
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import cv2
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Import your existing modules
from config import PRETRAINED_MODEL_PATH, SEGMENT_LENGTH, CLASS_NAMES, SENSOR_NAMES
from data import load_data, preprocess_file
from model import ParkinsonsGaitCNN

# Import XAI modules (from the provided files)
from gradCAM import GradCAM1D
from lrp import LRPModel


class XAIComparativeAnalyzer:
    """
    Comparative analyzer for GradCAM and LRP explanations on gait data.
    Allows selection of specific sensor channels and custom sequence lengths.
    """
    
    def __init__(self, model_path: str = PRETRAINED_MODEL_PATH):
        """
        Initialize the analyzer with pre-trained model.
        
        Args:
            model_path: Path to the pre-trained model weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load original model
        self.model = ParkinsonsGaitCNN(input_channels=16, sequence_length=SEGMENT_LENGTH)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        
        # Initialize XAI methods
        self.gradcam = GradCAM1D(self.model, target_layers=['conv4'])
        self.lrp_model = LRPModel(self.model)
        self.lrp_model.to(self.device).eval()
        
        
        self.colormap = LinearSegmentedColormap.from_list(
            'simple', ['blue', 'cyan', 'yellow', 'red']
        )
        
    def extract_sensor_data(self, 
                           patient_name: str,
                           sensor_idx: int,
                           start_time: int = 0,
                           sequence_length: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract data for a specific sensor channel and time window.
        
        Args:
            patient_name: Name of the patient ('Patient A' or 'Patient B')
            sensor_idx: Index of the sensor channel (0-15)
            start_time: Starting time point
            sequence_length: Length of the sequence to extract
            
        Returns:
            Tuple of (single_sensor_data, full_segment_data)
        """
        # Load patient data
        data = load_data()[patient_name]
        
        # Load raw data file
        import pandas as pd
        raw_data = pd.read_csv(data['gait_file'], sep='\t', header=None)
        
        if raw_data.shape[1] > 1:
            features = raw_data.iloc[:, 1:-2].values  # Skip time column
        else:
            features = raw_data.values
        
        # Extract the specified time window
        end_time = min(start_time + sequence_length, len(features))
        actual_length = end_time - start_time
        
        # Extract window
        window_data = features[start_time:end_time]
        
        # Pad if necessary
        if actual_length < sequence_length:
            padding = np.zeros((sequence_length - actual_length, window_data.shape[1]))
            window_data = np.vstack([window_data, padding])
        
        # Transpose to get (channels, time)
        window_data = window_data.T
        
        # Extract single sensor data
        single_sensor_data = window_data[sensor_idx]
        
        return single_sensor_data, window_data
    
    def compute_gradcam_relevance(self, 
                                  full_segment: np.ndarray,
                                  sensor_idx: int,
                                  target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute GradCAM relevance for a specific sensor.
        
        Args:
            full_segment: Full segment data (channels, time)
            sensor_idx: Index of the sensor to analyze
            target_class: Target class for explanation
            
        Returns:
            GradCAM relevance scores for the specified sensor
        """
        # Prepare input tensor
        input_tensor = torch.FloatTensor(full_segment).unsqueeze(0).to(self.device)
        
        # Generate GradCAM
        cams, pred_class = self.gradcam.generate_cam(input_tensor, target_class)
        
        # Get CAM for the last conv layer
        cam = cams['conv4'].cpu().numpy()
        
        # Upsample CAM to match input size
        if len(cam) != full_segment.shape[1]:
            cam = cv2.resize(cam.reshape(1, -1), 
                           (full_segment.shape[1], 1), 
                           interpolation=cv2.INTER_LINEAR).flatten()
        
        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def compute_lrp_relevance(self, 
                             full_segment: np.ndarray,
                             sensor_idx: int,
                             target_class: Optional[int] = None,
                             rule: str = 'epsilon') -> np.ndarray:
        """
        Compute LRP relevance for a specific sensor.
        
        Args:
            full_segment: Full segment data (channels, time)
            sensor_idx: Index of the sensor to analyze
            target_class: Target class for explanation
            rule: LRP rule to use
            
        Returns:
            LRP relevance scores for the specified sensor
        """
        # Prepare input tensor
        input_tensor = torch.FloatTensor(full_segment).unsqueeze(0).to(self.device)
        
        # Generate LRP explanation
        with torch.no_grad():
            relevance = self.lrp_model.explain(input_tensor, target_class, rule)
            relevance_scores = relevance[0].cpu().numpy()  # Shape: (channels, time)
        
        # Extract relevance for specific sensor
        sensor_relevance = relevance_scores[sensor_idx]
        
        # Normalize to [0, 1] (taking absolute values for comparison)
        sensor_relevance_abs = np.abs(sensor_relevance)
        if sensor_relevance_abs.max() > sensor_relevance_abs.min():
            sensor_relevance_norm = (sensor_relevance_abs - sensor_relevance_abs.min()) / \
                                   (sensor_relevance_abs.max() - sensor_relevance_abs.min())
        else:
            sensor_relevance_norm = sensor_relevance_abs
        
        return sensor_relevance_norm
    
    def plot_comparative_analysis(self,
                                 patient_name: str,
                                 sensor_idx: int,
                                 start_time: int = 0,
                                 sequence_length: int = 1000,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (15, 18)) -> plt.Figure:
        """
        Create comprehensive comparative visualization of GradCAM and LRP.
        
        Args:
            patient_name: Name of the patient
            sensor_idx: Index of the sensor channel (0-15)
            start_time: Starting time point
            sequence_length: Length of the sequence
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure with 5 subplots
        """
        # Extract data
        sensor_data, full_segment = self.extract_sensor_data(
            patient_name, sensor_idx, start_time, sequence_length
        )
        
        # Get model prediction
        input_tensor = torch.FloatTensor(full_segment).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Compute relevance scores
        gradcam_relevance = self.compute_gradcam_relevance(full_segment, sensor_idx)
        lrp_relevance = self.compute_lrp_relevance(full_segment, sensor_idx)
        
        # Create figure with 5 subplots
        fig, axes = plt.subplots(5, 1, figsize=figsize, facecolor='white')
        time_points = np.arange(len(sensor_data))
        
        # Get sensor name
        sensor_name = SENSOR_NAMES[sensor_idx] if sensor_idx < len(SENSOR_NAMES) else f"Sensor {sensor_idx}"
        
        # ---------------------------------- Figure 1: Raw data with GradCAM overlay ----------------------------------
        ax1 = axes[0]
        ax1.plot(time_points, sensor_data, 'k-', alpha=0.5, linewidth=1, label='Raw Data')
        
        # Create GradCAM overlay
        for i in range(len(time_points) - 1):
            color_intensity = gradcam_relevance[i]
            color = self.colormap(color_intensity)
            ax1.plot([time_points[i], time_points[i+1]], 
                    [sensor_data[i], sensor_data[i+1]], 
                    color=color, alpha=0.8, linewidth=2)
        
        ax1.set_title(f'Fig 1: Raw Data with GradCAM Overlay - {sensor_name}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Signal Amplitude', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(time_points)-1)
        
        # Add GradCAM colorbar
        sm1 = plt.cm.ScalarMappable(cmap=self.colormap, norm=plt.Normalize(0, 1))
        sm1.set_array([])
        cbar1 = plt.colorbar(sm1, ax=ax1, orientation='vertical', fraction=0.03, pad=0.02)
        cbar1.set_label('GradCAM Relevance', fontsize=9)
        
        # ---------------------------------- Figure 2: Raw data with LRP overlay ----------------------------------
        ax2 = axes[1]
        ax2.plot(time_points, sensor_data, 'k-', alpha=0.5, linewidth=1, label='Raw Data')
        
        # Create LRP overlay
        for i in range(len(time_points) - 1):
            color_intensity = lrp_relevance[i]
            color = self.colormap(color_intensity)
            ax2.plot([time_points[i], time_points[i+1]], 
                    [sensor_data[i], sensor_data[i+1]], 
                    color=color, alpha=0.8, linewidth=2)
        
        ax2.set_title(f'Fig 2: Raw Data with LRP Overlay - {sensor_name}', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Signal Amplitude', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, len(time_points)-1)
        
        # Add LRP colorbar
        sm2 = plt.cm.ScalarMappable(cmap=self.colormap, norm=plt.Normalize(0, 1))
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax2, orientation='vertical', fraction=0.03, pad=0.02)
        cbar2.set_label('LRP Relevance', fontsize=9)
        
        # ---------------------------------- Figure 3: Normalized relevance scores comparison ----------------------------------
        ax3 = axes[2]
        ax3.plot(time_points, gradcam_relevance, 'b-', linewidth=2, alpha=0.8, label='GradCAM')
        ax3.plot(time_points, lrp_relevance, 'r-', linewidth=2, alpha=0.8, label='LRP')
        ax3.set_title('Fig 3: Normalized Relevance Scores Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Normalized Relevance [0-1]', fontsize=10)
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        ax3.set_xlim(0, len(time_points)-1)
        
        # ---------------------------------- Figure 4: Absolute difference with highlighted regions ----------------------------------
        ax4 = axes[3]
        abs_diff = np.abs(gradcam_relevance - lrp_relevance)
        ax4.plot(time_points, abs_diff, 'purple', linewidth=2, label='|GradCAM - LRP|')
        
        # Highlight regions where difference > 0.5
        threshold = 0.5
        ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.5, label=f'Threshold = {threshold}')
        
        # Fill areas where difference > threshold
        high_discrepancy_mask = abs_diff > threshold
        ax4.fill_between(time_points, 0, abs_diff, where=high_discrepancy_mask, 
                        color='red', alpha=0.3, label='High Discrepancy')
        
        ax4.set_title('Fig 4: Absolute Difference |GradCAM - LRP| with High Discrepancy Regions', 
                     fontsize=12, fontweight='bold')
        ax4.set_ylabel('Absolute Difference', fontsize=10)
        ax4.set_ylim(-0.05, 1.05)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right')
        ax4.set_xlim(0, len(time_points)-1)
        
        # ---------------------------------- Figure 5: Original data with highlighted discrepancy regions ----------------------------------
        ax5 = axes[4]
        ax5.plot(time_points, sensor_data, 'k-', linewidth=1.5, label='Raw Data')
        
        # Highlight regions with high discrepancy
        for i in range(len(time_points)):
            if high_discrepancy_mask[i]:
                ax5.axvspan(max(0, i-1), min(len(time_points)-1, i+1), 
                          color='red', alpha=0.2)
        
        # Add a single patch for legend
        if np.any(high_discrepancy_mask):
            red_patch = mpatches.Patch(color='red', alpha=0.2, label='High Discrepancy Region')
            ax5.legend(handles=[ax5.get_lines()[0], red_patch], loc='upper right')
        else:
            ax5.legend(loc='upper right')
        
        ax5.set_title('Fig 5: Raw Data with High Discrepancy Regions Highlighted', 
                     fontsize=12, fontweight='bold')
        ax5.set_ylabel('Signal Amplitude', fontsize=10)
        ax5.set_xlabel('Time Steps', fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, len(time_points)-1)
        
        # Add overall title
        fig.suptitle(f'XAI Comparative Analysis - {patient_name}\n' + 
                    f'Predicted: {CLASS_NAMES[predicted_class]} (Confidence: {confidence:.3f})\n' +
                    f'Colormap: Blue (Low Relevance) → Red (High Relevance)',
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300) #, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    
    def analyze_multiple_sensors(self,
                                patient_name: str,
                                sensor_indices: List[int],
                                start_time: int = 0,
                                sequence_length: int = 1000,
                                save_dir: Optional[str] = None) -> Dict:
        """
        Analyze multiple sensors and generate comparative visualizations.
        
        Args:
            patient_name: Name of the patient
            sensor_indices: List of sensor indices to analyze
            start_time: Starting time point
            sequence_length: Length of the sequence
            save_dir: Directory to save figures
            
        Returns:
            Dictionary with analysis results for each sensor
        """
        import os
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        results = {}
        
        for sensor_idx in sensor_indices:
            sensor_name = SENSOR_NAMES[sensor_idx] if sensor_idx < len(SENSOR_NAMES) else f"Sensor {sensor_idx}"
            print(f"\nAnalyzing {sensor_name} (Index: {sensor_idx})...")
            
            # Generate save path
            save_path = None
            if save_dir:
                safe_sensor_name = sensor_name.replace(' ', '_').replace('-', '_')
                save_path = os.path.join(save_dir, 
                    f"{patient_name.replace(' ', '_')}_{safe_sensor_name}_comparison.png")
            
            # Create visualization
            fig = self.plot_comparative_analysis(
                patient_name=patient_name,
                sensor_idx=sensor_idx,
                start_time=start_time,
                sequence_length=sequence_length,
                save_path=save_path
            )
            
            # Store results
            results[sensor_name] = {
                'sensor_idx': sensor_idx,
                'figure': fig,
                'save_path': save_path
            }
            
            plt.close(fig)  # Close to free memory
        
        return results



if __name__ == "__main__":


    analyzer = XAIComparativeAnalyzer()

    # Get user input
    patient = input("Enter patient name (Patient A/Patient B): ")
    sensor_idx = int(input(f"Enter sensor index (0-15): "))
    start_time = int(input("Enter start time (default 0): ") or "0")
    # seq_length = int(input("Enter sequence length (default 1000): ") or "1000")
    seq_length = 1000
    
    # Generate analysis
    fig = analyzer.plot_comparative_analysis(
        patient_name=patient,
        sensor_idx=sensor_idx,
        start_time=start_time,
        sequence_length=seq_length,
        save_path=f"{patient}_{sensor_idx}_unified_analysis.png"
    )
    plt.show()