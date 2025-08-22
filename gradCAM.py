
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Union
import cv2
from matplotlib.colors import LinearSegmentedColormap

class GradCAM1D:
    """
    Gradient-weighted Class Activation Mapping for 1D CNNs applied to time series data.
    
    This implementation adapts the original Grad-CAM paper for 1D convolutional networks
    processing temporal data, specifically designed for Parkinson's gait analysis.
    """
    
    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Initialize Grad-CAM for 1D CNN.
        
        Args:
            model: The trained 1D CNN model
            target_layers: List of layer names to extract activations from
        """
        self.model = model
        self.model.eval()
        self.target_layers = target_layers
        
        # Storage for activations and gradients
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for target layers."""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(forward_hook(name))
                module.register_backward_hook(backward_hook(name))
    
    def generate_cam(self, 
                     input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None,
                     layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Generate Class Activation Maps for the input.
        
        Args:
            input_tensor: Input tensor of shape (batch_size, channels, sequence_length)
            target_class: Target class index. If None, uses predicted class
            layer_name: Specific layer to generate CAM for. If None, uses all target layers
        
        Returns:
            Dictionary mapping layer names to their CAM tensors
        """
        # Clear previous activations and gradients
        self.activations.clear()
        self.gradients.clear()
        
        # Forward pass
        input_tensor.requires_grad_()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)
        
        # Generate CAMs
        cams = {}
        target_layers = [layer_name] if layer_name else self.target_layers
        
        for layer in target_layers:
            if layer in self.activations and layer in self.gradients:
                activations = self.activations[layer][0]  # Shape: (channels, time)
                gradients = self.gradients[layer][0]      # Shape: (channels, time)
                
                # Global Average Pooling of gradients
                weights = gradients.mean(dim=1, keepdim=True)  # Shape: (channels, 1)
                
                # Weighted combination of activation maps
                cam = (weights * activations).sum(dim=0)  # Shape: (time,)
                
                # Apply ReLU
                cam = F.relu(cam)
                
                # Normalize CAM to [0, 1]
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                cams[layer] = cam
        
        return cams, target_class
    
    def generate_guided_gradcam(self, 
                               input_tensor: torch.Tensor,
                               target_class: Optional[int] = None,
                               layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Generate Guided Grad-CAM which combines Grad-CAM with Guided Backpropagation.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index
            layer_name: Specific layer name
        
        Returns:
            Dictionary mapping layer names to guided Grad-CAM tensors
        """
        # Get regular Grad-CAM
        cams, pred_class = self.generate_cam(input_tensor, target_class, layer_name)
        
        # Get guided backpropagation
        guided_gradients = self._guided_backpropagation(input_tensor, pred_class)
        
        # Combine CAM with guided gradients
        guided_gradcams = {}
        for layer_name, cam in cams.items():
            # Upsample CAM to match input size if necessary
            cam_upsampled = self._upsample_cam(cam, input_tensor.shape[-1])
            
            # Element-wise multiplication
            guided_gradcam = guided_gradients * cam_upsampled.unsqueeze(0)  # Broadcast over channels
            guided_gradcams[layer_name] = guided_gradcam
        
        return guided_gradcams, pred_class
    
    def _guided_backpropagation(self, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Perform guided backpropagation.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index
        
        Returns:
            Guided gradients
        """
        # Store original ReLU functions
        relu_outputs = []
        
        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        
        # Register hooks for guided backprop
        hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_backward_hook(relu_hook_function))
        
        # Forward pass
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients
        guided_gradients = input_tensor.grad.data[0]  # Remove batch dimension
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return guided_gradients
    
    def _upsample_cam(self, cam: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Upsample CAM to match input sequence length.
        
        Args:
            cam: CAM tensor of shape (time,)
            target_size: Target sequence length
        
        Returns:
            Upsampled CAM tensor
        """
        if cam.shape[0] == target_size:
            return cam
        
        # Convert to numpy for cv2 resize, then back to torch
        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np.reshape(1, -1), (target_size, 1), 
                                interpolation=cv2.INTER_LINEAR).flatten()
        return torch.from_numpy(cam_resized).to(cam.device)


class GaitGradCAMVisualizer:
    """
    Visualization utilities for Grad-CAM applied to gait analysis.
    """
    
    def __init__(self, sensor_names: List[str], class_names: List[str]):
        """
        Initialize visualizer.
        
        Args:
            sensor_names: Names of sensors/channels
            class_names: Names of classification classes
        """
        self.sensor_names = sensor_names
        self.class_names = class_names
        
        # Define colormap
        self.cam_colormap = LinearSegmentedColormap.from_list(
            'cam', ['blue', 'cyan', 'yellow', 'red']
        )
    
    def visualize_cam_overlay(self, 
                             input_data: np.ndarray,
                             cam: torch.Tensor,
                             predicted_class: int,
                             confidence: float,
                             layer_name: str,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Visualize Grad-CAM overlay on time series data.
        
        Args:
            input_data: Original input data (channels, time)
            cam: CAM tensor (time,)
            predicted_class: Predicted class index
            confidence: Prediction confidence
            layer_name: Name of the layer
            save_path: Path to save the figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(len(self.sensor_names), 1, figsize=figsize, 
                                sharex=True, facecolor='white')
        
        if len(self.sensor_names) == 1:
            axes = [axes]
        
        # Convert CAM to numpy
        cam_np = cam.cpu().numpy()
        time_points = np.arange(input_data.shape[1])
        
        # Ensure CAM matches input length
        if len(cam_np) != input_data.shape[1]:
            cam_np = cv2.resize(cam_np.reshape(1, -1), 
                               (input_data.shape[1], 1), 
                               interpolation=cv2.INTER_LINEAR).flatten()
        
        # Plot each sensor channel with CAM overlay
        for i, (sensor_name, ax) in enumerate(zip(self.sensor_names, axes)):
            # Plot original signal
            ax.plot(time_points, input_data[i], 'k-', alpha=0.7, linewidth=1)
            
            # Create CAM overlay
            cam_normalized = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            
            # Color-code the signal based on CAM intensity
            for j in range(len(time_points) - 1):
                color_intensity = cam_normalized[j]
                color = self.cam_colormap(color_intensity)
                ax.plot([time_points[j], time_points[j+1]], 
                       [input_data[i, j], input_data[i, j+1]], 
                       color=color, alpha=0.8, linewidth=2)
            
            ax.set_ylabel(sensor_name, fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cam_colormap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', 
                           fraction=0.05, pad=0.1)
        cbar.set_label('Grad-CAM Intensity', fontsize=12)
        
        # Set title and labels
        title = f'Grad-CAM Visualization - Layer: {layer_name}\n'
        title += f'Predicted: {self.class_names[predicted_class]} '
        title += f'(Confidence: {confidence:.2f})'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        axes[-1].set_xlabel('Time Steps', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_cam_heatmap(self, 
                             cam: torch.Tensor,
                             predicted_class: int,
                             layer_name: str,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Visualize CAM as a 1D heatmap.
        
        Args:
            cam: CAM tensor (time,)
            predicted_class: Predicted class index
            layer_name: Name of the layer
            save_path: Path to save the figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
        
        # Convert CAM to numpy and reshape for heatmap
        cam_np = cam.cpu().numpy().reshape(1, -1)
        
        # Create heatmap
        im = ax.imshow(cam_np, cmap=self.cam_colormap, aspect='auto', 
                      interpolation='bilinear')
        
        # Customize plot
        ax.set_title(f'Grad-CAM Heatmap - Layer: {layer_name}\n'
                    f'Predicted: {self.class_names[predicted_class]}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Activation', fontsize=12)
        ax.set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           fraction=0.1, pad=0.2)
        cbar.set_label('Grad-CAM Intensity', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_guided_gradcam(self,
                                input_data: np.ndarray,
                                guided_gradcam: torch.Tensor,
                                predicted_class: int,
                                layer_name: str,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Visualize Guided Grad-CAM.
        
        Args:
            input_data: Original input data (channels, time)
            guided_gradcam: Guided Grad-CAM tensor (channels, time)
            predicted_class: Predicted class index
            layer_name: Name of the layer
            save_path: Path to save the figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(len(self.sensor_names), 1, figsize=figsize,
                                sharex=True, facecolor='white')
        
        if len(self.sensor_names) == 1:
            axes = [axes]
        
        guided_gradcam_np = guided_gradcam.cpu().numpy()
        time_points = np.arange(input_data.shape[1])
        
        for i, (sensor_name, ax) in enumerate(zip(self.sensor_names, axes)):
            # Plot original signal
            ax.plot(time_points, input_data[i], 'k-', alpha=0.5, 
                   linewidth=1, label='Original')
            
            # Plot guided grad-cam
            guided_grad_normalized = guided_gradcam_np[i] / (np.abs(guided_gradcam_np[i]).max() + 1e-8)
            ax.plot(time_points, guided_grad_normalized, 'r-', 
                   linewidth=2, label='Guided Grad-CAM')
            
            ax.set_ylabel(sensor_name, fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        title = f'Guided Grad-CAM - Layer: {layer_name}\n'
        title += f'Predicted: {self.class_names[predicted_class]}'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        axes[-1].set_xlabel('Time Steps', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Integration with existing inference pipeline
def explain_gait_prediction(patient_name: str, 
                          segment_idx: int = 0,
                          target_layers: List[str] = None,
                          save_visualizations: bool = True,
                          output_dir: str = "gradcam_outputs") -> Dict:
    """
    Generate explanations for gait predictions using Grad-CAM.
    
    Args:
        patient_name: Name of the patient ('Patient A' or 'Patient B')
        segment_idx: Index of the segment to analyze
        target_layers: List of layer names to analyze
        save_visualizations: Whether to save visualization plots
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary containing explanations and visualizations
    """
    # Import required modules (assuming they're available)
    from config import PRETRAINED_MODEL_PATH, SEGMENT_LENGTH, CLASS_NAMES, SENSOR_NAMES
    from data import load_data, preprocess_file
    from model import ParkinsonsGaitCNN
    
    # Default target layers if none specified
    if target_layers is None:
        target_layers = ['conv4']  # Last convolutional layer
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParkinsonsGaitCNN(input_channels=16, sequence_length=SEGMENT_LENGTH)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    model.to(device).eval()
    
    # Load and preprocess data
    data = load_data()[patient_name]
    segments = preprocess_file(data['gait_file'], SEGMENT_LENGTH)
    
    if segment_idx >= len(segments):
        raise ValueError(f"Segment index {segment_idx} out of range. "
                        f"Available segments: {len(segments)}")
    
    # Get specific segment
    segment = segments[segment_idx]
    input_tensor = torch.FloatTensor(segment).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Initialize Grad-CAM
    gradcam = GradCAM1D(model, target_layers)
    visualizer = GaitGradCAMVisualizer(SENSOR_NAMES, CLASS_NAMES)
    
    # Generate CAMs
    cams, _ = gradcam.generate_cam(input_tensor, predicted_class)
    
    # Generate Guided Grad-CAM
    guided_gradcams, _ = gradcam.generate_guided_gradcam(input_tensor, predicted_class)
    
    # Create visualizations
    results = {
        'patient_info': data,
        'predicted_class': predicted_class,
        'predicted_label': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy(),
        'cams': cams,
        'guided_gradcams': guided_gradcams,
        'segment_data': segment
    }
    
    if save_visualizations:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for layer_name, cam in cams.items():
            # CAM overlay visualization
            fig1 = visualizer.visualize_cam_overlay(
                segment, cam, predicted_class, confidence, layer_name,
                save_path=os.path.join(output_dir, 
                    f"{patient_name}_{layer_name}_overlay.png")
            )
            
            # CAM heatmap visualization
            fig2 = visualizer.visualize_cam_heatmap(
                cam, predicted_class, layer_name,
                save_path=os.path.join(output_dir,
                    f"{patient_name}_{layer_name}_heatmap.png")
            )
            
            # # Guided Grad-CAM visualization
            # if layer_name in guided_gradcams:
            #     fig3 = visualizer.visualize_guided_gradcam(
            #         segment, guided_gradcams[layer_name], 
            #         predicted_class, layer_name,
            #         save_path=os.path.join(output_dir,
            #             f"{patient_name}_{layer_name}_guided.png")
            #     )
            
            plt.close('all')  # Close figures to free memory
    
    return results


# Example usage function
def analyze_all_patients():
    """
    Analyze both patients with Grad-CAM explanations.
    """
    patients = ['Patient A', 'Patient B']
    target_layers = ['conv4']  # Analyze multiple layers
    
    all_results = {}
    
    for patient in patients:
        print(f"\nAnalyzing {patient}...")
        try:
            results = explain_gait_prediction(
                patient_name=patient,
                segment_idx=0,
                target_layers=target_layers,
                save_visualizations=True,
                output_dir=f"gradcam_results_{patient.replace(' ', '_').lower()}"
            )
            
            all_results[patient] = results
            
            # Print summary
            print(f"Prediction: {results['predicted_label']} "
                  f"(Confidence: {results['confidence']:.3f})")
            
            for layer_name in target_layers:
                if layer_name in results['cams']:
                    cam = results['cams'][layer_name]
                    print(f"  {layer_name} CAM - "
                          f"Max activation: {cam.max():.3f}, "
                          f"Mean activation: {cam.mean():.3f}")
            
        except Exception as e:
            print(f"Error analyzing {patient}: {e}")
    
    return all_results


if __name__ == "__main__":
    # Run analysis
    results = analyze_all_patients()
    