import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, List
import os

# LRP Implementation for PyTorch (Based on your existing code)
class LRPUtil:
    """Utility class for Layer-Wise Relevance Propagation"""
    
    @staticmethod
    def safe_divide(a, b, eps=1e-12):
        """Safe division avoiding division by zero"""
        return a / (b + eps * torch.sign(b))
    
    @staticmethod
    def clone_layer(layer):
        """Clone a layer for LRP computation"""
        if isinstance(layer, nn.Conv1d):
            new_layer = nn.Conv1d(layer.in_channels, layer.out_channels, 
                                 layer.kernel_size, layer.stride, layer.padding)
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            return new_layer
        elif isinstance(layer, nn.Linear):
            new_layer = nn.Linear(layer.in_features, layer.out_features)
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            return new_layer
        else:
            return layer

class LRPModel(nn.Module):
    """LRP-enabled version of the Parkinson's CNN"""
    
    def __init__(self, original_model):
        super(LRPModel, self).__init__()
        self.original_model = original_model
        
        # Clone all layers for LRP
        self.conv1 = LRPUtil.clone_layer(original_model.conv1)
        self.conv2 = LRPUtil.clone_layer(original_model.conv2)
        self.conv3 = LRPUtil.clone_layer(original_model.conv3)
        self.conv4 = LRPUtil.clone_layer(original_model.conv4)
        
        self.fc1 = LRPUtil.clone_layer(original_model.fc1)
        self.fc2 = LRPUtil.clone_layer(original_model.fc2)
        self.fc3 = LRPUtil.clone_layer(original_model.fc3)
        
        # Store activations for LRP
        self.activations = {}
        
    def forward(self, x):
        """Forward pass storing activations"""
        self.activations['input'] = x
        
        # Conv blocks with ReLU and pooling
        x = F.relu(self.conv1(x))
        self.activations['conv1'] = x
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        self.activations['pool1'] = x
        
        x = F.relu(self.conv2(x))
        self.activations['conv2'] = x
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        self.activations['pool2'] = x
        
        x = F.relu(self.conv3(x))
        self.activations['conv3'] = x
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        self.activations['pool3'] = x
        
        x = F.relu(self.conv4(x))
        self.activations['conv4'] = x
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        self.activations['pool4'] = x
        
        # Flatten
        x = x.view(x.size(0), -1)
        self.activations['flatten'] = x
        
        # FC layers
        x = F.relu(self.fc1(x))
        self.activations['fc1'] = x
        x = F.relu(self.fc2(x))
        self.activations['fc2'] = x
        x = self.fc3(x)
        self.activations['output'] = x
        
        return x
    
    def lrp_linear(self, layer, activation_in, activation_out, relevance_out, rule='epsilon'):
        """LRP for linear layers"""
        if rule == 'epsilon':
            # LRP-epsilon rule
            eps = 1e-12
            z = torch.mm(activation_in, layer.weight.t())
            if layer.bias is not None:
                z = z + layer.bias.unsqueeze(0)
            s = relevance_out / (z + eps)
            c = torch.mm(s, layer.weight)
            relevance_in = activation_in * c
            
        elif rule == 'alpha_beta':
            # LRP-alpha-beta rule
            alpha = 2.0
            beta = -1.0
            
            w_pos = torch.clamp(layer.weight, min=0)
            w_neg = torch.clamp(layer.weight, max=0)
            
            z_pos = torch.mm(activation_in, w_pos.t())
            z_neg = torch.mm(activation_in, w_neg.t())
            
            if layer.bias is not None:
                bias_pos = torch.clamp(layer.bias, min=0)
                bias_neg = torch.clamp(layer.bias, max=0)
                z_pos = z_pos + bias_pos.unsqueeze(0)
                z_neg = z_neg + bias_neg.unsqueeze(0)
            
            s_pos = relevance_out / (z_pos + 1e-12)
            s_neg = relevance_out / (z_neg - 1e-12)
            
            c_pos = torch.mm(s_pos, w_pos)
            c_neg = torch.mm(s_neg, w_neg)
            
            relevance_in = activation_in * (alpha * c_pos + beta * c_neg)
            
        return relevance_in
    
    def lrp_conv1d(self, layer, activation_in, activation_out, relevance_out, rule='epsilon'):
        """LRP for 1D convolutional layers"""
        if rule == 'epsilon':
            eps = 1e-12
            z = F.conv1d(activation_in, layer.weight, layer.bias, 
                        layer.stride, layer.padding)
            s = relevance_out / (z + eps)
            
            # Gradient computation using conv_transpose1d
            c = F.conv_transpose1d(s, layer.weight, 
                                  stride=layer.stride, padding=layer.padding)
            relevance_in = activation_in * c
            
        elif rule == 'alpha_beta':
            alpha = 2.0
            beta = -1.0
            
            w_pos = torch.clamp(layer.weight, min=0)
            w_neg = torch.clamp(layer.weight, max=0)
            
            z_pos = F.conv1d(activation_in, w_pos, None, 
                           layer.stride, layer.padding)
            z_neg = F.conv1d(activation_in, w_neg, None, 
                           layer.stride, layer.padding)
            
            if layer.bias is not None:
                bias_pos = torch.clamp(layer.bias, min=0)
                bias_neg = torch.clamp(layer.bias, max=0)
                z_pos = z_pos + bias_pos.view(1, -1, 1)
                z_neg = z_neg + bias_neg.view(1, -1, 1)
            
            s_pos = relevance_out / (z_pos + 1e-12)
            s_neg = relevance_out / (z_neg - 1e-12)
            
            c_pos = F.conv_transpose1d(s_pos, w_pos, 
                                     stride=layer.stride, padding=layer.padding)
            c_neg = F.conv_transpose1d(s_neg, w_neg, 
                                     stride=layer.stride, padding=layer.padding)
            
            relevance_in = activation_in * (alpha * c_pos + beta * c_neg)
            
        return relevance_in
    
    def explain(self, input_tensor, target_class=None, rule='epsilon'):
        """Generate LRP explanation for input"""
        # Forward pass
        output = self.forward(input_tensor)
        
        # Initialize relevance with prediction
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # One-hot encode target class
        relevance = torch.zeros_like(output)
        relevance[0, target_class] = output[0, target_class]
        
        # Backward pass through layers
        # FC3 -> FC2
        relevance = self.lrp_linear(self.fc3, self.activations['fc2'], 
                                   self.activations['output'], relevance, rule)
        
        # FC2 -> FC1
        relevance = self.lrp_linear(self.fc2, self.activations['fc1'], 
                                   self.activations['fc2'], relevance, rule)
        
        # FC1 -> Flatten
        relevance = self.lrp_linear(self.fc1, self.activations['flatten'], 
                                   self.activations['fc1'], relevance, rule)
        
        # Reshape back to conv output
        relevance = relevance.view(self.activations['pool4'].shape)
        
        # Upsample through pooling layers
        relevance = F.interpolate(relevance, size=self.activations['conv4'].shape[-1], 
                                mode='nearest')
        
        # Conv4 -> Conv3
        relevance = self.lrp_conv1d(self.conv4, self.activations['pool3'], 
                                   self.activations['conv4'], relevance, rule)
        
        # Upsample
        relevance = F.interpolate(relevance, size=self.activations['conv3'].shape[-1], 
                                mode='nearest')
        
        # Conv3 -> Conv2
        relevance = self.lrp_conv1d(self.conv3, self.activations['pool2'], 
                                   self.activations['conv3'], relevance, rule)
        
        # Upsample
        relevance = F.interpolate(relevance, size=self.activations['conv2'].shape[-1], 
                                mode='nearest')
        
        # Conv2 -> Conv1
        relevance = self.lrp_conv1d(self.conv2, self.activations['pool1'], 
                                   self.activations['conv2'], relevance, rule)
        
        # Upsample
        relevance = F.interpolate(relevance, size=self.activations['conv1'].shape[-1], 
                                mode='nearest')
        
        # Conv1 -> Input
        relevance = self.lrp_conv1d(self.conv1, self.activations['input'], 
                                   self.activations['conv1'], relevance, rule)
        
        return relevance


class LRPVisualizer:
    """Focused visualization class for LRP explanations"""
    
    def __init__(self, sensor_names: List[str], class_names: List[str]):
        """
        Initialize LRP visualizer.
        
        Args:
            sensor_names: Names of sensors/channels
            class_names: Names of classification classes
        """
        self.sensor_names = sensor_names
        self.class_names = class_names
        
        # Define colormaps for relevance visualization
        self.relevance_colormap = LinearSegmentedColormap.from_list(
            'relevance', ['darkblue', 'blue', 'lightblue', 'white', 'lightcoral', 'red', 'darkred']
        )
        
    def plot_relevance_overlay(self, 
                              input_data: np.ndarray,
                              relevance_scores: np.ndarray,
                              predicted_class: int,
                              confidence: float,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Visualize LRP relevance scores overlaid on original time series.
        
        Args:
            input_data: Original input data (channels, time)
            relevance_scores: LRP relevance scores (channels, time)
            predicted_class: Predicted class index
            confidence: Prediction confidence
            save_path: Path to save the figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(len(self.sensor_names), 1, figsize=figsize,
                                sharex=True, facecolor='white')
        
        if len(self.sensor_names) == 1:
            axes = [axes]
        
        time_points = np.arange(input_data.shape[1])
        
        # Normalize relevance scores for color mapping
        relevance_max = np.abs(relevance_scores).max()
        if relevance_max > 0:
            relevance_normalized = relevance_scores / relevance_max
        else:
            relevance_normalized = relevance_scores
        
        for i, (sensor_name, ax) in enumerate(zip(self.sensor_names, axes)):
            # Plot original signal in light gray
            ax.plot(time_points, input_data[i], 'lightgray', alpha=0.7, linewidth=1,
                   label='Original Signal')
            
            # Create relevance-based colored overlay
            for j in range(len(time_points) - 1):
                relevance_val = relevance_normalized[i, j]
                
                # Map relevance to color (red for positive, blue for negative)
                color_val = (relevance_val + 1) / 2  # Map from [-1,1] to [0,1]
                color = self.relevance_colormap(color_val)
                
                # Plot segment with color and thickness based on relevance magnitude
                alpha = min(0.9, abs(relevance_val) * 0.8 + 0.3)
                linewidth = max(1, abs(relevance_val) * 3 + 1)
                
                ax.plot([time_points[j], time_points[j+1]], 
                       [input_data[i, j], input_data[i, j+1]], 
                       color=color, alpha=alpha, linewidth=linewidth)
            
            # Customize subplot
            ax.set_ylabel(sensor_name, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, len(time_points)-1)
            
            # Add relevance statistics
            rel_mean = relevance_scores[i].mean()
            rel_std = relevance_scores[i].std()
            ax.text(0.02, 0.98, f'μ={rel_mean:.3f}, σ={rel_std:.3f}',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.relevance_colormap)
        sm.set_array([-1, 1])
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal',
                           fraction=0.05, pad=0.1, shrink=0.8)
        cbar.set_label('LRP Relevance Score (Normalized)', fontsize=12)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(['Strong Negative', 'Negative', 'Neutral', 'Positive', 'Strong Positive'])
        
        # Set title and labels
        title = f'LRP Relevance Overlay Analysis\n'
        title += f'Predicted: {self.class_names[predicted_class]} (Confidence: {confidence:.3f})'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        axes[-1].set_xlabel('Time Steps', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Relevance overlay plot saved to: {save_path}")
        
        return fig
    
    def plot_relevance_heatmap(self,
                              relevance_scores: np.ndarray,
                              predicted_class: int,
                              confidence: float,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualize LRP relevance scores as a 2D heatmap.
        
        Args:
            relevance_scores: LRP relevance scores (channels, time)
            predicted_class: Predicted class index
            confidence: Prediction confidence
            save_path: Path to save the figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
        
        # Create heatmap with symmetric colormap around zero
        vmax = np.abs(relevance_scores).max()
        vmin = -vmax
        
        im = ax.imshow(relevance_scores, cmap=self.relevance_colormap, 
                      aspect='auto', interpolation='bilinear',
                      vmin=vmin, vmax=vmax)
        
        # Customize plot
        ax.set_title(f'LRP Relevance Heatmap\n'
                    f'Predicted: {self.class_names[predicted_class]} (Confidence: {confidence:.3f})',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Sensor Channels', fontsize=12)
        
        # Set y-axis labels
        ax.set_yticks(range(len(self.sensor_names)))
        ax.set_yticklabels(self.sensor_names, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Relevance Score', fontsize=12)
        
        # Add grid for better readability
        ax.set_xticks(np.arange(0, relevance_scores.shape[1], relevance_scores.shape[1]//10))
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Relevance heatmap saved to: {save_path}")
        
        return fig


class LRPExplainer:
    """Main class for LRP explanations with focused visualization"""
    
    def __init__(self, model, sensor_names: List[str], class_names: List[str], device='cpu'):
        """
        Initialize LRP explainer.
        
        Args:
            model: Trained PyTorch model
            sensor_names: List of sensor channel names
            class_names: List of class names
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.sensor_names = sensor_names
        self.class_names = class_names
        
        # Create LRP model
        self.lrp_model = LRPModel(model)
        self.lrp_model.to(device)
        self.lrp_model.eval()
        
        # Create visualizer
        self.visualizer = LRPVisualizer(sensor_names, class_names)
        
    def explain_and_visualize(self, 
                             input_data: np.ndarray,
                             target_class: Optional[int] = None,
                             rule: str = 'epsilon',
                             save_dir: Optional[str] = None,
                             prefix: str = 'lrp') -> dict:
        """
        Generate LRP explanation and create visualizations.
        
        Args:
            input_data: Input data (channels, time) or (batch, channels, time)
            target_class: Target class for explanation (if None, uses predicted class)
            rule: LRP rule to use ('epsilon' or 'alpha_beta')
            save_dir: Directory to save visualizations
            prefix: Prefix for saved files
        
        Returns:
            Dictionary containing results and paths to saved visualizations
        """
        # Ensure input is in correct format
        if input_data.ndim == 2:
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
            input_for_viz = input_data
        else:
            input_tensor = torch.FloatTensor(input_data).to(self.device)
            input_for_viz = input_data[0]  # Take first sample for visualization
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Generate LRP explanation
        with torch.no_grad():
            relevance = self.lrp_model.explain(input_tensor, target_class, rule)
            relevance_scores = relevance[0].cpu().numpy()  # Remove batch dimension
        
        # Prepare results
        results = {
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'relevance_scores': relevance_scores,
            'input_data': input_for_viz,
            'rule_used': rule
        }
        
        # Create visualizations
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Relevance overlay plot
            overlay_path = os.path.join(save_dir, f"{prefix}_relevance_overlay.png")
            fig1 = self.visualizer.plot_relevance_overlay(
                input_for_viz, relevance_scores, predicted_class, confidence, overlay_path
            )
            results['overlay_plot_path'] = overlay_path
            
            # Relevance heatmap
            heatmap_path = os.path.join(save_dir, f"{prefix}_relevance_heatmap.png")
            fig2 = self.visualizer.plot_relevance_heatmap(
                relevance_scores, predicted_class, confidence, heatmap_path
            )
            results['heatmap_plot_path'] = heatmap_path
            
            plt.close('all')  # Close figures to free memory
        else:
            # Just create figures without saving
            fig1 = self.visualizer.plot_relevance_overlay(
                input_for_viz, relevance_scores, predicted_class, confidence
            )
            fig2 = self.visualizer.plot_relevance_heatmap(
                relevance_scores, predicted_class, confidence
            )
            plt.show()
        
        return results


# Integration function for your existing workflow
def explain_gait_with_lrp_simple(patient_name: str,
                                 segment_idx: int = 0,
                                 rule: str = 'epsilon',
                                 save_visualizations: bool = True,
                                 output_dir: str = "lrp_simple_outputs") -> dict:
    """
    Simple LRP explanation function that integrates with your existing code.
    
    Args:
        patient_name: Name of the patient ('Patient A' or 'Patient B')
        segment_idx: Index of the segment to analyze
        rule: LRP rule to use ('epsilon' or 'alpha_beta')
        save_visualizations: Whether to save visualization plots
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary containing LRP explanations and visualization paths
    """
    # Import required modules (assuming they're available)
    from config import PRETRAINED_MODEL_PATH, SEGMENT_LENGTH, CLASS_NAMES, SENSOR_NAMES
    from data import load_data, preprocess_file
    from model import ParkinsonsGaitCNN
    
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
    
    # Initialize LRP explainer
    explainer = LRPExplainer(model, SENSOR_NAMES, CLASS_NAMES, device)
    
    # Generate explanation and visualizations
    results = explainer.explain_and_visualize(
        input_data=segment,
        rule=rule,
        save_dir=output_dir if save_visualizations else None,
        prefix=f"{patient_name.replace(' ', '_').lower()}_segment_{segment_idx}"
    )
    
    # Add patient info to results
    results['patient_info'] = data
    results['segment_idx'] = segment_idx
    
    return results


# Example usage
if __name__ == "__main__":
    print("Simplified LRP Analysis for Parkinson's Gait Classification")
    print("=" * 60)
    
    # Analyze Patient A
    try:
        results_a = explain_gait_with_lrp_simple(
            patient_name='Patient A',
            segment_idx=0,
            rule='epsilon',
            save_visualizations=True,
            output_dir="lrp_simple_patient_a"
        )
        print(f"✓ Patient A analysis completed")
        print(f"  Predicted: {results_a['predicted_label']} (Confidence: {results_a['confidence']:.3f})")
        
    except Exception as e:
        print(f"✗ Patient A analysis failed: {e}")
    
    # Analyze Patient B
    try:
        results_b = explain_gait_with_lrp_simple(
            patient_name='Patient B',
            segment_idx=0,
            rule='alpha_beta',
            save_visualizations=True,
            output_dir="lrp_simple_patient_b"
        )
        print(f"✓ Patient B analysis completed")
        print(f"  Predicted: {results_b['predicted_label']} (Confidence: {results_b['confidence']:.3f})")
        
    except Exception as e:
        print(f"✗ Patient B analysis failed: {e}")
    
    print("\nSimplified LRP Analysis Complete!")
    print("Check the output directories for visualizations.")