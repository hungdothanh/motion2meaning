
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

