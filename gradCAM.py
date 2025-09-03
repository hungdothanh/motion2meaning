# # -------------------gradCAM.py-------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Optional, Dict
# import cv2

# class GradCAM1D:
#     """
#     Gradient-weighted Class Activation Mapping for 1D CNNs (time series).
#     Works with single-channel or multi-channel 1D inputs.
#     """

#     def __init__(self, model: nn.Module, target_layers: List[str]):
#         self.model = model
#         self.model.eval()
#         self.target_layers = target_layers
#         self.activations: Dict[str, torch.Tensor] = {}
#         self.gradients: Dict[str, torch.Tensor] = {}
#         self._register_hooks()

#     def _register_hooks(self):
#         def forward_hook(name):
#             def hook(module, input, output):
#                 # Expect shape: (B, C, L)
#                 self.activations[name] = output.detach()
#             return hook

#         def backward_hook(name):
#             def hook(module, grad_input, grad_output):
#                 # Grad wrt output of the module: shape (B, C, L)
#                 self.gradients[name] = grad_output[0].detach()
#             return hook

#         for name, module in self.model.named_modules():
#             if name in self.target_layers:
#                 module.register_forward_hook(forward_hook(name))
#                 # register_backward_hook is deprecated but still works;
#                 # if using newer PyTorch, swap with register_full_backward_hook
#                 try:
#                     module.register_full_backward_hook(backward_hook(name))
#                 except Exception:
#                     module.register_backward_hook(backward_hook(name))

#     def generate_cam(
#         self,
#         input_tensor: torch.Tensor,
#         target_class: Optional[int] = None,
#         layer_name: Optional[str] = None
#     ):
#         """
#         Args:
#             input_tensor: (B, C, L)
#         Returns:
#             cams: {layer_name: 1D tensor of length L}, target_class
#         """
#         self.activations.clear()
#         self.gradients.clear()

#         input_tensor = input_tensor.requires_grad_(True)
#         output = self.model(input_tensor)  # (B, num_classes)

#         if target_class is None:
#             target_class = int(output.argmax(dim=1).item())

#         self.model.zero_grad()
#         class_score = output[0, target_class]
#         class_score.backward(retain_graph=True)

#         cams: Dict[str, torch.Tensor] = {}
#         target_layers = [layer_name] if layer_name else self.target_layers

#         for layer in target_layers:
#             if (layer not in self.activations) or (layer not in self.gradients):
#                 continue
#             # (B, C, L) -> (C, L)
#             A = self.activations[layer][0]
#             dA = self.gradients[layer][0]
#             # Global average pool gradients over L: (C, 1)
#             weights = dA.mean(dim=1, keepdim=True)
#             # Weighted sum over channels -> (L,)
#             cam = (weights * A).sum(dim=0)
#             cam = F.relu(cam)
#             if cam.max() > 0:
#                 cam = cam / cam.max()
#             cams[layer] = cam  # (L,)

#         return cams, target_class

#     def _upsample_cam(self, cam: torch.Tensor, target_size: int) -> torch.Tensor:
#         if cam.numel() == target_size:
#             return cam
#         cam_np = cam.detach().cpu().numpy()
#         cam_resized = cv2.resize(cam_np.reshape(1, -1), (target_size, 1),
#                                  interpolation=cv2.INTER_LINEAR).flatten()
#         return torch.from_numpy(cam_resized).to(cam.device)

# -------------------gradCAM.py-------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Union
import cv2

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
        
        # in GradCAM1D._register_hooks()
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(forward_hook(name))
                try:
                    module.register_full_backward_hook(backward_hook(name))
                except Exception:
                    module.register_backward_hook(backward_hook(name))  # fallback

    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        # Clear previous caches
        self.activations.clear()
        self.gradients.clear()

        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)

        # Collect per-layer cams into a list
        chosen_layers = [layer_name] if layer_name else self.target_layers
        cam_list = []

        for layer in chosen_layers:
            if layer not in self.activations or layer not in self.gradients:
                continue
            A = self.activations[layer][0]      # (C, L')
            dA = self.gradients[layer][0]       # (C, L')
            w = dA.mean(dim=1, keepdim=True)    # (C, 1)
            cam = (w * A).sum(dim=0)            # (L',)
            cam = F.relu(cam)
            if cam.max() > 0:
                cam = cam / cam.max()
            cam_list.append(cam)

        if not cam_list:
            return {}, target_class

        # ---- average across layers ----
        min_len = min(c.shape[0] for c in cam_list)
        cam_stack = torch.stack([c[:min_len] for c in cam_list], dim=0)  # (n_layers, L'')
        cam_avg = cam_stack.mean(dim=0)  # (L'')

        # Return a torch.Tensor (not numpy)
        return {"_aggregated_": cam_avg}, target_class


    
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


