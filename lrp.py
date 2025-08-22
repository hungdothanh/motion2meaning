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
