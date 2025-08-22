#------------model.py------------
import torch
import torch.nn as nn

class ParkinsonsGaitCNN(nn.Module):
    """CNN model for Parkinson's gait analysis matching the provided architecture"""
    
    def __init__(self, input_channels: int = 3, sequence_length: int = 1000):
        super(ParkinsonsGaitCNN, self).__init__()
        
        # First Conv2D + AveragePooling block
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Second Conv2D + AveragePooling block  
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Third Conv2D + AveragePooling block
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Fourth Conv2D + AveragePooling block
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Activation functions
        self.relu = nn.ReLU()
        # Calculate flattened size
        self.flattened_size = self._get_flattened_size(input_channels, sequence_length)
        
        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)  # Output layer for 4-class classification
        

        
    def _get_flattened_size(self, input_channels: int, sequence_length: int) -> int:
        """Calculate the size after convolution and pooling layers"""
        x = torch.randn(1, input_channels, sequence_length)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        return x.numel()
    
    def forward(self, x):
        # Conv2D + AveragePooling blocks
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = self.relu(self.conv4(x))
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x