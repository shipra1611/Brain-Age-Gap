"""
ResNet-18 based CNN for brain age regression
"""

import torch
import torch.nn as nn
import torchvision.models as models

class BrainAgeCNN(nn.Module):
    def __init__(self, pretrained=True, dropout=0.4, num_input_channels=3):
        """
        CNN for brain age prediction
        
        Args:
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout probability
            num_input_channels: Input channels (3 for RGB-like, 1 for grayscale)
        """
        super(BrainAgeCNN, self).__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv if using single channel
        if num_input_channels == 1:
            self.resnet.conv1 = nn.Conv2d(
                1, 64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
        
        # Replace final FC with regression head
        num_features = self.resnet.fc.in_features  # 512 for ResNet-18
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1)  # Single output: normalized age
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Predicted normalized age (B,)
        """
        output = self.resnet(x)
        return output.squeeze(-1)  # Remove last dimension


class BrainAgeCNNLight(nn.Module):
    """Lighter CNN architecture for comparison"""
    
    def __init__(self, dropout=0.4, num_input_channels=3):
        super(BrainAgeCNNLight, self).__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)