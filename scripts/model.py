import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, stride=1):    
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)

class CNN(nn.Module):                                   
    def __init__(self, in_channels=3, num_labels=14):
        super().__init__()

        act_name='GELU'
        base_channels=64
        dropout_rate=0.2      
        
        self.features = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=3, padding=1, stride=1),   
            ConvBlock(base_channels, base_channels*2, kernel_size=3, padding=1, stride=2),            
            ConvBlock(base_channels*2, base_channels*4, kernel_size=3, padding=1, stride=2),           
            ConvBlock(base_channels*4, base_channels*8, kernel_size=3, padding=1, stride=2),          
            ConvBlock(base_channels*8, base_channels*16, kernel_size=3, padding=1, stride=2),
            ConvBlock(base_channels*16, base_channels*16, kernel_size=3, padding=1, stride=2),          
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels*16, num_labels)
        )    

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
