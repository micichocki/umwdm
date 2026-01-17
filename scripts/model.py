import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):    
        super().__init__()
        
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        return self.block(x)

class CNN(nn.Module):                                             
    def __init__(self, in_channels=1, num_labels=14):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=7),   
            ConvBlock(32, 64, kernel_size=5),             
            ConvBlock(64, 128, kernel_size=5),            
            ConvBlock(128, 256, kernel_size=3),           
            ConvBlock(256, 512, kernel_size=3),           
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_labels)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
