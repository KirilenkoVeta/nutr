import torch
from torch import nn

from .resblock import ResnetBlock


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0
        )

    def forward(self, x):
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, 
                in_channels, 
                hidden_channels, 
                downsample_steps_before=4, 
                conv_steps=3, 
                downsample_steps_after=2,
                dropout=0.15):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(
                in_channels, 
                hidden_channels, 
                kernel_size=5, 
                stride=1, 
                padding=2
            )
        ])
        for _ in range(downsample_steps_before):
            self.layers.append(
                nn.Sequential(
                    ResnetBlock(hidden_channels, hidden_channels, dropout),
                    Downsample(hidden_channels)
                )
            )
        for _ in range(conv_steps):
            self.layers.append(
                ResnetBlock(hidden_channels, hidden_channels, dropout)
            )
        for _ in range(downsample_steps_after):
            self.layers.append(
                Downsample(hidden_channels)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
