
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        hidden_channels = max(1, channels // reduction)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape

        weights = self.pool(x)
        weights = weights.view(batch_size, channels)
        weights = self.fc(weights)
        weights = weights.view(batch_size, channels, 1, 1)

        return x * weights
