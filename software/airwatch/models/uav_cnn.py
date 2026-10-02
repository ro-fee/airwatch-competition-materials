"""One-dimensional CNN baseline for selected DroneRF bands."""
from __future__ import annotations

import torch
from torch import nn


class DroneRFCNN(nn.Module):
    """Compact baseline; input is ``[batch, channels, length]`` RF amplitude."""

    def __init__(self, num_classes: int = 4, in_channels: int = 2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if in_channels not in {1, 2}:
            raise ValueError("in_channels must be 1 or 2")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(256, num_classes))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"DroneRFCNN expects [batch, channels, length], got {tuple(inputs.shape)}"
            )
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {inputs.shape[1]}"
            )
        return self.features(inputs).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


__all__ = ["DroneRFCNN"]
