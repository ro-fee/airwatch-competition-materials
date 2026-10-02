"""Small one-dimensional CNN baseline for CWRU bearing classification."""

from __future__ import annotations

import torch
from torch import nn


class BearingCNN(nn.Module):
    """Compact four-class 1-D CNN baseline.

    Input shape is ``[batch, 1, signal_length]`` and the output is logits with
    shape ``[batch, num_classes]``.  The model is intentionally independent of
    the historical ``models/`` package and of Qt.
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 1) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if in_channels < 1:
            raise ValueError("in_channels must be at least 1")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"BearingCNN expects [batch, channels, length], got {tuple(inputs.shape)}"
            )
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channel(s), got {inputs.shape[1]}"
            )
        features = self.features(inputs)
        return self.classifier(features.flatten(1))
