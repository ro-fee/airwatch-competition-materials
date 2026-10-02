"""Isolated multi-scale temporal model for the preregistered A800 V2 matrix.

The original TCN implementation and its checkpoint names remain unchanged.
Both candidates expose three known-source logits and 256-dimensional features;
these outputs do not imply UAV detection or open-set rejection capability.
"""
from __future__ import annotations

from numbers import Integral

import torch
from torch import nn

from airwatch.models.uav_baselines import DroneRFTCN, _CausalConv1d


class _MultiScaleTCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            _CausalConv1d(in_channels, out_channels // 4, kernel, dilation)
            for kernel in (3, 7, 15)
        )
        self.network = nn.Sequential(
            nn.Conv1d(3 * (out_channels // 4), out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            _CausalConv1d(out_channels, out_channels, 5, dilation),
            nn.BatchNorm1d(out_channels),
        )
        self.projection = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        multiscale = torch.cat([branch(inputs) for branch in self.branches], dim=1)
        return self.activation(self.network(multiscale) + self.projection(inputs))


def _validate_num_classes(num_classes: int) -> int:
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2")
    return int(num_classes)


class MultiScaleRFTCN(nn.Module):
    """Three causal kernel scales per residual block, with the frozen TCN stem.

    Formal data uses [batch, 2, 4096]. Shorter lengths of at least 16 samples
    are also accepted for CPU engineering tests. The symmetric stem and
    training-mode batch normalization are not causal streaming operators.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 2) -> None:
        super().__init__()
        num_classes = _validate_num_classes(num_classes)
        if isinstance(in_channels, bool) or in_channels != 2:
            raise ValueError("A800 multi-scale TCN requires in_channels=2 for complex IQ")
        self.in_channels = 2
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, 15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            _MultiScaleTCNBlock(32, 64, dilation=1),
            nn.MaxPool1d(2),
            _MultiScaleTCNBlock(64, 128, dilation=2),
            nn.MaxPool1d(2),
            _MultiScaleTCNBlock(128, 128, dilation=4),
            _MultiScaleTCNBlock(128, 256, dilation=8),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValueError("MultiScaleRFTCN expects [batch, 2, length] IQ inputs")
        if inputs.shape[0] < 1 or inputs.shape[2] < 16:
            raise ValueError("MultiScaleRFTCN requires a nonempty batch and length >= 16")
        return self.pool(self.temporal(self.stem(inputs))).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


def build_a800_model(name: str, num_classes: int = 3) -> nn.Module:
    """Build exactly one registered A800 model without changing legacy weights."""
    num_classes = _validate_num_classes(num_classes)
    if name == "tcn":
        return DroneRFTCN(num_classes=num_classes, in_channels=2)
    if name == "multiscale_tcn":
        return MultiScaleRFTCN(num_classes=num_classes)
    raise ValueError(f"unregistered A800 model: {name!r}")


__all__ = ["MultiScaleRFTCN", "build_a800_model"]
