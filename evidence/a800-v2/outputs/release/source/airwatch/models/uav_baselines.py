"""ResNet and TCN baselines sharing an explicit DroneRF channel contract."""
from __future__ import annotations

import torch
from torch import nn


def _check_input(inputs: torch.Tensor, model_name: str, in_channels: int) -> None:
    if inputs.ndim != 3 or inputs.shape[1] != in_channels:
        raise ValueError(
            f"{model_name} expects [batch, {in_channels}, length], got {tuple(inputs.shape)}"
        )


class _ResidualBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.projection = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        outputs = self.activation(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.activation(outputs + residual)


class DroneRFResNet18(nn.Module):
    """One-dimensional ResNet-18 baseline for selected RF bands."""

    def __init__(self, num_classes: int = 4, in_channels: int = 2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if in_channels not in {1, 2}:
            raise ValueError("in_channels must be 1 or 2")
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._stage(32, 32, blocks=2, stride=1)
        self.stage2 = self._stage(32, 64, blocks=2, stride=2)
        self.stage3 = self._stage(64, 128, blocks=2, stride=2)
        self.stage4 = self._stage(128, 256, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_classes)

    @staticmethod
    def _stage(in_channels: int, out_channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        layers = [_ResidualBlock1D(in_channels, out_channels, stride)]
        layers.extend(_ResidualBlock1D(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        _check_input(inputs, type(self).__name__, self.in_channels)
        outputs = self.stem(inputs)
        outputs = self.stage4(self.stage3(self.stage2(self.stage1(outputs))))
        return self.pool(outputs).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


class _CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        self.causal_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels, out_channels, kernel_size,
            padding=self.causal_padding, dilation=dilation, bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = super().forward(inputs)
        return outputs[..., :-self.causal_padding] if self.causal_padding else outputs


class _TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _CausalConv1d(in_channels, out_channels, 5, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            _CausalConv1d(out_channels, out_channels, 5, dilation),
            nn.BatchNorm1d(out_channels),
        )
        self.projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(inputs) + self.projection(inputs))


class DroneRFTCN(nn.Module):
    """Dilated causal temporal-convolution baseline for selected RF bands."""

    def __init__(self, num_classes: int = 4, in_channels: int = 2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if in_channels not in {1, 2}:
            raise ValueError("in_channels must be 1 or 2")
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            _TCNBlock(32, 64, dilation=1),
            nn.MaxPool1d(2),
            _TCNBlock(64, 128, dilation=2),
            nn.MaxPool1d(2),
            _TCNBlock(128, 128, dilation=4),
            _TCNBlock(128, 256, dilation=8),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        _check_input(inputs, type(self).__name__, self.in_channels)
        return self.pool(self.temporal(self.stem(inputs))).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


__all__ = ["DroneRFResNet18", "DroneRFTCN"]
