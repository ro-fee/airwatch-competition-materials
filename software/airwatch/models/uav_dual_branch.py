"""Predeclared time-frequency fusion models for DroneRF ablation studies."""
from __future__ import annotations

import torch
from torch import nn

from .uav_baselines import DroneRFTCN


class DroneRFSpectralBackbone(nn.Module):
    """Deterministic STFT log-magnitude branch followed by a compact 2-D CNN."""

    feature_dim = 192

    def __init__(self, n_fft: int = 256, hop_length: int = 64) -> None:
        super().__init__()
        if n_fft < 32 or hop_length < 1 or hop_length > n_fft:
            raise ValueError("invalid STFT settings")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("stft_window", torch.hann_window(n_fft), persistent=True)
        self.network = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def spectrogram(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValueError(
                f"DroneRFSpectralBackbone expects [batch, 2, length], got {tuple(inputs.shape)}"
            )
        batch, channels, length = inputs.shape
        if length < self.n_fft:
            raise ValueError("input is shorter than n_fft")
        flattened = inputs.reshape(batch * channels, length)
        spectrum = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.stft_window,
            center=False,
            return_complex=True,
        )
        magnitude = torch.log1p(spectrum.abs())
        return magnitude.reshape(batch, channels, magnitude.shape[-2], magnitude.shape[-1])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(self.spectrogram(inputs)).flatten(1)


class DroneRFSpectralCNN(nn.Module):
    """Frequency-only ablation with the same type-classification contract."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = DroneRFSpectralBackbone()
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


class DroneRFDualBranch(nn.Module):
    """TCN time branch plus STFT-CNN branch with fixed or learned fusion."""

    feature_dim = 192

    def __init__(
        self,
        num_classes: int = 4,
        *,
        fusion_mode: str = "adaptive",
        presence_head: bool = False,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"concat", "adaptive"}:
            raise ValueError("fusion_mode must be concat or adaptive")
        time_model = DroneRFTCN(num_classes=num_classes)
        time_model.classifier = nn.Identity()
        self.time_backbone = time_model
        self.spectral_backbone = DroneRFSpectralBackbone()
        self.time_projection = nn.Sequential(
            nn.Linear(256, self.feature_dim), nn.LayerNorm(self.feature_dim), nn.ReLU()
        )
        self.frequency_projection = nn.Sequential(
            nn.Linear(self.spectral_backbone.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim), nn.ReLU(),
        )
        self.fusion_mode = fusion_mode
        self.has_presence_head = presence_head
        if fusion_mode == "concat":
            self.concat_fusion = nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.ReLU(),
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(self.feature_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        self.type_head = nn.Linear(self.feature_dim, num_classes)
        self.presence_head = nn.Linear(self.feature_dim, 2) if presence_head else None

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        time_features = self.time_projection(self.time_backbone.forward_features(inputs))
        frequency_features = self.frequency_projection(self.spectral_backbone(inputs))
        joined = torch.cat((time_features, frequency_features), dim=1)
        if self.fusion_mode == "concat":
            fused = self.concat_fusion(joined)
            fusion_weight = torch.full(
                (inputs.shape[0], 1), 0.5, dtype=inputs.dtype, device=inputs.device
            )
        else:
            fusion_weight = self.gate(joined)
            fused = fusion_weight * time_features + (1.0 - fusion_weight) * frequency_features
        outputs = {
            "type_logits": self.type_head(fused),
            "fusion_weight_time": fusion_weight,
        }
        if self.presence_head is not None:
            outputs["presence_logits"] = self.presence_head(fused)
        return outputs


__all__ = ["DroneRFDualBranch", "DroneRFSpectralBackbone", "DroneRFSpectralCNN"]
