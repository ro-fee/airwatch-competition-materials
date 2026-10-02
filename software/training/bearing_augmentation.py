"""Reproducible training-time augmentation for bearing signal windows.

This module is used only by the training split. Validation and test datasets
continue to expose audited, unmodified windows. Augmentation is deterministic
for (seed, epoch, sample_index) and always returns a new tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from airwatch.analysis.bearing_robustness import (
    BearingPerturbation,
    apply_bearing_perturbation,
)
from airwatch.data.cwru import CWRUDataError, normalize_windows


@dataclass(frozen=True)
class TrainingAugmentationConfig:
    """Configuration for deterministic clean/AWGN training augmentation."""

    clean_probability: float
    awgn_snr_db: tuple[float, ...]
    seed: int

    def __post_init__(self) -> None:
        probability = float(self.clean_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("clean_probability must be between 0 and 1")
        if not self.awgn_snr_db:
            raise ValueError("awgn_snr_db must contain at least one SNR value")
        for value in self.awgn_snr_db:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("awgn_snr_db must contain numeric values")
            if not bool(torch.isfinite(torch.tensor(float(value)))):
                raise ValueError("awgn_snr_db must contain finite values")


def _mix_seed(seed: int, epoch: int, sample_index: int) -> int:
    """Derive one stable, non-negative seed without global RNG state."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")

    value = int(seed) & 0xFFFF_FFFF_FFFF_FFFF
    value = (value + (int(epoch) + 1) * 0x9E37_79B9_7F4A_7C15) & 0xFFFF_FFFF_FFFF_FFFF
    value = (value + (int(sample_index) + 1) * 0xBF58_476D_1CE4_E5B9) & 0xFFFF_FFFF_FFFF_FFFF
    value ^= value >> 30
    value = (value * 0xBF58_476D_1CE4_E5B9) & 0xFFFF_FFFF_FFFF_FFFF
    value ^= value >> 27
    value = (value * 0x94D0_49BB_1331_11EB) & 0xFFFF_FFFF_FFFF_FFFF
    value ^= value >> 31
    return value % (2**63 - 1)


def _rng_for(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def apply_training_augmentation(
    signal: torch.Tensor,
    config: TrainingAugmentationConfig,
    *,
    epoch: int,
    sample_index: int,
) -> torch.Tensor:
    """Return one deterministic clean-or-AWGN copy of a signal window."""
    if not isinstance(signal, torch.Tensor):
        raise TypeError("signal must be a torch.Tensor")
    if signal.ndim != 2:
        raise ValueError(
            f"signal must have shape [channels, length], got {tuple(signal.shape)}"
        )
    if not signal.is_floating_point():
        raise ValueError("signal must use a floating-point dtype")
    if not bool(torch.isfinite(signal).all()):
        raise ValueError("signal contains NaN or infinite values")

    derived_seed = _mix_seed(config.seed, epoch, sample_index)
    generator = _rng_for(derived_seed)
    clean_draw = float(torch.rand((), generator=generator).item())
    if clean_draw < float(config.clean_probability):
        return signal.clone()

    snr_index = int(
        torch.randint(
            low=0,
            high=len(config.awgn_snr_db),
            size=(),
            generator=generator,
        ).item()
    )
    condition = BearingPerturbation(
        condition_id=f"training_awgn_{config.awgn_snr_db[snr_index]:g}db",
        kind="awgn_snr_db",
        value=float(config.awgn_snr_db[snr_index]),
        description="training-time additive white Gaussian noise",
    )
    return apply_bearing_perturbation(
        signal,
        condition,
        seed=derived_seed,
        sample_index=0,
    )


class AugmentedBearingDataset(Dataset[tuple[torch.Tensor, int]]):
    """Wrap a base dataset with training-only augmentation and normalization."""

    def __init__(
        self,
        base_dataset: Dataset[tuple[torch.Tensor, int]],
        *,
        config: TrainingAugmentationConfig,
        normalization: str = "window_zscore",
    ) -> None:
        if normalization not in {"none", "window_zscore"}:
            raise ValueError(
                "AugmentedBearingDataset supports normalization='none' or "
                "'window_zscore'"
            )
        self.base_dataset = base_dataset
        self.config = config
        self.normalization = normalization
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        signal, target = self.base_dataset[index]
        augmented = apply_training_augmentation(
            signal,
            self.config,
            epoch=self._epoch,
            sample_index=index,
        )
        if self.normalization == "window_zscore":
            try:
                normalized = normalize_windows(
                    augmented.detach().cpu().numpy(),
                    normalization="window_zscore",
                )
            except CWRUDataError as exc:
                raise ValueError(
                    f"cannot normalize augmented bearing window at index={index}: {exc}"
                ) from exc
            augmented = torch.from_numpy(normalized.copy())
        return augmented, int(target)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic augmentation stream for a training epoch."""
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def __getattr__(self, name: str) -> Any:
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)
