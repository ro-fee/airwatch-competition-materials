"""Deterministic perturbations for source-backed RF robustness experiments."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import Dataset


class ControlledNoiseError(ValueError):
    """Raised when a controlled-noise request is not numerically valid."""


def add_awgn_at_relative_snr(
    signal: torch.Tensor,
    *,
    snr_db: float,
    seed: int,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise at exact empirical signal/noise power ratio.

    The input is expected to be one normalized window shaped ``[channels, samples]``.
    The requested value is relative to the already captured window power; it is not
    an estimate of the physical acquisition SNR.
    """
    if signal.ndim != 2 or signal.shape[0] < 1 or signal.shape[1] < 2:
        raise ControlledNoiseError(f"expected [channels, samples], got {tuple(signal.shape)}")
    if not signal.is_floating_point() or not torch.isfinite(signal).all():
        raise ControlledNoiseError("signal must be finite floating-point values")
    if not math.isfinite(float(snr_db)) or not -(100.0) <= float(snr_db) <= 100.0:
        raise ControlledNoiseError("snr_db must be finite and between -100 and 100")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    noise = torch.randn(signal.shape, generator=generator, dtype=torch.float32)
    noise = noise - noise.mean(dim=1, keepdim=True)
    noise_power = noise.square().mean(dim=1, keepdim=True)
    if torch.any(noise_power <= 0):
        raise ControlledNoiseError("generated noise has zero power")
    unit_noise = noise / noise_power.sqrt()
    signal_float = signal.to(dtype=torch.float32)
    signal_power = signal_float.square().mean(dim=1, keepdim=True)
    if torch.any(signal_power <= 0):
        raise ControlledNoiseError("signal has zero power")
    ratio = 10.0 ** (float(snr_db) / 10.0)
    scaled_noise = unit_noise * (signal_power / ratio).sqrt()
    return signal_float + scaled_noise


def sample_noise_seed(base_seed: int, sample_index: int, snr_db: float) -> int:
    """Return a stable per-window, per-condition seed independent of batch order."""
    if sample_index < 0:
        raise ControlledNoiseError("sample_index must be non-negative")
    condition_code = int(round((float(snr_db) + 100.0) * 1000.0))
    return (
        int(base_seed) * 1_000_003
        + int(sample_index) * 97_409
        + condition_code * 65_537
    ) % (2**63 - 1)


class ControlledAWGNDataset(Dataset):
    """Read-only dataset view that adds deterministic AWGN after normalization."""

    def __init__(self, dataset: Dataset, *, snr_db: float, base_seed: int) -> None:
        self.dataset = dataset
        self.snr_db = float(snr_db)
        self.base_seed = int(base_seed)
        if not math.isfinite(self.snr_db):
            raise ControlledNoiseError("snr_db must be finite")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        signal, target, dataset_index = self.dataset[index]
        seed = sample_noise_seed(self.base_seed, int(dataset_index), self.snr_db)
        return (
            add_awgn_at_relative_snr(signal, snr_db=self.snr_db, seed=seed),
            target,
            dataset_index,
        )

    def sample_metadata(self, index: int) -> Any:
        return self.dataset.sample_metadata(index)

    @property
    def label_map(self) -> dict[str, int]:
        return self.dataset.label_map

    @property
    def recording_ids(self) -> set[str]:
        return self.dataset.recording_ids


class DeterministicNoiseAugmentedDataset(Dataset):
    """Epoch-specific training view mixing clean and deterministic noisy windows."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        snr_db_levels: list[float] | tuple[float, ...],
        noise_probability: float,
        base_seed: int,
        epoch: int,
    ) -> None:
        if not snr_db_levels or any(not math.isfinite(float(value)) for value in snr_db_levels):
            raise ControlledNoiseError("snr_db_levels must contain finite values")
        if not 0.0 <= float(noise_probability) <= 1.0:
            raise ControlledNoiseError("noise_probability must be between 0 and 1")
        if epoch < 1:
            raise ControlledNoiseError("epoch must be at least 1")
        self.dataset = dataset
        self.snr_db_levels = tuple(float(value) for value in snr_db_levels)
        self.noise_probability = float(noise_probability)
        self.base_seed = int(base_seed)
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        signal, target, dataset_index = self.dataset[index]
        decision_seed = sample_noise_seed(
            self.base_seed + self.epoch * 1_000_033, int(dataset_index), 0.0
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(decision_seed)
        if float(torch.rand((), generator=generator).item()) >= self.noise_probability:
            return signal.clone(), target, dataset_index
        level_index = int(
            torch.randint(len(self.snr_db_levels), (), generator=generator).item()
        )
        snr_db = self.snr_db_levels[level_index]
        noise_seed = sample_noise_seed(decision_seed, int(dataset_index), snr_db)
        return (
            add_awgn_at_relative_snr(signal, snr_db=snr_db, seed=noise_seed),
            target,
            dataset_index,
        )


__all__ = [
    "ControlledAWGNDataset",
    "ControlledNoiseError",
    "DeterministicNoiseAugmentedDataset",
    "add_awgn_at_relative_snr",
    "sample_noise_seed",
]
