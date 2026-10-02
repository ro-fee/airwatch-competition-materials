"""Deterministic in-memory perturbations for bearing robustness evaluation.

These helpers never modify the source dataset.  Every operation returns a new
Tensor containing a perturbed copy of one fixed-length signal window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

PerturbationKind = Literal[
    "clean",
    "awgn_snr_db",
    "amplitude_scale",
    "circular_shift_samples",
    "time_scale",
]


@dataclass(frozen=True)
class BearingPerturbation:
    """One named, reproducible pressure-test condition."""

    condition_id: str
    kind: PerturbationKind
    value: float | int | None
    description: str


def _require_numeric_value(condition: BearingPerturbation) -> float:
    value = condition.value
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"perturbation {condition.condition_id!r} requires a numeric value"
        )
    numeric = float(value)
    if not torch.isfinite(torch.tensor(numeric)):
        raise ValueError(
            f"perturbation {condition.condition_id!r} requires a finite value"
        )
    return numeric


def _validate_window(window: torch.Tensor) -> None:
    if not isinstance(window, torch.Tensor):
        raise TypeError("window must be a torch.Tensor")
    if window.ndim != 2:
        raise ValueError(
            f"bearing window must have shape [channels, length], got {tuple(window.shape)}"
        )
    if window.shape[-1] < 2:
        raise ValueError("bearing window length must be at least 2")
    if not window.is_floating_point():
        raise ValueError("bearing window must use a floating-point dtype")
    if not bool(torch.isfinite(window).all()):
        raise ValueError("bearing window contains non-finite values")


def _sample_seed(seed: int, sample_index: int) -> int:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    # Keep the mapping stable across batch sizes and evaluation order while
    # avoiding the highly correlated seed + index pattern.
    mixed = (int(seed) & 0x7FFF_FFFF_FFFF_FFFF) ^ (
        (int(sample_index) + 1) * 0x9E37_79B9_7F4A_7C15
    )
    return mixed % (2**63 - 1)


def _add_awgn(
    window: torch.Tensor,
    *,
    snr_db: float,
    seed: int,
    sample_index: int,
) -> torch.Tensor:
    signal_power = window.square().mean()
    if float(signal_power.item()) <= 0.0:
        raise ValueError("cannot apply an SNR condition to a zero-power window")

    generator = torch.Generator(device=window.device)
    generator.manual_seed(_sample_seed(seed, sample_index))
    noise = torch.randn(
        window.shape,
        dtype=window.dtype,
        device=window.device,
        generator=generator,
    )
    noise_rms = noise.square().mean().sqrt()
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = noise * (target_noise_power.sqrt() / noise_rms)
    return window.clone() + noise


def _time_scale(window: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError("time_scale must be greater than zero")
    original_length = int(window.shape[-1])
    intermediate_length = max(2, int(round(original_length / scale)))
    batched = window.unsqueeze(0)
    scaled = F.interpolate(
        batched,
        size=intermediate_length,
        mode="linear",
        align_corners=False,
    )

    if intermediate_length >= original_length:
        # Take the centered interval so scaling does not introduce a systematic
        # phase offset at only one edge.
        start = (intermediate_length - original_length) // 2
        result = scaled[..., start : start + original_length]
    else:
        # Restore the model contract while retaining the interpolation-induced
        # compression/expansion pattern as a mild speed-drift approximation.
        result = F.interpolate(
            scaled,
            size=original_length,
            mode="linear",
            align_corners=False,
        )
    return result.squeeze(0).clone()


def apply_bearing_perturbation(
    window: torch.Tensor,
    condition: BearingPerturbation,
    *,
    seed: int,
    sample_index: int,
) -> torch.Tensor:
    """Return a deterministic perturbed copy of one ``[channels, length]`` window.

    ``time_scale`` is intentionally a simple resampling approximation of mild
    rotational-speed/frequency drift.  It is not a bearing-system simulator.
    """

    _validate_window(window)
    kind = condition.kind
    if kind == "clean":
        if condition.value is not None:
            raise ValueError("clean perturbation value must be None")
        return window.clone()
    if kind == "awgn_snr_db":
        return _add_awgn(
            window,
            snr_db=_require_numeric_value(condition),
            seed=seed,
            sample_index=sample_index,
        )
    if kind == "amplitude_scale":
        scale = _require_numeric_value(condition)
        if scale < 0.0:
            raise ValueError("amplitude_scale must be non-negative")
        return window.clone() * scale
    if kind == "circular_shift_samples":
        shift_value = _require_numeric_value(condition)
        if not shift_value.is_integer():
            raise ValueError("circular_shift_samples must be an integer")
        return torch.roll(window.clone(), shifts=int(shift_value), dims=-1)
    if kind == "time_scale":
        return _time_scale(window, _require_numeric_value(condition))
    raise ValueError(f"unsupported perturbation kind: {kind!r}")
