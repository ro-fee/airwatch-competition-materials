"""Signal-quality checks applied before bearing model inference.

The checks in this module operate on raw windows. They intentionally run
before any per-window normalization, because normalization would make a flat
or extremely weak signal look artificially well-scaled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_RECAPTURE_MESSAGE = "信号质量不足，建议重新采集"
_ACCEPTED_MESSAGE = "信号质量检查通过"
_CAUTION_MESSAGE = "信号质量一般，诊断结果仅供参考，建议复核或重新采集"
_DEFAULT_NEAR_CONSTANT_STD = float(np.finfo(np.float32).eps)


@dataclass(frozen=True)
class BearingQualityThresholds:
    """Dimensionless/scale-aware rejection limits used by the quality gate.

    ``near_constant_std`` is evaluated on the raw acquisition units. The two
    ratio thresholds are dimensionless. Production values must be selected
    from validation data and then frozen before test evaluation.
    """

    near_constant_std: float = _DEFAULT_NEAR_CONSTANT_STD
    max_clipping_ratio: float = 0.08
    max_spectral_flatness: float = 0.45

    def __post_init__(self) -> None:
        values = {
            "near_constant_std": self.near_constant_std,
            "max_clipping_ratio": self.max_clipping_ratio,
            "max_spectral_flatness": self.max_spectral_flatness,
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.near_constant_std < 0.0:
            raise ValueError("near_constant_std must be non-negative")
        for name in ("max_clipping_ratio", "max_spectral_flatness"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class BearingQualityThresholdsV2:
    """Warning and rejection limits for the three-level quality gate."""

    near_constant_std: float = _DEFAULT_NEAR_CONSTANT_STD
    warning_clipping_ratio: float = 0.04
    reject_clipping_ratio: float = 0.08
    warning_spectral_flatness: float = 0.35
    reject_spectral_flatness: float = 0.45

    def __post_init__(self) -> None:
        values = {
            "near_constant_std": self.near_constant_std,
            "warning_clipping_ratio": self.warning_clipping_ratio,
            "reject_clipping_ratio": self.reject_clipping_ratio,
            "warning_spectral_flatness": self.warning_spectral_flatness,
            "reject_spectral_flatness": self.reject_spectral_flatness,
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.near_constant_std < 0.0:
            raise ValueError("near_constant_std must be non-negative")
        for warning_name, reject_name in (
            ("warning_clipping_ratio", "reject_clipping_ratio"),
            ("warning_spectral_flatness", "reject_spectral_flatness"),
        ):
            warning = float(getattr(self, warning_name))
            reject = float(getattr(self, reject_name))
            if not 0.0 <= warning <= reject <= 1.0:
                raise ValueError(
                    f"thresholds must satisfy 0 <= {warning_name} <= "
                    f"{reject_name} <= 1"
                )


@dataclass(frozen=True)
class BearingQualityResult:
    """Structured result returned by a bearing signal-quality check."""

    accepted: bool
    reason_code: str
    message: str
    quality_score: float
    features: dict[str, float]
    status: str


def load_bearing_quality_thresholds(
    path: str | Path,
) -> BearingQualityThresholds:
    """Load frozen quality-gate thresholds from a validation artifact.

    The loader deliberately requires ``calibration_split == "validation"``.
    This prevents a caller from accidentally using test-derived thresholds
    for inference or evaluation. The analysis code does not choose a default
    artifact path; callers must explicitly select the evidence file they want
    to freeze into an inference configuration.
    """

    artifact_path = Path(path)
    try:
        payload: Any = json.loads(artifact_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"cannot read bearing quality calibration artifact: {artifact_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"bearing quality calibration artifact is not valid JSON: {artifact_path}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError("bearing quality calibration artifact must be a JSON object")
    if payload.get("calibration_split") != "validation":
        raise ValueError(
            "bearing quality thresholds must come from calibration_split='validation'"
        )

    selected = payload.get("selected_thresholds")
    if not isinstance(selected, Mapping):
        raise ValueError("selected_thresholds must be a JSON object")

    required_fields = (
        "near_constant_std",
        "max_clipping_ratio",
        "max_spectral_flatness",
    )
    missing = [name for name in required_fields if name not in selected]
    if missing:
        raise ValueError(
            "selected_thresholds is missing required fields: " + ", ".join(missing)
        )

    numeric_values: dict[str, float] = {}
    for name in required_fields:
        value = selected[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"selected_thresholds.{name} must be a number")
        numeric_values[name] = float(value)

    return BearingQualityThresholds(**numeric_values)


def load_bearing_quality_thresholds_v2(
    path: str | Path,
) -> BearingQualityThresholdsV2:
    """Load frozen V2 thresholds selected exclusively on validation data."""

    artifact_path = Path(path)
    try:
        payload: Any = json.loads(artifact_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"cannot read bearing quality V2 calibration artifact: {artifact_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"bearing quality V2 calibration artifact is not valid JSON: {artifact_path}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError("bearing quality V2 calibration artifact must be a JSON object")
    if payload.get("calibration_split") != "validation":
        raise ValueError(
            "bearing quality V2 thresholds must come from "
            "calibration_split='validation'"
        )

    selected = payload.get("selected_thresholds")
    if not isinstance(selected, Mapping):
        raise ValueError("selected_thresholds must be a JSON object")

    required_fields = (
        "near_constant_std",
        "warning_clipping_ratio",
        "reject_clipping_ratio",
        "warning_spectral_flatness",
        "reject_spectral_flatness",
    )
    missing = [name for name in required_fields if name not in selected]
    if missing:
        raise ValueError(
            "selected_thresholds is missing required fields: " + ", ".join(missing)
        )

    numeric_values: dict[str, float] = {}
    for name in required_fields:
        value = selected[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"selected_thresholds.{name} must be a number")
        numeric_values[name] = float(value)

    return BearingQualityThresholdsV2(**numeric_values)


def _spectral_flatness(array: np.ndarray) -> float:
    centered = array - float(array.mean())
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size > 1:
        power = power[1:]
    floor = max(float(power.max(initial=0.0)) * 1e-12, np.finfo(np.float64).tiny)
    stabilized = power + floor
    return float(np.exp(np.mean(np.log(stabilized))) / np.mean(stabilized))


def assess_bearing_window(
    signal: object,
    *,
    thresholds: BearingQualityThresholds | None = None,
) -> BearingQualityResult:
    """Assess one raw bearing window before normalization or inference."""

    active_thresholds = thresholds or BearingQualityThresholds()
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        return BearingQualityResult(
            accepted=False,
            reason_code="invalid_signal",
            message=_RECAPTURE_MESSAGE,
            quality_score=0.0,
            features={},
            status="rejected",
        )
    if not np.all(np.isfinite(array)):
        return BearingQualityResult(
            accepted=False,
            reason_code="invalid_signal",
            message=_RECAPTURE_MESSAGE,
            quality_score=0.0,
            features={},
            status="rejected",
        )

    standard_deviation = float(array.std())
    rms = float(np.sqrt(np.mean(np.square(array))))
    peak_to_peak = float(np.ptp(array))
    extrema_tolerance = max(
        peak_to_peak * 1e-6,
        _DEFAULT_NEAR_CONSTANT_STD,
    )
    clipping_ratio = float(
        np.mean(
            np.isclose(array, float(array.min()), rtol=0.0, atol=extrema_tolerance)
            | np.isclose(array, float(array.max()), rtol=0.0, atol=extrema_tolerance)
        )
    )
    spectral_flatness = _spectral_flatness(array)
    features = {
        "standard_deviation": standard_deviation,
        "rms": rms,
        "peak_to_peak": peak_to_peak,
        "clipping_ratio": clipping_ratio,
        "spectral_flatness": spectral_flatness,
    }

    if standard_deviation <= active_thresholds.near_constant_std:
        return BearingQualityResult(
            accepted=False,
            reason_code="near_constant",
            message=_RECAPTURE_MESSAGE,
            quality_score=0.0,
            features=features,
            status="rejected",
        )

    if clipping_ratio > active_thresholds.max_clipping_ratio:
        return BearingQualityResult(
            accepted=False,
            reason_code="clipped",
            message=_RECAPTURE_MESSAGE,
            quality_score=0.0,
            features=features,
            status="rejected",
        )

    if spectral_flatness > active_thresholds.max_spectral_flatness:
        return BearingQualityResult(
            accepted=False,
            reason_code="excessive_noise",
            message=_RECAPTURE_MESSAGE,
            quality_score=0.0,
            features=features,
            status="rejected",
        )

    return BearingQualityResult(
        accepted=True,
        reason_code="accepted",
        message=_ACCEPTED_MESSAGE,
        quality_score=1.0,
        features=features,
        status="accepted",
    )


def assess_bearing_window_v2(
    signal: object,
    *,
    thresholds: BearingQualityThresholdsV2 | None = None,
) -> BearingQualityResult:
    """Assess one raw window using three-level quality semantics."""

    active = thresholds or BearingQualityThresholdsV2()
    result = assess_bearing_window(
        signal,
        thresholds=BearingQualityThresholds(
            near_constant_std=active.near_constant_std,
            max_clipping_ratio=active.reject_clipping_ratio,
            max_spectral_flatness=active.reject_spectral_flatness,
        ),
    )
    if result.status == "rejected":
        return result
    if result.features["clipping_ratio"] > active.warning_clipping_ratio:
        return BearingQualityResult(
            accepted=True,
            reason_code="clipping_warning",
            message=_CAUTION_MESSAGE,
            quality_score=0.5,
            features=result.features,
            status="caution",
        )
    if result.features["spectral_flatness"] > active.warning_spectral_flatness:
        return BearingQualityResult(
            accepted=True,
            reason_code="noise_warning",
            message=_CAUTION_MESSAGE,
            quality_score=0.5,
            features=result.features,
            status="caution",
        )
    return result


