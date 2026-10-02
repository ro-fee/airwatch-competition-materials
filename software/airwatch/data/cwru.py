"""Read-only CWRU MAT signal loading and deterministic window slicing.

This module deliberately does not train models, create derived datasets, or
modify source MAT files.  It provides the small data boundary that future
training/inference code can share.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import loadmat

Normalization = Literal["none", "zscore", "window_zscore"]


class CWRUDataError(ValueError):
    """Raised when a CWRU file cannot be safely interpreted as one signal."""


class AmbiguousSignalError(CWRUDataError):
    """Raised when a MAT file contains multiple candidate DE channels."""


@dataclass(frozen=True)
class SignalRecord:
    """A read-only representation of one selected CWRU DE signal."""

    path: Path
    sensor_key: str
    signal: np.ndarray
    source_shape: tuple[int, ...]
    normalization: Normalization


def _validate_signal_array(value: object, *, source: str) -> tuple[np.ndarray, tuple[int, ...]]:
    """Convert one MATLAB value to a finite, one-dimensional float32 array."""
    array = np.asarray(value)
    source_shape = tuple(int(dimension) for dimension in array.shape)

    if not np.issubdtype(array.dtype, np.number):
        raise CWRUDataError(f"{source} is not numeric: dtype={array.dtype}")

    # CWRU stores channels as N-by-1 (or 1-by-N).  Squeeze only singleton
    # dimensions; do not silently flatten a genuine multi-dimensional matrix.
    squeezed = np.squeeze(array)
    if squeezed.ndim != 1:
        raise CWRUDataError(
            f"{source} is not a single channel: original_shape={source_shape}, "
            f"squeezed_shape={tuple(squeezed.shape)}"
        )

    signal = np.asarray(squeezed, dtype=np.float32).copy()
    if signal.size == 0:
        raise CWRUDataError(f"{source} is empty")
    if not np.isfinite(signal).all():
        raise CWRUDataError(f"{source} contains NaN or infinite values")
    return signal, source_shape


def _apply_normalization(signal: np.ndarray, normalization: Normalization) -> np.ndarray:
    if normalization == "none":
        return signal.copy()
    if normalization == "window_zscore":
        raise CWRUDataError("window_zscore must be applied to windows after slicing")
    if normalization == "zscore":
        mean = float(signal.mean())
        std = float(signal.std())
        if std == 0.0:
            raise CWRUDataError("cannot z-score a constant signal")
        return ((signal - mean) / std).astype(np.float32, copy=False)
    raise CWRUDataError(f"unsupported normalization: {normalization!r}")


def normalize_windows(
    windows: np.ndarray,
    *,
    normalization: Normalization = "none",
) -> np.ndarray:
    """Return independently normalized signal windows as float32 data."""
    array = np.asarray(windows, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise CWRUDataError("windows must be a non-empty 2D matrix")
    if not np.isfinite(array).all():
        raise CWRUDataError("windows contain NaN or infinite values")
    if normalization == "none":
        normalized = array.copy()
    elif normalization == "window_zscore":
        means = array.mean(axis=1, keepdims=True)
        standard_deviations = array.std(axis=1, keepdims=True)
        if np.any(standard_deviations == 0.0):
            raise CWRUDataError("cannot z-score a constant window")
        normalized = np.asarray((array - means) / standard_deviations, dtype=np.float32)
    else:
        raise CWRUDataError(f"unsupported window normalization: {normalization!r}")
    normalized.setflags(write=False)
    return normalized


def _candidate_de_keys(mat: dict[str, object]) -> list[str]:
    return sorted(key for key in mat if key.endswith("_DE_time"))


def load_cwru_signal(
    path: str | Path,
    *,
    sensor_key: str | None = None,
    normalization: Normalization = "none",
) -> SignalRecord:
    """Load one CWRU drive-end channel without modifying the source file.

    If a MAT file has more than one ``*_DE_time`` key, callers must explicitly
    select ``sensor_key``.  This is intentional: the downloaded ``Normal_2``
    file contains two DE channels and silently choosing one would make the
    experiment non-auditable.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise CWRUDataError(f"MAT file does not exist: {source_path}")
    if source_path.suffix.lower() != ".mat":
        raise CWRUDataError(f"expected a .mat file: {source_path}")

    try:
        mat = loadmat(source_path)
    except (OSError, ValueError) as exc:
        raise CWRUDataError(f"failed to read MAT file {source_path}: {exc}") from exc

    candidates = _candidate_de_keys(mat)
    if not candidates:
        raise CWRUDataError(f"no *_DE_time channel found in {source_path}")
    if sensor_key is None:
        if len(candidates) != 1:
            raise AmbiguousSignalError(
                f"multiple DE channels in {source_path.name}: {candidates}; "
                "pass sensor_key explicitly"
            )
        selected_key = candidates[0]
    else:
        selected_key = sensor_key
        if selected_key not in candidates:
            raise CWRUDataError(
                f"sensor_key={selected_key!r} is not a DE channel in "
                f"{source_path.name}; available={candidates}"
            )

    signal, source_shape = _validate_signal_array(mat[selected_key], source=selected_key)
    normalized = _apply_normalization(signal, normalization)
    normalized.setflags(write=False)
    return SignalRecord(
        path=source_path,
        sensor_key=selected_key,
        signal=normalized,
        source_shape=source_shape,
        normalization=normalization,
    )


def slice_signal(
    signal: np.ndarray,
    *,
    window_size: int,
    step: int | None = None,
) -> np.ndarray:
    """Return complete, non-padded windows from one signal.

    The result has shape ``(window_count, window_size)``.  Incomplete trailing
    samples are ignored, and no randomization or augmentation is performed.
    A zero-window result is returned when the recording is shorter than one
    requested window.
    """
    if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
        raise CWRUDataError("window_size must be a positive integer")
    if step is None:
        step = window_size
    if not isinstance(step, (int, np.integer)) or step <= 0:
        raise CWRUDataError("step must be a positive integer")

    validated, _ = _validate_signal_array(signal, source="signal")
    if validated.size < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    starts = range(0, validated.size - window_size + 1, step)
    windows = np.stack([validated[start : start + window_size] for start in starts])
    windows.setflags(write=False)
    return windows
