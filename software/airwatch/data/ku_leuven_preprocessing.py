"""Frozen preprocessing contract for KU Leuven complex-IQ windows."""

from __future__ import annotations

import numpy as np


PREPROCESSING_ID = "ku-leuven-iq-dc-rms-v1"
INPUT_CHANNEL_SEMANTICS = ("in_phase", "quadrature")
INPUT_SAMPLE_RATE_HZ = 100_000_000
INPUT_WINDOW_SAMPLES = 4096
RMS_EPSILON = 1e-8


class KULeuvenPreprocessingError(ValueError):
    """Raised when a window cannot satisfy the frozen IQ contract."""


def preprocess_iq_window(samples: np.ndarray) -> np.ndarray:
    """Remove complex DC and normalize complex RMS without mutating input.

    The output is an immutable, C-contiguous ``float32`` array shaped
    ``[2, 4096]``. No information from other windows or dataset splits is used.
    """
    array = np.asarray(samples)
    if array.shape != (2, INPUT_WINDOW_SAMPLES):
        raise KULeuvenPreprocessingError(
            f"IQ 输入必须为 [2,{INPUT_WINDOW_SAMPLES}]，实际为 {array.shape}"
        )
    if array.dtype.kind not in "fiu":
        raise KULeuvenPreprocessingError("IQ 输入必须是实数数值数组")
    result = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise KULeuvenPreprocessingError("IQ 输入包含 NaN 或无穷值")
    result -= result.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    complex_power = np.square(result, dtype=np.float64).sum(axis=0).mean()
    rms = float(np.sqrt(complex_power))
    if not np.isfinite(rms) or rms <= RMS_EPSILON:
        raise KULeuvenPreprocessingError("IQ 窗口去直流后能量过低，拒绝归一化")
    result /= np.float32(rms)
    if not np.isfinite(result).all():
        raise KULeuvenPreprocessingError("IQ 归一化产生非有限值")
    result.setflags(write=False)
    return result


__all__ = [
    "INPUT_CHANNEL_SEMANTICS",
    "INPUT_SAMPLE_RATE_HZ",
    "INPUT_WINDOW_SAMPLES",
    "KULeuvenPreprocessingError",
    "PREPROCESSING_ID",
    "RMS_EPSILON",
    "preprocess_iq_window",
]
