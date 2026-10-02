"""Qt-free waterfall preparation, retaining the legacy relative-PSD convention."""
from dataclasses import dataclass

import numpy as np

from .signal_transforms import prepare_signal, compute_spectrogram


FLOOR_DB = -100.0
DISPLAY_FLOOR_DB = -80.0


def prepare_waterfall_signal(data):
    """Accept the historical single-channel row, then use canonical validation."""
    array = np.asarray(data)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return prepare_signal(array).samples


@dataclass(frozen=True)
class WaterfallView:
    frequencies: np.ndarray
    frame_times: np.ndarray
    spectra_db: np.ndarray
    hop_seconds: float
    two_sided: bool


def compute_waterfall(data, fs, n_fft, overlap_ratio=0.5):
    """Prepare frames; no Qt, files, timers, or playback state.

    Per-frame DC removal and recording-relative dB deliberately retain the old
    waterfall convention, unlike static spectrograms (which retain DC).
    """
    if isinstance(n_fft, (bool, np.bool_)) or not isinstance(n_fft, (int, np.integer)):
        raise ValueError('FFT 点数必须是整数。')
    if n_fft < 16 or n_fft & (n_fft - 1):
        raise ValueError('FFT 点数必须是至少为 16 的 2 的整数次幂。')
    overlap_ratio = float(overlap_ratio)
    if not np.isfinite(overlap_ratio) or not 0 <= overlap_ratio < 1:
        raise ValueError('频谱重叠率必须在 0（含）到 1（不含）之间。')
    noverlap = min(int(round(n_fft * overlap_ratio)), n_fft - 1)
    signal = prepare_waterfall_signal(data)
    view = compute_spectrogram(signal, sample_rate_hz=fs, nperseg=n_fft,
                               noverlap=noverlap, detrend='constant')
    if not np.isfinite(view.power).all():
        raise ValueError('信号幅值过大，频谱计算溢出，请检查数据量纲。')
    peak = float(np.max(view.power))
    if peak <= 0:
        relative = np.full(view.power.shape, FLOOR_DB, dtype=np.float32)
    else:
        relative = np.clip(10 * np.log10(np.maximum(view.power / peak,
                           np.finfo(np.float32).tiny)), FLOOR_DB, 0).astype(np.float32)
    relative.setflags(write=False)
    return WaterfallView(view.frequency_hz, view.time_s, relative,
                         view.time_step_s, view.two_sided)
