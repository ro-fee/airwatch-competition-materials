"""Validated real-signal multiresolution analysis, independent of Qt and files."""
from dataclasses import dataclass
import numpy as np
import pywt
from .signal_transforms import prepare_signal, _validate_sample_rate, _readonly


@dataclass(frozen=True)
class WaveletView:
    time_s: np.ndarray
    signal: np.ndarray
    approximations: tuple
    details: tuple
    level: int
    wavelet: str = 'db4'
    mode: str = 'smooth'


def prepare_wavelet_signal(data):
    array = np.asarray(data)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    prepared = prepare_signal(array)
    if prepared.kind != 'real':
        raise ValueError('小波分析目前仅支持实信号；不能偷偷丢弃 I/Q 通道。')
    return prepared.samples


def compute_wavelet(data, sample_rate_hz, *, start_sample=0):
    signal = prepare_wavelet_signal(data)
    fs = _validate_sample_rate(sample_rate_hz)
    if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
        raise ValueError('选区起点必须是非负整数采样点。')
    if signal.size > 65536:
        raise ValueError('本次小波显示最多 65536 点，请在时域图缩小选区。')
    wavelet = pywt.Wavelet('db4')
    level = min(5, pywt.dwt_max_level(signal.size, wavelet.dec_len))
    if level < 1:
        raise ValueError('db4 小波至少需要 14 个采样点，请扩大选区。')
    # PyWavelets' Cython kernels require a writable buffer in the supported runtime.
    a = signal.copy()
    approximations, details = [], []
    for i in range(level):
        a, d = pywt.dwt(a, wavelet, mode='smooth')
        # Use the same boundary extension for decomposition and reconstruction.
        ra = pywt.waverec([a, None] + [None] * i, wavelet, mode='smooth')[:signal.size]
        rd = pywt.waverec([None, d] + [None] * i, wavelet, mode='smooth')[:signal.size]
        if not np.isfinite(ra).all() or not np.isfinite(rd).all():
            raise ValueError('小波计算溢出，请检查输入幅值。')
        approximations.append(_readonly(ra))
        details.append(_readonly(rd))
    return WaveletView(_readonly((np.arange(signal.size) + start_sample) / fs),
                       signal, tuple(approximations), tuple(details), level)
