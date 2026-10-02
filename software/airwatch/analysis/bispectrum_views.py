"""Legacy indirect bispectrum estimator extracted without changing its lag layout.

This is an exploratory third-order statistic, not normalized bicoherence or a
calibrated power spectrum. Preserve the historical 10 records / 20 lags / 128 FFT.
"""
from dataclasses import dataclass
import numpy as np
from .signal_transforms import prepare_signal, _readonly, _validate_sample_rate


@dataclass(frozen=True)
class BispectrumView:
    magnitude: np.ndarray
    frequency_hz: np.ndarray
    cumulant: np.ndarray
    start_s: float
    end_s: float
    used_samples: int
    dropped_samples: int
    sample_rate_hz: float


def prepare_bispectrum_signal(data):
    array = np.asarray(data)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    prepared = prepare_signal(array)
    if prepared.kind != 'real':
        raise ValueError('双谱目前仅支持实信号，不能丢弃 I/Q 通道。')
    return prepared.samples


def compute_bispectrum(data, sample_rate_hz, *, start_sample=0):
    signal = prepare_bispectrum_signal(data)
    fs = _validate_sample_rate(sample_rate_hz)
    if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
        raise ValueError('选区起点必须是非负整数采样点。')
    if not 210 <= len(signal) <= 65536:
        raise ValueError('双谱选区须为 210–65536 点（10 段，每段超过 20 阶延迟）。')
    nlag, nfft, records = 20, 128, 10
    nsamp = len(signal) // records
    used = nsamp * records
    rows = signal[:used].reshape(records, nsamp)
    c3 = np.zeros((nlag + 1, nlag + 1), dtype=np.float64)
    with np.errstate(over='raise', invalid='raise'):
        for row in rows:
            x = row - np.mean(row)
            for j in range(nlag + 1):
                z = x[:nsamp-j] * x[j:]
                for i in range(j, nlag + 1):
                    c3[i,j] += np.dot(z[:nsamp-i], x[i:]) / nsamp
        c3 /= records
        c3 += np.tril(c3, -1).T
        c31 = c3[1:,1:]
        c32 = np.zeros((nlag,nlag))
        c33 = np.zeros((nlag,nlag))
        c34 = np.zeros((nlag,nlag))
        for i in range(nlag):
            x = c31[i:,i]
            c32[nlag-1-i,:nlag-i] = x
            c34[:nlag-i,nlag-1-i] = x
            if i < nlag-1:
                x = x[1:][::-1]
                c33 += np.diag(x,i+1) + np.diag(x,-(i+1))
        c33 += np.diag(c3[0,:0:-1])
        cmat = np.block([[c33,c32,np.zeros((nlag,1))],
                        [np.vstack([c34,np.zeros((1,nlag))]),c3]])
        magnitude = np.abs(np.fft.fftshift(np.fft.fft2(cmat,s=(nfft,nfft))))
    if not np.isfinite(magnitude).all():
        raise ValueError('双谱计算溢出，请检查信号幅值。')
    return BispectrumView(_readonly(magnitude), _readonly(np.fft.fftshift(np.fft.fftfreq(nfft,1/fs))),
        _readonly(c3), start_sample/fs, (start_sample+used)/fs, used, len(signal)-used, fs)
