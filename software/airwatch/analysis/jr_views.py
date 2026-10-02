"""Legacy J/R envelope statistics with explicit zero-window accounting."""
from dataclasses import dataclass
import numpy as np
from scipy.fftpack import hilbert
from .signal_transforms import prepare_signal, _readonly, _validate_sample_rate


@dataclass(frozen=True)
class JRView:
    j: np.ndarray
    r: np.ndarray
    window_indices: np.ndarray
    zero_window_indices: np.ndarray
    window_samples: int
    total_windows: int
    dropped_samples: int
    start_s: float
    end_s: float
    sample_rate_hz: float


def prepare_jr_signal(data):
    array = np.asarray(data)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    signal = prepare_signal(array)
    if signal.kind != 'real':
        raise ValueError('J/R 目前仅支持实信号，不能丢弃 I/Q 通道。')
    return signal.samples


def compute_jr(data, sample_rate_hz, *, start_sample=0):
    signal = prepare_jr_signal(data)
    fs = _validate_sample_rate(sample_rate_hz)
    if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
        raise ValueError('选区起点必须是非负整数采样点。')
    # Preserve floor(Fs/100), approximately 10 ms, rather than promise exact 10 ms.
    window = int(fs // 100)
    if window < 2:
        raise ValueError('J/R 每窗至少需要 2 点，采样率须不低于 200 Hz。')
    if len(signal) > 65536:
        raise ValueError('J/R 选区最多 65536 点，请缩小选区。')
    count = len(signal) // window
    if count == 0:
        raise ValueError(f'J/R 至少需要一个完整窗口（{window} 点）。')
    js, rs, indices, zeros = [], [], [], []
    for index in range(count):
        row = signal[index*window:(index+1)*window]
        scale = float(np.max(np.abs(row)))
        if scale == 0:
            zeros.append(index)
            continue
        # Both ratios are invariant to a common nonzero gain. Scaling here only
        # avoids overflow/underflow in moments; it does not change model inputs.
        y = row / scale
        h = hilbert(y)
        envelope_squared = y*y + h*h
        m2 = np.mean(envelope_squared)
        m4 = np.mean(envelope_squared**2)
        ps = np.mean(y*y) / 2
        r = abs((m4 - m2*m2)/(m2*m2))
        j = abs((m4 - 2*m2*m2)/(4*ps*ps))
        if not np.isfinite([j,r]).all():
            raise ValueError('J/R 计算产生无效数值，请检查输入。')
        js.append(j); rs.append(r); indices.append(index)
    return JRView(_readonly(np.array(js)), _readonly(np.array(rs)),
        _readonly(np.array(indices,dtype=np.int64)), _readonly(np.array(zeros,dtype=np.int64)),
        window, count, len(signal)%window, start_sample/fs,
        (start_sample+count*window)/fs, fs)
