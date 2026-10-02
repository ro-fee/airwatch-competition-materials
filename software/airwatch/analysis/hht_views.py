"""Optional EMD-signal + SciPy analysis; no Qt or legacy image fallback."""
from dataclasses import dataclass
from importlib.metadata import version, PackageNotFoundError
import numpy as np
from scipy.signal import hilbert
from .signal_transforms import prepare_signal, _readonly, _validate_sample_rate


class HHTUnavailable(RuntimeError):
    pass


def load_hht_backend():
    try:
        from PyEMD import EMD
        try:
            backend_name = f'EMD-signal {version("EMD-signal")} + SciPy phase derivative'
        except PackageNotFoundError:
            backend_name = 'EMD-signal (version metadata unavailable) + SciPy phase derivative'
    except Exception as exc:
        raise HHTUnavailable('HHT 暂不可用：需要 EMD-signal（导入名 PyEMD）；本次未生成结果。'
                             f'依赖详情：{type(exc).__name__}: {exc}') from exc

    class Decomposer:
        def __init__(self, signal):
            self.signal = signal
            self.backend = backend_name
            self.labels = ()

        def decompose(self):
            engine = EMD(spline_kind='cubic', nbsym=2, MAX_ITERATION=1000)
            engine.emd(self.signal, max_imf=8)
            imfs, residue = engine.get_imfs_and_residue()
            # Always show the explicitly returned residue, even if it is zero.
            self.labels = tuple(f'IMF {i+1}' for i in range(len(imfs))) + ('Residue',)
            parts = np.vstack([imfs, residue])
            if not np.allclose(parts.sum(axis=0), self.signal, rtol=1e-8, atol=1e-10):
                raise ValueError('EMD 分量无法重构输入信号，结果已拒绝。')
            return parts

    return Decomposer, instantaneous_frequency


def instantaneous_frequency(analytic):
    """Central phase derivative in cycles/sample, excluding endpoint estimates.

    Values outside the representable positive-frequency band are undefined,
    not clipped to look plausible. Edge effects and mode mixing still apply.
    """
    phase = np.unwrap(np.angle(analytic))
    return np.gradient(phase)[1:-1] / (2*np.pi), np.arange(1, len(analytic)-1)


def prepare_hht_signal(data):
    array = np.asarray(data)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    prepared = prepare_signal(array)
    if prepared.kind != 'real':
        raise ValueError('HHT 仅支持实信号，不能丢弃 I/Q 通道。')
    return prepared.samples


@dataclass(frozen=True)
class HHTInput:
    signal: np.ndarray
    sample_rate_hz: float
    start_sample: int


@dataclass(frozen=True)
class HHTComponent:
    values: np.ndarray
    envelope: np.ndarray
    frequency_time_s: np.ndarray
    frequency_hz: np.ndarray
    label: str = 'EMD component'


@dataclass(frozen=True)
class HHTView:
    time_s: np.ndarray
    components: tuple
    backend: str = 'unspecified EMD backend'


def prepare_hht(data, sample_rate_hz, start_sample=0):
    signal = prepare_hht_signal(data)
    rate = _validate_sample_rate(sample_rate_hz)
    if not 32 <= signal.size <= 8192:
        raise ValueError('HHT 选区须为 32–8192 点，请在时域图调整选区。')
    if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
        raise ValueError('HHT 选区起点须为非负整数。')
    if np.ptp(signal) == 0:
        raise ValueError('常数或全零信号不能进行有意义的 HHT 分解。')
    return HHTInput(signal, rate, start_sample)


def compute_hht(source, *, cancelled=lambda: False, progress=lambda message: None):
    def check():
        if cancelled():
            raise InterruptedError('HHT 已取消。')
    check()
    emd, inst_freq = load_hht_backend()
    progress('正在进行 EMD 分解（无百分比进度）')
    decomposer = emd(source.signal.copy())
    parts = np.asarray(decomposer.decompose())
    check()
    if (parts.ndim != 2 or not 1 <= parts.shape[0] <= 12 or
            parts.shape[1] != source.signal.size or np.iscomplexobj(parts) or
            not np.issubdtype(parts.dtype, np.number) or not np.isfinite(parts).all()):
        raise ValueError('EMD 分解返回无效形状、数值或超过 12 个分量；未绘图。')
    time_s = (np.arange(source.signal.size) + source.start_sample) / source.sample_rate_hz
    components = []
    labels = ('Raw',) + tuple(getattr(decomposer, 'labels', tuple(f'EMD component {i+1}' for i in range(len(parts)))))
    for index, row in enumerate([source.signal, *parts]):
        check()
        progress(f'正在分析 Hilbert 分量 {index+1}/{len(parts)+1}')
        analytic = hilbert(row)
        envelope = np.abs(analytic)
        if not np.isfinite(envelope).all():
            raise ValueError('Hilbert 包络溢出，请检查幅值。')
        if np.max(envelope) == 0:
            times, frequency = np.array([]), np.array([])
        else:
            frequency, stamps = inst_freq(analytic)
            frequency, stamps = np.asarray(frequency), np.asarray(stamps)
            if (frequency.ndim != 1 or stamps.shape != frequency.shape or
                    not np.isfinite(stamps).all() or not np.isfinite(frequency).all() or
                    np.any(stamps != np.floor(stamps)) or np.any(stamps < 0) or
                    np.any(stamps >= len(row)) or np.any(np.diff(stamps) <= 0)):
                raise ValueError('瞬时频率输出或采样点坐标无效。')
            # Phase is ill-defined near zero envelope. Do not fabricate zero Hz.
            samples = stamps.astype(int)
            # Derivative depends on neighbours; phase at a zero crossing is undefined.
            local = np.minimum.reduce([envelope[samples], envelope[np.maximum(samples-1,0)],
                                       envelope[np.minimum(samples+1,len(row)-1)]])
            valid = (local > np.max(envelope)*1e-8) & (frequency >= 0) & (frequency <= .5)
            frequency = np.where(valid, frequency * source.sample_rate_hz, np.nan)
            times = (stamps + source.start_sample) / source.sample_rate_hz
        components.append(HHTComponent(_readonly(row), _readonly(envelope),
                                        _readonly(times), _readonly(frequency), labels[index]))
    check()
    return HHTView(_readonly(time_s), tuple(components),
                   getattr(decomposer, 'backend', 'unspecified EMD backend'))
