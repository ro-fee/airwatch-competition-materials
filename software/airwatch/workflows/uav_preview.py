"""Bounded input preview, independent of Qt and UAV model inference."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from airwatch.analysis.signal_transforms import compute_waveform, compute_spectrum, compute_spectrogram
from airwatch.workflows.uav_intake import UAVPreparedInput, prepare_uav_input

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PREVIEW_SAMPLES = 65536


@dataclass(frozen=True)
class UAVPreview:
    path: str
    mode: str
    total_samples: int
    shown_samples: int
    channels: int
    sample_rate_hz: float | None
    view: object
    quality: object
    prepared_input: UAVPreparedInput


def prepare_uav_preview(
    path,
    mode,
    rate,
    window,
    cancelled,
    phase,
    prepared_input: UAVPreparedInput | None = None,
):
    def check():
        if cancelled.is_set():
            raise InterruptedError('预览已取消')
    check()
    if mode not in ('waveform', 'spectrum', 'spectrogram'):
        raise ValueError('不支持的预览类型')
    if rate is not None:
        try:
            if isinstance(rate, bool):
                raise ValueError
            rate = float(rate)
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError('采样率须为大于零的有限数值（Hz），请填写采集时的真实参数。') from None
    if mode != 'waveform' and rate is None:
        raise ValueError('频谱和时频图需要采样率（Hz）；不知道时请只看波形。')
    path = Path(path).resolve()
    if path.suffix.lower() != '.npy':
        raise ValueError('当前预览仅支持 NPY；DAT 需先确认数据格式并转换，不能猜测。')
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('预览文件上限为 64 MiB，请先拆分录制；不会静默忽略文件内容。')
    if prepared_input is None or prepared_input.identity.path != str(path):
        prepared_input = prepare_uav_input(
            path,
            declared_sample_rate_hz=rate,
            max_bytes=MAX_FILE_BYTES,
            cancelled=cancelled,
            phase=phase,
        )
    else:
        prepared_input = prepared_input.with_declared_sample_rate(rate)
    signal = prepared_input.signal
    check()
    samples = signal.samples[..., :MAX_PREVIEW_SAMPLES]
    if samples.ndim == 2 and samples.shape[0] == 1:
        samples = samples[0]
    quality = prepared_input.quality
    phase('正在计算图形…')
    if mode == 'waveform':
        view = compute_waveform(samples, sample_rate_hz=rate)
    elif mode == 'spectrum':
        view = compute_spectrum(samples, sample_rate_hz=rate)
    else:
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise ValueError('时频窗长必须是至少 2 的整数')
        if window > samples.shape[-1]:
            raise ValueError(f'时频窗长 {window} 超过预览长度 {samples.shape[-1]}，请减小窗长。')
        view = compute_spectrogram(samples, sample_rate_hz=rate, nperseg=window)
    check()
    return UAVPreview(
        str(path),
        mode,
        signal.sample_count,
        samples.shape[-1],
        signal.channels,
        rate,
        view,
        quality,
        prepared_input,
    )
