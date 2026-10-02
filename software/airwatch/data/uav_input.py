"""Explicit, read-only UAV signal input adapter.

The adapter accepts verified .npy arrays and explicitly-described raw .dat files.
It does not infer raw binary semantics from filenames.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class UAVFileSpec:
    dtype: Optional[str] = None
    channels: int = 1
    layout: str = "single"  # single, channel_first, interleaved_iq
    sample_rate_hz: Optional[float] = None

    def __post_init__(self):
        if self.channels not in (1, 2):
            raise ValueError("无人机输入目前只支持 1 或 2 通道")
        if self.layout not in ("single", "channel_first", "interleaved_iq"):
            raise ValueError("通道布局必须是 single、channel_first 或 interleaved_iq")
        if self.channels == 1 and self.layout != "single":
            raise ValueError("单通道输入必须使用 single 布局")
        if self.channels == 2 and self.layout == "single":
            raise ValueError("双通道输入必须明确 channel_first 或 interleaved_iq")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("采样率必须为正数")


@dataclass(frozen=True)
class UAVSignal:
    path: str
    samples: np.ndarray  # shape: (samples,) or (channels, samples), float32
    sample_rate_hz: Optional[float]
    source_format: str

    def __post_init__(self):
        array = np.asarray(self.samples, dtype=np.float32)
        if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[0] not in (1, 2)):
            raise ValueError("无人机信号必须是采样点数组或通道×采样点数组")
        if not array.size or not np.isfinite(array).all():
            raise ValueError("无人机信号必须是非空有限数值")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("采样率必须为正数")
        owned = np.frombuffer(array.tobytes(), dtype=np.float32).reshape(array.shape)
        owned.setflags(write=False)
        object.__setattr__(self, "samples", owned)

    @property
    def channels(self):
        return 1 if self.samples.ndim == 1 else self.samples.shape[0]

    @property
    def sample_count(self):
        return self.samples.shape[-1]


def _as_float32(array):
    if array.dtype.kind not in 'iufc':
        raise ValueError('NPY 信号必须包含实数或复数采样值')
    if np.iscomplexobj(array):
        if array.ndim != 1:
            raise ValueError('复数 IQ 必须是一维数组')
        return np.stack((array.real, array.imag), axis=0).astype(np.float32)
    canonical = np.asarray(array, dtype=np.float32)
    if canonical.ndim == 1:
        return canonical
    if canonical.ndim == 2:
        if canonical.shape[0] in (1, 2):
            return canonical
        if canonical.shape[1] in (1, 2):
            return canonical.T
        raise ValueError("NPY 二维数组必须是 [通道,采样点] 或 [采样点,通道]，通道数为 1 或 2")
    if canonical.ndim == 3 and canonical.shape[1] in (1, 2):
        # Frozen UAV contracts may store one recording as [window, channel, sample].
        # Concatenating in recording order gives the same canonical [channel, sample]
        # representation consumed by preview and inference.
        return canonical.transpose(1, 0, 2).reshape(canonical.shape[1], -1)
    raise ValueError(
        "NPY 信号必须是一维、[通道,采样点]、[采样点,通道]或[窗口,通道,采样点]"
    )


def load_uav_signal(path, spec: Optional[UAVFileSpec] = None, *, max_bytes=None) -> UAVSignal:
    """Load one signal without guessing raw binary layout."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到无人机信号文件：{path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            raw = np.load(path, allow_pickle=False, mmap_mode='r')
            if max_bytes is not None and raw.size * (8 if np.iscomplexobj(raw) else 4) > max_bytes:
                raise ValueError('转换后的信号超过预览内存上限，请先拆分录制')
        except Exception as exc:
            raise ValueError(f"无法读取 NPY 信号文件：{exc}") from exc
        array = _as_float32(raw)
        return UAVSignal(
            str(path.resolve()),
            array,
            spec.sample_rate_hz if spec else None,
            "npy",
        )
    if suffix == ".dat":
        if spec is None or not spec.dtype:
            raise ValueError("DAT 是原始二进制文件，必须先提供 dtype、通道布局和采样率")
        try:
            raw = np.fromfile(path, dtype=np.dtype(spec.dtype))
        except Exception as exc:
            raise ValueError(f"无法读取 DAT 信号文件：{exc}") from exc
        if not raw.size:
            raise ValueError("DAT 文件为空")
        if spec.layout == "interleaved_iq":
            if raw.size % 2:
                raise ValueError("交错 IQ 数据长度必须为偶数")
            array = np.stack((raw[0::2], raw[1::2]), axis=0)
        elif spec.layout == "channel_first":
            if raw.size % spec.channels:
                raise ValueError("通道优先 DAT 长度不能被通道数整除")
            array = raw.reshape(spec.channels, -1)
        else:
            array = raw
        return UAVSignal(str(path.resolve()), _as_float32(array), spec.sample_rate_hz, "dat")
    raise ValueError(f"暂不支持的无人机信号格式：{suffix or '无扩展名'}；仅支持 .npy 或 .dat")


__all__ = ["UAVFileSpec", "UAVSignal", "load_uav_signal"]
