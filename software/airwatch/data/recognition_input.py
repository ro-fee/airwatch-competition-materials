"""Decode and own bounded inputs for the five historical recognition tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from airwatch.general_recognition_contract import task_spec

# Bounds the UI-side copy and worst-case retained feature memory. Not a model limit.
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
MAX_WINDOWS = 2048
MAX_REPEATS = 100


def _owned_float32(raw: object) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(raw, dtype=np.float32, order="C")
    if not np.isfinite(converted).all():
        raise ValueError("输入幅度超出 float32 范围或包含非有限值")
    owned = np.frombuffer(converted.tobytes(), dtype=np.float32).reshape(converted.shape)
    owned.flags.writeable = False
    return owned


def _infer_directory_label(path: Path) -> str:
    parent = path.parent
    if "snr" in parent.name.lower() and parent.parent != parent:
        parent = parent.parent
    return parent.name


def _decode_file(task_name: str, path: Path) -> np.ndarray:
    spec = task_spec(task_name)
    if not path.is_file():
        raise ValueError(f"识别文件不存在：{path}")
    if path.stat().st_size % np.dtype(spec.sample_dtype).itemsize:
        raise ValueError(f"{path.name}：文件含截断的采样值，字节数必须整除 {spec.sample_dtype} 元素大小")
    try:
        raw = np.fromfile(path, dtype=np.dtype(spec.sample_dtype))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name}：无法按 {spec.sample_dtype} 读取") from exc
    if raw.size == 0:
        raise ValueError(f"{path.name}：文件为空")
    if spec.input_layout == "interleaved_iq":
        if raw.size % 2:
            raise ValueError(f"{path.name}：I/Q 交错数据的标量数必须为偶数")
        decoded = np.stack((raw[0::2], raw[1::2]), axis=0)
    elif spec.input_layout == "channel_major":
        if raw.size % spec.channels:
            raise ValueError(f"{path.name}：数据长度无法整除 {spec.channels} 个通道")
        decoded = raw.reshape(spec.channels, -1)
    elif spec.input_layout == "single_channel":
        decoded = raw.reshape(1, -1)
    else:
        raise ValueError(f"未实现的历史输入布局：{spec.input_layout}")
    decoded = _owned_float32(decoded)
    if decoded.shape[0] != spec.channels:
        raise ValueError(f"{path.name}：解码后通道数与任务契约不一致")
    if decoded.shape[-1] < spec.window_size:
        raise ValueError(
            f"{path.name}：仅 {decoded.shape[-1]} 个采样点，"
            f"少于任务窗长 {spec.window_size}"
        )
    if decoded.nbytes > MAX_SNAPSHOT_BYTES:
        raise ValueError(f"{path.name}：解码后超过 32 MiB 单文件上限")
    if decoded.shape[-1] // spec.window_size > MAX_WINDOWS:
        raise ValueError(f"{path.name}：超过 {MAX_WINDOWS} 个窗口上限")
    return decoded


@dataclass(frozen=True)
class GeneralRecognitionInput:
    """Owned decoded recordings that stay independent from other UI pages."""

    task_name: str
    filenames: tuple[str, ...]
    arrays: tuple[np.ndarray, ...]
    inferred_directory_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        spec = task_spec(self.task_name)
        if not self.filenames or len(self.filenames) != len(self.arrays):
            raise ValueError("识别文件与数组数量不一致")
        if len(self.inferred_directory_labels) != len(self.filenames):
            raise ValueError("目录标签数量与文件数不一致")
        for filename, array in zip(self.filenames, self.arrays):
            if array.dtype != np.float32 or array.ndim != 2:
                raise ValueError(f"{Path(filename).name}：解码数组必须是二维 float32")
            if array.shape[0] != spec.channels:
                raise ValueError(f"{Path(filename).name}：通道数与任务契约不一致")

    @property
    def file_count(self) -> int:
        return len(self.filenames)

    @property
    def display_names(self) -> tuple[str, ...]:
        return tuple(Path(path).name for path in self.filenames)

    def selected(self, index: int) -> np.ndarray:
        if not 0 <= index < self.file_count:
            index = 0
        return self.arrays[index]

    def snapshot(self, repeat_count: int) -> "RecognitionSnapshot":
        spec = task_spec(self.task_name)
        data: object = self.arrays[0] if self.file_count == 1 else self.arrays
        # Directory names are navigation hints, not verified ground truth.
        return RecognitionSnapshot(
            self.task_name,
            self.filenames,
            data,
            (),
            spec.window_size,
            repeat_count,
        )


def load_recognition_files(
    task_name: str,
    paths: Iterable[str | Path],
) -> GeneralRecognitionInput:
    """Decode selected raw files under one task contract, preserving order."""

    normalized = tuple(Path(path).expanduser().resolve() for path in paths)
    if not normalized:
        raise ValueError("至少需要一个识别文件")
    arrays = tuple(_decode_file(task_name, path) for path in normalized)
    return GeneralRecognitionInput(
        task_name=task_name,
        filenames=tuple(str(path) for path in normalized),
        arrays=arrays,
        inferred_directory_labels=tuple(_infer_directory_label(path) for path in normalized),
    )


def discover_recognition_files(folders: Iterable[str | Path]) -> tuple[Path, ...]:
    """Return deterministic direct child files from selected batch folders."""

    discovered: list[Path] = []
    for value in folders:
        folder = Path(value).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"批量识别目录不存在：{folder}")
        discovered.extend(sorted(
            (path for path in folder.iterdir() if path.is_file()),
            key=lambda path: path.name.casefold(),
        ))
    if not discovered:
        raise ValueError("所选目录中没有可读文件")
    return tuple(discovered)


@dataclass(frozen=True)
class RecognitionSnapshot:
    task_name: str
    filenames: tuple[str, ...]
    data: object
    labels: tuple[str, ...]
    window_size: int
    repeat_count: int = 1

    def __post_init__(self):
        if isinstance(self.window_size, bool) or not isinstance(self.window_size, int) or self.window_size <= 0:
            raise ValueError('窗口长度必须是正整数')
        if isinstance(self.repeat_count, bool) or not isinstance(self.repeat_count, (int, np.integer)) or self.repeat_count < 1 or self.repeat_count > MAX_REPEATS:
            raise ValueError(f'运行次数必须是 1-{MAX_REPEATS} 的整数')
        if isinstance(self.filenames, str) or not self.filenames or any(not str(x).strip() for x in self.filenames):
            raise ValueError('至少需要一个输入文件')
        if isinstance(self.labels, str) or len(self.labels) not in (0, len(self.filenames)):
            raise ValueError('真实标签数量必须为 0 或与文件数一致')
        arrays = self.data if isinstance(self.data, (tuple, list)) else (self.data,)
        if len(arrays) != len(self.filenames): raise ValueError('输入数组与文件数量不一致')
        if self.repeat_count * sum(max(0, np.asarray(a).shape[-1] // self.window_size) for a in arrays) > MAX_WINDOWS * MAX_REPEATS:
            raise ValueError('重复运行总窗口数超出上限，请减少文件或运行次数')
        normalized = []
        for raw in arrays:
            data = np.asarray(raw)
            if data.ndim not in (1, 2) or data.dtype.kind not in 'iuf':
                raise ValueError('输入须为实数数组：采样点或通道×采样点')
            if data.ndim == 2 and data.shape[0] not in (1, 2):
                raise ValueError('仅支持单通道或双通道；请核对文件解析格式')
            if data.size * 4 > MAX_SNAPSHOT_BYTES or data.shape[-1] // self.window_size > MAX_WINDOWS:
                raise ValueError('超出首版单文件预览上限（32 MiB / 2048 窗口），请缩短输入；不会截断后冒充完整结果')
            if not data.size or not np.isfinite(data).all(): raise ValueError('识别输入必须是非空有限数值')
            if data.shape[-1] < self.window_size: raise ValueError(f'信号长度不足 {self.window_size} 个采样点')
            owned = _owned_float32(data)
            normalized.append(owned)
        object.__setattr__(self, 'data', normalized[0] if len(normalized) == 1 else tuple(normalized))
        object.__setattr__(self, 'filenames', tuple(str(x) for x in self.filenames))
        object.__setattr__(self, 'labels', tuple(str(x) for x in self.labels))


__all__ = [
    "GeneralRecognitionInput",
    "MAX_REPEATS",
    "MAX_SNAPSHOT_BYTES",
    "MAX_WINDOWS",
    "RecognitionSnapshot",
    "discover_recognition_files",
    "load_recognition_files",
]

