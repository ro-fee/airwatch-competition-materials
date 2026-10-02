"""Bounded, read-only access to MATLAB v7.3 complex-IQ recordings.

MATLAB v7.3 files are HDF5 containers.  The KU Leuven Drone RF Dataset stores
``uhd_samps`` as a singleton row or column whose elements contain ``real`` and
``imag`` floating-point fields.  This module deliberately exposes window reads
only, so callers do not accidentally materialize an entire recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np


DEFAULT_DATASET_NAME = "uhd_samps"
DEFAULT_MAX_WINDOW_SAMPLES = 1_048_576


class MATV73IQError(ValueError):
    """Raised when a file does not satisfy the bounded complex-IQ contract."""


@dataclass(frozen=True)
class MATV73IQMetadata:
    path: str
    dataset_name: str
    sample_count: int
    sample_rate_hz: float
    duration_seconds: float
    source_shape: tuple[int, int]
    source_dtype: str
    matlab_class: str | None
    orientation: str


@dataclass(frozen=True)
class MATV73IQWindow:
    path: str
    dataset_name: str
    start_sample: int
    sample_rate_hz: float
    total_sample_count: int
    samples: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[1])

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.sample_count

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate_hz


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on caller environment
        raise RuntimeError(
            "读取 MATLAB v7.3 无人机数据需要 h5py；请安装 requirements-uav-data.txt"
        ) from exc
    return h5py


def _positive_sample_rate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MATV73IQError("sample_rate_hz 必须是有限正数")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise MATV73IQError("sample_rate_hz 必须是有限正数")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MATV73IQError(f"{name} 必须是整数")
    result = int(value)
    if result < 0:
        raise MATV73IQError(f"{name} 不能为负数")
    return result


def _positive_integer(value: int, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise MATV73IQError(f"{name} 必须大于 0")
    return result


def _decode_matlab_class(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    array = np.asarray(value)
    if array.size == 1:
        item = array.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def _validate_dataset(dataset: Any, *, path: Path, dataset_name: str) -> tuple[int, str]:
    if dataset.ndim != 2 or dataset.shape[0] != 1 and dataset.shape[1] != 1:
        raise MATV73IQError(
            f"{path}:{dataset_name} 必须是 1×N 或 N×1 复数 IQ 数据，实际为 {dataset.shape}"
        )
    sample_count = int(max(dataset.shape))
    if sample_count <= 0:
        raise MATV73IQError(f"{path}:{dataset_name} 没有采样点")
    fields = dataset.dtype.fields
    if fields is None or set(fields) != {"real", "imag"}:
        raise MATV73IQError(
            f"{path}:{dataset_name} 必须包含且仅包含 real/imag 数值字段"
        )
    for field_name in ("real", "imag"):
        field_dtype = np.dtype(fields[field_name][0])
        if field_dtype.kind != "f":
            raise MATV73IQError(
                f"{path}:{dataset_name}.{field_name} 必须是浮点字段"
            )
    orientation = "row" if dataset.shape[0] == 1 else "column"
    return sample_count, orientation


def inspect_mat_v73_iq(
    path: str | Path,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    sample_rate_hz: float = 100_000_000,
) -> MATV73IQMetadata:
    """Inspect metadata without reading any signal samples."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到 MATLAB v7.3 IQ 文件：{source}")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise MATV73IQError("dataset_name 不能为空")
    rate = _positive_sample_rate(sample_rate_hz)
    h5py = _require_h5py()
    try:
        with h5py.File(source, "r") as handle:
            return _inspect_h5py_handle(
                handle,
                source_label=str(source),
                dataset_name=dataset_name,
                sample_rate_hz=rate,
                h5py=h5py,
            )
    except OSError as exc:
        raise MATV73IQError(f"无法以 MATLAB v7.3/HDF5 格式读取 {source}：{exc}") from exc


def _inspect_h5py_handle(
    handle: Any,
    *,
    source_label: str,
    dataset_name: str,
    sample_rate_hz: float,
    h5py: Any,
) -> MATV73IQMetadata:
    if dataset_name not in handle:
        raise MATV73IQError(f"{source_label} 不包含数据集 {dataset_name!r}")
    dataset = handle[dataset_name]
    if not isinstance(dataset, h5py.Dataset):
        raise MATV73IQError(f"{source_label}:{dataset_name} 不是 HDF5 数据集")
    sample_count, orientation = _validate_dataset(
        dataset, path=Path(source_label), dataset_name=dataset_name
    )
    return MATV73IQMetadata(
        path=source_label,
        dataset_name=dataset_name,
        sample_count=sample_count,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=sample_count / sample_rate_hz,
        source_shape=tuple(int(value) for value in dataset.shape),
        source_dtype=str(dataset.dtype),
        matlab_class=_decode_matlab_class(dataset.attrs.get("MATLAB_class")),
        orientation=orientation,
    )


def inspect_mat_v73_iq_fileobj(
    fileobj: BinaryIO,
    *,
    source_name: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    sample_rate_hz: float = 100_000_000,
) -> MATV73IQMetadata:
    """Inspect a seekable binary stream, such as one member opened from a ZIP."""
    if not source_name.strip():
        raise MATV73IQError("source_name 不能为空")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise MATV73IQError("dataset_name 不能为空")
    rate = _positive_sample_rate(sample_rate_hz)
    h5py = _require_h5py()
    try:
        with h5py.File(fileobj, "r") as handle:
            return _inspect_h5py_handle(
                handle,
                source_label=source_name,
                dataset_name=dataset_name,
                sample_rate_hz=rate,
                h5py=h5py,
            )
    except OSError as exc:
        raise MATV73IQError(
            f"无法以 MATLAB v7.3/HDF5 格式读取 {source_name}：{exc}"
        ) from exc


def read_mat_v73_iq_window(
    path: str | Path,
    *,
    start_sample: int,
    window_size: int,
    dataset_name: str = DEFAULT_DATASET_NAME,
    sample_rate_hz: float = 100_000_000,
    max_window_samples: int = DEFAULT_MAX_WINDOW_SAMPLES,
) -> MATV73IQWindow:
    """Read one bounded window as immutable ``float32`` samples shaped ``[2, N]``."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到 MATLAB v7.3 IQ 文件：{source}")
    h5py = _require_h5py()
    try:
        with h5py.File(source, "r") as handle:
            return _read_window_from_handle(
                handle,
                source_label=str(source),
                start_sample=start_sample,
                window_size=window_size,
                dataset_name=dataset_name,
                sample_rate_hz=sample_rate_hz,
                max_window_samples=max_window_samples,
                h5py=h5py,
            )
    except OSError as exc:
        raise MATV73IQError(f"读取 {source} 的 IQ 窗口失败：{exc}") from exc


def _read_window_from_handle(
    handle: Any,
    *,
    source_label: str,
    start_sample: int,
    window_size: int,
    dataset_name: str,
    sample_rate_hz: float,
    max_window_samples: int,
    h5py: Any,
) -> MATV73IQWindow:
    start = _nonnegative_integer(start_sample, "start_sample")
    size = _positive_integer(window_size, "window_size")
    maximum = _positive_integer(max_window_samples, "max_window_samples")
    if size > maximum:
        raise MATV73IQError(
            f"window_size={size} 超过单次读取上限 {maximum}；请分批读取"
        )
    metadata = _inspect_h5py_handle(
        handle,
        source_label=source_label,
        dataset_name=dataset_name,
        sample_rate_hz=_positive_sample_rate(sample_rate_hz),
        h5py=h5py,
    )
    stop = start + size
    if stop > metadata.sample_count:
        raise MATV73IQError(
            f"请求区间 [{start}, {stop}) 超出总采样点数 {metadata.sample_count}"
        )

    dataset = handle[dataset_name]
    raw = (
        dataset[0, start:stop]
        if metadata.orientation == "row"
        else dataset[start:stop, 0]
    )

    samples = np.empty((2, size), dtype=np.float32)
    samples[0] = raw["real"]
    samples[1] = raw["imag"]
    if not np.isfinite(samples).all():
        raise MATV73IQError(f"{source_label} 的请求窗口包含 NaN 或无穷值")
    samples.setflags(write=False)
    return MATV73IQWindow(
        path=source_label,
        dataset_name=dataset_name,
        start_sample=start,
        sample_rate_hz=metadata.sample_rate_hz,
        total_sample_count=metadata.sample_count,
        samples=samples,
    )


def read_mat_v73_iq_window_fileobj(
    fileobj: BinaryIO,
    *,
    source_name: str,
    start_sample: int,
    window_size: int,
    dataset_name: str = DEFAULT_DATASET_NAME,
    sample_rate_hz: float = 100_000_000,
    max_window_samples: int = DEFAULT_MAX_WINDOW_SAMPLES,
) -> MATV73IQWindow:
    """Read one bounded IQ window from a seekable stream such as a ZIP member."""
    if not source_name.strip():
        raise MATV73IQError("source_name 不能为空")
    h5py = _require_h5py()
    try:
        with h5py.File(fileobj, "r") as handle:
            return _read_window_from_handle(
                handle,
                source_label=source_name,
                start_sample=start_sample,
                window_size=window_size,
                dataset_name=dataset_name,
                sample_rate_hz=sample_rate_hz,
                max_window_samples=max_window_samples,
                h5py=h5py,
            )
    except OSError as exc:
        raise MATV73IQError(f"读取 {source_name} 的 IQ 窗口失败：{exc}") from exc


def read_mat_v73_iq_windows_fileobj(
    fileobj: BinaryIO,
    *,
    source_name: str,
    start_samples: Iterable[int],
    window_size: int,
    dataset_name: str = DEFAULT_DATASET_NAME,
    sample_rate_hz: float = 100_000_000,
    max_window_samples: int = DEFAULT_MAX_WINDOW_SAMPLES,
) -> tuple[MATV73IQWindow, ...]:
    """Read several bounded windows while keeping one ZIP member/HDF5 handle open."""
    starts = tuple(start_samples)
    if not starts:
        raise MATV73IQError("start_samples 不能为空")
    if list(starts) != sorted(starts) or len(set(starts)) != len(starts):
        raise MATV73IQError("start_samples 必须严格递增且不重复")
    if not source_name.strip():
        raise MATV73IQError("source_name 不能为空")
    h5py = _require_h5py()
    try:
        with h5py.File(fileobj, "r") as handle:
            return tuple(
                _read_window_from_handle(
                    handle,
                    source_label=source_name,
                    start_sample=start,
                    window_size=window_size,
                    dataset_name=dataset_name,
                    sample_rate_hz=sample_rate_hz,
                    max_window_samples=max_window_samples,
                    h5py=h5py,
                )
                for start in starts
            )
    except OSError as exc:
        raise MATV73IQError(f"读取 {source_name} 的 IQ 窗口失败：{exc}") from exc


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_MAX_WINDOW_SAMPLES",
    "MATV73IQError",
    "MATV73IQMetadata",
    "MATV73IQWindow",
    "inspect_mat_v73_iq",
    "inspect_mat_v73_iq_fileobj",
    "read_mat_v73_iq_window",
    "read_mat_v73_iq_window_fileobj",
    "read_mat_v73_iq_windows_fileobj",
]
