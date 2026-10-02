"""Prepare one immutable UAV recording for preview and inference.

The prepared input is the seam between file-format handling and both runtime
consumers.  A selected file is read once; preview and inference then share the
same read-only samples and full-recording quality report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Callable

from airwatch.analysis.uav_quality import UAVQualityReport, assess_uav_signal
from airwatch.data.uav_input import UAVFileSpec, UAVSignal, load_uav_signal


@dataclass(frozen=True)
class UAVSourceIdentity:
    path: str
    size_bytes: int
    modified_time_ns: int

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise ValueError("无人机输入身份必须使用绝对路径")
        if self.size_bytes < 1 or self.modified_time_ns < 0:
            raise ValueError("无人机输入文件身份无效")


@dataclass(frozen=True)
class UAVPreparedInput:
    signal: UAVSignal
    quality: UAVQualityReport
    identity: UAVSourceIdentity
    declared_sample_rate_hz: float | None = None

    def __post_init__(self) -> None:
        if self.signal.path != self.identity.path:
            raise ValueError("无人机输入样本与文件身份不一致")
        if self.quality.sample_count != self.signal.sample_count:
            raise ValueError("无人机质量报告与完整记录采样点数不一致")
        if self.quality.channels != self.signal.channels:
            raise ValueError("无人机质量报告与完整记录通道数不一致")
        if self.declared_sample_rate_hz is not None and self.declared_sample_rate_hz <= 0:
            raise ValueError("声明采样率必须为正数")

    def with_declared_sample_rate(self, value: float | None) -> "UAVPreparedInput":
        return replace(self, declared_sample_rate_hz=value)


def prepare_uav_input(
    path: str | Path,
    *,
    declared_sample_rate_hz: float | None = None,
    max_bytes: int | None = None,
    cancelled: Event | None = None,
    phase: Callable[[str], None] | None = None,
) -> UAVPreparedInput:
    """Read and quality-check a complete recording exactly once."""

    source = Path(path).resolve()
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError("无人机输入准备已取消")
    try:
        stat = source.stat()
    except OSError as exc:
        raise ValueError(f"无法读取无人机输入文件信息：{exc}") from exc
    if phase is not None:
        phase("正在读取并校验完整 NPY 记录…")
    signal = load_uav_signal(
        source,
        UAVFileSpec(sample_rate_hz=declared_sample_rate_hz),
        max_bytes=max_bytes,
    )
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError("无人机输入准备已取消")
    quality = assess_uav_signal(signal.samples)
    identity = UAVSourceIdentity(
        path=str(source),
        size_bytes=int(stat.st_size),
        modified_time_ns=int(stat.st_mtime_ns),
    )
    return UAVPreparedInput(
        signal=signal,
        quality=quality,
        identity=identity,
        declared_sample_rate_hz=declared_sample_rate_hz,
    )


__all__ = [
    "UAVPreparedInput",
    "UAVSourceIdentity",
    "prepare_uav_input",
]
