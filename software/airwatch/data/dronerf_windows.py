"""Memory-bounded, deterministic window extraction from DroneRF CSV members."""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .dronerf import DroneRFRecording


class DroneRFWindowingError(ValueError):
    """Raised when a source member cannot produce a trustworthy window set."""


@dataclass(frozen=True)
class DroneRFWindowSpec:
    window_length: int = 4096
    windows_per_recording: int = 32
    expected_sample_count: int = 10_000_000
    seed: int = 20260908
    preprocessing_version: str = "dronerf-two-band-raw-windows-v1"

    def __post_init__(self):
        if self.window_length < 16:
            raise DroneRFWindowingError("窗口长度至少为 16")
        if self.windows_per_recording < 1:
            raise DroneRFWindowingError("每条录制至少提取一个窗口")
        if self.expected_sample_count < self.window_length:
            raise DroneRFWindowingError("预期采样点数不能小于窗口长度")
        if self.seed < 0:
            raise DroneRFWindowingError("窗口随机种子必须为非负整数")
        if not self.preprocessing_version.strip():
            raise DroneRFWindowingError("预处理版本不能为空")


@dataclass(frozen=True)
class DroneRFWindowResult:
    samples: np.ndarray
    offsets: tuple[int, ...]
    low_sample_count: int
    high_sample_count: int

    def __post_init__(self):
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 3 or samples.shape[1] != 2:
            raise DroneRFWindowingError("DroneRF 窗口必须是 [窗口, L/H 双频段, 采样点]")
        if samples.shape[0] != len(self.offsets):
            raise DroneRFWindowingError("窗口数量与偏移数量不一致")
        if not samples.size or not np.isfinite(samples).all():
            raise DroneRFWindowingError("DroneRF 窗口包含空值或非有限数值")
        owned = np.frombuffer(samples.tobytes(), dtype=np.float32).reshape(samples.shape)
        owned.setflags(write=False)
        object.__setattr__(self, "samples", owned)


def stratified_window_offsets(
    recording_id: str,
    spec: DroneRFWindowSpec,
) -> tuple[int, ...]:
    """Choose one stable, non-overlapping start from each temporal stratum."""
    usable = spec.expected_sample_count - spec.window_length + 1
    if usable < spec.windows_per_recording:
        raise DroneRFWindowingError("录制长度不足以容纳指定数量的分层窗口")
    edges = np.linspace(0, usable, spec.windows_per_recording + 1, dtype=np.int64)
    digest = hashlib.sha256(
        f"{spec.seed}\0{recording_id}\0{spec.preprocessing_version}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    offsets = []
    for start, stop in zip(edges[:-1], edges[1:]):
        low, high = int(start), int(stop)
        offsets.append(low if high <= low + 1 else int(rng.integers(low, high)))
    return tuple(offsets)


def _sample_csv_chunks(
    chunks: Iterable[bytes],
    offsets: tuple[int, ...],
    window_length: int,
    *,
    expected_sample_count: int,
) -> tuple[np.ndarray, int]:
    """Count every token but parse only requested ranges from CSV byte chunks."""
    if tuple(sorted(offsets)) != offsets or len(set(offsets)) != len(offsets):
        raise DroneRFWindowingError("窗口偏移必须严格递增且不重复")
    ranges = [(offset, offset + window_length) for offset in offsets]
    if ranges and (ranges[0][0] < 0 or ranges[-1][1] > expected_sample_count):
        raise DroneRFWindowingError("窗口偏移超出预期录制范围")

    collectors = [bytearray() for _ in ranges]
    completed = 0
    carry = b""
    for chunk in chunks:
        if not isinstance(chunk, (bytes, bytearray)):
            raise DroneRFWindowingError("CSV 流必须产生字节块")
        if not chunk:
            continue
        data = carry + bytes(chunk)
        raw = np.frombuffer(data, dtype=np.uint8)
        delimiters = np.flatnonzero(raw == ord(","))
        token_count = int(delimiters.size)
        upper = completed + token_count
        if token_count:
            for collector, (start, stop) in zip(collectors, ranges):
                lo, hi = max(start, completed), min(stop, upper)
                if lo >= hi:
                    continue
                relative_lo, relative_hi = lo - completed, hi - completed
                byte_lo = 0 if relative_lo == 0 else int(delimiters[relative_lo - 1]) + 1
                byte_hi = int(delimiters[relative_hi - 1]) + 1
                collector.extend(data[byte_lo:byte_hi])
            completed = upper
            carry = data[int(delimiters[-1]) + 1 :]
        else:
            carry = data

    if carry.strip():
        final_chunk = carry + b","
        for collector, (start, stop) in zip(collectors, ranges):
            if start <= completed < stop:
                collector.extend(final_chunk)
        completed += 1
    if completed != expected_sample_count:
        raise DroneRFWindowingError(
            f"CSV 采样点数不一致：期望 {expected_sample_count}，实际 {completed}"
        )

    windows = []
    for offset, collector in zip(offsets, collectors):
        parsed = np.fromstring(bytes(collector), dtype=np.float32, sep=",")
        if parsed.size != window_length:
            raise DroneRFWindowingError(
                f"窗口解析长度错误：偏移 {offset}，期望 {window_length}，实际 {parsed.size}"
            )
        if not np.isfinite(parsed).all():
            raise DroneRFWindowingError(f"窗口包含非有限值：偏移 {offset}")
        windows.append(parsed)
    return np.stack(windows, axis=0), completed


def _member_chunks(
    package_path: Path,
    member_path: str,
    *,
    chunk_bytes: int,
) -> Iterator[bytes]:
    process = subprocess.Popen(
        ["tar", "-xOf", str(package_path), member_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        while chunk := process.stdout.read(chunk_bytes):
            yield chunk
        error = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code:
            raise DroneRFWindowingError(
                f"无法读取 RAR 成员 {package_path}::{member_path}：{error or return_code}"
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def sample_csv_member(
    package_path: str | Path,
    member_path: str,
    offsets: tuple[int, ...],
    spec: DroneRFWindowSpec,
    *,
    chunk_bytes: int = 4 * 1024 * 1024,
) -> tuple[np.ndarray, int]:
    """Stream one RAR member and return only selected windows plus full token count."""
    package = Path(package_path)
    if not package.is_file():
        raise FileNotFoundError(f"DroneRF RAR 不存在：{package}")
    if chunk_bytes < 1024:
        raise DroneRFWindowingError("流式读取块大小至少为 1024 字节")
    return _sample_csv_chunks(
        _member_chunks(package, member_path, chunk_bytes=chunk_bytes),
        offsets,
        spec.window_length,
        expected_sample_count=spec.expected_sample_count,
    )


def extract_recording_windows(
    recording: DroneRFRecording,
    package_root: str | Path,
    spec: DroneRFWindowSpec,
) -> DroneRFWindowResult:
    """Extract aligned L/H windows for one recording without materializing its CSVs."""
    root = Path(package_root)
    offsets = stratified_window_offsets(recording.recording_id, spec)
    low, low_count = sample_csv_member(
        root / Path(recording.low_package_path), recording.low_member_path, offsets, spec
    )
    high, high_count = sample_csv_member(
        root / Path(recording.high_package_path), recording.high_member_path, offsets, spec
    )
    if low_count != high_count:
        raise DroneRFWindowingError(
            f"L/H 频段采样点数不一致：{recording.recording_id}"
        )
    samples = np.stack((low, high), axis=1)
    return DroneRFWindowResult(samples, offsets, low_count, high_count)


__all__ = [
    "DroneRFWindowResult",
    "DroneRFWindowSpec",
    "DroneRFWindowingError",
    "extract_recording_windows",
    "sample_csv_member",
    "stratified_window_offsets",
]
