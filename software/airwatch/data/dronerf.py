"""DroneRF v1 labels and recording-level manifest construction.

The official archive stores each recording as a lower-band (L) CSV and an
upper-band (H) CSV.  Package suffixes are storage chunks, not recording IDs:
for example, Phantom L1 contains segments 0--9, L2 contains 10--20, while one
H package contains all 21 segments.  Pairing therefore uses ``(code, index)``.

This module inventories source material only.  It does not extract, normalize,
window, split, or train on the recordings.
"""
from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


class DroneRFManifestError(ValueError):
    """Raised when the official package layout is incomplete or inconsistent."""


@dataclass(frozen=True)
class DroneRFLabel:
    code: str
    drone_present: bool
    drone_type: str
    operation_mode: str


@dataclass(frozen=True)
class DroneRFPackage:
    path: str
    code: str
    band: str
    storage_part: int | None


@dataclass(frozen=True)
class DroneRFArchiveEntry:
    package_path: str
    member_path: str
    code: str
    band: str
    segment_index: int


@dataclass(frozen=True)
class DroneRFRecording:
    recording_id: str
    source_scenario_group: str
    code: str
    drone_present: bool
    drone_type: str
    operation_mode: str
    segment_index: int
    low_package_path: str
    low_member_path: str
    high_package_path: str
    high_member_path: str


@dataclass(frozen=True)
class DroneRFSplitAssignment:
    recording_id: str
    source_scenario_group: str
    code: str
    split: str
    split_seed: int
    split_protocol: str = "dronerf-recording-stratified-v1"


_LABELS = {
    "00000": DroneRFLabel("00000", False, "background", "background"),
    "10000": DroneRFLabel("10000", True, "parrot_bebop", "on_connected"),
    "10001": DroneRFLabel("10001", True, "parrot_bebop", "hovering"),
    "10010": DroneRFLabel("10010", True, "parrot_bebop", "flying"),
    "10011": DroneRFLabel("10011", True, "parrot_bebop", "video_recording"),
    "10100": DroneRFLabel("10100", True, "parrot_ar", "on_connected"),
    "10101": DroneRFLabel("10101", True, "parrot_ar", "hovering"),
    "10110": DroneRFLabel("10110", True, "parrot_ar", "flying"),
    "10111": DroneRFLabel("10111", True, "parrot_ar", "video_recording"),
    "11000": DroneRFLabel("11000", True, "dji_phantom_3", "on_connected"),
}

_PACKAGE_RE = re.compile(
    r"^(?:RF|FR) Data_(?P<code>\d{5})_(?P<band>[LH])(?P<part>\d*)\.rar$",
    re.IGNORECASE,
)
_MEMBER_RE = re.compile(
    r"^(?P<code>\d{5})(?P<band>[LH])_(?P<index>\d+)\.csv$",
    re.IGNORECASE,
)


def decode_label(code: str) -> DroneRFLabel:
    """Return the documented DroneRF v1 label for a five-digit code."""
    try:
        return _LABELS[str(code)]
    except KeyError as exc:
        raise DroneRFManifestError(f"未知 DroneRF 标签码：{code}") from exc


def parse_package_path(path: str | Path) -> DroneRFPackage:
    """Parse one nested RAR package name, including the official ``FR`` typo."""
    normalized = str(path).replace("\\", "/")
    match = _PACKAGE_RE.fullmatch(Path(normalized).name)
    if not match:
        raise DroneRFManifestError(f"无法识别 DroneRF 压缩包名：{path}")
    code = match.group("code")
    decode_label(code)
    part_text = match.group("part")
    return DroneRFPackage(
        path=normalized,
        code=code,
        band=match.group("band").upper(),
        storage_part=int(part_text) if part_text else None,
    )


def parse_member_path(package: DroneRFPackage, member_path: str) -> DroneRFArchiveEntry | None:
    """Parse a CSV member; directory entries return ``None``."""
    normalized = str(member_path).replace("\\", "/").rstrip("/")
    if not normalized or not normalized.lower().endswith(".csv"):
        return None
    match = _MEMBER_RE.fullmatch(Path(normalized).name)
    if not match:
        raise DroneRFManifestError(
            f"压缩包 {package.path} 中存在无法识别的 CSV：{member_path}"
        )
    code = match.group("code")
    band = match.group("band").upper()
    if code != package.code or band != package.band:
        raise DroneRFManifestError(
            f"成员与压缩包标签不一致：{package.path} -> {member_path}"
        )
    return DroneRFArchiveEntry(
        package_path=package.path,
        member_path=normalized,
        code=code,
        band=band,
        segment_index=int(match.group("index")),
    )


def build_recordings(entries: Iterable[DroneRFArchiveEntry]) -> list[DroneRFRecording]:
    """Pair lower/upper band entries into stable recording-level records."""
    pairs: dict[tuple[str, int], dict[str, DroneRFArchiveEntry]] = {}
    for entry in entries:
        decode_label(entry.code)
        if entry.band not in {"L", "H"}:
            raise DroneRFManifestError(f"频段必须为 L 或 H：{entry.band}")
        key = (entry.code, entry.segment_index)
        by_band = pairs.setdefault(key, {})
        if entry.band in by_band:
            raise DroneRFManifestError(
                f"重复的 DroneRF 频段成员：{entry.code}/{entry.segment_index}/{entry.band}"
            )
        by_band[entry.band] = entry

    recordings: list[DroneRFRecording] = []
    for (code, segment_index), by_band in sorted(pairs.items()):
        missing = {"L", "H"} - set(by_band)
        if missing:
            raise DroneRFManifestError(
                f"DroneRF 录制缺少频段 {','.join(sorted(missing))}：{code}/{segment_index}"
            )
        label = decode_label(code)
        low, high = by_band["L"], by_band["H"]
        recordings.append(
            DroneRFRecording(
                recording_id=f"dronerf-v1:{code}:{segment_index:03d}",
                source_scenario_group=f"dronerf-v1:{code}",
                code=code,
                drone_present=label.drone_present,
                drone_type=label.drone_type,
                operation_mode=label.operation_mode,
                segment_index=segment_index,
                low_package_path=low.package_path,
                low_member_path=low.member_path,
                high_package_path=high.package_path,
                high_member_path=high.member_path,
            )
        )
    if not recordings:
        raise DroneRFManifestError("没有发现可配对的 DroneRF 录制")
    return recordings


def assign_recording_splits(
    recordings: Iterable[DroneRFRecording],
    *,
    seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> list[DroneRFSplitAssignment]:
    """Create a stable label-stratified split at the L/H recording-pair level.

    Each ``recording_id`` receives exactly one split before any windows are
    created.  DroneRF v1 does not publish independent session/device IDs, so
    this protocol must be reported as within-source recording-level evaluation,
    not cross-session or cross-device generalization.
    """
    if seed < 0:
        raise DroneRFManifestError("划分随机种子必须为非负整数")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise DroneRFManifestError("验证集和测试集比例必须位于 0 与 1 之间")
    if validation_fraction + test_fraction >= 1:
        raise DroneRFManifestError("验证集与测试集比例之和必须小于 1")

    by_code: dict[str, list[DroneRFRecording]] = {}
    seen: set[str] = set()
    for recording in recordings:
        if recording.recording_id in seen:
            raise DroneRFManifestError(f"重复 recording_id：{recording.recording_id}")
        seen.add(recording.recording_id)
        by_code.setdefault(recording.code, []).append(recording)
    if not by_code:
        raise DroneRFManifestError("不能划分空的 DroneRF 录制清单")

    assignments: list[DroneRFSplitAssignment] = []
    for code, items in sorted(by_code.items()):
        if len(items) < 3:
            raise DroneRFManifestError(f"标签 {code} 少于 3 条录制，无法生成三向划分")
        ordered = sorted(
            items,
            key=lambda item: hashlib.sha256(
                f"{seed}\0{item.recording_id}".encode("utf-8")
            ).digest(),
        )
        validation_count = max(1, round(len(ordered) * validation_fraction))
        test_count = max(1, round(len(ordered) * test_fraction))
        if validation_count + test_count >= len(ordered):
            raise DroneRFManifestError(f"标签 {code} 的训练录制不足")
        validation_ids = {item.recording_id for item in ordered[:validation_count]}
        test_ids = {
            item.recording_id
            for item in ordered[validation_count : validation_count + test_count]
        }
        for item in sorted(items, key=lambda value: value.recording_id):
            split = (
                "validation" if item.recording_id in validation_ids
                else "test" if item.recording_id in test_ids
                else "train"
            )
            assignments.append(
                DroneRFSplitAssignment(
                    recording_id=item.recording_id,
                    source_scenario_group=item.source_scenario_group,
                    code=item.code,
                    split=split,
                    split_seed=seed,
                )
            )
    return sorted(assignments, key=lambda item: item.recording_id)


def audit_recording_splits(
    recordings: Iterable[DroneRFRecording],
    assignments: Iterable[DroneRFSplitAssignment],
) -> dict:
    """Verify coverage, recording isolation, and per-label split coverage."""
    rows = list(recordings)
    split_rows = list(assignments)
    expected_ids = [item.recording_id for item in rows]
    assigned_ids = [item.recording_id for item in split_rows]
    duplicate_ids = sorted({item for item in assigned_ids if assigned_ids.count(item) > 1})
    missing_ids = sorted(set(expected_ids) - set(assigned_ids))
    unexpected_ids = sorted(set(assigned_ids) - set(expected_ids))
    invalid_splits = sorted({item.split for item in split_rows} - {"train", "validation", "test"})

    counts: dict[str, dict[str, int]] = {}
    for item in split_rows:
        counts.setdefault(item.code, {"train": 0, "validation": 0, "test": 0})
        if item.split in counts[item.code]:
            counts[item.code][item.split] += 1
    labels_missing_a_split = sorted(
        code for code, split_counts in counts.items() if not all(split_counts.values())
    )
    ok = not (duplicate_ids or missing_ids or unexpected_ids or invalid_splits or labels_missing_a_split)
    return {
        "ok": ok,
        "protocol": split_rows[0].split_protocol if split_rows else None,
        "seed": split_rows[0].split_seed if split_rows else None,
        "recording_count": len(rows),
        "assignment_count": len(split_rows),
        "counts_by_code": {code: counts[code] for code in sorted(counts)},
        "duplicate_recording_ids": duplicate_ids,
        "missing_recording_ids": missing_ids,
        "unexpected_recording_ids": unexpected_ids,
        "invalid_splits": invalid_splits,
        "labels_missing_a_split": labels_missing_a_split,
        "known_limitation": (
            "DroneRF v1 未提供独立采集会话、日期或物理设备实例 ID；"
            "本划分仅证明 L/H 录制对及其未来切片不跨集合，不能证明跨会话泛化。"
        ),
    }


def _list_with_tar(package_path: Path) -> Sequence[str]:
    completed = subprocess.run(
        ["tar", "-tf", str(package_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        detail = completed.stderr.strip() or f"退出码 {completed.returncode}"
        raise DroneRFManifestError(f"无法列出压缩包 {package_path}：{detail}")
    return completed.stdout.splitlines()


def discover_recordings(
    package_root: str | Path,
    *,
    member_lister: Callable[[Path], Sequence[str]] | None = None,
) -> list[DroneRFRecording]:
    """Inventory all nested RARs below ``package_root`` without extracting CSVs."""
    root = Path(package_root)
    if not root.is_dir():
        raise FileNotFoundError(f"DroneRF 压缩包目录不存在：{root}")
    lister = member_lister or _list_with_tar
    package_paths = sorted(root.rglob("*.rar"), key=lambda item: item.as_posix().lower())
    if not package_paths:
        raise DroneRFManifestError(f"目录中没有 DroneRF RAR 压缩包：{root}")

    entries: list[DroneRFArchiveEntry] = []
    for package_path in package_paths:
        relative = package_path.relative_to(root).as_posix()
        package = parse_package_path(relative)
        member_count = 0
        for member_path in lister(package_path):
            entry = parse_member_path(package, member_path)
            if entry is not None:
                entries.append(entry)
                member_count += 1
        if not member_count:
            raise DroneRFManifestError(f"压缩包中没有 CSV 成员：{relative}")
    return build_recordings(entries)


def write_recording_manifest(path: str | Path, recordings: Iterable[DroneRFRecording]) -> Path:
    """Write a deterministic CSV manifest consumed by later conversion steps."""
    rows = list(recordings)
    if not rows:
        raise DroneRFManifestError("不能写入空的 DroneRF 录制清单")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    temporary.replace(destination)
    return destination


def read_recording_manifest(path: str | Path) -> list[DroneRFRecording]:
    """Read and validate a recording manifest produced by this module."""
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DroneRFManifestError(f"DroneRF 录制清单为空：{source}")
    recordings = []
    for row in rows:
        present_text = row["drone_present"].strip().lower()
        if present_text not in {"true", "false"}:
            raise DroneRFManifestError(f"非法 drone_present：{row['drone_present']}")
        recording = DroneRFRecording(
            recording_id=row["recording_id"],
            source_scenario_group=row["source_scenario_group"],
            code=row["code"],
            drone_present=present_text == "true",
            drone_type=row["drone_type"],
            operation_mode=row["operation_mode"],
            segment_index=int(row["segment_index"]),
            low_package_path=row["low_package_path"],
            low_member_path=row["low_member_path"],
            high_package_path=row["high_package_path"],
            high_member_path=row["high_member_path"],
        )
        expected_label = decode_label(recording.code)
        if (
            recording.drone_present != expected_label.drone_present
            or recording.drone_type != expected_label.drone_type
            or recording.operation_mode != expected_label.operation_mode
        ):
            raise DroneRFManifestError(
                f"录制清单标签与代码不一致：{recording.recording_id}"
            )
        recordings.append(recording)
    if len({item.recording_id for item in recordings}) != len(recordings):
        raise DroneRFManifestError("录制清单包含重复 recording_id")
    return recordings


def write_split_manifest(
    path: str | Path, assignments: Iterable[DroneRFSplitAssignment]
) -> Path:
    """Write deterministic recording-level split assignments as CSV."""
    rows = list(assignments)
    if not rows:
        raise DroneRFManifestError("不能写入空的 DroneRF 划分清单")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    temporary.replace(destination)
    return destination


def read_split_manifest(path: str | Path) -> list[DroneRFSplitAssignment]:
    """Read split assignments and reject malformed or mixed protocols."""
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DroneRFManifestError(f"DroneRF 划分清单为空：{source}")
    assignments = [
        DroneRFSplitAssignment(
            recording_id=row["recording_id"],
            source_scenario_group=row["source_scenario_group"],
            code=row["code"],
            split=row["split"],
            split_seed=int(row["split_seed"]),
            split_protocol=row["split_protocol"],
        )
        for row in rows
    ]
    if len({item.recording_id for item in assignments}) != len(assignments):
        raise DroneRFManifestError("划分清单包含重复 recording_id")
    if len({(item.split_seed, item.split_protocol) for item in assignments}) != 1:
        raise DroneRFManifestError("划分清单混用了不同随机种子或协议")
    return assignments


__all__ = [
    "DroneRFArchiveEntry",
    "DroneRFLabel",
    "DroneRFManifestError",
    "DroneRFPackage",
    "DroneRFRecording",
    "DroneRFSplitAssignment",
    "assign_recording_splits",
    "audit_recording_splits",
    "build_recordings",
    "decode_label",
    "discover_recordings",
    "parse_member_path",
    "parse_package_path",
    "read_recording_manifest",
    "read_split_manifest",
    "write_recording_manifest",
    "write_split_manifest",
]
