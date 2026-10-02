"""Recording manifest and deterministic window plan for KU Leuven ZIP archives."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path, PurePosixPath
import re
from typing import Any, BinaryIO
import zipfile

import numpy as np

from .data_provenance import sha256_file
from .matlab_v73_iq import MATV73IQError, inspect_mat_v73_iq_fileobj


class KULeuvenManifestError(ValueError):
    """Raised when archive members cannot form a trustworthy recording manifest."""


@dataclass(frozen=True)
class KULeuvenArchiveSpec:
    archive_id: str
    member_pattern: str
    device_group: str
    device_label: str
    expected_recording_count: int
    sample_rate_hz: float = 100_000_000
    center_frequency_hz: float = 2_440_000_000
    dataset_name: str = "uhd_samps"
    dataset_version: str = "1.0"
    dataset_doi: str = "10.48804/HZRVNZ"
    expected_archive_bytes: int | None = None
    expected_archive_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("archive_id", "member_pattern", "device_group", "device_label"):
            if not str(getattr(self, name)).strip():
                raise KULeuvenManifestError(f"{name} 不能为空")
        if self.expected_recording_count < 1:
            raise KULeuvenManifestError("expected_recording_count 必须大于 0")
        if self.sample_rate_hz <= 0 or self.center_frequency_hz <= 0:
            raise KULeuvenManifestError("采样率和中心频率必须为正数")
        try:
            compiled = re.compile(self.member_pattern)
        except re.error as exc:
            raise KULeuvenManifestError(f"member_pattern 无效：{exc}") from exc
        if "index" not in compiled.groupindex:
            raise KULeuvenManifestError("member_pattern 必须包含命名组 (?P<index>...)")
        if self.expected_archive_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.expected_archive_sha256
        ):
            raise KULeuvenManifestError("expected_archive_sha256 必须是小写 SHA-256")


@dataclass(frozen=True)
class KULeuvenWindowSpec:
    window_length: int = 4096
    windows_per_recording: int = 32
    seed: int = 20260908
    preprocessing_version: str = "ku-leuven-single-band-iq-window-plan-v1"

    def __post_init__(self) -> None:
        for name in ("window_length", "windows_per_recording", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise KULeuvenManifestError(f"{name} 必须是整数")
        if self.window_length < 16:
            raise KULeuvenManifestError("window_length 至少为 16")
        if self.windows_per_recording < 1:
            raise KULeuvenManifestError("windows_per_recording 必须大于 0")
        if self.seed < 0:
            raise KULeuvenManifestError("seed 不能为负数")
        if not self.preprocessing_version.strip():
            raise KULeuvenManifestError("preprocessing_version 不能为空")


@dataclass(frozen=True)
class KULeuvenRecording:
    recording_id: str
    archive_id: str
    archive_path: str
    member_path: str
    member_index: int
    device_group: str
    device_label: str
    member_size_bytes: int
    member_compressed_bytes: int
    member_crc32: str
    member_sha256: str
    dataset_name: str
    source_shape: str
    source_dtype: str
    matlab_class: str | None
    sample_count: int
    sample_rate_hz: float
    center_frequency_hz: float
    duration_seconds: float
    split: str = "unassigned"


SJRC_PRO_SPEC = KULeuvenArchiveSpec(
    archive_id="sjrc-pro-v1",
    member_pattern=r"SJRC_pro/sjrc__(?P<index>\d+)\.mat",
    device_group="SJRC_pro",
    device_label="SJRC F11 Pro drone and remote-controller RF",
    expected_recording_count=51,
    expected_archive_bytes=1_444_331_879,
    expected_archive_sha256="4fc096a61f07cdc5e38b005578056264dad4742fa9a1778ed41abe89310f7cb9",
)


FRYSKY_SPEC = KULeuvenArchiveSpec(
    archive_id="frysky-v1",
    member_pattern=r"Frysky/Frysky_(?P<index>\d+)\.mat",
    device_group="Frysky",
    device_label="RadioMaster/Taranis Frysky remote-controller RF",
    expected_recording_count=51,
    expected_archive_bytes=1_707_367_398,
    expected_archive_sha256="062aa766b5c28adf317b6f8759b1892e0862fc94e52a3b5abb78aa42b98a4be1",
)


SPEKTRUM_DX4E_SPEC = KULeuvenArchiveSpec(
    archive_id="spektrum-dx4e-v1",
    member_pattern=r"Spektrum_DX4e/DX4e_(?P<index>\d+)\.mat",
    device_group="Spektrum_DX4e",
    device_label="Spektrum DX4e remote-controller RF",
    expected_recording_count=71,
    expected_archive_bytes=2_026_208_466,
    expected_archive_sha256="fc80f0552abd72f306088004b114d9329b7c86a47a7dee247400803f5f6bad06",
)


MINI2_RC_SPEC = KULeuvenArchiveSpec(
    archive_id="mini2-rc-v1",
    member_pattern=r"mini2RC/mini2_(?P<index>\d+)\.mat",
    device_group="mini2RC",
    device_label="DJI Mini 2 remote-controller RF",
    expected_recording_count=71,
    expected_archive_bytes=2_078_132_999,
    expected_archive_sha256="230ac862bfa50dad70082fbf413f13e2331a229e891eaeeb3ee278bb8de8b2d4",
)


ARCHIVE_SPECS = {
    "sjrc-pro": SJRC_PRO_SPEC,
    "frysky": FRYSKY_SPEC,
    "spektrum-dx4e": SPEKTRUM_DX4E_SPEC,
    "mini2-rc": MINI2_RC_SPEC,
}


RECORDING_FIELDS = tuple(KULeuvenRecording.__dataclass_fields__)
WINDOW_FIELDS = (
    "window_id",
    "recording_id",
    "archive_id",
    "member_path",
    "device_group",
    "device_label",
    "split",
    "start_sample",
    "end_sample_exclusive",
    "window_length",
    "sample_rate_hz",
    "center_frequency_hz",
    "preprocessing_version",
    "seed",
)


def _stream_sha256(stream: BinaryIO, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def inspect_zip_recordings(
    archive_path: str | Path,
    spec: KULeuvenArchiveSpec,
) -> tuple[KULeuvenRecording, ...]:
    """Audit every matching MAT member directly inside a ZIP, without extraction."""
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(f"找不到 KU Leuven ZIP：{archive}")
    if spec.expected_archive_bytes is not None and archive.stat().st_size != spec.expected_archive_bytes:
        raise KULeuvenManifestError(
            f"ZIP 字节数不一致：期望 {spec.expected_archive_bytes}，实际 {archive.stat().st_size}"
        )
    archive_sha256 = sha256_file(archive)
    if spec.expected_archive_sha256 is not None and archive_sha256 != spec.expected_archive_sha256:
        raise KULeuvenManifestError("ZIP SHA-256 与固定来源记录不一致")

    pattern = re.compile(spec.member_pattern)
    try:
        with zipfile.ZipFile(archive, "r") as handle:
            infos = [item for item in handle.infolist() if not item.is_dir()]
            if len({item.filename for item in infos}) != len(infos):
                raise KULeuvenManifestError("ZIP 包含重名成员")
            if any(not _safe_member_path(item.filename) for item in infos):
                raise KULeuvenManifestError("ZIP 包含不安全成员路径")

            matched: list[tuple[int, zipfile.ZipInfo]] = []
            unexpected = []
            for info in infos:
                match = pattern.fullmatch(info.filename)
                if match is None:
                    unexpected.append(info.filename)
                    continue
                matched.append((int(match.group("index")), info))
            if unexpected:
                raise KULeuvenManifestError(
                    f"ZIP 包含不符合固定命名规则的文件：{unexpected[:3]}"
                )
            if len(matched) != spec.expected_recording_count:
                raise KULeuvenManifestError(
                    f"录制数不一致：期望 {spec.expected_recording_count}，实际 {len(matched)}"
                )
            matched.sort(key=lambda pair: pair[0])
            indices = [index for index, _ in matched]
            if indices != list(range(spec.expected_recording_count)):
                raise KULeuvenManifestError("成员序号必须从 0 开始连续且不重复")

            recordings = []
            for index, info in matched:
                source_name = f"{archive.name}::{info.filename}"
                try:
                    with handle.open(info, "r") as member:
                        metadata = inspect_mat_v73_iq_fileobj(
                            member,
                            source_name=source_name,
                            dataset_name=spec.dataset_name,
                            sample_rate_hz=spec.sample_rate_hz,
                        )
                    with handle.open(info, "r") as member:
                        member_sha256 = _stream_sha256(member)
                except (OSError, zipfile.BadZipFile, MATV73IQError) as exc:
                    raise KULeuvenManifestError(
                        f"无法核验 ZIP 成员 {info.filename}：{exc}"
                    ) from exc
                recordings.append(
                    KULeuvenRecording(
                        recording_id=f"ku-leuven:{spec.archive_id}:{index:04d}",
                        archive_id=spec.archive_id,
                        archive_path=archive.as_posix(),
                        member_path=info.filename,
                        member_index=index,
                        device_group=spec.device_group,
                        device_label=spec.device_label,
                        member_size_bytes=info.file_size,
                        member_compressed_bytes=info.compress_size,
                        member_crc32=f"{info.CRC:08x}",
                        member_sha256=member_sha256,
                        dataset_name=metadata.dataset_name,
                        source_shape="x".join(str(value) for value in metadata.source_shape),
                        source_dtype=metadata.source_dtype,
                        matlab_class=metadata.matlab_class,
                        sample_count=metadata.sample_count,
                        sample_rate_hz=metadata.sample_rate_hz,
                        center_frequency_hz=spec.center_frequency_hz,
                        duration_seconds=metadata.duration_seconds,
                    )
                )
    except zipfile.BadZipFile as exc:
        raise KULeuvenManifestError(f"无效或损坏的 ZIP：{exc}") from exc
    return tuple(recordings)


def stratified_iq_window_offsets(
    recording_id: str,
    sample_count: int,
    spec: KULeuvenWindowSpec,
) -> tuple[int, ...]:
    """Choose stable windows fully contained in disjoint temporal strata."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise KULeuvenManifestError("sample_count 必须是整数")
    minimum = spec.window_length * spec.windows_per_recording
    if sample_count < minimum:
        raise KULeuvenManifestError(
            f"录制只有 {sample_count} 点，不足以容纳 {spec.windows_per_recording} 个互不重叠窗口"
        )
    edges = np.linspace(0, int(sample_count), spec.windows_per_recording + 1, dtype=np.int64)
    digest = hashlib.sha256(
        f"{spec.seed}\0{recording_id}\0{spec.preprocessing_version}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    offsets = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        low = int(lower)
        high_inclusive = int(upper) - spec.window_length
        if high_inclusive < low:
            raise KULeuvenManifestError("时间分层不足以容纳完整窗口")
        offsets.append(int(rng.integers(low, high_inclusive + 1)))
    return tuple(offsets)


def build_window_plan(
    recordings: tuple[KULeuvenRecording, ...],
    spec: KULeuvenWindowSpec,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for recording in recordings:
        offsets = stratified_iq_window_offsets(
            recording.recording_id, recording.sample_count, spec
        )
        for window_index, start in enumerate(offsets):
            rows.append(
                {
                    "window_id": f"{recording.recording_id}:w{window_index:03d}",
                    "recording_id": recording.recording_id,
                    "archive_id": recording.archive_id,
                    "member_path": recording.member_path,
                    "device_group": recording.device_group,
                    "device_label": recording.device_label,
                    "split": recording.split,
                    "start_sample": start,
                    "end_sample_exclusive": start + spec.window_length,
                    "window_length": spec.window_length,
                    "sample_rate_hz": recording.sample_rate_hz,
                    "center_frequency_hz": recording.center_frequency_hz,
                    "preprocessing_version": spec.preprocessing_version,
                    "seed": spec.seed,
                }
            )
    return tuple(rows)


def build_manifest_bundle(
    archive_path: str | Path,
    archive_spec: KULeuvenArchiveSpec,
    window_spec: KULeuvenWindowSpec,
) -> dict[str, Any]:
    recordings = inspect_zip_recordings(archive_path, archive_spec)
    windows = build_window_plan(recordings, window_spec)
    recording_ids = {row.recording_id for row in recordings}
    window_recording_ids = {row["recording_id"] for row in windows}
    nonoverlap = all(
        right["start_sample"] >= left["end_sample_exclusive"]
        for recording_id in recording_ids
        for left, right in zip(
            [row for row in windows if row["recording_id"] == recording_id][:-1],
            [row for row in windows if row["recording_id"] == recording_id][1:],
        )
    )
    verification = {
        "schema_version": "1.0",
        "artifact_type": "ku_leuven_recording_manifest_and_window_plan_verification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "KU Leuven Drone RF Dataset",
        "dataset_version": archive_spec.dataset_version,
        "dataset_doi": archive_spec.dataset_doi,
        "archive": {
            "path": Path(archive_path).as_posix(),
            "size_bytes": Path(archive_path).stat().st_size,
            "sha256": sha256_file(Path(archive_path)),
            "archive_id": archive_spec.archive_id,
        },
        "recording_count": len(recordings),
        "recording_ids_unique": len(recording_ids) == len(recordings),
        "member_sha256_complete": all(len(row.member_sha256) == 64 for row in recordings),
        "content_metadata_checked_for_every_recording": True,
        "sample_counts": sorted({row.sample_count for row in recordings}),
        "window_count": len(windows),
        "windows_per_recording": window_spec.windows_per_recording,
        "window_length": window_spec.window_length,
        "window_recording_ids_exact": window_recording_ids == recording_ids,
        "windows_nonoverlapping_within_recording": nonoverlap,
        "split_values": sorted({row.split for row in recordings}),
        "ok": bool(recordings) and nonoverlap and window_recording_ids == recording_ids,
        "training_eligible": False,
        "training_blocker": (
            "Window rows remain unassigned. Official metadata does not establish that each MAT "
            "file is an independent physical acquisition session. A defensible group split and "
            "compatible single-band IQ model protocol must be frozen before training or evaluation."
        ),
    }
    return {"recordings": recordings, "windows": windows, "verification": verification}


def _csv_text(rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_bundle_index(
    recording_manifest: str | Path,
    window_plan: str | Path,
    verification: str | Path,
) -> dict[str, Any]:
    paths = {
        "recording_manifest": Path(recording_manifest),
        "window_plan": Path(window_plan),
        "verification": Path(verification),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest evidence is incomplete: {missing}")
    return {
        "schema_version": "1.0",
        "artifact_type": "ku_leuven_manifest_bundle_index",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            name: {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "interpretation_limits": [
            "MAT member boundaries are verified, but independent physical acquisition sessions are not established by the official metadata.",
            "All recording and window rows remain split=unassigned; this bundle is not training or evaluation authorization.",
        ],
    }


def write_bundle_index(index: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite existing manifest bundle index")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_manifest_bundle(bundle: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    index_path = root / "bundle-index.json"
    destinations = {
        "recordings": root / "recording-manifest.csv",
        "windows": root / "window-plan.csv",
        "verification": root / "manifest-verification.json",
    }
    existing = [str(path) for path in (*destinations.values(), index_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing manifest evidence: {existing}")
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "recordings": _csv_text(
            [asdict(row) for row in bundle["recordings"]], RECORDING_FIELDS
        ),
        "windows": _csv_text(list(bundle["windows"]), WINDOW_FIELDS),
        "verification": json.dumps(
            bundle["verification"], ensure_ascii=False, indent=2
        ) + "\n",
    }
    temporaries = {name: path.with_suffix(path.suffix + ".tmp") for name, path in destinations.items()}
    try:
        for name, temporary in temporaries.items():
            temporary.write_text(payloads[name], encoding="utf-8", newline="")
        for name in ("recordings", "windows", "verification"):
            temporaries[name].replace(destinations[name])
    finally:
        for temporary in temporaries.values():
            if temporary.exists():
                temporary.unlink()
    write_bundle_index(
        build_bundle_index(
            destinations["recordings"], destinations["windows"], destinations["verification"]
        ),
        index_path,
    )
    destinations["index"] = index_path
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=tuple(ARCHIVE_SPECS), default="sjrc-pro",
        help="fixed archive identity and member naming contract",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=4096)
    parser.add_argument("--windows-per-recording", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    bundle = build_manifest_bundle(
        args.archive,
        ARCHIVE_SPECS[args.profile],
        KULeuvenWindowSpec(
            window_length=args.window_length,
            windows_per_recording=args.windows_per_recording,
            seed=args.seed,
        ),
    )
    outputs = write_manifest_bundle(bundle, args.output_dir)
    print(
        json.dumps(
            {
                "recording_count": len(bundle["recordings"]),
                "window_count": len(bundle["windows"]),
                "training_eligible": bundle["verification"]["training_eligible"],
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHIVE_SPECS",
    "FRYSKY_SPEC",
    "KULeuvenArchiveSpec",
    "KULeuvenManifestError",
    "KULeuvenRecording",
    "KULeuvenWindowSpec",
    "MINI2_RC_SPEC",
    "SJRC_PRO_SPEC",
    "SPEKTRUM_DX4E_SPEC",
    "build_bundle_index",
    "build_manifest_bundle",
    "build_window_plan",
    "inspect_zip_recordings",
    "stratified_iq_window_offsets",
    "write_bundle_index",
    "write_manifest_bundle",
]
