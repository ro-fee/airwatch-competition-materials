"""Materialize compact, recording-isolated DroneRF window artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .data_provenance import sha256_file
from .dronerf import (
    DroneRFRecording,
    audit_recording_splits,
    read_recording_manifest,
    read_split_manifest,
)
from .dronerf_windows import (
    DroneRFWindowResult,
    DroneRFWindowSpec,
    extract_recording_windows,
)


ProgressCallback = Callable[[int, int, str], None]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"不能写入空清单：{path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_array(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(samples, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def materialize(
    recording_manifest: Path,
    split_manifest: Path,
    package_root: Path,
    output_dir: Path,
    spec: DroneRFWindowSpec,
    *,
    progress: ProgressCallback | None = None,
) -> dict:
    """Build a complete dataset into a temporary directory, then rename atomically."""
    recordings = read_recording_manifest(recording_manifest)
    assignments = read_split_manifest(split_manifest)
    split_audit = audit_recording_splits(recordings, assignments)
    if not split_audit["ok"]:
        raise RuntimeError(f"拒绝转换：录制级划分审计失败：{split_audit}")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"发现未处理的中间目录，拒绝覆盖：{partial}")
    partial.mkdir(parents=True)

    split_by_id = {item.recording_id: item.split for item in assignments}
    artifact_rows: list[dict] = []
    window_rows: list[dict] = []
    total = len(recordings)
    try:
        for position, recording in enumerate(recordings, start=1):
            result: DroneRFWindowResult = extract_recording_windows(
                recording, package_root, spec
            )
            relative_path = Path("recordings") / recording.code / (
                f"{recording.segment_index:03d}.npy"
            )
            artifact_path = partial / relative_path
            _save_array(artifact_path, result.samples)
            digest = sha256_file(artifact_path)
            artifact_rows.append({
                "recording_id": recording.recording_id,
                "artifact_path": relative_path.as_posix(),
                "size_bytes": artifact_path.stat().st_size,
                "sha256": digest,
                "windows": result.samples.shape[0],
                "channels": result.samples.shape[1],
                "window_length": result.samples.shape[2],
                "low_sample_count": result.low_sample_count,
                "high_sample_count": result.high_sample_count,
                "preprocessing_version": spec.preprocessing_version,
            })
            split = split_by_id[recording.recording_id]
            for window_index, offset in enumerate(result.offsets):
                window_rows.append({
                    "sample_id": f"{recording.recording_id}:w{window_index:03d}",
                    "recording_id": recording.recording_id,
                    "source": "DroneRF",
                    "dataset_version": "1",
                    "split": split,
                    "code": recording.code,
                    "drone_present": recording.drone_present,
                    "drone_type": recording.drone_type,
                    "operation_mode": recording.operation_mode,
                    "window_index": window_index,
                    "start_sample": offset,
                    "window_length": spec.window_length,
                    "channels": 2,
                    "artifact_path": relative_path.as_posix(),
                    "preprocessing_version": spec.preprocessing_version,
                })
            if progress is not None:
                progress(position, total, recording.recording_id)

        _write_csv(partial / "recording-artifacts.csv", artifact_rows)
        _write_csv(partial / "window-manifest.csv", window_rows)
        split_counts = Counter(row["split"] for row in window_rows)
        code_counts = Counter(row["code"] for row in window_rows)
        metadata = {
            "schema_version": "1.0",
            "dataset": "DroneRF",
            "dataset_version": "1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "window_spec": asdict(spec),
            "source_recording_manifest": recording_manifest.name,
            "source_recording_manifest_sha256": sha256_file(recording_manifest),
            "source_split_manifest": split_manifest.name,
            "source_split_manifest_sha256": sha256_file(split_manifest),
            "recording_count": len(recordings),
            "window_count": len(window_rows),
            "counts_by_split": dict(sorted(split_counts.items())),
            "counts_by_code": dict(sorted(code_counts.items())),
            "artifact_inventory": "recording-artifacts.csv",
            "window_manifest": "window-manifest.csv",
            "split_protocol": split_audit["protocol"],
            "split_seed": split_audit["seed"],
            "known_limitation": split_audit["known_limitation"],
            "normalization": "none; raw float32 RF amplitude windows",
        }
        (partial / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        partial.replace(output_dir)
        verification = verify_materialized_dataset(
            output_dir,
            recording_manifest=recording_manifest,
            split_manifest=split_manifest,
        )
        write_verification_report(output_dir / "verification.json", verification)
        if not verification["ok"]:
            raise RuntimeError(f"DroneRF 落盘复核失败：{verification['findings']}")
        return metadata
    except BaseException:
        # Preserve the partial directory for forensic inspection and explicit resume/removal.
        raise


def verify_materialized_dataset(
    dataset_root: Path,
    *,
    recording_manifest: Path | None = None,
    split_manifest: Path | None = None,
) -> dict:
    """Re-open every artifact and audit hashes, shapes, indices, and split isolation."""
    root = Path(dataset_root)
    findings: list[str] = []
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "findings": [f"metadata.json 无法读取：{exc}"]}

    def read_rows(name: str) -> list[dict]:
        try:
            with (root / name).open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        except Exception as exc:
            findings.append(f"{name} 无法读取：{exc}")
            return []

    artifacts = read_rows("recording-artifacts.csv")
    windows = read_rows("window-manifest.csv")
    expected_paths = {row.get("artifact_path", "") for row in artifacts}
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*.npy")
    }
    if expected_paths != actual_paths:
        findings.append("NPY 文件集合与 recording-artifacts.csv 不一致")

    verified_bytes = 0
    artifact_windows: dict[str, int] = {}
    for row in artifacts:
        relative = row.get("artifact_path", "")
        path = root / relative
        if not path.is_file():
            findings.append(f"缺少窗口文件：{relative}")
            continue
        verified_bytes += path.stat().st_size
        if path.stat().st_size != int(row["size_bytes"]):
            findings.append(f"文件大小不一致：{relative}")
        if sha256_file(path) != row["sha256"]:
            findings.append(f"SHA-256 不一致：{relative}")
        try:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            expected_shape = (
                int(row["windows"]), int(row["channels"]), int(row["window_length"])
            )
            if array.dtype != np.float32 or array.shape != expected_shape:
                findings.append(f"数组类型或形状不一致：{relative}")
            elif not np.isfinite(array).all():
                findings.append(f"数组包含非有限值：{relative}")
            artifact_windows[relative] = expected_shape[0]
        except Exception as exc:
            findings.append(f"NPY 无法读取：{relative}：{exc}")

    sample_ids = [row.get("sample_id", "") for row in windows]
    if len(set(sample_ids)) != len(sample_ids):
        findings.append("window-manifest.csv 包含重复 sample_id")
    splits_by_recording: dict[str, set[str]] = {}
    windows_by_artifact = Counter()
    for row in windows:
        splits_by_recording.setdefault(row["recording_id"], set()).add(row["split"])
        windows_by_artifact[row["artifact_path"]] += 1
        start = int(row["start_sample"])
        length = int(row["window_length"])
        if start < 0 or start + length > metadata["window_spec"]["expected_sample_count"]:
            findings.append(f"窗口越界：{row['sample_id']}")
    crossing = sorted(
        recording_id for recording_id, splits in splits_by_recording.items() if len(splits) != 1
    )
    if crossing:
        findings.append(f"存在跨集合 recording_id：{crossing[:5]}")
    for relative, expected_count in artifact_windows.items():
        if windows_by_artifact[relative] != expected_count:
            findings.append(f"窗口索引数量不一致：{relative}")

    if len(artifacts) != metadata.get("recording_count"):
        findings.append("录制产物数量与 metadata.json 不一致")
    if len(windows) != metadata.get("window_count"):
        findings.append("窗口数量与 metadata.json 不一致")
    split_counts = dict(sorted(Counter(row["split"] for row in windows).items()))
    if split_counts != metadata.get("counts_by_split"):
        findings.append("集合窗口数量与 metadata.json 不一致")
    if recording_manifest is not None and sha256_file(recording_manifest) != metadata.get(
        "source_recording_manifest_sha256"
    ):
        findings.append("源录制清单 SHA-256 不一致")
    if split_manifest is not None and sha256_file(split_manifest) != metadata.get(
        "source_split_manifest_sha256"
    ):
        findings.append("源划分清单 SHA-256 不一致")

    return {
        "ok": not findings,
        "dataset": metadata.get("dataset"),
        "dataset_version": metadata.get("dataset_version"),
        "preprocessing_version": metadata.get("window_spec", {}).get("preprocessing_version"),
        "verified_recording_files": len(artifacts),
        "verified_window_rows": len(windows),
        "verified_bytes": verified_bytes,
        "counts_by_split": split_counts,
        "recordings_crossing_splits": crossing,
        "findings": findings,
    }


def write_verification_report(path: Path, report: dict) -> Path:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=4096)
    parser.add_argument("--windows-per-recording", type=int, default=32)
    parser.add_argument("--expected-sample-count", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = DroneRFWindowSpec(
        window_length=args.window_length,
        windows_per_recording=args.windows_per_recording,
        expected_sample_count=args.expected_sample_count,
        seed=args.seed,
    )

    def show_progress(current: int, total: int, recording_id: str) -> None:
        print(f"[{current}/{total}] {recording_id}", flush=True)

    metadata = materialize(
        args.recording_manifest,
        args.split_manifest,
        args.package_root,
        args.output_dir,
        spec,
        progress=show_progress,
    )
    print(json.dumps({
        "recording_count": metadata["recording_count"],
        "window_count": metadata["window_count"],
        "counts_by_split": metadata["counts_by_split"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
