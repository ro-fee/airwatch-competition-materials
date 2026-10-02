"""Materialize frozen KU Leuven train/validation IQ windows into NumPy arrays."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable
import zipfile

import numpy as np

from .data_provenance import sha256_file
from .ku_leuven_preprocessing import (
    INPUT_SAMPLE_RATE_HZ,
    INPUT_WINDOW_SAMPLES,
    PREPROCESSING_ID,
    preprocess_iq_window,
)
from .matlab_v73_iq import read_mat_v73_iq_windows_fileobj


ALLOWED_SPLITS = ("train", "validation")


class KULeuvenMaterializationError(ValueError):
    """Raised when frozen evidence cannot be materialized safely."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise KULeuvenMaterializationError(f"CSV 为空：{path}")
    return rows


def materialize_known_windows(
    *,
    split_assignments: str | Path,
    sources: Iterable[tuple[str, str | Path, str | Path]],
    output_dir: str | Path,
    splits: Iterable[str] = ALLOWED_SPLITS,
) -> Path:
    requested_splits = tuple(splits)
    if not requested_splits or len(set(requested_splits)) != len(requested_splits):
        raise KULeuvenMaterializationError("划分列表为空或重复")
    forbidden = sorted(set(requested_splits) - set(ALLOWED_SPLITS))
    if forbidden:
        raise KULeuvenMaterializationError(
            f"该入口只物化训练/验证集，拒绝：{forbidden}"
        )
    source_items = tuple((archive_id, Path(archive), Path(plan)) for archive_id, archive, plan in sources)
    source_map = {archive_id: (archive, plan) for archive_id, archive, plan in source_items}
    if len(source_map) != len(source_items):
        raise KULeuvenMaterializationError("archive_id 不得重复")

    split_path = Path(split_assignments)
    assignments = [
        row for row in _read_csv(split_path) if row["split"] in requested_splits
    ]
    if not assignments:
        raise KULeuvenMaterializationError("没有匹配的训练/验证成员")
    if set(row["archive_id"] for row in assignments) != set(source_map):
        raise KULeuvenMaterializationError("物化源与划分中的已知归档不一致")

    plans: dict[str, dict[str, list[dict[str, str]]]] = {}
    for archive_id, (_, plan_path) in source_map.items():
        by_recording: dict[str, list[dict[str, str]]] = {}
        for row in _read_csv(plan_path):
            by_recording.setdefault(row["recording_id"], []).append(row)
        for rows in by_recording.values():
            rows.sort(key=lambda row: int(row["start_sample"]))
        plans[archive_id] = by_recording

    counts = {split: 0 for split in requested_splits}
    for assignment in assignments:
        windows = plans[assignment["archive_id"]].get(assignment["recording_id"], [])
        if not windows:
            raise KULeuvenMaterializationError(
                f"{assignment['recording_id']} 没有冻结窗口"
            )
        if any(int(row["window_length"]) != INPUT_WINDOW_SAMPLES for row in windows):
            raise KULeuvenMaterializationError("窗口长度与预处理协议不一致")
        counts[assignment["split"]] += len(windows)

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite materialized dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
    arrays: dict[str, tuple[np.memmap, np.memmap]] = {}
    metadata_rows: dict[str, list[dict[str, str | int]]] = {
        split: [] for split in requested_splits
    }
    offsets = {split: 0 for split in requested_splits}
    try:
        for split, count in counts.items():
            split_dir = staging / split
            split_dir.mkdir()
            data = np.lib.format.open_memmap(
                split_dir / "data.npy", mode="w+", dtype=np.float32,
                shape=(count, 2, INPUT_WINDOW_SAMPLES),
            )
            labels = np.lib.format.open_memmap(
                split_dir / "labels.npy", mode="w+", dtype=np.int64, shape=(count,)
            )
            arrays[split] = (data, labels)

        processed_recordings = 0
        for archive_id, (archive_path, _) in source_map.items():
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            archive_assignments = sorted(
                (row for row in assignments if row["archive_id"] == archive_id),
                key=lambda row: int(row["member_index"]),
            )
            with zipfile.ZipFile(archive_path, "r") as archive:
                for assignment in archive_assignments:
                    window_rows = plans[archive_id][assignment["recording_id"]]
                    starts = tuple(int(row["start_sample"]) for row in window_rows)
                    with archive.open(assignment["member_path"], "r") as member:
                        raw_windows = read_mat_v73_iq_windows_fileobj(
                            member,
                            source_name=f"{archive_path.name}::{assignment['member_path']}",
                            start_samples=starts,
                            window_size=INPUT_WINDOW_SAMPLES,
                            sample_rate_hz=INPUT_SAMPLE_RATE_HZ,
                        )
                    split = assignment["split"]
                    data, labels = arrays[split]
                    for plan_row, raw_window in zip(window_rows, raw_windows):
                        index = offsets[split]
                        data[index] = preprocess_iq_window(raw_window.samples)
                        labels[index] = int(assignment["label_index"])
                        metadata_rows[split].append(
                            {
                                "data_index": index,
                                "window_id": plan_row["window_id"],
                                "recording_id": assignment["recording_id"],
                                "archive_id": archive_id,
                                "label_index": int(assignment["label_index"]),
                                "member_index": int(assignment["member_index"]),
                                "member_path": assignment["member_path"],
                                "member_sha256": assignment["member_sha256"],
                                "split": split,
                                "start_sample": int(plan_row["start_sample"]),
                                "end_sample_exclusive": int(plan_row["end_sample_exclusive"]),
                                "preprocessing_id": PREPROCESSING_ID,
                            }
                        )
                        offsets[split] += 1
                    processed_recordings += 1
                    if processed_recordings % 10 == 0:
                        print(f"materialized_recordings={processed_recordings}/{len(assignments)}", flush=True)

        for split, (data, labels) in arrays.items():
            data.flush()
            labels.flush()
            if offsets[split] != counts[split]:
                raise AssertionError(f"{split} materialized count mismatch")
        del data, labels
        arrays.clear()

        for split, rows in metadata_rows.items():
            path = staging / split / "windows.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        artifacts = {}
        for split in requested_splits:
            artifacts[split] = {}
            for name in ("data.npy", "labels.npy", "windows.csv"):
                path = staging / split / name
                artifacts[split][name] = {
                    "path": f"{split}/{name}",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        report = {
            "schema_version": "1.0",
            "artifact_type": "ku_leuven_materialized_known_train_validation",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "preprocessing_id": PREPROCESSING_ID,
            "shape_per_window": [2, INPUT_WINDOW_SAMPLES],
            "dtype": "float32",
            "split_counts": counts,
            "source_split_assignments": {
                "path": split_path.as_posix(), "sha256": sha256_file(split_path)
            },
            "source_window_plans": [
                {"path": plan.as_posix(), "sha256": sha256_file(plan)}
                for _, _, plan in source_items
            ],
            "artifacts": artifacts,
            "test_materialized": False,
            "unknown_materialized": False,
            "training_data_ready": True,
            "performance_evidence": False,
        }
        (staging / "dataset-metadata.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(destination)
    except Exception:
        arrays.clear()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument(
        "--source", action="append", nargs=3,
        metavar=("ARCHIVE_ID", "ZIP", "WINDOW_PLAN"), required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", choices=ALLOWED_SPLITS)
    args = parser.parse_args()
    output = materialize_known_windows(
        split_assignments=args.split_assignments,
        sources=[(archive_id, zip_path, plan) for archive_id, zip_path, plan in args.source],
        output_dir=args.output_dir,
        splits=args.split or ALLOWED_SPLITS,
    )
    print(json.dumps({"ok": True, "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = ["ALLOWED_SPLITS", "KULeuvenMaterializationError", "materialize_known_windows"]
