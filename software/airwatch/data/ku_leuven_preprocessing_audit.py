"""Run source-bound acceptance checks for the frozen KU Leuven IQ preprocessing."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .data_provenance import sha256_file
from .ku_leuven_preprocessing import (
    INPUT_CHANNEL_SEMANTICS,
    INPUT_SAMPLE_RATE_HZ,
    INPUT_WINDOW_SAMPLES,
    PREPROCESSING_ID,
    preprocess_iq_window,
)
from .matlab_v73_iq import read_mat_v73_iq_window_fileobj


class KULeuvenPreprocessingAuditError(ValueError):
    """Raised when evidence inputs do not satisfy the acceptance contract."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise KULeuvenPreprocessingAuditError(f"证据 CSV 为空：{path}")
    return rows


def audit_real_windows(
    *,
    split_assignments: str | Path,
    sources: list[tuple[str, str | Path, str | Path]],
) -> dict:
    split_path = Path(split_assignments)
    assignments = _read_csv(split_path)
    if len({archive_id for archive_id, _, _ in sources}) != len(sources):
        raise KULeuvenPreprocessingAuditError("源 archive_id 不得重复")
    results = []
    for archive_id, archive_value, plan_value in sources:
        archive = Path(archive_value)
        plan_path = Path(plan_value)
        if not archive.is_file():
            raise FileNotFoundError(archive)
        candidates = sorted(
            (
                row for row in assignments
                if row["archive_id"] == archive_id and row["split"] == "train"
            ),
            key=lambda row: int(row["member_index"]),
        )
        if not candidates:
            raise KULeuvenPreprocessingAuditError(f"{archive_id} 没有训练成员")
        assignment = candidates[0]
        plan_rows = _read_csv(plan_path)
        windows = sorted(
            (row for row in plan_rows if row["recording_id"] == assignment["recording_id"]),
            key=lambda row: row["window_id"],
        )
        if not windows:
            raise KULeuvenPreprocessingAuditError(
                f"{assignment['recording_id']} 没有预注册窗口"
            )
        window_row = windows[0]
        if window_row["member_path"] != assignment["member_path"]:
            raise KULeuvenPreprocessingAuditError("划分与窗口计划的成员路径不一致")
        start = int(window_row["start_sample"])
        with zipfile.ZipFile(archive, "r") as handle:
            with handle.open(assignment["member_path"], "r") as member:
                raw = read_mat_v73_iq_window_fileobj(
                    member,
                    source_name=f"{archive.name}::{assignment['member_path']}",
                    start_sample=start,
                    window_size=INPUT_WINDOW_SAMPLES,
                    sample_rate_hz=INPUT_SAMPLE_RATE_HZ,
                ).samples
        processed = preprocess_iq_window(raw)
        channel_means = processed.mean(axis=1, dtype=np.float64)
        complex_rms = float(
            np.sqrt(np.square(processed, dtype=np.float64).sum(axis=0).mean())
        )
        result = {
            "archive_id": archive_id,
            "archive_path": archive.as_posix(),
            "recording_id": assignment["recording_id"],
            "member_path": assignment["member_path"],
            "member_sha256": assignment["member_sha256"],
            "window_id": window_row["window_id"],
            "start_sample": start,
            "shape": list(processed.shape),
            "dtype": str(processed.dtype),
            "channel_means": [float(value) for value in channel_means],
            "complex_rms": complex_rms,
            "finite": bool(np.isfinite(processed).all()),
            "output_sha256": hashlib.sha256(processed.tobytes(order="C")).hexdigest(),
        }
        result["ok"] = (
            result["shape"] == [2, INPUT_WINDOW_SAMPLES]
            and result["dtype"] == "float32"
            and result["finite"]
            and max(abs(value) for value in result["channel_means"]) <= 1e-5
            and abs(result["complex_rms"] - 1.0) <= 1e-5
        )
        results.append(result)
    return {
        "schema_version": "1.0",
        "artifact_type": "ku_leuven_real_iq_preprocessing_acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "input_shape": [2, INPUT_WINDOW_SAMPLES],
            "channel_semantics": list(INPUT_CHANNEL_SEMANTICS),
            "sample_rate_hz": INPUT_SAMPLE_RATE_HZ,
            "operations": ["per_channel_dc_removal", "per_window_complex_rms_normalization"],
            "uses_cross_window_statistics": False,
            "uses_validation_or_test_statistics": False,
        },
        "source_evidence": {
            "split_assignments": {
                "path": split_path.as_posix(),
                "sha256": sha256_file(split_path),
            },
            "window_plans": [
                {"path": Path(plan).as_posix(), "sha256": sha256_file(Path(plan))}
                for _, _, plan in sources
            ],
        },
        "samples": results,
        "ok": bool(results) and all(row["ok"] for row in results),
        "performance_evidence": False,
        "interpretation": "This artifact validates data access and preprocessing only; it is not model-performance evidence.",
    }


def write_audit(report: dict, output_path: str | Path) -> Path:
    if not report.get("ok"):
        raise KULeuvenPreprocessingAuditError("拒绝写入未通过的验收报告")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("refusing to overwrite preprocessing acceptance evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument(
        "--source", action="append", nargs=3, metavar=("ARCHIVE_ID", "ZIP", "WINDOW_PLAN"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_real_windows(
        split_assignments=args.split_assignments,
        sources=[(archive_id, Path(zip_path), Path(plan)) for archive_id, zip_path, plan in args.source],
    )
    output = write_audit(report, args.output)
    print(json.dumps({"ok": report["ok"], "sample_count": len(report["samples"]), "output": str(output)}))


if __name__ == "__main__":
    main()


__all__ = ["KULeuvenPreprocessingAuditError", "audit_real_windows", "write_audit"]
