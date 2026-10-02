"""Rebuild DroneRF v1 recording/split manifests from verified nested RARs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data_provenance import sha256_file
from .dronerf import (
    assign_recording_splits,
    audit_recording_splits,
    discover_recordings,
    write_recording_manifest,
    write_split_manifest,
)


def prepare(
    package_root: Path,
    output_dir: Path,
    *,
    seed: int,
    source_archive: Path | None = None,
) -> dict:
    recordings = discover_recordings(package_root)
    assignments = assign_recording_splits(recordings, seed=seed)
    report = audit_recording_splits(recordings, assignments)
    if not report["ok"]:
        raise RuntimeError(f"DroneRF 划分审计失败：{report}")

    recording_path = write_recording_manifest(
        output_dir / "recording-manifest.csv", recordings
    )
    split_path = write_split_manifest(output_dir / "split-manifest.csv", assignments)
    report.update({
        "dataset": "DroneRF",
        "dataset_version": "1",
        "package_count": len(list(package_root.rglob("*.rar"))),
        "counts_by_type": dict(sorted(Counter(item.drone_type for item in recordings).items())),
        "recording_manifest": recording_path.name,
        "recording_manifest_sha256": sha256_file(recording_path),
        "split_manifest": split_path.name,
        "split_manifest_sha256": sha256_file(split_path),
        "source_archive_sha256": (
            sha256_file(source_archive) if source_archive is not None else None
        ),
        "training_eligible": False,
        "training_blocker": (
            "CSV 尚未转换并完成数值/长度校验；清单和录制级划分通过不等于训练数据已就绪。"
        ),
    })
    report_path = output_dir / "split-audit.json"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--source-archive", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare(
        args.package_root,
        args.output_dir,
        seed=args.seed,
        source_archive=args.source_archive,
    )
    print(json.dumps({
        "ok": report["ok"],
        "recording_count": report["recording_count"],
        "package_count": report["package_count"],
        "seed": report["seed"],
        "training_eligible": report["training_eligible"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
