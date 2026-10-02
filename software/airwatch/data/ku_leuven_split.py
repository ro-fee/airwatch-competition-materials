"""Conservative known-source split protocol for audited KU Leuven manifests."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .data_provenance import sha256_file


class KULeuvenSplitError(ValueError):
    """Raised when source manifests cannot satisfy the frozen split contract."""


KNOWN_ARCHIVES = {
    "frysky-v1": 0,
    "spektrum-dx4e-v1": 1,
    "mini2-rc-v1": 2,
}
SPLIT_PROTOCOL = "contiguous-member-blocks-with-five-member-guards-v1"
ACTIVE_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class KULeuvenKnownRecording:
    recording_id: str
    archive_id: str
    member_index: int
    member_path: str
    device_group: str
    device_label: str
    member_sha256: str


@dataclass(frozen=True)
class KULeuvenKnownSplitAssignment:
    recording_id: str
    archive_id: str
    label_index: int
    member_index: int
    member_path: str
    device_group: str
    device_label: str
    member_sha256: str
    acquisition_proxy_group: str
    split: str
    split_protocol: str = SPLIT_PROTOCOL


def read_known_recordings(manifest_paths: Iterable[str | Path]) -> tuple[KULeuvenKnownRecording, ...]:
    paths = tuple(Path(path) for path in manifest_paths)
    if not paths:
        raise KULeuvenSplitError("至少需要一个已审计录制清单")
    recordings: list[KULeuvenKnownRecording] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"找不到录制清单：{path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise KULeuvenSplitError(f"录制清单为空：{path}")
        for row in rows:
            if row.get("split") != "unassigned":
                raise KULeuvenSplitError("只接受 split=unassigned 的原始审计清单")
            archive_id = row.get("archive_id", "")
            if archive_id not in KNOWN_ARCHIVES:
                raise KULeuvenSplitError(f"非预注册已知源不得进入划分：{archive_id}")
            member_sha256 = row.get("member_sha256", "")
            if len(member_sha256) != 64:
                raise KULeuvenSplitError("录制成员缺少完整 SHA-256")
            recordings.append(
                KULeuvenKnownRecording(
                    recording_id=row["recording_id"],
                    archive_id=archive_id,
                    member_index=int(row["member_index"]),
                    member_path=row["member_path"],
                    device_group=row["device_group"],
                    device_label=row["device_label"],
                    member_sha256=member_sha256,
                )
            )
    ids = [row.recording_id for row in recordings]
    if len(ids) != len(set(ids)):
        raise KULeuvenSplitError("已知源清单包含重复 recording_id")
    archive_ids = {row.archive_id for row in recordings}
    if archive_ids != set(KNOWN_ARCHIVES):
        missing = sorted(set(KNOWN_ARCHIVES) - archive_ids)
        extra = sorted(archive_ids - set(KNOWN_ARCHIVES))
        raise KULeuvenSplitError(f"已知源集合不完整：missing={missing}, extra={extra}")
    return tuple(recordings)


def assign_contiguous_proxy_splits(
    recordings: Iterable[KULeuvenKnownRecording],
    *,
    guard_members: int = 5,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[KULeuvenKnownSplitAssignment, ...]:
    """Split ordered members into contiguous blocks separated by unused guards.

    MAT files are the finest public grouping unit. The repository does not claim
    they are independent physical sessions; guard blocks only reduce adjacency.
    """
    if guard_members < 1:
        raise KULeuvenSplitError("guard_members 至少为 1")
    if not 0 < validation_fraction < 0.5 or not 0 < test_fraction < 0.5:
        raise KULeuvenSplitError("验证与测试比例必须介于 0 和 0.5 之间")
    if validation_fraction + test_fraction >= 1:
        raise KULeuvenSplitError("验证与测试比例之和必须小于 1")

    rows = tuple(recordings)
    assignments: list[KULeuvenKnownSplitAssignment] = []
    for archive_id in KNOWN_ARCHIVES:
        source = sorted(
            (row for row in rows if row.archive_id == archive_id),
            key=lambda row: row.member_index,
        )
        if [row.member_index for row in source] != list(range(len(source))):
            raise KULeuvenSplitError(f"{archive_id} 的成员序号必须从 0 连续递增")
        usable = len(source) - 2 * guard_members
        if usable < 3:
            raise KULeuvenSplitError(f"{archive_id} 不足以划分三个集合和两个隔离带")
        validation_count = max(1, round(usable * validation_fraction))
        test_count = max(1, round(usable * test_fraction))
        train_count = usable - validation_count - test_count
        if train_count < 1:
            raise KULeuvenSplitError(f"{archive_id} 的训练块为空")

        boundaries = (
            ("train", train_count),
            ("guard", guard_members),
            ("validation", validation_count),
            ("guard", guard_members),
            ("test", test_count),
        )
        cursor = 0
        guard_index = 0
        for split, count in boundaries:
            block = source[cursor : cursor + count]
            if split == "guard":
                guard_index += 1
                proxy_group = f"{archive_id}:guard-{guard_index}"
            else:
                proxy_group = f"{archive_id}:{split}-block"
            for row in block:
                assignments.append(
                    KULeuvenKnownSplitAssignment(
                        recording_id=row.recording_id,
                        archive_id=row.archive_id,
                        label_index=KNOWN_ARCHIVES[row.archive_id],
                        member_index=row.member_index,
                        member_path=row.member_path,
                        device_group=row.device_group,
                        device_label=row.device_label,
                        member_sha256=row.member_sha256,
                        acquisition_proxy_group=proxy_group,
                        split=split,
                    )
                )
            cursor += count
        if cursor != len(source):
            raise AssertionError("split construction did not consume every member")
    return tuple(sorted(assignments, key=lambda row: (row.label_index, row.member_index)))


def audit_known_splits(
    recordings: Iterable[KULeuvenKnownRecording],
    assignments: Iterable[KULeuvenKnownSplitAssignment],
    *,
    guard_members: int,
) -> dict[str, Any]:
    source_rows = tuple(recordings)
    split_rows = tuple(assignments)
    source_ids = {row.recording_id for row in source_rows}
    assigned_ids = [row.recording_id for row in split_rows]
    counts: dict[str, dict[str, int]] = {}
    separation_ok = True
    for archive_id in KNOWN_ARCHIVES:
        group = [row for row in split_rows if row.archive_id == archive_id]
        counts[archive_id] = {
            split: sum(row.split == split for row in group)
            for split in (*ACTIVE_SPLITS, "guard")
        }
        ranges = {
            split: [row.member_index for row in group if row.split == split]
            for split in ACTIVE_SPLITS
        }
        if any(not values for values in ranges.values()):
            separation_ok = False
        else:
            separation_ok &= (
                min(ranges["validation"]) - max(ranges["train"]) - 1 >= guard_members
                and min(ranges["test"]) - max(ranges["validation"]) - 1 >= guard_members
            )
    duplicate_ids = sorted({item for item in assigned_ids if assigned_ids.count(item) > 1})
    hashes = [row.member_sha256 for row in split_rows]
    duplicate_hashes = sorted({item for item in hashes if hashes.count(item) > 1})
    active_proxy_groups = {
        row.acquisition_proxy_group: row.split
        for row in split_rows
        if row.split in ACTIVE_SPLITS
    }
    proxy_group_disjoint = len(active_proxy_groups) == sum(
        1 for row in split_rows if row.split in ACTIVE_SPLITS and row.member_index == min(
            candidate.member_index
            for candidate in split_rows
            if candidate.acquisition_proxy_group == row.acquisition_proxy_group
        )
    )
    ok = (
        not duplicate_ids
        and not duplicate_hashes
        and set(assigned_ids) == source_ids
        and len(assigned_ids) == len(source_rows)
        and separation_ok
        and proxy_group_disjoint
        and all(all(counts[a][s] > 0 for s in ACTIVE_SPLITS) for a in KNOWN_ARCHIVES)
    )
    return {
        "ok": ok,
        "protocol": SPLIT_PROTOCOL,
        "source_recording_count": len(source_rows),
        "assignment_count": len(split_rows),
        "counts_by_archive": counts,
        "guard_members_per_boundary": guard_members,
        "active_split_recording_count": sum(row.split in ACTIVE_SPLITS for row in split_rows),
        "guard_recording_count": sum(row.split == "guard" for row in split_rows),
        "recording_ids_exact": set(assigned_ids) == source_ids and len(assigned_ids) == len(source_rows),
        "member_sha256_unique": not duplicate_hashes,
        "active_proxy_groups_split_disjoint": proxy_group_disjoint,
        "contiguous_blocks_separated_by_required_guards": separation_ok,
        "unknown_sources_present": False,
        "physical_session_independence_established": False,
        "interpretation_limit": (
            "Official metadata does not map MAT members to independent physical acquisition sessions. "
            "Contiguous blocks and five-member guards reduce adjacent-sample leakage but do not prove "
            "cross-session generalization."
        ),
        "split_ready": ok,
        "training_eligible": False,
        "training_blocker": (
            "The split is frozen, but IQ preprocessing, member-level aggregation and the training "
            "configuration must be frozen and bound to this split before training."
        ),
    }


def write_split_evidence(
    assignments: Iterable[KULeuvenKnownSplitAssignment],
    audit: dict[str, Any],
    manifest_paths: Iterable[str | Path],
    output_dir: str | Path,
) -> dict[str, Path]:
    rows = tuple(assignments)
    if not rows or not audit.get("ok"):
        raise KULeuvenSplitError("拒绝写入空或审计未通过的划分")
    root = Path(output_dir)
    destinations = {
        "assignments": root / "known-split-assignments.csv",
        "verification": root / "known-split-verification.json",
        "index": root / "bundle-index.json",
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite split evidence: {existing}")
    root.mkdir(parents=True, exist_ok=True)
    assignments_tmp = destinations["assignments"].with_suffix(".csv.tmp")
    verification_tmp = destinations["verification"].with_suffix(".json.tmp")
    try:
        with assignments_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        verification = dict(audit)
        verification.update(
            {
                "schema_version": "1.0",
                "artifact_type": "ku_leuven_known_source_split_verification",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_manifests": [
                    {
                        "path": Path(path).as_posix(),
                        "sha256": sha256_file(Path(path)),
                    }
                    for path in manifest_paths
                ],
            }
        )
        verification_tmp.write_text(
            json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        assignments_tmp.replace(destinations["assignments"])
        verification_tmp.replace(destinations["verification"])
    finally:
        for temporary in (assignments_tmp, verification_tmp):
            if temporary.exists():
                temporary.unlink()
    index = {
        "schema_version": "1.0",
        "artifact_type": "ku_leuven_known_split_bundle_index",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            name: {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in destinations.items()
            if name != "index"
        },
    }
    destinations["index"].write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--guard-members", type=int, default=5)
    args = parser.parse_args()
    recordings = read_known_recordings(args.manifest)
    assignments = assign_contiguous_proxy_splits(
        recordings, guard_members=args.guard_members
    )
    audit = audit_known_splits(
        recordings, assignments, guard_members=args.guard_members
    )
    outputs = write_split_evidence(
        assignments, audit, args.manifest, args.output_dir
    )
    print(
        json.dumps(
            {
                "split_ready": audit["split_ready"],
                "training_eligible": audit["training_eligible"],
                "counts_by_archive": audit["counts_by_archive"],
                "outputs": {name: str(path) for name, path in outputs.items()},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ACTIVE_SPLITS",
    "KNOWN_ARCHIVES",
    "KULeuvenKnownRecording",
    "KULeuvenKnownSplitAssignment",
    "KULeuvenSplitError",
    "SPLIT_PROTOCOL",
    "assign_contiguous_proxy_splits",
    "audit_known_splits",
    "read_known_recordings",
    "write_split_evidence",
]
