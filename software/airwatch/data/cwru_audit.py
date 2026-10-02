"""Read-only manifest auditing for CWRU bearing recordings.

The auditor validates the bookkeeping around a manifest without creating
processed data, changing MAT files, training a model, or touching checkpoints.
It is intentionally separate from the dataset reader so it can be run before
any experiment consumes a manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .cwru import CWRUDataError, load_cwru_signal

_BASE_COLUMNS = {
    "dataset",
    "local_path",
    "original_filename",
    "class_name",
    "load_hp",
    "rpm",
    "sample_rate_hz",
    "sensor_key",
}
_OPTIONAL_COLUMNS = {
    "label_index",
    "fault_diameter_inch",
    "source_url",
    "downloaded_at",
    "sha256",
    "inspection_status",
    "rpm_source",
    "notes",
    "window_size",
    "step",
    "window_count",
    "split",
    "split_group",
}
_VALID_SPLITS = {"train", "validation", "test"}


class CWRUManifestAuditError(ValueError):
    """Raised when the manifest cannot be read as a CSV."""


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    row_number: int | None = None
    local_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.row_number is not None:
            result["row_number"] = self.row_number
        if self.local_path is not None:
            result["local_path"] = self.local_path
        return result


def _infer_project_root(manifest_path: Path) -> Path:
    # <project>/datasets/bearing/cwru/<manifest>.csv
    if len(manifest_path.parents) >= 4 and manifest_path.parent.name == "cwru":
        return manifest_path.parents[3]
    return manifest_path.parent


def _read_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not path.is_file():
        raise CWRUManifestAuditError(f"manifest does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = _BASE_COLUMNS - columns
            if missing:
                raise CWRUManifestAuditError(
                    f"manifest missing required columns: {sorted(missing)}"
                )
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise CWRUManifestAuditError(f"failed to read manifest {path}: {exc}") from exc
    if not rows:
        raise CWRUManifestAuditError(f"manifest contains no rows: {path}")
    return rows, columns


def _load_label_map(path: Path | None) -> dict[int, str]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {int(index): str(name) for index, name in payload["labels"].items()}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CWRUManifestAuditError(f"invalid label map: {path}: {exc}") from exc


def _resolve_path(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _int_field(row: dict[str, str], name: str) -> int | None:
    value = (row.get(name) or "").strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _float_field(row: dict[str, str], name: str) -> float | None:
    value = (row.get(name) or "").strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _reference_by_path(
    reference_manifest_path: Path | None,
    project_root: Path,
) -> dict[Path, dict[str, str]]:
    if reference_manifest_path is None:
        return {}
    rows, _ = _read_csv(reference_manifest_path)
    result: dict[Path, dict[str, str]] = {}
    for row in rows:
        local_path = (row.get("local_path") or "").strip()
        if local_path:
            result[_resolve_path(project_root, local_path)] = row
    return result


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def audit_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    label_map_path: str | Path | None = None,
    reference_manifest_path: str | Path | None = None,
    expected_sample_rate_hz: int | None = 12000,
) -> dict[str, Any]:
    """Audit one manifest and return a JSON-serialisable report.

    ``reference_manifest_path`` is useful for a derived split manifest: the
    split file can be checked against the original manifest's SHA-256 and
    source metadata without copying those fields into the split file.
    """
    manifest = Path(manifest_path).resolve()
    root = Path(project_root).resolve() if project_root else _infer_project_root(manifest)
    label_path = (
        Path(label_map_path).resolve()
        if label_map_path is not None
        else manifest.parent / "label-map.json"
    )
    labels = _load_label_map(label_path)
    rows, columns = _read_csv(manifest)
    reference = _reference_by_path(
        Path(reference_manifest_path).resolve() if reference_manifest_path else None,
        root,
    )

    issues: list[AuditIssue] = []
    if not labels:
        issues.append(
            AuditIssue(
                "warning",
                "label_map_missing",
                f"label map unavailable; class/label consistency was not checked: {label_path}",
            )
        )

    missing_recommended = {
        "source_url",
        "downloaded_at",
        "sha256",
    } - columns
    if missing_recommended and not reference:
        issues.append(
            AuditIssue(
                "warning",
                "traceability_columns_missing",
                "manifest is missing traceability columns: "
                + ", ".join(sorted(missing_recommended)),
            )
        )

    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    windows_by_split: Counter[str] = Counter()
    groups_to_splits: defaultdict[str, set[str]] = defaultdict(set)
    paths_to_splits: defaultdict[Path, set[str]] = defaultdict(set)
    seen_paths: dict[Path, int] = {}
    audited_rows: list[dict[str, Any]] = []
    total_windows = 0
    valid_rows = 0

    for row_number, row in enumerate(rows, start=2):
        local_value = (row.get("local_path") or "").strip()
        local_path = _resolve_path(root, local_value) if local_value else None
        row_issues: list[AuditIssue] = []
        def add(severity: str, code: str, message: str) -> None:
            issue = AuditIssue(severity, code, message, row_number, local_value or None)
            issues.append(issue)
            row_issues.append(issue)

        dataset = (row.get("dataset") or "").strip()
        class_name = (row.get("class_name") or "").strip()
        sensor_key = (row.get("sensor_key") or "").strip()
        split = (row.get("split") or "").strip()
        split_group = (row.get("split_group") or "").strip()
        class_counts[class_name or "<missing>"] += 1
        if split:
            split_counts[split] += 1
            if split not in _VALID_SPLITS:
                add("error", "invalid_split", f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}")
            if split_group:
                groups_to_splits[split_group].add(split)
        elif "split" in columns:
            add("error", "missing_split", "split is empty")

        if dataset != "CWRU":
            add("error", "invalid_dataset", f"dataset must be 'CWRU', got {dataset!r}")
        if not class_name:
            add("error", "missing_class_name", "class_name is empty")
        elif labels and class_name not in labels.values():
            add("error", "unknown_class_name", f"class_name {class_name!r} is absent from label map")

        label_index = _int_field(row, "label_index")
        if "label_index" in columns and label_index is None:
            add("error", "invalid_label_index", "label_index must be an integer")
        if labels and label_index is not None:
            expected_class = labels.get(label_index)
            if expected_class is None:
                add("error", "unknown_label_index", f"label_index {label_index} is absent from label map")
            elif expected_class != class_name:
                add("error", "label_mismatch", f"label_index {label_index} maps to {expected_class!r}, not {class_name!r}")
        elif labels and label_index is None:
            inverse = {name: index for index, name in labels.items()}
            label_index = inverse.get(class_name)

        if not local_value:
            add("error", "missing_local_path", "local_path is empty")
        elif local_path is not None:
            if not _inside(root, local_path):
                add("error", "path_outside_project", f"local_path resolves outside project root: {local_path}")
            if not local_path.is_file():
                add("error", "file_missing", f"file does not exist: {local_path}")
            else:
                if local_path in seen_paths:
                    add("error", "duplicate_path", f"same file already appears on row {seen_paths[local_path]}")
                else:
                    seen_paths[local_path] = row_number
                if (row.get("original_filename") or "").strip() != local_path.name:
                    add("error", "filename_mismatch", "original_filename does not match the local file name")
                paths_to_splits[local_path].add(split or "<unsplit>")

        if not sensor_key:
            add("error", "missing_sensor_key", "sensor_key is empty")
        elif "|" in sensor_key:
            add("error", "ambiguous_sensor_key", "sensor_key contains multiple channels; choose exactly one channel")
        elif not sensor_key.endswith("_DE_time"):
            add("error", "invalid_sensor_key", "sensor_key must end with '_DE_time'")

        load_hp = _int_field(row, "load_hp")
        if load_hp is None or load_hp < 0:
            add("error", "invalid_load_hp", "load_hp must be a non-negative integer")
        rpm = _int_field(row, "rpm")
        if rpm is None or rpm <= 0:
            add("error", "invalid_rpm", "rpm must be a positive integer")
        sample_rate = _int_field(row, "sample_rate_hz")
        if sample_rate is None or sample_rate <= 0:
            add("error", "invalid_sample_rate", "sample_rate_hz must be a positive integer")
        elif expected_sample_rate_hz is not None and sample_rate != expected_sample_rate_hz:
            add("error", "unexpected_sample_rate", f"sample_rate_hz must be {expected_sample_rate_hz}, got {sample_rate}")

        reference_row = reference.get(local_path) if local_path else None
        expected_hash = (row.get("sha256") or "").strip()
        if not expected_hash and reference_row:
            expected_hash = (reference_row.get("sha256") or "").strip()
        actual_hash: str | None = None
        if local_path is not None and local_path.is_file():
            try:
                actual_hash = _sha256(local_path)
            except OSError as exc:
                add("error", "hash_read_failed", f"could not hash file: {exc}")
            if expected_hash and actual_hash != expected_hash:
                add("error", "sha256_mismatch", f"manifest SHA-256 {expected_hash} != actual {actual_hash}")
            elif not expected_hash:
                add("warning", "sha256_missing", "no SHA-256 found in this manifest or its reference manifest")

        signal_length: int | None = None
        source_shape: list[int] | None = None
        candidate_keys: list[str] = []
        signal_stats: dict[str, float] = {}
        if local_path is not None and local_path.is_file() and sensor_key and "|" not in sensor_key:
            try:
                mat = loadmat(local_path)
                candidate_keys = sorted(key for key in mat if key.endswith("_DE_time"))
                if not candidate_keys:
                    add("error", "de_channel_missing", "MAT file contains no *_DE_time channel")
                elif len(candidate_keys) != 1:
                    add("warning", "multiple_de_channels", f"MAT file contains {len(candidate_keys)} DE channels; manifest explicitly selects {sensor_key}")
                record = load_cwru_signal(local_path, sensor_key=sensor_key)
                signal = record.signal
                signal_length = int(signal.size)
                source_shape = [int(dimension) for dimension in record.source_shape]
                signal_stats = {
                    "min": float(np.min(signal)),
                    "max": float(np.max(signal)),
                    "mean": float(np.mean(signal)),
                    "std": float(np.std(signal)),
                }
            except (CWRUDataError, OSError, ValueError) as exc:
                add("error", "signal_invalid", str(exc))

        window_size = _int_field(row, "window_size")
        step = _int_field(row, "step")
        declared_window_count = _int_field(row, "window_count")
        expected_window_count: int | None = None
        if any(name in columns for name in ("window_size", "step", "window_count")):
            if window_size is None or window_size <= 0:
                add("error", "invalid_window_size", "window_size must be a positive integer")
            if step is None or step <= 0:
                add("error", "invalid_step", "step must be a positive integer")
            if declared_window_count is None or declared_window_count < 0:
                add("error", "invalid_window_count", "window_count must be a non-negative integer")
            if signal_length is not None and window_size and step and window_size > 0 and step > 0:
                expected_window_count = max(0, (signal_length - window_size) // step + 1) if signal_length >= window_size else 0
                if declared_window_count is not None and declared_window_count != expected_window_count:
                    add("error", "window_count_mismatch", f"declared window_count={declared_window_count}, actual={expected_window_count}")
            if split and declared_window_count is not None and declared_window_count >= 0:
                windows_by_split[split] += declared_window_count
                total_windows += declared_window_count

        if not row_issues:
            valid_rows += 1
        audited_rows.append(
            {
                "row_number": row_number,
                "local_path": local_value,
                "resolved_path": str(local_path) if local_path else None,
                "class_name": class_name,
                "label_index": label_index,
                "split": split or None,
                "split_group": split_group or None,
                "sensor_key": sensor_key,
                "candidate_de_channels": candidate_keys,
                "signal_length": signal_length,
                "source_shape": source_shape,
                "signal_stats": signal_stats,
                "declared_window_count": declared_window_count,
                "expected_window_count": expected_window_count,
                "sha256_expected": expected_hash or None,
                "sha256_actual": actual_hash,
                "status": "error" if any(item.severity == "error" for item in row_issues) else ("warning" if row_issues else "pass"),
                "issues": [item.as_dict() for item in row_issues],
            }
        )

    for group, splits in sorted(groups_to_splits.items()):
        if len(splits) > 1:
            issues.append(AuditIssue("error", "split_group_leakage", f"split_group {group!r} appears in multiple splits: {sorted(splits)}"))
    for path, splits in sorted(paths_to_splits.items(), key=lambda item: str(item[0])):
        if len(splits) > 1:
            issues.append(AuditIssue("error", "recording_split_leakage", f"recording appears in multiple splits: {sorted(splits)}", local_path=str(path)))

    errors = [item for item in issues if item.severity == "error"]
    warnings = [item for item in issues if item.severity == "warning"]
    status = "error" if errors else ("warning" if warnings else "pass")
    return {
        "audit_name": "cwru_manifest_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest),
        "project_root": str(root),
        "label_map_path": str(label_path),
        "reference_manifest_path": str(Path(reference_manifest_path).resolve()) if reference_manifest_path else None,
        "columns": sorted(columns),
        "status": status,
        "summary": {
            "row_count": len(rows),
            "valid_rows": valid_rows,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "class_counts": dict(sorted(class_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "windows_by_split": dict(sorted(windows_by_split.items())),
            "total_window_count": total_windows,
            "unique_files": len(seen_paths),
        },
        "issues": [item.as_dict() for item in issues],
        "rows": audited_rows,
    }


def write_audit_json(report: dict[str, Any], destination: str | Path) -> Path:
    """Atomically write an audit report and return its absolute path."""
    return _atomic_write_json(report, Path(destination))


__all__ = [
    "AuditIssue",
    "CWRUManifestAuditError",
    "audit_manifest",
    "write_audit_json",
]
