"""Fail-closed validation for the KU Leuven SJRC external-unknown protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from airwatch.data.data_provenance import sha256_file


class KULeuvenExternalProtocolError(ValueError):
    """Raised when the external-unknown protocol or its evidence has drifted."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KULeuvenExternalProtocolError(f"cannot read JSON evidence {path}: {exc}") from exc


def _resolve_verified(root: Path, item: dict[str, Any], label: str) -> Path:
    path = root / item.get("path", item.get("local_path", ""))
    if not path.is_file():
        raise KULeuvenExternalProtocolError(f"missing {label}: {path}")
    if sha256_file(path) != item.get("sha256"):
        raise KULeuvenExternalProtocolError(f"{label} SHA-256 mismatch: {path}")
    return path


def validate_external_unknown_protocol(
    protocol_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root)
    protocol_file = Path(protocol_path)
    protocol = _load_json(protocol_file)
    if protocol.get("schema_version") != "1.0":
        raise KULeuvenExternalProtocolError("unsupported protocol schema")
    dataset = protocol.get("dataset", {})
    if dataset.get("signal_type") != "RC" or dataset.get("paper_role") != "unknown_source":
        raise KULeuvenExternalProtocolError("SJRC protocol must preserve the sourced RC/unknown role")
    role = protocol.get("role_assignment", {})
    required_role = {
        "entire_archive_role": "external_unknown_test",
        "all_recordings_in_one_locked_evaluation_group": True,
        "member_level_split_allowed": False,
        "training_allowed": False,
        "validation_allowed": False,
        "threshold_selection_allowed": False,
        "model_selection_allowed": False,
    }
    for key, expected in required_role.items():
        if role.get(key) != expected:
            raise KULeuvenExternalProtocolError(f"unsafe role assignment: {key}")
    if protocol.get("status") != "blocked" or protocol.get("evaluation_eligible") is not False:
        raise KULeuvenExternalProtocolError("protocol must remain blocked until prerequisites are met")
    if protocol.get("training_eligible") is not False:
        raise KULeuvenExternalProtocolError("external unknown data cannot be training eligible")

    official = protocol.get("source_evidence", [])[0]
    official_path = _resolve_verified(root, official, "official dataset README")
    artifacts = protocol.get("artifacts", {})
    index_path = _resolve_verified(root, artifacts.get("manifest_bundle_index", {}), "manifest bundle index")
    acceptance_path = _resolve_verified(
        root, artifacts.get("window_reader_acceptance", {}), "window reader acceptance"
    )
    index = _load_json(index_path)
    bundle_root = index_path.parent
    resolved_bundle = {}
    for name, item in index.get("artifacts", {}).items():
        resolved_bundle[name] = _resolve_verified(bundle_root, item, f"bundle artifact {name}")

    recording_path = resolved_bundle.get("recording_manifest")
    window_path = resolved_bundle.get("window_plan")
    if recording_path is None or window_path is None:
        raise KULeuvenExternalProtocolError("bundle index lacks recording or window artifact")
    with recording_path.open(encoding="utf-8", newline="") as handle:
        recordings = list(csv.DictReader(handle))
    with window_path.open(encoding="utf-8", newline="") as handle:
        windows = list(csv.DictReader(handle))
    recording_ids = {row["recording_id"] for row in recordings}
    if len(recordings) != 51 or len(recording_ids) != 51:
        raise KULeuvenExternalProtocolError("expected 51 unique SJRC MAT-member records")
    if {row["split"] for row in recordings} != {"unassigned"}:
        raise KULeuvenExternalProtocolError("source manifest split must remain unassigned")
    if len(windows) != 1632 or {row["recording_id"] for row in windows} != recording_ids:
        raise KULeuvenExternalProtocolError("window plan does not exactly cover 51 recordings")
    if {row["split"] for row in windows} != {"unassigned"}:
        raise KULeuvenExternalProtocolError("source window split must remain unassigned")
    acceptance = _load_json(acceptance_path)
    if not acceptance.get("passed") or acceptance.get("training_eligible") is not False:
        raise KULeuvenExternalProtocolError("window-reader acceptance is missing or unsafe")

    return {
        "ok": True,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_file),
        "official_readme": str(official_path),
        "recording_count": len(recordings),
        "window_count": len(windows),
        "assigned_external_role": role["entire_archive_role"],
        "source_manifest_split": "unassigned",
        "training_data_used": False,
        "validation_data_used": False,
        "threshold_selection_data_used": False,
        "evaluation_run": False,
        "evaluation_eligible": False,
        "blocker": protocol["blocker"],
    }


def write_validation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite external-protocol validation evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = [
    "KULeuvenExternalProtocolError",
    "validate_external_unknown_protocol",
    "write_validation_report",
]
