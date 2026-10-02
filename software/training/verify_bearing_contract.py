"""Verify the frozen anti-noise bearing model contract.

This command is intentionally read-only. It checks the model checkpoint,
label mapping, preprocessing configuration, quality-gate calibration artifact,
the implementation files that enforce the gate, and the referenced CWRU
inputs. A changed file is a hard failure; callers should create a new
contract version instead of editing a frozen one in place.

Run from the project root with::

    python -m training.verify_bearing_contract
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from airwatch.models import BearingCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "artifacts/evidence/bearing/anti_noise/"
    / "bearing_cnn_anti_noise_20260904_frozen_contract.json"
)


class BearingContractError(ValueError):
    """Raised when the frozen bearing contract is invalid or has drifted."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Return the stable byte representation used for the contract digest."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve_project_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BearingContractError(f"invalid project-relative path: {value!r}")
    path = Path(value)
    if path.is_absolute():
        raise BearingContractError(f"contract paths must be relative: {value!r}")
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise BearingContractError(
            f"contract path escapes project root: {value!r}"
        ) from exc
    return resolved


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BearingContractError(f"{name} must be a JSON object")
    return value


def _require_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise BearingContractError(
            f"{name} mismatch: expected {expected!r}, got {actual!r}"
        )


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BearingContractError(f"cannot read {name}: {path}") from exc
    return _require_mapping(payload, name)


def _verify_contract_digest(contract: Mapping[str, Any]) -> None:
    digest = contract.get("contract_digest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise BearingContractError("contract_digest_sha256 is missing or malformed")
    without_digest = dict(contract)
    without_digest.pop("contract_digest_sha256", None)
    actual = hashlib.sha256(_canonical_json(without_digest)).hexdigest().upper()
    _require_equal(actual, digest.upper(), "contract digest")


def _verify_file_records(contract: Mapping[str, Any]) -> int:
    integrity = _require_mapping(contract.get("integrity"), "integrity")
    records = integrity.get("files")
    if not isinstance(records, list) or not records:
        raise BearingContractError("integrity.files must be a non-empty list")

    seen_paths: set[str] = set()
    for index, record_value in enumerate(records):
        record = _require_mapping(record_value, f"integrity.files[{index}]")
        relative_path = record.get("path")
        if not isinstance(relative_path, str):
            raise BearingContractError(
                f"integrity.files[{index}].path must be a string"
            )
        if relative_path in seen_paths:
            raise BearingContractError(f"duplicate integrity path: {relative_path}")
        seen_paths.add(relative_path)
        path = _resolve_project_path(relative_path)
        if not path.is_file():
            raise BearingContractError(f"frozen file is missing: {path}")
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise BearingContractError(
                f"integrity.files[{index}].sha256 is malformed"
            )
        _require_equal(
            _sha256(path), expected_hash.upper(), f"SHA-256 for {relative_path}"
        )
        expected_size = record.get("size_bytes")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise BearingContractError(
                f"integrity.files[{index}].size_bytes is malformed"
            )
        _require_equal(path.stat().st_size, expected_size, f"size for {relative_path}")
    return len(records)


def _file_record(contract: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    records = contract["integrity"]["files"]
    for record in records:
        if isinstance(record, Mapping) and record.get("role") == role:
            return record
    raise BearingContractError(f"integrity record not found for role={role!r}")


def _verify_label_map(contract: Mapping[str, Any]) -> None:
    input_contract = _require_mapping(contract.get("input_contract"), "input_contract")
    label_contract = _require_mapping(
        input_contract.get("label_map"), "input_contract.label_map"
    )
    path = _resolve_project_path(label_contract.get("path"))
    payload = _load_json(path, "label map")
    expected_order = label_contract.get("label_order")
    expected_labels = label_contract.get("labels")
    _require_equal(payload.get("dataset"), "CWRU", "label_map.dataset")
    _require_equal(payload.get("label_order"), expected_order, "label_map.label_order")
    _require_equal(payload.get("labels"), expected_labels, "label_map.labels")
    record = _file_record(contract, "label_map")
    _require_equal(record.get("path"), label_contract.get("path"), "label-map record path")
    _require_equal(record.get("sha256"), label_contract.get("sha256"), "label-map SHA-256")
    _require_equal(record.get("size_bytes"), path.stat().st_size, "label-map size")


def _verify_model_record_links(contract: Mapping[str, Any]) -> None:
    """Ensure the model section points at the registered checkpoint record."""

    model_contract = _require_mapping(contract.get("model"), "model")
    checkpoint_contract = _require_mapping(
        model_contract.get("checkpoint"), "model.checkpoint"
    )
    record = _file_record(contract, "checkpoint")
    for field in ("path", "sha256", "size_bytes"):
        _require_equal(
            checkpoint_contract.get(field),
            record.get(field),
            f"model.checkpoint.{field}",
        )


def _verify_historical_boundary(contract: Mapping[str, Any]) -> None:
    """Confirm the historical checkpoint named in the boundary was untouched."""

    boundary = _require_mapping(
        contract.get("scope_boundary"), "scope_boundary"
    )
    historical = _require_mapping(
        boundary.get("historical_checkpoint"),
        "scope_boundary.historical_checkpoint",
    )
    path = _resolve_project_path(historical.get("path"))
    if not path.is_file():
        raise BearingContractError(f"historical checkpoint is missing: {path}")
    expected_hash = historical.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise BearingContractError("historical checkpoint SHA-256 is malformed")
    _require_equal(
        _sha256(path), expected_hash.upper(),
        f"historical checkpoint SHA-256 for {historical['path']}",
    )


def _verify_checkpoint(contract: Mapping[str, Any]) -> None:
    model_contract = _require_mapping(contract.get("model"), "model")
    checkpoint_contract = _require_mapping(
        model_contract.get("checkpoint"), "model.checkpoint"
    )
    checkpoint_path = _resolve_project_path(checkpoint_contract.get("path"))
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BearingContractError(
            f"cannot load frozen checkpoint: {checkpoint_path}"
        ) from exc
    checkpoint = _require_mapping(payload, "checkpoint")
    _require_equal(
        checkpoint.get("model_name"), model_contract.get("name"),
        "checkpoint.model_name"
    )

    embedded_config = _require_mapping(checkpoint.get("config"), "checkpoint.config")
    dataset = _require_mapping(
        embedded_config.get("dataset"), "checkpoint.config.dataset"
    )
    input_contract = _require_mapping(contract.get("input_contract"), "input_contract")
    label_contract = _require_mapping(input_contract["label_map"], "input_contract.label_map")
    normalization = _require_mapping(
        input_contract.get("normalization"), "input_contract.normalization"
    )
    _require_equal(dataset.get("window_size"), input_contract.get("window_size"), "checkpoint window_size")
    _require_equal(dataset.get("step"), input_contract.get("step"), "checkpoint step")
    _require_equal(dataset.get("normalization"), normalization.get("name"), "checkpoint normalization")
    _require_equal(dataset.get("num_classes"), len(label_contract["labels"]), "checkpoint num_classes")
    _require_equal(dataset.get("label_map"), label_contract.get("path"), "checkpoint label-map path")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise BearingContractError("frozen checkpoint has no model_state_dict")
    model = BearingCNN(num_classes=int(dataset["num_classes"]), in_channels=1)
    expected_keys = set(model.state_dict())
    if set(state_dict) != expected_keys:
        raise BearingContractError("frozen checkpoint state_dict keys do not match BearingCNN")
    parameter_count = sum(int(value.numel()) for value in model.parameters())
    _require_equal(
        parameter_count,
        model_contract.get("parameter_count"),
        "checkpoint parameter_count",
    )
    _require_equal(
        int(dataset.get("num_classes")),
        int(model_contract.get("num_classes")),
        "model num_classes",
    )
    _require_equal(
        int(model_contract.get("input_channels")),
        1,
        "model input_channels",
    )


def _verify_training_config(contract: Mapping[str, Any]) -> None:
    config_record = _file_record(contract, "training_config")
    config = _load_json(_resolve_project_path(config_record["path"]), "training config")
    input_contract = _require_mapping(contract.get("input_contract"), "input_contract")
    dataset = _require_mapping(config.get("dataset"), "training config.dataset")
    training = _require_mapping(config.get("training"), "training config.training")
    normalization = _require_mapping(
        input_contract.get("normalization"), "input_contract.normalization"
    )
    label_contract = _require_mapping(input_contract["label_map"], "input_contract.label_map")
    _require_equal(dataset.get("window_size"), input_contract.get("window_size"), "training config window_size")
    _require_equal(dataset.get("step"), input_contract.get("step"), "training config step")
    _require_equal(dataset.get("normalization"), normalization.get("name"), "training config normalization")
    _require_equal(dataset.get("label_map"), label_contract.get("path"), "training config label-map path")
    _require_equal(training.get("augmentation"), contract.get("training_recipe", {}).get("augmentation"), "training augmentation recipe")
    _require_equal(training.get("seed"), contract.get("training_recipe", {}).get("seed"), "training seed")


def _verify_quality_gate(contract: Mapping[str, Any]) -> None:
    quality_contract = _require_mapping(contract.get("quality_gate"), "quality_gate")
    calibration = _require_mapping(
        quality_contract.get("calibration"), "quality_gate.calibration"
    )
    calibration_path = _resolve_project_path(calibration.get("path"))
    payload = _load_json(calibration_path, "quality calibration artifact")
    _require_equal(payload.get("calibration_split"), "validation", "quality calibration split")
    _require_equal(payload.get("selected_thresholds"), quality_contract.get("thresholds"), "quality thresholds")
    _require_equal(calibration.get("test_used_for_threshold_selection"), False, "quality calibration test-use declaration")
    _require_equal(quality_contract.get("normalization_before_gate"), "none", "quality gate input normalization")
    _require_equal(quality_contract.get("normalization_after_gate"), "window_zscore", "quality gate post-entry normalization")
    states = _require_mapping(quality_contract.get("states"), "quality_gate.states")
    for state in ("accepted", "caution", "rejected"):
        _require_mapping(states.get(state), f"quality_gate.states.{state}")
    _require_equal(states["rejected"].get("model_entry"), False, "rejected model-entry rule")
    _require_equal(states["rejected"].get("prediction"), None, "rejected prediction rule")
    _require_equal(states["rejected"].get("confidence"), None, "rejected confidence rule")


def verify_contract(contract_path: str | Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Verify one frozen contract without writing or modifying any file."""
    path = Path(contract_path).resolve()
    if not path.is_file():
        raise BearingContractError(f"contract does not exist: {path}")
    contract = _load_json(path, "frozen bearing contract")
    _require_equal(contract.get("schema_version"), 1, "contract schema_version")
    _require_equal(contract.get("status"), "frozen", "contract status")
    _verify_contract_digest(contract)
    checked_files = _verify_file_records(contract)
    _verify_historical_boundary(contract)
    _verify_model_record_links(contract)
    _verify_label_map(contract)
    _verify_training_config(contract)
    _verify_checkpoint(contract)
    _verify_quality_gate(contract)
    return {
        "passed": True,
        "contract_id": contract.get("contract_id"),
        "checked_files": checked_files,
        "checkpoint": contract["model"]["checkpoint"]["path"],
        "label_map": contract["input_contract"]["label_map"]["path"],
        "normalization": contract["input_contract"]["normalization"]["name"],
        "quality_gate_version": contract["quality_gate"].get("version"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    try:
        result = verify_contract(args.contract)
    except (BearingContractError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
