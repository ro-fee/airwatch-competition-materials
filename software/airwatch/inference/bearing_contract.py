"""Runtime loader for the frozen anti-noise bearing model contract.

The desktop application must not import the training package merely to find a
checkpoint.  This module resolves only the artifacts needed at runtime and
verifies the frozen contract digest plus the registered runtime files.  It is
read-only and never rewrites the contract, model, labels, or calibration data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from airwatch.runtime_paths import resource_root, resource_path


# Keep the contract lookup independent of the current working directory.
# ``resource_root`` points at the source tree during development and at
# PyInstaller's read-only resource directory in a packaged build.
PROJECT_ROOT = resource_root()
DEFAULT_FROZEN_BEARING_CONTRACT = resource_path(
    "artifacts",
    "evidence",
    "bearing",
    "anti_noise",
    "bearing_cnn_anti_noise_20260904_frozen_contract.json",
)


class BearingRuntimeContractError(ValueError):
    """Raised when a frozen contract is unusable by desktop inference."""


@dataclass(frozen=True)
class FrozenBearingRuntimeContract:
    """Resolved, verified paths and metadata used by desktop inference."""

    contract_path: Path
    project_root: Path
    contract_id: str
    checkpoint_path: Path
    label_map_path: Path
    quality_calibration_path: Path
    manifest_path: Path
    quality_gate_version: str
    normalization: str
    window_size: int
    step: int
    label_order: tuple[str, ...]
    quality_messages: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "contract_path",
            "project_root",
            "checkpoint_path",
            "label_map_path",
            "quality_calibration_path",
            "manifest_path",
        ):
            payload[name] = str(payload[name])
        payload["label_order"] = list(self.label_order)
        payload["quality_messages"] = dict(self.quality_messages)
        return payload


_RUNTIME_INTEGRITY_ROLES = {
    "checkpoint",
    "label_map",
    "quality_calibration",
    "model_definition",
    "quality_gate_implementation",
    "preprocessing_implementation",
    "inference_implementation",
    "dataset_manifest",
}


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BearingRuntimeContractError(f"{name} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _resolve_project_path(project_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BearingRuntimeContractError(f"invalid contract path: {value!r}")
    relative = Path(value)
    if relative.is_absolute():
        raise BearingRuntimeContractError(
            f"frozen contract paths must be project-relative: {value!r}"
        )
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BearingRuntimeContractError(
            f"contract path escapes the project directory: {value!r}"
        ) from exc
    return resolved


def _read_contract(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BearingRuntimeContractError(f"cannot read frozen contract: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BearingRuntimeContractError(
            f"frozen contract is not valid JSON: {path}"
        ) from exc
    return _mapping(payload, "frozen contract")


def _verify_contract_digest(contract: Mapping[str, Any]) -> None:
    expected = contract.get("contract_digest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise BearingRuntimeContractError("contract digest is missing or malformed")
    body = dict(contract)
    body.pop("contract_digest_sha256", None)
    actual = hashlib.sha256(_canonical_json(body)).hexdigest().upper()
    if actual != expected.upper():
        raise BearingRuntimeContractError("frozen contract digest does not match")


def _integrity_records(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    integrity = _mapping(contract.get("integrity"), "integrity")
    raw_records = integrity.get("files")
    if not isinstance(raw_records, list):
        raise BearingRuntimeContractError("integrity.files must be a list")
    records: dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, f"integrity.files[{index}]")
        role = record.get("role")
        if isinstance(role, str) and role in _RUNTIME_INTEGRITY_ROLES:
            if role in records:
                raise BearingRuntimeContractError(
                    f"duplicate runtime integrity role: {role}"
                )
            records[role] = record
    missing = sorted(_RUNTIME_INTEGRITY_ROLES - set(records))
    if missing:
        raise BearingRuntimeContractError(
            "contract is missing runtime integrity records: " + ", ".join(missing)
        )
    return records


def _verify_runtime_files(
    contract: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for role, record in _integrity_records(contract).items():
        path = _resolve_project_path(project_root, record.get("path"))
        if not path.is_file():
            raise BearingRuntimeContractError(f"frozen runtime file is missing: {path}")
        expected_size = record.get("size_bytes")
        expected_hash = record.get("sha256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise BearingRuntimeContractError(f"invalid size for runtime role {role}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise BearingRuntimeContractError(f"invalid SHA-256 for runtime role {role}")
        if path.stat().st_size != expected_size:
            raise BearingRuntimeContractError(
                f"frozen runtime file size changed for role {role}: {path}"
            )
        if _sha256(path) != expected_hash.upper():
            raise BearingRuntimeContractError(
                f"frozen runtime file SHA-256 changed for role {role}: {path}"
            )
        resolved[role] = path
    return resolved


def load_frozen_bearing_runtime_contract(
    contract_path: str | Path = DEFAULT_FROZEN_BEARING_CONTRACT,
    *,
    project_root: str | Path | None = None,
) -> FrozenBearingRuntimeContract:
    """Load and verify the runtime portion of one frozen bearing contract."""

    path = Path(contract_path).resolve()
    root = resource_root() if project_root is None else Path(project_root).resolve()
    if not path.is_file():
        raise BearingRuntimeContractError(f"frozen contract does not exist: {path}")
    contract = _read_contract(path)
    if contract.get("schema_version") != 1:
        raise BearingRuntimeContractError("unsupported frozen contract schema")
    if contract.get("status") != "frozen":
        raise BearingRuntimeContractError("bearing contract is not frozen")
    _verify_contract_digest(contract)
    runtime_files = _verify_runtime_files(contract, root)

    model = _mapping(contract.get("model"), "model")
    checkpoint = _mapping(model.get("checkpoint"), "model.checkpoint")
    input_contract = _mapping(contract.get("input_contract"), "input_contract")
    label_map = _mapping(input_contract.get("label_map"), "input_contract.label_map")
    normalization = _mapping(
        input_contract.get("normalization"), "input_contract.normalization"
    )
    quality_gate = _mapping(contract.get("quality_gate"), "quality_gate")
    calibration = _mapping(
        quality_gate.get("calibration"), "quality_gate.calibration"
    )
    states = _mapping(quality_gate.get("states"), "quality_gate.states")

    links = {
        "checkpoint": checkpoint.get("path"),
        "label_map": label_map.get("path"),
        "quality_calibration": calibration.get("path"),
        "dataset_manifest": _mapping(
            contract.get("data_contract"), "data_contract"
        ).get("manifest_path"),
    }
    for role, relative_path in links.items():
        expected = _resolve_project_path(root, relative_path)
        if expected != runtime_files[role]:
            raise BearingRuntimeContractError(
                f"contract section and integrity record disagree for role {role}"
            )

    if quality_gate.get("version") != "V2":
        raise BearingRuntimeContractError("desktop bearing inference requires quality gate V2")
    if normalization.get("name") != "window_zscore":
        raise BearingRuntimeContractError(
            "desktop bearing inference requires frozen window_zscore preprocessing"
        )
    if quality_gate.get("normalization_before_gate") != "none":
        raise BearingRuntimeContractError("quality gate must run on the raw window")
    if quality_gate.get("normalization_after_gate") != "window_zscore":
        raise BearingRuntimeContractError(
            "accepted windows must use window_zscore before model inference"
        )

    try:
        window_size = int(input_contract["window_size"])
        step = int(input_contract["step"])
        label_order = tuple(str(value) for value in label_map["label_order"])
        quality_messages = {
            state: str(_mapping(states[state], f"quality_gate.states.{state}")["user_message"])
            for state in ("accepted", "caution", "rejected")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise BearingRuntimeContractError(
            "frozen contract is missing required runtime metadata"
        ) from exc
    if window_size <= 0 or step <= 0 or not label_order:
        raise BearingRuntimeContractError("frozen runtime dimensions or labels are invalid")

    return FrozenBearingRuntimeContract(
        contract_path=path,
        project_root=root,
        contract_id=str(contract.get("contract_id")),
        checkpoint_path=runtime_files["checkpoint"],
        label_map_path=runtime_files["label_map"],
        quality_calibration_path=runtime_files["quality_calibration"],
        manifest_path=runtime_files["dataset_manifest"],
        quality_gate_version="V2",
        normalization="window_zscore",
        window_size=window_size,
        step=step,
        label_order=label_order,
        quality_messages=quality_messages,
    )


__all__ = [
    "BearingRuntimeContractError",
    "DEFAULT_FROZEN_BEARING_CONTRACT",
    "FrozenBearingRuntimeContract",
    "load_frozen_bearing_runtime_contract",
]
