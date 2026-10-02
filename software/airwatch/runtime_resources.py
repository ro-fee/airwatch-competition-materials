"""Build and verify the minimal read-only resource closure for AirWatch.

This module is deliberately free of Qt, NumPy and PyTorch.  PyInstaller specs,
release validation and tests all use the same manifest so packaging cannot
silently drift away from the frozen runtime contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from airwatch.general_recognition_contract import TASKS
from airwatch import runtime_paths


MANIFEST_SCHEMA_VERSION = 1
RUNTIME_MANIFEST_NAME = "runtime-resources.json"

BEARING_RUNTIME_CONTRACT = (
    "artifacts/evidence/bearing/anti_noise/"
    "bearing_cnn_anti_noise_20260904_frozen_contract.json"
)
KU_LEUVEN_DEVELOPMENT_CONTRACT = (
    "artifacts/evidence/uav/ku_leuven/local_development/"
    "ku_leuven_tcn_strong_noise_seed20260910_v2_runtime_contract.json"
)
KU_LEUVEN_SOFTWARE_CONTRACT = (
    "artifacts/evidence/uav/ku_leuven/local_development/"
    "ku_leuven_tcn_weighted_worst_noise_seed20260910_v4_runtime_contract.json"
)

GENERAL_TASK_CHECKPOINTS: dict[str, str] = {
    task: spec.checkpoint for task, spec in TASKS.items()
}

GENERAL_TASK_SLOT_IDS: dict[str, str] = {
    "信号个体识别": "historical.individual",
    "信号调制识别": "historical.modulation",
    "信号通联识别": "historical.communication",
    "信号业务识别": "historical.service",
    "信号编码识别": "historical.coding",
}
HISTORICAL_GENERATOR_SLOT_ID = "historical.generator"
HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID = "historical.individual_reference"

_BEARING_RUNTIME_ROLES = frozenset(
    {
        "checkpoint",
        "label_map",
        "quality_calibration",
        "model_definition",
        "quality_gate_implementation",
        "preprocessing_implementation",
        "inference_implementation",
        "dataset_manifest",
    }
)
_KU_LEUVEN_RUNTIME_ROLES = frozenset(
    {
        "checkpoint",
        "resolved_config",
        "model_definition",
        "preprocessing_implementation",
        "inference_implementation",
    }
)


class RuntimeResourceError(ValueError):
    """Raised when a runtime resource closure is missing or inconsistent."""


@dataclass(frozen=True)
class ModelSlot:
    """One truthful model position exposed to packaging and future UI work."""

    slot_id: str
    display_name: str
    status: str
    release_tier: str
    resource: str | None = None
    limitation: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "reserved"}:
            raise RuntimeResourceError(f"unsupported model slot status: {self.status}")
        if self.status == "available" and not self.resource:
            raise RuntimeResourceError(f"available model slot has no resource: {self.slot_id}")
        if self.status == "reserved" and self.resource is not None:
            raise RuntimeResourceError(f"reserved model slot must not ship a resource: {self.slot_id}")


MODEL_SLOTS: tuple[ModelSlot, ...] = (
    *(
        ModelSlot(
            slot_id=GENERAL_TASK_SLOT_IDS[task],
            display_name=task,
            status="available",
            release_tier="historical_demo",
            resource=checkpoint,
            limitation="历史原型任务，不代表无人机竞赛性能。",
        )
        for task, checkpoint in GENERAL_TASK_CHECKPOINTS.items()
    ),
    ModelSlot(
        slot_id=HISTORICAL_GENERATOR_SLOT_ID,
        display_name="历史条件 GAN 信号生成",
        status="available",
        release_tier="historical_demo",
        resource="models/999G_plus.ckpt",
        limitation="仅用于历史合成信号演示，不代表真实无人机信号分布。",
    ),
    ModelSlot(
        slot_id=HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID,
        display_name="个体识别 TCN 原始对照模型",
        status="available",
        release_tier="historical_demo",
        resource="models/newdata_TCN1122.pkl",
        limitation="仅作为历史轻量化页面的对照模型，不是竞赛端侧学生模型。",
    ),
    ModelSlot(
        slot_id="bearing.diagnosis",
        display_name="轴承抗噪声诊断",
        status="available",
        release_tier="frozen",
        resource=BEARING_RUNTIME_CONTRACT,
        limitation="独立技术验证，不是无人机频谱识别结果。",
    ),
    ModelSlot(
        slot_id="uav.known_source",
        display_name="无人机已知源识别",
        status="available",
        release_tier="development_frozen",
        resource=KU_LEUVEN_SOFTWARE_CONTRACT,
        limitation="仅开发版已知源分类，不提供未知拒识结论。",
    ),
    ModelSlot(
        slot_id="uav.multitask",
        display_name="无人机多任务识别",
        status="reserved",
        release_tier="not_available",
        limitation="等待后续冻结模型、标签和评估证据接入。",
    ),
    ModelSlot(
        slot_id="uav.open_set",
        display_name="无人机未知信号预警",
        status="reserved",
        release_tier="not_available",
        limitation="等待后续冻结验证阈值和独立测试证据。",
    ),
    ModelSlot(
        slot_id="edge.student",
        display_name="端侧轻量化学生模型",
        status="reserved",
        release_tier="not_available",
        limitation="等待后续模型大小、时延和精度证据接入。",
    ),
)


def model_slot(slot_id: str) -> ModelSlot:
    """Return one registered model slot without implying reserved capability."""

    for slot in MODEL_SLOTS:
        if slot.slot_id == slot_id:
            return slot
    raise KeyError(f"unknown model slot: {slot_id}")


def general_task_slot_id(task_name: str) -> str:
    """Return the stable model-slot id for one historical recognition task."""

    try:
        return GENERAL_TASK_SLOT_IDS[task_name]
    except KeyError as exc:
        raise KeyError(f"unknown historical recognition task: {task_name}") from exc


def model_slot_resource_path(slot_id: str) -> Path:
    """Resolve and verify one available model slot in source or packaged mode."""

    slot = model_slot(slot_id)
    if slot.status != "available" or not slot.resource:
        raise RuntimeResourceError(
            f"model slot is not available: {slot.display_name} ({slot.slot_id})"
        )
    path = runtime_paths.resource_path(slot.resource)
    if not path.is_file():
        raise RuntimeResourceError(
            f"model slot resource is missing: {slot.display_name} ({slot.resource})"
        )
    return path


@dataclass(frozen=True)
class RuntimeResource:
    path: str
    size_bytes: int
    sha256: str
    source: str


def _canonical_relative(value: str | Path) -> str:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeResourceError(f"runtime resource path must be project-relative: {value}")
    return relative.as_posix()


def _resolve(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    candidate = (root / _canonical_relative(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeResourceError(f"runtime resource escapes project root: {relative}") from exc
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resource(root: Path, relative: str, source: str) -> RuntimeResource:
    path = _resolve(root, relative)
    if not path.is_file():
        raise RuntimeResourceError(f"runtime resource is missing: {relative}")
    return RuntimeResource(
        path=_canonical_relative(relative),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        source=source,
    )


def _contract_resources(
    root: Path,
    contract_relative: str,
    roles: frozenset[str],
    source: str,
) -> list[RuntimeResource]:
    contract_path = _resolve(root, contract_relative)
    if not contract_path.is_file():
        raise RuntimeResourceError(f"runtime contract is missing: {contract_relative}")
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        records = payload["integrity"]["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeResourceError(f"invalid runtime contract: {contract_relative}") from exc
    if not isinstance(records, list):
        raise RuntimeResourceError(f"runtime contract has no integrity file list: {contract_relative}")

    selected: dict[str, Mapping[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise RuntimeResourceError(f"invalid integrity record in: {contract_relative}")
        role = raw.get("role")
        if role in roles:
            if role in selected:
                raise RuntimeResourceError(f"duplicate runtime role {role}: {contract_relative}")
            selected[str(role)] = raw
    missing = sorted(roles - set(selected))
    if missing:
        raise RuntimeResourceError(
            f"runtime contract is missing roles {', '.join(missing)}: {contract_relative}"
        )

    resources = [_resource(root, contract_relative, f"{source}:contract")]
    for role in sorted(roles):
        record = selected[role]
        relative = _canonical_relative(str(record.get("path", "")))
        resource = _resource(root, relative, f"{source}:{role}")
        if resource.size_bytes != record.get("size_bytes"):
            raise RuntimeResourceError(f"contract size mismatch for {role}: {relative}")
        expected_hash = str(record.get("sha256", "")).upper()
        if resource.sha256 != expected_hash:
            raise RuntimeResourceError(f"contract hash mismatch for {role}: {relative}")
        resources.append(resource)
    return resources


def _tree_resources(root: Path, relative_root: str, source: str) -> Iterable[RuntimeResource]:
    directory = _resolve(root, relative_root)
    if not directory.is_dir():
        raise RuntimeResourceError(f"runtime resource directory is missing: {relative_root}")
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        yield _resource(root, path.relative_to(root).as_posix(), source)


def build_runtime_resource_manifest(project_root: str | Path) -> dict[str, Any]:
    """Return the verified minimal runtime resource manifest for one source tree."""

    root = Path(project_root).resolve()
    resources: dict[str, RuntimeResource] = {}

    def add(items: Iterable[RuntimeResource]) -> None:
        for item in items:
            previous = resources.get(item.path)
            if previous and (previous.size_bytes, previous.sha256) != (
                item.size_bytes,
                item.sha256,
            ):
                raise RuntimeResourceError(f"conflicting resource registration: {item.path}")
            resources.setdefault(item.path, item)

    add(_tree_resources(root, "assets", "static:assets"))
    add(_tree_resources(root, "data", "static:demo_data"))
    contract_slot_ids = {"bearing.diagnosis", "uav.known_source"}
    add(
        _resource(root, str(slot.resource), f"model:{slot.slot_id}")
        for slot in MODEL_SLOTS
        if slot.status == "available" and slot.slot_id not in contract_slot_ids
    )
    add(
        _contract_resources(
            root,
            BEARING_RUNTIME_CONTRACT,
            _BEARING_RUNTIME_ROLES,
            "model:bearing.diagnosis",
        )
    )
    add(
        _contract_resources(
            root,
            KU_LEUVEN_SOFTWARE_CONTRACT,
            _KU_LEUVEN_RUNTIME_ROLES,
            "model:uav.known_source",
        )
    )

    ordered = [resources[path] for path in sorted(resources)]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "resources": [asdict(item) for item in ordered],
        "model_slots": [asdict(slot) for slot in MODEL_SLOTS],
        "summary": {
            "resource_files": len(ordered),
            "total_bytes": sum(item.size_bytes for item in ordered),
            "available_model_slots": sum(slot.status == "available" for slot in MODEL_SLOTS),
            "reserved_model_slots": sum(slot.status == "reserved" for slot in MODEL_SLOTS),
        },
    }


def write_runtime_resource_manifest(
    project_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build and atomically write a runtime resource manifest."""

    manifest = build_runtime_resource_manifest(project_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    try:
        if output.is_file() and output.read_text(encoding="utf-8") == serialized:
            return manifest
    except OSError:
        # Fall through to the existing atomic replacement path; the write will
        # surface a useful filesystem error if the destination is unusable.
        pass
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    return manifest


def pyinstaller_datas(
    project_root: str | Path,
    manifest_path: str | Path,
) -> list[tuple[str, str]]:
    """Return PyInstaller ``datas`` entries driven by the verified manifest."""

    root = Path(project_root).resolve()
    manifest = write_runtime_resource_manifest(root, manifest_path)
    datas = [
        (str(_resolve(root, item["path"])), str(Path(item["path"]).parent))
        for item in manifest["resources"]
    ]
    datas.append((str(Path(manifest_path).resolve()), "."))
    return datas


def validate_runtime_resource_manifest(
    staged_root: str | Path,
    manifest: Mapping[str, Any] | str | Path,
) -> dict[str, int]:
    """Validate every declared file in a staged source or packaged resource root."""

    root = Path(staged_root).resolve()
    if isinstance(manifest, (str, Path)):
        try:
            payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeResourceError(f"cannot read runtime resource manifest: {manifest}") from exc
    else:
        payload = dict(manifest)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeResourceError("unsupported runtime resource manifest schema")
    raw_resources = payload.get("resources")
    if not isinstance(raw_resources, list) or not raw_resources:
        raise RuntimeResourceError("runtime resource manifest is empty")

    seen: set[str] = set()
    total_bytes = 0
    for raw in raw_resources:
        if not isinstance(raw, Mapping):
            raise RuntimeResourceError("runtime resource entry must be an object")
        relative = _canonical_relative(str(raw.get("path", "")))
        if relative in seen:
            raise RuntimeResourceError(f"duplicate runtime resource entry: {relative}")
        seen.add(relative)
        resource = _resolve(root, relative)
        if not resource.is_file():
            raise RuntimeResourceError(f"staged runtime resource is missing: {relative}")
        expected_size = raw.get("size_bytes")
        expected_hash = str(raw.get("sha256", "")).upper()
        if resource.stat().st_size != expected_size:
            raise RuntimeResourceError(f"staged runtime resource size mismatch: {relative}")
        if _sha256(resource) != expected_hash:
            raise RuntimeResourceError(f"staged runtime resource hash mismatch: {relative}")
        total_bytes += resource.stat().st_size
    return {"resource_files": len(seen), "total_bytes": total_bytes}


__all__ = [
    "BEARING_RUNTIME_CONTRACT",
    "GENERAL_TASK_CHECKPOINTS",
    "GENERAL_TASK_SLOT_IDS",
    "HISTORICAL_GENERATOR_SLOT_ID",
    "HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID",
    "KU_LEUVEN_DEVELOPMENT_CONTRACT",
    "KU_LEUVEN_SOFTWARE_CONTRACT",
    "MANIFEST_SCHEMA_VERSION",
    "MODEL_SLOTS",
    "ModelSlot",
    "RUNTIME_MANIFEST_NAME",
    "RuntimeResourceError",
    "build_runtime_resource_manifest",
    "general_task_slot_id",
    "model_slot",
    "model_slot_resource_path",
    "pyinstaller_datas",
    "validate_runtime_resource_manifest",
    "write_runtime_resource_manifest",
]
