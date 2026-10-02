"""Verified software-demo input for the historical two-model comparison.

Schema v1 intentionally refuses performance evidence.  It proves file identity
and preprocessing compatibility, but the bundled synthetic labels remain
navigation metadata rather than ground truth for accuracy claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from airwatch.data.recognition_input import (
    GeneralRecognitionInput,
    RecognitionSnapshot,
    load_recognition_files,
)
from airwatch.general_recognition_contract import task_spec
from airwatch.runtime_paths import resource_path


SCHEMA_VERSION = 1
TASK_NAME = "信号个体识别"
DEFAULT_MANIFEST_RESOURCE = (
    "data/individual/software_demo_comparison_manifest.json"
)
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_RECORDINGS = 256
MAX_COMPARISON_WINDOWS = 4096
PERFORMANCE_STATUS = "not_eligible_software_demo"
PERFORMANCE_LIMITATION = (
    "输入清单只证明软件演示文件身份与格式兼容；不存在经审计真值测试集，"
    "因此不发布 OA、宏平均精度、F1 或端侧时延。"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def default_historical_comparison_manifest() -> Path:
    return resource_path(DEFAULT_MANIFEST_RESOURCE)


@dataclass(frozen=True)
class PreparedHistoricalComparison:
    """Immutable, checksum-verified input set with an explicit evidence gate."""

    dataset_id: str
    purpose: str
    manifest_path: Path
    manifest_sha256: str
    inputs: GeneralRecognitionInput
    recording_ids: tuple[str, ...]
    declared_demo_labels: tuple[str, ...]
    acquisition_groups: tuple[str, ...]
    windows_per_recording: tuple[int, ...]
    performance_evidence_eligible: bool = False
    performance_status: str = PERFORMANCE_STATUS
    limitation: str = PERFORMANCE_LIMITATION

    def __post_init__(self) -> None:
        count = self.inputs.file_count
        for values in (
            self.recording_ids,
            self.declared_demo_labels,
            self.acquisition_groups,
            self.windows_per_recording,
        ):
            if len(values) != count:
                raise ValueError("历史比较清单字段数量与文件数不一致。")
        if self.performance_evidence_eligible:
            raise ValueError("历史比较清单 schema v1 不允许发布性能证据。")
        if sum(self.windows_per_recording) > MAX_COMPARISON_WINDOWS:
            raise ValueError(
                f"历史比较窗口总数超过 {MAX_COMPARISON_WINDOWS} 个安全上限。"
            )

    @property
    def recording_count(self) -> int:
        return self.inputs.file_count

    @property
    def total_windows(self) -> int:
        return sum(self.windows_per_recording)

    def snapshot(self) -> RecognitionSnapshot:
        """Return the shared model input without promoting demo labels to truth."""

        spec = task_spec(TASK_NAME)
        data: object = (
            self.inputs.arrays[0]
            if self.inputs.file_count == 1
            else self.inputs.arrays
        )
        return RecognitionSnapshot(
            task_name=TASK_NAME,
            filenames=self.inputs.filenames,
            data=data,
            labels=(),
            window_size=spec.window_size,
            repeat_count=1,
        )


def prepare_historical_comparison(
    manifest_path: str | Path,
) -> PreparedHistoricalComparison:
    """Verify one schema-v1 manifest and decode its bounded recordings."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"历史比较清单不存在：{manifest}")
    if manifest.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("历史比较清单超过 1 MiB 上限。")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取历史比较清单：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("历史比较清单根节点必须是 JSON 对象。")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"仅支持历史比较清单 schema_version={SCHEMA_VERSION}。")
    if payload.get("task_name") != TASK_NAME:
        raise ValueError(f"历史比较清单仅支持任务：{TASK_NAME}。")
    if payload.get("purpose") != "software_demo":
        raise ValueError("schema v1 仅接受 purpose=software_demo。")
    if payload.get("performance_evidence") is not False:
        raise ValueError(
            "schema v1 必须声明 performance_evidence=false；"
            "未来真实评测需接入独立审计契约。"
        )
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("历史比较清单缺少 dataset_id。")
    if not isinstance(payload.get("origin"), str) or not payload["origin"].strip():
        raise ValueError("历史比较清单缺少 origin。")
    if not isinstance(payload.get("label_provenance"), str) or not payload[
        "label_provenance"
    ].strip():
        raise ValueError("历史比较清单缺少 label_provenance。")

    spec = task_spec(TASK_NAME)
    expected_fields = {
        "sample_dtype": spec.sample_dtype,
        "input_layout": spec.input_layout,
        "channels": spec.channels,
        "window_size": spec.window_size,
    }
    for key, expected in expected_fields.items():
        if payload.get(key) != expected:
            raise ValueError(f"清单字段 {key} 与历史任务契约不一致。")
    if tuple(payload.get("label_order", ())) != spec.classes:
        raise ValueError("清单 label_order 与冻结历史任务标签顺序不一致。")

    rows = payload.get("recordings")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_RECORDINGS:
        raise ValueError(f"recordings 数量应在 1–{MAX_RECORDINGS} 之间。")
    base = manifest.parent.resolve()
    ids: list[str] = []
    relative_paths: list[str] = []
    files: list[Path] = []
    labels: list[str] = []
    groups: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"recordings[{index}] 必须是 JSON 对象。")
        recording_id = row.get("recording_id")
        relative = row.get("path")
        label = row.get("label")
        group = row.get("acquisition_group")
        expected_hash = row.get("sha256")
        expected_bytes = row.get("bytes")
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise ValueError(f"recordings[{index}] 缺少 recording_id。")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"recordings[{index}] 缺少相对 path。")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"recordings[{index}] path 必须位于清单目录内。")
        path = (base / relative_path).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"recordings[{index}] path 逃逸清单目录。") from exc
        if not path.is_file():
            raise ValueError(f"recordings[{index}] 文件不存在：{relative}")
        if path.suffix.lower() not in {".dat", ".bin", ".raw"}:
            raise ValueError(f"recordings[{index}] 不是受支持的原始信号文件。")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ValueError(f"recordings[{index}] bytes 必须是整数。")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"recordings[{index}] 文件大小与清单不一致。")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"recordings[{index}] sha256 格式无效。")
        if _sha256(path) != expected_hash.upper():
            raise ValueError(f"recordings[{index}] SHA-256 与清单不一致。")
        if label not in spec.classes:
            raise ValueError(f"recordings[{index}] label 不在冻结标签顺序中。")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"recordings[{index}] 缺少 acquisition_group。")
        ids.append(recording_id)
        relative_paths.append(relative_path.as_posix().casefold())
        files.append(path)
        labels.append(label)
        groups.append(group)
    if len(set(ids)) != len(ids):
        raise ValueError("历史比较清单包含重复 recording_id。")
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("历史比较清单包含重复 path。")

    loaded = load_recognition_files(TASK_NAME, files)
    windows = tuple(array.shape[-1] // spec.window_size for array in loaded.arrays)
    return PreparedHistoricalComparison(
        dataset_id=dataset_id.strip(),
        purpose="software_demo",
        manifest_path=manifest,
        manifest_sha256=_sha256(manifest),
        inputs=loaded,
        recording_ids=tuple(ids),
        declared_demo_labels=tuple(labels),
        acquisition_groups=tuple(groups),
        windows_per_recording=windows,
    )


__all__ = [
    "DEFAULT_MANIFEST_RESOURCE",
    "MAX_COMPARISON_WINDOWS",
    "PERFORMANCE_LIMITATION",
    "PERFORMANCE_STATUS",
    "PreparedHistoricalComparison",
    "default_historical_comparison_manifest",
    "prepare_historical_comparison",
]
