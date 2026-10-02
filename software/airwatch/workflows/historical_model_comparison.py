"""Cancellable, evidence-gated comparison of two historical individual models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from threading import Event, Lock
import time
from typing import Callable
from uuid import uuid4

import numpy as np
import torch

from airwatch.data.historical_comparison import (
    PERFORMANCE_LIMITATION,
    PERFORMANCE_STATUS,
    PreparedHistoricalComparison,
    prepare_historical_comparison,
)
from airwatch.general_recognition_contract import task_spec
from airwatch.runtime_resources import (
    HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID,
    general_task_slot_id,
    model_slot_resource_path,
)
from models.ResNet18_TCN import ResNet18_TCN
from models.individual_light import CNNNEW_zl


TASK_NAME = "信号个体识别"
EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZE = 256
COMPARISON_LIMITATION = (
    PERFORMANCE_LIMITATION
    + " 两模型窗口预测一致率只描述输出相似程度，不代表任一模型正确。"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class ComparisonRunRequest:
    manifest_path: Path
    output_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_path", Path(self.manifest_path).expanduser().resolve()
        )
        object.__setattr__(
            self, "output_root", Path(self.output_root).expanduser().resolve()
        )


@dataclass(frozen=True)
class LoadedComparisonModel:
    role: str
    slot_id: str
    model: torch.nn.Module
    checkpoint_path: Path


@dataclass(frozen=True)
class ModelComparisonMeasurement:
    role: str
    slot_id: str
    model_name: str
    checkpoint_name: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    parameter_count: int
    file_predictions: tuple[str, ...]
    window_predictions: tuple[int, ...]


@dataclass(frozen=True)
class HistoricalModelComparisonResult:
    run_id: str
    evidence_directory: Path
    evidence_file: Path
    dataset_id: str
    manifest_name: str
    manifest_sha256: str
    recording_count: int
    windows_per_recording: tuple[int, ...]
    total_windows: int
    reference: ModelComparisonMeasurement
    lightweight: ModelComparisonMeasurement
    agreement_count: int
    agreement_rate: float
    workflow_elapsed_seconds: float
    performance_status: str = PERFORMANCE_STATUS
    accuracy: None = None
    macro_precision: None = None
    macro_f1: None = None
    limitation: str = COMPARISON_LIMITATION

    def __post_init__(self) -> None:
        if not 0 <= self.agreement_count <= self.total_windows:
            raise ValueError("窗口预测一致数量超出总窗口数。")
        expected = self.agreement_count / self.total_windows
        if abs(self.agreement_rate - expected) > 1e-12:
            raise ValueError("窗口预测一致率与计数不一致。")
        if any(value is not None for value in (self.accuracy, self.macro_precision, self.macro_f1)):
            raise ValueError("软件演示比较不得发布准确率类指标。")


ModelLoader = Callable[[], tuple[LoadedComparisonModel, LoadedComparisonModel]]


def _default_model_loader(
    device: torch.device,
) -> tuple[LoadedComparisonModel, LoadedComparisonModel]:
    spec = task_spec(TASK_NAME)
    reference_path = model_slot_resource_path(HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID)
    reference = ResNet18_TCN(num_classes=len(spec.classes))
    reference.load_state_dict(
        torch.load(reference_path, map_location=device, weights_only=True), strict=True
    )
    reference.to(device).eval()

    lightweight_slot = general_task_slot_id(TASK_NAME)
    lightweight_path = model_slot_resource_path(lightweight_slot)
    lightweight = CNNNEW_zl(len(spec.classes))
    lightweight.load_state_dict(
        torch.load(lightweight_path, map_location=device, weights_only=True), strict=True
    )
    lightweight.to(device).eval()
    return (
        LoadedComparisonModel(
            "历史 TCN 对照模型",
            HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID,
            reference,
            reference_path,
        ),
        LoadedComparisonModel(
            "历史 10 层轻量模型",
            lightweight_slot,
            lightweight,
            lightweight_path,
        ),
    )


class HistoricalModelComparisonWorkflow:
    """Deep Module hiding manifest preparation, both forwards and Evidence."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model_loader: ModelLoader | None = None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 1024:
            raise ValueError("历史比较批大小应在 1–1024 之间。")
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self._model_loader = model_loader
        self._models: tuple[LoadedComparisonModel, LoadedComparisonModel] | None = None
        self._load_lock = Lock()

    @staticmethod
    def _check_cancelled(cancelled: Event) -> None:
        if cancelled.is_set():
            raise InterruptedError("历史模型比较已取消，不发布部分结果。")

    def _load_models(self) -> tuple[LoadedComparisonModel, LoadedComparisonModel]:
        with self._load_lock:
            if self._models is None:
                models = (
                    _default_model_loader(self.device)
                    if self._model_loader is None
                    else self._model_loader()
                )
                if len(models) != 2:
                    raise ValueError("历史比较必须提供一个对照模型和一个轻量模型。")
                normalized = []
                for item in models:
                    item.model.to(self.device).eval()
                    normalized.append(
                        LoadedComparisonModel(
                            item.role,
                            item.slot_id,
                            item.model,
                            Path(item.checkpoint_path).resolve(),
                        )
                    )
                self._models = tuple(normalized)  # type: ignore[assignment]
            return self._models

    @staticmethod
    def _prepare_windows(prepared: PreparedHistoricalComparison) -> np.ndarray:
        spec = task_spec(TASK_NAME)
        total = prepared.total_windows
        windows = np.empty((total, spec.channels, spec.window_size), dtype=np.float32)
        cursor = 0
        for array, count in zip(prepared.inputs.arrays, prepared.windows_per_recording):
            usable = array[:, : count * spec.window_size]
            block = usable.reshape(spec.channels, count, spec.window_size).transpose(1, 0, 2)
            windows[cursor : cursor + count] = block
            cursor += count
        return windows

    def _predict(
        self,
        loaded: LoadedComparisonModel,
        windows: np.ndarray,
        windows_per_recording: tuple[int, ...],
        cancelled: Event,
        progress: Callable[[int, int], None] | None,
        progress_offset: int,
        progress_total: int,
    ) -> ModelComparisonMeasurement:
        self._check_cancelled(cancelled)
        spec = task_spec(TASK_NAME)
        model = loaded.model
        # Warm-up is excluded from progress and is not published as latency.
        warm_count = min(len(windows), self.batch_size)
        with torch.inference_mode():
            warm = torch.from_numpy(np.array(windows[:warm_count], copy=True)).to(self.device)
            _, warm_logits = model(warm)
            if tuple(warm_logits.shape) != (warm_count, len(spec.classes)):
                raise ValueError(f"{loaded.role} 输出类别数量与冻结标签顺序不一致。")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            predictions: list[int] = []
            done = 0
            for start in range(0, len(windows), self.batch_size):
                self._check_cancelled(cancelled)
                stop = min(start + self.batch_size, len(windows))
                batch = torch.from_numpy(np.array(windows[start:stop], copy=True)).to(
                    self.device
                )
                _, logits = model(batch)
                if tuple(logits.shape) != (stop - start, len(spec.classes)):
                    raise ValueError(f"{loaded.role} 输出形状与冻结标签顺序不一致。")
                if not torch.isfinite(logits).all():
                    raise ValueError(f"{loaded.role} 输出包含非有限值。")
                predictions.extend(logits.argmax(1).cpu().tolist())
                done += stop - start
                if progress:
                    progress(progress_offset + done, progress_total)

        file_labels: list[str] = []
        cursor = 0
        for count in windows_per_recording:
            votes = predictions[cursor : cursor + count]
            winner = int(np.bincount(votes, minlength=len(spec.classes)).argmax())
            file_labels.append(spec.classes[winner])
            cursor += count
        path = loaded.checkpoint_path
        if not path.is_file():
            raise FileNotFoundError(f"比较模型权重不存在：{path}")
        return ModelComparisonMeasurement(
            role=loaded.role,
            slot_id=loaded.slot_id,
            model_name=type(model).__name__,
            checkpoint_name=path.name,
            checkpoint_sha256=_sha256(path),
            checkpoint_bytes=path.stat().st_size,
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            file_predictions=tuple(file_labels),
            window_predictions=tuple(predictions),
        )

    @staticmethod
    def _session_paths(root: Path) -> tuple[str, Path, Path]:
        for _ in range(10):
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
            final = root / f"historical-comparison-{run_id}"
            staging = root / f".historical-comparison-{run_id}.staging"
            if not final.exists() and not staging.exists():
                return run_id, staging, final
        raise RuntimeError("无法分配唯一的历史比较 Evidence 目录。")

    @staticmethod
    def _cleanup_staging(staging: Path, root: Path) -> None:
        if not staging.exists():
            return
        resolved = staging.resolve()
        if resolved.parent != root.resolve() or not resolved.name.startswith(
            ".historical-comparison-"
        ):
            raise RuntimeError("拒绝清理不属于本次历史比较的目录。")
        shutil.rmtree(resolved)

    def run(
        self,
        request: ComparisonRunRequest,
        cancelled: Event,
        progress: Callable[[int, int], None] | None = None,
        phase: Callable[[str], None] | None = None,
    ) -> HistoricalModelComparisonResult:
        if not isinstance(request, ComparisonRunRequest):
            raise TypeError("历史比较任务输入必须是 ComparisonRunRequest。")
        started = time.perf_counter()
        self._check_cancelled(cancelled)
        if phase:
            phase("正在校验演示清单与文件 SHA-256…")
        prepared = prepare_historical_comparison(request.manifest_path)
        self._check_cancelled(cancelled)
        if prepared.performance_evidence_eligible:
            raise ValueError("schema v1 不允许把软件演示数据升级为性能证据。")
        if phase:
            phase("正在加载两个历史模型（严格权重匹配）…")
        reference_model, lightweight_model = self._load_models()
        self._check_cancelled(cancelled)
        windows = self._prepare_windows(prepared)
        total_progress = prepared.total_windows * 2
        if progress:
            progress(0, total_progress)
        if phase:
            phase("正在运行历史 TCN 对照模型…")
        reference = self._predict(
            reference_model,
            windows,
            prepared.windows_per_recording,
            cancelled,
            progress,
            0,
            total_progress,
        )
        self._check_cancelled(cancelled)
        if phase:
            phase("正在运行历史 10 层轻量模型…")
        lightweight = self._predict(
            lightweight_model,
            windows,
            prepared.windows_per_recording,
            cancelled,
            progress,
            prepared.total_windows,
            total_progress,
        )
        self._check_cancelled(cancelled)
        agreement_count = sum(
            left == right
            for left, right in zip(
                reference.window_predictions, lightweight.window_predictions
            )
        )
        agreement_rate = agreement_count / prepared.total_windows

        root = request.output_root
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise ValueError(f"Evidence 输出路径不是文件夹：{root}")
        run_id, staging, final = self._session_paths(root)
        staging.mkdir(parents=False, exist_ok=False)
        try:
            payload = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "historical software-path comparison; not performance evidence",
                "dataset": {
                    "dataset_id": prepared.dataset_id,
                    "purpose": prepared.purpose,
                    "manifest": prepared.manifest_path.name,
                    "manifest_sha256": prepared.manifest_sha256,
                    "recording_count": prepared.recording_count,
                    "windows_per_recording": list(prepared.windows_per_recording),
                    "total_windows": prepared.total_windows,
                },
                "models": {
                    "reference": self._measurement_payload(reference),
                    "lightweight": self._measurement_payload(lightweight),
                },
                "descriptive_comparison": {
                    "window_prediction_agreement_count": agreement_count,
                    "window_prediction_agreement_rate": agreement_rate,
                    "meaning": "output similarity only; not correctness",
                },
                "performance_metrics": {
                    "status": prepared.performance_status,
                    "accuracy": None,
                    "macro_precision": None,
                    "macro_f1": None,
                    "latency": None,
                },
                "limitation": COMPARISON_LIMITATION,
            }
            evidence_name = "comparison-result.json"
            (staging / evidence_name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._check_cancelled(cancelled)
            staging.rename(final)
        except BaseException:
            self._cleanup_staging(staging, root)
            raise
        elapsed = time.perf_counter() - started
        return HistoricalModelComparisonResult(
            run_id=run_id,
            evidence_directory=final.resolve(),
            evidence_file=(final / "comparison-result.json").resolve(),
            dataset_id=prepared.dataset_id,
            manifest_name=prepared.manifest_path.name,
            manifest_sha256=prepared.manifest_sha256,
            recording_count=prepared.recording_count,
            windows_per_recording=prepared.windows_per_recording,
            total_windows=prepared.total_windows,
            reference=reference,
            lightweight=lightweight,
            agreement_count=agreement_count,
            agreement_rate=agreement_rate,
            workflow_elapsed_seconds=elapsed,
        )

    @staticmethod
    def _measurement_payload(value: ModelComparisonMeasurement) -> dict[str, object]:
        return {
            "role": value.role,
            "slot_id": value.slot_id,
            "model_name": value.model_name,
            "checkpoint_name": value.checkpoint_name,
            "checkpoint_sha256": value.checkpoint_sha256,
            "checkpoint_bytes": value.checkpoint_bytes,
            "parameter_count": value.parameter_count,
            "file_predictions": list(value.file_predictions),
        }


__all__ = [
    "COMPARISON_LIMITATION",
    "ComparisonRunRequest",
    "HistoricalModelComparisonResult",
    "HistoricalModelComparisonWorkflow",
    "LoadedComparisonModel",
    "ModelComparisonMeasurement",
]
