"""Atomic, cancellable workflow for the historical conditional GAN demo.

This Module owns the complete generation Implementation: request validation,
model loading, bounded batch inference, within-class descriptive scoring and an
all-or-nothing session directory.  It has no Qt dependency and performs no
training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from threading import Event, Lock
import time
from typing import Callable, Protocol
from uuid import uuid4

import numpy as np
import torch

from airwatch.runtime_resources import (
    HISTORICAL_GENERATOR_SLOT_ID,
    model_slot,
    model_slot_resource_path,
)
from models.generator import Generator


GENERATION_CLASSES: tuple[str, ...] = (
    "32PSK",
    "16APSK",
    "32QAM",
    "FM",
    "GMSK",
    "32APSK",
    "OQPSK",
    "8ASK",
    "BPSK",
    "8PSK",
    "AM-SSB-SC",
    "4ASK",
    "16PSK",
    "64APSK",
    "128QAM",
)
ALL_CLASSES = "全选"
SAMPLE_LENGTH = 1024
LATENT_DIMENSION = 100
MIN_SAMPLES_PER_CLASS = 2
MAX_SAMPLES_PER_CLASS = 10_000
MAX_TOTAL_SAMPLES = 30_000
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
DEFAULT_BATCH_SIZE = 256
SESSION_SCHEMA_VERSION = 1
CONSISTENCY_NAME = "批内信号统计偏离"
CONSISTENCY_LIMITATION = (
    "仅描述同一条件类别、同一生成批次内的统计离散程度；"
    "不验证类别正确性、真实性或模型质量，也不可作为 FID 或竞赛指标。"
)


class GenerationProgress(Protocol):
    def __call__(self, current: int, total: int) -> None: ...


class GenerationPhase(Protocol):
    def __call__(self, message: str) -> None: ...


ModelLoader = Callable[[], tuple[torch.nn.Module, Path]]


def _readonly_float32(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return np.frombuffer(array.tobytes(), dtype=np.float32).reshape(array.shape)


@dataclass(frozen=True)
class GenerationRequest:
    """Immutable user request captured before background execution starts."""

    output_root: Path
    class_name: str
    samples_per_class: int
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        root = Path(self.output_root).expanduser().resolve()
        object.__setattr__(self, "output_root", root)
        if self.class_name != ALL_CLASSES and self.class_name not in GENERATION_CLASSES:
            raise ValueError(f"不支持的生成类别：{self.class_name}")
        if isinstance(self.samples_per_class, bool) or not isinstance(
            self.samples_per_class, int
        ):
            raise ValueError("每类样本数必须是整数。")
        if not MIN_SAMPLES_PER_CLASS <= self.samples_per_class <= MAX_SAMPLES_PER_CLASS:
            raise ValueError(
                f"每类样本数应在 {MIN_SAMPLES_PER_CLASS}–{MAX_SAMPLES_PER_CLASS} 之间。"
            )
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ValueError("生成批大小必须是整数。")
        if not 1 <= self.batch_size <= 2048:
            raise ValueError("生成批大小应在 1–2048 之间。")
        if self.total_samples > MAX_TOTAL_SAMPLES:
            raise ValueError(
                f"本次共 {self.total_samples} 个样本，超过安全上限 {MAX_TOTAL_SAMPLES}；"
                "请减少每类样本数或改为单类生成。"
            )
        if self.estimated_output_bytes > MAX_OUTPUT_BYTES:
            raise ValueError("本次生成数据超过 128 MiB 安全上限。")

    @property
    def class_names(self) -> tuple[str, ...]:
        return GENERATION_CLASSES if self.class_name == ALL_CLASSES else (self.class_name,)

    @property
    def total_samples(self) -> int:
        return self.samples_per_class * len(self.class_names)

    @property
    def estimated_output_bytes(self) -> int:
        return self.total_samples * SAMPLE_LENGTH * np.dtype(np.float32).itemsize


@dataclass(frozen=True)
class GeneratedClassResult:
    label: str
    samples: np.ndarray
    deviation_scores: np.ndarray
    output_file: Path
    elapsed_seconds: float

    def __post_init__(self) -> None:
        samples = _readonly_float32(self.samples)
        scores = _readonly_float32(self.deviation_scores)
        if samples.ndim != 3 or samples.shape[1:] != (1, SAMPLE_LENGTH):
            raise ValueError("生成结果形状必须为 (样本数, 1, 1024)。")
        if scores.shape != (samples.shape[0],):
            raise ValueError("偏离分数必须与生成样本一一对应。")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "deviation_scores", scores)
        object.__setattr__(self, "output_file", Path(self.output_file).resolve())


@dataclass(frozen=True)
class GenerationResult:
    session_id: str
    session_directory: Path
    classes: tuple[GeneratedClassResult, ...]
    checkpoint_name: str
    model_name: str
    device_name: str
    elapsed_seconds: float
    limitation: str = CONSISTENCY_LIMITATION

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(item.label for item in self.classes)

    @property
    def batches(self) -> tuple[np.ndarray, ...]:
        return tuple(item.samples for item in self.classes)

    @property
    def total_samples(self) -> int:
        return sum(item.samples.shape[0] for item in self.classes)

    @property
    def mean_deviation(self) -> float:
        scores = [item.deviation_scores for item in self.classes]
        return float(np.mean(np.concatenate(scores))) if scores else float("nan")

    def score_for(self, class_name: str, sample_number: int) -> float:
        for item in self.classes:
            if item.label == class_name:
                if not 1 <= sample_number <= len(item.deviation_scores):
                    raise IndexError(
                        f"{class_name} 的样本序号应在 1–{len(item.deviation_scores)} 之间。"
                    )
                return float(item.deviation_scores[sample_number - 1])
        raise KeyError(f"本次生成结果不包含类别：{class_name}")


def signal_statistics_deviation(samples: np.ndarray) -> np.ndarray:
    """Return transparent, within-class descriptive deviations per sample.

    Five features are compared with the class-batch median.  The robust scale is
    bounded by feature magnitude so the batch mean is not mechanically fixed at
    one, unlike z-scoring against the same batch.  The result remains descriptive
    only and deliberately makes no claim about realism or class correctness.
    """

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (1, SAMPLE_LENGTH):
        raise ValueError("一致性分析需要形状为 (样本数, 1, 1024) 的数据。")
    if array.shape[0] < MIN_SAMPLES_PER_CLASS:
        raise ValueError("一致性分析至少需要两个同类样本。")
    values = array[:, 0, :]
    rms = np.sqrt(np.mean(values**2, axis=1))
    peak = np.max(np.abs(values), axis=1)
    zero_crossing = np.mean(np.diff(np.signbit(values), axis=1), axis=1)
    features = np.column_stack(
        (
            np.mean(values, axis=1),
            np.std(values, axis=1),
            rms,
            peak / np.maximum(rms, np.finfo(float).eps),
            zero_crossing,
        )
    )
    center = np.median(features, axis=0)
    robust_spread = 1.4826 * np.median(np.abs(features - center), axis=0)
    magnitude_floor = np.maximum(np.abs(center), 1e-6)
    scale = np.maximum(robust_spread, magnitude_floor)
    normalized = (features - center) / scale
    scores = np.sqrt(np.mean(normalized**2, axis=1))
    if not np.isfinite(scores).all():
        raise ValueError("生成数据的批内统计偏离包含非有限值。")
    return scores.astype(np.float32, copy=False)


def _default_model_loader(device: torch.device) -> tuple[torch.nn.Module, Path]:
    checkpoint = model_slot_resource_path(HISTORICAL_GENERATOR_SLOT_ID)
    generator = Generator()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    generator.load_state_dict(state, strict=True)
    generator.eval()
    return generator.to(device), checkpoint


class HistoricalGenerationWorkflow:
    """Deep Interface for one atomic historical-generation session."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model_loader = model_loader
        self._model: torch.nn.Module | None = None
        self._checkpoint: Path | None = None
        self._load_lock = Lock()

    def _load_model(self) -> tuple[torch.nn.Module, Path]:
        with self._load_lock:
            if self._model is None or self._checkpoint is None:
                if self._model_loader is None:
                    model, checkpoint = _default_model_loader(self.device)
                else:
                    model, checkpoint = self._model_loader()
                    model = model.to(self.device).eval()
                self._model = model
                self._checkpoint = Path(checkpoint).resolve()
            return self._model, self._checkpoint

    @staticmethod
    def _check_cancelled(cancelled: Event) -> None:
        if cancelled.is_set():
            raise InterruptedError("生成已取消，不保留部分结果。")

    @staticmethod
    def _new_session_paths(root: Path) -> tuple[str, Path, Path]:
        for _ in range(10):
            session_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + uuid4().hex[:8]
            )
            final = root / f"generation-{session_id}"
            staging = root / f".generation-{session_id}.staging"
            if not final.exists() and not staging.exists():
                return session_id, staging, final
        raise RuntimeError("无法分配唯一的生成会话目录。")

    @staticmethod
    def _cleanup_staging(staging: Path, root: Path) -> None:
        if not staging.exists():
            return
        resolved = staging.resolve()
        if resolved.parent != root.resolve() or not resolved.name.startswith(".generation-"):
            raise RuntimeError("拒绝清理不属于本次生成会话的目录。")
        shutil.rmtree(resolved)

    def run(
        self,
        request: GenerationRequest,
        cancelled: Event,
        progress: GenerationProgress | None = None,
        phase: GenerationPhase | None = None,
    ) -> GenerationResult:
        if not isinstance(request, GenerationRequest):
            raise TypeError("生成任务输入必须是 GenerationRequest。")
        root = request.output_root
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise ValueError(f"输出路径不是文件夹：{root}")
        free_bytes = shutil.disk_usage(root).free
        required_bytes = request.estimated_output_bytes + 16 * 1024 * 1024
        if free_bytes < required_bytes:
            raise OSError(
                f"输出磁盘空间不足：至少需要约 {required_bytes / 1024**2:.1f} MiB。"
            )

        workflow_started = time.perf_counter()
        self._check_cancelled(cancelled)
        if phase:
            phase("正在加载历史条件 GAN（仅演示，不训练）…")
        model, checkpoint = self._load_model()
        self._check_cancelled(cancelled)

        session_id, staging, final = self._new_session_paths(root)
        staging.mkdir(parents=False, exist_ok=False)
        class_payloads: list[tuple[str, np.ndarray, np.ndarray, float]] = []
        total_done = 0
        generation_started = time.perf_counter()
        if progress:
            progress(0, request.total_samples)
        try:
            for label in request.class_names:
                self._check_cancelled(cancelled)
                if phase:
                    phase(f"正在生成 {label}：0 / {request.samples_per_class}")
                class_started = time.perf_counter()
                class_index = GENERATION_CLASSES.index(label)
                output = np.empty(
                    (request.samples_per_class, 1, SAMPLE_LENGTH), dtype=np.float32
                )
                with torch.inference_mode():
                    for start in range(0, request.samples_per_class, request.batch_size):
                        self._check_cancelled(cancelled)
                        stop = min(start + request.batch_size, request.samples_per_class)
                        count = stop - start
                        noise = torch.randn(count, LATENT_DIMENSION, device=self.device)
                        labels = torch.full(
                            (count,), class_index, dtype=torch.long, device=self.device
                        )
                        generated = model(noise, labels)
                        if tuple(generated.shape) != (count, 1, SAMPLE_LENGTH):
                            raise ValueError(
                                "历史生成模型输出形状与 (批大小, 1, 1024) 契约不一致。"
                            )
                        if not torch.isfinite(generated).all():
                            raise ValueError("历史生成模型输出包含非有限值。")
                        output[start:stop] = generated.detach().cpu().numpy().astype(
                            np.float32, copy=False
                        )
                        total_done += count
                        if progress:
                            progress(total_done, request.total_samples)
                        if phase:
                            phase(
                                f"正在生成 {label}：{stop} / {request.samples_per_class}"
                            )
                scores = signal_statistics_deviation(output)
                np.save(staging / f"{label}.npy", output, allow_pickle=False)
                class_payloads.append(
                    (label, output, scores, time.perf_counter() - class_started)
                )

            self._check_cancelled(cancelled)
            generation_elapsed = time.perf_counter() - generation_started
            slot = model_slot(HISTORICAL_GENERATOR_SLOT_ID)
            manifest = {
                "schema_version": SESSION_SCHEMA_VERSION,
                "session_id": session_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_slot": slot.slot_id,
                "release_tier": slot.release_tier,
                "checkpoint": checkpoint.name,
                "model_class": type(model).__name__,
                "device": str(self.device),
                "requested_class": request.class_name,
                "classes": list(request.class_names),
                "samples_per_class": request.samples_per_class,
                "total_samples": request.total_samples,
                "sample_shape": [1, SAMPLE_LENGTH],
                "dtype": "float32",
                "generation_elapsed_seconds": generation_elapsed,
                "workflow_elapsed_seconds_before_commit": (
                    time.perf_counter() - workflow_started
                ),
                "descriptive_score": CONSISTENCY_NAME,
                "score_limitation": CONSISTENCY_LIMITATION,
                "capability_limitation": slot.limitation,
            }
            (staging / "generation-session.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._check_cancelled(cancelled)
            staging.rename(final)
        except BaseException:
            self._cleanup_staging(staging, root)
            raise

        class_results = tuple(
            GeneratedClassResult(
                label=label,
                samples=samples,
                deviation_scores=scores,
                output_file=final / f"{label}.npy",
                elapsed_seconds=elapsed,
            )
            for label, samples, scores, elapsed in class_payloads
        )
        return GenerationResult(
            session_id=session_id,
            session_directory=final.resolve(),
            classes=class_results,
            checkpoint_name=checkpoint.name,
            model_name=type(model).__name__,
            device_name=str(self.device),
            elapsed_seconds=time.perf_counter() - workflow_started,
        )


__all__ = [
    "ALL_CLASSES",
    "CONSISTENCY_LIMITATION",
    "CONSISTENCY_NAME",
    "DEFAULT_BATCH_SIZE",
    "GENERATION_CLASSES",
    "GeneratedClassResult",
    "GenerationRequest",
    "GenerationResult",
    "HistoricalGenerationWorkflow",
    "MAX_SAMPLES_PER_CLASS",
    "MAX_TOTAL_SAMPLES",
    "signal_statistics_deviation",
]
