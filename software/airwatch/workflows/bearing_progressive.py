"""Progressive, cancellable orchestration for frozen bearing inference.

This module deliberately wraps :class:`BearingDiagnosisWorkflow` instead of
changing the frozen predictor implementation.  Raw windows are processed in
small batches, allowing a Qt worker to report progress and honor cancellation
between model calls while preserving the frozen quality and preprocessing
rules.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from airwatch.analysis.bearing_quality import BearingQualityThresholdsV2
from airwatch.data.cwru import CWRUDataError, load_cwru_signal

from .bearing_diagnosis import BearingDiagnosisWorkflow


ProgressCallback = Callable[[int, int, str], None]
CancellationCheck = Callable[[], bool]


class BearingDiagnosisCancelled(RuntimeError):
    """Raised when a progressive diagnosis is cooperatively cancelled."""


@dataclass(frozen=True)
class ProgressiveBearingDiagnosisResult:
    """Structured file-level result suitable for a desktop presentation layer."""

    source_path: str | None
    sensor_key: str | None
    device: str
    checkpoint_path: str
    window_size: int
    step: int
    window_count: int
    processed_window_count: int
    quality_status: str
    quality_message: str
    quality_status_counts: dict[str, int]
    predicted_label: str | None
    predicted_class: int | None
    mean_confidence: float | None
    class_counts: dict[str, int]
    elapsed_seconds: float
    windows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_status_counts"] = dict(self.quality_status_counts)
        payload["class_counts"] = dict(self.class_counts)
        payload["windows"] = [dict(row) for row in self.windows]
        return payload


class ProgressiveBearingDiagnosisWorkflow:
    """Run the existing diagnosis workflow in progress-reporting batches."""

    def __init__(
        self,
        workflow: BearingDiagnosisWorkflow,
        *,
        thresholds: BearingQualityThresholdsV2,
        quality_messages: dict[str, str] | None = None,
    ) -> None:
        self.workflow = workflow
        self.thresholds = thresholds
        self.quality_messages = dict(quality_messages or {})

    def run_file(
        self,
        path: str | Path,
        *,
        sensor_key: str | None = None,
        batch_size: int = 32,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
    ) -> ProgressiveBearingDiagnosisResult:
        """Load one CWRU MAT file and diagnose it without blocking the Qt thread."""

        self._raise_if_cancelled(is_cancelled)
        try:
            record = load_cwru_signal(path, sensor_key=sensor_key, normalization="none")
        except CWRUDataError:
            raise
        return self.run_signal(
            record.signal,
            source_path=str(record.path.resolve()),
            sensor_key=record.sensor_key,
            batch_size=batch_size,
            progress=progress,
            is_cancelled=is_cancelled,
        )

    def run_signal(
        self,
        signal: object,
        *,
        source_path: str | None = None,
        sensor_key: str | None = None,
        batch_size: int = 32,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
    ) -> ProgressiveBearingDiagnosisResult:
        """Process a finite one-dimensional signal in cancellable model batches."""

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        predictor = self.workflow.adapter.predictor
        if predictor.step != predictor.window_size:
            raise ValueError(
                "progressive inference currently requires non-overlapping frozen windows"
            )
        array = np.asarray(signal)
        if array.ndim != 1:
            raise CWRUDataError(
                f"bearing signal must be one-dimensional, got shape={array.shape}"
            )
        if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
            raise CWRUDataError("bearing signal must be a real numeric time series")
        converted = np.asarray(array, dtype=np.float32)
        if converted.size < predictor.window_size:
            windows = np.empty((0, predictor.window_size), dtype=np.float32)
        else:
            windows = np.stack(
                [
                    converted[start : start + predictor.window_size]
                    for start in range(
                        0,
                        converted.size - predictor.window_size + 1,
                        predictor.step,
                    )
                ]
            )
        total = int(windows.shape[0])
        if total == 0:
            raise CWRUDataError(
                f"信号长度不足：至少需要 {predictor.window_size} 个采样点"
            )

        self._raise_if_cancelled(is_cancelled)
        if progress is not None:
            progress(0, total, f"已读取信号，共 {total} 个窗口")
        started_at = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for start in range(0, total, batch_size):
            self._raise_if_cancelled(is_cancelled)
            stop = min(start + batch_size, total)
            batch_signal = np.asarray(
                windows[start:stop], dtype=np.float32
            ).reshape(-1)
            batch_rows = self.workflow.run_signal_with_quality_v2(
                batch_signal,
                thresholds=self.thresholds,
            )
            for local_row in batch_rows:
                row = dict(local_row)
                global_index = start + int(row["window_index"])
                row["window_index"] = global_index
                if row.get("prediction") is not None:
                    prediction = dict(row["prediction"])
                    prediction["window_index"] = global_index
                    row["prediction"] = prediction
                rows.append(row)
            if progress is not None:
                progress(stop, total, f"正在诊断：{stop} / {total} 个窗口")
            self._raise_if_cancelled(is_cancelled)

        return self._summarize(
            rows,
            source_path=source_path,
            sensor_key=sensor_key,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    @staticmethod
    def _raise_if_cancelled(is_cancelled: CancellationCheck | None) -> None:
        if is_cancelled is not None and is_cancelled():
            raise BearingDiagnosisCancelled("bearing diagnosis was cancelled")

    def _summarize(
        self,
        rows: list[dict[str, Any]],
        *,
        source_path: str | None,
        sensor_key: str | None,
        elapsed_seconds: float,
    ) -> ProgressiveBearingDiagnosisResult:
        predictor = self.workflow.adapter.predictor
        status_counts = Counter(str(row["status"]) for row in rows)
        normalized_status_counts = {
            state: int(status_counts.get(state, 0))
            for state in ("accepted", "caution", "rejected")
        }
        predictions = [
            dict(row["prediction"])
            for row in rows
            if row.get("prediction") is not None
        ]
        if not predictions:
            quality_status = "rejected"
            predicted_class = None
            predicted_label = None
            mean_confidence = None
            class_counts: dict[str, int] = {}
        else:
            counts = Counter(int(row["predicted_class"]) for row in predictions)
            predicted_class = min(
                counts,
                key=lambda class_index: (-counts[class_index], class_index),
            )
            predicted_label = predictor.label_map[predicted_class]
            mean_confidence = float(
                np.mean([float(row["confidence"]) for row in predictions])
            )
            class_counts = {
                predictor.label_map[index]: int(counts[index])
                for index in sorted(counts)
            }
            quality_status = (
                "accepted"
                if normalized_status_counts["caution"] == 0
                and normalized_status_counts["rejected"] == 0
                else "caution"
            )

        if quality_status == "caution" and normalized_status_counts["rejected"]:
            quality_message = (
                f"有 {normalized_status_counts['rejected']} 个窗口质量不足，未进入模型；"
                "其余窗口的诊断结果仅供参考，建议复核或重新采集"
            )
        else:
            quality_message = self.quality_messages.get(
                quality_status,
                {
                    "accepted": "信号质量检查通过",
                    "caution": "信号质量一般，诊断结果仅供参考，建议复核或重新采集",
                    "rejected": "信号质量不足，建议重新采集",
                }[quality_status],
            )

        return ProgressiveBearingDiagnosisResult(
            source_path=source_path,
            sensor_key=sensor_key,
            device=str(predictor.device),
            checkpoint_path=str(predictor.checkpoint_path),
            window_size=predictor.window_size,
            step=predictor.step,
            window_count=len(rows),
            processed_window_count=len(rows),
            quality_status=quality_status,
            quality_message=quality_message,
            quality_status_counts=normalized_status_counts,
            predicted_label=predicted_label,
            predicted_class=predicted_class,
            mean_confidence=mean_confidence,
            class_counts=class_counts,
            elapsed_seconds=float(elapsed_seconds),
            windows=tuple(rows),
        )


__all__ = [
    "BearingDiagnosisCancelled",
    "ProgressiveBearingDiagnosisResult",
    "ProgressiveBearingDiagnosisWorkflow",
]
