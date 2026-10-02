"""Application-facing adapter for the frozen bearing inference service.

The adapter is deliberately UI-agnostic.  It turns the window-level output of
:class:`airwatch.inference.bearing.BearingPredictor` into one stable result
object that a future workflow or Qt page can consume without knowing about
PyTorch, checkpoint internals, or CWRU file parsing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from airwatch.analysis.bearing_quality import BearingQualityThresholdsV2

from .bearing import BearingPredictor, WindowPrediction


@dataclass(frozen=True)
class BearingInferenceResult:
    """Stable, JSON-friendly result returned to an upper-layer workflow."""

    source_path: str | None
    sensor_key: str | None
    device: str
    checkpoint_path: str
    window_size: int
    step: int
    window_count: int
    predicted_label: str
    predicted_class: int
    mean_confidence: float
    class_counts: dict[str, int]
    windows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary for logging, JSON, or a future UI."""
        payload = asdict(self)
        payload["windows"] = [dict(window) for window in self.windows]
        payload["class_counts"] = dict(self.class_counts)
        return payload


class BearingInferenceAdapter:
    """Small application boundary around a frozen :class:`BearingPredictor`.

    The adapter owns no model weights and never writes to the input file or
    checkpoint.  A caller can create it once and reuse it for multiple files,
    which avoids repeatedly loading the model when a future workflow processes
    a batch of recordings.
    """

    def __init__(self, predictor: BearingPredictor) -> None:
        self.predictor = predictor

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        label_map_path: str | Path | None = None,
    ) -> "BearingInferenceAdapter":
        """Create an adapter while loading one frozen checkpoint once."""
        return cls(
            BearingPredictor(
                checkpoint_path,
                device=device,
                label_map_path=label_map_path,
            )
        )

    def predict_signal(self, signal: object) -> BearingInferenceResult:
        """Predict a one-dimensional in-memory signal."""
        windows = self.predictor.predict_signal(signal)
        return self._build_result(windows, source_path=None, sensor_key=None)

    def predict_file(
        self,
        path: str | Path,
        *,
        sensor_key: str | None = None,
    ) -> BearingInferenceResult:
        """Predict a CWRU MAT file and return both details and a summary."""
        windows = self.predictor.predict_file(path, sensor_key=sensor_key)
        return self._build_result(
            windows,
            source_path=str(Path(path).resolve()),
            sensor_key=self._resolve_sensor_key(path, sensor_key),
        )

    def predict_signal_with_quality_v2(
        self,
        signal: object,
        *,
        thresholds: BearingQualityThresholdsV2 | None = None,
    ) -> list[dict[str, Any]]:
        """Return three-state window results without inventing a summary."""
        return self.predictor.predict_signal_with_quality_v2(
            signal, thresholds=thresholds
        )

    def predict_file_with_quality_v2(
        self,
        path: str | Path,
        *,
        sensor_key: str | None = None,
        thresholds: BearingQualityThresholdsV2 | None = None,
    ) -> list[dict[str, Any]]:
        """Apply the V2 quality gate to one file and return window rows."""
        return self.predictor.predict_file_with_quality_v2(
            path,
            sensor_key=sensor_key,
            thresholds=thresholds,
        )

    def predict(
        self,
        source: object,
        *,
        sensor_key: str | None = None,
    ) -> BearingInferenceResult:
        """Dispatch to file or array inference using one simple entry point."""
        if isinstance(source, (str, Path)):
            return self.predict_file(source, sensor_key=sensor_key)
        return self.predict_signal(source)

    def _resolve_sensor_key(
        self,
        path: str | Path,
        sensor_key: str | None,
    ) -> str | None:
        """Recover the selected key without making the array path ambiguous."""
        if sensor_key is not None:
            return sensor_key
        # ``BearingPredictor.predict_file`` has already validated that an
        # implicit selection is unambiguous.  The public result should still
        # expose the key, so read the metadata through the same loader only in
        # this uncommon reporting step.  This does not mutate the MAT file.
        from airwatch.data.cwru import load_cwru_signal

        return load_cwru_signal(
            path,
            normalization=self.predictor.normalization,
        ).sensor_key

    def _build_result(
        self,
        windows: list[dict[str, Any]],
        *,
        source_path: str | None,
        sensor_key: str | None,
    ) -> BearingInferenceResult:
        if not windows:
            raise RuntimeError("predictor returned no window predictions")

        counts_by_class = Counter(int(row["predicted_class"]) for row in windows)
        majority_class = min(
            counts_by_class,
            key=lambda class_index: (-counts_by_class[class_index], class_index),
        )
        class_counts = {
            self.predictor.label_map[class_index]: counts_by_class[class_index]
            for class_index in sorted(counts_by_class)
        }
        mean_confidence = float(
            np.mean([float(row["confidence"]) for row in windows])
        )
        return BearingInferenceResult(
            source_path=source_path,
            sensor_key=sensor_key,
            device=str(self.predictor.device),
            checkpoint_path=str(self.predictor.checkpoint_path),
            window_size=self.predictor.window_size,
            step=self.predictor.step,
            window_count=len(windows),
            predicted_label=self.predictor.label_map[majority_class],
            predicted_class=majority_class,
            mean_confidence=mean_confidence,
            class_counts=class_counts,
            windows=tuple(
                WindowPrediction(
                    window_index=int(row["window_index"]),
                    predicted_label=str(row["predicted_label"]),
                    predicted_class=int(row["predicted_class"]),
                    confidence=float(row["confidence"]),
                    probabilities=tuple(float(value) for value in row["probabilities"]),
                ).to_dict()
                for row in windows
            ),
        )


__all__ = ["BearingInferenceAdapter", "BearingInferenceResult"]
