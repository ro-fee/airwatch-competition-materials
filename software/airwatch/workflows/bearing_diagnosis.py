"""Backend workflow for one frozen-checkpoint bearing diagnosis run.

This module is intentionally small: it coordinates the existing inference
adapter and exposes a stable entry point for a future UI or service layer.
It does not train, alter, or save models; it also never modifies the input
signal or source file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from airwatch.analysis.bearing_quality import BearingQualityThresholdsV2
from airwatch.inference.adapter import BearingInferenceAdapter, BearingInferenceResult


class BearingDiagnosisWorkflow:
    """Run a complete bearing diagnosis using a reusable inference adapter.

    The workflow is the boundary that a future Qt worker can call.  Keeping
    this layer free of Qt means the same behaviour can be tested and reused in
    a command-line tool, a batch job, or the desktop application later.
    """

    def __init__(self, adapter: BearingInferenceAdapter) -> None:
        self.adapter = adapter

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        label_map_path: str | Path | None = None,
    ) -> "BearingDiagnosisWorkflow":
        """Build a workflow around one frozen checkpoint loaded once."""
        return cls(
            BearingInferenceAdapter.from_checkpoint(
                checkpoint_path,
                device=device,
                label_map_path=label_map_path,
            )
        )

    def run_file(
        self,
        path: str | Path,
        *,
        sensor_key: str | None = None,
    ) -> BearingInferenceResult:
        """Diagnose one CWRU ``.mat`` file without changing it."""
        return self.adapter.predict_file(path, sensor_key=sensor_key)

    def run_signal(self, signal: object) -> BearingInferenceResult:
        """Diagnose one in-memory one-dimensional signal."""
        return self.adapter.predict_signal(signal)

    def run_signal_with_quality_v2(
        self,
        signal: object,
        *,
        thresholds: BearingQualityThresholdsV2 | None = None,
    ) -> list[dict[str, Any]]:
        """Apply the three-state quality gate before window inference."""
        return self.adapter.predict_signal_with_quality_v2(
            signal,
            thresholds=thresholds,
        )

    def run_file_with_quality_v2(
        self,
        path: str | Path,
        *,
        sensor_key: str | None = None,
        thresholds: BearingQualityThresholdsV2 | None = None,
    ) -> list[dict[str, Any]]:
        """Quality-check and diagnose one CWRU MAT file without changing it."""
        return self.adapter.predict_file_with_quality_v2(
            path,
            sensor_key=sensor_key,
            thresholds=thresholds,
        )

    def run(
        self,
        source: object,
        *,
        sensor_key: str | None = None,
    ) -> BearingInferenceResult:
        """Diagnose either a path-like source or an in-memory signal."""
        if isinstance(source, (str, Path)):
            return self.run_file(source, sensor_key=sensor_key)
        if sensor_key is not None:
            raise TypeError("sensor_key is only valid when source is a MAT file path")
        return self.run_signal(source)

    def run_with_quality_v2(
        self,
        source: object,
        *,
        sensor_key: str | None = None,
        thresholds: BearingQualityThresholdsV2 | None = None,
    ) -> list[dict[str, Any]]:
        """Dispatch V2 quality-aware inference for either a file or signal."""
        if isinstance(source, (str, Path)):
            return self.run_file_with_quality_v2(
                source,
                sensor_key=sensor_key,
                thresholds=thresholds,
            )
        if sensor_key is not None:
            raise TypeError("sensor_key is only valid when source is a MAT file path")
        return self.run_signal_with_quality_v2(source, thresholds=thresholds)

    @staticmethod
    def render_json(
        result: BearingInferenceResult,
        *,
        indent: int = 2,
    ) -> str:
        """Render a result for logging or transport without writing a file."""
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=indent)

    @staticmethod
    def render_quality_json_v2(
        rows: list[dict[str, Any]],
        *,
        indent: int = 2,
    ) -> str:
        """Render V2 rows as strict JSON without writing an evidence file."""
        return json.dumps(
            rows,
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )


__all__ = ["BearingDiagnosisWorkflow"]
