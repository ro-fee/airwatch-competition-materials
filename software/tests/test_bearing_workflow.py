"""Regression tests for the UI-independent bearing diagnosis workflow."""

from __future__ import annotations

from pathlib import Path
import json
import unittest

import numpy as np

from airwatch.analysis import BearingQualityThresholdsV2
from airwatch.inference import InsufficientSignalError
from airwatch.workflows import BearingDiagnosisWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/bearing/bearing_cnn_baseline_best.pt"
BALL_FILE = PROJECT_ROOT / "datasets/bearing/cwru/raw/ball/B007_3.mat"


class BearingDiagnosisWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = BearingDiagnosisWorkflow.from_checkpoint(CHECKPOINT, device="cpu")

    def test_run_signal_returns_complete_diagnosis(self) -> None:
        result = self.workflow.run_signal(np.ones(1024 * 2, dtype=np.float32))

        self.assertEqual(result.window_count, 2)
        self.assertEqual(sum(result.class_counts.values()), 2)
        self.assertEqual(len(result.windows), 2)
        self.assertIsNone(result.source_path)
        self.assertIsNone(result.sensor_key)

    def test_run_file_uses_existing_mat_and_reports_sensor(self) -> None:
        result = self.workflow.run_file(BALL_FILE, sensor_key="X121_DE_time")

        self.assertEqual(result.source_path, str(BALL_FILE.resolve()))
        self.assertEqual(result.sensor_key, "X121_DE_time")
        self.assertEqual(result.predicted_label, "ball")
        self.assertEqual(result.window_count, 118)

    def test_generic_run_dispatches_to_file_or_signal(self) -> None:
        signal_result = self.workflow.run(np.zeros(1024, dtype=np.float32))
        file_result = self.workflow.run(BALL_FILE, sensor_key="X121_DE_time")

        self.assertIsNone(signal_result.source_path)
        self.assertEqual(file_result.sensor_key, "X121_DE_time")

    def test_v2_workflow_routes_three_quality_states(self) -> None:
        phase = np.linspace(
            0.0, 16.0 * np.pi, self.workflow.adapter.predictor.window_size,
            endpoint=False,
        )
        accepted = (np.sin(phase) + 0.2 * np.sin(3.0 * phase)).astype(np.float32)
        caution = np.random.default_rng(20260903).normal(
            0.0, 1.0, self.workflow.adapter.predictor.window_size
        ).astype(np.float32)
        rejected = np.full(
            self.workflow.adapter.predictor.window_size, 0.25, dtype=np.float32
        )
        thresholds = BearingQualityThresholdsV2(
            warning_clipping_ratio=1.0,
            reject_clipping_ratio=1.0,
            warning_spectral_flatness=0.20,
            reject_spectral_flatness=0.99,
        )

        rows = self.workflow.run_signal_with_quality_v2(
            np.concatenate([accepted, caution, rejected]),
            thresholds=thresholds,
        )

        self.assertEqual([row["status"] for row in rows], [
            "accepted", "caution", "rejected"
        ])
        self.assertIsNotNone(rows[0]["prediction"])
        self.assertIsNotNone(rows[1]["prediction"])
        self.assertIsNone(rows[2]["prediction"])

    def test_v2_all_rejected_has_no_fabricated_diagnosis(self) -> None:
        rows = self.workflow.run_signal_with_quality_v2(
            np.full(1024 * 2, 0.25, dtype=np.float32)
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "rejected" for row in rows))
        self.assertTrue(all(row["prediction"] is None for row in rows))

    def test_v2_generic_dispatches_file_and_signal(self) -> None:
        signal_rows = self.workflow.run_with_quality_v2(
            np.full(1024, 0.25, dtype=np.float32)
        )
        file_rows = self.workflow.run_with_quality_v2(
            BALL_FILE, sensor_key="X121_DE_time"
        )

        self.assertEqual(len(signal_rows), 1)
        self.assertEqual(len(file_rows), 118)

    def test_v2_sensor_key_is_rejected_for_in_memory_signal(self) -> None:
        with self.assertRaises(TypeError):
            self.workflow.run_with_quality_v2(
                np.zeros(1024, dtype=np.float32), sensor_key="X121_DE_time"
            )

    def test_v2_json_rendering_is_strict_and_does_not_write(self) -> None:
        rows = self.workflow.run_signal_with_quality_v2(
            np.full(1024, 0.25, dtype=np.float32)
        )
        rendered = self.workflow.render_quality_json_v2(rows)
        payload = json.loads(rendered)

        self.assertEqual(payload[0]["status"], "rejected")
        self.assertIsNone(payload[0]["prediction"])

    def test_json_rendering_is_serializable_and_does_not_write(self) -> None:
        result = self.workflow.run_signal(np.zeros(1024, dtype=np.float32))
        rendered = self.workflow.render_json(result)
        payload = json.loads(rendered)

        self.assertEqual(payload["window_count"], 1)
        self.assertIsInstance(payload["windows"], list)

    def test_short_signal_raises_clear_inference_error(self) -> None:
        with self.assertRaises(InsufficientSignalError):
            self.workflow.run_signal(np.zeros(1023, dtype=np.float32))

    def test_sensor_key_is_rejected_for_in_memory_signal(self) -> None:
        with self.assertRaises(TypeError):
            self.workflow.run(np.zeros(1024, dtype=np.float32), sensor_key="X121_DE_time")


if __name__ == "__main__":
    unittest.main()
