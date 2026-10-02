"""Regression tests for the UI-independent bearing inference adapter."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from airwatch.analysis import BearingQualityThresholdsV2
from airwatch.inference import BearingInferenceAdapter, BearingInferenceResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/bearing/bearing_cnn_baseline_best.pt"


class BearingInferenceAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = BearingInferenceAdapter.from_checkpoint(CHECKPOINT, device="cpu")

    def test_signal_result_has_stable_summary_and_window_details(self) -> None:
        signal = np.zeros(1024 * 2, dtype=np.float32)
        result = self.adapter.predict_signal(signal)

        self.assertIsInstance(result, BearingInferenceResult)
        self.assertIsNone(result.source_path)
        self.assertIsNone(result.sensor_key)
        self.assertEqual(result.window_count, 2)
        self.assertEqual(result.window_size, 1024)
        self.assertEqual(result.step, 1024)
        self.assertEqual(sum(result.class_counts.values()), result.window_count)
        self.assertIn(result.predicted_label, self.adapter.predictor.label_map.values())
        self.assertEqual(len(result.windows), 2)

    def test_v2_returns_window_rows_without_fabricating_a_summary(self) -> None:
        phase = np.linspace(
            0.0, 16.0 * np.pi, self.adapter.predictor.window_size, endpoint=False
        )
        caution = np.random.default_rng(20260903).normal(
            0.0, 1.0, self.adapter.predictor.window_size
        ).astype(np.float32)
        rejected = np.full(
            self.adapter.predictor.window_size, 0.25, dtype=np.float32
        )
        thresholds = BearingQualityThresholdsV2(
            warning_clipping_ratio=1.0,
            reject_clipping_ratio=1.0,
            warning_spectral_flatness=0.20,
            reject_spectral_flatness=0.99,
        )

        rows = self.adapter.predict_signal_with_quality_v2(
            np.concatenate([np.sin(phase).astype(np.float32), caution, rejected]),
            thresholds=thresholds,
        )

        self.assertIsInstance(rows, list)
        self.assertEqual([row["status"] for row in rows], [
            "accepted", "caution", "rejected"
        ])
        self.assertIsNotNone(rows[1]["prediction"])
        self.assertIsNone(rows[2]["prediction"])

    def test_v2_file_adapter_reads_file_without_creating_summary(self) -> None:
        path = PROJECT_ROOT / "datasets/bearing/cwru/raw/ball/B007_3.mat"

        rows = self.adapter.predict_file_with_quality_v2(
            path, sensor_key="X121_DE_time"
        )

        self.assertEqual(len(rows), 118)
        self.assertIn(rows[0]["status"], {"accepted", "caution", "rejected"})

    def test_generic_dispatch_accepts_numpy_signal(self) -> None:
        result = self.adapter.predict(np.ones(1024, dtype=np.float32))
        self.assertEqual(result.window_count, 1)
        self.assertIsNone(result.source_path)

    def test_result_is_json_friendly(self) -> None:
        result = self.adapter.predict_signal(np.ones(1024, dtype=np.float32))
        payload = result.to_dict()

        self.assertIsInstance(payload["windows"], list)
        self.assertIsInstance(payload["class_counts"], dict)
        self.assertIsInstance(payload["windows"][0]["probabilities"], list)
        self.assertEqual(payload["window_count"], 1)

    def test_file_dispatch_reports_source_and_sensor(self) -> None:
        path = PROJECT_ROOT / "datasets/bearing/cwru/raw/ball/B007_3.mat"
        result = self.adapter.predict_file(path, sensor_key="X121_DE_time")

        self.assertEqual(result.source_path, str(path.resolve()))
        self.assertEqual(result.sensor_key, "X121_DE_time")
        self.assertEqual(result.window_count, 118)
        self.assertEqual(result.predicted_label, "ball")


if __name__ == "__main__":
    unittest.main()
