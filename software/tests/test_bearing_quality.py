"""Tests for bearing signal-quality rejection before model inference."""

from __future__ import annotations

from dataclasses import asdict
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from airwatch.analysis.bearing_quality import (
    BearingQualityThresholds,
    assess_bearing_window,
    load_bearing_quality_thresholds,
)


class BearingQualityTests(unittest.TestCase):
    def test_near_constant_signal_is_rejected(self) -> None:
        signal = np.full(1024, 0.25, dtype=np.float32)

        result = assess_bearing_window(signal)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "near_constant")
        self.assertIn("重新采集", result.message)

    def test_thresholds_are_validated_and_json_serializable(self) -> None:
        thresholds = BearingQualityThresholds(
            near_constant_std=1e-6,
            max_clipping_ratio=0.1,
            max_spectral_flatness=0.5,
        )

        rendered = json.dumps(asdict(thresholds), sort_keys=True)

        self.assertIn('"near_constant_std": 1e-06', rendered)
        invalid_values = (
            {"near_constant_std": -1.0},
            {"near_constant_std": np.inf},
            {"max_clipping_ratio": -0.01},
            {"max_clipping_ratio": 1.01},
            {"max_spectral_flatness": np.nan},
            {"max_spectral_flatness": 1.01},
        )
        for overrides in invalid_values:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                BearingQualityThresholds(**overrides)

    def test_custom_thresholds_control_rejection_and_features_include_rms(self) -> None:
        phase = np.linspace(0.0, 16.0 * np.pi, 1024, endpoint=False)
        signal = (0.01 * np.sin(phase)).astype(np.float32)

        accepted = assess_bearing_window(signal)
        rejected = assess_bearing_window(
            signal,
            thresholds=BearingQualityThresholds(near_constant_std=0.02),
        )

        self.assertTrue(accepted.accepted)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason_code, "near_constant")
        self.assertAlmostEqual(
            accepted.features["rms"],
            float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))),
        )

    def test_obviously_clipped_signal_is_rejected(self) -> None:
        phase = np.linspace(0.0, 8.0 * np.pi, 1024, endpoint=False)
        signal = np.clip(np.sin(phase), -0.35, 0.35).astype(np.float32)

        result = assess_bearing_window(signal)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "clipped")
        self.assertGreater(result.features["clipping_ratio"], 0.08)

    def test_white_noise_is_rejected_as_excessive_noise(self) -> None:
        rng = np.random.default_rng(20260903)
        signal = rng.normal(0.0, 1.0, 1024).astype(np.float32)

        result = assess_bearing_window(signal)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "excessive_noise")
        self.assertGreater(result.features["spectral_flatness"], 0.45)

    def test_structured_vibration_signal_is_accepted(self) -> None:
        phase = np.linspace(0.0, 16.0 * np.pi, 1024, endpoint=False)
        signal = (np.sin(phase) + 0.2 * np.sin(3.0 * phase)).astype(np.float32)

        result = assess_bearing_window(signal)

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason_code, "accepted")
        self.assertEqual(result.message, "信号质量检查通过")
        self.assertGreater(result.quality_score, 0.0)

    def test_nonfinite_signal_is_rejected_as_invalid(self) -> None:
        for bad_value in (np.nan, np.inf):
            with self.subTest(bad_value=bad_value):
                signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 1024)).astype(np.float32)
                signal[100] = bad_value

                result = assess_bearing_window(signal)

                self.assertFalse(result.accepted)
                self.assertEqual(result.reason_code, "invalid_signal")
                self.assertIn("重新采集", result.message)

    def test_empty_or_non_vector_signal_is_rejected_as_invalid(self) -> None:
        for signal in (np.array([], dtype=np.float32), np.ones((2, 512), dtype=np.float32)):
            with self.subTest(shape=signal.shape):
                result = assess_bearing_window(signal)

                self.assertFalse(result.accepted)
                self.assertEqual(result.reason_code, "invalid_signal")


class BearingQualityThresholdLoadingTests(unittest.TestCase):
    def _write_artifact(self, payload: object) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "quality-calibration.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_thresholds_from_validation_artifact(self) -> None:
        path = self._write_artifact(
            {
                "calibration_split": "validation",
                "selected_thresholds": {
                    "near_constant_std": 0.001,
                    "max_clipping_ratio": 0.003,
                    "max_spectral_flatness": 0.325,
                },
            }
        )

        thresholds = load_bearing_quality_thresholds(path)

        self.assertEqual(thresholds.near_constant_std, 0.001)
        self.assertEqual(thresholds.max_clipping_ratio, 0.003)
        self.assertEqual(thresholds.max_spectral_flatness, 0.325)

    def test_rejects_thresholds_derived_from_test_split(self) -> None:
        path = self._write_artifact(
            {
                "calibration_split": "test",
                "selected_thresholds": {
                    "near_constant_std": 0.001,
                    "max_clipping_ratio": 0.003,
                    "max_spectral_flatness": 0.325,
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "calibration_split='validation'"):
            load_bearing_quality_thresholds(path)

    def test_rejects_missing_threshold_field(self) -> None:
        path = self._write_artifact(
            {
                "calibration_split": "validation",
                "selected_thresholds": {
                    "near_constant_std": 0.001,
                    "max_clipping_ratio": 0.003,
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "max_spectral_flatness"):
            load_bearing_quality_thresholds(path)

    def test_rejects_non_numeric_threshold(self) -> None:
        path = self._write_artifact(
            {
                "calibration_split": "validation",
                "selected_thresholds": {
                    "near_constant_std": 0.001,
                    "max_clipping_ratio": True,
                    "max_spectral_flatness": 0.325,
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "max_clipping_ratio must be a number"):
            load_bearing_quality_thresholds(path)


if __name__ == "__main__":
    unittest.main()
