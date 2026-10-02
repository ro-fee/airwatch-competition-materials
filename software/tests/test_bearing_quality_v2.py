"""Tests for the three-level bearing signal-quality gate (V2)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from airwatch.analysis import (
    BearingQualityThresholdsV2,
    assess_bearing_window_v2,
    load_bearing_quality_thresholds_v2,
)


class BearingQualityV2Tests(unittest.TestCase):
    def test_structured_signal_is_accepted(self) -> None:
        phase = np.linspace(0.0, 16.0 * np.pi, 1024, endpoint=False)
        signal = (np.sin(phase) + 0.2 * np.sin(3.0 * phase)).astype(np.float32)

        result = assess_bearing_window_v2(signal)

        self.assertEqual(result.status, "accepted")
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason_code, "accepted")

    def test_moderate_clipping_returns_caution(self) -> None:
        phase = np.linspace(0.0, 16.0 * np.pi, 1024, endpoint=False)
        signal = np.sin(phase).astype(np.float32)
        signal[:30] = -1.0
        signal[30:60] = 1.0
        thresholds = BearingQualityThresholdsV2(
            warning_clipping_ratio=0.04,
            reject_clipping_ratio=0.08,
            warning_spectral_flatness=1.0,
            reject_spectral_flatness=1.0,
        )

        result = assess_bearing_window_v2(signal, thresholds=thresholds)

        self.assertEqual(result.status, "caution")
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason_code, "clipping_warning")
        self.assertIn("仅供参考", result.message)
        self.assertGreater(result.features["clipping_ratio"], 0.04)
        self.assertLessEqual(result.features["clipping_ratio"], 0.08)

    def test_moderate_spectral_flatness_returns_caution(self) -> None:
        rng = np.random.default_rng(20260903)
        signal = rng.normal(0.0, 1.0, 1024).astype(np.float32)
        thresholds = BearingQualityThresholdsV2(
            warning_clipping_ratio=1.0,
            reject_clipping_ratio=1.0,
            warning_spectral_flatness=0.20,
            reject_spectral_flatness=0.99,
        )

        result = assess_bearing_window_v2(signal, thresholds=thresholds)

        self.assertEqual(result.status, "caution")
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason_code, "noise_warning")
        self.assertIn("仅供参考", result.message)
        self.assertGreater(result.features["spectral_flatness"], 0.20)
        self.assertLessEqual(result.features["spectral_flatness"], 0.99)

    def test_hard_failures_are_rejected(self) -> None:
        cases = {
            "empty": np.array([], dtype=np.float32),
            "two_dimensional": np.zeros((2, 8), dtype=np.float32),
            "nan": np.array([0.0, np.nan, 1.0], dtype=np.float32),
            "infinite": np.array([0.0, np.inf, 1.0], dtype=np.float32),
            "near_constant": np.full(1024, 0.25, dtype=np.float32),
        }

        for name, signal in cases.items():
            with self.subTest(name=name):
                result = assess_bearing_window_v2(signal)
                self.assertEqual(result.status, "rejected")
                self.assertFalse(result.accepted)

    def test_clipping_above_reject_limit_is_rejected(self) -> None:
        phase = np.linspace(0.0, 16.0 * np.pi, 1024, endpoint=False)
        signal = np.sin(phase).astype(np.float32)
        signal[:60] = -1.0
        signal[60:120] = 1.0
        thresholds = BearingQualityThresholdsV2(
            warning_clipping_ratio=0.04,
            reject_clipping_ratio=0.08,
            warning_spectral_flatness=1.0,
            reject_spectral_flatness=1.0,
        )

        result = assess_bearing_window_v2(signal, thresholds=thresholds)

        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "clipped")

    def test_noise_above_reject_limit_is_rejected(self) -> None:
        signal = np.random.default_rng(20260903).normal(0.0, 1.0, 4096).astype(np.float32)
        thresholds = BearingQualityThresholdsV2(
            warning_clipping_ratio=1.0,
            reject_clipping_ratio=1.0,
            warning_spectral_flatness=0.20,
            reject_spectral_flatness=0.30,
        )

        result = assess_bearing_window_v2(signal, thresholds=thresholds)

        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "excessive_noise")


class BearingQualityThresholdLoadingV2Tests(unittest.TestCase):
    def _write_artifact(self, payload: object) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "quality-calibration-v2.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_v2_thresholds_from_validation_artifact(self) -> None:
        path = self._write_artifact(
            {
                "calibration_split": "validation",
                "selected_thresholds": {
                    "near_constant_std": 0.001,
                    "warning_clipping_ratio": 0.02,
                    "reject_clipping_ratio": 0.08,
                    "warning_spectral_flatness": 0.30,
                    "reject_spectral_flatness": 0.45,
                },
            }
        )

        thresholds = load_bearing_quality_thresholds_v2(path)

        self.assertEqual(thresholds.near_constant_std, 0.001)
        self.assertEqual(thresholds.warning_clipping_ratio, 0.02)
        self.assertEqual(thresholds.reject_clipping_ratio, 0.08)
        self.assertEqual(thresholds.warning_spectral_flatness, 0.30)
        self.assertEqual(thresholds.reject_spectral_flatness, 0.45)


    def test_rejects_non_validation_artifact(self) -> None:
        path = self._write_artifact(
            {
                "calibration_split": "test",
                "selected_thresholds": {
                    "near_constant_std": 0.001,
                    "warning_clipping_ratio": 0.02,
                    "reject_clipping_ratio": 0.08,
                    "warning_spectral_flatness": 0.30,
                    "reject_spectral_flatness": 0.45,
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "validation"):
            load_bearing_quality_thresholds_v2(path)

    def test_rejects_missing_or_non_numeric_thresholds(self) -> None:
        valid = {
            "near_constant_std": 0.001,
            "warning_clipping_ratio": 0.02,
            "reject_clipping_ratio": 0.08,
            "warning_spectral_flatness": 0.30,
            "reject_spectral_flatness": 0.45,
        }
        cases = {
            "missing": {key: value for key, value in valid.items() if key != "reject_clipping_ratio"},
            "string": {**valid, "warning_spectral_flatness": "0.30"},
            "boolean": {**valid, "warning_clipping_ratio": True},
        }

        for name, thresholds in cases.items():
            with self.subTest(name=name):
                path = self._write_artifact(
                    {"calibration_split": "validation", "selected_thresholds": thresholds}
                )
                with self.assertRaises(ValueError):
                    load_bearing_quality_thresholds_v2(path)

    def test_rejects_invalid_warning_reject_order(self) -> None:
        path = self._write_artifact(
            {
                "calibration_split": "validation",
                "selected_thresholds": {
                    "near_constant_std": 0.001,
                    "warning_clipping_ratio": 0.09,
                    "reject_clipping_ratio": 0.08,
                    "warning_spectral_flatness": 0.30,
                    "reject_spectral_flatness": 0.45,
                },
            }
        )

        with self.assertRaises(ValueError):
            load_bearing_quality_thresholds_v2(path)


if __name__ == "__main__":
    unittest.main()




