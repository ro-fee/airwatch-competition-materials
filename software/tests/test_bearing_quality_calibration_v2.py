"""Tests for validation-only three-level bearing quality calibration (V2)."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from airwatch.analysis import BearingQualityThresholdsV2
from training.calibrate_bearing_quality_v2 import (
    CALIBRATION_SEED,
    CalibrationV2Error,
    calibrate_quality_features_v2,
    write_calibration_outputs_v2,
)


def _feature_records() -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    values = {
        "clean": [0.03, 0.04, 0.05, 0.06],
        "awgn_20db": [0.05, 0.06, 0.07, 0.08],
        "awgn_10db": [0.12, 0.14, 0.16, 0.18],
        "awgn_5db": [0.22, 0.24, 0.26, 0.28],
        "awgn_0db": [0.34, 0.36, 0.38, 0.40],
        "awgn_minus5db": [0.46, 0.48, 0.50, 0.52],
    }
    for condition_id, flatness_values in values.items():
        for flatness in flatness_values:
            records.append(
                {
                    "condition_id": condition_id,
                    "standard_deviation": 0.05,
                    "clipping_ratio": 0.001,
                    "spectral_flatness": flatness,
                }
            )
    return records


class BearingQualityCalibrationV2Tests(unittest.TestCase):
    def test_calibration_only_accepts_validation_split(self) -> None:
        for split in ("train", "test"):
            with self.subTest(split=split), self.assertRaisesRegex(
                CalibrationV2Error, "validation"
            ):
                calibrate_quality_features_v2(
                    _feature_records(), calibration_split=split
                )

    def test_calibration_is_deterministic_and_strict_json_serializable(self) -> None:
        first = calibrate_quality_features_v2(
            _feature_records(), calibration_split="validation"
        )
        second = calibrate_quality_features_v2(
            _feature_records(), calibration_split="validation"
        )

        self.assertEqual(first, second)
        rendered = json.dumps(
            first, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        self.assertIn('"calibration_split": "validation"', rendered)
        self.assertEqual(first["seed"], CALIBRATION_SEED)

    def test_selected_thresholds_restore_v2_threshold_object(self) -> None:
        result = calibrate_quality_features_v2(
            _feature_records(), calibration_split="validation"
        )

        thresholds = BearingQualityThresholdsV2(**result["selected_thresholds"])

        self.assertEqual(asdict(thresholds), result["selected_thresholds"])
        self.assertLessEqual(
            thresholds.warning_clipping_ratio,
            thresholds.reject_clipping_ratio,
        )
        self.assertLessEqual(
            thresholds.warning_spectral_flatness,
            thresholds.reject_spectral_flatness,
        )

    def test_evidence_contains_all_candidates_and_three_state_statistics(self) -> None:
        result = calibrate_quality_features_v2(
            _feature_records(), calibration_split="validation"
        )

        expected = {
            "near_constant_std",
            "warning_clipping_ratio",
            "reject_clipping_ratio",
            "warning_spectral_flatness",
            "reject_spectral_flatness",
        }
        self.assertEqual(set(result["candidate_thresholds"]), expected)
        self.assertEqual(set(result["candidate_evaluations"]), expected)
        for field in expected:
            self.assertEqual(
                len(result["candidate_thresholds"][field]),
                len(result["candidate_evaluations"][field]),
            )
            for candidate in result["candidate_evaluations"][field]:
                self.assertIn("selection_reason", candidate)
                self.assertIn("selected", candidate)
                self.assertTrue(candidate["conditions"])
                for condition in candidate["conditions"]:
                    self.assertEqual(
                        set(condition["status_counts"]),
                        {"accepted", "caution", "rejected"},
                    )
                    self.assertEqual(
                        set(condition["status_ratios"]),
                        {"accepted", "caution", "rejected"},
                    )
                    self.assertEqual(
                        sum(condition["status_counts"].values()),
                        condition["window_count"],
                    )
                    self.assertAlmostEqual(
                        sum(condition["status_ratios"].values()), 1.0
                    )

    def test_condition_report_includes_clean_and_all_controlled_noise_levels(self) -> None:
        result = calibrate_quality_features_v2(
            _feature_records(), calibration_split="validation"
        )
        conditions = {row["condition_id"]: row for row in result["conditions"]}

        self.assertEqual(
            set(conditions),
            {
                "clean",
                "awgn_20db",
                "awgn_10db",
                "awgn_5db",
                "awgn_0db",
                "awgn_minus5db",
            },
        )
        self.assertFalse(conditions["awgn_0db"]["used_for_selection"])
        for condition in conditions.values():
            self.assertEqual(
                sum(condition["status_counts"].values()),
                condition["window_count"],
            )
            self.assertIn("feature_distributions", condition)

    def test_constant_nan_and_inf_are_always_rejected(self) -> None:
        result = calibrate_quality_features_v2(
            _feature_records(), calibration_split="validation"
        )
        checks = {row["condition_id"]: row for row in result["hard_error_checks"]}

        self.assertEqual(set(checks), {"constant", "nan", "inf"})
        for row in checks.values():
            self.assertEqual(row["status"], "rejected")
            self.assertEqual(row["rejected_ratio"], 1.0)

    def test_calibration_does_not_invent_classification_metrics(self) -> None:
        result = calibrate_quality_features_v2(
            _feature_records(), calibration_split="validation"
        )
        keys: list[str] = []

        def collect(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    keys.append(str(key).lower())
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(result)
        self.assertNotIn("accuracy", keys)
        self.assertNotIn("f1", keys)
        self.assertNotIn("macro_f1", keys)

    def test_existing_output_is_never_overwritten(self) -> None:
        result = calibrate_quality_features_v2(
            _feature_records(), calibration_split="validation"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            json_path = output_dir / "run.json"
            csv_path = output_dir / "run.csv"
            json_path.write_text("keep me", encoding="utf-8")

            with self.assertRaisesRegex(CalibrationV2Error, "overwrite"):
                write_calibration_outputs_v2(
                    result,
                    json_path=json_path,
                    csv_path=csv_path,
                )

            self.assertEqual(json_path.read_text(encoding="utf-8"), "keep me")
            self.assertFalse(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
