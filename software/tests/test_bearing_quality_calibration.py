"""Tests for validation-only bearing signal-quality calibration."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from airwatch.analysis import BearingQualityThresholds
from training.calibrate_bearing_quality import (
    CALIBRATION_SEED,
    CalibrationError,
    calibrate_quality_features,
    write_calibration_outputs,
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
                    "clipping_ratio": 0.002,
                    "spectral_flatness": flatness,
                }
            )
    return records


class BearingQualityCalibrationTests(unittest.TestCase):
    def test_calibration_only_accepts_validation_split(self) -> None:
        for split in ("train", "test"):
            with self.subTest(split=split), self.assertRaisesRegex(
                CalibrationError, "validation"
            ):
                calibrate_quality_features(
                    _feature_records(), calibration_split=split
                )

    def test_calibration_is_deterministic_and_json_serializable(self) -> None:
        first = calibrate_quality_features(
            _feature_records(), calibration_split="validation"
        )
        second = calibrate_quality_features(
            _feature_records(), calibration_split="validation"
        )

        self.assertEqual(first, second)
        rendered = json.dumps(first, ensure_ascii=False, sort_keys=True)
        self.assertIn('"calibration_split": "validation"', rendered)
        self.assertEqual(first["seed"], CALIBRATION_SEED)

    def test_selected_thresholds_restore_quality_threshold_object(self) -> None:
        result = calibrate_quality_features(
            _feature_records(), calibration_split="validation"
        )

        thresholds = BearingQualityThresholds(**result["selected_thresholds"])

        self.assertEqual(asdict(thresholds), result["selected_thresholds"])

    def test_five_db_is_report_only_and_not_used_for_selection(self) -> None:
        result = calibrate_quality_features(
            _feature_records(), calibration_split="validation"
        )
        conditions = {
            row["condition_id"]: row for row in result["conditions"]
        }

        self.assertEqual(conditions["awgn_5db"]["role"], "gray_report_only")
        self.assertFalse(conditions["awgn_5db"]["used_for_selection"])
        self.assertNotIn(
            "awgn_5db", result["selection_rule"]["retain_minimum_accepted_ratio"]
        )
        self.assertNotIn(
            "awgn_5db", result["selection_rule"]["severe_noise_conditions"]
        )

    def test_calibration_does_not_invent_classification_metrics(self) -> None:
        result = calibrate_quality_features(
            _feature_records(), calibration_split="validation"
        )
        serialized_keys: list[str] = []

        def collect_keys(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    serialized_keys.append(str(key).lower())
                    collect_keys(child)
            elif isinstance(value, list):
                for child in value:
                    collect_keys(child)

        collect_keys(result)
        self.assertNotIn("accuracy", serialized_keys)
        self.assertNotIn("f1", serialized_keys)
        self.assertNotIn("macro_f1", serialized_keys)

    def test_existing_output_is_never_overwritten(self) -> None:
        result = calibrate_quality_features(
            _feature_records(), calibration_split="validation"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            json_path = output_dir / "run.json"
            csv_path = output_dir / "run.csv"
            json_path.write_text("keep me", encoding="utf-8")

            with self.assertRaisesRegex(CalibrationError, "overwrite"):
                write_calibration_outputs(
                    result,
                    json_path=json_path,
                    csv_path=csv_path,
                )

            self.assertEqual(json_path.read_text(encoding="utf-8"), "keep me")
            self.assertFalse(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
