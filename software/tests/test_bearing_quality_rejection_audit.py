from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from airwatch.analysis.bearing_quality import (
    assess_bearing_window,
    load_bearing_quality_thresholds,
)
from training.audit_bearing_quality_rejection import (
    HIGH_CONFIDENCE_THRESHOLD,
    apply_audit_condition,
    audit_conditions,
    build_raw_test_dataset,
    ensure_outputs_absent,
    output_paths,
    strict_json_dumps,
    summarize_condition,
    validate_calibration_payload,
    validate_quality_prediction_rows,
    write_outputs_atomically,
)


class QualityRejectionAuditTest(unittest.TestCase):
    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_dir = Path(temp_dir)
            paths = output_paths(evidence_dir, "audit_v1")
            for key in paths:
                with self.subTest(key=key):
                    for path in paths.values():
                        path.unlink(missing_ok=True)
                    paths[key].parent.mkdir(parents=True, exist_ok=True)
                    paths[key].write_text("historical evidence", encoding="utf-8")

                    with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                        ensure_outputs_absent(paths)

                    self.assertEqual(
                        paths[key].read_text(encoding="utf-8"),
                        "historical evidence",
                    )

    def test_invalid_and_clipped_conditions_are_rejected_without_mutating_input(self) -> None:
        thresholds = load_bearing_quality_thresholds(
            Path("artifacts/evidence/bearing/bearing_quality_calibration_20260903.json")
        )
        original = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
        original_copy = original.copy()
        expected_reasons = {
            "near_constant": "near_constant",
            "nan": "invalid_signal",
            "inf": "invalid_signal",
            "clipping": "clipped",
        }
        conditions = {item.condition_id: item for item in audit_conditions()}

        for condition_id, expected_reason in expected_reasons.items():
            with self.subTest(condition_id=condition_id):
                changed = apply_audit_condition(
                    original,
                    conditions[condition_id],
                    seed=20260903,
                    sample_index=7,
                )
                result = assess_bearing_window(changed, thresholds=thresholds)
                self.assertFalse(result.accepted)
                self.assertEqual(result.reason_code, expected_reason)

        np.testing.assert_array_equal(original, original_copy)

    def test_summary_keeps_coverage_separate_from_selective_accuracy(self) -> None:
        rows = [
            {"target": 0, "accepted": True, "reason_code": "accepted", "prediction": 0, "confidence": 0.99, "baseline_evaluable": True, "baseline_prediction": 0, "baseline_confidence": 0.99},
            {"target": 0, "accepted": False, "reason_code": "excessive_noise", "prediction": None, "confidence": None, "baseline_evaluable": True, "baseline_prediction": 1, "baseline_confidence": 0.95},
            {"target": 0, "accepted": True, "reason_code": "accepted", "prediction": 1, "confidence": 0.91, "baseline_evaluable": True, "baseline_prediction": 1, "baseline_confidence": 0.91},
            {"target": 0, "accepted": False, "reason_code": "clipped", "prediction": None, "confidence": None, "baseline_evaluable": True, "baseline_prediction": 0, "baseline_confidence": 0.98},
        ]

        summary = summarize_condition(rows, num_classes=4, high_confidence_threshold=0.9)

        self.assertEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["selective_accuracy"], 0.5)
        self.assertEqual(summary["baseline_accuracy"], 0.5)
        self.assertEqual(summary["baseline_errors_rejected_rate"], 0.5)
        self.assertEqual(summary["baseline_correct_rejected_rate"], 0.5)
        self.assertEqual(summary["high_confidence_baseline_error_count"], 2)
        self.assertEqual(summary["high_confidence_baseline_errors_rejected_count"], 1)
        self.assertEqual(summary["accepted_error_mean_confidence"], 0.91)
        self.assertEqual(summary["rejected_baseline_error_mean_confidence"], 0.95)

    def test_no_accepted_windows_uses_json_null_not_nan(self) -> None:
        rows = [
            {"target": 0, "accepted": False, "reason_code": "invalid_signal", "prediction": None, "confidence": None, "baseline_evaluable": False, "baseline_prediction": None, "baseline_confidence": None},
        ]
        summary = summarize_condition(rows, num_classes=4, high_confidence_threshold=0.9)

        self.assertIsNone(summary["baseline_accuracy"])
        self.assertIsNone(summary["selective_accuracy"])
        self.assertIsNone(summary["selective_macro_f1"])
        serialized = strict_json_dumps({"summary": summary})
        self.assertNotIn("NaN", serialized)
        self.assertNotIn("Infinity", serialized)
        self.assertIn('"selective_accuracy": null', serialized)
        with self.assertRaises(ValueError):
            strict_json_dumps({"forbidden": float("nan")})

    def test_awgn_condition_is_deterministic(self) -> None:
        original = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
        condition = next(item for item in audit_conditions() if item.condition_id == "awgn_0db")
        first = apply_audit_condition(original, condition, seed=20260903, sample_index=12)
        second = apply_audit_condition(original, condition, seed=20260903, sample_index=12)
        np.testing.assert_array_equal(first, second)

    def test_calibration_must_come_from_validation_split(self) -> None:
        valid = {"calibration_split": "validation", "selected_thresholds": {}}
        validate_calibration_payload(valid)
        with self.assertRaisesRegex(ValueError, "validation"):
            validate_calibration_payload({"calibration_split": "test"})

    def test_rejected_rows_cannot_contain_a_diagnosis_or_confidence(self) -> None:
        validate_quality_prediction_rows(
            [{"accepted": False, "prediction": None, "confidence": None}]
        )
        with self.assertRaisesRegex(ValueError, "rejected"):
            validate_quality_prediction_rows(
                [{"accepted": False, "prediction": 2, "confidence": 0.99}]
            )

    def test_atomic_write_failure_leaves_no_partial_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = output_paths(Path(temp_dir), "audit_v1")
            rows = [{"condition_id": "clean", "value": 1}]
            real_replace = __import__("os").replace
            calls = 0

            def fail_second_replace(source: object, destination: object) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated move failure")
                real_replace(source, destination)

            with self.assertRaisesRegex(OSError, "simulated"):
                write_outputs_atomically(
                    paths,
                    {"schema_version": 1},
                    rows,
                    rows,
                    replace_func=fail_second_replace,
                )
            self.assertTrue(all(not path.exists() for path in paths.values()))

    def test_dataset_contract_is_fixed_test_and_raw(self) -> None:
        config = {
            "dataset": {
                "manifest": "datasets/bearing/cwru/split-manifest.csv",
                "label_map": "datasets/bearing/cwru/label-map.json",
            }
        }
        fake_dataset = object()
        with mock.patch(
            "training.audit_bearing_quality_rejection.CWRUBearingDataset",
            return_value=fake_dataset,
        ) as constructor:
            result = build_raw_test_dataset(config, Path.cwd())
        self.assertIs(result, fake_dataset)
        self.assertEqual(constructor.call_args.kwargs["split"], "test")
        self.assertEqual(constructor.call_args.kwargs["normalization"], "none")

    def test_high_confidence_threshold_is_frozen(self) -> None:
        self.assertEqual(HIGH_CONFIDENCE_THRESHOLD, 0.9)


if __name__ == "__main__":
    unittest.main()
