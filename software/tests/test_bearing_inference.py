"""Regression tests for the frozen bearing inference interface."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from airwatch.analysis import BearingQualityThresholds, BearingQualityThresholdsV2
from airwatch.data import CWRUBearingDataset
from airwatch.inference import (
    BearingInferenceError,
    BearingPredictor,
    InsufficientSignalError,
    QualityAwareWindowPrediction,
    QualityAwareWindowPredictionV2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/bearing/bearing_cnn_baseline_best.pt"
MANIFEST = PROJECT_ROOT / "datasets/bearing/cwru/split-manifest.csv"
LABEL_MAP = PROJECT_ROOT / "datasets/bearing/cwru/label-map.json"
PREDICTIONS = PROJECT_ROOT / (
    "artifacts/evidence/bearing/bearing_cnn_baseline_test_predictions.csv"
)


class BearingInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = BearingPredictor(CHECKPOINT, device="cpu")

    def test_checkpoint_loads_model_contract(self) -> None:
        self.assertEqual(self.predictor.window_size, 1024)
        self.assertEqual(self.predictor.step, 1024)
        self.assertEqual(self.predictor.normalization, "none")
        self.assertEqual(self.predictor.num_classes, 4)
        self.assertEqual(
            self.predictor.label_map,
            {0: "normal", 1: "inner_race", 2: "ball", 3: "outer_race_6"},
        )
        self.assertFalse(self.predictor.model.training)

    def test_input_shape_and_finite_value_validation(self) -> None:
        with self.assertRaisesRegex(BearingInferenceError, "one-dimensional"):
            self.predictor.predict_signal(np.zeros((2, 1024), dtype=np.float32))
        with self.assertRaisesRegex(BearingInferenceError, "NaN"):
            invalid = np.zeros(1024, dtype=np.float32)
            invalid[0] = np.nan
            self.predictor.predict_signal(invalid)
        with self.assertRaisesRegex(BearingInferenceError, "real-valued"):
            self.predictor.predict_signal(np.ones(1024, dtype=np.complex64))

    def test_short_signal_fails_with_explicit_message(self) -> None:
        with self.assertRaisesRegex(InsufficientSignalError, "one window requires 1024"):
            self.predictor.predict_signal(np.zeros(1023, dtype=np.float32))

    def test_prediction_output_is_model_derived_and_well_formed(self) -> None:
        rng = np.random.default_rng(20260903)
        result = self.predictor.predict_signal(
            rng.normal(size=self.predictor.window_size).astype(np.float32)
        )
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["window_index"], 0)
        self.assertIn(row["predicted_label"], self.predictor.label_map.values())
        self.assertIn(row["predicted_class"], self.predictor.label_map)
        self.assertGreaterEqual(row["confidence"], 0.0)
        self.assertLessEqual(row["confidence"], 1.0)
        self.assertEqual(len(row["probabilities"]), 4)
        self.assertAlmostEqual(sum(row["probabilities"]), 1.0, places=5)

    def test_window_zscore_inference_is_invariant_to_gain_and_offset(self) -> None:
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        payload["config"] = dict(payload["config"])
        payload["config"]["dataset"] = dict(payload["config"]["dataset"])
        payload["config"]["dataset"]["normalization"] = "window_zscore"

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "window-zscore.pt"
            torch.save(payload, checkpoint)
            predictor = BearingPredictor(checkpoint, device="cpu", label_map_path=LABEL_MAP)

            base = np.linspace(-2.0, 3.0, predictor.window_size, dtype=np.float32)
            original = predictor.predict_signal(base)[0]
            transformed = predictor.predict_signal(base * 2.5 + 17.0)[0]

        np.testing.assert_allclose(
            original["probabilities"], transformed["probabilities"], atol=1e-5
        )
        self.assertEqual(original["predicted_class"], transformed["predicted_class"])

    def test_v2_routes_accepted_caution_and_rejected_windows(self) -> None:
        phase = np.linspace(
            0.0, 16.0 * np.pi, self.predictor.window_size, endpoint=False
        )
        accepted = (np.sin(phase) + 0.2 * np.sin(3.0 * phase)).astype(np.float32)
        caution = np.random.default_rng(20260903).normal(
            0.0, 1.0, self.predictor.window_size
        ).astype(np.float32)
        rejected = np.full(self.predictor.window_size, 0.25, dtype=np.float32)
        thresholds = BearingQualityThresholdsV2(
            warning_clipping_ratio=1.0,
            reject_clipping_ratio=1.0,
            warning_spectral_flatness=0.20,
            reject_spectral_flatness=0.99,
        )

        rows = self.predictor.predict_signal_with_quality_v2(
            np.concatenate([accepted, caution, rejected]),
            thresholds=thresholds,
        )

        self.assertEqual([row["status"] for row in rows], [
            "accepted", "caution", "rejected"
        ])
        self.assertIsNotNone(rows[0]["prediction"])
        self.assertIsNotNone(rows[1]["prediction"])
        self.assertTrue(rows[1]["accepted"])
        self.assertIsNone(rows[2]["prediction"])

    def test_v2_caution_enters_model_and_rejected_window_does_not(self) -> None:
        phase = np.linspace(
            0.0, 16.0 * np.pi, self.predictor.window_size, endpoint=False
        )
        accepted = (np.sin(phase) + 0.2 * np.sin(3.0 * phase)).astype(np.float32)
        caution = np.random.default_rng(20260903).normal(
            0.0, 1.0, self.predictor.window_size
        ).astype(np.float32)
        rejected = np.full(self.predictor.window_size, 0.25, dtype=np.float32)
        thresholds = BearingQualityThresholdsV2(
            warning_clipping_ratio=1.0,
            reject_clipping_ratio=1.0,
            warning_spectral_flatness=0.20,
            reject_spectral_flatness=0.99,
        )

        with mock.patch.object(
            self.predictor,
            "_predict_windows",
            wraps=self.predictor._predict_windows,
        ) as predict_windows:
            rows = self.predictor.predict_signal_with_quality_v2(
                np.concatenate([accepted, caution, rejected]),
                thresholds=thresholds,
            )

        predict_windows.assert_called_once()
        model_windows = predict_windows.call_args.args[0]
        self.assertEqual(model_windows.shape, (2, self.predictor.window_size))
        self.assertEqual(predict_windows.call_args.kwargs["window_indices"], [0, 1])
        self.assertEqual(rows[1]["status"], "caution")
        self.assertIsNotNone(rows[1]["prediction"])
        self.assertEqual(rows[2]["status"], "rejected")
        self.assertIsNone(rows[2]["prediction"])

    def test_v2_all_rejected_windows_never_call_model(self) -> None:
        rejected = np.full(self.predictor.window_size * 2, 0.25, dtype=np.float32)

        with mock.patch.object(self.predictor, "_predict_windows") as predict_windows:
            rows = self.predictor.predict_signal_with_quality_v2(rejected)

        predict_windows.assert_not_called()
        self.assertEqual([row["status"] for row in rows], ["rejected", "rejected"])
        self.assertTrue(all(row["prediction"] is None for row in rows))

    def test_v2_result_is_strict_json_serializable(self) -> None:
        phase = np.linspace(
            0.0, 16.0 * np.pi, self.predictor.window_size, endpoint=False
        )
        accepted = np.sin(phase).astype(np.float32)
        rejected = np.full(self.predictor.window_size, 0.25, dtype=np.float32)

        rows = self.predictor.predict_signal_with_quality_v2(
            np.concatenate([accepted, rejected])
        )
        encoded = json.dumps(rows, allow_nan=False, ensure_ascii=False)

        self.assertIn('"status": "accepted"', encoded)
        self.assertIn('"status": "rejected"', encoded)
        self.assertIn('"prediction": null', encoded)

    def test_v2_file_inference_reads_real_validation_record(self) -> None:
        with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
            validation_row = next(
                row for row in csv.DictReader(handle) if row["split"] == "validation"
            )
        path = PROJECT_ROOT / validation_row["local_path"]

        rows = self.predictor.predict_file_with_quality_v2(
            path,
            sensor_key=validation_row["sensor_key"],
        )

        self.assertEqual(len(rows), int(validation_row["window_count"]))
        self.assertEqual(rows[0]["window_index"], 0)
        self.assertEqual(
            set(rows[0]),
            {
                "window_index",
                "status",
                "accepted",
                "reason_code",
                "message",
                "quality_score",
                "quality_features",
                "prediction",
            },
        )

    def test_quality_aware_inference_rejects_before_window_zscore(self) -> None:
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        payload["config"] = dict(payload["config"])
        payload["config"]["dataset"] = dict(payload["config"]["dataset"])
        payload["config"]["dataset"]["normalization"] = "window_zscore"

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "window-zscore.pt"
            torch.save(payload, checkpoint)
            predictor = BearingPredictor(checkpoint, device="cpu", label_map_path=LABEL_MAP)

            rows = predictor.predict_signal_with_quality(
                np.full(predictor.window_size, 3.0, dtype=np.float32)
            )

        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["accepted"])
        self.assertEqual(rows[0]["reason_code"], "near_constant")
        self.assertIsNone(rows[0]["prediction"])

    def test_quality_aware_inference_handles_accepted_and_rejected_windows(self) -> None:
        phase = np.linspace(0.0, 16.0 * np.pi, self.predictor.window_size, endpoint=False)
        structured = (np.sin(phase) + 0.2 * np.sin(3.0 * phase)).astype(np.float32)
        mixed = np.concatenate(
            [np.zeros(self.predictor.window_size, dtype=np.float32), structured]
        )

        rows = self.predictor.predict_signal_with_quality(mixed)

        self.assertEqual([row["window_index"] for row in rows], [0, 1])
        self.assertFalse(rows[0]["accepted"])
        self.assertEqual(rows[0]["reason_code"], "near_constant")
        self.assertIsNone(rows[0]["prediction"])
        self.assertTrue(rows[1]["accepted"])
        self.assertEqual(rows[1]["reason_code"], "accepted")
        self.assertIsNotNone(rows[1]["prediction"])
        self.assertEqual(rows[1]["prediction"]["window_index"], 1)
        self.assertAlmostEqual(sum(rows[1]["prediction"]["probabilities"]), 1.0, places=5)

    def test_quality_aware_inference_uses_supplied_thresholds(self) -> None:
        phase = np.linspace(0.0, 16.0 * np.pi, self.predictor.window_size, endpoint=False)
        weak_signal = (0.01 * np.sin(phase)).astype(np.float32)

        row = self.predictor.predict_signal_with_quality(
            weak_signal,
            thresholds=BearingQualityThresholds(near_constant_std=0.02),
        )[0]

        self.assertFalse(row["accepted"])
        self.assertEqual(row["reason_code"], "near_constant")
        self.assertIsNone(row["prediction"])

    def test_quality_aware_prediction_type_is_publicly_exported(self) -> None:
        self.assertEqual(QualityAwareWindowPrediction.__name__, "QualityAwareWindowPrediction")

    def test_v2_quality_aware_prediction_type_is_publicly_exported(self) -> None:
        self.assertEqual(
            QualityAwareWindowPredictionV2.__name__,
            "QualityAwareWindowPredictionV2",
        )

    def test_non_finite_windows_are_rejected_without_blocking_other_windows(self) -> None:
        phase = np.linspace(0.0, 16.0 * np.pi, self.predictor.window_size, endpoint=False)
        structured = (np.sin(phase) + 0.2 * np.sin(3.0 * phase)).astype(np.float32)
        nan_window = structured.copy()
        nan_window[17] = np.nan
        inf_window = structured.copy()
        inf_window[29] = np.inf

        rows = self.predictor.predict_signal_with_quality(
            np.concatenate([nan_window, inf_window, structured])
        )

        self.assertEqual([row["window_index"] for row in rows], [0, 1, 2])
        for row in rows[:2]:
            self.assertFalse(row["accepted"])
            self.assertEqual(row["reason_code"], "invalid_signal")
            self.assertIsNone(row["prediction"])
        self.assertTrue(rows[2]["accepted"])
        self.assertIsNotNone(rows[2]["prediction"])
        self.assertEqual(rows[2]["prediction"]["window_index"], 2)

    def test_quality_aware_result_is_strict_json_serializable(self) -> None:
        phase = np.linspace(0.0, 16.0 * np.pi, self.predictor.window_size, endpoint=False)
        rows = self.predictor.predict_signal_with_quality(np.sin(phase).astype(np.float32))

        encoded = json.dumps(rows, allow_nan=False, ensure_ascii=False)

        self.assertIn('"accepted": true', encoded)
        self.assertIn('"prediction": {', encoded)

    def test_quality_aware_file_inference_reads_real_validation_record(self) -> None:
        with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
            validation_row = next(
                row for row in csv.DictReader(handle) if row["split"] == "validation"
            )
        path = PROJECT_ROOT / validation_row["local_path"]

        rows = self.predictor.predict_file_with_quality(
            path,
            sensor_key=validation_row["sensor_key"],
        )

        self.assertEqual(len(rows), int(validation_row["window_count"]))
        self.assertEqual(rows[0]["window_index"], 0)
        self.assertEqual(
            set(rows[0]),
            {
                "window_index",
                "accepted",
                "reason_code",
                "message",
                "quality_score",
                "quality_features",
                "prediction",
            },
        )

    def test_legacy_prediction_interface_remains_unchanged(self) -> None:
        phase = np.linspace(0.0, 16.0 * np.pi, self.predictor.window_size, endpoint=False)
        row = self.predictor.predict_signal(np.sin(phase).astype(np.float32))[0]

        self.assertEqual(
            set(row),
            {"window_index", "predicted_label", "predicted_class", "confidence", "probabilities"},
        )

    def test_fixed_test_predictions_match_training_evidence(self) -> None:
        dataset = CWRUBearingDataset(
            MANIFEST,
            split="test",
            label_map_path=LABEL_MAP,
            normalization="none",
            project_root=PROJECT_ROOT,
        )
        with PREDICTIONS.open("r", encoding="utf-8", newline="") as handle:
            expected = list(csv.DictReader(handle))
        self.assertEqual(len(expected), len(dataset))

        actual: list[int] = []
        actual_labels: list[str] = []
        for path in dataset.source_files:
            indices = [
                index
                for index in range(len(dataset))
                if dataset.sample_metadata(index).path == path
            ]
            self.assertTrue(indices)
            first = dataset.sample_metadata(indices[0])
            rows = self.predictor.predict_file(path, sensor_key=first.sensor_key)
            self.assertEqual(len(rows), len(indices))
            actual.extend(int(row["predicted_class"]) for row in rows)
            actual_labels.extend(str(row["predicted_label"]) for row in rows)

        expected_predictions = [int(row["prediction"]) for row in expected]
        self.assertEqual(actual, expected_predictions)
        self.assertTrue(
            all(label in self.predictor.label_map.values() for label in actual_labels)
        )


if __name__ == "__main__":
    unittest.main()


