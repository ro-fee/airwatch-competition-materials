"""Regression checks for the frozen anti-noise backend acceptance run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import uuid

import numpy as np

from airwatch.analysis import load_bearing_quality_thresholds_v2
from airwatch.inference import BearingPredictor
from airwatch.workflows import BearingDiagnosisWorkflow
from training.accept_bearing_inference import (
    _write_outputs,
    run_acceptance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT_ROOT / (
    "artifacts/evidence/bearing/anti_noise/"
    "bearing_cnn_anti_noise_20260904_frozen_contract.json"
)
CHECKPOINT = PROJECT_ROOT / (
    "artifacts/checkpoints/bearing/anti_noise/"
    "bearing_cnn_anti_noise_20260904_best.pt"
)
LABEL_MAP = PROJECT_ROOT / "datasets/bearing/cwru/label-map.json"
CALIBRATION = PROJECT_ROOT / (
    "artifacts/evidence/bearing/bearing_quality_calibration_20260903_v2.json"
)
ACCEPTANCE_JSON = PROJECT_ROOT / (
    "artifacts/evidence/bearing/anti_noise/"
    "bearing_cnn_anti_noise_20260904_inference_acceptance.json"
)
ACCEPTANCE_CSV = PROJECT_ROOT / (
    "artifacts/evidence/bearing/anti_noise/"
    "bearing_cnn_anti_noise_20260904_inference_acceptance_files.csv"
)


class FrozenBearingInferenceAcceptanceTests(unittest.TestCase):
    def test_recorded_acceptance_evidence_is_strict_and_complete(self) -> None:
        payload = json.loads(ACCEPTANCE_JSON.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["contract"]["contract_id"],
            "bearing_cnn_anti_noise_20260904_v1",
        )
        self.assertEqual(payload["contract"]["checked_files"], 29)
        self.assertEqual(payload["model"]["normalization"], "window_zscore")
        self.assertEqual(payload["quality_gate"]["version"], "V2")
        self.assertEqual(payload["dataset"]["manifest_row_count"], 16)
        self.assertEqual(payload["results"]["file_count"], 16)
        self.assertEqual(payload["protected_inputs"]["changed_paths"], [])
        self.assertFalse(payload["safety_declarations"]["training_performed"])
        self.assertFalse(payload["safety_declarations"]["checkpoint_modified"])
        self.assertFalse(payload["safety_declarations"]["ui_modified"])

        with ACCEPTANCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {row["split"] for row in rows},
            {"train", "validation", "test"},
        )

    def test_recorded_test_files_match_frozen_label_contract(self) -> None:
        payload = json.loads(ACCEPTANCE_JSON.read_text(encoding="utf-8"))
        test_rows = [row for row in payload["files"] if row["split"] == "test"]

        self.assertEqual(len(test_rows), 4)
        self.assertTrue(all(row["quality_file_correct"] for row in test_rows))
        self.assertEqual(
            {
                row["original_filename"]: row["quality_majority_class"]
                for row in test_rows
            },
            {
                "B007_3.mat": "ball",
                "IR007_3.mat": "inner_race",
                "Normal_3.mat": "normal",
                "OR007@6_3.mat": "outer_race_6",
            },
        )

    def test_frozen_quality_gate_routes_all_three_states(self) -> None:
        workflow = BearingDiagnosisWorkflow.from_checkpoint(
            CHECKPOINT,
            device="cpu",
            label_map_path=LABEL_MAP,
        )
        thresholds = load_bearing_quality_thresholds_v2(CALIBRATION)
        window_size = workflow.adapter.predictor.window_size
        phase = np.linspace(0.0, 2.0 * np.pi, window_size, endpoint=False)
        accepted = np.sin(phase).astype(np.float32)
        caution = (
            np.sin(phase)
            + 0.5 * np.random.default_rng(20260904).normal(size=window_size)
        ).astype(np.float32)
        rejected = np.full(window_size, 0.25, dtype=np.float32)

        rows = workflow.run_signal_with_quality_v2(
            np.concatenate([accepted, caution, rejected]),
            thresholds=thresholds,
        )

        self.assertEqual(
            [row["status"] for row in rows],
            ["accepted", "caution", "rejected"],
        )
        self.assertIsNotNone(rows[0]["prediction"])
        self.assertIsNotNone(rows[1]["prediction"])
        self.assertIsNone(rows[2]["prediction"])

    def test_rejected_windows_do_not_call_the_model(self) -> None:
        predictor = BearingPredictor(
            CHECKPOINT,
            device="cpu",
            label_map_path=LABEL_MAP,
        )
        rejected = np.full(predictor.window_size * 2, 0.25, dtype=np.float32)

        with mock.patch.object(predictor, "_predict_windows") as predict_windows:
            rows = predictor.predict_signal_with_quality_v2(rejected)

        predict_windows.assert_not_called()
        self.assertEqual([row["status"] for row in rows], ["rejected", "rejected"])
        self.assertTrue(all(row["prediction"] is None for row in rows))

    def test_acceptance_outputs_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "acceptance.json"
            csv_path = Path(directory) / "acceptance.csv"
            json_path.write_text("original", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                _write_outputs(json_path, csv_path, {"passed": True}, [{"ok": True}])

            self.assertEqual(json_path.read_text(encoding="utf-8"), "original")
            self.assertFalse(csv_path.exists())

    def test_acceptance_command_can_run_under_a_new_non_overwriting_name(self) -> None:
        run_name = (
            "bearing_cnn_anti_noise_20260904_acceptance_test_"
            + uuid.uuid4().hex
        )
        output_dir = PROJECT_ROOT / "artifacts/evidence/bearing/anti_noise"
        json_path = output_dir / f"{run_name}.json"
        csv_path = output_dir / f"{run_name}_files.csv"
        try:
            result = run_acceptance(
                contract_path=CONTRACT,
                run_name=run_name,
                device_name="cpu",
            )
            self.assertEqual(result["results"]["file_count"], 16)
            self.assertEqual(
                result["results"]["test_file_accuracy_against_manifest_labels"],
                1.0,
            )
        finally:
            json_path.unlink(missing_ok=True)
            csv_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
