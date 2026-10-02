"""Runtime-contract and CPU inference tests for the KU Leuven TCN candidate."""

from __future__ import annotations

import csv
from pathlib import Path
from threading import Event
import json
import tempfile
import unittest

import numpy as np

from airwatch.data import KULeuvenMaterializedDataset
from airwatch.inference import (
    DEFAULT_KU_LEUVEN_DEVELOPMENT_CONTRACT,
    KULeuvenInferenceError,
    KULeuvenKnownSourcePredictor,
    aggregate_single_recording_probabilities,
    load_ku_leuven_runtime_contract,
)
from airwatch.workflows import (
    DEFAULT_KU_LEUVEN_SOFTWARE_CONTRACT,
    KULeuvenRecognitionWorkflow,
    QualityStatus,
    RecognitionStatus,
    UAVInputInfo,
    prepare_uav_input,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"D:\AirWatch_Datasets\ku_leuven_drone_rf\prepared\known-iq-v1")
PREDICTIONS = ROOT / (
    "artifacts/evidence/uav/ku_leuven/local_development/"
    "ku_leuven_tcn_weighted_worst_noise_validation_robustness_v4_recording_predictions.csv"
)


class KULeuvenRuntimeContractTests(unittest.TestCase):
    def test_repository_contract_loads_and_preserves_claim_boundaries(self):
        contract = load_ku_leuven_runtime_contract()
        self.assertEqual(contract.release_tier, "development_only")
        self.assertEqual(contract.model_name, "DroneRFTCN")
        self.assertEqual(contract.training_seed, 20260910)
        self.assertEqual(contract.windows_per_recording, 32)
        self.assertEqual(contract.open_set_status, "not_evaluated")
        self.assertEqual(contract.label_map["dji_mini2_rc"], 2)
        self.assertTrue(contract.checkpoint_path.is_file())

    def test_modified_contract_is_rejected_before_resources_are_loaded(self):
        payload = json.loads(
            DEFAULT_KU_LEUVEN_DEVELOPMENT_CONTRACT.read_text(encoding="utf-8")
        )
        payload["model"]["version"] = "tampered"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(KULeuvenInferenceError, "摘要不一致"):
                load_ku_leuven_runtime_contract(path, project_root=ROOT)

    def test_accepted_v4_software_contract_loads(self):
        contract = load_ku_leuven_runtime_contract(
            DEFAULT_KU_LEUVEN_SOFTWARE_CONTRACT
        )
        self.assertEqual(contract.contract_id, "ku-leuven-known-source-runtime-dev-v2")
        self.assertEqual(
            contract.model_version,
            "ku-leuven-tcn-weighted-worst-noise-dev-20260910-v4",
        )
        self.assertEqual(contract.training_seed, 20260910)
        self.assertEqual(contract.open_set_status, "not_evaluated")

    def test_single_recording_aggregation_matches_frozen_mean_rule(self):
        rows = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=np.float64)
        actual = aggregate_single_recording_probabilities(
            rows, expected_window_count=2
        )
        np.testing.assert_allclose(actual, [0.5, 0.2, 0.3])
        with self.assertRaisesRegex(ValueError, "window count"):
            aggregate_single_recording_probabilities(rows, expected_window_count=3)


@unittest.skipUnless(
    DATASET_ROOT.is_dir()
    and DEFAULT_KU_LEUVEN_SOFTWARE_CONTRACT.is_file()
    and PREDICTIONS.is_file(),
    "local KU Leuven runtime evidence is unavailable",
)
class KULeuvenKnownSourcePredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_ku_leuven_runtime_contract(
            DEFAULT_KU_LEUVEN_SOFTWARE_CONTRACT
        )
        cls.predictor = KULeuvenKnownSourcePredictor(
            cls.contract, device="cpu", batch_size=8
        )
        cls.dataset = KULeuvenMaterializedDataset(
            DATASET_ROOT, split="validation", verify_hashes=True
        )
        cls.recording_id = cls.dataset.samples[0].recording_id
        cls.indices = [
            index
            for index, sample in enumerate(cls.dataset.samples)
            if sample.recording_id == cls.recording_id
        ]
        cls.windows = np.stack(
            [cls.dataset[index][0].numpy() for index in cls.indices]
        )

    @classmethod
    def tearDownClass(cls):
        cls.dataset.close()

    def test_rejects_incomplete_recording_before_model_inference(self):
        with self.assertRaisesRegex(KULeuvenInferenceError, "严格要求 32 个窗口"):
            self.predictor.predict_recording(self.windows[:-1])

    def test_cpu_prediction_matches_frozen_evaluation_artifact(self):
        progress = []
        result = self.predictor.predict_recording(
            self.windows, progress=lambda current, total: progress.append((current, total))
        )
        with PREDICTIONS.open("r", encoding="utf-8", newline="") as handle:
            expected = next(
                row
                for row in csv.DictReader(handle)
                if row["condition_id"] == "clean"
                and row["predictor_id"] == "tcn_seed20260910"
                and row["recording_id"] == self.recording_id
            )
        expected_probabilities = np.asarray(
            [
                expected["prob_frysky"],
                expected["prob_spektrum_dx4e"],
                expected["prob_dji_mini2_rc"],
            ],
            dtype=np.float64,
        )
        self.assertEqual(result.predicted_class, int(expected["prediction"]))
        np.testing.assert_allclose(
            result.probabilities, expected_probabilities, rtol=0, atol=1e-4
        )
        self.assertEqual(progress, [(8, 32), (16, 32), (24, 32), (32, 32)])
        self.assertEqual(result.open_set_status, "not_evaluated")
        self.assertIsNone(result.known_unknown)
        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_background_workflow_returns_caution_without_unknown_claim(self):
        workflow = KULeuvenRecognitionWorkflow(self.predictor)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "known-iq.npy"
            np.save(path, self.windows)
            prepared = prepare_uav_input(
                path,
                declared_sample_rate_hz=100_000_000,
            )
            progress = []
            result = workflow.as_background_backend(
                prepared, Event(), lambda current, total: progress.append((current, total))
            )
        self.assertEqual(result.quality_status, QualityStatus.CAUTION)
        self.assertEqual(result.recognition_status, RecognitionStatus.COMPLETED)
        self.assertIsNone(result.known_unknown)
        self.assertIsNotNone(result.label)
        self.assertEqual(result.window_count, 32)
        self.assertEqual(result.open_set_status, "not_evaluated")
        self.assertEqual(progress[-1], (32, 32))

    def test_background_workflow_rejects_wrong_declared_sample_rate(self):
        workflow = KULeuvenRecognitionWorkflow(self.predictor)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "known-iq.npy"
            np.save(path, self.windows)
            prepared = prepare_uav_input(
                path,
                declared_sample_rate_hz=40_000_000,
            )
            with self.assertRaisesRegex(KULeuvenInferenceError, "模型要求 100000000 Hz"):
                workflow.as_background_backend(prepared, Event(), lambda *_: None)

    def test_cancelled_file_workflow_stops_before_read(self):
        workflow = KULeuvenRecognitionWorkflow(self.predictor)
        cancelled = Event()
        cancelled.set()
        with self.assertRaises(InterruptedError):
            workflow.predict_file("missing.npy", cancelled=cancelled)


if __name__ == "__main__":
    unittest.main()
