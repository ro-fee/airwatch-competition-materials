from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from threading import Event
import tempfile
import unittest

import numpy as np

from airwatch.inference.ku_leuven import (
    KULeuvenInferenceError,
    KULeuvenKnownSourcePredictor,
)
from airwatch.workflows.ku_leuven_recognition import KULeuvenRecognitionWorkflow
from airwatch.workflows.uav_contract import QualityStatus, RecognitionStatus
from airwatch.workflows.uav_intake import prepare_uav_input


class UAVIntakeWorkflowTests(unittest.TestCase):
    def predictor(self, calls):
        predictor = object.__new__(KULeuvenKnownSourcePredictor)
        predictor.contract = SimpleNamespace(
            sample_rate_hz=100_000_000,
            input_channels=2,
            windows_per_recording=2,
            window_samples=8,
            model_version="fake-known-source-v1",
            open_set_status="not_evaluated",
            contract_id="fake-contract-v1",
            limitations=("development only",),
        )

        def predict_recording(samples, *, cancelled=None, progress=None):
            calls.append(np.array(samples, copy=True))
            if progress is not None:
                progress(1, 2)
                progress(2, 2)
            return SimpleNamespace(
                display_label="测试已知源",
                confidence=0.75,
                model_version="fake-known-source-v1",
                elapsed_seconds=0.01,
                window_count=2,
                open_set_status="not_evaluated",
                contract_id="fake-contract-v1",
                limitations=("development only",),
            )

        predictor.predict_recording = predict_recording
        return predictor

    def test_preview_snapshot_is_reused_after_source_file_changes(self):
        calls = []
        workflow = KULeuvenRecognitionWorkflow(self.predictor(calls))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.npy"
            original = np.stack(
                (
                    np.arange(16, dtype=np.float32),
                    np.arange(16, dtype=np.float32) + 0.5,
                )
            )
            np.save(path, original)
            prepared = prepare_uav_input(
                path, declared_sample_rate_hz=100_000_000
            )
            np.save(path, np.zeros_like(original))
            progress = []
            result = workflow.as_background_backend(
                prepared,
                Event(),
                lambda current, total: progress.append((current, total)),
            )
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0], original)
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertEqual(result.window_count, 2)
        self.assertEqual(result.open_set_status, "not_evaluated")
        self.assertEqual(result.contract_id, "fake-contract-v1")
        self.assertEqual(result.quality_status, QualityStatus.CAUTION)
        self.assertEqual(result.recognition_status, RecognitionStatus.COMPLETED)

    def test_full_recording_quality_rejection_skips_predictor(self):
        calls = []
        workflow = KULeuvenRecognitionWorkflow(self.predictor(calls))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constant.npy"
            np.save(path, np.ones((2, 16), dtype=np.float32))
            prepared = prepare_uav_input(
                path, declared_sample_rate_hz=100_000_000
            )
            result = workflow.as_background_backend(prepared, Event(), lambda *_: None)
        self.assertEqual(calls, [])
        self.assertEqual(result.quality_status, QualityStatus.REJECTED)
        self.assertEqual(result.recognition_status, RecognitionStatus.NOT_RUN)
        self.assertFalse(result.can_publish_prediction)

    def test_sample_rate_must_be_explicit_and_match_contract(self):
        workflow = KULeuvenRecognitionWorkflow(self.predictor([]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.npy"
            samples = np.stack((np.arange(16), np.arange(16) + 1)).astype(np.float32)
            np.save(path, samples)
            missing = prepare_uav_input(path)
            wrong = missing.with_declared_sample_rate(40_000_000)
        with self.assertRaisesRegex(KULeuvenInferenceError, "必须填写真实采样率"):
            workflow.as_background_backend(missing, Event(), lambda *_: None)
        with self.assertRaisesRegex(KULeuvenInferenceError, "模型要求 100000000 Hz"):
            workflow.as_background_backend(wrong, Event(), lambda *_: None)


if __name__ == "__main__":
    unittest.main()
