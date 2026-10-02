"""Regression tests for batch bearing diagnosis and evidence aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from airwatch.inference import BearingInferenceResult
from airwatch.workflows import BearingBatchDiagnosisWorkflow, BearingBatchError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakePredictor:
    checkpoint_path = PROJECT_ROOT / "fake-checkpoint.pt"
    device = "cpu"
    window_size = 4
    step = 4
    label_map = {0: "normal", 1: "fault"}


class _FakeWorkflow:
    def __init__(self) -> None:
        self.adapter = type("Adapter", (), {"predictor": _FakePredictor()})()

    def run_file(self, path: Path, *, sensor_key: str) -> BearingInferenceResult:
        predicted_class = 1 if path.name.startswith("fault") else 0
        windows = tuple(
            {
                "window_index": index,
                "predicted_label": self.adapter.predictor.label_map[predicted_class],
                "predicted_class": predicted_class,
                "confidence": 0.9,
                "probabilities": [0.1, 0.9]
                if predicted_class == 1
                else [0.9, 0.1],
            }
            for index in range(2)
        )
        return BearingInferenceResult(
            source_path=str(path),
            sensor_key=sensor_key,
            device="cpu",
            checkpoint_path=str(self.adapter.predictor.checkpoint_path),
            window_size=4,
            step=4,
            window_count=2,
            predicted_label=self.adapter.predictor.label_map[predicted_class],
            predicted_class=predicted_class,
            mean_confidence=0.9,
            class_counts={
                self.adapter.predictor.label_map[predicted_class]: 2,
            },
            windows=windows,
        )


class BearingBatchWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = BearingBatchDiagnosisWorkflow(
            _FakeWorkflow(),
            project_root=PROJECT_ROOT,
        )

    def test_aggregates_file_and_window_metrics_from_predictions(self) -> None:
        result = self.workflow.run_rows(
            [
                {
                    "local_path": "data/normal.mat",
                    "original_filename": "normal.mat",
                    "class_name": "normal",
                    "label_index": "0",
                    "sensor_key": "DE",
                    "split": "test",
                    "split_group": "group_a",
                },
                {
                    "local_path": "data/fault.mat",
                    "original_filename": "fault.mat",
                    "class_name": "fault",
                    "label_index": "1",
                    "sensor_key": "DE",
                    "split": "test",
                    "split_group": "group_b",
                },
            ],
            split="test",
        )

        self.assertEqual(result.file_count, 2)
        self.assertEqual(result.window_count, 4)
        self.assertEqual(result.file_accuracy, 1.0)
        self.assertEqual(result.window_accuracy, 1.0)
        self.assertEqual(result.macro_f1, 1.0)
        self.assertEqual(result.confusion_matrix, ((1, 0), (0, 1)))
        self.assertEqual(result.files[0].window_correct, 2)
        self.assertEqual(result.to_dict()["files"][1]["sensor_key"], "DE")

    def test_max_files_keeps_manifest_order_and_filter(self) -> None:
        rows = [
            {
                "local_path": f"data/{name}.mat",
                "original_filename": f"{name}.mat",
                "class_name": "normal",
                "label_index": "0",
                "sensor_key": "DE",
                "split": split,
            }
            for name, split in (("normal", "train"), ("fault", "test"))
        ]
        result = self.workflow.run_rows(rows, split="test")
        self.assertEqual(result.file_count, 1)
        self.assertEqual(result.files[0].original_filename, "fault.mat")

    def test_refuses_ambiguous_sensor_key_in_manifest(self) -> None:
        with self.assertRaisesRegex(BearingBatchError, "refuses to guess"):
            self.workflow.run_rows(
                [
                    {
                        "local_path": "data/normal.mat",
                        "original_filename": "normal.mat",
                        "class_name": "normal",
                        "label_index": "0",
                        "sensor_key": "A|B",
                    }
                ]
            )

    def test_derives_label_index_from_original_manifest_class_name(self) -> None:
        result = self.workflow.run_rows(
            [
                {
                    "local_path": "data/normal.mat",
                    "original_filename": "normal.mat",
                    "class_name": "normal",
                    "sensor_key": "DE",
                }
            ]
        )
        self.assertEqual(result.files[0].label_index, 0)
        self.assertEqual(result.files[0].class_name, "normal")
        self.assertTrue(result.files[0].file_correct)

    def test_rejects_class_name_and_label_index_mismatch(self) -> None:
        with self.assertRaisesRegex(BearingBatchError, "mismatch"):
            self.workflow.run_rows(
                [
                    {
                        "local_path": "data/normal.mat",
                        "original_filename": "normal.mat",
                        "class_name": "normal",
                        "label_index": "1",
                        "sensor_key": "DE",
                    }
                ]
            )

    def test_write_json_is_round_trippable_and_does_not_need_training(self) -> None:
        result = self.workflow.run_rows(
            [
                {
                    "local_path": "data/normal.mat",
                    "original_filename": "normal.mat",
                    "class_name": "normal",
                    "label_index": "0",
                    "sensor_key": "DE",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "batch.json"
            written = self.workflow.write_json(result, destination)
            self.assertEqual(written, destination.resolve())
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload["file_count"], 1)
            self.assertEqual(payload["files"][0]["window_count"], 2)

    def test_real_split_manifest_can_be_read_and_first_file_evaluated(self) -> None:
        checkpoint = PROJECT_ROOT / "artifacts/checkpoints/bearing/bearing_cnn_baseline_best.pt"
        if not checkpoint.is_file():
            self.skipTest("frozen bearing checkpoint is unavailable")
        workflow = BearingBatchDiagnosisWorkflow.from_checkpoint(
            checkpoint,
            device="cpu",
            project_root=PROJECT_ROOT,
        )
        result = workflow.run_manifest(
            PROJECT_ROOT / "datasets/bearing/cwru/split-manifest.csv",
            split="test",
            max_files=1,
        )
        self.assertEqual(result.file_count, 1)
        self.assertEqual(result.files[0].class_name, "ball")
        self.assertGreater(result.files[0].window_count, 0)


if __name__ == "__main__":
    unittest.main()
