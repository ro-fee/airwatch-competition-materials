import json
from pathlib import Path
from threading import Event
import tempfile
import unittest

import numpy as np
import torch

from airwatch.workflows.historical_generation import (
    ALL_CLASSES,
    GENERATION_CLASSES,
    GenerationRequest,
    HistoricalGenerationWorkflow,
    MAX_TOTAL_SAMPLES,
    signal_statistics_deviation,
)


class FakeGenerator(torch.nn.Module):
    def __init__(self, cancel_event=None, bad_shape=False):
        super().__init__()
        self.cancel_event = cancel_event
        self.bad_shape = bad_shape

    def forward(self, noise, labels):
        count = noise.shape[0]
        length = 512 if self.bad_shape else 1024
        axis = torch.linspace(-1.0, 1.0, length, device=noise.device)
        output = axis.view(1, 1, -1).repeat(count, 1, 1)
        output = output + labels.float().view(-1, 1, 1) / 100.0
        output = output + noise[:, :1].view(-1, 1, 1) / 1000.0
        if self.cancel_event is not None:
            self.cancel_event.set()
        return output


def workflow(model):
    return HistoricalGenerationWorkflow(
        device="cpu", model_loader=lambda: (model, Path("fake-generator.ckpt"))
    )


class HistoricalGenerationWorkflowTests(unittest.TestCase):
    def test_request_is_bounded_before_any_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in (0, 1, 10_001, True, 1.5):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    GenerationRequest(root, "32PSK", value)
            with self.assertRaises(ValueError):
                GenerationRequest(root, ALL_CLASSES, MAX_TOTAL_SAMPLES // 15 + 1)
            with self.assertRaises(ValueError):
                GenerationRequest(root, "not-a-class", 2)

    def test_single_class_is_committed_as_unique_immutable_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = GenerationRequest(root, "FM", 5, batch_size=2)
            progress = []
            result = workflow(FakeGenerator()).run(
                request, Event(), lambda current, total: progress.append((current, total))
            )

            self.assertTrue(result.session_directory.is_dir())
            self.assertEqual(result.labels, ("FM",))
            self.assertEqual(result.total_samples, 5)
            self.assertEqual(progress[0], (0, 5))
            self.assertEqual(progress[-1], (5, 5))
            self.assertFalse(result.batches[0].flags.writeable)
            np.testing.assert_array_equal(
                np.load(result.classes[0].output_file, allow_pickle=False),
                result.batches[0],
            )
            manifest = json.loads(
                (result.session_directory / "generation-session.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["classes"], ["FM"])
            self.assertEqual(manifest["total_samples"], 5)
            self.assertIn("不验证类别正确性", manifest["score_limitation"])
            self.assertFalse(any(path.name.endswith(".staging") for path in root.iterdir()))

            second = workflow(FakeGenerator()).run(request, Event())
            self.assertNotEqual(result.session_directory, second.session_directory)
            self.assertTrue(result.classes[0].output_file.is_file())
            self.assertTrue(second.classes[0].output_file.is_file())

    def test_all_classes_keep_scores_with_their_own_class(self):
        with tempfile.TemporaryDirectory() as directory:
            result = workflow(FakeGenerator()).run(
                GenerationRequest(Path(directory), ALL_CLASSES, 2, batch_size=1), Event()
            )
            self.assertEqual(result.labels, GENERATION_CLASSES)
            self.assertEqual(len(result.classes), 15)
            for item in result.classes:
                self.assertEqual(item.deviation_scores.shape, (2,))
                self.assertEqual(result.score_for(item.label, 1), item.deviation_scores[0])
            with self.assertRaises(KeyError):
                result.score_for("missing", 1)

    def test_identical_samples_have_zero_descriptive_deviation(self):
        samples = np.ones((3, 1, 1024), dtype=np.float32)
        np.testing.assert_array_equal(
            signal_statistics_deviation(samples), np.zeros(3, dtype=np.float32)
        )

    def test_cancel_removes_staging_and_never_commits_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cancelled = Event()
            runner = workflow(FakeGenerator(cancel_event=cancelled))
            with self.assertRaises(InterruptedError):
                runner.run(
                    GenerationRequest(root, "32PSK", 5, batch_size=2), cancelled
                )
            self.assertEqual(list(root.iterdir()), [])

    def test_model_contract_failure_removes_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "输出形状"):
                workflow(FakeGenerator(bad_shape=True)).run(
                    GenerationRequest(root, "32PSK", 2), Event()
                )
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
