"""Evidence-gate tests for the historical two-model comparison Workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event
import tempfile
import unittest

import numpy as np
import torch

from airwatch.general_recognition_contract import task_spec
from airwatch.workflows.historical_model_comparison import (
    ComparisonRunRequest,
    HistoricalModelComparisonWorkflow,
    LoadedComparisonModel,
)


class _ConstantModel(torch.nn.Module):
    def __init__(self, winner: int, *, invalid: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))
        self.winner = winner
        self.invalid = invalid

    def forward(self, data):
        logits = torch.zeros((len(data), 15), device=data.device)
        logits[:, self.winner] = float("nan") if self.invalid else 10.0
        return data.mean(dim=-1), logits


def _fixture(root: Path) -> Path:
    spec = task_spec("信号个体识别")
    signal = np.linspace(-1, 1, 4096, dtype=np.float32).reshape(2048, 2)
    signal_path = root / "signal.dat"
    signal.tofile(signal_path)
    payload = {
        "schema_version": 1,
        "dataset_id": "workflow-fixture",
        "task_name": "信号个体识别",
        "purpose": "software_demo",
        "performance_evidence": False,
        "origin": "unit test fixture",
        "label_provenance": "not ground truth",
        "sample_dtype": spec.sample_dtype,
        "input_layout": spec.input_layout,
        "channels": spec.channels,
        "window_size": spec.window_size,
        "label_order": list(spec.classes),
        "recordings": [{
            "recording_id": "r1",
            "path": signal_path.name,
            "sha256": hashlib.sha256(signal_path.read_bytes()).hexdigest().upper(),
            "bytes": signal_path.stat().st_size,
            "label": spec.classes[0],
            "acquisition_group": "fixture",
        }],
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return manifest


def _loader(root: Path, *, invalid_light: bool = False):
    reference_path = root / "reference.pt"
    light_path = root / "light.pt"
    reference_path.write_bytes(b"reference checkpoint")
    light_path.write_bytes(b"light checkpoint")

    def load():
        return (
            LoadedComparisonModel("reference", "reference.slot", _ConstantModel(2), reference_path),
            LoadedComparisonModel("light", "light.slot", _ConstantModel(2, invalid=invalid_light), light_path),
        )

    return load


class HistoricalModelComparisonTests(unittest.TestCase):
    def test_workflow_publishes_atomic_descriptive_evidence_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root)
            output = root / "evidence"
            workflow = HistoricalModelComparisonWorkflow(
                device="cpu", batch_size=1, model_loader=_loader(root)
            )
            progress = []
            result = workflow.run(
                ComparisonRunRequest(manifest, output), Event(),
                lambda current, total: progress.append((current, total)),
            )

            self.assertEqual(result.total_windows, 2)
            self.assertEqual(result.agreement_count, 2)
            self.assertEqual(result.agreement_rate, 1.0)
            self.assertIsNone(result.accuracy)
            self.assertIsNone(result.macro_precision)
            self.assertIsNone(result.macro_f1)
            self.assertEqual(progress[-1], (4, 4))
            self.assertTrue(result.evidence_file.is_file())
            self.assertFalse(any(path.name.endswith(".staging") for path in output.iterdir()))
            payload = json.loads(result.evidence_file.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["performance_metrics"],
                {
                    "status": "not_eligible_software_demo",
                    "accuracy": None,
                    "macro_precision": None,
                    "macro_f1": None,
                    "latency": None,
                },
            )
            self.assertEqual(
                payload["descriptive_comparison"]["meaning"],
                "output similarity only; not correctness",
            )

    def test_repeated_runs_get_unique_committed_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root)
            workflow = HistoricalModelComparisonWorkflow(
                device="cpu", model_loader=_loader(root)
            )
            first = workflow.run(ComparisonRunRequest(manifest, root / "out"), Event())
            second = workflow.run(ComparisonRunRequest(manifest, root / "out"), Event())
            self.assertNotEqual(first.run_id, second.run_id)
            self.assertTrue(first.evidence_directory.is_dir())
            self.assertTrue(second.evidence_directory.is_dir())

    def test_cancellation_never_commits_partial_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root)
            output = root / "out"
            workflow = HistoricalModelComparisonWorkflow(
                device="cpu", batch_size=1, model_loader=_loader(root)
            )
            cancelled = Event()

            def cancel_after_first(current, total):
                if current:
                    cancelled.set()

            with self.assertRaises(InterruptedError):
                workflow.run(
                    ComparisonRunRequest(manifest, output),
                    cancelled,
                    cancel_after_first,
                )
            self.assertFalse(output.exists())

    def test_invalid_model_output_never_commits_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root)
            output = root / "out"
            workflow = HistoricalModelComparisonWorkflow(
                device="cpu", model_loader=_loader(root, invalid_light=True)
            )
            with self.assertRaisesRegex(ValueError, "非有限值"):
                workflow.run(ComparisonRunRequest(manifest, output), Event())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
