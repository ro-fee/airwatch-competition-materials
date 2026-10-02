"""Qt Adapter and maintained-window tests for historical model comparison."""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from PyQt5.QtCore import QThread
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

import main
from airwatch.data.historical_comparison import default_historical_comparison_manifest
from airwatch.ui.historical_comparison_background import HistoricalComparisonTask
from airwatch.workflows.historical_model_comparison import (
    ComparisonRunRequest,
    HistoricalModelComparisonWorkflow,
    LoadedComparisonModel,
)


class _RecordingModel(torch.nn.Module):
    def __init__(self, calls, winner):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.ones(2))
        self.calls = calls
        self.winner = winner

    def forward(self, data):
        self.calls.append(QThread.currentThread())
        logits = torch.zeros((len(data), 15), device=data.device)
        logits[:, self.winner] = 5.0
        return data.mean(dim=-1), logits


def _workflow(root: Path, calls):
    reference_path = root / "reference.pt"
    light_path = root / "light.pt"
    reference_path.write_bytes(b"reference")
    light_path.write_bytes(b"light")

    def loader():
        return (
            LoadedComparisonModel(
                "reference", "reference.slot", _RecordingModel(calls, 1), reference_path
            ),
            LoadedComparisonModel(
                "light", "light.slot", _RecordingModel(calls, 1), light_path
            ),
        )

    return HistoricalModelComparisonWorkflow(
        device="cpu", batch_size=8, model_loader=loader
    )


class HistoricalComparisonBackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def pump(self, predicate, timeout=5000):
        deadline = time.monotonic() + timeout / 1000
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertTrue(predicate())

    def test_task_runs_both_models_off_gui_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls, results, progress = [], [], []
            task = HistoricalComparisonTask(_workflow(root, calls))
            task.completed.connect(results.append)
            task.progress.connect(lambda current, total: progress.append((current, total)))
            task.start(
                ComparisonRunRequest(
                    default_historical_comparison_manifest(), root / "evidence"
                )
            )
            self.pump(lambda: not task.busy)

            self.assertEqual(len(results), 1)
            self.assertTrue(calls)
            self.assertTrue(all(thread != self.app.thread() for thread in calls))
            self.assertEqual(progress[-1], (64, 64))

    def test_main_window_keeps_general_page_state_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            window = main.MyWindow()
            self.addCleanup(window.close)
            window.historical_comparison_task.workflow = _workflow(root, calls)
            window.runtime_directories["comparison"] = root / "evidence"
            general_data = object()
            general_labels = object()
            general_files = ["general-page-sentinel"]
            window.data = general_data
            window.label = general_labels
            window.filename = general_files

            with patch.object(main.QMessageBox, "critical") as critical:
                window.start_historical_comparison()
                self.pump(lambda: not window.historical_comparison_task.busy)
                self.assertFalse(critical.called)

            self.assertIs(window.data, general_data)
            self.assertIs(window.label, general_labels)
            self.assertIs(window.filename, general_files)
            self.assertEqual(window.comboBox_task_2.count(), 1)
            self.assertEqual(window.comboBox_task_2.currentText(), "信号个体识别")
            self.assertIsNotNone(window.light_comparison_result)
            self.assertIn("这不是准确率", window.textEdit_6.toPlainText())
            self.assertIn("Evidence：", window.textEdit_7.toPlainText())
            self.assertTrue(window.light_comparison_result.evidence_file.is_file())


if __name__ == "__main__":
    unittest.main()
