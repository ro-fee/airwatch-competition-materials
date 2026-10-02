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
from airwatch.ui.generation_background import GenerationTask
from airwatch.workflows.historical_generation import (
    GenerationRequest,
    HistoricalGenerationWorkflow,
)


class RecordingGenerator(torch.nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def forward(self, noise, labels):
        self.calls.append(QThread.currentThread())
        axis = torch.linspace(-1.0, 1.0, 1024, device=noise.device)
        return axis.view(1, 1, -1).repeat(noise.shape[0], 1, 1)


def fake_workflow(calls):
    return HistoricalGenerationWorkflow(
        device="cpu",
        model_loader=lambda: (RecordingGenerator(calls), Path("fake-generator.ckpt")),
    )


class GenerationBackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def pump(self, predicate, timeout=3000):
        deadline = time.monotonic() + timeout / 1000
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertTrue(predicate())

    def test_task_runs_generation_off_gui_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, results, progress = [], [], []
            task = GenerationTask(fake_workflow(calls))
            task.completed.connect(results.append)
            task.progress.connect(lambda current, total: progress.append((current, total)))
            task.start(GenerationRequest(Path(directory), "FM", 3, batch_size=2))
            self.pump(lambda: not task.busy)

            self.assertEqual(len(results), 1)
            self.assertTrue(calls)
            self.assertTrue(all(thread != self.app.thread() for thread in calls))
            self.assertEqual(progress[-1], (3, 3))

    def test_main_window_publishes_only_completed_session(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            window = main.MyWindow()
            self.addCleanup(window.close)
            window.generation_task.workflow = fake_workflow(calls)
            window.save_path = directory
            window.lineEdit_savepath.setText(directory)
            window.comboBox_class.setCurrentText("FM")
            window.lineEdit_samplenum.setText("3")

            with patch.object(main.QMessageBox, "critical") as critical:
                window.generate_data()
                self.pump(lambda: not window.generation_task.busy)
                self.assertFalse(critical.called)

            result = window.generation_result
            self.assertIsNotNone(result)
            self.assertEqual(window.gen_labels, ["FM"])
            self.assertEqual(len(window.signals), 1)
            self.assertEqual(window.signals[0].shape, (3, 1, 1024))
            self.assertEqual(window.comboBox_showclass.currentText(), "FM")
            self.assertTrue(window.lineEdit_as.text())
            self.assertTrue(window.lineEdit_score.text())
            self.assertTrue(window.pushButton_switchsignal.isEnabled())
            self.assertTrue(result.session_directory.is_dir())
            self.assertFalse(any(path.name.endswith(".staging") for path in Path(directory).iterdir()))
            self.assertIn("不验证类别正确性", window.textEdit_log_2.toPlainText())

    def test_save_dialog_starts_in_user_writable_generation_directory(self):
        window = main.MyWindow()
        self.addCleanup(window.close)
        selected = str(window.runtime_directories["generation"])
        with patch.object(
            main.QFileDialog, "getExistingDirectory", return_value=selected
        ) as dialog:
            window.select_save_path()
        self.assertEqual(dialog.call_args.args[2], selected)
        self.assertEqual(window.lineEdit_savepath.text(), str(Path(selected).resolve()))


if __name__ == "__main__":
    unittest.main()
