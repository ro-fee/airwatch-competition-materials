from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from airwatch.workflows.uav_contract import (
    QualityStatus,
    RecognitionStatus,
    UAVInputInfo,
    UAVRecognitionResult,
)
from main import MyWindow


class UAVMainIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def pump(self, predicate, timeout_seconds=4.0):
        deadline = time.monotonic() + timeout_seconds
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertTrue(predicate())

    def test_start_waits_for_full_input_and_explicit_sample_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.npy"
            samples = np.stack(
                (
                    np.arange(131072, dtype=np.float32),
                    np.arange(131072, dtype=np.float32) + 0.5,
                )
            )
            np.save(path, samples)
            window = MyWindow()
            try:
                plots = window.recognition_workbench.uav_plots
                plots.select_file(path)
                self.pump(lambda: plots.thread is None and plots.view is not None)
                self.assertFalse(plots.is_recognition_ready)
                self.assertFalse(window.recognition_workbench.uav_start.isEnabled())
                plots.rate.setText("100000000")
                self.pump(
                    lambda: plots.thread is None and plots.is_recognition_ready
                )
                self.assertTrue(window.recognition_workbench.uav_start.isEnabled())
            finally:
                window.close()
                self.app.processEvents()

    def test_completed_card_uses_result_contract_and_real_window_count(self):
        window = MyWindow()
        try:
            result = UAVRecognitionResult(
                input_info=UAVInputInfo(
                    str(Path("recording.npy").resolve()),
                    sample_rate_hz=100_000_000,
                    sample_count=28,
                    channels=2,
                ),
                quality_status=QualityStatus.CAUTION,
                quality_message="开发质量门",
                recognition_status=RecognitionStatus.COMPLETED,
                label="测试已知源",
                confidence=0.75,
                model_version="model-v1",
                window_count=7,
                open_set_status="not_evaluated",
                contract_id="contract-v1",
                limitations=("development only",),
            )
            window._on_uav_recognition_completed(result)
            workbench = window.recognition_workbench
            self.assertEqual(workbench.uav_progress.maximum(), 7)
            self.assertEqual(workbench.uav_progress.value(), 7)
            self.assertIn("7 / 7 个窗口", workbench.uav_progress.format())
            self.assertIn("contract-v1", workbench.uav_result_label.text())
            self.assertIn("未评估（阈值尚未冻结）", workbench.uav_result_label.text())
        finally:
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
