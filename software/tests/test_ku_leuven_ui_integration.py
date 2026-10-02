"""End-to-end Qt integration for the KU Leuven development model."""

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

from airwatch.data import KULeuvenMaterializedDataset
from main import MyWindow


DATASET_ROOT = Path(r"D:\AirWatch_Datasets\ku_leuven_drone_rf\prepared\known-iq-v1")


@unittest.skipUnless(DATASET_ROOT.is_dir(), "local KU Leuven prepared data is unavailable")
class KULeuvenUIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        dataset = KULeuvenMaterializedDataset(
            DATASET_ROOT, split="validation", verify_hashes=True
        )
        try:
            recording_id = dataset.samples[0].recording_id
            indices = [
                index
                for index, sample in enumerate(dataset.samples)
                if sample.recording_id == recording_id
            ]
            cls.windows = np.stack(
                [dataset[index][0].numpy() for index in indices]
            )
        finally:
            dataset.close()

    def _pump_until(self, predicate, timeout_seconds=5.0):
        deadline = time.monotonic() + timeout_seconds
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertTrue(predicate(), "Qt background inference did not finish in time")

    def test_uav_page_runs_real_model_and_keeps_open_set_unpublished(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ku-leuven-known-recording.npy"
            np.save(path, np.concatenate(self.windows, axis=-1))
            window = MyWindow()
            try:
                workbench = window.recognition_workbench
                workbench.select_module(0)
                workbench.uav_plots.select_file(path)
                workbench.uav_plots.rate.setText("100000000")
                self._pump_until(
                    lambda: workbench.uav_plots.thread is None
                    and workbench.uav_plots.is_recognition_ready
                )
                self.assertIsNotNone(workbench.uav_plots.view)
                self.assertEqual(workbench.uav_plots.view.channels, 2)
                self.assertEqual(workbench.uav_plots.view.total_samples, 32 * 4096)
                self.assertTrue(workbench.uav_start.isEnabled())
                self.assertFalse(window.uav_backend.loaded)

                completed = []
                failures = []
                window.uav_task.completed.connect(completed.append)
                window.uav_task.failed.connect(failures.append)
                window.start_uav_recognition()
                self.assertTrue(window.uav_task.busy)
                self._pump_until(lambda: not window.uav_task.busy)

                self.assertEqual(failures, [])
                self.assertEqual(len(completed), 1)
                self.assertTrue(window.uav_backend.loaded)
                self.assertEqual(
                    completed[0].model_version,
                    "ku-leuven-tcn-weighted-worst-noise-dev-20260910-v4",
                )
                self.assertIsNone(completed[0].known_unknown)
                self.assertEqual(completed[0].window_count, 32)
                self.assertEqual(completed[0].open_set_status, "not_evaluated")
                self.assertIn("已完成（开发版）", workbench.uav_result_label.text())
                self.assertIn("未评估（阈值尚未冻结）", workbench.uav_result_label.text())
                self.assertIn("32 / 32", workbench.uav_progress.format())
                self.assertTrue(workbench.uav_start.isEnabled())
                self.assertFalse(workbench.uav_cancel.isEnabled())
            finally:
                window.close()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
