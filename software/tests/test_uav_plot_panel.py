import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from airwatch.ui.uav_plot_panel import UAVPlotPanel


class UAVPlotPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def pump(self, predicate, timeout=3000):
        deadline = time.monotonic() + timeout / 1000
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents(); QTest.qWait(5)
        self.assertTrue(predicate())

    def test_waveform_then_spectrum_then_spectrogram(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'uav.npy'
            np.save(path, np.stack((np.sin(np.arange(2048) / 5), np.cos(np.arange(2048) / 5))))
            panel = UAVPlotPanel(); panel.show(); panel.select_file(path)
            self.pump(lambda: panel.thread is None and panel.view is not None)
            self.assertEqual(panel.view.mode, 'waveform')
            self.assertFalse(panel.is_recognition_ready)
            panel.rate.setText('1000')
            panel.mode.setCurrentIndex(1)
            self.pump(lambda: panel.thread is None and panel.view is not None and panel.view.mode == 'spectrum')
            self.assertTrue(panel.view.view.two_sided)
            self.assertTrue(panel.is_recognition_ready)
            panel.mode.setCurrentIndex(2)
            self.pump(lambda: panel.thread is None and panel.view is not None and panel.view.mode == 'spectrogram')
            self.assertGreater(panel.view.view.power.shape[0], 1)
            panel.close()

    def test_frequency_modes_require_rate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'uav.npy'; np.save(path, np.sin(np.arange(512) / 5))
            panel = UAVPlotPanel(); panel.select_file(path); self.pump(lambda: panel.thread is None and panel.view is not None)
            panel.mode.setCurrentIndex(1); self.pump(lambda: panel.thread is None)
            self.assertIn('需要采样率', panel.caption.text())
            panel.close()

    def test_rate_change_refits_downsampled_waveform_to_seconds(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rate.npy'
            np.save(path, np.stack((np.sin(np.arange(131072) / 5),
                                    np.cos(np.arange(131072) / 5))))
            panel = UAVPlotPanel(); panel.resize(900, 600); panel.show()
            try:
                panel.select_file(path)
                self.pump(lambda: panel.thread is None and panel.view is not None)
                self.assertGreater(panel.plot.viewRange()[0][1], 60000)
                panel.rate.setText('100000000')
                self.pump(lambda: panel.thread is None and panel.view is not None)
                QTest.qWait(50)
                low, high = panel.plot.viewRange()[0]
                self.assertLess(high - low, 0.002)
                self.assertLessEqual(low, 0)
                self.assertGreaterEqual(high, panel.view.view.x[-1])
                for curve in panel.plot.listDataItems():
                    self.assertGreater(len(curve.getData()[0]), 2)
            finally:
                panel.cancel(); self.pump(lambda: panel.thread is None)
                panel.close(); panel.deleteLater()

    def test_cancel_prevents_stale_result(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'uav.npy'; np.save(path, np.ones(65536))
            panel = UAVPlotPanel(); panel.select_file(path); panel.cancel(); self.app.processEvents()
            self.assertIsNone(panel.view)
            self.assertFalse(panel.is_recognition_ready)
            panel.close()

    def test_rejected_full_recording_never_enables_recognition(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'constant.npy'
            np.save(path, np.ones((2, 131072), dtype=np.float32))
            panel = UAVPlotPanel(); panel.rate.setText('100000000'); panel.select_file(path)
            panel.rate.setText('100000000')
            self.pump(lambda: panel.thread is None and panel.view is not None)
            self.assertEqual(panel.view.quality.status.value, 'rejected')
            self.assertFalse(panel.is_recognition_ready)
            panel.close()


if __name__ == '__main__':
    unittest.main()


