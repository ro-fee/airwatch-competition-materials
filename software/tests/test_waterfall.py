"""Regression tests for the animated spectrum waterfall integration."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
os.environ['QT_API'] = 'pyqt5'

import numpy as np
import pyqtgraph as pg
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from waterfall_plotter import WaterfallPlotter


class WaterfallPlotterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.plot = pg.PlotWidget()
        self.waterfall = WaterfallPlotter(self.plot)
        t = np.arange(4096) / 9600
        self.signal = np.sin(2 * np.pi * 700 * t).astype(np.float32)

    def tearDown(self):
        self.waterfall.stop(clear=True)
        self.plot.close()

    def test_start_builds_relative_db_frames_and_dynamic_axes(self):
        self.waterfall.start(self.signal, 9600, 256, history_lines=100)
        self.assertTrue(self.waterfall.is_running)
        self.assertEqual(self.waterfall.buffer.shape, (100, 129))
        self.assertEqual(self.waterfall.spectra_db.shape[0], 129)
        self.assertLessEqual(float(self.waterfall.spectra_db.max()), 0.0001)
        self.assertGreaterEqual(float(self.waterfall.spectra_db.min()), -100.0001)
        self.assertAlmostEqual(self.waterfall.frequencies[-1], 4800, places=4)
        self.assertGreaterEqual(self.waterfall.frame_interval_ms, 1)

    def test_frame_update_pause_resume_and_finish_are_consistent(self):
        self.waterfall.start(self.signal, 9600, 256, history_lines=80)
        self.waterfall.pause()
        self.assertTrue(self.waterfall.is_paused)
        pointer = self.waterfall.current_frame
        self.waterfall._update_frame()
        self.assertEqual(self.waterfall.current_frame, pointer)
        self.waterfall.resume()
        self.assertTrue(self.waterfall.is_running)
        self.waterfall._advance_to_frame(self.waterfall.total_frames)
        self.assertTrue(self.waterfall.is_finished)
        self.assertFalse(self.waterfall.timer.isActive())

    def test_repeated_start_does_not_leave_extra_timers(self):
        timer = self.waterfall.timer
        for _ in range(25):
            self.waterfall.start(self.signal, 9600, 256, history_lines=60)
            self.waterfall.pause()
        self.assertIs(self.waterfall.timer, timer)
        self.assertTrue(self.waterfall.is_paused)
        self.waterfall.stop(clear=False)
        self.assertFalse(timer.isActive())
        self.assertIsNotNone(self.waterfall.image_item)
        self.waterfall.stop(clear=True)
        self.assertIsNone(self.waterfall.image_item)

    def test_timer_advances_during_event_loop(self):
        self.waterfall.start(self.signal, 9600, 256, history_lines=60, playback_speed=4)
        pointer = self.waterfall.current_frame
        self.assertTrue(self.waterfall.timer.isActive())
        QTest.qWait(max(100, self.waterfall.frame_interval_ms * 10))
        self.assertGreater(self.waterfall.current_frame, pointer)

    def test_zero_signal_stays_at_display_floor(self):
        self.waterfall.start(np.zeros(4096, dtype=np.float32), 9600, 256, history_lines=60)
        self.assertTrue(np.all(self.waterfall.spectra_db == self.waterfall.floor_db))
        self.assertTrue(np.all(self.waterfall.buffer[0] == self.waterfall.floor_db))

    def test_high_sample_rate_uses_capped_render_timer(self):
        self.waterfall.start(self.signal, 100_000_000, 256, history_lines=60)
        self.assertGreaterEqual(self.waterfall.frame_interval_ms, 30)
        self.assertLessEqual(self.waterfall.frame_interval_ms, 40)

    def test_paused_timer_callback_does_not_advance_pointer(self):
        self.waterfall.start(self.signal, 9600, 256, history_lines=60)
        self.waterfall.pause()
        pointer = self.waterfall.current_frame
        self.waterfall._update_frame()
        self.assertEqual(self.waterfall.current_frame, pointer)

    def test_complex_iq_uses_two_sided_frequency_axis(self):
        t = np.arange(4096) / 9600
        iq = np.exp(1j * 2 * np.pi * 1200 * t)
        self.waterfall.start(iq, 9600, 256, history_lines=60)
        self.assertAlmostEqual(self.waterfall.frequencies[0], -4800, places=4)
        self.assertAlmostEqual(self.waterfall.frequencies[-1], 4762.5, places=4)
        peak_frequency = self.waterfall.frequencies[np.argmax(self.waterfall.spectra_db[:, 0])]
        self.assertAlmostEqual(peak_frequency, 1200, delta=50)

    def test_non_finite_inputs_fail_cleanly(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            signal = self.signal.copy()
            signal[10] = bad_value
            with self.subTest(value=bad_value):
                with self.assertRaises(ValueError):
                    self.waterfall.start(signal, 9600, 256, history_lines=60)
                self.assertFalse(self.waterfall.timer.isActive())

    def test_high_sample_rate_batches_due_frames_and_finishes(self):
        self.waterfall.start(self.signal, 100_000_000, 256, history_lines=60)
        QTest.qWait(100)
        self.assertTrue(self.waterfall.is_finished)
        self.assertEqual(self.waterfall.current_frame, self.waterfall.total_frames)
        self.assertFalse(self.waterfall.timer.isActive())

    def test_finished_playback_can_restart_from_first_frame(self):
        self.waterfall.start(self.signal, 9600, 256, history_lines=60)
        total = self.waterfall.total_frames
        self.waterfall._advance_to_frame(total)
        self.assertTrue(self.waterfall.is_finished)
        self.waterfall.start(self.signal, 9600, 256, history_lines=60)
        self.assertTrue(self.waterfall.is_running)
        self.assertEqual(self.waterfall.current_frame, 1)
        self.assertEqual(self.waterfall.total_frames, total)

    def test_invalid_inputs_fail_cleanly(self):
        invalid_cases = [
            (np.array([]), 9600, 256, 100, 1),
            (self.signal, 0, 256, 100, 1),
            (self.signal[:100], 9600, 256, 100, 1),
            (self.signal, 9600, 250, 100, 1),
            (self.signal, 9600, 256, 5, 1),
            (self.signal, 9600, 256, 100, 0),
        ]
        for data, fs, n_fft, history, speed in invalid_cases:
            with self.subTest(fs=fs, n_fft=n_fft, history=history, speed=speed):
                with self.assertRaises(ValueError):
                    self.waterfall.start(data, fs, n_fft, history, playback_speed=speed)
                self.assertFalse(self.waterfall.timer.isActive())


class MainWindowWaterfallIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main
        cls.main = main
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = self.main.MyWindow()
        t = np.arange(8192) / 9600
        self.window.filename = ['waterfall_test.npy']
        self.window.data = np.sin(2 * np.pi * 600 * t).astype(np.float32)
        self.window.home_data = self.window.data
        self.window.home_file = 'waterfall_test.npy'
        self.window.pushButton_reload_TF.setEnabled(True)

    def tearDown(self):
        self.window.close()

    def test_waterfall_signal_accepts_only_documented_shapes(self):
        n = 32
        real = np.arange(n, dtype=np.float32)
        for shaped in (real, real[None, :]):
            self.window.home_data = shaped
            np.testing.assert_array_equal(self.window._waterfall_signal(), real)

        iq_channels = np.vstack((real, -real))
        self.window.home_data = iq_channels
        np.testing.assert_array_equal(self.window._waterfall_signal(), real - 1j * real)
        self.window.home_data = iq_channels.T
        np.testing.assert_array_equal(self.window._waterfall_signal(), real - 1j * real)

        for invalid in (np.zeros((3, n)), np.zeros((2, 3, n)), np.zeros((n, 1))):
            with self.subTest(shape=invalid.shape):
                self.window.home_data = invalid
                with self.assertRaises(ValueError):
                    self.window._waterfall_signal()

    def test_feature_selection_uses_home_data_and_ignores_waterfall_hz_axis(self):
        self.window.data = np.arange(10, dtype=np.float32)
        self.window.home_data = np.arange(512, dtype=np.float32)
        self.window.radioButton_time_1.setChecked(True)
        self.window.label_signalshow_1.setXRange(20, 39, padding=0)
        start, end = self.window._feature_selection_bounds(self.window.home_data)
        self.assertEqual((start, end), (20, 40))

        self.window.radioButton_waterfall.setChecked(True)
        self.window.waterfall_plotter.pause()
        start, end = self.window._feature_selection_bounds(self.window.home_data)
        self.assertEqual((start, end), (0, 512))
        self.assertTrue(self.window.waterfall_plotter.is_paused)

    def test_features_reads_home_data_not_recognition_data(self):
        self.window.data = np.full(512, -1, dtype=np.float32)
        self.window.home_data = np.arange(512, dtype=np.float32)
        self.window.radioButton_time_1.setChecked(True)
        self.window.label_signalshow_1.setXRange(20, 39, padding=0)
        captured = []

        def capture_then_stop(data, *args, **kwargs):
            captured.append(np.asarray(data).copy())
            raise RuntimeError('stop after capture')

        with patch('airwatch.analysis.wavelet_views.pywt.dwt', side_effect=capture_then_stop):
            self.window.features('小波特征')
            end = time.monotonic() + 10
            while self.window.home_feature_task.busy and time.monotonic() < end:
                QTest.qWait(10)
        self.assertFalse(self.window.home_feature_task.busy)
        self.assertEqual(len(captured), 1)
        np.testing.assert_array_equal(captured[0], np.arange(20, 40, dtype=np.float32))

    def test_leaving_home_pauses_without_advancing(self):
        self.window.tabWidget.setCurrentWidget(self.window.tab1)
        self.window.radioButton_waterfall.setChecked(True)
        self.assertTrue(self.window.waterfall_plotter.is_running)
        home_tab = self.window.tabWidget.indexOf(self.window.tab1)
        other_tab = next(i for i in range(self.window.tabWidget.count()) if i != home_tab)
        self.window.tabWidget.setCurrentIndex(other_tab)
        self.assertTrue(self.window.waterfall_plotter.is_paused)
        pointer = self.window.waterfall_plotter.current_frame
        QTest.qWait(120)
        self.assertEqual(self.window.waterfall_plotter.current_frame, pointer)

    def test_starting_new_home_signal_replaces_cached_spectra(self):
        self.window.radioButton_waterfall.setChecked(True)
        old_spectra = self.window.waterfall_plotter.spectra_db.copy()
        t = np.arange(8192) / 9600
        self.window.home_data = np.sin(2 * np.pi * 1600 * t).astype(np.float32)
        self.window.home_file = 'new_waterfall_test.npy'
        self.window.start_waterfall_playback()
        new_spectra = self.window.waterfall_plotter.spectra_db
        self.assertEqual(self.window.waterfall_plotter.current_frame, 1)
        self.assertFalse(np.array_equal(old_spectra, new_spectra))

    def test_opening_new_file_restarts_waterfall_with_new_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = os.path.join(tmpdir, 'first.npy')
            second_path = os.path.join(tmpdir, 'second.npy')
            np.save(first_path, np.sin(np.linspace(0, 30, 4096)).astype(np.float32))
            np.save(second_path, np.cos(np.linspace(0, 70, 4096)).astype(np.float32))
            with patch.object(self.main.QFileDialog, 'getOpenFileName', return_value=(first_path, '')):
                self.window.openfile()
            self.window.radioButton_waterfall.setChecked(True)
            first_spectra = self.window.waterfall_plotter.spectra_db.copy()
            with patch.object(self.main.QFileDialog, 'getOpenFileName', return_value=(second_path, '')):
                self.window.openfile()
            self.assertEqual(self.window.home_file, second_path)
            self.assertTrue(self.window.waterfall_plotter.is_running)
            self.assertFalse(np.array_equal(first_spectra, self.window.waterfall_plotter.spectra_db))

    def test_switching_from_waterfall_clears_axis_state_and_limits(self):
        self.window.radioButton_waterfall.setChecked(True)
        view_box = self.window.label_signalshow_1.getViewBox()
        self.assertTrue(view_box.state['yInverted'])
        self.window.radioButton_time_1.setChecked(True)
        QTest.qWait(20)
        self.assertFalse(view_box.state['yInverted'])
        self.assertIsNone(self.window.label_signalshow_1.getAxis('left')._tickLevels)
        self.assertIsNone(self.window.label_signalshow_1.getAxis('bottom')._tickLevels)
        self.assertIsNone(view_box.state['limits']['xLimits'][1])
        self.assertGreaterEqual(view_box.viewRange()[0][1], len(self.window.home_data) - 1)

    def test_ui_controls_and_mode_switch_lifecycle(self):
        self.assertEqual(self.window.radioButton_waterfall.text(), '动态瀑布图')
        self.window.radioButton_waterfall.setChecked(True)
        self.assertTrue(self.window.waterfall_plotter.is_running)
        self.assertEqual(self.window.pushButton_waterfall_play.text(), '暂停')
        self.window.toggle_waterfall_playback()
        self.assertTrue(self.window.waterfall_plotter.is_paused)
        self.assertEqual(self.window.pushButton_waterfall_play.text(), '继续')
        self.window.toggle_waterfall_playback()
        self.assertTrue(self.window.waterfall_plotter.is_running)
        self.window.radioButton_time_1.setChecked(True)
        self.assertFalse(self.window.waterfall_plotter.timer.isActive())

    def test_repeated_mode_switch_and_close_stop_timer(self):
        for _ in range(20):
            self.window.radioButton_waterfall.setChecked(True)
            self.assertTrue(self.window.waterfall_plotter.timer.isActive())
            self.window.radioButton_TF_1.setChecked(True)
            self.assertFalse(self.window.waterfall_plotter.timer.isActive())
        self.window.radioButton_waterfall.setChecked(True)
        timer = self.window.waterfall_plotter.timer
        self.window.close()
        self.assertFalse(timer.isActive())


if __name__ == '__main__':
    unittest.main(verbosity=2)
