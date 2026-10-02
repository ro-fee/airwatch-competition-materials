"""Acceptance for migrated waterfall analysis and Qt playback."""
import os
import unittest
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
import numpy as np
import pyqtgraph as pg
from scipy.signal import spectrogram
from PyQt5.QtCore import QPointF, QTimer
from PyQt5.QtWidgets import QApplication
from airwatch.analysis.waterfall import compute_waterfall
from airwatch.ui.plots.waterfall import WaterfallPlotter


class WaterfallMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.plot = pg.PlotWidget()
        self.w = WaterfallPlotter(self.plot)
        self.x = np.exp(-2j * np.pi * 8 * np.arange(256) / 64)
        self.addCleanup(self.plot.close)
        self.addCleanup(lambda: self.w.stop(clear=True))

    def test_compatibility_import_is_same_class(self):
        from waterfall_plotter import WaterfallPlotter as Old
        self.assertIs(Old, WaterfallPlotter)

    def test_frames_match_legacy_scipy_convention(self):
        for data in (self.x, self.x.real + 3):
            f, t, p = spectrogram(data, fs=64, window='hann', nperseg=16,
                noverlap=8, detrend='constant', scaling='density', mode='psd',
                return_onesided=not np.iscomplexobj(data))
            if np.iscomplexobj(data):
                order = np.argsort(f); f = f[order]; p = p[order]
            expected = np.clip(10*np.log10(np.maximum(p/p.max(), np.finfo(np.float32).tiny)), -100, 0)
            v = compute_waterfall(data, 64, 16)
            np.testing.assert_allclose(v.frequencies, f)
            np.testing.assert_allclose(v.frame_times, t)
            np.testing.assert_allclose(v.spectra_db, expected, atol=1e-5)
            self.assertFalse(v.spectra_db.flags.writeable)

    def test_iq_shapes_and_input_immutability(self):
        expected = compute_waterfall(self.x, 64, 16)
        for data in (self.x, np.stack([self.x.real,self.x.imag]), np.column_stack([self.x.real,self.x.imag])):
            before = data.copy()
            self.w.start(data, 64, 16, history_lines=20)
            np.testing.assert_allclose(self.w.spectra_db, expected.spectra_db)
            np.testing.assert_array_equal(data, before)

    def test_constant_and_zero_remain_dark(self):
        for data in (np.zeros(256), np.ones((1,256))):
            self.w.start(data, 64, 16, history_lines=20)
            self.assertTrue((self.w.spectra_db == -100).all())

    def test_invalid_shapes_and_fractional_parameters_are_rejected(self):
        for data, fft in ((np.ones((3,32)),16), (np.ones((32,1)),16),
                          (np.ones((2,32),dtype=complex),16), (self.x,16.5), (self.x,True)):
            with self.assertRaises(ValueError):
                self.w.start(data, 64, fft, history_lines=20)
            self.assertFalse(self.w.timer.isActive())
            self.assertIsNone(self.w.image_item)
        for overlap in (-1, 1, np.nan):
            with self.assertRaises(ValueError):
                compute_waterfall(self.x, 64, 16, overlap)
        with self.assertRaises(ValueError):
            self.w.start(self.x, 64, 16, history_lines=20.5)

    def test_pixel_centres_and_axis_order_are_explicit(self):
        old = pg.getConfigOption('imageAxisOrder')
        try:
            for order in ('row-major','col-major'):
                pg.setConfigOption('imageAxisOrder', order)
                self.w.start(self.x, 64, 16, history_lines=20)
                image = self.w.image_item
                self.assertEqual(image.axisOrder, 'row-major')
                self.assertEqual(image.image.shape, (20,16))
                for col in (0, 8, 15):
                    p = image.mapToParent(QPointF(col + .5, .5))
                    self.assertAlmostEqual(p.x(), self.w.frequencies[col])
                    self.assertAlmostEqual(p.y(), 0)
                np.testing.assert_array_equal(image.getLevels(), [-80,0])
        finally:
            pg.setConfigOption('imageAxisOrder', old)

    def test_clock_pause_resume_and_newest_row_order(self):
        clock = [0.0]
        self.w._clock = lambda: clock[0]
        self.w.start(self.x,64,16,history_lines=20,playback_speed=2)
        self.w.timer.stop()
        clock[0] += 3*self.w.frame_duration_seconds
        self.w._update_frame()
        self.assertEqual(self.w.current_frame,4)
        np.testing.assert_array_equal(self.w.buffer[:4], self.w.spectra_db[:,:4].T[::-1])
        self.w.pause(); clock[0] += 100
        self.w._update_frame(); self.assertEqual(self.w.current_frame,4)
        self.w.resume(); self.w.timer.stop(); self.w._update_frame()
        self.assertEqual(self.w.current_frame,4)
        self.w._advance_to_frame(self.w.total_frames)
        self.assertTrue(self.w.is_finished)
        np.testing.assert_array_equal(self.w.buffer, self.w.spectra_db[:,-20:].T[::-1])
        self.assertEqual(len(self.w.findChildren(QTimer)),1)

    def test_axis_state_is_reset_on_start_and_clear(self):
        self.plot.getViewBox().setLimits(xMin=1,xMax=100,yMin=1,yMax=100,
                                         minXRange=2,maxXRange=2)
        self.plot.setLogMode(x=True,y=True)
        self.plot.invertX(True)
        self.w.start(self.x,64,16,history_lines=20)
        self.assertFalse(self.plot.getAxis('bottom').logMode)
        self.assertFalse(self.plot.getViewBox().state['xInverted'])
        self.w.stop(clear=True)
        self.assertFalse(self.plot.getViewBox().state['yInverted'])
        self.assertIsNone(self.plot.getAxis('left')._tickLevels)
        self.assertEqual(self.plot.getViewBox().state['limits']['xRange'],[None,None])

    def test_bad_ui_parameter_clears_previous_playback(self):
        import main
        window = main.MyWindow()
        self.addCleanup(window.close)
        window.home_data = self.x
        window.comboBox_waterfall_fft.setCurrentText('256')
        window.lineEdit_Fs_1.setText('64.5')
        self.assertTrue(window.start_waterfall_playback())
        window.lineEdit_Fs_1.setText('')
        with patch.object(main.QMessageBox,'warning') as warning:
            self.assertFalse(window.start_waterfall_playback())
            warning.assert_called_once()
        self.assertIsNone(window.waterfall_plotter.spectra_db)
        self.assertIsNone(window.waterfall_plotter.image_item)
        self.assertFalse(window.waterfall_plotter.timer.isActive())
        self.assertEqual(window.label_waterfall_progress.text(),'无法播放')


if __name__ == '__main__':
    unittest.main()
