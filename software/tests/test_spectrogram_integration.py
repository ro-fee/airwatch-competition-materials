"""Numerical and real Qt acceptance for static spectrogram migration."""
import os
import unittest
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
os.environ['QT_API'] = 'pyqt5'

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QApplication
from scipy.signal import spectrogram

import main
from airwatch.analysis.signal_transforms import compute_spectrogram, SignalViewError
from airwatch.ui.plots.pyqtgraph_renderer import render_spectrogram, render_waveform, render_spectrum


class SpectrogramIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = main.MyWindow()
        self.plot = self.window.label_signal
        self.window.filename = ['test.npy']
        self.window.lineEdit_Fs_2.setText('64')
        self.window.lineEdit_windowlength_3.setText('16')
        self.window.lineEdit_Fs_1.setText('128')
        self.window.lineEdit_windowlength_1.setText('32')
        self.signal = np.cos(2 * np.pi * 8 * np.arange(64) / 64)
        for name in ('about', 'warning'):
            p = patch.object(main.QMessageBox, name)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.window.close)

    def image(self, plot=None):
        images = [i for i in (plot or self.plot).plotItem.items if isinstance(i, pg.ImageItem)]
        self.assertEqual(len(images), 1)
        return images[0]

    def test_real_psd_matches_explicit_scipy_contract(self):
        f, t, p = spectrogram(self.signal, fs=64, window='hann', nperseg=16,
                              noverlap=8, detrend=False, scaling='density', mode='psd')
        view = compute_spectrogram(self.signal, sample_rate_hz=64, nperseg=16)
        np.testing.assert_allclose(view.frequency_hz, f)
        np.testing.assert_allclose(view.time_s, t)
        np.testing.assert_allclose(view.power, p)
        self.assertFalse(view.power.flags.writeable)
        self.assertAlmostEqual(np.sum(view.power[:, 0]) * view.frequency_step_hz, 0.5)
        self.window.plotspec(self.signal, self.plot)
        np.testing.assert_allclose(self.image().image, view.power_db)

    def test_iq_layouts_keep_negative_frequency_and_input(self):
        iq = np.exp(-2j * np.pi * 8 * np.arange(64) / 64)
        for data in (iq, np.stack([iq.real, iq.imag]), np.column_stack([iq.real, iq.imag])):
            with self.subTest(shape=data.shape):
                before = data.copy()
                self.window.plotspec(data, self.plot)
                image = self.image()
                self.assertEqual(image.image.shape, (16, 7))
                row = np.argmax(image.image[:, 0])
                self.assertAlmostEqual(image.mapToParent(QPointF(0.5, row + 0.5)).y(), -8)
                np.testing.assert_array_equal(data, before)

    def test_pixel_centres_match_axes_even_single_frame_or_bin(self):
        for count, window in ((64, 16), (16, 16), (1, 1)):
            view = compute_spectrogram(np.ones(count), sample_rate_hz=64, nperseg=window)
            image = render_spectrogram(self.plot.plotItem, view)
            for x, y in ((0, 0), (len(view.time_s) - 1, len(view.frequency_hz) - 1)):
                p = image.mapToParent(QPointF(x + 0.5, y + 0.5))
                self.assertAlmostEqual(p.x(), view.time_s[x])
                self.assertAlmostEqual(p.y(), view.frequency_hz[y])
            self.assertGreater(image.mapRectToParent(image.boundingRect()).width(), 0)

    def test_global_image_order_cannot_transpose_time_and_frequency(self):
        old = pg.getConfigOption('imageAxisOrder')
        try:
            for order in ('row-major', 'col-major'):
                pg.setConfigOption('imageAxisOrder', order)
                self.window.plotspec(self.signal, self.plot)
                image = self.image()
                self.assertEqual(image.axisOrder, 'row-major')
                self.assertEqual(image.image.shape, (9, 7))
        finally:
            pg.setConfigOption('imageAxisOrder', old)

    def test_colour_levels_are_explicit_and_zero_energy_is_dark(self):
        self.window.plotspec(self.signal, self.plot)
        image = self.image()
        peak = image.image.max()
        np.testing.assert_allclose(image.getLevels(), [peak - 80, peak])
        self.window.plotspec(np.zeros(64), self.plot)
        image = self.image()
        self.assertTrue(np.isfinite(image.image).all())
        self.assertTrue((image.image <= image.getLevels()[0]).all())
        self.assertIn('非 dBm', self.plot.plotItem.titleLabel.text)

    def test_rate_change_and_no_filename_override(self):
        for filename in ('仿真.npy', '实采A.npy', 'anything.npy'):
            self.window.filename = [filename]
            self.window.lineEdit_Fs_2.setText('128')
            self.window.plotspec(self.signal, self.plot)
            image = self.image()
            row = np.argmax(image.image[:, 0])
            self.assertAlmostEqual(image.mapToParent(QPointF(0.5, row + 0.5)).y(), 16)

    def test_multifile_selection_before_conversion_and_default_first(self):
        combo = self.window.comboBox
        combo.blockSignals(True)
        combo.addItems(['first', 'second'])
        self.window.filename = ['a.npy', 'b.npy']
        for index, count in ((1, 128), (0, 64), (-1, 64)):
            combo.setCurrentIndex(index)
            self.window.plotspec([self.signal, np.zeros(128)], self.plot)
            self.assertEqual(self.image().image.shape[1], (count - 16) // 8 + 1)
        combo.blockSignals(False)

    def test_home_button_is_independent_of_recognition_state(self):
        self.window.filename = ['other.npy', 'another.npy']
        self.window.home_data = self.signal
        self.window.home_file = '仿真.npy'
        self.window.radioButton_time_1.setChecked(True)
        self.window.radioButton_TF_1.setChecked(True)
        image = self.image(self.window.label_signalshow_1)
        self.assertEqual(image.image.shape, (17, 3))
        row = np.argmax(image.image[:, 0])
        self.assertAlmostEqual(image.mapToParent(QPointF(0.5, row + 0.5)).y(), 16)

    def test_recognition_button_uses_recognition_parameters(self):
        self.window.data = self.signal
        self.window.radioButton_time_3.setChecked(True)
        self.window.radioButton_TF_3.setChecked(True)
        self.assertEqual(self.image().image.shape, (9, 7))
        # The secondary feature slot must not parse invalid text outside its try block.
        self.window.radioButton_time_3.setChecked(True)
        self.window.lineEdit_Fs_2.setText('')
        self.window.radioButton_TF_3.setChecked(True)
        self.assertFalse(self.plot.plotItem.items)
        self.assertIn('时频图无法显示', self.window.statusbar.currentMessage())

    def test_invalid_input_clears_stale_plot_and_reports_reason(self):
        for data, rate, window in ((None, 64, 16), ([], 64, 16),
                                  ([1, np.nan], 64, 2), ([1, np.inf], 64, 2),
                                  (np.ones((3, 8)), 64, 4), (self.signal, '', 16),
                                  (self.signal, 0, 16), (self.signal, 'nan', 16),
                                  (self.signal, 64, 65), (self.signal, 64, 0),
                                  (self.signal, 64, '1.5'), (self.signal, 64, True)):
            with self.subTest(rate=rate, window=window):
                self.window.plotspec(self.signal, self.plot)
                self.window.plotspec(data, self.plot, rate, window)
                self.assertFalse(self.plot.plotItem.items)
                self.assertIn('时频图', self.window.statusbar.currentMessage())

    def test_legacy_row_and_no_filename(self):
        self.window.filename = []
        self.window.plotspec(self.signal[None, :], self.plot)
        self.assertEqual(self.image().image.shape, (9, 7))

    def test_repeated_switch_resets_axes_and_does_not_accumulate_images(self):
        for _ in range(3):
            box = self.plot.getViewBox()
            box.setLimits(xMin=1, xMax=101, yMin=1, yMax=100)
            self.plot.setLogMode(x=True, y=True)
            box.invertY(True)
            box.invertX(True)
            box.setLimits(xMin=100, xMax=101, minXRange=1, maxXRange=1)
            self.window.plotspec(self.signal, self.plot)
            self.image()
            self.assertFalse(box.state['yInverted'])
            self.assertFalse(box.state['xInverted'])
            self.assertIsNone(box.state['limits']['xLimits'][0])
            self.assertEqual(self.plot.getAxis('bottom').labelUnits, 's')
            self.assertFalse(self.plot.getAxis('bottom').logMode)
            render_spectrum(self.plot, self.signal, sample_rate_hz=64)
            self.assertFalse(any(isinstance(i, pg.ImageItem) for i in self.plot.plotItem.items))
            self.assertEqual(self.plot.getAxis('bottom').labelUnits, 'Hz')
            render_waveform(self.plot, self.signal)
            self.assertEqual(self.plot.getAxis('bottom').labelUnits, '')

    def test_window_and_overlap_contract(self):
        for kwargs in ({'nperseg': 0}, {'nperseg': 3.5}, {'nperseg': True},
                       {'nperseg': 16, 'noverlap': -1}, {'nperseg': 16, 'noverlap': 16}):
            with self.assertRaises(SignalViewError):
                compute_spectrogram(self.signal, sample_rate_hz=64, **kwargs)
        view = compute_spectrogram(self.signal, sample_rate_hz=64, nperseg=16, noverlap=0)
        self.assertEqual(view.time_step_s, 0.25)
        self.assertEqual(view.frequency_step_hz, 4)


if __name__ == '__main__':
    unittest.main()
