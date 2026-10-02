"""Regression checks for real Qt spectrum slots and migration neighbours."""

import os
import unittest
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
os.environ['QT_API'] = 'pyqt5'

import numpy as np
from PyQt5.QtWidgets import QApplication

import main
from airwatch.ui.plots import render_spectrum, render_waveform


class SpectrumIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = main.MyWindow()
        self.plot = self.window.label_signal
        self.window.filename = ['test.npy']
        self.window.lineEdit_Fs_2.setText('64')
        self.window.lineEdit_Fs_1.setText('128')
        self.t = np.arange(64) / 64
        self.signal = np.cos(2 * np.pi * 8 * self.t)
        self.dialogs = patch.object(main.QMessageBox, 'about')
        self.dialogs.start()
        self.addCleanup(self.dialogs.stop)
        self.addCleanup(self.window.close)

    def select_files(self, index):
        combo = self.window.comboBox
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(['first', 'second'])
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self.window.filename = ['first.npy', 'second.npy']

    def curve(self, plot=None):
        curves = (plot or self.plot).plotItem.listDataItems()
        self.assertEqual(len(curves), 1, 'spectrum slot must draw exactly one curve')
        return curves[0].getData()

    def test_real_spectrum_coordinates_and_amplitude(self):
        self.window.plotspectrum(self.signal, self.plot)
        x, y = self.curve()
        np.testing.assert_array_equal(x, np.arange(33))
        self.assertEqual(x[np.argmax(y)], 8)
        self.assertAlmostEqual(y[8], 0.5)
        self.assertEqual(self.plot.getAxis('bottom').labelUnits, 'Hz')

    def test_frequency_button_uses_real_slot(self):
        self.window.data = self.signal
        self.window.radioButton_time_3.setChecked(True)
        self.window.radioButton_PP_3.setChecked(True)
        self.app.processEvents()
        x, y = self.curve()
        self.assertEqual(x[np.argmax(y)], 8)

    def test_rate_change_changes_frequency_axis(self):
        self.window.lineEdit_Fs_2.setText('128')
        self.window.plotspectrum(self.signal, self.plot)
        x, y = self.curve()
        self.assertEqual(x[np.argmax(y)], 16)

    def test_missing_file_selection_index_does_not_use_last_file(self):
        self.select_files(-1)
        self.window.plotspectrum([self.signal, np.zeros(64)], self.plot)
        x, y = self.curve()
        np.testing.assert_array_equal(x, np.arange(33))
        self.assertEqual(x[np.argmax(y)], 8)
        self.assertAlmostEqual(y[8], 0.5)
        self.assertEqual(self.plot.getAxis('bottom').labelUnits, 'Hz')

    def test_iq_formats_keep_both_channels(self):
        iq = np.exp(-2j * np.pi * 8 * self.t)
        for source in (iq, np.array([iq.real, iq.imag]), np.column_stack([iq.real, iq.imag])):
            with self.subTest(shape=source.shape):
                before = source.copy()
                self.window.plotspectrum(source, self.plot)
                x, y = self.curve()
                np.testing.assert_array_equal(x, np.arange(-32, 32))
                self.assertEqual(x[np.argmax(y)], -8)
                self.assertAlmostEqual(y.max(), 1)
                np.testing.assert_array_equal(source, before)

    def test_single_channel_legacy_row(self):
        self.window.plotspectrum(self.signal[None, :], self.plot)
        x, y = self.curve()
        self.assertEqual(x[np.argmax(y)], 8)

    def test_multiple_files_select_before_array_conversion(self):
        signals = [self.signal, np.cos(2 * np.pi * 12 * np.arange(128) / 64)]
        for index, expected in ((1, 12), (0, 8), (-1, 8)):
            with self.subTest(index=index):
                self.select_files(index)
                self.window.plotspectrum(signals, self.plot)
                x, y = self.curve()
                self.assertEqual(x[np.argmax(y)], expected)

    def test_home_ignores_recognition_selection_and_rate(self):
        self.select_files(1)
        home = self.window.label_signalshow_1
        self.window.plotspectrum(self.signal, home)
        x, y = self.curve(home)
        self.assertEqual(x[np.argmax(y)], 16)
        self.assertEqual(x[-1], 64)

    def test_missing_filename_does_not_index_single_signal(self):
        self.window.filename = []
        self.window.plotspectrum(self.signal, self.plot)
        x, y = self.curve()
        self.assertEqual(x[np.argmax(y)], 8)

    def test_bad_sample_rate_clears_stale_spectrum_and_reports_chinese(self):
        for rate in ('', '0', '-1', 'nan', 'invalid'):
            with self.subTest(rate=rate):
                render_spectrum(self.plot, self.signal, sample_rate_hz=64)
                self.window.lineEdit_Fs_2.setText(rate)
                self.window.plotspectrum(self.signal, self.plot)
                self.assertFalse(self.plot.plotItem.listDataItems())
                self.assertIn('采样率', self.window.statusbar.currentMessage())

    def test_bad_shape_nan_and_empty_clear_stale_result(self):
        for source in ([], [1, np.nan], [1, np.inf], np.zeros((3, 8))):
            with self.subTest(shape=np.shape(source)):
                render_spectrum(self.plot, self.signal, sample_rate_hz=64)
                self.window.plotspectrum(source, self.plot)
                self.assertFalse(self.plot.plotItem.listDataItems())
                self.assertIn('频谱', self.window.statusbar.currentMessage())

    def test_no_input_is_nonfatal_and_clears_old_graph(self):
        render_spectrum(self.plot, self.signal, sample_rate_hz=64)
        self.window.plotspectrum(None, self.plot)
        self.assertFalse(self.plot.plotItem.listDataItems())

    def test_repeated_mode_switch_resets_axes_and_does_not_accumulate(self):
        for _ in range(3):
            self.plot.invertY(True)
            self.plot.setLogMode(x=True, y=True)
            self.plot.getViewBox().setLimits(xMin=0, xMax=4, yMin=0, yMax=2)
            render_spectrum(self.plot, np.exp(-2j * np.pi * 8 * self.t), sample_rate_hz=64)
            x, y = self.curve()
            self.assertEqual(x[np.argmax(y)], -8)
            self.assertFalse(self.plot.getViewBox().state['yInverted'])
            self.assertLess(self.plot.getViewBox().viewRange()[0][0], -8)
            render_waveform(self.plot, self.signal)
            self.assertEqual(self.plot.getAxis('bottom').labelUnits, '')
            self.window.plotspectrum(self.signal, self.plot)
            self.curve()

    def test_renderer_plotitem_db_zero_input(self):
        view = render_spectrum(self.plot.plotItem, np.zeros(8), sample_rate_hz=8, use_db=True)
        self.assertTrue(np.isfinite(view.power_db).all())
        np.testing.assert_allclose(self.curve()[1], view.power_db)

    def test_feature_waveform_single_file_not_indexed_twice(self):
        self.window.feature = [1]
        self.window.currentIndex = 0
        data = np.arange(16).reshape(2, 8)
        self.window.plotsig_featuremap(data, self.window.label_featuremap)
        x, y = self.curve(self.window.label_featuremap)
        np.testing.assert_array_equal(y, data[0])


if __name__ == '__main__':
    unittest.main()
