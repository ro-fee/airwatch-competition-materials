"""Real file/dialog-boundary regressions: imports must not mix page identities."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PyQt5.QtWidgets import QApplication
import main


class HomeImportIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.w = main.MyWindow()
        self.addCleanup(self.w.close)
        self.w.lineEdit_Fs_1.setText('48000')
        self.w.radioButton_spectrum_1.setChecked(True)

    def load(self, name, array):
        path = Path(self.tmp.name) / name
        np.save(path, array)
        with patch.object(main.QFileDialog, 'getOpenFileName', return_value=(str(path), '')), \
             patch.object(main.QMessageBox, 'critical'):
            self.w.openfile()
        return path

    def test_home_import_preserves_recognition_input_identity(self):
        original = np.arange(512.)
        self.w.data = original
        self.w.filename = ['recognition.dat']
        self.w.data_file = 'recognition.dat'
        self.load('home.npy', np.ones(1024))
        self.assertIs(self.w.data, original)
        self.assertEqual(self.w.filename, ['recognition.dat'])
        self.assertEqual(self.w.data_file, 'recognition.dat')

    def test_failed_import_retains_previous_valid_identity_atomically(self):
        old = self.load('valid.npy', np.ones(1024))
        for bad in (np.array([]), np.array([1., np.nan]), np.zeros((3, 256))):
            with self.subTest(shape=bad.shape):
                self.load('invalid.npy', bad)
                self.assertEqual(self.w.home_file, str(old))
                self.assertEqual(self.w.label_filename_1.text(), old.name)
                np.testing.assert_array_equal(self.w.home_data, np.ones(1024))

    def test_iq_representations_have_same_imported_spectrum(self):
        iq = np.exp(1j * (np.pi / 4 + 2 * np.pi * np.arange(1024) / 4))
        peaks = []
        for value in (iq, np.array([iq.real, iq.imag]), np.column_stack([iq.real, iq.imag])):
            self.load('iq.npy', value)
            _, y = self.w.label_signalshow_1.plotItem.listDataItems()[0].getData()
            peaks.append(float(y.max()))
        np.testing.assert_allclose(peaks, [1., 1., 1.], atol=1e-12)

    def test_invalid_waveform_clears_previous_curves(self):
        plot = self.w.label_signalshow_1
        for bad in (None, np.array([1., np.nan]), np.zeros((3, 256))):
            with self.subTest(shape=np.shape(bad)):
                self.w.plotsig(np.ones(32), plot)
                self.w.plotsig(bad, plot)
                self.assertEqual(plot.plotItem.listDataItems(), [])

    def test_invalid_plot_parameters_are_not_reported_as_successful_render(self):
        self.w.lineEdit_Fs_1.setText('0')
        self.load('valid.npy', np.ones(1024))
        self.assertIn('采样率', self.w.statusbar.currentMessage())

    def test_zero_signal_is_valid_for_plotting(self):
        self.load('zero.npy', np.zeros(1024))
        _, y = self.w.label_signalshow_1.plotItem.listDataItems()[0].getData()
        np.testing.assert_array_equal(y, np.zeros(513))

    def test_import_does_not_silently_change_amplitude(self):
        self.load('quarter.npy', .25*np.sin(2*np.pi*np.arange(1024)/8))
        _, y = self.w.label_signalshow_1.plotItem.listDataItems()[0].getData()
        self.assertAlmostEqual(float(y.max()), .125)

    def test_single_sample_can_be_plotted_as_waveform(self):
        self.w.plotsig(np.array([.25]), self.w.label_signalshow_1)
        curves = self.w.label_signalshow_1.plotItem.listDataItems()
        self.assertEqual(len(curves), 1)
        np.testing.assert_array_equal(curves[0].getData()[1], [.25])


if __name__ == '__main__':
    unittest.main()
