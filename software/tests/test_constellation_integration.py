"""Constellation input provenance, geometry and real generation-page acceptance."""
import os
import unittest
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
import numpy as np
import pyqtgraph as pg
from scipy.signal import hilbert
from PyQt5.QtWidgets import QApplication
import main
from airwatch.analysis.signal_transforms import compute_constellation
from airwatch.analysis.generated_views import select_generated_sample
from airwatch.ui.plots.pyqtgraph_renderer import render_constellation, render_waveform


class ConstellationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.w = main.MyWindow()
        self.plot = self.w.label_genesig
        self.addCleanup(self.w.close)
        self.x = np.cos(2*np.pi*np.arange(64)/8)
        self.w.signals = [np.stack([self.x, 2*self.x])[:,None,:],
                          np.stack([3*self.x, 4*self.x])[:,None,:]]
        self.w.gen_labels = ['A','B']
        self.w.comboBox_showclass.blockSignals(True)
        self.w.comboBox_showclass.clear()
        self.w.comboBox_showclass.addItems(['全选','A','B'])
        self.w.comboBox_showclass.setCurrentText('A')
        self.w.comboBox_showclass.blockSignals(False)
        self.w.lineEdit_switchsignal.setText('1')

    def points(self):
        items = [i for i in self.plot.plotItem.items if isinstance(i,pg.ScatterPlotItem)]
        self.assertEqual(len(items),1)
        return items[0].getData()

    def test_iq_renderer_preserves_geometry_and_input(self):
        iq = np.array([1+2j,-1-2j,3-4j])
        for data in (iq, np.stack([iq.real,iq.imag]), np.column_stack([iq.real,iq.imag])):
            before = data.copy()
            self.w.xzt(data,self.plot)
            i,q = self.points()
            np.testing.assert_array_equal(i,iq.real)
            np.testing.assert_array_equal(q,iq.imag)
            np.testing.assert_array_equal(data,before)
            self.assertEqual(self.plot.getViewBox().state['aspectLocked'],1)

    def test_direct_entry_never_reads_demo_or_generation_files(self):
        with patch.object(np,'load',side_effect=AssertionError('unexpected disk read')):
            self.w.xzt(self.x+1j*self.x,self.plot)
            self.points()
            self.w.radioButton_2.setChecked(True)
            i,q = self.points()
            np.testing.assert_allclose(i,self.x)
            np.testing.assert_allclose(q,hilbert(self.x).imag)

    def test_raw_real_requires_explicit_compatibility(self):
        with self.assertRaises(ValueError):
            compute_constellation(self.x)
        view = compute_constellation(self.x,analytic_from_real=True)
        self.assertEqual(view.source_kind,'analytic_real')
        np.testing.assert_allclose(view.q,hilbert(self.x).imag)
        render_constellation(self.plot,view)
        self.assertIn('非真实',self.plot.plotItem.titleLabel.text)

    def test_class_and_sample_switch_keep_constellation_mode(self):
        self.w.radioButton_2.setChecked(True)
        self.w.comboBox_showclass.setCurrentText('B')
        self.w.lineEdit_switchsignal.setText('2')
        self.w.switch_single_sig(self.w.signals,self.plot)
        i,q = self.points()
        np.testing.assert_allclose(i,4*self.x)
        np.testing.assert_allclose(q,hilbert(4*self.x).imag)
        self.assertIn('B 第 2',self.w.statusbar.currentMessage())

    def test_fifteen_samples_are_not_mistaken_for_fifteen_classes(self):
        batches = [np.tile(self.x,(15,1,1))]
        label,sample = select_generated_sample(batches,['A'],'全选',15)
        self.assertEqual(label,'A')
        np.testing.assert_array_equal(sample,self.x)
        self.assertFalse(sample.flags.writeable)

    def test_invalid_selection_does_not_fall_back_or_leave_old_graph(self):
        self.w.radioButton_2.setChecked(True)
        for number in ('0','3','bad'):
            self.w.lineEdit_switchsignal.setText(number)
            with patch.object(main.QMessageBox,'warning'):
                self.w.switch_gene_sig(self.w.signals,self.plot)
            self.assertFalse(self.plot.plotItem.items)
        for labels, name in ((['A','B'],'C'),([], 'A')):
            with self.assertRaises(ValueError):
                select_generated_sample(self.w.signals,labels,name,1)

    def test_bad_direct_data_clears_old_scatter(self):
        for data in ([],[np.nan],np.ones((3,4)),self.x):
            self.w.xzt(self.x+1j*self.x,self.plot)
            self.w.xzt(data,self.plot)
            self.assertFalse(self.plot.plotItem.items)
            self.assertIn('星座图无法显示',self.w.statusbar.currentMessage())

    def test_switching_back_restores_waveform_and_clears_units(self):
        for _ in range(3):
            self.w.radioButton_2.setChecked(True)
            self.points()
            self.w.radioButton_TF_2.setChecked(True)
            self.assertEqual(len(self.plot.plotItem.listDataItems()),2)
            self.assertFalse(any(isinstance(i,pg.ScatterPlotItem) for i in self.plot.plotItem.items))
            self.assertFalse(self.plot.getViewBox().state['aspectLocked'])
            self.assertEqual(self.plot.getAxis('bottom').labelUnits,'')

    def test_decimation_is_deterministic_and_does_not_normalize(self):
        iq = np.arange(10000)+1j*np.arange(10000)*2
        view = compute_constellation(iq,max_points=5000)
        indices = np.linspace(0,9999,5000,dtype=np.int64)
        np.testing.assert_array_equal(view.i,iq.real[indices])
        np.testing.assert_array_equal(view.q,iq.imag[indices])
        self.assertFalse(view.i.flags.writeable)

    def test_empty_session_radio_click_is_nonfatal_and_honest(self):
        self.w.signals=[]; self.w.gen_labels=[]
        self.w.radioButton_2.setChecked(True)
        self.assertFalse(self.plot.plotItem.items)
        self.assertIn('请先生成',self.w.statusbar.currentMessage())


if __name__ == '__main__':
    unittest.main()
