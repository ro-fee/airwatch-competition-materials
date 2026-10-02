"""Real optional PyEMD backend acceptance, separate from lifecycle stubs."""
import os
import time
import unittest
import importlib.util
import numpy as np
from scipy.signal import chirp
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['PYQTGRAPH_QT_LIB']='PyQt5'
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from airwatch.analysis.hht_views import prepare_hht, compute_hht


@unittest.skipUnless(importlib.util.find_spec('PyEMD'), 'optional EMD-signal not installed')
class RealHHTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def test_two_tones_reconstruct_and_recover_frequencies(self):
        t=np.arange(2048)/1024
        x=np.cos(2*np.pi*160*t)+.5*np.cos(2*np.pi*32*t)
        result=compute_hht(prepare_hht(x,1024,128))
        np.testing.assert_allclose(sum(c.values for c in result.components[1:]),x,atol=1e-8)
        self.assertEqual(result.components[-1].label,'Residue')
        self.assertAlmostEqual(np.nanmedian(result.components[1].frequency_hz[128:-128]),160,delta=2)
        self.assertAlmostEqual(np.nanmedian(result.components[2].frequency_hz[128:-128]),32,delta=2)
        self.assertAlmostEqual(result.time_s[0],.125)

    def test_chirp_and_monotonic_trend(self):
        t=np.arange(2048)/1024
        x=chirp(t,f0=40,t1=t[-1],f1=160)
        v=compute_hht(prepare_hht(x,1024))
        c=v.components[1]
        expected=40+120*c.frequency_time_s/t[-1]
        self.assertLess(np.nanmedian(np.abs(c.frequency_hz[128:-128]-expected[128:-128])),3)
        v=compute_hht(prepare_hht(t,1024))
        np.testing.assert_allclose(sum(c.values for c in v.components[1:]),t,atol=1e-8)

    def test_repeatable_and_max_input(self):
        x=np.cos(2*np.pi*64*np.arange(8192)/1024)
        a=compute_hht(prepare_hht(x,1024)); b=compute_hht(prepare_hht(x,1024))
        self.assertEqual(len(a.components),len(b.components))
        for left,right in zip(a.components,b.components):
            np.testing.assert_array_equal(left.values,right.values)
            np.testing.assert_allclose(left.frequency_hz,right.frequency_hz,equal_nan=True)

    def test_real_qt_background_entry_and_render(self):
        import main
        w=main.MyWindow(); self.addCleanup(w.close)
        w.home_data=np.cos(2*np.pi*64*np.arange(512)/1024)
        w.lineEdit_Fs_1.setText('1024')
        w.radioButton_time_1.setChecked(True); w.label_signalshow_1.setXRange(0,511,padding=0)
        before=w.num; target=w.label_featureshow_1 if before%2==0 else w.label_featureshow_2
        w.features('HHT特征')
        end=time.monotonic()+15
        while w.home_feature_task.busy and time.monotonic()<end: QTest.qWait(10)
        self.assertFalse(w.home_feature_task.busy)
        self.assertEqual(w.num,before+1,w.statusbar.currentMessage())
        self.assertFalse(target.pixmap().isNull())


if __name__=='__main__': unittest.main()
