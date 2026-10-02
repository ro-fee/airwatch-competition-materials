"""Numerical, unit and real Qt acceptance for the bispectrum migration."""
import os
import time
import unittest
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['PYQTGRAPH_QT_LIB']='PyQt5'
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
import main
from airwatch.analysis.bispectrum_views import compute_bispectrum
from airwatch.ui.plots.bispectrum_renderer import render_bispectrum_image


class BispectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def wait_for_feature(self, window):
        end=time.monotonic()+10
        while window.home_feature_task.busy and time.monotonic()<end:
            QTest.qWait(10)
        self.assertFalse(window.home_feature_task.busy)

    def signal(self,n=517):
        t=np.arange(n)
        return np.cos(.13*t)+.4*np.cos(.26*t+.2)

    def test_cumulants_match_independent_scalar_sum(self):
        x=self.signal(); before=x.copy(); v=compute_bispectrum(x,100)
        rows=x[:510].reshape(10,51); rows=rows-rows.mean(axis=1,keepdims=True)
        for i,j in ((0,0),(1,0),(20,3),(20,20),(7,14)):
            a,b=max(i,j),min(i,j)
            expected=sum(sum(row[k]*row[k+b]*row[k+a] for k in range(51-a))/51 for row in rows)/10
            self.assertAlmostEqual(v.cumulant[i,j],expected,places=12)
        np.testing.assert_array_equal(x,before)
        self.assertFalse(v.magnitude.flags.writeable)
        self.assertFalse(v.frequency_hz.flags.writeable)

    def test_rate_time_tail_and_shape(self):
        a=compute_bispectrum(self.signal(),100,start_sample=100)
        b=compute_bispectrum(self.signal(),200,start_sample=100)
        self.assertEqual(a.magnitude.shape,(128,128))
        np.testing.assert_array_equal(a.magnitude,b.magnitude)
        np.testing.assert_allclose(b.frequency_hz,a.frequency_hz*2)
        self.assertEqual(a.used_samples,510); self.assertEqual(a.dropped_samples,7)
        self.assertEqual(a.start_s,1); self.assertEqual(a.end_s,6.1)
        self.assertEqual(a.frequency_hz[0],-50)

    def test_gain_cubed_zero_constant_and_offset(self):
        x=self.signal(); a=compute_bispectrum(x,100)
        np.testing.assert_allclose(compute_bispectrum(2*x,100).magnitude,8*a.magnitude,atol=1e-10)
        np.testing.assert_allclose(compute_bispectrum(x+3,100).magnitude,a.magnitude,atol=1e-10)
        for x in (np.zeros(210),np.ones((1,210))):
            self.assertTrue((compute_bispectrum(x,100).magnitude==0).all())

    def test_bad_input_is_rejected(self):
        for x,fs in (([],1),(np.ones(209),1),(np.ones(65537),1),
                     (np.ones((2,256)),1),(np.ones(256,dtype=complex),1),
                     (np.full(256,np.nan),1),(self.signal(),0),(self.signal(),float('nan'))):
            with self.assertRaises(ValueError): compute_bispectrum(x,fs)
        with self.assertRaises(ValueError): compute_bispectrum(self.signal(),1,start_sample=-1)

    def test_render_has_no_disk_io_or_pyplot_figures(self):
        before=plt.get_fignums()
        with patch.object(plt,'savefig',side_effect=AssertionError('write')), \
             patch.object(Image,'open',side_effect=AssertionError('read')):
            for x in (self.signal(),np.zeros(210)):
                self.assertFalse(render_bispectrum_image(compute_bispectrum(x,100)).isNull())
        self.assertEqual(plt.get_fignums(),before)

    def test_main_uses_home_selection_and_current_rate(self):
        w=main.MyWindow(); self.addCleanup(w.close)
        w.home_data=self.signal(1024); w.data=np.zeros(1024)
        w.lineEdit_Fs_1.setText('100'); w.str2_fs='wrong'
        w.radioButton_time_1.setChecked(True); w.label_signalshow_1.setXRange(100,616,padding=0)
        calls=[]
        def compute(data,fs,**kw):
            calls.append((data.copy(),fs,kw)); return compute_bispectrum(data,fs,**kw)
        with patch('airwatch.analysis.home_features.compute_bispectrum',side_effect=compute):
            w.features('双谱特征'); self.wait_for_feature(w)
            w.features('双谱特征'); self.wait_for_feature(w)
        np.testing.assert_array_equal(calls[0][0],w.home_data[100:617])
        self.assertEqual(calls[0][1],100.0); self.assertEqual(calls[0][2]['start_sample'],100)
        self.assertFalse(w.label_featureshow_1.pixmap().isNull())
        self.assertFalse(w.label_featureshow_2.pixmap().isNull())
        slot=w.num; target=w.label_featureshow_1 if slot%2==0 else w.label_featureshow_2
        w.home_data=np.ones((2,512)); w.features('双谱特征')
        self.assertEqual(w.num,slot); self.assertIn('I/Q',target.text())


if __name__=='__main__': unittest.main()
