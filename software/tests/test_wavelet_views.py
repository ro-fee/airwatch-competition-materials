"""Wavelet numerical correctness, selection and disk-free legacy UI integration."""
import os
import time
import unittest
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['PYQTGRAPH_QT_LIB']='PyQt5'
import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
import main
from airwatch.analysis.wavelet_views import compute_wavelet
from airwatch.ui.plots.wavelet_renderer import render_wavelet_image


class WaveletViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def wait_for_feature(self, window):
        end=time.monotonic()+10
        while window.home_feature_task.busy and time.monotonic()<end:
            QTest.qWait(10)
        self.assertFalse(window.home_feature_task.busy)

    def test_reconstruction_length_and_components_even_and_odd(self):
        for n in (20,64,512,513):
            x=np.cos(np.arange(n)/5)+np.arange(n)/n
            before=x.copy()
            view=compute_wavelet(x,100,start_sample=10)
            self.assertEqual(view.level,min(5,pywt.dwt_max_level(n,7+1)))
            for y in view.approximations+view.details:
                self.assertEqual(len(y),n)
                self.assertFalse(y.flags.writeable)
            np.testing.assert_allclose(view.approximations[-1]+sum(view.details),x,atol=1e-10)
            np.testing.assert_allclose(view.time_s,(np.arange(n)+10)/100)
            np.testing.assert_array_equal(x,before)

    def test_invalid_shape_iq_rate_and_size_rejected(self):
        for x,fs in (([],10),(np.ones(13),10),(np.ones(65537),10),
                     (np.ones((2,32)),10),(np.ones(32,dtype=complex),10),
                     (np.ones(32),0),(np.full(32,np.nan),10)):
            with self.assertRaises(ValueError): compute_wavelet(x,fs)
        view=compute_wavelet(np.zeros((1,64)),100)
        self.assertTrue(all(np.all(y==0) for y in view.approximations+view.details))

    def test_renderer_is_disk_free_and_does_not_register_pyplot_figures(self):
        view=compute_wavelet(np.sin(np.arange(64)),64)
        before=plt.get_fignums()
        with patch.object(plt,'savefig',side_effect=AssertionError('write')), \
             patch.object(Image,'open',side_effect=AssertionError('read')):
            pix=render_wavelet_image(view)
        self.assertFalse(pix.isNull())
        self.assertEqual(plt.get_fignums(),before)

    def test_home_selection_current_rate_and_alternating_slots(self):
        w=main.MyWindow(); self.addCleanup(w.close)
        w.home_data=np.sin(np.arange(512)/3); w.data=np.zeros(512)
        w.lineEdit_Fs_1.setText('128'); w.str2_fs='wrong'
        w.radioButton_time_1.setChecked(True)
        w.label_signalshow_1.setXRange(20,83,padding=0)
        calls=[]
        def compute(data,fs,**kwargs):
            calls.append((data.copy(),fs,kwargs)); return compute_wavelet(data,fs,**kwargs)
        with patch('airwatch.analysis.home_features.compute_wavelet',side_effect=compute):
            w.features('小波特征'); self.wait_for_feature(w)
            w.features('小波特征'); self.wait_for_feature(w)
        np.testing.assert_array_equal(calls[0][0],w.home_data[20:84])
        self.assertEqual(calls[0][1],128.0)
        self.assertEqual(calls[0][2]['start_sample'],20)
        self.assertFalse(w.label_featureshow_1.pixmap().isNull())
        self.assertFalse(w.label_featureshow_2.pixmap().isNull())

    def test_error_clears_target_and_does_not_advance_slot(self):
        w=main.MyWindow(); self.addCleanup(w.close)
        w.home_data=np.sin(np.arange(512)); w.features('小波特征'); self.wait_for_feature(w)
        slot=w.num; target=w.label_featureshow_1 if slot%2==0 else w.label_featureshow_2
        w.home_data=np.ones((2,512)); w.features('小波特征')
        self.assertEqual(w.num,slot)
        self.assertIn('I/Q',target.text())
        self.assertTrue(target.pixmap() is None or target.pixmap().isNull())


if __name__=='__main__': unittest.main()
