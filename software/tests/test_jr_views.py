"""J/R formula, degeneracy, axes and legacy page acceptance."""
import os
import time
import unittest
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['PYQTGRAPH_QT_LIB']='PyQt5'
import numpy as np
from scipy.fftpack import hilbert
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
import main
from airwatch.analysis.jr_views import compute_jr
from airwatch.ui.plots.jr_renderer import render_jr_image


class JRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def wait_for_feature(self, window):
        end=time.monotonic()+10
        while window.home_feature_task.busy and time.monotonic()<end:
            QTest.qWait(10)
        self.assertFalse(window.home_feature_task.busy)

    def signal(self): return np.cos(np.arange(103)*.7)+.3*np.sin(np.arange(103)*.21)

    def test_matches_legacy_formulas_with_correct_named_fields(self):
        data=self.signal(); before=data.copy(); view=compute_jr(data,1000)
        j=[]; r=[]
        for y in data[:100].reshape(10,10):
            z=np.sqrt(y**2+hilbert(y)**2)
            m2=np.mean(z**2); m4=np.mean(z**4); ps=np.mean(y**2)/2
            r.append(abs((m4-m2**2)/m2**2))
            j.append(abs((m4-2*m2**2)/(4*ps**2)))
        np.testing.assert_allclose(view.j,j,atol=1e-12)
        np.testing.assert_allclose(view.r,r,atol=1e-12)
        np.testing.assert_array_equal(data,before)
        self.assertFalse(view.j.flags.writeable)

    def test_extreme_gain_invariance(self):
        base=compute_jr(self.signal(),1000)
        for gain in (1e-250,1e250,-3):
            result=compute_jr(self.signal()*gain,1000)
            np.testing.assert_allclose(result.j,base.j,atol=1e-12)
            np.testing.assert_allclose(result.r,base.r,atol=1e-12)

    def test_zero_windows_excluded_and_constants_are_valid(self):
        view=compute_jr(np.r_[np.zeros(10),np.ones(10),np.zeros(10)],1000)
        np.testing.assert_array_equal(view.window_indices,[1])
        np.testing.assert_array_equal(view.zero_window_indices,[0,2])
        np.testing.assert_allclose(view.j,[1]); np.testing.assert_allclose(view.r,[0])
        zero=compute_jr(np.zeros((1,20)),1000)
        self.assertEqual(len(zero.j),0); self.assertEqual(zero.total_windows,2)
        self.assertFalse(render_jr_image(zero).isNull())

    def test_time_window_and_tail_are_explicit(self):
        view=compute_jr(self.signal(),1050,start_sample=21)
        self.assertEqual(view.window_samples,10)
        self.assertEqual(view.dropped_samples,3)
        self.assertEqual(view.start_s,21/1050)
        self.assertEqual(view.end_s,121/1050)

    def test_bad_input_rejected(self):
        for data,fs in (([],1000),(np.ones(3),1000),(np.ones(20),199),
                        (np.ones(65537),1000),(np.ones((2,32)),1000),
                        (np.ones(20,dtype=complex),1000),(np.full(20,np.nan),1000),
                        (np.ones(20),float('inf'))):
            with self.assertRaises(ValueError): compute_jr(data,fs)
        with self.assertRaises(ValueError): compute_jr(self.signal(),1000,start_sample=-1)

    def test_render_axes_order_no_disk_or_pyplot_leak(self):
        view=compute_jr(self.signal(),1000); original=Axes.scatter; points=[]
        def capture(axis,x,y,*args,**kw):
            points.append((np.array(x),np.array(y)))
            return original(axis,x,y,*args,**kw)
        before=plt.get_fignums()
        with patch.object(Axes,'scatter',capture), \
             patch.object(plt,'savefig',side_effect=AssertionError('disk')), \
             patch.object(Image,'open',side_effect=AssertionError('disk')):
            self.assertFalse(render_jr_image(view).isNull())
        np.testing.assert_array_equal(points[0][0],view.j)
        np.testing.assert_array_equal(points[0][1],view.r)
        self.assertEqual(plt.get_fignums(),before)

    def test_home_selection_current_rate_and_alternation(self):
        w=main.MyWindow(); self.addCleanup(w.close)
        w.home_data=np.tile(self.signal(),4); w.data=np.zeros(412)
        w.radioButton_time_1.setChecked(True); w.lineEdit_Fs_1.setText('1000'); w.str2_fs='bad'
        w.label_signalshow_1.setXRange(20,122,padding=0)
        calls=[]
        def capture(data,fs,**kw):
            calls.append((data.copy(),fs,kw)); return compute_jr(data,fs,**kw)
        with patch('airwatch.analysis.home_features.compute_jr',side_effect=capture):
            w.features('J、R特征'); self.wait_for_feature(w)
            w.features('J、R特征'); self.wait_for_feature(w)
        np.testing.assert_array_equal(calls[0][0],w.home_data[20:123])
        self.assertEqual(calls[0][1],1000.0); self.assertEqual(calls[0][2]['start_sample'],20)
        self.assertFalse(w.label_featureshow_1.pixmap().isNull())
        self.assertFalse(w.label_featureshow_2.pixmap().isNull())
        slot=w.num; target=w.label_featureshow_1 if slot%2==0 else w.label_featureshow_2
        w.home_data=np.ones((2,512)); w.features('J、R特征')
        self.assertEqual(w.num,slot); self.assertIn('I/Q',target.text())


if __name__=='__main__': unittest.main()
