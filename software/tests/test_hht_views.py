"""HHT contracts and unified home-feature background lifecycle acceptance."""
import os
import threading
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

from airwatch.analysis.hht_views import (
    HHTUnavailable,
    compute_hht,
    load_hht_backend,
    prepare_hht,
)
from airwatch.analysis.home_features import (
    HomeFeatureRequest,
    HomeFeatureResult,
    prepare_home_feature_request,
)
from airwatch.ui.home_feature_background import HomeFeatureTask
from airwatch.ui.plots.hht_renderer import render_hht_image
import main


class FakeEMD:
    def __init__(self,x): self.x=x
    def decompose(self): return np.stack([self.x,np.zeros_like(self.x)])


def fake_frequency(x):
    # Known stub solely for checking conversion and data contracts.
    return np.full(len(x)-2,.125), np.arange(1,len(x)-1)


class HHTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def source(self): return prepare_hht(np.cos(np.arange(64)*2*np.pi/8),128,128)

    def request(self):
        source=self.source()
        return HomeFeatureRequest(
            'hht',source.signal,source.sample_rate_hz,source.start_sample
        )

    def wait(self,predicate):
        end=time.monotonic()+10
        while not predicate() and time.monotonic()<end: QTest.qWait(10)
        self.assertTrue(predicate())

    def test_input_contract_and_immutability(self):
        source=self.source(); self.assertFalse(source.signal.flags.writeable)
        for x,fs in ((np.ones(64),128),(np.zeros(64),128),(np.ones((2,64)),128),
                     (np.ones(64,complex),128),(np.arange(31),128),(np.arange(8193),128),
                     (np.arange(64),0),(np.full(64,np.nan),128)):
            with self.assertRaises(ValueError): prepare_hht(x,fs)

    def test_backend_import_failure_is_explicit(self):
        with patch.dict('sys.modules',{'PyEMD':None}):
            with self.assertRaisesRegex(HHTUnavailable,'EMD-signal'): load_hht_backend()

    def test_stub_components_units_zero_residual_and_no_disk(self):
        with patch('airwatch.analysis.hht_views.load_hht_backend',return_value=(FakeEMD,fake_frequency)):
            view=compute_hht(self.source())
        self.assertEqual(len(view.components),3)
        np.testing.assert_allclose(view.components[0].frequency_hz,16)
        np.testing.assert_allclose(view.components[0].frequency_time_s,(np.arange(1,63)+128)/128)
        self.assertEqual(view.components[-1].frequency_hz.size,0)
        self.assertFalse(view.components[1].values.flags.writeable)
        with patch.object(Image,'open',side_effect=AssertionError('read')), patch.object(plt,'savefig',side_effect=AssertionError('write')):
            self.assertFalse(render_hht_image(view).isNull())

    def test_bad_backend_result_and_cancellation_rejected(self):
        class Bad:
            def __init__(self,x): pass
            def decompose(self): return np.zeros((2,3))
        with patch('airwatch.analysis.hht_views.load_hht_backend',return_value=(Bad,fake_frequency)):
            with self.assertRaises(ValueError): compute_hht(self.source())
        with self.assertRaises(InterruptedError): compute_hht(self.source(),cancelled=lambda:True)

    def test_worker_off_thread_error_and_cleanup(self):
        task=HomeFeatureTask(); errors=[]; ids=[]
        task.failed.connect(errors.append)
        def fail(*args,**kw): ids.append(threading.get_ident()); raise ValueError('backend failed')
        with patch('airwatch.ui.home_feature_background.compute_home_feature',side_effect=fail):
            task.start(self.request()); self.wait(lambda:not task.busy)
        self.assertEqual(errors,['backend failed'])
        self.assertNotEqual(ids[0],threading.get_ident())

    def test_cancel_discards_queued_result_and_refuses_overlap(self):
        task=HomeFeatureTask(); gate=threading.Event(); entered=threading.Event(); results=[]
        task.completed.connect(results.append)
        def slow(*args,**kw): entered.set(); gate.wait(5); return object()
        with patch('airwatch.ui.home_feature_background.compute_home_feature',side_effect=slow):
            try:
                task.start(self.request()); self.wait(entered.is_set); task.cancel()
                with self.assertRaises(ValueError): task.start(self.request())
                task._completed(task.token-1,object())
            finally: gate.set(); self.wait(lambda:not task.busy)
        self.assertEqual(results,[])

    def test_page_missing_dependency_no_old_image_or_slot_advance(self):
        w=main.MyWindow(); self.addCleanup(w.close)
        w.home_data=self.source().signal; before=w.num
        target=w.label_featureshow_1 if before%2==0 else w.label_featureshow_2
        with patch('airwatch.analysis.hht_views.load_hht_backend',side_effect=HHTUnavailable('缺少 EMD-signal')), patch.object(Image,'open',side_effect=AssertionError('disk')):
            w.features('HHT特征'); self.wait(lambda:not w.home_feature_task.busy)
        self.assertEqual(w.num,before); self.assertIn('EMD-signal',target.text())

    def test_page_success_and_close_while_busy(self):
        """A running decomposition must survive a rejected close until it exits."""
        w=main.MyWindow(); w.home_data=self.source().signal
        w.radioButton_time_1.setChecked(True); w.label_signalshow_1.setXRange(0,63,padding=0)
        gate=threading.Event(); entered=threading.Event()
        def slow(request,**kw):
            entered.set(); gate.wait(5)
            with patch('airwatch.analysis.hht_views.load_hht_backend',return_value=(FakeEMD,fake_frequency)):
                view=compute_hht(prepare_hht(request.signal,request.sample_rate_hz,request.start_sample))
            return HomeFeatureResult('hht',request,view)
        with patch('airwatch.ui.home_feature_background.compute_home_feature',side_effect=slow):
            try:
                w.features('HHT特征'); self.wait(entered.is_set)
                from PyQt5.QtGui import QCloseEvent
                event=QCloseEvent(); w.closeEvent(event)
                self.assertFalse(event.isAccepted())
            finally:
                gate.set(); self.wait(lambda:not w.home_feature_task.busy); w.close()
        self.assertIsNone(w.home_feature_target)

    def test_page_stub_success_renders_and_advances_slot(self):
        w=main.MyWindow(); self.addCleanup(w.close)
        w.home_data=self.source().signal; w.lineEdit_Fs_1.setText('128')
        w.radioButton_time_1.setChecked(True); w.label_signalshow_1.setXRange(0,63,padding=0)
        before=w.num; target=w.label_featureshow_1 if before%2==0 else w.label_featureshow_2
        with patch('airwatch.analysis.hht_views.load_hht_backend',return_value=(FakeEMD,fake_frequency)):
            w.features('HHT特征'); self.wait(lambda:not w.home_feature_task.busy)
        self.assertEqual(w.num,before+1)
        self.assertFalse(target.pixmap().isNull())


if __name__=='__main__': unittest.main()
