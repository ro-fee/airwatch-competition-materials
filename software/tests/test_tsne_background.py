"""Small-sample numerical and deterministic Qt lifecycle tests."""
import os
import time
import threading
import unittest
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['PYQTGRAPH_QT_LIB']='PyQt5'
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from airwatch.analysis.tsne_views import prepare_tsne, compute_tsne, TsneView
from airwatch.ui.tsne_background import TsneTask
import main


class TsneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def source(self):
        return prepare_tsne(np.arange(24).reshape(6,4),['a','b'],[2,4])

    def wait(self, predicate):
        end=time.monotonic()+10
        while not predicate() and time.monotonic()<end:
            QTest.qWait(10)
        self.assertTrue(predicate())

    def fake(self, source):
        return TsneView(np.arange(12).reshape(6,2),source.file_ids,source.file_names,1.0,42)

    def test_snapshot_mapping_and_no_mutation(self):
        source=self.source()
        np.testing.assert_array_equal(source.file_ids,[0,0,1,1,1,1])
        self.assertFalse(source.features.flags.writeable)
        self.assertFalse(source.file_ids.flags.writeable)
        data=np.arange(8).reshape(2,4)
        source=prepare_tsne(data,['a'],None); data[:]=0
        self.assertNotEqual(source.features.sum(),0)

    def test_bad_input_and_missing_mapping(self):
        for data in ([],np.ones((1,4)),np.ones((3,4)),np.full((3,4),np.nan),np.ones((2001,2))):
            with self.assertRaises(ValueError): prepare_tsne(data,['a'],None)
        with self.assertRaises(ValueError): prepare_tsne(np.arange(24).reshape(6,4),['a','b'],None)

    def test_real_tsne_two_samples_and_determinism(self):
        source=prepare_tsne(np.array([[0.,1.,2.],[2.,0.,1.]]),['a'],None)
        a=compute_tsne(source); b=compute_tsne(source)
        self.assertEqual(a.coordinates.shape,(2,2))
        self.assertTrue(np.isfinite(a.coordinates).all())
        np.testing.assert_array_equal(a.coordinates,b.coordinates)
        self.assertLess(a.perplexity,2)

    def test_worker_runs_off_ui_thread(self):
        task=TsneTask(); results=[]; ids=[]
        def compute(source):
            ids.append(threading.get_ident()); return self.fake(source)
        task.completed.connect(results.append)
        with patch('airwatch.ui.tsne_background.compute_tsne',side_effect=compute):
            task.start(self.source()); self.wait(lambda:not task.busy)
        self.assertEqual(len(results),1)
        self.assertNotEqual(ids[0],threading.get_ident())

    def test_cancel_blocks_overlap_and_late_result(self):
        task=TsneTask(); gate=threading.Event(); entered=threading.Event(); results=[]
        task.completed.connect(results.append)
        def compute(source):
            entered.set(); gate.wait(5); return self.fake(source)
        with patch('airwatch.ui.tsne_background.compute_tsne',side_effect=compute):
            try:
                task.start(self.source()); self.wait(entered.is_set)
                task.cancel()
                with self.assertRaises(ValueError): task.start(self.source())
                task._result(task.token-1,self.fake(self.source()))
                self.assertEqual(results,[])
            finally:
                gate.set(); self.wait(lambda:not task.busy)
        self.assertEqual(results,[])

    def test_worker_error_and_restart_cleanup(self):
        task=TsneTask(); errors=[]; task.failed.connect(errors.append)
        with patch('airwatch.ui.tsne_background.compute_tsne',side_effect=ValueError('bad input')):
            task.start(self.source()); self.wait(lambda:not task.busy)
        self.assertEqual(errors,['bad input'])
        with patch('airwatch.ui.tsne_background.compute_tsne',side_effect=self.fake):
            task.start(self.source()); self.wait(lambda:not task.busy)
        self.assertIsNone(task.thread)

    def test_page_entry_uses_current_features_no_file_fallback(self):
        w=main.MyWindow(); self.addCleanup(w.close)
        w.filename=['a','b']; w.feature_map=self.source().features; w.feature_row_counts=[2,4]
        with patch('airwatch.ui.tsne_background.compute_tsne',side_effect=self.fake), patch.object(np,'load',side_effect=AssertionError('disk')):
            w.comboBox_choose_feature.setCurrentText('tsne')
            self.wait(lambda:not w.tsne_task.busy)
        points=[i for i in w.label_featuremap.plotItem.items if isinstance(i,pg.ScatterPlotItem)]
        self.assertEqual(len(points),1)
        self.assertEqual(len(points[0].points()),6)
        self.assertIn('文件 2',points[0].points()[2].data())
        self.assertEqual(w.label_featuremap.getAxis('bottom').labelUnits,'')

    def test_switch_task_invalidates_running_result_and_close_waits(self):
        w=main.MyWindow(); gate=threading.Event(); entered=threading.Event()
        w.filename=['a','b']; w.feature_map=self.source().features; w.feature_row_counts=[2,4]
        def compute(source):
            entered.set(); gate.wait(5); return self.fake(source)
        with patch('airwatch.ui.tsne_background.compute_tsne',side_effect=compute):
            try:
                w.comboBox_choose_feature.setCurrentText('tsne'); self.wait(entered.is_set)
                w._on_task_changed()
                from PyQt5.QtGui import QCloseEvent
                event=QCloseEvent(); w.closeEvent(event)
                self.assertFalse(event.isAccepted())
            finally:
                gate.set(); self.wait(lambda:not w.tsne_task.busy)
                self.assertFalse(w.label_featuremap.plotItem.items)
                w.close()


if __name__=='__main__': unittest.main()
