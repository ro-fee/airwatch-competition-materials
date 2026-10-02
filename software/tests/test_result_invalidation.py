"""Failure/cancel publication tests using real workflows and controlled models."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
import main
from tests.test_generation_background import fake_workflow
from tests.test_historical_comparison_background import _workflow
from airwatch.workflows.historical_generation import HistoricalGenerationWorkflow


class ResultInvalidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.w=main.MyWindow(); self.addCleanup(self.w.close)
        p=patch.object(main.QMessageBox,'critical'); p.start(); self.addCleanup(p.stop)

    def wait(self,task):
        end=time.monotonic()+10
        while task.busy and time.monotonic()<end: QTest.qWait(5)
        self.assertFalse(task.busy)

    def test_failed_generation_cannot_keep_previous_session_as_current(self):
        self.w.generation_task.workflow=fake_workflow([])
        self.w.lineEdit_savepath.setText(str(self.root))
        self.w.comboBox_class.setCurrentText('FM')
        self.w.lineEdit_samplenum.setText('2')
        self.w.generate_data(); self.wait(self.w.generation_task)
        old=self.w.generation_result.session_directory
        self.assertTrue(self.w.label_genesig.listDataItems())
        def fail(): raise RuntimeError('controlled checkpoint read failure')
        self.w.generation_task.workflow=HistoricalGenerationWorkflow(device='cpu',model_loader=fail)
        self.w.generate_data(); self.wait(self.w.generation_task)
        self.assertIsNone(self.w.generation_result)
        self.assertFalse(self.w.label_genesig.listDataItems())
        self.assertFalse(self.w.pushButton_switchsignal.isEnabled())
        self.assertEqual(self.w.signals,[])
        self.assertTrue((old/'generation-session.json').exists())

    def test_bad_comparison_manifest_does_not_leave_old_model_results(self):
        self.w.historical_comparison_task.workflow=_workflow(self.root,[])
        self.w.runtime_directories['comparison']=self.root/'evidence'
        self.w.start_historical_comparison(); self.wait(self.w.historical_comparison_task)
        old=self.w.light_comparison_result
        self.assertIn(old.reference.checkpoint_name,self.w.textEdit_6.toPlainText())
        broken=self.root/'broken.json'; broken.write_text('{',encoding='utf8')
        with patch.object(main.QFileDialog,'getOpenFileName',return_value=(str(broken),'')):
            self.w.load_data_light()
        self.w.start_historical_comparison(); self.wait(self.w.historical_comparison_task)
        self.assertIsNone(self.w.light_comparison_result)
        self.assertNotIn(old.reference.checkpoint_name,self.w.textEdit_6.toPlainText())
        self.assertNotIn(old.lightweight.checkpoint_name,self.w.textEdit_7.toPlainText())
        self.assertTrue(old.evidence_file.is_file())

    def test_general_failure_clears_visible_model_feature(self):
        self.w.label_featuremap.plot(np.arange(10))
        self.w._on_general_recognition_failed('controlled error')
        self.assertFalse(self.w.label_featuremap.listDataItems())

    def test_general_failure_progress_does_not_become_cancelled_on_finish(self):
        self.w._on_general_recognition_failed('controlled error')
        self.w._on_general_recognition_busy(False)
        self.assertEqual(self.w.recognition_workbench.general_progress.format(),'失败')


if __name__=='__main__': unittest.main()
