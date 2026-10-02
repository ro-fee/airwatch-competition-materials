"""Completed file selectors must release directory-watcher threads."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PyQt5.QtCore import QCoreApplication,QEvent
from PyQt5.QtWidgets import QApplication,QFileDialog
import main

class FileDialogLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.w=main.MyWindow();self.tmp=tempfile.TemporaryDirectory()
        self.file=Path(self.tmp.name)/'signal.dat'
        np.ones(512,dtype=np.int16).tofile(self.file)
        self.w.comboBox_task.setCurrentText('信号编码识别')
    def tearDown(self): self.w.close();self.tmp.cleanup()
    def check_dialogs(self,accepted,path):
        for _ in range(4):
            with patch.object(QFileDialog,'exec_',return_value=accepted),patch.object(QFileDialog,'selectedFiles',return_value=[str(path)]),patch.object(main.QMessageBox,'critical'):
                self.w.load_data()
            QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
            self.app.processEvents()
        self.assertEqual(self.w.findChildren(QFileDialog),[],'file selectors and their filesystem workers remain owned by the window')
    def test_accepted_selector_is_disposed(self): self.check_dialogs(1,self.file)
    def test_cancelled_selector_is_disposed(self): self.check_dialogs(0,self.file)
    def test_failed_import_selector_is_disposed(self): self.check_dialogs(1,self.file.with_suffix('.missing'))

if __name__=='__main__': unittest.main()
