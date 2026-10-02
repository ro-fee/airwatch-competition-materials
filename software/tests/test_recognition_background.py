import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import time
import unittest
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from airwatch.ui.recognition_background import RecognitionResult, RecognitionTask
from airwatch.data.recognition_input import RecognitionSnapshot
from airwatch.inference.general_models import TASKS, load_general_model
from airwatch.workflows.general_recognition import predict_one

class RecognitionBackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def pump(self, predicate, timeout=2000):
        deadline = time.monotonic() + timeout / 1000
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents(); QTest.qWait(5)
        self.assertTrue(predicate())

    def snapshot(self):
        data = np.arange(32, dtype=np.float32).reshape(2, 16)
        return RecognitionSnapshot('test', ('a.dat',), data, (), 8)

    def test_snapshot_copies_and_freezes_input(self):
        source = np.ones((2, 8), dtype=np.float32)
        snap = RecognitionSnapshot('test', ('a.dat',), source, (), 8)
        source[0, 0] = 99
        self.assertEqual(snap.data[0, 0], 1)
        self.assertFalse(snap.data.flags.writeable)
        with self.assertRaises(ValueError): snap.data[0, 0] = 0

    def test_task_runs_loader_and_predictor_off_gui_thread(self):
        gui = self.app.thread(); calls=[]; loaded=[]
        def loader(task):
            loaded.append((task, self.app.thread() == __import__('PyQt5').QtCore.QThread.currentThread()))
            return 'model'
        def predictor(model, snap, cancelled):
            calls.append(__import__('PyQt5').QtCore.QThread.currentThread() == gui)
            return RecognitionResult(snap.task_name, snap.filenames, ('normal',), ((0,1),), np.zeros((2,3)), (2,), 'model.pkl', .01, False)
        task=RecognitionTask(loader,predictor); results=[]; task.completed.connect(results.append)
        task.start(self.snapshot()); self.pump(lambda: not task.busy)
        self.assertEqual(len(results),1); self.assertEqual(loaded[0][0],'test'); self.assertEqual(calls,[False])

    def test_duplicate_start_cancel_and_stale_result(self):
        release=[]
        def loader(task): return 'model'
        def predictor(model, snap, cancelled):
            while not cancelled.is_set(): time.sleep(.005)
            raise InterruptedError
        task=RecognitionTask(loader,predictor); task.start(self.snapshot())
        with self.assertRaises(RuntimeError): task.start(self.snapshot())
        self.assertTrue(task.cancel()); self.pump(lambda: not task.busy)
        self.assertFalse(task.cancel())

    def test_registry_loads_from_other_working_directory(self):
        import os
        old = Path.cwd()
        try:
            os.chdir(Path(__file__).resolve().parents[1] / 'tests')
            model, path = load_general_model('信号编码识别', 'cpu')
            self.assertTrue(path.is_file())
            self.assertEqual(next(model.parameters()).device.type, 'cpu')
        finally:
            os.chdir(old)

    def test_multiple_input_snapshot_preserves_file_order(self):
        snap = RecognitionSnapshot('test', ('a.dat', 'b.dat'), (np.ones((1, 8)), np.zeros((1, 8))), (), 8)
        self.assertEqual(len(snap.data), 2)
        self.assertEqual(snap.filenames, ('a.dat', 'b.dat'))

    def test_repeat_count_is_preserved_and_progress_budget_is_validated(self):
        snap = RecognitionSnapshot('test', ('a.dat',), np.ones((1, 16)), (), 8, 3)
        self.assertEqual(snap.repeat_count, 3)

    def test_snapshot_rejects_invalid_input(self):
        for data in (np.array([]), np.array([[np.nan, 1]])):
            with self.assertRaises(ValueError): RecognitionSnapshot('test', ('a',), data, (), 8)

    def test_snapshot_rejects_invalid_repeat_counts(self):
        for repeat in (0, -1, True, 1.5, float('nan'), 101):
            with self.assertRaises(ValueError):
                RecognitionSnapshot('test', ('a',), np.ones((1, 8)), (), 8, repeat)

    def test_cancel_reports_once_and_does_not_emit_result(self):
        def loader(task): return 'model'
        def predictor(model, snap, cancelled):
            while not cancelled.is_set(): time.sleep(.002)
            raise InterruptedError
        task = RecognitionTask(loader, predictor)
        cancelled, results = [], []
        task.cancelled.connect(lambda: cancelled.append(True)); task.completed.connect(results.append)
        task.start(self.snapshot()); self.assertTrue(task.cancel()); self.pump(lambda: not task.busy)
        self.assertEqual(cancelled, [True]); self.assertEqual(results, [])

    def test_loader_failure_after_cancel_reports_cancelled_and_cleans_up(self):
        from threading import Event
        entered, release = Event(), Event()
        def loader(_):
            entered.set()
            if not release.wait(2): raise RuntimeError('probe timeout')
            raise RuntimeError('checkpoint failed after cancellation')
        task = RecognitionTask(loader, lambda *args: None)
        cancelled, errors = [], []
        task.cancelled.connect(lambda: cancelled.append(True)); task.failed.connect(errors.append)
        task.start(self.snapshot()); self.pump(entered.is_set)
        task.cancel(); release.set(); self.pump(lambda: not task.busy)
        self.assertEqual(cancelled, [True]); self.assertEqual(errors, [])

if __name__ == '__main__': unittest.main()
