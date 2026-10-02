import os, time, unittest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from airwatch.analysis.uav_quality import assess_uav_signal
from airwatch.data.uav_input import UAVSignal
from airwatch.ui.uav_background import UAVRecognitionTask
from airwatch.workflows.uav_contract import (UAVInputInfo, UAVRecognitionResult, QualityStatus, RecognitionStatus)
from airwatch.workflows.uav_intake import UAVPreparedInput, UAVSourceIdentity

class UAVBackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def pump(self, pred, timeout=2000):
        end=time.monotonic()+timeout/1000
        while not pred() and time.monotonic()<end: self.app.processEvents(); QTest.qWait(5)
        self.assertTrue(pred())
    def prepared(self):
        path=str(Path("uav.npy").resolve())
        samples=np.stack((np.arange(16,dtype=np.float32),np.arange(16,dtype=np.float32)+1))
        signal=UAVSignal(path,samples,1e6,"npy")
        return UAVPreparedInput(
            signal,
            assess_uav_signal(signal.samples),
            UAVSourceIdentity(path,1,0),
            1e6,
        )
    def test_default_backend_fails_without_fake_result(self):
        task=UAVRecognitionTask(); errors=[]; results=[]
        task.failed.connect(errors.append); task.completed.connect(results.append)
        task.start(self.prepared()); self.pump(lambda:not task.busy)
        self.assertEqual(results, []); self.assertIn("模型尚未接入", errors[0])
    def test_injected_backend_runs_off_gui_and_returns_structured_result(self):
        gui=self.app.thread(); calls=[]
        def backend(prepared, cancelled, progress):
            calls.append(__import__("PyQt5").QtCore.QThread.currentThread()==gui)
            progress(1, 1)
            info=UAVInputInfo(
                prepared.identity.path,
                prepared.declared_sample_rate_hz,
                prepared.signal.sample_count,
                prepared.signal.channels,
            )
            return UAVRecognitionResult(info, QualityStatus.ACCEPTED, "良好", RecognitionStatus.COMPLETED, "目标A", "known", .8, "test")
        task=UAVRecognitionTask(backend); results=[]; progress=[]
        task.completed.connect(results.append); task.progress.connect(lambda c,t:progress.append((c,t)))
        task.start(self.prepared()); self.pump(lambda:not task.busy)
        self.assertEqual(len(results),1); self.assertEqual(calls,[False]); self.assertEqual(progress,[(1,1)])
    def test_cancel_emits_once_without_result(self):
        def backend(prepared, cancelled, progress):
            while not cancelled.is_set(): time.sleep(.002)
            raise InterruptedError
        task=UAVRecognitionTask(backend); cancelled=[]; task.cancelled.connect(lambda:cancelled.append(1))
        task.start(self.prepared()); self.assertTrue(task.cancel()); self.pump(lambda:not task.busy)
        self.assertEqual(cancelled,[1])
if __name__ == "__main__": unittest.main()
