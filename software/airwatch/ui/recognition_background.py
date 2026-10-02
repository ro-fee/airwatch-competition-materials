"""Background boundary for one general-signal recognition job."""
from __future__ import annotations
from threading import Event
from typing import Any, Callable
import inspect
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from airwatch.data.recognition_input import RecognitionSnapshot
from airwatch.workflows.general_recognition import RecognitionResult

Loader = Callable[[str], Any]
Predictor = Callable[..., RecognitionResult]

class _RecognitionThread(QThread):
    completed = pyqtSignal(int, object)
    phase = pyqtSignal(int, str)
    progress = pyqtSignal(int, int, int)
    failed = pyqtSignal(int, str)
    cancelled_signal = pyqtSignal(int)
    def __init__(self, token, snapshot, loader, predictor, parent):
        super().__init__(parent)
        self.token, self.snapshot = token, snapshot
        self.loader, self.predictor = loader, predictor
        self.cancelled = Event()
    def run(self):
        try:
            self.phase.emit(self.token, '正在加载历史识别模型…')
            if self.cancelled.is_set(): raise InterruptedError
            model = self.loader(self.snapshot.task_name)
            if self.cancelled.is_set(): raise InterruptedError
            self.phase.emit(self.token, '正在后台处理信号…')
            progress = lambda current, total: self.progress.emit(self.token, current, total)
            if len(inspect.signature(self.predictor).parameters) >= 4:
                result = self.predictor(model, self.snapshot, self.cancelled, progress)
            else:
                result = self.predictor(model, self.snapshot, self.cancelled)
            if self.cancelled.is_set(): raise InterruptedError
            self.completed.emit(self.token, result)
        except InterruptedError:
            self.cancelled_signal.emit(self.token)
        except Exception as exc:
            if self.cancelled.is_set(): self.cancelled_signal.emit(self.token)
            else: self.failed.emit(self.token, str(exc))
        finally:
            self.snapshot = None

class RecognitionTask(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    busy_changed = pyqtSignal(bool)
    phase = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    def __init__(self, loader, predictor, parent=None):
        super().__init__(parent)
        self.loader, self.predictor = loader, predictor
        self.thread = None; self.token = 0; self._cancel_requested = False
    @property
    def busy(self): return self.thread is not None
    def start(self, snapshot):
        if self.busy: raise RuntimeError('上一项通用信号识别仍在运行或收尾')
        self.token += 1; self._cancel_requested = False
        thread = _RecognitionThread(self.token, snapshot, self.loader, self.predictor, self)
        self.thread = thread
        thread.completed.connect(self._completed); thread.failed.connect(self._failed)
        thread.cancelled_signal.connect(self._cancelled); thread.phase.connect(self._phase)
        thread.progress.connect(self._progress); thread.finished.connect(self._finished)
        self.busy_changed.emit(True); thread.start()
    def cancel(self, *, invalidate=False):
        if self.thread is None: return False
        # Input/task changes invalidate callbacks; the explicit cancel button keeps them.
        if invalidate: self.token += 1
        self._cancel_requested = True; self.thread.cancelled.set(); return True
    def wait(self, milliseconds=5000):
        return self.thread is None or self.thread.wait(milliseconds)
    def _valid(self, token): return self.thread is not None and token == self.token
    def _completed(self, token, result):
        if self._valid(token): self.completed.emit(result)
    def _failed(self, token, message):
        if self._valid(token): self.failed.emit(message)
    def _cancelled(self, token):
        if self._valid(token) and self._cancel_requested: self.cancelled.emit()
    def _phase(self, token, message):
        if self._valid(token): self.phase.emit(message)
    def _progress(self, token, current, total):
        if self._valid(token): self.progress.emit(current, total)
    def _finished(self):
        thread = self.thread; self.thread = None
        if thread is not None: thread.deleteLater()
        self.busy_changed.emit(False)

__all__ = ['RecognitionSnapshot', 'RecognitionResult', 'RecognitionTask']
