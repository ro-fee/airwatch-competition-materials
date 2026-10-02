"""Qt background boundary for future UAV inference.

The task is usable with an injected backend for tests, but the default backend is
explicitly unavailable: it never fabricates a UAV prediction.
"""
from __future__ import annotations
from threading import Event
from typing import Callable, Any
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from airwatch.workflows.uav_contract import UAVRecognitionResult
from airwatch.workflows.uav_intake import UAVPreparedInput

Backend = Callable[[UAVPreparedInput, Event, Callable[[int, int], None]], UAVRecognitionResult]


def unavailable_backend(input_info, cancelled, progress):
    raise RuntimeError('无人机模型尚未接入：未发布识别结果')


class _UAVThread(QThread):
    completed = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    cancelled_signal = pyqtSignal(int)
    phase = pyqtSignal(int, str)
    progress = pyqtSignal(int, int, int)

    def __init__(self, token, prepared_input, backend, parent):
        super().__init__(parent)
        self.token, self.prepared_input, self.backend = token, prepared_input, backend
        self.cancelled = Event()

    def run(self):
        try:
            self.phase.emit(self.token, '正在准备无人机识别任务…')
            if self.cancelled.is_set():
                raise InterruptedError
            report = lambda current, total: self.progress.emit(self.token, current, total)
            result = self.backend(self.prepared_input, self.cancelled, report)
            if self.cancelled.is_set():
                raise InterruptedError
            if not isinstance(result, UAVRecognitionResult):
                raise TypeError('无人机后台必须返回 UAVRecognitionResult')
            self.completed.emit(self.token, result)
        except InterruptedError:
            self.cancelled_signal.emit(self.token)
        except Exception as exc:
            if not self.cancelled.is_set():
                self.failed.emit(self.token, str(exc))
        finally:
            self.prepared_input = None


class UAVRecognitionTask(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    busy_changed = pyqtSignal(bool)
    phase = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, backend=unavailable_backend, parent=None):
        super().__init__(parent)
        self.backend, self.thread, self.token = backend, None, 0
        self._cancel_requested = False

    @property
    def busy(self):
        return self.thread is not None

    def start(self, prepared_input):
        if self.busy:
            raise RuntimeError('上一项无人机识别仍在运行或收尾')
        if not isinstance(prepared_input, UAVPreparedInput):
            raise TypeError('无人机任务输入必须是 UAVPreparedInput')
        self.token += 1
        self._cancel_requested = False
        thread = _UAVThread(self.token, prepared_input, self.backend, self)
        self.thread = thread
        thread.completed.connect(self._completed)
        thread.failed.connect(self._failed)
        thread.cancelled_signal.connect(self._cancelled)
        thread.phase.connect(self._phase)
        thread.progress.connect(self._progress)
        thread.finished.connect(self._finished)
        self.busy_changed.emit(True)
        thread.start()

    def cancel(self, *, invalidate=False):
        if self.thread is None:
            return False
        if invalidate:
            self.token += 1
        self._cancel_requested = True
        self.thread.cancelled.set()
        return True

    def wait(self, milliseconds=5000):
        return self.thread is None or self.thread.wait(milliseconds)

    def _valid(self, token):
        return self.thread is not None and token == self.token

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
        thread = self.thread
        self.thread = None
        if thread is not None: thread.deleteLater()
        self.busy_changed.emit(False)


__all__ = ['UAVRecognitionTask', 'unavailable_backend']
