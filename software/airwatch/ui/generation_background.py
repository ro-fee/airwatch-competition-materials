"""Qt Adapter for the UI-independent historical generation Workflow."""

from __future__ import annotations

from threading import Event

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from airwatch.workflows.historical_generation import (
    GenerationRequest,
    GenerationResult,
    HistoricalGenerationWorkflow,
)


class _GenerationThread(QThread):
    completed = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    cancelled_signal = pyqtSignal(int)
    phase = pyqtSignal(int, str)
    progress = pyqtSignal(int, int, int)

    def __init__(self, token, request, workflow, parent):
        super().__init__(parent)
        self.token = token
        self.request = request
        self.workflow = workflow
        self.cancelled = Event()

    def run(self):
        try:
            result = self.workflow.run(
                self.request,
                self.cancelled,
                lambda current, total: self.progress.emit(self.token, current, total),
                lambda message: self.phase.emit(self.token, message),
            )
            if not isinstance(result, GenerationResult):
                raise TypeError("生成后台必须返回 GenerationResult。")
            # A committed atomic session is a completed result even if the user
            # clicks cancel in the tiny interval after the final rename.
            self.completed.emit(self.token, result)
        except InterruptedError:
            self.cancelled_signal.emit(self.token)
        except Exception as exc:
            if self.cancelled.is_set():
                self.cancelled_signal.emit(self.token)
            else:
                self.failed.emit(self.token, str(exc))
        finally:
            self.request = None


class GenerationTask(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    busy_changed = pyqtSignal(bool)
    phase = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, workflow=None, parent=None):
        super().__init__(parent)
        self.workflow = workflow or HistoricalGenerationWorkflow()
        self.thread = None
        self.token = 0
        self._cancel_requested = False

    @property
    def busy(self):
        return self.thread is not None

    def start(self, request):
        if self.busy:
            raise RuntimeError("上一项生成任务仍在运行或收尾。")
        if not isinstance(request, GenerationRequest):
            raise TypeError("生成任务输入必须是 GenerationRequest。")
        self.token += 1
        self._cancel_requested = False
        thread = _GenerationThread(self.token, request, self.workflow, self)
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
        if self._valid(token):
            self.completed.emit(result)

    def _failed(self, token, message):
        if self._valid(token):
            self.failed.emit(message)

    def _cancelled(self, token):
        if self._valid(token) and self._cancel_requested:
            self.cancelled.emit()

    def _phase(self, token, message):
        if self._valid(token):
            self.phase.emit(message)

    def _progress(self, token, current, total):
        if self._valid(token):
            self.progress.emit(current, total)

    def _finished(self):
        thread = self.thread
        self.thread = None
        if thread is not None:
            thread.deleteLater()
        self.busy_changed.emit(False)


__all__ = ["GenerationTask"]
