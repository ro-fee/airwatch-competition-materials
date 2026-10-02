"""One background t-SNE job with cooperative cancellation and stale-result rejection."""
from threading import Event
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from airwatch.analysis.tsne_views import compute_tsne


class _TsneThread(QThread):
    result = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, token, source, parent):
        super().__init__(parent)
        self.token, self.source = token, source
        self.cancelled = Event()

    def run(self):
        try:
            if self.cancelled.is_set():
                return
            view = compute_tsne(self.source)
            if not self.cancelled.is_set():
                self.result.emit(self.token, view)
        except Exception as exc:
            if not self.cancelled.is_set():
                self.failed.emit(self.token, str(exc))
        finally:
            self.source = None


class TsneTask(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)
    phase = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.token = 0

    @property
    def busy(self):
        return self.thread is not None

    def start(self, source):
        if self.busy:
            raise ValueError('上一项 t-SNE 仍在计算或收尾，请稍后重试。')
        self.token += 1
        thread = _TsneThread(self.token, source, self)
        self.thread = thread
        thread.result.connect(self._result)
        thread.failed.connect(self._failed)
        thread.finished.connect(self._finished)
        self.busy_changed.emit(True)
        self.phase.emit('正在后台计算 t-SNE（算法不提供百分比进度）')
        thread.start()

    def cancel(self):
        self.token += 1  # Also invalidates already queued completion signals.
        if self.thread is not None:
            self.thread.cancelled.set()
            self.phase.emit('已取消显示；后台算法仍在收尾，完成后可重新计算')

    def _result(self, token, result):
        if token == self.token:
            self.completed.emit(result)

    def _failed(self, token, message):
        if token == self.token:
            self.failed.emit(message)

    def _finished(self):
        thread = self.thread
        self.thread = None
        if thread is not None:
            thread.deleteLater()
        self.busy_changed.emit(False)

    def wait(self, milliseconds):
        return self.thread is None or self.thread.wait(milliseconds)
