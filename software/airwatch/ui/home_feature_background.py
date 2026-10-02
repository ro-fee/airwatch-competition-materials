"""Single-flight Qt Adapter for frozen home-page expert-analysis requests."""

from __future__ import annotations

from threading import Event

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from airwatch.analysis.home_features import HomeFeatureRequest, compute_home_feature
from airwatch.ui.plots.home_feature_renderer import prepare_home_feature


class _HomeFeatureThread(QThread):
    completed = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    phase = pyqtSignal(int, str)

    def __init__(self, token: int, request: HomeFeatureRequest, parent: QObject):
        super().__init__(parent)
        self.token = token
        self.request = request
        self.cancelled = Event()

    def run(self) -> None:
        try:
            result = compute_home_feature(
                self.request,
                cancelled=self.cancelled.is_set,
                progress=lambda message: self.phase.emit(self.token, message),
            )
            if self.cancelled.is_set():
                return
            self.phase.emit(self.token, "正在后台生成专家图像")
            prepared = prepare_home_feature(result)
            if not self.cancelled.is_set():
                self.completed.emit(self.token, prepared)
        except InterruptedError:
            pass
        except Exception as exc:
            if not self.cancelled.is_set():
                self.failed.emit(self.token, str(exc))
        finally:
            self.request = None


class HomeFeatureTask(QObject):
    """Single-flight analysis plus rasterization; completed publishes QImage data.

    Widgets and QPixmap creation remain with the UI receiver. Cancellation
    invalidates both numerical and raster stages without terminating a thread.
    """
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    phase = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.thread: _HomeFeatureThread | None = None
        self.token = 0

    @property
    def busy(self) -> bool:
        return self.thread is not None

    def start(self, request: HomeFeatureRequest) -> None:
        if self.busy:
            raise ValueError("上一项主页专家分析仍在计算或收尾，请稍后重试。")
        if not isinstance(request, HomeFeatureRequest):
            raise TypeError("request must be a HomeFeatureRequest")
        self.token += 1
        thread = _HomeFeatureThread(self.token, request, self)
        self.thread = thread
        thread.completed.connect(self._completed)
        thread.failed.connect(self._failed)
        thread.phase.connect(self._phase)
        thread.finished.connect(self._finished)
        self.busy_changed.emit(True)
        thread.start()

    def cancel(self) -> None:
        self.token += 1
        if self.thread is not None:
            self.thread.cancelled.set()
            self.phase.emit("已取消显示；后台分析仍在安全收尾")

    def _completed(self, token: int, result: object) -> None:
        if token == self.token:
            self.completed.emit(result)

    def _failed(self, token: int, message: str) -> None:
        if token == self.token:
            self.failed.emit(message)

    def _phase(self, token: int, message: str) -> None:
        if token == self.token:
            self.phase.emit(message)

    def _finished(self) -> None:
        thread, self.thread = self.thread, None
        if thread is not None:
            thread.deleteLater()
        self.busy_changed.emit(False)

    def wait(self, milliseconds: int) -> bool:
        return self.thread is None or self.thread.wait(milliseconds)


__all__ = ["HomeFeatureTask"]
