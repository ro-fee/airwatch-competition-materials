"""QThread task for the frozen bearing diagnosis workflow.

The worker is the only object that enters the background thread.  It never
touches a widget.  The controller remains in the GUI thread and forwards
structured progress, completion, cancellation, and failure signals.
"""

from __future__ import annotations

from airwatch.data.bearing_metadata import find_bearing_metadata, metadata_sensor_key
from pathlib import Path
import threading
import traceback
from typing import Any, Callable

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot

from airwatch.analysis.bearing_quality import load_bearing_quality_thresholds_v2
from airwatch.data.cwru import CWRUDataError
from airwatch.inference.bearing import BearingInferenceError
from airwatch.inference.bearing_contract import (
    BearingRuntimeContractError,
    DEFAULT_FROZEN_BEARING_CONTRACT,
    load_frozen_bearing_runtime_contract,
)
from airwatch.workflows.bearing_diagnosis import BearingDiagnosisWorkflow
from airwatch.workflows.bearing_progressive import (
    BearingDiagnosisCancelled,
    ProgressiveBearingDiagnosisWorkflow,
)


class BearingInferenceWorker(QObject):
    """Execute one frozen-contract diagnosis in a worker thread."""

    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(dict)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(
        self,
        *,
        input_path: str | Path,
        contract_path: str | Path = DEFAULT_FROZEN_BEARING_CONTRACT,
        device: str = "cpu",
        sensor_key: str | None = None,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.input_path = Path(input_path).resolve()
        self.contract_path = Path(contract_path).resolve()
        self.device = str(device)
        self.sensor_key = sensor_key
        self.batch_size = int(batch_size)
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        """Set the cooperative cancellation flag; safe from the GUI thread."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            self._raise_if_cancelled()
            self.progress.emit(0, 0, "正在核对冻结模型契约…")
            runtime = load_frozen_bearing_runtime_contract(self.contract_path)
            self._raise_if_cancelled()

            sensor_key = self.sensor_key or self._sensor_key_from_manifest(
                runtime.manifest_path,
                runtime.project_root,
                self.input_path,
            )
            self.progress.emit(0, 0, "正在加载冻结模型和质量门…")
            thresholds = load_bearing_quality_thresholds_v2(
                runtime.quality_calibration_path
            )
            workflow = BearingDiagnosisWorkflow.from_checkpoint(
                runtime.checkpoint_path,
                device=self.device,
                label_map_path=runtime.label_map_path,
            )
            self._raise_if_cancelled()
            progressive = ProgressiveBearingDiagnosisWorkflow(
                workflow,
                thresholds=thresholds,
                quality_messages=runtime.quality_messages,
            )
            result = progressive.run_file(
                self.input_path,
                sensor_key=sensor_key,
                batch_size=self.batch_size,
                progress=self.progress.emit,
                is_cancelled=self.is_cancelled,
            )
            self._raise_if_cancelled()
            payload = result.to_dict()
            payload["contract_id"] = runtime.contract_id
            payload["contract_path"] = str(runtime.contract_path)
            payload["quality_gate_version"] = runtime.quality_gate_version
            payload["normalization"] = runtime.normalization
            self.completed.emit(payload)
        except BearingDiagnosisCancelled:
            self.cancelled.emit("轴承诊断已取消，未生成新的诊断结果。")
        except Exception as exc:  # noqa: BLE001 - converted to a safe UI error
            self.failed.emit(self._error_payload(exc))
        finally:
            self.finished.emit()

    def _raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise BearingDiagnosisCancelled("bearing diagnosis was cancelled")

    @staticmethod
    def _sensor_key_from_manifest(
        manifest_path: Path,
        project_root: Path,
        input_path: Path,
    ) -> str | None:
        """Use the audited sensor key when the chosen file is in the manifest."""
        return metadata_sensor_key(find_bearing_metadata(manifest_path, project_root, input_path))

    @staticmethod
    def _error_payload(exc: Exception) -> dict[str, str]:
        if isinstance(exc, BearingRuntimeContractError):
            message = (
                "冻结模型或配套文件的完整性检查失败。为了避免使用错配模型，"
                "本次诊断已停止。"
            )
        elif isinstance(exc, CWRUDataError):
            message = f"无法读取轴承振动数据：{exc}"
        elif isinstance(exc, BearingInferenceError):
            message = f"冻结模型无法处理这份数据：{exc}"
        elif isinstance(exc, FileNotFoundError):
            message = f"诊断所需文件不存在：{exc}"
        elif isinstance(exc, MemoryError):
            message = "可用内存不足，轴承诊断已停止。请关闭其他任务后重试。"
        else:
            message = f"后台轴承诊断失败：{exc}"
        return {
            "title": "轴承诊断失败",
            "message": message,
            "exception_type": type(exc).__name__,
            "technical_detail": "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip(),
        }


class BearingInferenceTask(QObject):
    """GUI-thread controller that owns one worker and one QThread at a time."""

    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(dict)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(dict)
    state_changed = pyqtSignal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        worker_factory: Callable[..., BearingInferenceWorker] = BearingInferenceWorker,
    ) -> None:
        super().__init__(parent)
        self._worker_factory = worker_factory
        self._thread: QThread | None = None
        self._worker: BearingInferenceWorker | None = None
        self._state = "idle"

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(
        self,
        *,
        input_path: str | Path,
        contract_path: str | Path = DEFAULT_FROZEN_BEARING_CONTRACT,
        device: str = "cpu",
        sensor_key: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if self.is_running or self._thread is not None:
            raise RuntimeError("a bearing inference task is already running")
        worker = self._worker_factory(
            input_path=input_path,
            contract_path=contract_path,
            device=device,
            sensor_key=sensor_key,
            batch_size=batch_size,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.progress.emit)
        worker.completed.connect(self.completed.emit)
        worker.cancelled.connect(self.cancelled.emit)
        worker.failed.connect(self.failed.emit)
        # Quit directly from the worker-emitting thread.  This matters when the
        # window is closing and the GUI thread is waiting for a cooperative
        # cancellation to finish.
        worker.finished.connect(thread.quit, Qt.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._worker = worker
        self._thread = thread
        self._set_state("running")
        thread.start()

    def cancel(self) -> bool:
        if not self.is_running or self._worker is None:
            return False
        self._worker.request_cancel()
        self._set_state("cancelling")
        return True

    def wait(self, timeout_ms: int = 5000) -> bool:
        """Wait for a cancelled task to leave its thread safely.

        This is intentionally a wait, not ``terminate()``: model execution is
        allowed to finish its current batch and release PyTorch/file resources
        normally.  The method is mainly used by ``QMainWindow.closeEvent`` and
        by integration tests; ordinary UI code remains signal-driven.
        """
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool):
            raise TypeError("timeout_ms must be an integer")
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        thread = self._thread
        if thread is None:
            return True
        if not thread.wait(timeout_ms):
            return False
        # ``thread.finished`` normally reaches this controller through the Qt
        # event queue. A close event may be waiting synchronously, so perform the
        # same small bookkeeping here when the thread is already known to have
        # stopped. The queued callback is harmless afterwards.
        if self._thread is thread:
            self._worker = None
            self._thread = None
            self._set_state("idle")
        return True

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self.state_changed.emit(state)

    @pyqtSlot()
    def _thread_finished(self) -> None:
        self._worker = None
        self._thread = None
        self._set_state("idle")


__all__ = ["BearingInferenceTask", "BearingInferenceWorker"]
