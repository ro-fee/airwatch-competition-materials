"""Regression tests for the cancellable bearing UI boundary.

These tests keep the model out of the Qt controller tests. The progressive
workflow is exercised with a small fake backend, while the QThread controller
is exercised with a fake worker exposing the same signal contract.
"""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import time
import unittest

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication

from airwatch.ui import BearingInferenceTask, build_bearing_diagnosis_presentation
from airwatch.workflows import BearingDiagnosisCancelled, ProgressiveBearingDiagnosisWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakePredictor:
    window_size = 4
    step = 4
    device = "cpu"
    checkpoint_path = PROJECT_ROOT / "fake-bearing.pt"
    label_map = {0: "normal", 1: "inner_race", 2: "ball", 3: "outer_race_6"}


class _FakeQualityWorkflow:
    def __init__(self, statuses: list[str] | None = None) -> None:
        self.adapter = type("Adapter", (), {"predictor": _FakePredictor()})()
        self.statuses = statuses
        self.offset = 0

    def run_signal_with_quality_v2(self, signal, *, thresholds):
        batch_count = len(np.asarray(signal)) // self.adapter.predictor.window_size
        rows = []
        for local_index in range(batch_count):
            global_index = self.offset + local_index
            status = self.statuses[global_index] if self.statuses is not None else "accepted"
            prediction = None
            if status != "rejected":
                predicted_class = global_index % 4
                prediction = {
                    "window_index": local_index,
                    "predicted_label": self.adapter.predictor.label_map[predicted_class],
                    "predicted_class": predicted_class,
                    "confidence": 0.8,
                    "probabilities": [0.1, 0.2, 0.3, 0.4],
                }
            rows.append({
                "window_index": local_index,
                "status": status,
                "accepted": status != "rejected",
                "reason_code": status,
                "message": status,
                "quality_score": 1.0 if status == "accepted" else 0.5,
                "quality_features": {},
                "prediction": prediction,
            })
        self.offset += batch_count
        return rows


class ProgressiveBearingWorkflowTests(unittest.TestCase):
    def test_reports_progress_and_preserves_global_window_indices(self) -> None:
        workflow = ProgressiveBearingDiagnosisWorkflow(_FakeQualityWorkflow(), thresholds=object())
        progress = []
        result = workflow.run_signal(
            np.arange(24, dtype=np.float32),
            batch_size=2,
            progress=lambda current, total, message: progress.append((current, total, message)),
        )
        self.assertEqual(
            [(current, total) for current, total, _ in progress],
            [(0, 6), (2, 6), (4, 6), (6, 6)],
        )
        self.assertEqual([row["window_index"] for row in result.windows], list(range(6)))
        self.assertEqual(result.quality_status, "accepted")
        self.assertEqual(result.processed_window_count, 6)

    def test_rejected_windows_have_no_prediction_and_summary_is_rejected(self) -> None:
        workflow = ProgressiveBearingDiagnosisWorkflow(
            _FakeQualityWorkflow(["rejected", "rejected", "rejected"]),
            thresholds=object(),
        )
        result = workflow.run_signal(np.ones(12, dtype=np.float32), batch_size=2)
        self.assertEqual(result.quality_status, "rejected")
        self.assertIsNone(result.predicted_label)
        self.assertIsNone(result.predicted_class)
        self.assertIsNone(result.mean_confidence)
        self.assertEqual(result.quality_status_counts["rejected"], 3)
        self.assertTrue(all(row["prediction"] is None for row in result.windows))

    def test_caution_is_inferred_but_summary_is_marked_caution(self) -> None:
        workflow = ProgressiveBearingDiagnosisWorkflow(
            _FakeQualityWorkflow(["accepted", "caution"]), thresholds=object()
        )
        result = workflow.run_signal(np.ones(8, dtype=np.float32), batch_size=1)
        self.assertEqual(result.quality_status, "caution")
        self.assertIsNotNone(result.predicted_label)
        self.assertIsNotNone(result.mean_confidence)
        self.assertEqual(result.quality_status_counts["caution"], 1)

    def test_cancellation_is_checked_between_batches(self) -> None:
        workflow = ProgressiveBearingDiagnosisWorkflow(_FakeQualityWorkflow(), thresholds=object())
        calls = []

        def is_cancelled() -> bool:
            calls.append(True)
            return len(calls) >= 3

        with self.assertRaises(BearingDiagnosisCancelled):
            workflow.run_signal(
                np.ones(16, dtype=np.float32), batch_size=1, is_cancelled=is_cancelled
            )
        self.assertGreaterEqual(len(calls), 3)

    def test_invalid_batch_size_is_rejected(self) -> None:
        workflow = ProgressiveBearingDiagnosisWorkflow(_FakeQualityWorkflow(), thresholds=object())
        with self.assertRaises(ValueError):
            workflow.run_signal(np.ones(4, dtype=np.float32), batch_size=0)


class PresentationRuleTests(unittest.TestCase):
    def _result(self, status: str) -> dict[str, object]:
        payload = {
            "quality_status": status,
            "quality_message": "质量提示",
            "window_count": 3,
            "quality_status_counts": {"accepted": 2, "caution": 1, "rejected": 0},
            "elapsed_seconds": 0.25,
        }
        if status != "rejected":
            payload.update({"predicted_label": "ball", "mean_confidence": 0.91})
        return payload

    def test_accepted_shows_diagnosis_and_confidence(self) -> None:
        presentation = build_bearing_diagnosis_presentation(self._result("accepted"))
        self.assertEqual(presentation.state_title, "质量通过")
        self.assertIn("滚动体故障", presentation.diagnosis_text)
        self.assertIn("91.00%", presentation.confidence_text)
        self.assertTrue(presentation.can_show_prediction)
        self.assertIn('模型契约', presentation.details_text)
        self.assertIn('归一化', presentation.details_text)

    def test_caution_shows_diagnosis_and_keeps_warning_semantics(self) -> None:
        presentation = build_bearing_diagnosis_presentation(self._result("caution"))
        self.assertEqual(presentation.state_title, "质量提醒")
        self.assertIn("滚动体故障", presentation.diagnosis_text)
        self.assertTrue(presentation.can_show_prediction)

    def test_rejected_never_shows_model_class_or_confidence(self) -> None:
        presentation = build_bearing_diagnosis_presentation(self._result("rejected"))
        self.assertEqual(presentation.state_title, "已拒绝诊断")
        self.assertEqual(presentation.diagnosis_text, "未输出故障类别")
        self.assertEqual(presentation.confidence_text, "未输出置信度")
        self.assertFalse(presentation.can_show_prediction)


class _FakeWorker(QObject):
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(dict)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, *, mode="complete", **kwargs) -> None:
        super().__init__()
        self.mode = mode
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True

    @pyqtSlot()
    def run(self) -> None:
        if self.mode == "complete":
            self.progress.emit(1, 1, "完成")
            self.completed.emit({
                "quality_status": "accepted",
                "quality_message": "ok",
                "window_count": 1,
                "quality_status_counts": {"accepted": 1, "caution": 0, "rejected": 0},
                "predicted_label": "normal",
                "mean_confidence": 0.9,
                "elapsed_seconds": 0.01,
            })
            self.finished.emit()
            return
        for _ in range(100):
            if self.cancel_requested:
                self.cancelled.emit("cancelled")
                self.finished.emit()
                return
            time.sleep(0.005)
        self.failed.emit({"title": "failed", "message": "boom"})
        self.finished.emit()


class BearingInferenceTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Use QApplication for the whole module: another test class creates the
        # real main window, and Qt permits only one application object.
        cls.app = QApplication.instance() or QApplication([])

    def _pump_until(self, predicate, timeout_ms=2000) -> None:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertTrue(predicate())

    def test_completion_returns_to_idle_and_duplicate_start_is_rejected(self) -> None:
        task = BearingInferenceTask(worker_factory=_FakeWorker)
        completed = []
        task.completed.connect(completed.append)
        task.start(input_path="fake.mat")
        with self.assertRaises(RuntimeError):
            task.start(input_path="second.mat")
        self._pump_until(lambda: task.state == "idle")
        self.assertEqual(len(completed), 1)
        self.assertFalse(task.is_running)

    def test_cancel_requests_worker_and_returns_to_idle(self) -> None:
        task = BearingInferenceTask(
            worker_factory=lambda **kwargs: _FakeWorker(mode="cancel", **kwargs)
        )
        cancelled = []
        task.cancelled.connect(cancelled.append)
        task.start(input_path="fake.mat")
        QTimer.singleShot(20, task.cancel)
        self._pump_until(lambda: task.state == "idle")
        self.assertEqual(cancelled, ["cancelled"])
        self.assertFalse(task.cancel())

    def test_failed_worker_error_is_forwarded_and_wait_is_safe(self) -> None:
        task = BearingInferenceTask(
            worker_factory=lambda **kwargs: _FakeWorker(mode="failed", **kwargs)
        )
        failed = []
        task.failed.connect(failed.append)
        task.start(input_path="fake.mat")
        self._pump_until(lambda: task.state == "idle")
        self.assertEqual(failed, [{"title": "failed", "message": "boom"}])
        self.assertTrue(task.wait())

    def test_wait_rejects_invalid_timeout(self) -> None:
        task = BearingInferenceTask(worker_factory=_FakeWorker)
        self.assertTrue(task.wait())
        with self.assertRaises(TypeError):
            task.wait(1.5)
        with self.assertRaises(ValueError):
            task.wait(-1)


class MainWindowBearingPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_panel_is_present_and_rejected_display_is_safe(self) -> None:
        import main

        window = main.MyWindow()
        self.assertTrue(hasattr(window, "bearing_panel"))
        self.assertFalse(window.bearing_start_button.isEnabled())
        window._on_bearing_completed({
            "quality_status": "rejected",
            "quality_message": "信号质量不足，建议重新采集",
            "window_count": 1,
            "quality_status_counts": {"accepted": 0, "caution": 0, "rejected": 1},
            "elapsed_seconds": 0.01,
        })
        self.assertEqual(window.bearing_diagnosis_label.text(), "未输出故障类别")
        self.assertEqual(window.bearing_confidence_label.text(), "未输出置信度")
        window.close()


if __name__ == "__main__":
    unittest.main()
