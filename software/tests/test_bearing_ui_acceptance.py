"""Real frozen-model Qt acceptance, not an accuracy benchmark.

Only the native file picker and blocking message boxes are intercepted. Buttons,
preview, QThread, model, quality gate and result widgets execute normally.
Set AIRWATCH_UI_QA_OUTPUT to a writable directory to save computed evidence.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
import numpy as np
import torch
from scipy.io import savemat
import main
from airwatch.inference.bearing_contract import load_frozen_bearing_runtime_contract
from airwatch.ui.bearing_presentation import build_bearing_diagnosis_presentation

ROOT = Path(__file__).resolve().parents[1]


class BearingUIAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if Path('C:/Windows/Fonts/msyh.ttc').exists():
            # QA-only font discovery workaround, not a packaged application path.
            QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cpu = patch.object(main, 'device', torch.device('cpu'))
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.w = main.MyWindow()
        self.addCleanup(self.cleanup_window)
        self.w.resize(1180, 760)
        self.w.tabWidget.setCurrentWidget(self.w.tab_3)
        self.w.recognition_workbench.select_module(2)
        self.w.show()
        QTest.qWait(20)
        self.p = self.w.recognition_workbench.bearing_plots
        self.results, self.errors, self.progress = [], [], []
        self.w.bearing_task.completed.connect(self.results.append)
        self.w.bearing_task.failed.connect(self.errors.append)
        self.w.bearing_task.progress.connect(lambda *args: self.progress.append(args))
        self.critical = patch.object(main.QMessageBox, 'critical')
        self.modal = self.critical.start()
        self.addCleanup(self.critical.stop)
        self.ticks = 0
        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.addCleanup(self.timer.stop)

    def tick(self):
        self.ticks += 1

    def until(self, predicate, timeout=40):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(predicate(), 'UI/background task did not settle within timeout')
        self.app.processEvents()

    def cleanup_window(self):
        self.w.bearing_task.cancel()
        self.p.cancel()
        self.until(lambda: not self.w.bearing_task.is_running and self.p.thread is None)
        self.w.close()
        self.app.processEvents()

    def select(self, path):
        with patch.object(main.QFileDialog, 'getOpenFileName', return_value=(str(path), '')):
            QTest.mouseClick(self.w.bearing_select_button, Qt.LeftButton)
        self.until(lambda: self.p.thread is None)
        self.assertEqual(Path(self.w.bearing_file_edit.text()), path.resolve())
        self.assertEqual(self.w.bearing_diagnosis_label.text(), '—')

    def start_and_wait(self):
        previous = len(self.results)
        ticks = self.ticks
        QTest.mouseClick(self.w.bearing_start_button, Qt.LeftButton)
        self.assertFalse(self.w.bearing_select_button.isEnabled())
        self.assertFalse(self.w.bearing_start_button.isEnabled())
        self.until(lambda: not self.w.bearing_task.is_running)
        self.assertEqual(self.errors, [])
        self.assertEqual(len(self.results), previous + 1)
        self.assertGreater(self.ticks, ticks, 'UI event timer must still fire during inference')
        payload = self.results[-1]
        display = build_bearing_diagnosis_presentation(payload)
        self.assertEqual(self.w.bearing_diagnosis_label.text(), display.diagnosis_text)
        self.assertEqual(self.w.bearing_confidence_label.text(), display.confidence_text)
        self.assertEqual(self.w.bearing_quality_status_label.text(), display.state_title)
        self.assertEqual(self.w.bearing_quality_card.property('qualityState'), payload['quality_status'])
        self.assertEqual(self.w.bearing_progress.value(), payload['window_count'])
        self.assertEqual(self.w.bearing_task_state_label.text(), '已完成')
        self.assertTrue(self.w.bearing_select_button.isEnabled())
        self.assertFalse(self.w.bearing_cancel_button.isEnabled())
        return payload

    def evidence(self, name, value):
        output = os.environ.get('AIRWATCH_UI_QA_OUTPUT')
        if output:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / (name + '.json')).write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
            self.w.grab().save(str(directory / (name + '.png')))

    def test_four_real_files_through_buttons_and_unchanged_frozen_resources(self):
        runtime = load_frozen_bearing_runtime_contract()
        paths = [runtime.contract_path, runtime.checkpoint_path, runtime.label_map_path,
                 runtime.quality_calibration_path, runtime.manifest_path]
        files = [ROOT / 'datasets/bearing/cwru/raw' / relative for relative in (
            'normal/Normal_0.mat', 'ball/B007_0.mat',
            'inner_race/IR007_0.mat', 'outer_race_6/OR007@6_0.mat')]
        paths += files
        def hashes():
            return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        before = hashes()
        for path in files:
            with self.subTest(file=path.name):
                self.select(path)
                self.assertEqual(self.p.view.path, str(path.resolve()))
                self.assertEqual(self.p._frequency_lines, [])
                self.assertTrue(all(not e.text() for e in self.p.geometry_edits.values()))
                QTest.mouseClick(self.p.spectrum_button, Qt.LeftButton)
                self.assertTrue(self.p.plot.listDataItems())
                result = self.start_and_wait()
                self.assertEqual(result['source_path'], str(path.resolve()))
                self.evidence(path.stem, {'scope': 'CPU real Qt buttons + frozen model; not accuracy',
                                         'result': result, 'ui_diagnosis': self.w.bearing_diagnosis_label.text()})
        after = hashes()
        self.assertEqual(before, after)
        self.evidence('integrity', {'before': before, 'after': after, 'unchanged': before == after})
        self.assertTrue(self.w.close())
        self.assertFalse(self.w.isVisible())

    def test_plot_parameters_do_not_change_diagnosis(self):
        path = ROOT / 'datasets/bearing/cwru/raw/normal/Normal_0.mat'
        self.select(path)
        first = self.start_and_wait()
        QTest.mouseClick(self.p.spectrum_button, Qt.LeftButton)
        QTest.mouseClick(self.p.frequency_toggle, Qt.LeftButton)
        for key, value in {'roller_count':'8', 'roller_diameter_mm':'10', 'pitch_diameter_mm':'50',
                           'shaft_rpm':'1200', 'contact_angle_deg':'0'}.items():
            self.p.geometry_edits[key].setText(value)
        QTest.mouseClick(self.p.frequency_button, Qt.LeftButton)
        self.assertEqual(len(self.p._frequency_lines), 4)
        second = self.start_and_wait()
        for key in ('predicted_label', 'predicted_class', 'quality_status', 'quality_status_counts',
                    'window_count', 'contract_id', 'normalization'):
            self.assertEqual(first[key], second[key])
        self.assertAlmostEqual(first['mean_confidence'], second['mean_confidence'], places=7)
        self.evidence('plot_isolation', {'geometry': 'hypothetical QA only, not CWRU metadata',
                                       'before': first, 'after': second})

    def test_rejection_and_corrupt_input_clear_previous_results(self):
        constant = Path(self.temp.name) / 'constant.mat'
        savemat(constant, {'X_DE_time': np.ones(4096)})
        self.select(constant)
        self.assertIsNone(self.p.view.spectrum)
        result = self.start_and_wait()
        self.assertEqual(result['quality_status'], 'rejected')
        self.assertIsNone(result['predicted_label'])
        self.assertEqual(self.w.bearing_diagnosis_label.text(), '未输出故障类别')
        self.assertEqual(self.w.bearing_confidence_label.text(), '未输出置信度')
        self.evidence('synthetic_constant_rejection', {'scope': 'synthetic failure fixture', 'result': result})
        corrupt = Path(self.temp.name) / 'corrupt.mat'
        corrupt.write_bytes(b'not a MAT file')
        self.select(corrupt)
        self.assertIsNone(self.p.view)
        self.assertFalse(self.p.plot.listDataItems())
        QTest.mouseClick(self.w.bearing_start_button, Qt.LeftButton)
        self.until(lambda: not self.w.bearing_task.is_running)
        self.assertEqual(len(self.errors), 1)
        self.modal.assert_called_once()
        self.assertEqual(self.w.bearing_task_state_label.text(), '失败')
        self.assertEqual(self.w.bearing_diagnosis_label.text(), '—')
        self.evidence('corrupt_input', {'error': self.errors[-1], 'ui_state': '失败'})

    def test_real_task_cancel_and_close_while_running(self):
        self.select(ROOT / 'datasets/bearing/cwru/raw/normal/Normal_0.mat')
        QTest.mouseClick(self.w.bearing_start_button, Qt.LeftButton)
        QTest.mouseClick(self.w.bearing_cancel_button, Qt.LeftButton)
        self.until(lambda: not self.w.bearing_task.is_running)
        self.assertEqual(self.w.bearing_task_state_label.text(), '已取消')
        self.assertEqual(self.w.bearing_diagnosis_label.text(), '—')
        self.assertEqual(self.results, [])
        self.assertEqual(self.errors, [])
        QTest.mouseClick(self.w.bearing_start_button, Qt.LeftButton)
        self.assertTrue(self.w.close())
        self.assertFalse(self.w.bearing_task.is_running)
        self.assertFalse(self.w.isVisible())


if __name__ == '__main__':
    unittest.main()
