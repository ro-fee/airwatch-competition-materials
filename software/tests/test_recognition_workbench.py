"""Navigation/layout integration tests; no model training or fabricated UI metrics."""
import os
import unittest
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
import numpy as np
import tempfile
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtGui import QFontDatabase
from pathlib import Path
import main


class RecognitionWorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if Path('C:/Windows/Fonts/msyh.ttc').exists():
            QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')

    def setUp(self):
        self.w = main.MyWindow()
        self.view = self.w.recognition_workbench
        self.w.tabWidget.setCurrentWidget(self.w.tab_3)
        self.w.show(); QTest.qWait(20)

    def tearDown(self):
        self.w.close(); self.app.processEvents()

    def test_idle_startup_never_opens_background_progress_dialogs(self):
        # QProgressDialog starts a 4-second auto-show timer on construction.
        QTest.qWait(5200)
        self.assertTrue(self.w.isVisible())
        self.assertFalse(self.w.home_feature_progress.isVisible())
        self.assertFalse(self.w.tsne_progress.isVisible())

    def test_progress_dialogs_still_follow_explicit_busy_state(self):
        for handler, dialog in ((self.w._on_home_feature_busy, self.w.home_feature_progress),
                                (self.w._on_tsne_busy, self.w.tsne_progress)):
            handler(True)
            self.assertTrue(dialog.isVisible())
            handler(False)
            self.assertFalse(dialog.isVisible())

    def test_uav_page_is_contract_shell_without_fake_inference(self):
        self.view.select_module(0)
        self.assertFalse(self.view.uav_start.isEnabled())
        self.assertTrue(self.view.uav_choose.isEnabled())
        self.assertIn('等待接入无人机数据', self.view.uav_plot_placeholder.plotItem.titleLabel.text)
        self.assertIn('信号质量：待检测', self.view.uav_quality_label.text())
        self.assertIn('置信度：—', self.view.uav_result_label.text())

    def test_three_modules_exclusive_and_uav_never_runs_legacy_model(self):
        with patch.object(main, 'load_general_model') as loader:
            for key,index in (('uav',0),('bearing',2),('general',1)):
                self.view.module_buttons[key].click()
                self.assertEqual(self.view.pages.currentIndex(),index)
                self.assertEqual(sum(b.isChecked() for b in self.view.module_buttons.values()),1)
                self.assertFalse(self.view.module_buttons[key].icon().isNull())
            loader.assert_not_called()
        self.view.select_module(0)
        self.assertFalse(self.view.uav_start.isEnabled())
        self.assertFalse(self.w.bearing_start_button.isVisible())
        self.assertFalse(self.w.pushButton_recognition.isVisible())

    def test_canvas_dominates_and_command_fits_minimum_window(self):
        for width,height in ((1180,760),(1440,920)):
            self.w.resize(width,height); QTest.qWait(30)
            self.assertGreater(self.view.plot_stack.width(),self.view.width()*.65)
            self.assertGreater(self.view.plot_stack.height(),self.view.height()*.5)
            p=self.w.pushButton_recognition.mapTo(self.view,self.w.pushButton_recognition.rect().bottomRight())
            self.assertLess(p.x(),self.view.width())
            self.assertLess(p.y(),self.view.height())
        self.assertTrue(self.view.details.isHidden())
        self.view.details_toggle.click()
        self.assertTrue(self.w.comboBox_choose_feature.isVisible())

    def test_real_signal_spectrum_and_feature_routes_keep_widgets(self):
        w=self.w
        w.data=np.cos(2*np.pi*64*np.arange(1024)/1024)
        w.filename=['synthetic']; w.lineEdit_Fs_2.setText('1024')
        w.radioButton_PP_3.click()
        x,y=w.label_signal.listDataItems()[0].getData()
        self.assertAlmostEqual(x[np.argmax(y)],64,delta=1)
        self.view.feature_button.click()
        self.assertIs(self.view.plot_stack.currentWidget(),w.label_featuremap)
        self.assertIn('非原始信号',self.view.plot_caption.text())
        self.view.signal_button.click()
        self.assertIs(self.view.plot_stack.currentWidget(),w.label_signal)
        self.assertTrue(w.radioButton_PP_3.isChecked())

    def test_multi_file_result_card_is_one_row_per_file(self):
        from airwatch.workflows.general_recognition import RecognitionResult
        result = RecognitionResult(
            'test', ('first.dat', 'second.dat'), ('BCH', 'LDPC'),
            ((0,), (1,)), np.zeros((2, 3), dtype=np.float32), (1, 1),
            'fixture.pth', .125, False, repeat_count=2,
            repeat_labels=(('BCH', 'LDPC'), ('BCH', 'BCH')))
        card = main.MyWindow._format_general_result_card(result)
        self.assertIn('1. first.dat → BCH（1 窗口；各轮：BCH / BCH）', card)
        self.assertIn('2. second.dat → LDPC（1 窗口；各轮：LDPC / BCH）', card)
        self.assertIn('运行次数：2', card)

    def test_bearing_rejection_is_separate_from_general_results(self):
        self.w.textEdit_log_3.setPlainText('general-only')
        self.view.select_module(2)
        self.w._on_bearing_completed({
            'quality_status':'rejected','quality_message':'重新采集',
            'window_count':1,'quality_status_counts':{'accepted':0,'caution':0,'rejected':1},
            'elapsed_seconds':.01})
        self.assertEqual(self.w.bearing_diagnosis_label.text(),'未输出故障类别')
        self.assertEqual(self.w.bearing_confidence_label.text(),'未输出置信度')
        self.assertIn('轴承后台诊断完成',self.w.bearing_log.toPlainText())
        self.assertEqual(self.w.textEdit_log_3.toPlainText(),'general-only')
        self.assertTrue(self.w.bearing_start_button.isVisible())
        self.view.select_module(0)
        self.assertFalse(self.w.bearing_diagnosis_label.isVisible())

    def test_switch_keeps_bearing_task_and_file_but_invalidates_tsne(self):
        self.w.bearing_file_edit.setText('fixture.mat')
        token=self.w.tsne_task.token
        with patch.object(self.w.bearing_task,'cancel') as cancel:
            self.view.select_module(2); self.view.select_module(0); self.view.select_module(1)
            cancel.assert_not_called()
        self.assertGreater(self.w.tsne_task.token,token)
        self.assertEqual(self.w.bearing_file_edit.text(),'fixture.mat')

    def test_bearing_failure_and_cancel_stay_in_bearing_log(self):
        before=self.w.textEdit_log_3.toPlainText()
        with patch.object(main.QMessageBox,'critical'):
            self.w._on_bearing_failed({'message':'fixture-error'})
        self.assertIn('fixture-error',self.w.bearing_log.toPlainText())
        self.w._on_bearing_cancelled('取消')
        self.assertEqual(self.w.textEdit_log_3.toPlainText(),before)
        self.assertEqual(self.w.bearing_task_state_label.text(),'已取消')

    def test_task_change_resets_model_hint_without_loading(self):
        self.view.model_hint.setText('old-model')
        with patch.object(main, 'load_general_model') as load:
            self.w.comboBox_task.setCurrentIndex(1)
            load.assert_not_called()
        self.assertIn('请重新选择',self.view.model_hint.text())

    def test_general_snapshot_is_not_overwritten_by_home_or_lightweight_state(self):
        from airwatch.data.recognition_input import load_recognition_files
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'BCH' / 'coding.dat'
            path.parent.mkdir()
            np.arange(1024, dtype=np.int16).tofile(path)
            self.w.comboBox_task.setCurrentText('信号编码识别')
            self.w.general_input = load_recognition_files('信号编码识别', [path])
            expected = self.w.general_input.arrays[0].copy()

            self.w.data = np.full((2, 4096), 99, dtype=np.float32)
            self.w.filename = ['unrelated-lightweight.dat']
            snapshot = self.w._make_general_snapshot()

            np.testing.assert_array_equal(snapshot.data, expected)
            self.assertEqual(snapshot.filenames, (str(path.resolve()),))

    def test_file_selection_uses_task_contract_and_updates_truthful_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'BCH' / 'coding.dat'
            path.parent.mkdir()
            np.arange(1024, dtype=np.int16).tofile(path)
            self.w.comboBox_task.setCurrentText('信号编码识别')
            self.w.comboBox_mod.setCurrentText('单脉冲识别')
            with patch.object(main.QFileDialog, 'exec_', return_value=1), \
                    patch.object(main.QFileDialog, 'selectedFiles', return_value=[str(path)]):
                self.w.load_data()

            self.assertIsNotNone(self.w.general_input)
            self.assertEqual(self.w.general_input.arrays[0].shape, (1, 1024))
            self.assertEqual(self.w.general_input.arrays[0].dtype, np.float32)
            self.assertIn('int16', self.view.model_hint.text())
            self.assertIn('窗长 512', self.view.model_hint.text())
            self.assertTrue(self.w.pushButton_recognition.isEnabled())

    def test_task_change_invalidates_previous_general_input(self):
        from airwatch.data.recognition_input import load_recognition_files
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'BCH' / 'coding.dat'
            path.parent.mkdir()
            np.arange(1024, dtype=np.int16).tofile(path)
            self.w.comboBox_task.setCurrentText('信号编码识别')
            self.w.general_input = load_recognition_files('信号编码识别', [path])
            self.w.pushButton_recognition.setEnabled(True)

            self.w.comboBox_task.setCurrentText('信号通联识别')

            self.assertIsNone(self.w.general_input)
            self.assertFalse(self.w.pushButton_recognition.isEnabled())
            self.assertIn('请重新选择', self.view.model_hint.text())


if __name__=='__main__': unittest.main()
