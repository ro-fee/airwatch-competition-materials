"""Real Qt canvas layout/routing tests, without training or checkpoint changes."""
import os
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QFontDatabase
from PyQt5.QtTest import QTest
import main


class SignalWorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # Offscreen Windows plugin does not discover system fonts automatically.
        font = Path('C:/Windows/Fonts/msyh.ttc')
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))

    def setUp(self):
        self.w = main.MyWindow()
        self.w.tabWidget.setCurrentWidget(self.w.tab1)
        self.w.show()
        QTest.qWait(20)
        self.canvas = self.w.signal_workbench

    def tearDown(self):
        self.w.close()
        self.app.processEvents()

    def data(self):
        self.w.home_data = np.cos(2*np.pi*64*np.arange(2048)/1024)
        self.w.lineEdit_Fs_1.setText('1024')
        self.w.lineEdit_windowlength_1.setText('128')

    def wait_for_feature(self):
        end = time.monotonic() + 10
        while self.w.home_feature_task.busy and time.monotonic() < end:
            QTest.qWait(10)
        self.assertFalse(self.w.home_feature_task.busy)

    def test_canvas_dominates_at_two_sizes_and_expert_is_collapsed(self):
        self.assertTrue(self.canvas.expert_panel.isHidden())
        for size in ((1180,760),(1440,920)):
            self.w.resize(*size); QTest.qWait(30)
            self.assertGreater(self.canvas.stack.width(), self.canvas.width()*.65)
            self.assertGreater(self.canvas.stack.height(), self.canvas.height()*.5)
            self.assertTrue(self.w.textEdit_log_1.isVisible())
            self.assertLess(self.w.pushButton_reload_TF.mapTo(self.canvas, self.w.pushButton_reload_TF.rect().bottomRight()).x(), self.canvas.width())
        self.canvas.expert_toggle.click()
        self.assertTrue(self.canvas.expert_panel.isVisible())

    def test_real_spectrum_refresh_and_exclusive_modes(self):
        self.data()
        self.w.radioButton_spectrum_1.setChecked(True)
        items = self.w.label_signalshow_1.listDataItems()
        self.assertTrue(items)
        x,y = items[0].getData()
        self.assertAlmostEqual(x[np.argmax(y)],64,delta=1)
        self.w._refresh_home_view()
        self.assertTrue(self.w.radioButton_spectrum_1.isChecked())
        self.w.radioButton_time_1.setChecked(True)
        self.assertFalse(self.w.radioButton_spectrum_1.isChecked())
        self.assertEqual(self.canvas.stack.currentIndex(),0)

    def test_expert_actual_result_alternation_and_return(self):
        self.data()
        self.w.radioButton_time_1.setChecked(True)
        self.w.label_signalshow_1.setXRange(0,2047,padding=0)
        self.canvas.analyse_button.click()
        self.wait_for_feature()
        first = self.canvas.stack.currentIndex()
        self.assertIn(first,(1,2))
        self.assertFalse(self.canvas.images[first-1].source.isNull())
        self.canvas.analyse_button.click()
        self.wait_for_feature()
        self.assertNotEqual(self.canvas.stack.currentIndex(),first)
        self.canvas.return_button.click()
        self.assertEqual(self.canvas.stack.currentIndex(),0)

    def test_image_aspect_ratio_original_size_and_scroll(self):
        image = self.canvas.images[0]
        pixmap = QPixmap(800,1500); pixmap.fill()
        image.setPixmap(pixmap); self.app.processEvents()
        image.update_size()
        fitted = image.pixmap()
        self.assertAlmostEqual(fitted.width()/fitted.height(),800/1500,delta=.005)
        self.canvas.zoom_button.click(); self.app.processEvents()
        self.assertEqual(image.pixmap().size(),pixmap.size())
        self.assertGreater(self.canvas.stack.currentWidget().verticalScrollBar().maximum(),0)

    def test_new_file_clears_prior_images_without_reading_history(self):
        self.canvas.images[0].setPixmap(QPixmap(40,40))
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'signal.npy'; np.save(file,np.arange(512,dtype=float))
            with patch.object(main.QFileDialog,'getOpenFileName',return_value=(str(file),'')):
                self.w.openfile()
        self.assertTrue(all(image.source.isNull() for image in self.canvas.images))
        self.assertEqual(self.canvas.stack.currentIndex(),0)
        self.assertIn('signal.npy',self.w.label_filename_1.toolTip())

    def test_no_data_and_invalid_expert_input_have_visible_error(self):
        self.canvas.analyse_button.click()
        image=self.canvas.images[self.canvas.stack.currentIndex()-1]
        self.assertIn('未完成',image.text())
        self.assertEqual(self.w.num,0)
        self.w.home_data=np.ones((2,512))
        self.canvas.analyse_button.click()
        self.assertTrue(image.source.isNull())
        self.assertEqual(self.w.num,0)

    def test_return_and_mode_switch_cancel_home_feature_before_showing_signal(self):
        self.data()
        self.w.radioButton_waterfall.setChecked(True)
        self.canvas.show_result(self.canvas.stack.widget(1))
        self.assertTrue(self.w.waterfall_plotter.is_paused)
        self.w.home_feature_target=self.canvas.images[0]
        self.canvas.images[0].setText('后台计算中')
        self.canvas.return_button.click()
        self.assertIsNone(self.w.home_feature_target)
        self.assertEqual(self.canvas.stack.currentIndex(),0)
        self.w.home_feature_target=self.canvas.images[0]
        self.w.radioButton_spectrum_1.setChecked(True)
        self.assertIsNone(self.w.home_feature_target)
        self.assertEqual(self.canvas.stack.currentIndex(),0)


if __name__=='__main__': unittest.main()
