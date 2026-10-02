"""Deep-module contracts for the unified home expert-analysis session."""
import os
import threading
import time
import unittest
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from matplotlib.backends.backend_agg import FigureCanvasAgg

import main
from airwatch.analysis.home_features import (
    HomeFeatureRequest,
    compute_home_feature,
    prepare_home_feature_request,
)
from airwatch.ui.plots.home_feature_renderer import HomeFeaturePresentation


class HomeFeatureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_for_feature(self, window):
        end = time.monotonic() + 10
        while window.home_feature_task.busy and time.monotonic() < end:
            QTest.qWait(10)
        self.assertFalse(window.home_feature_task.busy)

    def test_factory_freezes_exact_selection_and_source_mutation_cannot_drift(self):
        source = np.arange(512, dtype=np.float32)
        request = prepare_home_feature_request(
            'wavelet', source, '128', start_sample=20, end_sample=84
        )
        source[20:84] = -1
        np.testing.assert_array_equal(request.signal, np.arange(20, 84))
        self.assertFalse(request.signal.flags.writeable)
        self.assertEqual(request.sample_rate_hz, 128.0)
        self.assertEqual((request.start_sample, request.end_sample), (20, 84))

    def test_unknown_kind_and_direct_invalid_hht_request_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '未登记'):
            prepare_home_feature_request('unknown', np.arange(64), 128)
        request = HomeFeatureRequest('hht', np.ones(64), 128, 0)
        with self.assertRaises(ValueError):
            compute_home_feature(request)

    def test_worker_computes_off_thread_but_window_renders_on_ui_thread(self):
        window = main.MyWindow()
        self.addCleanup(window.close)
        window.home_data = np.arange(512, dtype=np.float64)
        main_thread = threading.get_ident()
        worker_threads = []
        renderer_threads = []
        computed = compute_home_feature(prepare_home_feature_request(
            'wavelet', window.home_data, 48000))

        def compute(request, **_kwargs):
            worker_threads.append(threading.get_ident())
            return computed

        def render(_result):
            renderer_threads.append(threading.get_ident())
            pixmap = QPixmap(12, 12)
            pixmap.fill()
            return HomeFeaturePresentation(pixmap, 'fixture', 'fixture rendered')

        with patch(
            'airwatch.ui.home_feature_background.compute_home_feature',
            side_effect=compute,
        ), patch.object(main, 'render_home_feature', side_effect=render):
            window.features('小波特征')
            self.wait_for_feature(window)

        self.assertEqual(renderer_threads, [main_thread])
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], main_thread)
        self.assertEqual(window.num, 1)

    def test_expert_rasterization_does_not_run_on_ui_thread(self):
        """Moving only signal math off-thread must not leave slow Agg drawing in Qt."""
        window = main.MyWindow()
        self.addCleanup(window.close)
        window.home_data = np.sin(np.arange(256) / 8)
        window.lineEdit_Fs_1.setText('48000')
        ui_thread = threading.get_ident()
        draw_threads = []
        pixmap_threads = []
        original_draw = FigureCanvasAgg.draw
        original_from_image = QPixmap.fromImage

        def draw(canvas):
            draw_threads.append(threading.get_ident())
            return original_draw(canvas)

        def publish(image, *args, **kwargs):
            pixmap_threads.append(threading.get_ident())
            return original_from_image(image, *args, **kwargs)

        with patch.object(FigureCanvasAgg, 'draw', draw), patch.object(QPixmap, 'fromImage', publish):
            window.features('小波特征')
            self.wait_for_feature(window)
        self.assertEqual(window.num, 1)
        self.assertFalse(window.label_featureshow_1.source.isNull())
        self.assertTrue(draw_threads)
        self.assertNotIn(ui_thread, draw_threads)
        self.assertTrue(pixmap_threads)
        self.assertEqual(set(pixmap_threads), {ui_thread})

    def test_cancel_during_rasterization_discards_image_and_cleans_worker(self):
        window = main.MyWindow()
        self.addCleanup(window.close)
        window.home_data = np.sin(np.arange(256) / 8)
        window.lineEdit_Fs_1.setText('48000')
        entered, release = threading.Event(), threading.Event()
        original_draw = FigureCanvasAgg.draw

        def draw(canvas):
            entered.set()
            release.wait(1)
            return original_draw(canvas)

        with patch.object(FigureCanvasAgg, 'draw', draw):
            try:
                window.features('小波特征')
                deadline = time.monotonic() + 10
                while not entered.is_set() and time.monotonic() < deadline:
                    QTest.qWait(10)
                self.assertTrue(entered.is_set())
                window._cancel_home_feature()
            finally:
                release.set()
                self.wait_for_feature(window)
        self.assertEqual(window.num, 0)
        self.assertTrue(window.label_featureshow_1.source.isNull())


if __name__ == '__main__':
    unittest.main()
