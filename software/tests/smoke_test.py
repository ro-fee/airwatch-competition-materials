"""Non-interactive startup and core-function smoke test."""

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from pyqtgraph import ImageItem
from PyQt5.QtWidgets import QApplication

import main


def run(check_models=False):
    app = QApplication.instance() or QApplication([])
    window = main.MyWindow()
    assert window.tabWidget.count() == 4
    assert not window.pushButton_recognition.isEnabled()

    signal = np.sin(np.linspace(0, 40 * np.pi, 4096)).astype(np.float32)
    window.filename = ['smoke.npy']
    window.data = signal
    window.plotsig(signal, window.label_signalshow_1)
    curves = window.label_signalshow_1.plotItem.listDataItems()
    assert len(curves) == 1
    np.testing.assert_allclose(curves[0].getData()[1], signal)
    window.lineEdit_Fs_1.setText('9600')
    window.plotspectrum(signal, window.label_signalshow_1)
    curves = window.label_signalshow_1.plotItem.listDataItems()
    assert len(curves) == 1
    spectrum_x, spectrum_y = curves[0].getData()
    np.testing.assert_allclose(spectrum_x, np.fft.rfftfreq(signal.size, d=1 / 9600))
    np.testing.assert_allclose(spectrum_y, np.abs(np.fft.rfft(signal.astype(float))) / signal.size)
    window.plotspec(signal, window.label_signalshow_1, 9600, 256)
    from airwatch.analysis.signal_transforms import compute_spectrogram
    images = [item for item in window.label_signalshow_1.plotItem.items
              if isinstance(item, ImageItem)]
    assert len(images) == 1
    expected = compute_spectrogram(signal, sample_rate_hz=9600, nperseg=256)
    np.testing.assert_allclose(images[0].image, expected.power_db)
    assert images[0].axisOrder == 'row-major'

    if check_models:
        from airwatch.inference.general_models import TASKS, load_general_model
        from airwatch.workflows.historical_generation import (
            GenerationRequest,
            HistoricalGenerationWorkflow,
        )
        from airwatch.data.historical_comparison import (
            default_historical_comparison_manifest,
        )
        from airwatch.workflows.historical_model_comparison import (
            ComparisonRunRequest,
            HistoricalModelComparisonWorkflow,
        )

        for task_name in TASKS:
            model, checkpoint = load_general_model(task_name, main.device)
            assert model is not None
            assert checkpoint.is_file()
        with TemporaryDirectory(prefix='airwatch-generation-smoke-') as directory:
            generated = HistoricalGenerationWorkflow(device='cpu').run(
                GenerationRequest(Path(directory), '32PSK', 2), Event()
            )
            assert generated.batches[0].shape == (2, 1, 1024)
            assert generated.session_directory.is_dir()
        with TemporaryDirectory(prefix='airwatch-comparison-smoke-') as directory:
            compared = HistoricalModelComparisonWorkflow(device='cpu').run(
                ComparisonRunRequest(
                    default_historical_comparison_manifest(),
                    Path(directory) / 'evidence',
                ),
                Event(),
            )
            assert compared.total_windows == 32
            assert compared.accuracy is None
            assert compared.evidence_file.is_file()

    window.close()
    print('SMOKE TEST PASSED' + (' (including model weights)' if check_models else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', action='store_true', help='also load every recognition/GAN weight')
    run(parser.parse_args().models)
