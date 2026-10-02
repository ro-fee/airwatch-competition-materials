"""UAV input plots with latest-request-only background preparation."""
from pathlib import Path
from threading import Event
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                            QSpinBox, QComboBox, QPushButton, QSizePolicy)
from airwatch.workflows.uav_preview import prepare_uav_preview
from airwatch.ui.plots.pyqtgraph_renderer import _reset_axes, render_spectrogram


class PreviewThread(QThread):
    result = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    phase = pyqtSignal(int, str)

    def __init__(self, request, parent):
        super().__init__(parent)
        self.request = request
        self.cancelled = Event()

    def run(self):
        token, path, mode, rate, window, prepared_input = self.request
        try:
            result = prepare_uav_preview(path, mode, rate, window, self.cancelled,
                                         lambda text: self.phase.emit(token, text),
                                         prepared_input)
            if not self.cancelled.is_set():
                self.result.emit(token, result)
        except InterruptedError:
            pass
        except Exception as exc:
            if not self.cancelled.is_set():
                self.failed.emit(token, str(exc))


class UAVPlotPanel(QWidget):
    input_changed = pyqtSignal(str)
    quality_changed = pyqtSignal(str, str)
    recognition_ready_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = None
        self.thread = None
        self.pending = None
        self.token = 0
        self.view = None
        self.prepared_input = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#0c1012')
        self.plot.setMinimumSize(480, 280)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot.setTitle('等待接入无人机数据')
        self.plot.addLegend(offset=(10, 10))
        layout.addWidget(self.plot, 1)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.mode = QComboBox()
        for title, key in (('波形', 'waveform'), ('频谱', 'spectrum'), ('时频图', 'spectrogram')):
            self.mode.addItem(title, key)
        row.addWidget(self.mode)
        row.addWidget(QLabel('采样率 Hz'))
        self.rate = QLineEdit()
        self.rate.setPlaceholderText('识别前必须填写')
        self.rate.setMaximumWidth(120)
        self.rate.setToolTip(
            '用户填写的真实采集采样率，未核验；用于图形刻度并与模型契约核对。'
        )
        row.addWidget(self.rate)
        row.addWidget(QLabel('窗长'))
        self.window = QSpinBox()
        self.window.setRange(2, 65536)
        self.window.setValue(256)
        self.window.setMaximumWidth(85)
        row.addWidget(self.window)
        row.addStretch()
        self.cancel_button = QPushButton('取消预览')
        self.cancel_button.clicked.connect(self.cancel)
        row.addWidget(self.cancel_button)
        self.caption = QLabel('请选择 NPY 文件；采样率未知时可查看采样点波形。')
        self.caption.setWordWrap(True)
        self.caption.setTextFormat(Qt.PlainText)
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.caption)
        self.mode.currentIndexChanged.connect(self.reload)
        self.rate.textChanged.connect(self.reload)
        self.window.valueChanged.connect(self.reload)
        self._buttons()

    def select_file(self, path):
        self.path = str(Path(path).resolve())
        self.prepared_input = None
        self.rate.blockSignals(True)
        self.rate.clear()
        self.rate.blockSignals(False)
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0)
        self.mode.blockSignals(False)
        self.reload()

    def reload(self, *_):
        self.window.setEnabled(self.mode.currentData() == 'spectrogram')
        if self.path is None:
            return
        self.token += 1
        self.view = None
        self.recognition_ready_changed.emit(False)
        self.plot.clear()
        self.plot.setTitle(None)
        self.input_changed.emit(f'文件：{Path(self.path).name}\n状态：准备预览')
        self.caption.setText('正在准备预览…')
        self.pending = (
            self.token,
            self.path,
            self.mode.currentData(),
            self.rate.text().strip() or None,
            self.window.value(),
            self.prepared_input,
        )
        if self.thread is None:
            self._launch()
        else:
            self.thread.cancelled.set()
        self._buttons()

    def _launch(self):
        self.thread = PreviewThread(self.pending, self)
        self.pending = None
        self.thread.result.connect(self._result)
        self.thread.failed.connect(self._failed)
        self.thread.phase.connect(self._phase)
        self.thread.finished.connect(self._finished)
        self.thread.start()

    def _phase(self, token, text):
        if token == self.token:
            self.caption.setText(text)

    def _failed(self, token, text):
        if token == self.token:
            self.recognition_ready_changed.emit(False)
            self.caption.setText(f'无法显示：{text}')
            self.input_changed.emit(f'文件：{Path(self.path).name}\n状态：预览未完成')
            self.quality_changed.emit('rejected', str(text))

    def _result(self, token, result):
        if token != self.token or result.path != self.path:
            return
        self.view = result
        self.prepared_input = result.prepared_input
        item = self.plot.plotItem
        _reset_axes(item)
        item.clear()
        item.getAxis('bottom').setTicks(None)
        item.getAxis('left').setTicks(None)
        view = result.view
        if result.mode == 'spectrogram':
            render_spectrogram(self.plot, view)
        elif result.mode == 'waveform':
            item.setLabel('bottom', view.x_label, units='')
            item.setLabel('left', '原文件幅度（单位未核验）', units='')
            item.setTitle('I / Q 波形' if result.channels == 2 else '实信号波形')
            curves = []
            for i, label in enumerate(view.channel_labels):
                curve = item.plot(view.x, view.values[i], name=label,
                                  pen=pg.mkPen(('#70e0d6', '#ff83c7')[i], width=1))
                curves.append(curve)
            # Fit raw coordinates before adaptive downsampling: an old sample-
            # index range can otherwise collapse a seconds-scale curve to zero
            # display points, leaving autoRange with no bounds to fit.
            item.getViewBox().autoRange()
            for curve in curves:
                curve.setDownsampling(auto=True, method='peak')
        else:
            item.setLabel('bottom', '基带相对频率' if view.two_sided else '频率', units='Hz')
            item.setLabel('left', '|FFT| / N（非校准功率）', units='')
            item.setTitle('IQ 双边频谱' if view.two_sided else '实信号单边频谱')
            item.plot(view.frequency_hz, view.magnitude, pen=pg.mkPen('#ff83c7', width=1))
        item.getViewBox().autoRange()
        rate = f'{result.sample_rate_hz:g} Hz（用户填写）' if result.sample_rate_hz else '未提供'
        quality_text = ('谨慎：' if result.quality.status.value == 'caution' else '') + result.quality.message
        self.input_changed.emit(f'文件：{Path(result.path).name}\n通道：{result.channels}\n' 
                                f'总采样点：{result.total_samples:,}\n采样率：{rate}')
        self.quality_changed.emit(result.quality.status.value, quality_text)
        self.recognition_ready_changed.emit(self.is_recognition_ready)
        self.caption.setText(f'显示前 {result.shown_samples:,} / {result.total_samples:,} 点；'
                             '质量检查覆盖完整记录；图形未做模型归一化。')

    def _finished(self):
        old = self.thread
        self.thread = None
        old.deleteLater()
        if self.pending is not None:
            self._launch()
        self._buttons()

    def _buttons(self):
        self.cancel_button.setEnabled(self.thread is not None)
        self.window.setEnabled(self.mode.currentData() == 'spectrogram')

    def cancel(self):
        self.token += 1
        self.pending = None
        if self.thread is not None:
            self.thread.cancelled.set()
        self.view = None
        self.prepared_input = None
        self.recognition_ready_changed.emit(False)
        self.plot.clear()
        self.plot.setTitle(None)
        self.caption.setText('预览已取消；后台读取或计算正在安全收尾。')
        self.input_changed.emit('状态：预览已取消')
        self._buttons()

    def wait(self, milliseconds):
        return self.thread is None or self.thread.wait(milliseconds)

    @property
    def is_recognition_ready(self):
        return (
            self.view is not None
            and self.prepared_input is not None
            and self.prepared_input.quality.status.value != 'rejected'
            and self.prepared_input.declared_sample_rate_hz is not None
        )
