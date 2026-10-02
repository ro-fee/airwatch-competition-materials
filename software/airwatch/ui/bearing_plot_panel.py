"""Bearing preview widget + single background job, latest-request-wins routing."""
from pathlib import Path
from threading import Event
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QGroupBox, QGridLayout, QSizePolicy)
from airwatch.analysis.bearing_fault_frequencies import BearingGeometry, compute_characteristic_frequencies
import pyqtgraph as pg
from airwatch.workflows.bearing_preview import load_bearing_preview
from .plots.pyqtgraph_renderer import _reset_axes


class PreviewThread(QThread):
    result = pyqtSignal(int, object)
    error = pyqtSignal(int, str)
    phase = pyqtSignal(int, str)

    def __init__(self, token, path, rate, parent):
        super().__init__(parent)
        self.token, self.path, self.rate = token, path, rate
        self.cancelled = Event()

    def run(self):
        try:
            result = load_bearing_preview(self.path, sample_rate_hz=self.rate,
                cancelled=self.cancelled.is_set,
                progress=lambda message: self.phase.emit(self.token, message))
            if not self.cancelled.is_set():
                self.result.emit(self.token, result)
        except InterruptedError:
            pass
        except Exception as exc:
            if not self.cancelled.is_set():
                self.error.emit(self.token, f'轴承图无法显示：{exc}')


class BearingPlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.pending = None
        self.token = 0
        self.path = None
        self.view = None
        self.mode = 'waveform'
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.caption = QLabel('请选择轴承 MAT 文件；图形预览不等于诊断结论。')
        self.caption.setWordWrap(True)
        self.caption.setTextFormat(Qt.PlainText)
        layout.addWidget(self.caption)
        self.plot = pg.PlotWidget()
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot.setBackground('#0c1012')
        self.plot.setMinimumSize(480,220)
        self.plot.showGrid(x=True,y=True,alpha=.18)
        layout.addWidget(self.plot,1)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.wave_button = QPushButton('振动波形')
        self.spectrum_button = QPushButton('频谱')
        for button,mode in ((self.wave_button,'waveform'),(self.spectrum_button,'spectrum')):
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, mode=mode: self.render(mode))
            row.addWidget(button)
        self.frequency_toggle = QPushButton('参考线参数 +')
        self.frequency_toggle.setCheckable(True)
        self.frequency_toggle.toggled.connect(self._toggle_frequency_controls)
        row.addWidget(self.frequency_toggle)
        row.addStretch()
        row.addWidget(QLabel('采样率 Hz'))
        self.rate = QLineEdit()
        self.rate.setPlaceholderText('清单自动 / 手填')
        self.rate.setMaximumWidth(145)
        self.rate.setToolTip('仅影响绘图；清单外文件请填写真实采样率，不会改变模型预处理。')
        row.addWidget(self.rate)
        self.refresh = QPushButton('刷新预览')
        self.refresh.clicked.connect(self.reload)
        row.addWidget(self.refresh)
        self.cancel_button = QPushButton('取消预览')
        self.cancel_button.clicked.connect(self.cancel)
        row.addWidget(self.cancel_button)
        self._build_frequency_controls(layout)
        self._buttons()

    def _build_frequency_controls(self, layout):
        self.frequency_group = QGroupBox('故障特征频率标线（仅参数对照）')
        grid = QGridLayout(self.frequency_group)
        self.frequency_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.geometry_edits = {}
        fields = (
            ('roller_count', '滚动体数', '例如 8'),
            ('roller_diameter_mm', '滚动体径 mm', '例如 10'),
            ('pitch_diameter_mm', '节径 mm', '例如 50'),
            ('shaft_rpm', '转速 RPM', '例如 1200'),
            ('contact_angle_deg', '接触角 °', '请填写'),
        )
        for col, (key, label, placeholder) in enumerate(fields):
            grid.addWidget(QLabel(label), 0, col)
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setMinimumWidth(0)
            edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            edit.textChanged.connect(self._clear_frequency_lines)
            self.geometry_edits[key] = edit
            grid.addWidget(edit, 1, col)
        self.frequency_button = QPushButton('画参考线')
        self.frequency_button.clicked.connect(self.update_frequency_lines)
        self.frequency_button.setToolTip('只根据输入的几何和转速画理论参考线，不自动判断故障类型。')
        grid.addWidget(self.frequency_button, 1, 5)
        self.frequency_note = QLabel('尚未计算参考线。')
        self.frequency_note.setWordWrap(True)
        self.frequency_note.setTextFormat(Qt.PlainText)
        self.frequency_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        clear_button = QPushButton('清除')
        clear_button.clicked.connect(self._clear_frequency_lines)
        grid.addWidget(clear_button, 1, 6)
        hint = QLabel('参数来源：用户填写（未核验）。节径不是外径；不清楚参数时请勿猜填。')
        hint.setWordWrap(True)
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        grid.addWidget(hint, 2, 0, 1, 7)

        layout.addWidget(self.frequency_group)
        layout.addWidget(self.frequency_note)
        self.frequency_note.hide()
        self.frequency_group.hide()
        self._frequency_lines = []

    def _toggle_frequency_controls(self, checked):
        self.frequency_group.setVisible(checked and self.mode == 'spectrum' and self.view is not None)
        self.frequency_toggle.setText('参考线参数 −' if checked else '参考线参数 +')

    def _clear_frequency_lines(self):
        for line in self._frequency_lines:
            self.plot.removeItem(line)
        self._frequency_lines = []
        if hasattr(self, 'frequency_note'):
            self.frequency_note.setText('参考线未显示。尺寸和转速决定参考位置；出现峰值也不等于确认故障。')

    def update_frequency_lines(self):
        self._clear_frequency_lines()
        if self.view is None or self.mode != 'spectrum' or self.view.spectrum is None:
            self.frequency_note.setText('请先加载文件并切换到频谱视图。')
            return
        rate = self.view.waveform.sample_rate_hz
        if not rate:
            self.frequency_note.setText('当前采样率未知，无法在频谱上放置参考线。')
            return
        try:
            values = {key: edit.text().strip() for key, edit in self.geometry_edits.items()}
            geometry = BearingGeometry(
                roller_count=int(values['roller_count']),
                roller_diameter_mm=float(values['roller_diameter_mm']),
                pitch_diameter_mm=float(values['pitch_diameter_mm']),
                shaft_rpm=float(values['shaft_rpm']),
                contact_angle_deg=float(values['contact_angle_deg']),
            )
            frequencies = compute_characteristic_frequencies(geometry)
        except (TypeError, ValueError, OverflowError) as exc:
            self.frequency_note.setText(f'参数无效：{exc}')
            return
        labels = (('FTF', frequencies.ftf_hz, '#ffe08a'), ('BPFO', frequencies.bpfo_hz, '#ff83c7'),
                  ('BPFI', frequencies.bpfi_hz, '#70e0d6'), ('BSF', frequencies.bsf_hz, '#b69cff'))
        nyquist = rate / 2.0
        max_frequency = float(self.view.spectrum.frequency_hz[-1])
        shown, omitted = [], []
        for label, frequency, color in labels:
            if frequency <= 0 or frequency > nyquist or frequency > max_frequency:
                omitted.append(f'{label} {frequency:.2f} Hz（超出当前频带）')
                continue
            line = pg.InfiniteLine(pos=frequency, angle=90, movable=False,
                                    pen=pg.mkPen(color, width=1.5),
                                    label=f'{label} {frequency:.2f} Hz',
                                    labelOpts={'color': color, 'position': 0.08 + len(shown) * 0.12})
            self.plot.addItem(line)
            self._frequency_lines.append(line)
            shown.append(f'{label} {frequency:.2f} Hz')
        note = '；'.join(shown) if shown else '没有参考线落在当前频带内'
        note += '。参数来源：用户填写（未核验）'
        if omitted:
            note += '。未显示：' + '；'.join(omitted)
        self.frequency_note.setText(note + '。峰值不等于确认故障。')

    def _buttons(self):
        spectrum_active = self.view is not None and self.mode == 'spectrum' and self.view.spectrum is not None
        self.frequency_toggle.setVisible(spectrum_active)
        self.frequency_note.setVisible(spectrum_active)
        self._toggle_frequency_controls(self.frequency_toggle.isChecked())
        self.wave_button.setEnabled(self.view is not None)
        self.spectrum_button.setEnabled(self.view is not None and self.view.spectrum is not None)
        self.wave_button.setChecked(self.mode == 'waveform')
        self.spectrum_button.setChecked(self.mode == 'spectrum')
        self.cancel_button.setEnabled(self.thread is not None)
        self.refresh.setEnabled(self.path is not None)

    def select_file(self,path):
        self.path = str(Path(path).resolve())
        self.rate.clear()
        for edit in self.geometry_edits.values():
            edit.clear()
        self.frequency_toggle.setChecked(False)
        self.reload()

    def reload(self):
        if self.path is None:
            return
        self.token += 1
        self.view = None
        self.plot.clear()
        self._clear_frequency_lines()
        self.frequency_group.hide()
        self.plot.setTitle(None)
        self.caption.setText('正在准备所选文件预览…')
        self.pending = (self.token,self.path,self.rate.text().strip() or None)
        if self.thread is not None:
            self.thread.cancelled.set()
            self.caption.setText('旧预览正在收尾，随后读取最新选择；不会显示旧文件结果。')
        else:
            self._launch()
        self._buttons()

    def _launch(self):
        token,path,rate = self.pending
        self.pending = None
        thread = PreviewThread(token,path,rate,self)
        self.thread = thread
        thread.result.connect(self._result)
        thread.error.connect(self._error)
        thread.phase.connect(self._phase)
        thread.finished.connect(self._finished)
        thread.start()
        self._buttons()

    def _phase(self,token,message):
        if token == self.token:
            self.caption.setText(message)

    def _result(self,token,view):
        if token != self.token or view.path != self.path:
            return
        self.view = view
        self.render(self.mode if view.spectrum else 'waveform')

    def _error(self,token,message):
        if token == self.token:
            self.view = None
            self.plot.clear()
            self._clear_frequency_lines()
            self.frequency_group.hide()
            self.caption.setText(message)
            self._buttons()

    def _finished(self):
        old = self.thread
        self.thread = None
        if old is not None:
            old.deleteLater()
        if self.pending is not None:
            self._launch()
        self._buttons()

    def cancel(self):
        self.token += 1
        self.pending = None
        if self.thread is not None:
            self.thread.cancelled.set()
        self.view = None
        self.plot.clear()
        self._clear_frequency_lines()
        self.frequency_group.hide()
        self.caption.setText('预览已取消；读取/计算可能仍在收尾，不影响后台诊断。')
        self._buttons()

    def wait(self,milliseconds):
        return self.thread is None or self.thread.wait(milliseconds)

    def render(self,mode):
        if self.view is None or (mode == 'spectrum' and self.view.spectrum is None):
            return
        self._clear_frequency_lines()
        self.mode = mode
        v = self.view
        item = self.plot.plotItem
        _reset_axes(item)
        item.clear()
        item.getAxis('left').setTicks(None)
        item.getAxis('bottom').setTicks(None)
        item.setLabel('left','原文件幅度（单位未核验）')
        if mode == 'waveform':
            item.setLabel('bottom',v.waveform.x_label)
            curve = item.plot(v.waveform.x,v.waveform.values[0],pen=pg.mkPen('#70e0d6',width=1))
            # Establish seconds-scale bounds before adaptive downsampling uses
            # a stale sample-index or spectrum viewport to choose its stride.
            item.getViewBox().autoRange()
            curve.setDownsampling(auto=True,method='peak')
            item.setTitle('振动波形 · 未做模型归一化')
        else:
            item.setLabel('bottom','频率',units='Hz')
            item.setLabel('left','|FFT| / N（非校准功率）')
            item.plot(v.spectrum.frequency_hz,v.spectrum.magnitude,pen=pg.mkPen('#ff83c7',width=1))
            item.setTitle('当前预览段频谱 · 未加窗 / 未去均值 / 单边未倍增')
        item.getViewBox().autoRange()
        rate = v.waveform.sample_rate_hz
        units = f'{rate:g} Hz' if rate else '采样率未知，频谱禁用'
        self.caption.setText(f'{Path(v.path).name} · {v.sensor_key} · {units}（{v.rate_source}）\n'
            f'显示前 {v.shown_samples} / {v.total_samples} 点；仅辅助观察，不标注故障类型。')
        self._buttons()
