"""Animated spectrum-waterfall widget controller for pyqtgraph."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from airwatch.analysis.waterfall import compute_waterfall, FLOOR_DB, DISPLAY_FLOOR_DB
from .pyqtgraph_renderer import _reset_axes


class WaterfallPlotter(QtCore.QObject):
    """Pre-compute a spectrogram and replay it as an animated waterfall."""

    state_changed = QtCore.pyqtSignal(str)
    progress_changed = QtCore.pyqtSignal(int, int)
    playback_finished = QtCore.pyqtSignal()

    floor_db = FLOOR_DB
    display_floor_db = DISPLAY_FLOOR_DB
    render_fps = 30.0

    def __init__(self, plot_widget):
        super().__init__(plot_widget)
        self.plot = plot_widget
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._update_frame)
        self.image_item = None
        self.buffer = None
        self.spectra_db = None
        self.frequencies = None
        self.frame_times = None
        self.current_frame = 0
        self.total_frames = 0
        self.frame_interval_ms = 0
        self.history_lines = 0
        self.hop_seconds = 0.0
        self.frame_duration_seconds = 0.0
        self.playback_speed = 1.0
        self._clock = time.perf_counter
        self._playback_anchor_time = 0.0
        self._playback_anchor_frame = 0
        self._state = 'stopped'

    @property
    def state(self):
        return self._state

    @property
    def is_running(self):
        return self._state == 'running'

    @property
    def is_paused(self):
        return self._state == 'paused'

    @property
    def is_finished(self):
        return self._state == 'finished'

    def start(
        self,
        data,
        fs,
        n_fft,
        history_lines=300,
        overlap_ratio=0.5,
        playback_speed=1.0,
    ):
        """Start a fresh playback after validating and analysing the signal."""
        self.stop(clear=True)
        history_lines, playback_speed = self._validate_playback(history_lines, playback_speed)
        view = compute_waterfall(data, fs, n_fft, overlap_ratio)
        self.spectra_db = view.spectra_db
        self.frequencies = view.frequencies
        self.frame_times = view.frame_times
        self.total_frames = self.spectra_db.shape[1]
        if self.total_frames == 0:
            raise ValueError('无法从当前信号计算频谱帧。')

        self.history_lines = history_lines
        self.buffer = np.full(
            (history_lines, self.spectra_db.shape[0]),
            self.floor_db,
            dtype=np.float32,
        )
        self.current_frame = 0
        self.hop_seconds = view.hop_seconds
        self.playback_speed = playback_speed
        self.frame_duration_seconds = self.hop_seconds / playback_speed
        self.frame_interval_ms = max(1, int(round(1000.0 / self.render_fps)))
        self._configure_plot()
        self._set_state('running')
        self._advance_to_frame(1)
        if not self.is_finished:
            self._restart_playback_clock()
            self.timer.start(self.frame_interval_ms)

    def pause(self):
        if self.is_running:
            self.timer.stop()
            self._set_state('paused')

    def resume(self):
        if self.is_paused and self.current_frame < self.total_frames:
            self._set_state('running')
            self._restart_playback_clock()
            self.timer.start(self.frame_interval_ms)

    def stop(self, clear=False):
        self.timer.stop()
        self.current_frame = 0
        if clear:
            self.plot.clear()
            _reset_axes(self.plot.plotItem)
            self.plot.setTitle(None)
            self.plot.getAxis('left').setTicks(None)
            self.plot.getAxis('bottom').setTicks(None)
            self.plot.getViewBox().setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
            self.image_item = None
            self.buffer = None
            self.spectra_db = None
            self.frequencies = None
            self.frame_times = None
            self.total_frames = 0
        self._set_state('stopped')

    def _validate_playback(self, history_lines, playback_speed):
        if isinstance(history_lines, (bool, np.bool_)) or not isinstance(history_lines, (int, np.integer)):
            raise ValueError('历史行数必须是整数。')
        if history_lines < 20:
            raise ValueError('历史行数不能小于 20。')
        playback_speed = float(playback_speed)
        if not np.isfinite(playback_speed) or playback_speed <= 0:
            raise ValueError('播放速度必须大于 0。')
        return history_lines, playback_speed

    def _configure_plot(self):
        self.plot.clear()
        _reset_axes(self.plot.plotItem)
        self.plot.invertY(True)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setLabel('bottom', '频率', units='Hz')
        self.plot.setLabel('left', '历史时间', units='s')
        self.plot.setTitle('动态瀑布：本记录峰值 0 dB，暗→亮 -80→0 dB（去均值，非 dBm）', color='#dbe7f5', size='11pt')

        self.image_item = pg.ImageItem(axisOrder='row-major')
        colors = np.array([
            [5, 12, 28, 255],
            [0, 67, 128, 255],
            [0, 190, 178, 255],
            [255, 165, 48, 255],
            [255, 246, 204, 255],
        ], dtype=np.ubyte)
        positions = np.array([0.0, 0.28, 0.55, 0.8, 1.0])
        color_map = pg.ColorMap(positions, colors)
        self.image_item.setLookupTable(color_map.getLookupTable(nPts=512))
        self.image_item.setImage(
            self.buffer,
            autoLevels=False,
            levels=(self.display_floor_db, 0.0),
        )
        frequency_step = (
            abs(self.frequencies[1] - self.frequencies[0])
            if self.frequencies.size > 1 else 1.0
        )
        left = float(self.frequencies[0] - frequency_step / 2)
        width = float(self.frequencies[-1] - self.frequencies[0] + frequency_step)
        self.image_item.setRect(QtCore.QRectF(left, -0.5, width, self.history_lines))
        self.plot.addItem(self.image_item)
        self.plot.setXRange(left, left + width, padding=0)
        self.plot.setYRange(-0.5, self.history_lines - 0.5, padding=0)
        self.plot.getViewBox().setLimits(
            xMin=left,
            xMax=left + width,
            yMin=-0.5,
            yMax=self.history_lines - 0.5,
        )
        self._set_time_ticks()

    def _set_time_ticks(self):
        positions = np.linspace(0, self.history_lines - 1, 6, dtype=int)
        ticks = []
        for line in positions:
            label = '现在' if line == 0 else f'-{line * self.hop_seconds:.2f}'
            ticks.append((int(line), label))
        self.plot.getAxis('left').setTicks([ticks])

    def _restart_playback_clock(self):
        self._playback_anchor_time = self._clock()
        self._playback_anchor_frame = self.current_frame

    @QtCore.pyqtSlot()
    def _update_frame(self):
        if not self.is_running or self.spectra_db is None:
            return
        if self.current_frame >= self.total_frames:
            self._finish_playback()
            return
        elapsed = max(0.0, self._clock() - self._playback_anchor_time)
        due_frames = int(elapsed / self.frame_duration_seconds)
        target_frame = min(
            self.total_frames,
            self._playback_anchor_frame + due_frames,
        )
        if target_frame > self.current_frame:
            self._advance_to_frame(target_frame)

    def _advance_to_frame(self, target_frame):
        target_frame = min(int(target_frame), self.total_frames)
        if target_frame <= self.current_frame:
            return
        new_rows = self.spectra_db[:, self.current_frame:target_frame].T[::-1]
        count = new_rows.shape[0]
        if count >= self.history_lines:
            self.buffer[:] = new_rows[:self.history_lines]
        else:
            self.buffer[count:] = self.buffer[:self.history_lines - count].copy()
            self.buffer[:count] = new_rows
        self.current_frame = target_frame
        self.image_item.setImage(
            self.buffer,
            autoLevels=False,
            levels=(self.display_floor_db, 0.0),
        )
        self.progress_changed.emit(self.current_frame, self.total_frames)
        if self.current_frame >= self.total_frames:
            self._finish_playback()

    def _finish_playback(self):
        self.timer.stop()
        if not self.is_finished:
            self._set_state('finished')
            self.playback_finished.emit()

    def _set_state(self, state):
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)
