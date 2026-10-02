"""Small Qt renderers for data prepared by the analysis layer.

The renderers know about pyqtgraph widgets and visual styling, but they do not
load files, load models, or decide what a signal means. The transformation and
validation remain in airwatch.analysis.signal_transforms so the same input
contract can later be used by a background workflow or a report.
"""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PyQt5.QtCore import QRectF
import numpy as np

from airwatch.analysis.signal_transforms import (
    ConstellationView,
    SpectrogramView,
    SpectrumView,
    WaveformView,
    compute_spectrum,
    compute_waveform,
)


_WAVEFORM_PENS = (
    pg.mkPen("#42bfd7", width=1.2),
    pg.mkPen("#f4b942", width=1.2),
)


def _plot_item(plot_widget: Any):
    """Return the PlotItem for either a PlotWidget or a PlotItem."""

    return getattr(plot_widget, "plotItem", plot_widget)


def _reset_axes(item):
    """Remove log, inversion and range constraints left by other plot modes."""
    item.setLogMode(x=False, y=False)
    view_box = item.getViewBox()
    view_box.invertY(False)
    view_box.invertX(False)
    view_box.setAspectLocked(False)
    view_box.setLimits(
        xMin=None, xMax=None, yMin=None, yMax=None,
        minXRange=None, maxXRange=None, minYRange=None, maxYRange=None,
    )


def render_waveform(
    plot_widget: Any,
    samples: object,
    *,
    sample_rate_hz: float | None = None,
) -> WaveformView:
    """Prepare and render one real or IQ waveform."""

    view = compute_waveform(samples, sample_rate_hz=sample_rate_hz)
    plot_item = _plot_item(plot_widget)
    _reset_axes(plot_item)
    invert_y = getattr(plot_widget, "invertY", None)
    if callable(invert_y):
        invert_y(False)
    plot_item.clear()
    plot_item.setTitle(None)
    plot_item.getAxis("left").setTicks(None)
    plot_item.getAxis("bottom").setTicks(None)
    plot_item.setLabel("bottom", view.x_label, units="")
    plot_item.setLabel("left", "幅度", units="")

    for index, label in enumerate(view.channel_labels):
        pen = _WAVEFORM_PENS[index % len(_WAVEFORM_PENS)]
        plot_item.plot(view.x, view.values[index], pen=pen, name=label)
    plot_item.getViewBox().autoRange()
    return view


_SPECTRUM_PEN = pg.mkPen("#f4b942", width=1.2)


def render_constellation(plot_widget: Any, view: ConstellationView) -> pg.ScatterPlotItem:
    """Draw sample I/Q points, not symbol-synchronized or demodulated decisions."""
    item = _plot_item(plot_widget)
    _reset_axes(item)
    item.clear()
    item.getAxis('left').setTicks(None)
    item.getAxis('bottom').setTicks(None)
    item.setLabel('bottom', 'I / 实部', units='')
    item.setLabel('left', 'Q / 虚部', units='')
    title = ('解析信号兼容显示（Hilbert 推导 Q，非真实双路采集）'
             if view.source_kind == 'analytic_real' else 'I/Q 样本散点（未做符号同步）')
    item.setTitle(title)
    item.getViewBox().setAspectLocked(True, ratio=1)
    scatter = pg.ScatterPlotItem(x=view.i, y=view.q, size=4, pen=None,
                                 brush=pg.mkBrush('#f28fbd'), pxMode=True)
    item.addItem(scatter)
    item.getViewBox().autoRange()
    return scatter


def render_spectrogram(plot_widget: Any, view: SpectrogramView) -> pg.ImageItem:
    """Render PSD dB with physical bin centres, independent of global image order.

    Colour spans 80 dB below this recording's peak; this is not calibrated dBm.
    No filesystem assets or matplotlib figures are needed.
    """
    item = _plot_item(plot_widget)
    _reset_axes(item)
    item.clear()
    item.getAxis("left").setTicks(None)
    item.getAxis("bottom").setTicks(None)
    item.setLabel("bottom", "时间", units="s")
    item.setLabel("left", "频率", units="Hz")
    item.setTitle("颜色：暗→亮 = 本记录峰值以下 80 dB→峰值（非 dBm）")
    peak = float(np.max(view.power_db))
    # Zero energy must remain dark, not become the brightest relative bin.
    levels = (peak - 80.0, peak) if np.any(view.power > 0) else (-80.0, 0.0)
    image = pg.ImageItem(axisOrder="row-major")
    image.setImage(view.power_db, autoLevels=False, levels=levels)
    image.setColorMap(pg.colormap.get("CET-L9"))
    image.setRect(QRectF(
        float(view.time_s[0] - view.time_step_s / 2),
        float(view.frequency_hz[0] - view.frequency_step_hz / 2),
        float(view.time_s.size * view.time_step_s),
        float(view.frequency_hz.size * view.frequency_step_hz),
    ))
    item.addItem(image)
    item.getViewBox().autoRange()
    return image


def render_spectrum(
    plot_widget: Any,
    samples: object,
    *,
    sample_rate_hz: float,
    use_db: bool = False,
) -> SpectrumView:
    """Prepare and render a real or IQ amplitude spectrum."""

    view = compute_spectrum(samples, sample_rate_hz=sample_rate_hz)
    plot_item = _plot_item(plot_widget)
    _reset_axes(plot_item)
    invert_y = getattr(plot_widget, "invertY", None)
    if callable(invert_y):
        invert_y(False)
    plot_item.clear()
    plot_item.setTitle(None)
    plot_item.getAxis("left").setTicks(None)
    plot_item.getAxis("bottom").setTicks(None)
    plot_item.setLabel("bottom", "频率", units="Hz")
    plot_item.setLabel("left", "幅度（dB，参考值 1）" if use_db else "幅度（|FFT|/N）", units="")
    values = view.power_db if use_db else view.magnitude
    plot_item.plot(view.frequency_hz, values, pen=_SPECTRUM_PEN, name=view.channel_label)
    plot_item.getViewBox().autoRange()
    return view
