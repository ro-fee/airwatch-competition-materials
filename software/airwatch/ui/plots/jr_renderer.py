"""Disk-free J/R scatter rendering for legacy QLabel destinations."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtGui import QImage, QPixmap


def render_jr_image(view):
    """Compatibility UI-thread entry point returning a QPixmap."""
    return QPixmap.fromImage(render_jr_qimage(view))


def render_jr_qimage(view):
    figure = Figure(figsize=(6,5), dpi=110, constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    if view.j.size:
        axis.scatter(view.j, view.r, marker='x', s=30, color='#b84c81')
    else:
        axis.text(.5,.5,'No valid points: all windows have zero energy',
                  ha='center', va='center', transform=axis.transAxes)
    axis.set_xlabel('J (dimensionless)')
    axis.set_ylabel('R (dimensionless)')
    axis.set_title(f'J/R envelope statistics | {view.start_s:.4g}–{view.end_s:.4g} s\n'
        f'{len(view.j)}/{view.total_windows} valid | zero: {len(view.zero_window_indices)} | tail: {view.dropped_samples}\n'
        f'Window: {view.window_samples} samples ({1000*view.window_samples/view.sample_rate_hz:.4g} ms)')
    axis.grid(alpha=.2)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    image = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy()
    figure.clear()
    return image
