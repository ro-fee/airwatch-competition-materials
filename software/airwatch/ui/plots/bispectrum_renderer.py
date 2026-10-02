"""In-memory bispectrum rendering for existing QLabel slots."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtGui import QImage, QPixmap


def render_bispectrum_image(view):
    """Compatibility UI-thread entry point returning a QPixmap."""
    return QPixmap.fromImage(render_bispectrum_qimage(view))


def render_bispectrum_qimage(view):
    figure = Figure(figsize=(6,5), dpi=110, constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    step = view.sample_rate_hz / 128
    edges = np.r_[view.frequency_hz-step/2, view.frequency_hz[-1]+step/2]
    image = axis.pcolormesh(edges, edges, view.magnitude, shading='flat',
                           cmap='magma', vmin=0, vmax=float(view.magnitude.max()) or 1)
    axis.set_xlabel('f2 (Hz)')
    axis.set_ylabel('f1 (Hz)')
    axis.set_aspect('equal')
    axis.set_title(f'Legacy indirect bispectrum | {view.start_s:.4g}–{view.end_s:.4g} s\n'
                   f'10 records / lag 20 / FFT 128 | tail omitted: {view.dropped_samples}')
    figure.colorbar(image, ax=axis, label='|B| (uncalibrated; not bicoherence)')
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy()
    figure.clear()
    return qimage
