"""In-memory component/envelope/frequency panels, not a Hilbert energy spectrum."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtGui import QImage, QPixmap


def render_hht_image(view):
    """Compatibility UI-thread entry point returning a QPixmap."""
    return QPixmap.fromImage(render_hht_qimage(view))


def render_hht_qimage(view):
    figure = Figure(figsize=(8, max(4, 1.5*len(view.components))), dpi=100, constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(len(view.components), 2, squeeze=False)
    for i, part in enumerate(view.components):
        left, right = axes[i]
        left.plot(view.time_s, part.values, lw=.7)
        left.plot(view.time_s, part.envelope, lw=.7, color='#b84c81')
        left.set_title(f'{part.label} + envelope')
        left.set_ylabel('Amplitude')
        right.plot(part.frequency_time_s, part.frequency_hz, lw=.7)
        right.set_title('Instantaneous frequency\n(gaps = undefined)', fontsize=10)
        right.set_ylabel('Hz')
        left.set_xlabel('Time (s)'); right.set_xlabel('Time (s)')
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    image = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy()
    figure.clear()
    return image
