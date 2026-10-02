"""In-memory wavelet figure for legacy QLabel slots; no PNG read/write."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtGui import QImage, QPixmap


def render_wavelet_image(view):
    """Compatibility UI-thread entry point returning a QPixmap."""
    return QPixmap.fromImage(render_wavelet_qimage(view))


def render_wavelet_qimage(view):
    """Rasterize to independently owned pixels, without touching GUI resources."""
    figure = Figure(figsize=(8, 7), dpi=110, constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    grid = figure.add_gridspec(view.level + 1, 2)
    main = figure.add_subplot(grid[0, :])
    main.plot(view.time_s, view.signal, linewidth=.7)
    main.set_title(f'db4 / smooth / {view.level} levels | Original signal')
    main.set_ylabel('Amplitude')
    main.set_xlabel('Time (s)')
    for i, (a, d) in enumerate(zip(view.approximations, view.details)):
        for col, values, name, color in ((0, a, f'A{i+1}', '#b84c81'), (1, d, f'D{i+1}', '#197a91')):
            axis = figure.add_subplot(grid[i+1, col])
            axis.plot(view.time_s, values, color=color, linewidth=.7)
            axis.set_ylabel(name)
            if i == view.level-1:
                axis.set_xlabel('Time (s)')
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    # Copy detaches image ownership from the short-lived Agg canvas.
    image = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy()
    figure.clear()
    return image
