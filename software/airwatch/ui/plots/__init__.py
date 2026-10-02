"""Plot metadata and renderer seams for the migrated UI."""

from .plot_registry import (
    PlotAvailability,
    SignalViewSpec,
    get_signal_view_spec,
    list_signal_view_specs,
)
from .pyqtgraph_renderer import render_constellation, render_spectrogram, render_spectrum, render_waveform

__all__ = [
    "PlotAvailability",
    "SignalViewSpec",
    "get_signal_view_spec",
    "list_signal_view_specs",
    "render_spectrum",
    "render_constellation",
    "render_spectrogram",
    "render_waveform",
]
