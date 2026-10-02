"""Pure signal-to-plot transformations for the migrated workbench.

This module is deliberately independent of Qt, model loading, files, and page
layout. It validates signal shapes and returns read-only NumPy-backed view data
for renderers to consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import hilbert, spectrogram


SignalKind = Literal["real", "iq"]


class SignalViewError(ValueError):
    """Raised when a signal cannot satisfy a plot's input contract."""


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array).copy()
    result.setflags(write=False)
    return result


def _validate_sample_rate(sample_rate_hz: float) -> float:
    if isinstance(sample_rate_hz, bool):
        raise SignalViewError("sample_rate_hz must be a positive finite number")
    try:
        value = float(sample_rate_hz)
    except (TypeError, ValueError) as exc:
        raise SignalViewError("sample_rate_hz must be a positive finite number") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise SignalViewError("sample_rate_hz must be a positive finite number")
    return value


@dataclass(frozen=True)
class PreparedSignal:
    """Canonical one-dimensional signal used by all plot transformations."""

    samples: np.ndarray
    kind: SignalKind
    source_shape: tuple[int, ...]

    @property
    def sample_count(self) -> int:
        return int(self.samples.size)


def prepare_signal(samples: object) -> PreparedSignal:
    """Validate and canonicalize real or IQ samples without normalizing them.

    A two-channel real matrix is interpreted as I/Q only when one dimension is
    exactly two. Other matrices are rejected instead of silently flattened.
    """

    array = np.asarray(samples)
    source_shape = tuple(int(dimension) for dimension in array.shape)
    if array.ndim == 0 or array.ndim > 2:
        raise SignalViewError(
            f"signal must be one-dimensional or a two-channel matrix, got shape={source_shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise SignalViewError(f"signal must be numeric, got dtype={array.dtype}")

    if array.ndim == 1:
        if array.size == 0:
            raise SignalViewError("signal must not be empty")
        if not np.isfinite(array).all():
            raise SignalViewError("signal contains NaN or infinite values")
        kind: SignalKind = "iq" if np.iscomplexobj(array) else "real"
        canonical = np.asarray(
            array,
            dtype=np.complex128 if kind == "iq" else np.float64,
        )
        return PreparedSignal(_readonly(canonical), kind, source_shape)

    if np.iscomplexobj(array):
        raise SignalViewError(
            "a two-channel matrix must contain real I/Q components; use one-dimensional complex IQ instead"
        )
    if array.shape[0] == 2:
        channels = array
    elif array.shape[1] == 2:
        channels = array.T
    else:
        raise SignalViewError(
            f"two-dimensional signal must have exactly two channels, got shape={source_shape}"
        )
    if channels.shape[1] == 0:
        raise SignalViewError("signal must not be empty")
    if not np.isfinite(channels).all():
        raise SignalViewError("signal contains NaN or infinite values")
    canonical = np.asarray(channels[0], dtype=np.float64) + 1j * np.asarray(
        channels[1], dtype=np.float64
    )
    return PreparedSignal(_readonly(canonical), "iq", source_shape)


@dataclass(frozen=True)
class WaveformView:
    """Line-plot data for one real signal or the I/Q channels."""

    x: np.ndarray
    values: np.ndarray
    channel_labels: tuple[str, ...]
    x_label: str
    sample_rate_hz: float | None


def compute_waveform(
    samples: object,
    *,
    sample_rate_hz: float | None = None,
) -> WaveformView:
    """Return time/sample coordinates and one or two waveform channels."""

    prepared = prepare_signal(samples)
    rate = None if sample_rate_hz is None else _validate_sample_rate(sample_rate_hz)
    if rate is None:
        x = np.arange(prepared.sample_count, dtype=np.float64)
        x_label = "采样点"
    else:
        x = np.arange(prepared.sample_count, dtype=np.float64) / rate
        x_label = "时间 (s)"

    if prepared.kind == "iq":
        values = np.vstack((prepared.samples.real, prepared.samples.imag))
        labels = ("I", "Q")
    else:
        values = prepared.samples.reshape(1, -1)
        labels = ("幅度",)
    return WaveformView(
        x=_readonly(x),
        values=_readonly(values),
        channel_labels=labels,
        x_label=x_label,
        sample_rate_hz=rate,
    )


@dataclass(frozen=True)
class SpectrumView:
    """Frequency-domain line-plot data."""

    frequency_hz: np.ndarray
    magnitude: np.ndarray
    power_db: np.ndarray
    channel_label: str
    two_sided: bool
    sample_rate_hz: float


def compute_spectrum(samples: object, *, sample_rate_hz: float) -> SpectrumView:
    """Compute a one-sided real or centered complex/IQ amplitude spectrum."""

    prepared = prepare_signal(samples)
    rate = _validate_sample_rate(sample_rate_hz)
    count = prepared.sample_count
    if prepared.kind == "iq":
        transformed = np.fft.fftshift(np.fft.fft(prepared.samples))
        frequency = np.fft.fftshift(np.fft.fftfreq(count, d=1.0 / rate))
        label = "IQ"
        two_sided = True
    else:
        transformed = np.fft.rfft(prepared.samples)
        frequency = np.fft.rfftfreq(count, d=1.0 / rate)
        label = "实信号"
        two_sided = False
    magnitude = np.abs(transformed) / count
    power_db = 20.0 * np.log10(np.maximum(magnitude, np.finfo(np.float64).tiny))
    return SpectrumView(
        frequency_hz=_readonly(frequency),
        magnitude=_readonly(magnitude),
        power_db=_readonly(power_db),
        channel_label=label,
        two_sided=two_sided,
        sample_rate_hz=rate,
    )


@dataclass(frozen=True)
class SpectrogramView:
    """Static time-frequency image data in linear and dB form."""

    time_s: np.ndarray
    frequency_hz: np.ndarray
    power: np.ndarray
    power_db: np.ndarray
    sample_rate_hz: float
    two_sided: bool
    time_step_s: float
    frequency_step_hz: float


def compute_spectrogram(
    samples: object,
    *,
    sample_rate_hz: float,
    nperseg: int,
    noverlap: int | None = None,
    detrend: Literal[False, "constant"] = False,
) -> SpectrogramView:
    """Compute a static STFT-style spectrogram for real or IQ data."""

    if detrend is not False and detrend != "constant":
        raise SignalViewError("detrend must be False or 'constant'")

    prepared = prepare_signal(samples)
    rate = _validate_sample_rate(sample_rate_hz)
    if isinstance(nperseg, bool) or not isinstance(nperseg, (int, np.integer)):
        raise SignalViewError("nperseg must be a positive integer")
    window = int(nperseg)
    if window <= 0:
        raise SignalViewError("nperseg must be a positive integer")
    if window > prepared.sample_count:
        raise SignalViewError("nperseg cannot exceed signal length")
    if noverlap is not None:
        if isinstance(noverlap, bool) or not isinstance(noverlap, (int, np.integer)):
            raise SignalViewError("noverlap must be an integer smaller than nperseg")
        overlap = int(noverlap)
        if overlap < 0 or overlap >= window:
            raise SignalViewError("noverlap must be an integer smaller than nperseg")
    else:
        # Keep the time resolution stable across SciPy versions. SciPy's
        # implicit default is not part of this module's input contract, so the
        # migrated workbench uses the common 50% overlap explicitly.
        overlap = window // 2

    frequencies, times, power = spectrogram(
        prepared.samples,
        fs=rate,
        window="hann",
        nperseg=window,
        noverlap=overlap,
        detrend=detrend,
        mode="psd",
        scaling="density",
        return_onesided=prepared.kind != "iq",
    )
    if prepared.kind == "iq":
        order = np.argsort(frequencies)
        frequencies = frequencies[order]
        power = power[order, :]
    power = np.asarray(power, dtype=np.float64)
    power_db = 10.0 * np.log10(np.maximum(power, np.finfo(np.float64).tiny))
    return SpectrogramView(
        time_s=_readonly(times),
        frequency_hz=_readonly(frequencies),
        power=_readonly(power),
        power_db=_readonly(power_db),
        sample_rate_hz=rate,
        two_sided=prepared.kind == "iq",
        time_step_s=(window - overlap) / rate,
        frequency_step_hz=rate / window,
    )


@dataclass(frozen=True)
class ConstellationView:
    """I/Q point data suitable for a scatter plot."""

    i: np.ndarray
    q: np.ndarray
    source_kind: Literal["iq", "analytic_real"]

    @property
    def point_count(self) -> int:
        return int(self.i.size)


def compute_constellation(
    samples: object,
    *,
    max_points: int = 5000,
    analytic_from_real: bool = False,
) -> ConstellationView:
    """Return decimated I/Q points, optionally using a Hilbert analytic signal.

    A real signal has no independent Q channel and is rejected by default.
    The legacy modulation view used a Hilbert transform; callers may request
    that explicit compatibility mode with analytic_from_real=True.
    """

    prepared = prepare_signal(samples)
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)):
        raise SignalViewError("max_points must be a positive integer")
    point_limit = int(max_points)
    if point_limit <= 0:
        raise SignalViewError("max_points must be a positive integer")

    if prepared.kind == "real":
        if not analytic_from_real:
            raise SignalViewError(
                "constellation view requires IQ data or analytic_from_real=True"
            )
        analytic = hilbert(prepared.samples)
        source_kind: Literal["iq", "analytic_real"] = "analytic_real"
    else:
        analytic = prepared.samples
        source_kind = "iq"

    if analytic.size > point_limit:
        indices = np.linspace(0, analytic.size - 1, point_limit, dtype=np.int64)
        analytic = analytic[indices]
    return ConstellationView(
        i=_readonly(np.asarray(analytic.real, dtype=np.float64)),
        q=_readonly(np.asarray(analytic.imag, dtype=np.float64)),
        source_kind=source_kind,
    )


__all__ = [
    "ConstellationView",
    "PreparedSignal",
    "SignalKind",
    "SignalViewError",
    "SpectrogramView",
    "SpectrumView",
    "WaveformView",
    "compute_constellation",
    "compute_spectrogram",
    "compute_spectrum",
    "compute_waveform",
    "prepare_signal",
]
