"""Frozen requests for the four home-page expert signal analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .bispectrum_views import compute_bispectrum, prepare_bispectrum_signal
from .hht_views import compute_hht, prepare_hht, prepare_hht_signal
from .jr_views import compute_jr, prepare_jr_signal
from .signal_transforms import _readonly, _validate_sample_rate
from .wavelet_views import compute_wavelet, prepare_wavelet_signal


HomeFeatureKind = Literal["wavelet", "bispectrum", "jr", "hht"]
HOME_FEATURE_KINDS = frozenset({"wavelet", "bispectrum", "jr", "hht"})

_PREPARERS = {
    "wavelet": prepare_wavelet_signal,
    "bispectrum": prepare_bispectrum_signal,
    "jr": prepare_jr_signal,
    "hht": prepare_hht_signal,
}


@dataclass(frozen=True)
class HomeFeatureRequest:
    """Owned analysis input that cannot drift after a task starts."""

    kind: HomeFeatureKind
    signal: np.ndarray
    sample_rate_hz: float
    start_sample: int

    def __post_init__(self) -> None:
        if self.kind not in HOME_FEATURE_KINDS:
            raise ValueError(f"未登记的主页专家分析：{self.kind}")
        rate = _validate_sample_rate(self.sample_rate_hz)
        if (
            isinstance(self.start_sample, bool)
            or not isinstance(self.start_sample, (int, np.integer))
            or self.start_sample < 0
        ):
            raise ValueError("选区起点必须是非负整数采样点。")
        signal = np.asarray(self.signal)
        if signal.ndim != 1 or np.iscomplexobj(signal) or signal.dtype.kind not in "iuf":
            raise ValueError("主页专家分析请求必须包含一维实信号。")
        if signal.size == 0 or not np.isfinite(signal).all():
            raise ValueError("主页专家分析请求必须包含非空有限信号。")
        object.__setattr__(self, "signal", _readonly(np.asarray(signal, dtype=np.float64)))
        object.__setattr__(self, "sample_rate_hz", rate)
        object.__setattr__(self, "start_sample", int(self.start_sample))

    @property
    def end_sample(self) -> int:
        return self.start_sample + int(self.signal.size)


@dataclass(frozen=True)
class HomeFeatureResult:
    kind: HomeFeatureKind
    request: HomeFeatureRequest
    view: object


def prepare_home_feature_request(
    kind: HomeFeatureKind,
    data: object,
    sample_rate_hz: object,
    *,
    start_sample: int = 0,
    end_sample: int | None = None,
) -> HomeFeatureRequest:
    """Validate the full source and freeze exactly the selected real samples."""

    if kind not in HOME_FEATURE_KINDS:
        raise ValueError(f"未登记的主页专家分析：{kind}")
    signal = _PREPARERS[kind](data)
    if (
        isinstance(start_sample, bool)
        or not isinstance(start_sample, (int, np.integer))
        or start_sample < 0
    ):
        raise ValueError("选区起点必须是非负整数采样点。")
    stop = len(signal) if end_sample is None else end_sample
    if isinstance(stop, bool) or not isinstance(stop, (int, np.integer)):
        raise ValueError("选区终点必须是整数采样点。")
    start, stop = int(start_sample), int(stop)
    if start >= stop or stop > len(signal):
        raise ValueError("主页专家分析选区无效，请重新选择时域范围。")
    selected = _readonly(signal[start:stop])
    if kind == "hht":
        prepared = prepare_hht(selected, sample_rate_hz, start)
        selected, rate = prepared.signal, prepared.sample_rate_hz
    else:
        rate = _validate_sample_rate(sample_rate_hz)
    return HomeFeatureRequest(kind, selected, rate, start)


def compute_home_feature(
    request: HomeFeatureRequest,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[str], None] = lambda _message: None,
) -> HomeFeatureResult:
    """Run one frozen request without importing Qt or rendering an image."""

    if not isinstance(request, HomeFeatureRequest):
        raise TypeError("request must be a HomeFeatureRequest")
    if cancelled():
        raise InterruptedError("主页专家分析已取消。")
    if request.kind == "hht":
        # Revalidate the specialised HHT contract even when a caller constructs
        # HomeFeatureRequest directly instead of using the public factory.
        source = prepare_hht(
            request.signal,
            request.sample_rate_hz,
            request.start_sample,
        )
        view = compute_hht(source, cancelled=cancelled, progress=progress)
    else:
        labels = {"wavelet": "小波", "bispectrum": "双谱", "jr": "J/R"}
        progress(f"正在后台计算{labels[request.kind]}（无百分比进度）")
        if request.kind == "wavelet":
            view = compute_wavelet(
                request.signal,
                request.sample_rate_hz,
                start_sample=request.start_sample,
            )
        elif request.kind == "bispectrum":
            view = compute_bispectrum(
                request.signal,
                request.sample_rate_hz,
                start_sample=request.start_sample,
            )
        else:
            view = compute_jr(
                request.signal,
                request.sample_rate_hz,
                start_sample=request.start_sample,
            )
    if cancelled():
        raise InterruptedError("主页专家分析已取消。")
    return HomeFeatureResult(request.kind, request, view)


__all__ = [
    "HOME_FEATURE_KINDS",
    "HomeFeatureKind",
    "HomeFeatureRequest",
    "HomeFeatureResult",
    "compute_home_feature",
    "prepare_home_feature_request",
]
