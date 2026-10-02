"""A truthful registry of signal visualisations.

The registry is deliberately plain Python data.  It gives the future Qt page
one stable source for labels, explanations, input requirements, and migration
status without importing widgets or pretending that a planned renderer exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PlotAvailability = Literal[
    "current",
    "legacy_to_migrate",
    "legacy_entry_to_verify",
    "planned",
    "reserved",
]


@dataclass(frozen=True)
class SignalViewSpec:
    """Metadata needed to present one visualisation in the workbench."""

    key: str
    title: str
    plain_explanation: str
    required_data: str
    requires_sample_rate: bool
    supports_real: bool
    supports_iq: bool
    supports_dynamic_playback: bool
    availability: PlotAvailability
    workspaces: tuple[str, ...]
    legacy_location: str
    renderer_key: str


_SPECS: tuple[SignalViewSpec, ...] = (
    SignalViewSpec(
        key="waveform",
        title="时域波形",
        plain_explanation="看信号随时间（或采样点）上下变化，适合观察脉冲、突变和整体幅度。",
        required_data="real_or_iq",
        requires_sample_rate=False,
        supports_real=True,
        supports_iq=True,
        supports_dynamic_playback=False,
        availability="current",
        workspaces=("general", "uav", "bearing"),
        legacy_location="main.py:plotsig; newwindow_.py:label_signal/label_signalshow_1",
        renderer_key="pyqtgraph_line",
    ),
    SignalViewSpec(
        key="spectrum",
        title="频域 / 频谱图",
        plain_explanation="把信号拆成不同频率，帮助观察能量集中在哪些频段。",
        required_data="real_or_iq",
        requires_sample_rate=True,
        supports_real=True,
        supports_iq=True,
        supports_dynamic_playback=False,
        availability="current",
        workspaces=("general", "uav", "bearing"),
        legacy_location="main.py:plotspectrum; newwindow_.py:radioButton_PP_3",
        renderer_key="pyqtgraph_line",
    ),
    SignalViewSpec(
        key="spectrogram",
        title="静态时频图",
        plain_explanation="同时看时间和频率：能量什么时候出现、频率有没有变化。",
        required_data="real_or_iq",
        requires_sample_rate=True,
        supports_real=True,
        supports_iq=True,
        supports_dynamic_playback=False,
        availability="current",
        workspaces=("general", "uav", "bearing"),
        legacy_location="main.py:plotspec; waterfall_plotter.py:spectrogram",
        renderer_key="pyqtgraph_image",
    ),
    SignalViewSpec(
        key="waterfall",
        title="动态频谱瀑布图",
        plain_explanation="把连续的频谱一行行堆起来并播放，适合观察频段随时间移动或出现。",
        required_data="real_or_iq",
        requires_sample_rate=True,
        supports_real=True,
        supports_iq=True,
        supports_dynamic_playback=True,
        availability="current",
        workspaces=("general", "uav"),
        legacy_location="waterfall_plotter.py:WaterfallPlotter; main.py:_create_waterfall_controls",
        renderer_key="waterfall_plotter",
    ),
    SignalViewSpec(
        key="constellation",
        title="星座图",
        plain_explanation="把 I/Q 两路样本画成点，观察调制信号的点群形状是否清晰。",
        required_data="iq_or_explicit_analytic_real",
        requires_sample_rate=False,
        supports_real=False,
        supports_iq=True,
        supports_dynamic_playback=False,
        availability="current",
        workspaces=("general",),
        legacy_location="main.py:xzt/_display_generated_sample; newwindow_.py:radioButton_2",
        renderer_key="pyqtgraph_scatter",
    ),
    SignalViewSpec(
        key="feature_map",
        title="中间层特征图",
        plain_explanation="看模型中间层提取出的模式，帮助专业用户理解模型在关注什么。",
        required_data="model_feature_tensor",
        requires_sample_rate=False,
        supports_real=True,
        supports_iq=True,
        supports_dynamic_playback=False,
        availability="current",
        workspaces=("general",),
        legacy_location="main.py:plotspec_featuremap/plotsig_featuremap",
        renderer_key="pyqtgraph_line_or_image",
    ),
    SignalViewSpec(
        key="tsne",
        title="特征分布图",
        plain_explanation="把模型提取的高维特征压到二维，观察不同样本是否形成分开的分组。",
        required_data="batch_feature_matrix",
        requires_sample_rate=False,
        supports_real=True,
        supports_iq=True,
        supports_dynamic_playback=False,
        availability="current",
        workspaces=("general",),
        legacy_location="main.py:choose_feature_signal; features/<task>_tsne.png",
        renderer_key="pyqtgraph_scatter",
    ),
    SignalViewSpec(
        key="wavelet",
        title="小波特征",
        plain_explanation="把信号拆成不同快慢的变化，适合找突发、冲击和局部细节。",
        required_data="real_signal",
        requires_sample_rate=True,
        supports_real=True,
        supports_iq=False,
        supports_dynamic_playback=False,
        availability="current",
        workspaces=("general",),
        legacy_location="main.py:features('小波特征'); result/WVTSigs.png",
        renderer_key="image_or_multi_line",
    ),
    SignalViewSpec(
        key="bispectrum",
        title="双谱特征",
        plain_explanation="检查不同频率之间是否存在更复杂的联动，属于专家分析图。",
        required_data="real_signal",
        requires_sample_rate=True,
        supports_real=True,
        supports_iq=False,
        supports_dynamic_playback=False,
        availability="current",
        workspaces=("general",),
        legacy_location="main.py:_show_bispectrum_feature; airwatch/analysis/bispectrum_views.py",
        renderer_key="image",
    ),
    SignalViewSpec(
        key="jr",
        title="J/R 特征",
        plain_explanation="用两个统计量概括信号形状，适合做样本之间的对比。",
        required_data="real_signal",
        requires_sample_rate=True,
        supports_real=True,
        supports_iq=False,
        supports_dynamic_playback=False,
        availability="current",
        workspaces=("general",),
        legacy_location="main.py:_show_jr_feature; airwatch/analysis/jr_views.py",
        renderer_key="image_or_scatter",
    ),
    SignalViewSpec(
        key="hht",
        title="HHT 特征",
        plain_explanation="分解信号并观察瞬时频率；需可选 EMD-signal，边界效应和噪声可能影响解释。",
        required_data="real_signal_plus_optional_hht_dependencies",
        requires_sample_rate=True,
        supports_real=True,
        supports_iq=False,
        supports_dynamic_playback=False,
        availability="current",
        workspaces=("general",),
        legacy_location="main.py:_start_hht_feature; airwatch/analysis/hht_views.py",
        renderer_key="image_or_multi_line",
    ),
    SignalViewSpec(
        key="bearing_fft",
        title="轴承故障频率图",
        plain_explanation="查看振动信号的频率峰值，辅助解释轴承可能对应的故障特征。",
        required_data="bearing_real_signal_plus_sampling_metadata",
        requires_sample_rate=True,
        supports_real=True,
        supports_iq=False,
        supports_dynamic_playback=False,
        availability="planned",
        workspaces=("bearing",),
        legacy_location="airwatch/inference/bearing.py:inference only; no legacy plot renderer found",
        renderer_key="pyqtgraph_line_with_markers",
    ),
    SignalViewSpec(
        key="event_timeline",
        title="事件时间轴",
        plain_explanation="把连续检测结果合并成可追溯的事件，方便回看何时出现异常。",
        required_data="structured_detection_events",
        requires_sample_rate=False,
        supports_real=False,
        supports_iq=False,
        supports_dynamic_playback=True,
        availability="reserved",
        workspaces=("uav",),
        legacy_location="docs/前端产品结构与功能规划.md:频谱事件区域",
        renderer_key="timeline",
    ),
)


def list_signal_view_specs(
    *,
    workspace: str | None = None,
    availability: PlotAvailability | None = None,
) -> tuple[SignalViewSpec, ...]:
    """Return registered views, optionally filtered for one work area."""

    if workspace is None and availability is None:
        return _SPECS
    return tuple(
        spec
        for spec in _SPECS
        if (workspace is None or workspace in spec.workspaces)
        and (availability is None or spec.availability == availability)
    )


def get_signal_view_spec(key: str) -> SignalViewSpec:
    """Look up one view by stable key."""

    for spec in _SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown signal view: {key!r}")


__all__ = [
    "PlotAvailability",
    "SignalViewSpec",
    "get_signal_view_spec",
    "list_signal_view_specs",
]
