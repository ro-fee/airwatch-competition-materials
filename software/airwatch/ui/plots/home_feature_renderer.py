"""Worker-safe raster preparation and UI-only publication of expert images."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from PyQt5.QtGui import QImage, QPixmap

from airwatch.analysis.home_features import HomeFeatureResult
from .bispectrum_renderer import render_bispectrum_qimage
from .hht_renderer import render_hht_qimage
from .jr_renderer import render_jr_qimage
from .wavelet_renderer import render_wavelet_qimage


_raster_lock = RLock()


@dataclass(frozen=True)
class PreparedHomeFeature:
    """Detached QImage pixels; immutable-by-contract after queued publication."""
    image: QImage
    tooltip: str
    status: str


@dataclass(frozen=True)
class HomeFeaturePresentation:
    pixmap: QPixmap
    tooltip: str
    status: str


def prepare_home_feature(result: HomeFeatureResult) -> PreparedHomeFeature:
    """Build the entire Agg image off-thread, serializing our Matplotlib users."""
    with _raster_lock:
        return _prepare_home_feature(result)


def _prepare_home_feature(result: HomeFeatureResult) -> PreparedHomeFeature:

    if not isinstance(result, HomeFeatureResult):
        raise TypeError("result must be a HomeFeatureResult")
    view = result.view
    if result.kind == "wavelet":
        image = render_wavelet_qimage(view)
        tooltip = f"小波 db4 / smooth，{view.level} 层；A 为近似分量，D 为细节分量。"
        status = (
            f"小波图已更新：采样点 [{result.request.start_sample}, "
            f"{result.request.end_sample})，{view.level} 层；"
            "A=近似，D=细节；无自动写盘。"
        )
    elif result.kind == "bispectrum":
        image = render_bispectrum_qimage(view)
        tooltip = "双谱：观察三阶频率关联；颜色为未校准幅值，不是功率或归一化双相干。"
        status = (
            f"双谱图已更新：{view.start_s:g}–{view.end_s:g} 秒，"
            f"使用 {view.used_samples} 点，末尾舍弃 {view.dropped_samples} 点；无自动写盘。"
        )
    elif result.kind == "jr":
        image = render_jr_qimage(view)
        tooltip = "横轴 J，纵轴 R；均为无量纲包络统计量，不是识别结果。零能量窗口不绘制假点。"
        status = (
            f"J/R：有效 {len(view.j)}/{view.total_windows} 窗，"
            f"零能量 {len(view.zero_window_indices)} 窗，"
            f"末尾舍弃 {view.dropped_samples} 点；横轴 J，纵轴 R。"
        )
    elif result.kind == "hht":
        image = render_hht_qimage(view)
        tooltip = (
            "EMD-signal：IMF 与残差分别标记；最多 8 个 IMF，筛分迭代上限 1000。"
            "噪声/边界效应会影响瞬时频率，不是故障分类结果。"
        )
        status = (
            f"HHT 已更新（{view.backend}）；8 IMF / 1000 次筛分上限，"
            "瞬时频率不是分类结论。"
        )
    else:
        raise ValueError(f"未登记的主页专家分析结果：{result.kind}")
    if image.isNull():
        raise ValueError("主页专家分析渲染返回空图像。")
    return PreparedHomeFeature(image, tooltip, status)


def render_home_feature(result: PreparedHomeFeature | HomeFeatureResult) -> HomeFeaturePresentation:
    """Publish prepared pixels as a QPixmap; call only on the Qt UI thread.

    Direct numerical/legacy callers may still supply an analysis result. The
    maintained HomeFeatureTask always supplies already-rasterized pixels.
    """
    prepared = result if isinstance(result, PreparedHomeFeature) else prepare_home_feature(result)
    return HomeFeaturePresentation(QPixmap.fromImage(prepared.image), prepared.tooltip, prepared.status)


__all__ = ["HomeFeaturePresentation", "PreparedHomeFeature", "prepare_home_feature", "render_home_feature"]
