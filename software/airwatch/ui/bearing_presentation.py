"""Presentation rules for frozen bearing diagnosis results.

This module contains no Qt and no model code.  It translates structured
workflow output into the three user-facing quality states used by the desktop
page, making the safety rule for rejected signals easy to unit-test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


_LABELS_ZH = {
    "normal": "正常",
    "inner_race": "内圈故障",
    "ball": "滚动体故障",
    "outer_race_6": "外圈故障（6 点钟方向）",
}

_STATE_TITLES = {
    "accepted": "质量通过",
    "caution": "质量提醒",
    "rejected": "已拒绝诊断",
}


@dataclass(frozen=True)
class BearingDiagnosisPresentation:
    """Plain text and style state consumed by the Qt main window."""

    quality_status: str
    state_title: str
    diagnosis_text: str
    confidence_text: str
    quality_message: str
    details_text: str
    can_show_prediction: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _counts_text(value: object) -> str:
    counts = value if isinstance(value, Mapping) else {}
    accepted = int(counts.get("accepted", 0))
    caution = int(counts.get("caution", 0))
    rejected = int(counts.get("rejected", 0))
    return f"通过 {accepted} / 提醒 {caution} / 拒绝 {rejected}"


def build_bearing_diagnosis_presentation(
    result: Mapping[str, Any],
) -> BearingDiagnosisPresentation:
    """Apply the frozen three-state display contract to one workflow result."""

    status = str(result.get("quality_status", ""))
    if status not in _STATE_TITLES:
        raise ValueError(f"unsupported bearing quality status: {status!r}")
    quality_message = str(result.get("quality_message") or "")
    window_count = int(result.get("window_count", 0))
    elapsed_seconds = float(result.get("elapsed_seconds", 0.0))
    details = (
        f"模型契约：{str(result.get('contract_id') or '未提供')}；"
        f"模型版本文件：{str(result.get('checkpoint_path') or '未提供').split(chr(92))[-1].split('/')[-1]}；"
        f"归一化：{str(result.get('normalization') or '未提供')}；"
        f"窗口：{window_count}；质量统计："
        f"{_counts_text(result.get('quality_status_counts'))}；"
        f"后台耗时：{elapsed_seconds:.3f} 秒"
    )

    if status == "rejected":
        return BearingDiagnosisPresentation(
            quality_status=status,
            state_title=_STATE_TITLES[status],
            diagnosis_text="未输出故障类别",
            confidence_text="未输出置信度",
            quality_message=quality_message or "信号质量不足，建议重新采集",
            details_text=details,
            can_show_prediction=False,
        )

    raw_label = result.get("predicted_label")
    raw_confidence = result.get("mean_confidence")
    if raw_label is None or raw_confidence is None:
        raise ValueError(f"{status} result must contain prediction and confidence")
    confidence = float(raw_confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("bearing confidence must be between 0 and 1")
    label = str(raw_label)
    translated = _LABELS_ZH.get(label, label)
    return BearingDiagnosisPresentation(
        quality_status=status,
        state_title=_STATE_TITLES[status],
        diagnosis_text=f"诊断结果：{translated}",
        confidence_text=f"平均置信度：{confidence * 100:.2f}%",
        quality_message=quality_message,
        details_text=details,
        can_show_prediction=True,
    )


__all__ = [
    "BearingDiagnosisPresentation",
    "build_bearing_diagnosis_presentation",
]
