"""Stable, UI-independent result contract for future UAV inference.

This module intentionally contains no model loading or signal processing. It defines
what a UAV backend must return before the UI is allowed to publish a diagnosis.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QualityStatus(str, Enum):
    ACCEPTED = "accepted"
    CAUTION = "caution"
    REJECTED = "rejected"


class RecognitionStatus(str, Enum):
    NOT_RUN = "not_run"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class UAVInputInfo:
    filename: str
    sample_rate_hz: Optional[float] = None
    sample_count: Optional[int] = None
    channels: Optional[int] = None

    def __post_init__(self):
        if not str(self.filename).strip():
            raise ValueError("无人机输入文件名不能为空")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("采样率必须为正数")
        if self.sample_count is not None and (isinstance(self.sample_count, bool) or self.sample_count <= 0):
            raise ValueError("采样点数必须为正整数")
        if self.channels is not None and (isinstance(self.channels, bool) or self.channels <= 0):
            raise ValueError("通道数必须为正整数")


@dataclass(frozen=True)
class UAVRecognitionResult:
    input_info: UAVInputInfo
    quality_status: QualityStatus
    quality_message: str
    recognition_status: RecognitionStatus
    label: Optional[str] = None
    known_unknown: Optional[str] = None
    confidence: Optional[float] = None
    model_version: str = ""
    elapsed_seconds: Optional[float] = None
    window_count: Optional[int] = None
    open_set_status: Optional[str] = None
    contract_id: str = ""
    limitations: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.quality_status, QualityStatus):
            raise ValueError("质量状态必须使用 QualityStatus")
        if not isinstance(self.recognition_status, RecognitionStatus):
            raise ValueError("识别状态必须使用 RecognitionStatus")
        if not str(self.quality_message).strip():
            raise ValueError("必须提供可理解的质量说明")
        if self.known_unknown not in (None, "known", "unknown"):
            raise ValueError("目标状态只能是 known、unknown 或空")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("置信度必须在 0 到 1 之间")
        if self.elapsed_seconds is not None and self.elapsed_seconds < 0:
            raise ValueError("耗时不能为负数")
        if self.window_count is not None and (
            isinstance(self.window_count, bool) or self.window_count < 1
        ):
            raise ValueError("窗口数必须为正整数")
        if self.open_set_status is not None and not str(self.open_set_status).strip():
            raise ValueError("开放集状态不能为空字符串")
        if not isinstance(self.limitations, tuple) or any(
            not str(item).strip() for item in self.limitations
        ):
            raise ValueError("限制说明必须是非空字符串元组")
        if self.quality_status is QualityStatus.REJECTED:
            if self.label is not None or self.known_unknown is not None or self.confidence is not None:
                raise ValueError("质量拒绝时不得发布类别、目标状态或置信度")
        if self.recognition_status is not RecognitionStatus.COMPLETED:
            if any(value is not None for value in (self.label, self.known_unknown, self.confidence)):
                raise ValueError("识别未完成时不得发布识别结果")
        if self.known_unknown == "unknown" and self.label not in (None, "未知信号"):
            raise ValueError("未知目标不得伪装成已知类别")

    @property
    def can_publish_prediction(self) -> bool:
        return (self.quality_status is not QualityStatus.REJECTED and
                self.recognition_status is RecognitionStatus.COMPLETED and
                self.label is not None)


__all__ = ["QualityStatus", "RecognitionStatus", "UAVInputInfo", "UAVRecognitionResult"]
