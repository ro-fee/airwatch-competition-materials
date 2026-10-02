"""Application workflow for the development-only KU Leuven runtime release."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Callable

from airwatch.analysis.uav_quality import UAVQualityStatus
from airwatch.inference.ku_leuven import (
    KULeuvenInferenceError,
    KULeuvenKnownSourcePredictor,
    KULeuvenRecordingPrediction,
    load_ku_leuven_runtime_contract,
)
from airwatch.runtime_paths import resource_path
from airwatch.runtime_resources import KU_LEUVEN_SOFTWARE_CONTRACT
from airwatch.workflows.uav_intake import UAVPreparedInput, prepare_uav_input

from .uav_contract import (
    QualityStatus,
    RecognitionStatus,
    UAVInputInfo,
    UAVRecognitionResult,
)


DEFAULT_KU_LEUVEN_SOFTWARE_CONTRACT = resource_path(KU_LEUVEN_SOFTWARE_CONTRACT)


class KULeuvenRecognitionWorkflow:
    """Bridge verified known-source inference to the stable UI result type."""

    def __init__(self, predictor: KULeuvenKnownSourcePredictor) -> None:
        if not isinstance(predictor, KULeuvenKnownSourcePredictor):
            raise TypeError("predictor 必须是 KULeuvenKnownSourcePredictor")
        self.predictor = predictor

    @classmethod
    def from_contract(
        cls,
        contract_path: str | Path = DEFAULT_KU_LEUVEN_SOFTWARE_CONTRACT,
        *,
        project_root: str | Path | None = None,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> "KULeuvenRecognitionWorkflow":
        contract = load_ku_leuven_runtime_contract(
            contract_path, project_root=project_root
        )
        return cls(
            KULeuvenKnownSourcePredictor(
                contract, device=device, batch_size=batch_size
            )
        )

    def predict_array(
        self,
        samples: object,
        *,
        cancelled: Event | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> KULeuvenRecordingPrediction:
        return self.predictor.predict_recording(
            samples, cancelled=cancelled, progress=progress
        )

    def predict_file(
        self,
        path: str | Path,
        *,
        sample_rate_hz: float | None = None,
        cancelled: Event | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> KULeuvenRecordingPrediction:
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError
        try:
            prepared = prepare_uav_input(
                path,
                declared_sample_rate_hz=sample_rate_hz,
                cancelled=cancelled,
            )
        except (OSError, ValueError) as exc:
            raise KULeuvenInferenceError(f"无法读取无人机 NPY 文件：{exc}") from exc
        return self.predict_prepared(
            prepared,
            cancelled=cancelled,
            progress=progress,
        )

    def _validate_prepared_input(self, prepared: UAVPreparedInput) -> UAVInputInfo:
        if not isinstance(prepared, UAVPreparedInput):
            raise TypeError("无人机后台输入必须是 UAVPreparedInput")
        contract = self.predictor.contract
        if prepared.declared_sample_rate_hz is None:
            raise KULeuvenInferenceError(
                f"识别前必须填写真实采样率；当前模型要求 {contract.sample_rate_hz} Hz"
            )
        if prepared.declared_sample_rate_hz != contract.sample_rate_hz:
            raise KULeuvenInferenceError(
                f"模型要求 {contract.sample_rate_hz} Hz，当前输入声明为 "
                f"{prepared.declared_sample_rate_hz:g} Hz"
            )
        signal = prepared.signal
        if signal.channels != contract.input_channels:
            raise KULeuvenInferenceError(
                f"模型要求 {contract.input_channels} 通道 IQ，当前输入为 "
                f"{signal.channels} 通道"
            )
        expected_samples = contract.windows_per_recording * contract.window_samples
        if signal.sample_count != expected_samples:
            raise KULeuvenInferenceError(
                f"模型严格要求 {expected_samples} 个连续采样点，当前输入为 "
                f"{signal.sample_count} 个"
            )
        return UAVInputInfo(
            filename=prepared.identity.path,
            sample_rate_hz=prepared.declared_sample_rate_hz,
            sample_count=signal.sample_count,
            channels=signal.channels,
        )

    def predict_prepared(
        self,
        prepared: UAVPreparedInput,
        *,
        cancelled: Event | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> KULeuvenRecordingPrediction:
        self._validate_prepared_input(prepared)
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError
        return self.predict_array(
            prepared.signal.samples,
            cancelled=cancelled,
            progress=progress,
        )

    def as_background_backend(
        self,
        prepared: UAVPreparedInput,
        cancelled: Event,
        progress: Callable[[int, int], None],
    ) -> UAVRecognitionResult:
        """Callable matching :class:`airwatch.ui.uav_background.UAVRecognitionTask`."""
        input_info = self._validate_prepared_input(prepared)
        contract = self.predictor.contract
        quality = prepared.quality
        if quality.status is UAVQualityStatus.REJECTED:
            return UAVRecognitionResult(
                input_info=input_info,
                quality_status=QualityStatus.REJECTED,
                quality_message=quality.message,
                recognition_status=RecognitionStatus.NOT_RUN,
                model_version=contract.model_version,
                open_set_status=contract.open_set_status,
                contract_id=contract.contract_id,
                limitations=contract.limitations,
            )
        detail = self.predict_prepared(
            prepared, cancelled=cancelled, progress=progress
        )
        quality_message = quality.message
        quality_message += (
            "；尚未建立物理采集质量门，当前仅发布开发版已知信号源分类。"
        )
        return UAVRecognitionResult(
            input_info=input_info,
            quality_status=QualityStatus.CAUTION,
            quality_message=quality_message,
            recognition_status=RecognitionStatus.COMPLETED,
            label=detail.display_label,
            known_unknown=None,
            confidence=detail.confidence,
            model_version=detail.model_version,
            elapsed_seconds=detail.elapsed_seconds,
            window_count=detail.window_count,
            open_set_status=detail.open_set_status,
            contract_id=detail.contract_id,
            limitations=detail.limitations,
        )


class LazyKULeuvenRecognitionBackend:
    """Load the verified model on the first worker-thread invocation only."""

    def __init__(
        self,
        contract_path: str | Path = DEFAULT_KU_LEUVEN_SOFTWARE_CONTRACT,
        *,
        project_root: str | Path | None = None,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.contract_path = Path(contract_path)
        self.project_root = project_root
        self.device = device
        self.batch_size = batch_size
        self._workflow: KULeuvenRecognitionWorkflow | None = None

    @property
    def loaded(self) -> bool:
        return self._workflow is not None

    def __call__(
        self,
        prepared: UAVPreparedInput,
        cancelled: Event,
        progress: Callable[[int, int], None],
    ) -> UAVRecognitionResult:
        if cancelled.is_set():
            raise InterruptedError
        if self._workflow is None:
            self._workflow = KULeuvenRecognitionWorkflow.from_contract(
                self.contract_path,
                project_root=self.project_root,
                device=self.device,
                batch_size=self.batch_size,
            )
        if cancelled.is_set():
            raise InterruptedError
        return self._workflow.as_background_backend(prepared, cancelled, progress)


__all__ = [
    "DEFAULT_KU_LEUVEN_SOFTWARE_CONTRACT",
    "KULeuvenRecognitionWorkflow",
    "LazyKULeuvenRecognitionBackend",
]
