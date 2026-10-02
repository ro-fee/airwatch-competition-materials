"""Lazy public exports for UI-independent workflows."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BearingBatchDiagnosisWorkflow": (".bearing_batch", "BearingBatchDiagnosisWorkflow"),
    "BearingBatchError": (".bearing_batch", "BearingBatchError"),
    "BearingBatchFileResult": (".bearing_batch", "BearingBatchFileResult"),
    "BearingBatchResult": (".bearing_batch", "BearingBatchResult"),
    "BearingDiagnosisWorkflow": (".bearing_diagnosis", "BearingDiagnosisWorkflow"),
    "BearingDiagnosisCancelled": (".bearing_progressive", "BearingDiagnosisCancelled"),
    "ProgressiveBearingDiagnosisResult": (".bearing_progressive", "ProgressiveBearingDiagnosisResult"),
    "ProgressiveBearingDiagnosisWorkflow": (".bearing_progressive", "ProgressiveBearingDiagnosisWorkflow"),
    "GenerationRequest": (".historical_generation", "GenerationRequest"),
    "GenerationResult": (".historical_generation", "GenerationResult"),
    "HistoricalGenerationWorkflow": (".historical_generation", "HistoricalGenerationWorkflow"),
    "QualityStatus": (".uav_contract", "QualityStatus"),
    "RecognitionStatus": (".uav_contract", "RecognitionStatus"),
    "UAVInputInfo": (".uav_contract", "UAVInputInfo"),
    "UAVRecognitionResult": (".uav_contract", "UAVRecognitionResult"),
    "UAVPreparedInput": (".uav_intake", "UAVPreparedInput"),
    "UAVSourceIdentity": (".uav_intake", "UAVSourceIdentity"),
    "prepare_uav_input": (".uav_intake", "prepare_uav_input"),
    "DEFAULT_KU_LEUVEN_SOFTWARE_CONTRACT": (".ku_leuven_recognition", "DEFAULT_KU_LEUVEN_SOFTWARE_CONTRACT"),
    "KULeuvenRecognitionWorkflow": (".ku_leuven_recognition", "KULeuvenRecognitionWorkflow"),
    "LazyKULeuvenRecognitionBackend": (".ku_leuven_recognition", "LazyKULeuvenRecognitionBackend"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
