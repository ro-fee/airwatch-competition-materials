"""Lazy exports for frozen inference modules."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BearingInferenceAdapter": (".adapter", "BearingInferenceAdapter"),
    "BearingInferenceResult": (".adapter", "BearingInferenceResult"),
    "BearingRuntimeContractError": (".bearing_contract", "BearingRuntimeContractError"),
    "DEFAULT_FROZEN_BEARING_CONTRACT": (".bearing_contract", "DEFAULT_FROZEN_BEARING_CONTRACT"),
    "FrozenBearingRuntimeContract": (".bearing_contract", "FrozenBearingRuntimeContract"),
    "load_frozen_bearing_runtime_contract": (".bearing_contract", "load_frozen_bearing_runtime_contract"),
    "BearingInferenceError": (".bearing", "BearingInferenceError"),
    "BearingPredictor": (".bearing", "BearingPredictor"),
    "InsufficientSignalError": (".bearing", "InsufficientSignalError"),
    "QualityAwareWindowPrediction": (".bearing", "QualityAwareWindowPrediction"),
    "QualityAwareWindowPredictionV2": (".bearing", "QualityAwareWindowPredictionV2"),
    "WindowPrediction": (".bearing", "WindowPrediction"),
    "ClassPrototypes": (".uav_open_set", "ClassPrototypes"),
    "OpenSetCalibration": (".uav_open_set", "OpenSetCalibration"),
    "UAVOpenSetError": (".uav_open_set", "UAVOpenSetError"),
    "aggregate_recording_scores": (".uav_open_set", "aggregate_recording_scores"),
    "aggregate_recording_probabilities": (".uav_open_set", "aggregate_recording_probabilities"),
    "aggregate_single_recording_probabilities": (".uav_open_set", "aggregate_single_recording_probabilities"),
    "calibrate_known_acceptance": (".uav_open_set", "calibrate_known_acceptance"),
    "decide_known": (".uav_open_set", "decide_known"),
    "energy_knownness": (".uav_open_set", "energy_knownness"),
    "fit_class_prototypes": (".uav_open_set", "fit_class_prototypes"),
    "maximum_softmax_knownness": (".uav_open_set", "maximum_softmax_knownness"),
    "prototype_knownness": (".uav_open_set", "prototype_knownness"),
    "DEFAULT_KU_LEUVEN_DEVELOPMENT_CONTRACT": (".ku_leuven", "DEFAULT_KU_LEUVEN_DEVELOPMENT_CONTRACT"),
    "KULeuvenInferenceError": (".ku_leuven", "KULeuvenInferenceError"),
    "KULeuvenKnownSourcePredictor": (".ku_leuven", "KULeuvenKnownSourcePredictor"),
    "KULeuvenRecordingPrediction": (".ku_leuven", "KULeuvenRecordingPrediction"),
    "KULeuvenRuntimeContract": (".ku_leuven", "KULeuvenRuntimeContract"),
    "KULeuvenWindowPrediction": (".ku_leuven", "KULeuvenWindowPrediction"),
    "load_ku_leuven_runtime_contract": (".ku_leuven", "load_ku_leuven_runtime_contract"),
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
