"""Lazy public exports for signal-analysis modules."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BearingFrequencies": (".bearing_fault_frequencies", "BearingFrequencies"),
    "BearingGeometry": (".bearing_fault_frequencies", "BearingGeometry"),
    "compute_characteristic_frequencies": (
        ".bearing_fault_frequencies",
        "compute_characteristic_frequencies",
    ),
    "BearingQualityResult": (".bearing_quality", "BearingQualityResult"),
    "BearingQualityThresholds": (".bearing_quality", "BearingQualityThresholds"),
    "BearingQualityThresholdsV2": (".bearing_quality", "BearingQualityThresholdsV2"),
    "assess_bearing_window": (".bearing_quality", "assess_bearing_window"),
    "assess_bearing_window_v2": (".bearing_quality", "assess_bearing_window_v2"),
    "load_bearing_quality_thresholds": (".bearing_quality", "load_bearing_quality_thresholds"),
    "load_bearing_quality_thresholds_v2": (".bearing_quality", "load_bearing_quality_thresholds_v2"),
    "BearingPerturbation": (".bearing_robustness", "BearingPerturbation"),
    "PerturbationKind": (".bearing_robustness", "PerturbationKind"),
    "apply_bearing_perturbation": (".bearing_robustness", "apply_bearing_perturbation"),
    "ControlledAWGNDataset": (".controlled_noise", "ControlledAWGNDataset"),
    "ControlledNoiseError": (".controlled_noise", "ControlledNoiseError"),
    "DeterministicNoiseAugmentedDataset": (".controlled_noise", "DeterministicNoiseAugmentedDataset"),
    "add_awgn_at_relative_snr": (".controlled_noise", "add_awgn_at_relative_snr"),
    "sample_noise_seed": (".controlled_noise", "sample_noise_seed"),
    "DeterministicIQPerturbationDataset": (".iq_perturbations", "DeterministicIQPerturbationDataset"),
    "IQPerturbationError": (".iq_perturbations", "IQPerturbationError"),
    "IQPerturbationSpec": (".iq_perturbations", "IQPerturbationSpec"),
    "MULTIPATH_PROFILE_NAMES": (".iq_perturbations", "MULTIPATH_PROFILE_NAMES"),
    "MULTIPATH_PROFILES": (".iq_perturbations", "MULTIPATH_PROFILES"),
    "add_complex_awgn": (".iq_perturbations", "add_complex_awgn"),
    "apply_carrier_frequency_offset": (".iq_perturbations", "apply_carrier_frequency_offset"),
    "apply_iq_perturbation": (".iq_perturbations", "apply_iq_perturbation"),
    "apply_multipath_channel": (".iq_perturbations", "apply_multipath_channel"),
    "complex_signal_power": (".iq_perturbations", "complex_signal_power"),
    "multipath_profile_taps": (".iq_perturbations", "multipath_profile_taps"),
    "normalize_complex_iq_window": (".iq_perturbations", "normalize_complex_iq_window"),
    "recording_iq_perturbation_seed": (".iq_perturbations", "recording_iq_perturbation_seed"),
    "sample_iq_perturbation_seed": (".iq_perturbations", "sample_iq_perturbation_seed"),
    "window_iq_perturbation_seed": (".iq_perturbations", "window_iq_perturbation_seed"),
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
