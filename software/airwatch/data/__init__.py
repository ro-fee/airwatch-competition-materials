"""Lazy public exports for data modules.

Desktop code imports concrete runtime submodules. The lazy package Interface
keeps source compatibility for training and tests without making PyInstaller
pull every offline audit and dataset implementation into the desktop build.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BearingDatasetError": (".bearing_dataset", "BearingDatasetError"),
    "CWRUBearingDataset": (".bearing_dataset", "CWRUBearingDataset"),
    "AmbiguousSignalError": (".cwru", "AmbiguousSignalError"),
    "CWRUDataError": (".cwru", "CWRUDataError"),
    "SignalRecord": (".cwru", "SignalRecord"),
    "load_cwru_signal": (".cwru", "load_cwru_signal"),
    "normalize_windows": (".cwru", "normalize_windows"),
    "slice_signal": (".cwru", "slice_signal"),
    "AuditIssue": (".cwru_audit", "AuditIssue"),
    "CWRUManifestAuditError": (".cwru_audit", "CWRUManifestAuditError"),
    "audit_manifest": (".cwru_audit", "audit_manifest"),
    "write_audit_json": (".cwru_audit", "write_audit_json"),
    "CompatibilityCheckError": (".external_compat", "CompatibilityCheckError"),
    "ExternalReferenceSpec": (".external_compat", "ExternalReferenceSpec"),
    "build_cnn_transformer_compatibility_report": (
        ".external_compat",
        "build_cnn_transformer_compatibility_report",
    ),
    "write_report": (".external_compat", "write_report"),
    "RecognitionSnapshot": (".recognition_input", "RecognitionSnapshot"),
    "KU_LEUVEN_KNOWN_LABELS": (".ku_leuven_dataset", "KU_LEUVEN_KNOWN_LABELS"),
    "KULeuvenDatasetError": (".ku_leuven_dataset", "KULeuvenDatasetError"),
    "KULeuvenMaterializedDataset": (".ku_leuven_dataset", "KULeuvenMaterializedDataset"),
    "KULeuvenWindowSample": (".ku_leuven_dataset", "KULeuvenWindowSample"),
    "DroneRFArchiveEntry": (".dronerf", "DroneRFArchiveEntry"),
    "DroneRFLabel": (".dronerf", "DroneRFLabel"),
    "DroneRFManifestError": (".dronerf", "DroneRFManifestError"),
    "DroneRFPackage": (".dronerf", "DroneRFPackage"),
    "DroneRFRecording": (".dronerf", "DroneRFRecording"),
    "DroneRFSplitAssignment": (".dronerf", "DroneRFSplitAssignment"),
    "assign_recording_splits": (".dronerf", "assign_recording_splits"),
    "audit_recording_splits": (".dronerf", "audit_recording_splits"),
    "build_recordings": (".dronerf", "build_recordings"),
    "decode_label": (".dronerf", "decode_label"),
    "discover_recordings": (".dronerf", "discover_recordings"),
    "parse_member_path": (".dronerf", "parse_member_path"),
    "parse_package_path": (".dronerf", "parse_package_path"),
    "read_recording_manifest": (".dronerf", "read_recording_manifest"),
    "read_split_manifest": (".dronerf", "read_split_manifest"),
    "write_recording_manifest": (".dronerf", "write_recording_manifest"),
    "write_split_manifest": (".dronerf", "write_split_manifest"),
    "DRONE_TYPE_LABELS": (".dronerf_dataset", "DRONE_TYPE_LABELS"),
    "DroneRFDatasetError": (".dronerf_dataset", "DroneRFDatasetError"),
    "DroneRFWindowDataset": (".dronerf_dataset", "DroneRFWindowDataset"),
    "DroneRFWindowSample": (".dronerf_dataset", "DroneRFWindowSample"),
    "DroneRFWindowResult": (".dronerf_windows", "DroneRFWindowResult"),
    "DroneRFWindowSpec": (".dronerf_windows", "DroneRFWindowSpec"),
    "DroneRFWindowingError": (".dronerf_windows", "DroneRFWindowingError"),
    "extract_recording_windows": (".dronerf_windows", "extract_recording_windows"),
    "sample_csv_member": (".dronerf_windows", "sample_csv_member"),
    "stratified_window_offsets": (".dronerf_windows", "stratified_window_offsets"),
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
