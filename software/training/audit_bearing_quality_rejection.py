"""Audit the frozen bearing quality gate on the fixed CWRU test split.

This module is evaluation-only. It never creates an optimizer, calls backward,
or modifies the checkpoint, source data, or desktop UI.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy
import sklearn
import torch
from sklearn.metrics import f1_score

from airwatch.analysis import load_bearing_quality_thresholds
from airwatch.analysis.bearing_robustness import (
    BearingPerturbation,
    apply_bearing_perturbation,
)
from airwatch.data import CWRUBearingDataset
from airwatch.inference import BearingPredictor
from training.common import (
    collect_environment,
    load_config,
    project_root_from_config,
    resolve_device,
    resolve_project_path,
)


HIGH_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_CONFIG = "training/configs/bearing_cnn_window_zscore.json"
DEFAULT_CHECKPOINT = (
    "artifacts/checkpoints/bearing/window_zscore/"
    "bearing_cnn_window_zscore_20260903_best.pt"
)
DEFAULT_CALIBRATION = (
    "artifacts/evidence/bearing/bearing_quality_calibration_20260903.json"
)
DEFAULT_EVIDENCE_DIR = "artifacts/evidence/bearing/quality_rejection_audit"
DEFAULT_RUN_NAME = "bearing_quality_rejection_test_audit_20260903_v1"


@dataclass(frozen=True)
class AuditCondition:
    condition_id: str
    kind: str
    value: float | None
    description: str


def audit_conditions() -> tuple[AuditCondition, ...]:
    """Return the fixed, non-tunable audit matrix."""
    return (
        AuditCondition("clean", "clean", None, "Unmodified test window"),
        AuditCondition("near_constant", "near_constant", 0.0, "All-zero window"),
        AuditCondition("nan", "nan", None, "One sample replaced with NaN"),
        AuditCondition("inf", "inf", None, "One sample replaced with positive infinity"),
        AuditCondition("clipping", "clipping", 0.125, "First 12.5% forced to extrema"),
        AuditCondition("awgn_20db", "awgn_snr_db", 20.0, "AWGN at 20 dB SNR"),
        AuditCondition("awgn_10db", "awgn_snr_db", 10.0, "AWGN at 10 dB SNR"),
        AuditCondition("awgn_5db", "awgn_snr_db", 5.0, "AWGN at 5 dB SNR"),
        AuditCondition("awgn_0db", "awgn_snr_db", 0.0, "AWGN at 0 dB SNR"),
        AuditCondition("awgn_minus5db", "awgn_snr_db", -5.0, "AWGN at -5 dB SNR"),
    )


def apply_audit_condition(
    window: object,
    condition: AuditCondition,
    *,
    seed: int,
    sample_index: int,
) -> np.ndarray:
    """Return a modified copy for one fixed audit condition."""
    source = np.asarray(window, dtype=np.float32)
    if source.ndim != 1 or source.size == 0:
        raise ValueError(f"audit window must be non-empty and one-dimensional: {source.shape}")
    changed = np.array(source, dtype=np.float32, copy=True)

    if condition.kind == "clean":
        return changed
    if condition.kind == "near_constant":
        changed.fill(float(condition.value or 0.0))
        return changed
    if condition.kind == "nan":
        changed[0] = np.nan
        return changed
    if condition.kind == "inf":
        changed[0] = np.inf
        return changed
    if condition.kind == "clipping":
        count = max(2, int(round(changed.size * float(condition.value or 0.125))))
        half = count // 2
        changed[:half] = float(np.nanmin(source))
        changed[half:count] = float(np.nanmax(source))
        return changed
    if condition.kind == "awgn_snr_db":
        perturbation = BearingPerturbation(
            condition_id=condition.condition_id,
            kind="awgn_snr_db",
            value=condition.value,
            description=condition.description,
        )
        tensor = torch.from_numpy(changed).unsqueeze(0)
        perturbed = apply_bearing_perturbation(
            tensor,
            perturbation,
            seed=seed,
            sample_index=sample_index,
        )
        return np.array(perturbed.squeeze(0).cpu().numpy(), dtype=np.float32, copy=True)
    raise ValueError(f"unsupported audit condition kind: {condition.kind!r}")


def build_raw_test_dataset(
    config: Mapping[str, Any], project_root: Path
) -> CWRUBearingDataset:
    """Build the immutable audit dataset: fixed test split, no normalization."""
    dataset_config = config["dataset"]
    manifest_path = resolve_project_path(project_root, dataset_config["manifest"])
    label_map_path = resolve_project_path(project_root, dataset_config["label_map"])
    return CWRUBearingDataset(
        manifest_path,
        split="test",
        label_map_path=label_map_path,
        normalization="none",
        project_root=project_root,
        verify_window_counts=True,
    )


def validate_calibration_payload(payload: object) -> None:
    """Reject thresholds that were not frozen using validation data."""
    if not isinstance(payload, Mapping):
        raise ValueError("calibration payload must be a JSON object")
    if payload.get("calibration_split") != "validation":
        raise ValueError("quality thresholds must come from calibration_split='validation'")
    if not isinstance(payload.get("selected_thresholds"), Mapping):
        raise ValueError("selected_thresholds must be a JSON object")


def validate_quality_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Ensure a rejected signal can never carry a diagnosis or confidence."""
    for index, row in enumerate(rows):
        if not bool(row.get("accepted")) and (
            row.get("prediction") is not None or row.get("confidence") is not None
        ):
            raise ValueError(
                f"rejected row {index} must have prediction=None and confidence=None"
            )


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize_condition(
    rows: Sequence[Mapping[str, Any]],
    *,
    num_classes: int,
    high_confidence_threshold: float,
) -> dict[str, Any]:
    """Summarize quality-gate coverage separately from accepted accuracy."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not 0.0 <= high_confidence_threshold <= 1.0:
        raise ValueError("high_confidence_threshold must be between 0 and 1")

    total = len(rows)
    accepted_rows = [
        row
        for row in rows
        if bool(row.get("accepted")) and row.get("prediction") is not None
    ]
    rejected_rows = [row for row in rows if not bool(row.get("accepted"))]
    baseline_rows = [row for row in rows if bool(row.get("baseline_evaluable"))]

    accepted_correct = sum(
        int(row["prediction"]) == int(row["target"]) for row in accepted_rows
    )
    baseline_correct_rows = [
        row
        for row in baseline_rows
        if row.get("baseline_prediction") is not None
        and int(row["baseline_prediction"]) == int(row["target"])
    ]
    baseline_error_rows = [
        row
        for row in baseline_rows
        if row.get("baseline_prediction") is not None
        and int(row["baseline_prediction"]) != int(row["target"])
    ]
    accepted_error_rows = [
        row for row in accepted_rows if int(row["prediction"]) != int(row["target"])
    ]
    rejected_baseline_error_rows = [
        row for row in baseline_error_rows if not bool(row.get("accepted"))
    ]
    rejected_baseline_correct_rows = [
        row for row in baseline_correct_rows if not bool(row.get("accepted"))
    ]
    high_confidence_baseline_error_rows = [
        row
        for row in baseline_error_rows
        if row.get("baseline_confidence") is not None
        and float(row["baseline_confidence"]) >= high_confidence_threshold
    ]
    rejected_high_confidence_baseline_error_rows = [
        row
        for row in high_confidence_baseline_error_rows
        if not bool(row.get("accepted"))
    ]

    reason_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason_code") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    selective_macro_f1: float | None = None
    if accepted_rows:
        selective_macro_f1 = float(
            f1_score(
                [int(row["target"]) for row in accepted_rows],
                [int(row["prediction"]) for row in accepted_rows],
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )

    return {
        "total_windows": total,
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "coverage": _safe_rate(len(accepted_rows), total),
        "rejection_rate": _safe_rate(len(rejected_rows), total),
        "baseline_evaluable_count": len(baseline_rows),
        "baseline_accuracy": _safe_rate(len(baseline_correct_rows), len(baseline_rows)),
        "selective_accuracy": _safe_rate(accepted_correct, len(accepted_rows)),
        "selective_macro_f1": selective_macro_f1,
        "rejection_reason_counts": reason_counts,
        "baseline_error_count": len(baseline_error_rows),
        "baseline_errors_rejected_count": len(rejected_baseline_error_rows),
        "baseline_errors_rejected_rate": _safe_rate(
            len(rejected_baseline_error_rows), len(baseline_error_rows)
        ),
        "baseline_correct_count": len(baseline_correct_rows),
        "baseline_correct_rejected_count": len(rejected_baseline_correct_rows),
        "baseline_correct_rejected_rate": _safe_rate(
            len(rejected_baseline_correct_rows), len(baseline_correct_rows)
        ),
        "accepted_error_count": len(accepted_error_rows),
        "accepted_error_mean_confidence": _mean_or_none(
            [
                float(row["confidence"])
                for row in accepted_error_rows
                if row.get("confidence") is not None
            ]
        ),
        "rejected_baseline_error_mean_confidence": _mean_or_none(
            [
                float(row["baseline_confidence"])
                for row in rejected_baseline_error_rows
                if row.get("baseline_confidence") is not None
            ]
        ),
        "high_confidence_baseline_error_count": len(high_confidence_baseline_error_rows),
        "high_confidence_baseline_errors_rejected_count": len(
            rejected_high_confidence_baseline_error_rows
        ),
        "high_confidence_baseline_errors_rejected_rate": _safe_rate(
            len(rejected_high_confidence_baseline_error_rows),
            len(high_confidence_baseline_error_rows),
        ),
    }


def strict_json_dumps(payload: object) -> str:
    """Serialize evidence as standards-compliant JSON; NaN/Inf are forbidden."""
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


def output_paths(evidence_dir: Path, run_name: str) -> dict[str, Path]:
    """Return the three immutable evidence paths for one audit run."""
    if not run_name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in run_name
    ):
        raise ValueError("run_name must contain only letters, digits, underscore, or hyphen")
    return {
        "json": evidence_dir / f"{run_name}.json",
        "conditions_csv": evidence_dir / f"{run_name}_conditions.csv",
        "predictions_csv": evidence_dir / f"{run_name}_predictions.csv",
    }


def ensure_outputs_absent(paths: Mapping[str, Path]) -> None:
    """Fail before evaluation when any target evidence file already exists."""
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing evidence: {joined}")


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return "" if value is None else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def write_outputs_atomically(
    paths: Mapping[str, Path],
    json_payload: Mapping[str, Any],
    condition_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    replace_func: Callable[[object, object], None] = os.replace,
) -> None:
    """Write all evidence or none, never overwriting a previous audit."""
    ensure_outputs_absent(paths)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    temporary_paths: dict[str, Path] = {}
    moved_destinations: list[Path] = []
    try:
        for key, destination in paths.items():
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            os.close(fd)
            temporary_paths[key] = Path(temporary_name)

        temporary_paths["json"].write_text(
            strict_json_dumps(json_payload) + "\n", encoding="utf-8"
        )
        _write_csv(temporary_paths["conditions_csv"], condition_rows)
        _write_csv(temporary_paths["predictions_csv"], prediction_rows)

        ensure_outputs_absent(paths)
        for key in ("json", "conditions_csv", "predictions_csv"):
            destination = paths[key]
            replace_func(temporary_paths[key], destination)
            moved_destinations.append(destination)
    except Exception:
        for destination in moved_destinations:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary_path in temporary_paths.values():
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_records(paths: Sequence[Path], project_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted({item.resolve() for item in paths}, key=lambda item: str(item)):
        try:
            display_path = str(path.relative_to(project_root))
        except ValueError:
            display_path = str(path)
        records.append(
            {
                "path": display_path,
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _flatten_quality_prediction(
    quality_row: Mapping[str, Any],
) -> tuple[int | None, str | None, float | None]:
    prediction = quality_row.get("prediction")
    if not isinstance(prediction, Mapping):
        return None, None, None
    return (
        int(prediction["predicted_class"]),
        str(prediction["predicted_label"]),
        float(prediction["confidence"]),
    )


def _condition_csv_row(condition: AuditCondition, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {**asdict(condition), **dict(summary)}


def run_quality_rejection_audit(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    calibration_path: str | Path = DEFAULT_CALIBRATION,
    evidence_dir: str | Path = DEFAULT_EVIDENCE_DIR,
    run_name: str = DEFAULT_RUN_NAME,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run the fixed, evaluation-only audit and write immutable evidence."""
    config = load_config(config_path)
    project_root = project_root_from_config(config)
    config_path_resolved = Path(config["_config_path"]).resolve()
    checkpoint = resolve_project_path(project_root, checkpoint_path)
    calibration = resolve_project_path(project_root, calibration_path)
    evidence = resolve_project_path(project_root, evidence_dir)
    paths = output_paths(evidence, run_name)
    ensure_outputs_absent(paths)

    manifest = resolve_project_path(project_root, config["dataset"]["manifest"])
    label_map = resolve_project_path(project_root, config["dataset"]["label_map"])
    for required in (checkpoint, calibration, manifest, label_map):
        if not required.is_file():
            raise FileNotFoundError(f"required audit input does not exist: {required}")

    calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
    validate_calibration_payload(calibration_payload)
    thresholds = load_bearing_quality_thresholds(calibration)

    requested_device = device_name or str(config["training"].get("device", "auto"))
    device = resolve_device(requested_device)
    seed = int(config["training"]["seed"])
    num_classes = int(config["dataset"]["num_classes"])

    checkpoint_sha_before = _sha256(checkpoint)
    dataset = build_raw_test_dataset(config, project_root)
    if dataset.window_size != int(config["dataset"]["window_size"]):
        raise ValueError("test dataset window size does not match frozen config")
    if dataset.step != int(config["dataset"]["step"]):
        raise ValueError("test dataset step does not match frozen config")
    if dataset.num_classes != num_classes:
        raise ValueError("test dataset class count does not match frozen config")

    source_paths = [dataset.sample_metadata(index).path for index in range(len(dataset))]
    protected_inputs = [config_path_resolved, checkpoint, calibration, manifest, label_map]
    protected_inputs.extend(sorted(set(source_paths), key=lambda item: str(item)))
    inputs_before = _file_records(protected_inputs, project_root)
    before_hashes = {row["absolute_path"]: row["sha256"] for row in inputs_before}

    predictor = BearingPredictor(checkpoint, device=device, label_map_path=label_map)
    label_by_index = {int(index): str(name) for index, name in dataset.label_map.items()}

    raw_windows: list[np.ndarray] = []
    targets: list[int] = []
    metadata_rows: list[Any] = []
    for index in range(len(dataset)):
        tensor, target = dataset[index]
        raw_windows.append(np.array(tensor.squeeze(0).numpy(), dtype=np.float32, copy=True))
        targets.append(int(target))
        metadata_rows.append(dataset.sample_metadata(index))

    prediction_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    baseline_kinds = {"clean", "clipping", "awgn_snr_db"}

    for condition in audit_conditions():
        print(f"[audit] condition={condition.condition_id} windows={len(raw_windows)}", flush=True)
        changed_windows = [
            apply_audit_condition(
                window,
                condition,
                seed=seed,
                sample_index=index,
            )
            for index, window in enumerate(raw_windows)
        ]
        signal = np.concatenate(changed_windows).astype(np.float32, copy=False)
        quality_predictions = predictor.predict_signal_with_quality(
            signal, thresholds=thresholds
        )
        if len(quality_predictions) != len(raw_windows):
            raise RuntimeError("quality prediction count does not match test window count")

        baseline_evaluable = condition.kind in baseline_kinds
        baseline_predictions: list[Mapping[str, Any] | None]
        if baseline_evaluable:
            predicted = predictor.predict_signal(signal)
            if len(predicted) != len(raw_windows):
                raise RuntimeError("baseline prediction count does not match test window count")
            baseline_predictions = list(predicted)
        else:
            baseline_predictions = [None] * len(raw_windows)

        current_rows: list[dict[str, Any]] = []
        for sample_index, (quality, baseline) in enumerate(
            zip(quality_predictions, baseline_predictions)
        ):
            predicted_class, predicted_label, confidence = _flatten_quality_prediction(quality)
            features = quality.get("quality_features")
            if not isinstance(features, Mapping):
                features = {}
            metadata = metadata_rows[sample_index]
            baseline_prediction = (
                int(baseline["predicted_class"]) if isinstance(baseline, Mapping) else None
            )
            baseline_label = (
                str(baseline["predicted_label"]) if isinstance(baseline, Mapping) else None
            )
            baseline_confidence = (
                float(baseline["confidence"]) if isinstance(baseline, Mapping) else None
            )
            row = {
                "condition_id": condition.condition_id,
                "sample_index": sample_index,
                "target": targets[sample_index],
                "target_label": label_by_index[targets[sample_index]],
                "accepted": bool(quality["accepted"]),
                "reason_code": str(quality["reason_code"]),
                "quality_score": float(quality["quality_score"]),
                "quality_std": features.get("standard_deviation"),
                "quality_rms": features.get("rms"),
                "quality_peak_to_peak": features.get("peak_to_peak"),
                "quality_clipping_ratio": features.get("clipping_ratio"),
                "quality_spectral_flatness": features.get("spectral_flatness"),
                "prediction": predicted_class,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "baseline_evaluable": baseline_evaluable,
                "baseline_prediction": baseline_prediction,
                "baseline_predicted_label": baseline_label,
                "baseline_confidence": baseline_confidence,
                "original_filename": metadata.original_filename,
                "source_path": str(metadata.path.relative_to(project_root)),
                "source_window_index": metadata.window_index,
                "sensor_key": metadata.sensor_key,
                "split_group": metadata.split_group,
            }
            current_rows.append(row)

        validate_quality_prediction_rows(current_rows)
        summary = summarize_condition(
            current_rows,
            num_classes=num_classes,
            high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
        )
        summaries[condition.condition_id] = summary
        condition_rows.append(_condition_csv_row(condition, summary))
        prediction_rows.extend(current_rows)

    aggregate_summary = summarize_condition(
        prediction_rows,
        num_classes=num_classes,
        high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
    )

    inputs_after = _file_records(protected_inputs, project_root)
    after_hashes = {row["absolute_path"]: row["sha256"] for row in inputs_after}
    changed_inputs = sorted(
        path for path, before_hash in before_hashes.items() if after_hashes.get(path) != before_hash
    )
    if changed_inputs:
        raise RuntimeError(f"protected audit inputs changed during evaluation: {changed_inputs}")
    checkpoint_sha_after = _sha256(checkpoint)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("checkpoint SHA-256 changed during evaluation")

    environment = collect_environment(device)
    environment.update(
        {
            "executable": os.path.abspath(os.sys.executable),
            "platform": platform.platform(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        }
    )

    test_recordings: list[dict[str, Any]] = []
    seen_recordings: set[Path] = set()
    for metadata in metadata_rows:
        if metadata.path in seen_recordings:
            continue
        seen_recordings.add(metadata.path)
        count = sum(item.path == metadata.path for item in metadata_rows)
        test_recordings.append(
            {
                "original_filename": metadata.original_filename,
                "class_name": metadata.class_name,
                "label_index": metadata.label_index,
                "sensor_key": metadata.sensor_key,
                "window_count": count,
                "path": str(metadata.path.relative_to(project_root)),
                "sha256": before_hashes[str(metadata.path.resolve())],
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_name": run_name,
        "audit_date": date.today().isoformat(),
        "purpose": "Frozen quality-gate audit on the fixed CWRU test split",
        "evaluation_contract": {
            "split": "test",
            "normalization_before_quality_gate": "none",
            "model_preprocessing": predictor.normalization,
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
            "seed": seed,
            "test_used_for_threshold_selection": False,
        },
        "safety_declarations": {
            "training_performed": False,
            "optimizer_created": False,
            "backward_called": False,
            "checkpoint_modified": False,
            "ui_modified": False,
            "source_data_modified": False,
        },
        "environment": environment,
        "inputs": {
            "config_path": str(config_path_resolved),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "calibration_path": str(calibration),
            "calibration_split": calibration_payload["calibration_split"],
            "selected_thresholds": dict(calibration_payload["selected_thresholds"]),
            "manifest_path": str(manifest),
            "label_map_path": str(label_map),
            "protected_files_before": inputs_before,
            "protected_files_after": inputs_after,
            "protected_files_changed": changed_inputs,
        },
        "test_data": {
            "dataset": "CWRU bearing test-rig data",
            "window_count": len(dataset),
            "recording_count": len(test_recordings),
            "recordings": test_recordings,
        },
        "conditions": [
            {**asdict(condition), "summary": summaries[condition.condition_id]}
            for condition in audit_conditions()
        ],
        "overall_summary": aggregate_summary,
        "focus_results": {
            condition_id: summaries[condition_id]
            for condition_id in ("awgn_0db", "awgn_minus5db")
        },
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
        "limitations": [
            "CWRU is laboratory bearing test-rig data, not industrial field data.",
            "AWGN and clipping are offline mathematical perturbations.",
            "The test split was not used to select quality thresholds.",
            "Unknown fault classes were not tested.",
            "Cross-dataset generalization was not tested.",
        ],
    }

    write_outputs_atomically(paths, payload, condition_rows, prediction_rows)
    print(f"[audit] evidence_json={paths['json']}", flush=True)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit frozen bearing signal-quality rejection on the test split."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    run_quality_rejection_audit(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        calibration_path=args.calibration,
        evidence_dir=args.evidence_dir,
        run_name=args.run_name,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
