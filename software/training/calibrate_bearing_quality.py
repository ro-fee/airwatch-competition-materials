"""Calibrate the bearing signal-quality gate using validation windows only.

This command never loads a classifier, trains a model, reads the test split, or
modifies source recordings.  It derives quality thresholds from the audited
CWRU validation split and deterministic in-memory AWGN perturbations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

from airwatch.analysis import (
    BearingPerturbation,
    BearingQualityThresholds,
    apply_bearing_perturbation,
    assess_bearing_window,
)
from airwatch.data import CWRUBearingDataset


CALIBRATION_SEED = 20260903
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _PROJECT_ROOT / "datasets/bearing/cwru/split-manifest.csv"
_DEFAULT_LABEL_MAP = _PROJECT_ROOT / "datasets/bearing/cwru/label-map.json"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "artifacts/evidence/bearing"
_DEFAULT_JSON = _DEFAULT_OUTPUT_DIR / "bearing_quality_calibration_20260903.json"
_DEFAULT_CSV = _DEFAULT_OUTPUT_DIR / "bearing_quality_calibration_20260903.csv"

_NEAR_CONSTANT_CANDIDATES = (
    float(np.finfo(np.float32).eps),
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
)
_CLIPPING_CANDIDATES = (0.002, 0.003, 0.005, 0.01, 0.02, 0.04, 0.08)
_FLATNESS_CANDIDATES = tuple(round(value, 3) for value in np.arange(0.05, 0.6001, 0.005))

_RETAIN_MINIMUMS = {
    "clean": 0.99,
    "awgn_20db": 0.99,
    "awgn_10db": 0.95,
}
_SEVERE_NOISE_CONDITIONS = ("awgn_0db", "awgn_minus5db")
_CONDITION_ORDER = (
    "clean",
    "awgn_20db",
    "awgn_10db",
    "awgn_5db",
    "awgn_0db",
    "awgn_minus5db",
)
_CONDITION_ROLES = {
    "clean": "retain",
    "awgn_20db": "retain",
    "awgn_10db": "retain",
    "awgn_5db": "gray_report_only",
    "awgn_0db": "severe_noise_reject",
    "awgn_minus5db": "severe_noise_reject",
}


class CalibrationError(ValueError):
    """Raised when validation-only quality calibration cannot be completed safely."""


def _finite_float(record: Mapping[str, Any], name: str) -> float:
    try:
        value = float(record[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"record has invalid {name!r}: {record!r}") from exc
    if not np.isfinite(value):
        raise CalibrationError(f"record has non-finite {name!r}: {record!r}")
    return value


def _group_feature_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, float]]]:
    grouped = {condition_id: [] for condition_id in _CONDITION_ORDER}
    for record in records:
        condition_id = str(record.get("condition_id", ""))
        if condition_id not in grouped:
            raise CalibrationError(f"unsupported condition_id: {condition_id!r}")
        grouped[condition_id].append(
            {
                "standard_deviation": _finite_float(record, "standard_deviation"),
                "clipping_ratio": _finite_float(record, "clipping_ratio"),
                "spectral_flatness": _finite_float(record, "spectral_flatness"),
            }
        )
    missing = [condition_id for condition_id, rows in grouped.items() if not rows]
    if missing:
        raise CalibrationError(f"missing calibration conditions: {missing}")
    return grouped


def _accepted_ratio(
    rows: list[dict[str, float]],
    *,
    near_constant_std: float,
    max_clipping_ratio: float,
    max_spectral_flatness: float,
) -> float:
    accepted = sum(
        row["standard_deviation"] > near_constant_std
        and row["clipping_ratio"] <= max_clipping_ratio
        and row["spectral_flatness"] <= max_spectral_flatness
        for row in rows
    )
    return float(accepted / len(rows))


def _select_near_constant_threshold(clean: list[dict[str, float]]) -> float:
    valid = [
        candidate
        for candidate in _NEAR_CONSTANT_CANDIDATES
        if all(row["standard_deviation"] > candidate for row in clean)
    ]
    if not valid:
        raise CalibrationError("no near-constant candidate preserves all clean validation windows")
    return float(max(valid))


def _select_clipping_threshold(clean: list[dict[str, float]]) -> float:
    valid = [
        candidate
        for candidate in _CLIPPING_CANDIDATES
        if all(row["clipping_ratio"] <= candidate for row in clean)
    ]
    if not valid:
        raise CalibrationError("no clipping candidate preserves all clean validation windows")
    return float(min(valid))


def _select_flatness_threshold(
    grouped: dict[str, list[dict[str, float]]],
    *,
    near_constant_std: float,
    max_clipping_ratio: float,
) -> tuple[float, float]:
    feasible: list[tuple[float, float]] = []
    for candidate in _FLATNESS_CANDIDATES:
        retain_ratios = {
            condition_id: _accepted_ratio(
                grouped[condition_id],
                near_constant_std=near_constant_std,
                max_clipping_ratio=max_clipping_ratio,
                max_spectral_flatness=candidate,
            )
            for condition_id in _RETAIN_MINIMUMS
        }
        if not all(
            retain_ratios[condition_id] >= minimum
            for condition_id, minimum in _RETAIN_MINIMUMS.items()
        ):
            continue
        severe_rejection = float(
            np.mean(
                [
                    1.0
                    - _accepted_ratio(
                        grouped[condition_id],
                        near_constant_std=near_constant_std,
                        max_clipping_ratio=max_clipping_ratio,
                        max_spectral_flatness=candidate,
                    )
                    for condition_id in _SEVERE_NOISE_CONDITIONS
                ]
            )
        )
        feasible.append((candidate, severe_rejection))

    if not feasible:
        raise CalibrationError("no spectral-flatness candidate satisfies retention constraints")

    best_rejection = max(rejection for _, rejection in feasible)
    tied = [
        candidate
        for candidate, rejection in feasible
        if np.isclose(rejection, best_rejection, rtol=0.0, atol=1e-12)
    ]
    # A larger threshold is less aggressive.  Use it when severe-noise rejection
    # is tied, reducing avoidable false rejection on retained/gray conditions.
    return float(max(tied)), float(best_rejection)


def _distribution(rows: list[dict[str, float]], feature: str) -> dict[str, float]:
    values = np.asarray([row[feature] for row in rows], dtype=np.float64)
    return {
        "minimum": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def calibrate_quality_features(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration_split: str,
) -> dict[str, Any]:
    """Select deterministic gate thresholds from precomputed validation features."""

    if calibration_split != "validation":
        raise CalibrationError("quality thresholds may only be calibrated on validation data")

    grouped = _group_feature_records(records)
    near_constant_std = _select_near_constant_threshold(grouped["clean"])
    max_clipping_ratio = _select_clipping_threshold(grouped["clean"])
    max_spectral_flatness, severe_rejection = _select_flatness_threshold(
        grouped,
        near_constant_std=near_constant_std,
        max_clipping_ratio=max_clipping_ratio,
    )
    thresholds = BearingQualityThresholds(
        near_constant_std=near_constant_std,
        max_clipping_ratio=max_clipping_ratio,
        max_spectral_flatness=max_spectral_flatness,
    )

    condition_rows: list[dict[str, Any]] = []
    for condition_id in _CONDITION_ORDER:
        rows = grouped[condition_id]
        accepted_ratio = _accepted_ratio(
            rows,
            near_constant_std=thresholds.near_constant_std,
            max_clipping_ratio=thresholds.max_clipping_ratio,
            max_spectral_flatness=thresholds.max_spectral_flatness,
        )
        condition_rows.append(
            {
                "condition_id": condition_id,
                "role": _CONDITION_ROLES[condition_id],
                "used_for_selection": condition_id != "awgn_5db",
                "window_count": int(len(rows)),
                "accepted_ratio": accepted_ratio,
                "rejected_ratio": float(1.0 - accepted_ratio),
                "feature_distributions": {
                    feature: _distribution(rows, feature)
                    for feature in (
                        "standard_deviation",
                        "clipping_ratio",
                        "spectral_flatness",
                    )
                },
            }
        )

    return {
        "schema_version": 1,
        "calibration_split": calibration_split,
        "seed": CALIBRATION_SEED,
        "selected_thresholds": asdict(thresholds),
        "selection_rule": {
            "near_constant": "largest candidate rejecting zero clean validation windows",
            "clipping": "smallest candidate rejecting zero clean validation windows",
            "spectral_flatness": (
                "maximize mean rejection on severe noise while satisfying retention; "
                "choose the larger threshold on ties"
            ),
            "retain_minimum_accepted_ratio": dict(_RETAIN_MINIMUMS),
            "severe_noise_conditions": list(_SEVERE_NOISE_CONDITIONS),
            "gray_report_only_conditions": ["awgn_5db"],
            "selected_mean_severe_noise_rejection_ratio": severe_rejection,
        },
        "candidate_thresholds": {
            "near_constant_std": list(_NEAR_CONSTANT_CANDIDATES),
            "max_clipping_ratio": list(_CLIPPING_CANDIDATES),
            "max_spectral_flatness": list(_FLATNESS_CANDIDATES),
        },
        "conditions": condition_rows,
        "limitations": [
            "Thresholds are calibrated only on the CWRU validation split.",
            "Noise cases are deterministic synthetic AWGN, not field sensor failures.",
            "The 5 dB condition is report-only and does not influence selection.",
            "No classifier or classification metric is used in this calibration.",
        ],
    }


def _csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = payload["selected_thresholds"]
    rows: list[dict[str, Any]] = []
    for condition in payload["conditions"]:
        distributions = condition["feature_distributions"]
        rows.append(
            {
                "calibration_split": payload["calibration_split"],
                "seed": payload["seed"],
                "condition_id": condition["condition_id"],
                "role": condition["role"],
                "used_for_selection": condition["used_for_selection"],
                "window_count": condition["window_count"],
                "accepted_ratio": condition["accepted_ratio"],
                "rejected_ratio": condition["rejected_ratio"],
                "near_constant_std": thresholds["near_constant_std"],
                "max_clipping_ratio": thresholds["max_clipping_ratio"],
                "max_spectral_flatness": thresholds["max_spectral_flatness"],
                "spectral_flatness_min": distributions["spectral_flatness"]["minimum"],
                "spectral_flatness_p95": distributions["spectral_flatness"]["p95"],
                "spectral_flatness_max": distributions["spectral_flatness"]["maximum"],
            }
        )
    return rows


def _write_temp_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def write_calibration_outputs(
    payload: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Write JSON and CSV evidence without overwriting any existing result."""

    json_path = Path(json_path)
    csv_path = Path(csv_path)
    existing = [str(path) for path in (json_path, csv_path) if path.exists()]
    if existing:
        raise CalibrationError(f"refusing to overwrite existing output: {existing}")

    json_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    csv_rows = _csv_rows(payload)
    if not csv_rows:
        raise CalibrationError("calibration payload has no condition rows")
    fieldnames = list(csv_rows[0])
    from io import StringIO

    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)

    json_temp: Path | None = None
    csv_temp: Path | None = None
    json_created = False
    try:
        json_temp = _write_temp_text(json_path, json_text)
        csv_temp = _write_temp_text(csv_path, csv_buffer.getvalue())
        # Recheck after preparing both files.  os.replace is then used only for
        # the destinations confirmed absent in this single-process workflow.
        existing = [str(path) for path in (json_path, csv_path) if path.exists()]
        if existing:
            raise CalibrationError(f"refusing to overwrite existing output: {existing}")
        os.replace(json_temp, json_path)
        json_created = True
        json_temp = None
        os.replace(csv_temp, csv_path)
        csv_temp = None
    except Exception:
        if json_created:
            json_path.unlink(missing_ok=True)
        raise
    finally:
        if json_temp is not None:
            json_temp.unlink(missing_ok=True)
        if csv_temp is not None:
            csv_temp.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _conditions() -> tuple[BearingPerturbation, ...]:
    return (
        BearingPerturbation("clean", "clean", None, "Unmodified validation window"),
        BearingPerturbation("awgn_20db", "awgn_snr_db", 20.0, "AWGN at 20 dB SNR"),
        BearingPerturbation("awgn_10db", "awgn_snr_db", 10.0, "AWGN at 10 dB SNR"),
        BearingPerturbation("awgn_5db", "awgn_snr_db", 5.0, "AWGN at 5 dB SNR; report only"),
        BearingPerturbation("awgn_0db", "awgn_snr_db", 0.0, "AWGN at 0 dB SNR"),
        BearingPerturbation("awgn_minus5db", "awgn_snr_db", -5.0, "AWGN at -5 dB SNR"),
    )


def collect_validation_quality_features(
    *,
    manifest_path: Path,
    label_map_path: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only validation windows and extract raw quality features."""

    dataset = CWRUBearingDataset(
        manifest_path,
        split="validation",
        label_map_path=label_map_path,
        normalization="none",
        project_root=project_root,
    )
    records: list[dict[str, Any]] = []
    conditions = _conditions()
    for sample_index in range(len(dataset)):
        window, _ = dataset[sample_index]
        metadata = dataset.sample_metadata(sample_index)
        for condition in conditions:
            perturbed = apply_bearing_perturbation(
                window,
                condition,
                seed=CALIBRATION_SEED,
                sample_index=sample_index,
            )
            result = assess_bearing_window(perturbed.squeeze(0).cpu().numpy())
            if not result.features:
                raise CalibrationError(
                    f"cannot extract quality features for sample {sample_index}, "
                    f"condition {condition.condition_id}"
                )
            records.append(
                {
                    "condition_id": condition.condition_id,
                    "sample_index": int(sample_index),
                    "original_filename": metadata.original_filename,
                    "window_index": int(metadata.window_index),
                    **{name: float(value) for name, value in result.features.items()},
                }
            )

    source = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "label_map_path": str(label_map_path.resolve()),
        "validation_recording_count": int(len(dataset.source_files)),
        "validation_window_count": int(len(dataset)),
        "window_size": int(dataset.window_size),
        "step": int(dataset.step),
        "normalization_during_calibration": "none",
        "conditions": [asdict(condition) for condition in conditions],
    }
    return records, source


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--label-map", type=Path, default=_DEFAULT_LABEL_MAP)
    parser.add_argument("--json-output", type=Path, default=_DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=_DEFAULT_CSV)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.json_output.exists() or args.csv_output.exists():
        raise CalibrationError(
            "refusing to overwrite existing calibration output; choose new output paths"
        )
    records, source = collect_validation_quality_features(
        manifest_path=args.manifest,
        label_map_path=args.label_map,
        project_root=_PROJECT_ROOT,
    )
    payload = calibrate_quality_features(records, calibration_split="validation")
    payload["created_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    payload["source"] = source
    write_calibration_outputs(
        payload,
        json_path=args.json_output,
        csv_path=args.csv_output,
    )
    print(
        json.dumps(
            {
                "json_output": str(args.json_output.resolve()),
                "csv_output": str(args.csv_output.resolve()),
                "selected_thresholds": payload["selected_thresholds"],
                "validation_window_count": source["validation_window_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
