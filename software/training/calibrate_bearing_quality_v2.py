"""Calibrate the three-level bearing signal-quality gate on validation data only.

This command reads the audited CWRU validation split, creates deterministic
in-memory AWGN pressure-test copies, and selects warning/rejection thresholds.
It never loads or trains a classifier, reads the test split, modifies Qt files,
or changes source recordings/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

from airwatch.analysis import (
    BearingPerturbation,
    BearingQualityThresholdsV2,
    apply_bearing_perturbation,
    assess_bearing_window_v2,
)
from airwatch.data import CWRUBearingDataset


CALIBRATION_SEED = 20260903
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _PROJECT_ROOT / "datasets/bearing/cwru/split-manifest.csv"
_DEFAULT_LABEL_MAP = _PROJECT_ROOT / "datasets/bearing/cwru/label-map.json"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "artifacts/evidence/bearing"
_DEFAULT_JSON = _DEFAULT_OUTPUT_DIR / "bearing_quality_calibration_20260903_v2.json"
_DEFAULT_CSV = _DEFAULT_OUTPUT_DIR / "bearing_quality_calibration_20260903_v2.csv"

_NEAR_CONSTANT_CANDIDATES = (
    float(np.finfo(np.float32).eps),
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
)
_CLIPPING_CANDIDATES = (0.002, 0.003, 0.005, 0.01, 0.02, 0.04, 0.08)
_FLATNESS_CANDIDATES = tuple(
    round(float(value), 3) for value in np.arange(0.05, 0.6001, 0.005)
)
_CONDITION_ORDER = (
    "clean",
    "awgn_20db",
    "awgn_10db",
    "awgn_5db",
    "awgn_0db",
    "awgn_minus5db",
)
_CONDITION_ROLES = {
    "clean": "accepted_reference",
    "awgn_20db": "accepted_reference",
    "awgn_10db": "caution_target",
    "awgn_5db": "caution_target_and_reject_retention",
    "awgn_0db": "transition_report_only",
    "awgn_minus5db": "severe_noise_reject_reference",
}
_WARNING_ACCEPTED_MINIMUMS = {"clean": 0.99, "awgn_20db": 0.99}
_REJECT_NONREJECTED_MINIMUMS = {
    "clean": 0.99,
    "awgn_20db": 0.99,
    "awgn_10db": 0.95,
    "awgn_5db": 0.95,
}
_WARNING_TARGETS = ("awgn_10db", "awgn_5db")
_REJECT_TARGETS = ("awgn_minus5db",)


class CalibrationV2Error(ValueError):
    """Raised when validation-only V2 calibration cannot be completed safely."""


def _finite_float(record: Mapping[str, Any], name: str) -> float:
    try:
        value = float(record[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationV2Error(f"record has invalid {name!r}: {record!r}") from exc
    if not np.isfinite(value):
        raise CalibrationV2Error(f"record has non-finite {name!r}: {record!r}")
    return value


def _group_feature_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, float]]]:
    grouped = {condition_id: [] for condition_id in _CONDITION_ORDER}
    for record in records:
        condition_id = str(record.get("condition_id", ""))
        if condition_id not in grouped:
            raise CalibrationV2Error(f"unsupported condition_id: {condition_id!r}")
        grouped[condition_id].append(
            {
                "standard_deviation": _finite_float(record, "standard_deviation"),
                "clipping_ratio": _finite_float(record, "clipping_ratio"),
                "spectral_flatness": _finite_float(record, "spectral_flatness"),
            }
        )
    missing = [condition_id for condition_id, rows in grouped.items() if not rows]
    if missing:
        raise CalibrationV2Error(f"missing calibration conditions: {missing}")
    return grouped


def _status(row: Mapping[str, float], thresholds: BearingQualityThresholdsV2) -> str:
    if (
        row["standard_deviation"] <= thresholds.near_constant_std
        or row["clipping_ratio"] > thresholds.reject_clipping_ratio
        or row["spectral_flatness"] > thresholds.reject_spectral_flatness
    ):
        return "rejected"
    if (
        row["clipping_ratio"] > thresholds.warning_clipping_ratio
        or row["spectral_flatness"] > thresholds.warning_spectral_flatness
    ):
        return "caution"
    return "accepted"


def _condition_statistics(
    condition_id: str,
    rows: list[dict[str, float]],
    thresholds: BearingQualityThresholdsV2,
) -> dict[str, Any]:
    counts = {"accepted": 0, "caution": 0, "rejected": 0}
    for row in rows:
        counts[_status(row, thresholds)] += 1
    total = len(rows)
    return {
        "condition_id": condition_id,
        "window_count": int(total),
        "status_counts": counts,
        "status_ratios": {
            name: float(count / total) for name, count in counts.items()
        },
    }


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


def _select_near_constant(clean: list[dict[str, float]]) -> float:
    feasible = [
        candidate
        for candidate in _NEAR_CONSTANT_CANDIDATES
        if all(row["standard_deviation"] > candidate for row in clean)
    ]
    if not feasible:
        raise CalibrationV2Error(
            "no near-constant candidate preserves all clean validation windows"
        )
    return float(max(feasible))


def _select_reject_clipping(clean: list[dict[str, float]]) -> float:
    feasible = [
        candidate
        for candidate in _CLIPPING_CANDIDATES
        if all(row["clipping_ratio"] <= candidate for row in clean)
    ]
    if not feasible:
        raise CalibrationV2Error(
            "no clipping-rejection candidate preserves all clean validation windows"
        )
    return float(min(feasible))


def _select_warning_clipping(
    grouped: Mapping[str, list[dict[str, float]]], *, reject_clipping_ratio: float
) -> float:
    feasible = []
    for candidate in _CLIPPING_CANDIDATES:
        if candidate > reject_clipping_ratio:
            continue
        if all(
            sum(row["clipping_ratio"] <= candidate for row in grouped[condition_id])
            / len(grouped[condition_id])
            >= minimum
            for condition_id, minimum in _WARNING_ACCEPTED_MINIMUMS.items()
        ):
            feasible.append(candidate)
    if not feasible:
        # Equality disables unsupported clipping warnings while keeping rejection valid.
        return float(reject_clipping_ratio)
    return float(max(feasible))


def _nonrejected_ratio(
    rows: list[dict[str, float]],
    *,
    near_constant_std: float,
    reject_clipping_ratio: float,
    reject_spectral_flatness: float,
) -> float:
    return float(
        sum(
            row["standard_deviation"] > near_constant_std
            and row["clipping_ratio"] <= reject_clipping_ratio
            and row["spectral_flatness"] <= reject_spectral_flatness
            for row in rows
        )
        / len(rows)
    )


def _select_reject_flatness(
    grouped: Mapping[str, list[dict[str, float]]],
    *,
    near_constant_std: float,
    reject_clipping_ratio: float,
) -> tuple[float, float]:
    feasible: list[tuple[float, float]] = []
    for candidate in _FLATNESS_CANDIDATES:
        retained = all(
            _nonrejected_ratio(
                grouped[condition_id],
                near_constant_std=near_constant_std,
                reject_clipping_ratio=reject_clipping_ratio,
                reject_spectral_flatness=candidate,
            )
            >= minimum
            for condition_id, minimum in _REJECT_NONREJECTED_MINIMUMS.items()
        )
        if not retained:
            continue
        severe_rejection = float(
            np.mean(
                [
                    1.0
                    - _nonrejected_ratio(
                        grouped[condition_id],
                        near_constant_std=near_constant_std,
                        reject_clipping_ratio=reject_clipping_ratio,
                        reject_spectral_flatness=candidate,
                    )
                    for condition_id in _REJECT_TARGETS
                ]
            )
        )
        feasible.append((candidate, severe_rejection))
    if not feasible:
        raise CalibrationV2Error(
            "no spectral-flatness rejection candidate satisfies retention constraints"
        )
    best = max(score for _, score in feasible)
    tied = [
        candidate
        for candidate, score in feasible
        if np.isclose(score, best, rtol=0.0, atol=1e-12)
    ]
    return float(max(tied)), float(best)


def _select_warning_flatness(
    grouped: Mapping[str, list[dict[str, float]]],
    *,
    near_constant_std: float,
    warning_clipping_ratio: float,
    reject_clipping_ratio: float,
    reject_spectral_flatness: float,
) -> tuple[float, float]:
    feasible: list[tuple[float, float]] = []
    for candidate in _FLATNESS_CANDIDATES:
        if candidate > reject_spectral_flatness:
            continue
        thresholds = BearingQualityThresholdsV2(
            near_constant_std=near_constant_std,
            warning_clipping_ratio=warning_clipping_ratio,
            reject_clipping_ratio=reject_clipping_ratio,
            warning_spectral_flatness=candidate,
            reject_spectral_flatness=reject_spectral_flatness,
        )
        reports = {
            condition_id: _condition_statistics(
                condition_id, grouped[condition_id], thresholds
            )
            for condition_id in _CONDITION_ORDER
        }
        retains_accepted = all(
            reports[condition_id]["status_ratios"]["accepted"] >= minimum
            for condition_id, minimum in _WARNING_ACCEPTED_MINIMUMS.items()
        )
        if not retains_accepted:
            continue
        mean_caution = float(
            np.mean(
                [
                    reports[condition_id]["status_ratios"]["caution"]
                    for condition_id in _WARNING_TARGETS
                ]
            )
        )
        feasible.append((candidate, mean_caution))
    if not feasible:
        raise CalibrationV2Error(
            "no spectral-flatness warning candidate satisfies clean/20 dB constraints"
        )
    best = max(score for _, score in feasible)
    tied = [
        candidate
        for candidate, score in feasible
        if np.isclose(score, best, rtol=0.0, atol=1e-12)
    ]
    return float(max(tied)), float(best)


def _candidate_values_for_field(
    field: str, selected: BearingQualityThresholdsV2
) -> list[float]:
    if field == "near_constant_std":
        return [float(value) for value in _NEAR_CONSTANT_CANDIDATES]
    if field == "warning_clipping_ratio":
        return [
            float(value)
            for value in _CLIPPING_CANDIDATES
            if value <= selected.reject_clipping_ratio
        ]
    if field == "reject_clipping_ratio":
        return [
            float(value)
            for value in _CLIPPING_CANDIDATES
            if value >= selected.warning_clipping_ratio
        ]
    if field == "warning_spectral_flatness":
        return [
            float(value)
            for value in _FLATNESS_CANDIDATES
            if value <= selected.reject_spectral_flatness
        ]
    if field == "reject_spectral_flatness":
        return [
            float(value)
            for value in _FLATNESS_CANDIDATES
            if value >= selected.warning_spectral_flatness
        ]
    raise AssertionError(field)


def _replace_threshold(
    selected: BearingQualityThresholdsV2, field: str, candidate: float
) -> BearingQualityThresholdsV2:
    values = asdict(selected)
    values[field] = float(candidate)
    return BearingQualityThresholdsV2(**values)


def _candidate_reason(field: str, selected: bool) -> str:
    if selected:
        return {
            "near_constant_std": "selected: largest candidate preserving every clean validation window",
            "warning_clipping_ratio": "selected: conservative warning limit supported by clean and 20 dB validation windows",
            "reject_clipping_ratio": "selected: smallest rejection limit preserving every clean validation window",
            "warning_spectral_flatness": "selected: maximizes caution coverage at 10/5 dB while retaining clean/20 dB as accepted; larger tied limit preferred",
            "reject_spectral_flatness": "selected: maximizes -5 dB rejection while retaining clean/20/10/5 dB; larger tied limit preferred",
        }[field]
    return "not selected: candidate statistics are reported for audit against the stated selection rule"


def _candidate_evaluations(
    grouped: Mapping[str, list[dict[str, float]]],
    selected: BearingQualityThresholdsV2,
) -> tuple[dict[str, list[float]], dict[str, list[dict[str, Any]]]]:
    fields = (
        "near_constant_std",
        "warning_clipping_ratio",
        "reject_clipping_ratio",
        "warning_spectral_flatness",
        "reject_spectral_flatness",
    )
    candidates: dict[str, list[float]] = {}
    evaluations: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        values = _candidate_values_for_field(field, selected)
        candidates[field] = values
        evaluations[field] = []
        for candidate in values:
            thresholds = _replace_threshold(selected, field, candidate)
            is_selected = bool(
                np.isclose(
                    candidate, getattr(selected, field), rtol=0.0, atol=1e-15
                )
            )
            evaluations[field].append(
                {
                    "candidate": float(candidate),
                    "selected": is_selected,
                    "selection_reason": _candidate_reason(field, is_selected),
                    "conditions": [
                        _condition_statistics(
                            condition_id, grouped[condition_id], thresholds
                        )
                        for condition_id in _CONDITION_ORDER
                    ],
                }
            )
    return candidates, evaluations


def _hard_error_checks(thresholds: BearingQualityThresholdsV2) -> list[dict[str, Any]]:
    signals = {
        "constant": np.zeros(1024, dtype=np.float64),
        "nan": np.concatenate(
            [np.zeros(1023, dtype=np.float64), np.asarray([np.nan])]
        ),
        "inf": np.concatenate(
            [np.zeros(1023, dtype=np.float64), np.asarray([np.inf])]
        ),
    }
    rows = []
    for condition_id, signal in signals.items():
        result = assess_bearing_window_v2(signal, thresholds=thresholds)
        rows.append(
            {
                "condition_id": condition_id,
                "window_count": 1,
                "status": result.status,
                "reason_code": result.reason_code,
                "rejected_ratio": 1.0 if result.status == "rejected" else 0.0,
            }
        )
    return rows


def calibrate_quality_features_v2(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration_split: str,
) -> dict[str, Any]:
    """Select deterministic warning/rejection limits from validation features."""

    if calibration_split != "validation":
        raise CalibrationV2Error(
            "bearing quality V2 thresholds may only be calibrated on validation data"
        )
    grouped = _group_feature_records(records)
    near_constant_std = _select_near_constant(grouped["clean"])
    reject_clipping_ratio = _select_reject_clipping(grouped["clean"])
    warning_clipping_ratio = _select_warning_clipping(
        grouped, reject_clipping_ratio=reject_clipping_ratio
    )
    reject_spectral_flatness, severe_rejection = _select_reject_flatness(
        grouped,
        near_constant_std=near_constant_std,
        reject_clipping_ratio=reject_clipping_ratio,
    )
    warning_spectral_flatness, target_caution = _select_warning_flatness(
        grouped,
        near_constant_std=near_constant_std,
        warning_clipping_ratio=warning_clipping_ratio,
        reject_clipping_ratio=reject_clipping_ratio,
        reject_spectral_flatness=reject_spectral_flatness,
    )
    selected = BearingQualityThresholdsV2(
        near_constant_std=near_constant_std,
        warning_clipping_ratio=warning_clipping_ratio,
        reject_clipping_ratio=reject_clipping_ratio,
        warning_spectral_flatness=warning_spectral_flatness,
        reject_spectral_flatness=reject_spectral_flatness,
    )
    candidates, evaluations = _candidate_evaluations(grouped, selected)

    conditions = []
    for condition_id in _CONDITION_ORDER:
        row = _condition_statistics(condition_id, grouped[condition_id], selected)
        row.update(
            {
                "role": _CONDITION_ROLES[condition_id],
                "used_for_selection": condition_id != "awgn_0db",
                "feature_distributions": {
                    feature: _distribution(grouped[condition_id], feature)
                    for feature in (
                        "standard_deviation",
                        "clipping_ratio",
                        "spectral_flatness",
                    )
                },
            }
        )
        conditions.append(row)

    return {
        "schema_version": 2,
        "calibration_split": calibration_split,
        "seed": CALIBRATION_SEED,
        "selected_thresholds": asdict(selected),
        "selection_rule": {
            "near_constant": "largest candidate preserving every clean validation window",
            "warning_clipping": "largest feasible warning candidate not exceeding the selected rejection limit; equality disables unsupported clipping-only warnings",
            "reject_clipping": "smallest candidate preserving every clean validation window",
            "warning_spectral_flatness": "maximize mean caution ratio at 10 dB and 5 dB while clean and 20 dB remain at least 99% accepted; choose the larger threshold on ties",
            "reject_spectral_flatness": "maximize rejection at -5 dB while clean/20/10/5 dB satisfy non-rejection constraints; choose the larger threshold on ties",
            "warning_accepted_minimums": dict(_WARNING_ACCEPTED_MINIMUMS),
            "reject_nonrejected_minimums": dict(_REJECT_NONREJECTED_MINIMUMS),
            "warning_target_conditions": list(_WARNING_TARGETS),
            "reject_target_conditions": list(_REJECT_TARGETS),
            "report_only_conditions": ["awgn_0db"],
            "selected_mean_warning_target_caution_ratio": target_caution,
            "selected_mean_severe_noise_rejection_ratio": severe_rejection,
        },
        "candidate_thresholds": candidates,
        "candidate_evaluations": evaluations,
        "conditions": conditions,
        "hard_error_checks": _hard_error_checks(selected),
        "limitations": [
            "CWRU is experimental test-rig bearing data, not field deployment data.",
            "AWGN cases are deterministic controlled pressure tests, not field sensor failures.",
            "The 0 dB condition is report-only and does not influence threshold selection.",
            "No classifier is loaded and no classification metric is produced.",
        ],
    }


def _csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = payload["selected_thresholds"]
    rows: list[dict[str, Any]] = []
    for condition in payload["conditions"]:
        distributions = condition["feature_distributions"]
        rows.append(
            {
                "row_type": "selected_condition",
                "field": "",
                "candidate": "",
                "selected": "",
                "selection_reason": "",
                "condition_id": condition["condition_id"],
                "role": condition["role"],
                "used_for_selection": condition["used_for_selection"],
                "window_count": condition["window_count"],
                "accepted_count": condition["status_counts"]["accepted"],
                "caution_count": condition["status_counts"]["caution"],
                "rejected_count": condition["status_counts"]["rejected"],
                "accepted_ratio": condition["status_ratios"]["accepted"],
                "caution_ratio": condition["status_ratios"]["caution"],
                "rejected_ratio": condition["status_ratios"]["rejected"],
                "spectral_flatness_min": distributions["spectral_flatness"]["minimum"],
                "spectral_flatness_p95": distributions["spectral_flatness"]["p95"],
                "spectral_flatness_max": distributions["spectral_flatness"]["maximum"],
                **thresholds,
            }
        )
    for field, evaluations in payload["candidate_evaluations"].items():
        for evaluation in evaluations:
            for condition in evaluation["conditions"]:
                rows.append(
                    {
                        "row_type": "candidate_condition",
                        "field": field,
                        "candidate": evaluation["candidate"],
                        "selected": evaluation["selected"],
                        "selection_reason": evaluation["selection_reason"],
                        "condition_id": condition["condition_id"],
                        "role": "",
                        "used_for_selection": condition["condition_id"] != "awgn_0db",
                        "window_count": condition["window_count"],
                        "accepted_count": condition["status_counts"]["accepted"],
                        "caution_count": condition["status_counts"]["caution"],
                        "rejected_count": condition["status_counts"]["rejected"],
                        "accepted_ratio": condition["status_ratios"]["accepted"],
                        "caution_ratio": condition["status_ratios"]["caution"],
                        "rejected_ratio": condition["status_ratios"]["rejected"],
                        "spectral_flatness_min": "",
                        "spectral_flatness_p95": "",
                        "spectral_flatness_max": "",
                        **thresholds,
                    }
                )
    for check in payload["hard_error_checks"]:
        rows.append(
            {
                "row_type": "hard_error_check",
                "field": "",
                "candidate": "",
                "selected": "",
                "selection_reason": check["reason_code"],
                "condition_id": check["condition_id"],
                "role": "hard_error",
                "used_for_selection": False,
                "window_count": check["window_count"],
                "accepted_count": 0,
                "caution_count": 0,
                "rejected_count": check["window_count"],
                "accepted_ratio": 0.0,
                "caution_ratio": 0.0,
                "rejected_ratio": check["rejected_ratio"],
                "spectral_flatness_min": "",
                "spectral_flatness_p95": "",
                "spectral_flatness_max": "",
                **thresholds,
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


def write_calibration_outputs_v2(
    payload: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Atomically write strict JSON/CSV evidence without overwriting history."""

    json_path = Path(json_path)
    csv_path = Path(csv_path)
    existing = [str(path) for path in (json_path, csv_path) if path.exists()]
    if existing:
        raise CalibrationV2Error(f"refusing to overwrite existing output: {existing}")
    try:
        json_text = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationV2Error("payload is not strict-JSON serializable") from exc
    csv_rows = _csv_rows(payload)
    if not csv_rows:
        raise CalibrationV2Error("calibration payload has no CSV rows")
    fieldnames = list(csv_rows[0])
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
        existing = [str(path) for path in (json_path, csv_path) if path.exists()]
        if existing:
            raise CalibrationV2Error(f"refusing to overwrite existing output: {existing}")
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
        BearingPerturbation("clean", "clean", None, "unmodified validation window"),
        BearingPerturbation("awgn_20db", "awgn_snr_db", 20.0, "controlled AWGN at 20 dB SNR"),
        BearingPerturbation("awgn_10db", "awgn_snr_db", 10.0, "controlled AWGN at 10 dB SNR"),
        BearingPerturbation("awgn_5db", "awgn_snr_db", 5.0, "controlled AWGN at 5 dB SNR"),
        BearingPerturbation("awgn_0db", "awgn_snr_db", 0.0, "controlled AWGN at 0 dB SNR; report only"),
        BearingPerturbation("awgn_minus5db", "awgn_snr_db", -5.0, "controlled AWGN at -5 dB SNR"),
    )


def collect_validation_quality_features_v2(
    *,
    manifest_path: Path,
    label_map_path: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only validation windows and extract pre-normalization quality features."""

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
            result = assess_bearing_window_v2(perturbed.squeeze(0).cpu().numpy())
            if not result.features:
                raise CalibrationV2Error(
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
        "dataset_name": "Case Western Reserve University Bearing Data Center",
        "dataset_scope": "experimental test-rig bearing data",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "label_map_path": str(label_map_path.resolve()),
        "label_map_sha256": _sha256(label_map_path),
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
        raise CalibrationV2Error(
            "refusing to overwrite existing V2 calibration output; choose new paths"
        )
    records, source = collect_validation_quality_features_v2(
        manifest_path=args.manifest,
        label_map_path=args.label_map,
        project_root=_PROJECT_ROOT,
    )
    payload = calibrate_quality_features_v2(records, calibration_split="validation")
    payload["created_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    payload["source"] = source
    write_calibration_outputs_v2(
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
                "validation_recording_count": source["validation_recording_count"],
                "validation_window_count": source["validation_window_count"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
