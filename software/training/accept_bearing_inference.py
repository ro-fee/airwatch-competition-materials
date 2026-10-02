"""Backend acceptance run for the frozen anti-noise bearing contract.

This is an inference-only acceptance command. It loads the frozen contract,
uses its checkpoint/labels/calibration/manifest, exercises the UI-independent
workflow, and writes new acceptance evidence without changing any input.

Run from the project root with::

    python -m training.accept_bearing_inference
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from airwatch.analysis import load_bearing_quality_thresholds_v2
from airwatch.workflows import BearingDiagnosisWorkflow
from training.common import collect_environment, resolve_project_path, resolve_device
from training.verify_bearing_contract import (
    DEFAULT_CONTRACT,
    BearingContractError,
    verify_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_NAME = "bearing_cnn_anti_noise_20260904_inference_acceptance"


class BearingInferenceAcceptanceError(ValueError):
    """Raised when the frozen backend inference acceptance fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve_contract_path(contract: Mapping[str, Any], *keys: str) -> Path:
    value: Any = contract
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise BearingInferenceAcceptanceError(
                "frozen contract is missing path: " + ".".join(keys)
            )
        value = value[key]
    if not isinstance(value, str) or not value:
        raise BearingInferenceAcceptanceError(
            "frozen contract path is not a non-empty string: " + ".".join(keys)
        )
    return resolve_project_path(PROJECT_ROOT, value)


def _load_contract(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BearingInferenceAcceptanceError(
            f"cannot read frozen contract: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise BearingInferenceAcceptanceError("frozen contract must be a JSON object")
    return payload


def _read_manifest(path: Path) -> list[dict[str, str]]:
    required = {
        "local_path",
        "original_filename",
        "class_name",
        "label_index",
        "sensor_key",
        "window_count",
        "split",
    }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise BearingInferenceAcceptanceError(
                    "manifest is missing required columns: " + ", ".join(missing)
                )
            rows = [
                {str(key): str(value or "") for key, value in row.items()}
                for row in reader
            ]
    except UnicodeDecodeError as exc:
        raise BearingInferenceAcceptanceError(f"manifest is not UTF-8 text: {path}") from exc
    if not rows:
        raise BearingInferenceAcceptanceError(f"manifest contains no rows: {path}")
    return rows


def _label_map(contract: Mapping[str, Any]) -> dict[int, str]:
    labels = contract["input_contract"]["label_map"]["labels"]
    if not isinstance(labels, Mapping):
        raise BearingInferenceAcceptanceError("contract label map is not an object")
    return {int(index): str(label) for index, label in labels.items()}


def _majority(values: Sequence[int]) -> int | None:
    if not values:
        return None
    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    return min(counts, key=lambda value: (-counts[value], value))


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _validate_quality_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_map: Mapping[int, str],
    expected_count: int,
) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise BearingInferenceAcceptanceError(
            f"quality result count mismatch: expected {expected_count}, got {len(rows)}"
        )
    status_counts = _status_counts(rows)
    prediction_count = 0
    for index, row in enumerate(rows):
        status = row.get("status")
        if status not in {"accepted", "caution", "rejected"}:
            raise BearingInferenceAcceptanceError(
                f"window {index} has invalid quality status: {status!r}"
            )
        prediction = row.get("prediction")
        if status == "rejected":
            if bool(row.get("accepted")) or prediction is not None:
                raise BearingInferenceAcceptanceError(
                    f"rejected window {index} carried model output"
                )
        else:
            if not bool(row.get("accepted")) or not isinstance(prediction, Mapping):
                raise BearingInferenceAcceptanceError(
                    f"{status} window {index} did not carry model output"
                )
            predicted_class = int(prediction["predicted_class"])
            if predicted_class not in label_map:
                raise BearingInferenceAcceptanceError(
                    f"window {index} emitted unknown class {predicted_class}"
                )
            confidence = float(prediction["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise BearingInferenceAcceptanceError(
                    f"window {index} emitted invalid confidence {confidence}"
                )
            prediction_count += 1
    return {
        "window_count": len(rows),
        "status_counts": status_counts,
        "prediction_count": prediction_count,
        "rejected_count": status_counts.get("rejected", 0),
    }


def _check_synthetic_quality_states(
    workflow: BearingDiagnosisWorkflow,
    thresholds: Any,
    *,
    window_size: int,
) -> list[dict[str, Any]]:
    # Keep one cycle across the window.  At this sampling density, repeating
    # many cycles creates more samples at the exact extrema and trips the
    # frozen clipping threshold; this check is meant to exercise the accepted
    # path, not to characterize a particular bearing waveform.
    phase = np.linspace(0.0, 2.0 * np.pi, window_size, endpoint=False)
    structured = np.sin(phase).astype(np.float32)
    caution = (
        np.sin(phase)
        + 0.5 * np.random.default_rng(20260904).normal(size=window_size)
    ).astype(np.float32)
    rejected = np.full(window_size, 0.25, dtype=np.float32)
    cases = {
        "accepted_signal": (structured, "accepted", True),
        "caution_signal": (caution, "caution", True),
        "near_constant_signal": (rejected, "rejected", False),
    }
    results: list[dict[str, Any]] = []
    for name, (signal, expected_status, should_enter_model) in cases.items():
        rows = workflow.run_signal_with_quality_v2(signal, thresholds=thresholds)
        if len(rows) != 1:
            raise BearingInferenceAcceptanceError(
                f"synthetic case {name} did not produce one window"
            )
        row = rows[0]
        actual_status = str(row["status"])
        has_prediction = row.get("prediction") is not None
        if actual_status != expected_status or has_prediction != should_enter_model:
            raise BearingInferenceAcceptanceError(
                f"synthetic case {name} expected status={expected_status!r}, "
                f"model_entry={should_enter_model}; got status={actual_status!r}, "
                f"model_entry={has_prediction}"
            )
        results.append(
            {
                "case": name,
                "expected_status": expected_status,
                "actual_status": actual_status,
                "model_entry": has_prediction,
                "reason_code": row["reason_code"],
                "quality_features": dict(row["quality_features"]),
            }
        )
    return results


def _write_outputs(
    json_path: Path,
    csv_path: Path,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if json_path.exists() or csv_path.exists():
        existing = [str(path) for path in (json_path, csv_path) if path.exists()]
        raise FileExistsError(
            "refusing to overwrite inference acceptance evidence: "
            + ", ".join(existing)
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        json_fd, json_name = tempfile.mkstemp(
            prefix=f".{json_path.name}.", suffix=".tmp", dir=json_path.parent
        )
        os.close(json_fd)
        csv_fd, csv_name = tempfile.mkstemp(
            prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent
        )
        os.close(csv_fd)
        temporary_paths.extend([Path(json_name), Path(csv_name)])
        temporary_paths[0].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        fieldnames: list[str] = []
        for row in rows:
            for name in row:
                if name not in fieldnames:
                    fieldnames.append(name)
        with temporary_paths[1].open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    name: json.dumps(value, ensure_ascii=False, allow_nan=False)
                    if isinstance(value, (dict, list, tuple))
                    else ("" if value is None else value)
                    for name, value in row.items()
                })
        if json_path.exists() or csv_path.exists():
            raise FileExistsError("inference acceptance output appeared during write")
        temporary_paths[0].replace(json_path)
        temporary_paths[1].replace(csv_path)
    except BaseException:
        for path in (json_path, csv_path):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def run_acceptance(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT,
    run_name: str = DEFAULT_RUN_NAME,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Run frozen-contract inference checks and write new evidence."""
    contract_file = Path(contract_path).resolve()
    try:
        contract_check = verify_contract(contract_file)
    except (BearingContractError, OSError, KeyError, TypeError, ValueError) as exc:
        raise BearingInferenceAcceptanceError(
            f"frozen contract verification failed: {exc}"
        ) from exc
    contract = _load_contract(contract_file)
    label_map = _label_map(contract)
    expected_order = list(contract["input_contract"]["label_map"]["label_order"])
    if [label_map[index] for index in sorted(label_map)] != expected_order:
        raise BearingInferenceAcceptanceError("label order does not match frozen contract")

    checkpoint = _resolve_contract_path(contract, "model", "checkpoint", "path")
    label_map_path = _resolve_contract_path(
        contract, "input_contract", "label_map", "path"
    )
    calibration = _resolve_contract_path(
        contract, "quality_gate", "calibration", "path"
    )
    manifest = _resolve_contract_path(contract, "data_contract", "manifest_path")
    thresholds = load_bearing_quality_thresholds_v2(calibration)
    device = resolve_device(device_name)

    rows = _read_manifest(manifest)
    if len(rows) != 16:
        raise BearingInferenceAcceptanceError(
            f"expected 16 manifest rows, got {len(rows)}"
        )
    required_splits = {"train", "validation", "test"}
    if {row["split"] for row in rows} != required_splits:
        raise BearingInferenceAcceptanceError("manifest split coverage is incomplete")

    source_paths = [resolve_project_path(PROJECT_ROOT, row["local_path"]) for row in rows]
    protected_paths = [contract_file, checkpoint, label_map_path, calibration, manifest]
    protected_paths.extend(source_paths)
    before = {
        str(path.resolve()): {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in protected_paths
    }

    workflow = BearingDiagnosisWorkflow.from_checkpoint(
        checkpoint,
        device=str(device),
        label_map_path=label_map_path,
    )
    predictor = workflow.adapter.predictor
    if predictor.normalization != contract["input_contract"]["normalization"]["name"]:
        raise BearingInferenceAcceptanceError("loaded predictor normalization drifted")
    if predictor.window_size != int(contract["input_contract"]["window_size"]):
        raise BearingInferenceAcceptanceError("loaded predictor window size drifted")
    if predictor.step != int(contract["input_contract"]["step"]):
        raise BearingInferenceAcceptanceError("loaded predictor step drifted")
    if predictor.label_map != label_map:
        raise BearingInferenceAcceptanceError("loaded predictor label map drifted")

    synthetic_results = _check_synthetic_quality_states(
        workflow,
        thresholds,
        window_size=predictor.window_size,
    )

    file_results: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_path = resolve_project_path(PROJECT_ROOT, row["local_path"])
        expected_label_index = int(row["label_index"])
        expected_window_count = int(row["window_count"])
        direct = workflow.run_file(source_path, sensor_key=row["sensor_key"])
        quality_rows = workflow.run_file_with_quality_v2(
            source_path,
            sensor_key=row["sensor_key"],
            thresholds=thresholds,
        )
        quality_summary = _validate_quality_rows(
            quality_rows,
            label_map=label_map,
            expected_count=expected_window_count,
        )
        direct_classes = [int(item["predicted_class"]) for item in direct.windows]
        quality_classes = [
            int(item["prediction"]["predicted_class"])
            for item in quality_rows
            if item.get("prediction") is not None
        ]
        direct_majority = _majority(direct_classes)
        quality_majority = _majority(quality_classes)
        direct_matches_quality = (
            len(quality_classes) == len(direct_classes)
            and direct_classes == quality_classes
        )
        expected_label = label_map[expected_label_index]
        file_result = {
            "manifest_row": index + 2,
            "split": row["split"],
            "original_filename": row["original_filename"],
            "source_path": row["local_path"],
            "sensor_key": row["sensor_key"],
            "expected_class": expected_label,
            "expected_class_index": expected_label_index,
            "window_count": expected_window_count,
            "direct_window_count": direct.window_count,
            "quality_window_count": quality_summary["window_count"],
            "direct_majority_class": label_map[direct_majority]
            if direct_majority is not None
            else None,
            "quality_majority_class": label_map[quality_majority]
            if quality_majority is not None
            else None,
            "direct_file_correct": direct_majority == expected_label_index,
            "quality_file_correct": quality_majority == expected_label_index,
            "direct_mean_confidence": direct.mean_confidence,
            "quality_status_counts": quality_summary["status_counts"],
            "quality_prediction_count": quality_summary["prediction_count"],
            "direct_quality_prediction_agreement": direct_matches_quality,
        }
        file_results.append(file_result)
        file_rows.append(file_result)

    after = {
        path: {
            "size_bytes": Path(path).stat().st_size,
            "sha256": _sha256(Path(path)),
        }
        for path in before
    }
    changed_paths = [path for path in before if before[path] != after[path]]
    if changed_paths:
        raise BearingInferenceAcceptanceError(
            "protected input files changed during inference: " + ", ".join(changed_paths)
        )

    test_results = [item for item in file_results if item["split"] == "test"]
    file_accuracy = sum(bool(item["quality_file_correct"]) for item in file_results) / len(file_results)
    test_file_accuracy = (
        sum(bool(item["quality_file_correct"]) for item in test_results) / len(test_results)
        if test_results
        else None
    )
    all_quality_statuses: dict[str, int] = {}
    for item in file_results:
        for status, count in item["quality_status_counts"].items():
            all_quality_statuses[status] = all_quality_statuses.get(status, 0) + int(count)

    output_dir = PROJECT_ROOT / "artifacts/evidence/bearing/anti_noise"
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}_files.csv"
    environment = collect_environment(device)
    environment.update({
        "executable": str(Path(os.sys.executable).resolve()),
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    })
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_name": run_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Backend inference acceptance for the frozen anti-noise bearing model contract",
        "contract": {
            **contract_check,
            "contract_path": str(contract_file),
            "contract_digest_sha256": contract["contract_digest_sha256"],
        },
        "model": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": before[str(checkpoint.resolve())]["sha256"],
            "name": predictor._EXPECTED_MODEL_NAME,
            "device": str(device),
            "window_size": predictor.window_size,
            "step": predictor.step,
            "normalization": predictor.normalization,
            "label_map": {str(index): label for index, label in label_map.items()},
        },
        "quality_gate": {
            "version": contract["quality_gate"]["version"],
            "calibration_path": str(calibration),
            "calibration_split": "validation",
            "thresholds": {
                "near_constant_std": thresholds.near_constant_std,
                "warning_clipping_ratio": thresholds.warning_clipping_ratio,
                "reject_clipping_ratio": thresholds.reject_clipping_ratio,
                "warning_spectral_flatness": thresholds.warning_spectral_flatness,
                "reject_spectral_flatness": thresholds.reject_spectral_flatness,
            },
            "synthetic_state_checks": synthetic_results,
        },
        "dataset": {
            "manifest_path": str(manifest),
            "manifest_row_count": len(rows),
            "raw_file_count": len(source_paths),
            "split_window_counts": {
                split: sum(int(row["window_count"]) for row in rows if row["split"] == split)
                for split in ("train", "validation", "test")
            },
        },
        "results": {
            "file_count": len(file_results),
            "quality_status_counts": all_quality_statuses,
            "quality_file_accuracy_against_manifest_labels": file_accuracy,
            "test_file_accuracy_against_manifest_labels": test_file_accuracy,
            "all_files_have_expected_window_count": all(
                item["quality_window_count"] == item["window_count"] for item in file_results
            ),
            "all_rejected_windows_have_no_prediction": all(
                item["quality_status_counts"].get("rejected", 0) == 0
                or item["quality_prediction_count"]
                <= item["quality_window_count"] - item["quality_status_counts"].get("rejected", 0)
                for item in file_results
            ),
            "direct_and_quality_predictions_agree_when_all_windows_enter_model": all(
                bool(item["direct_quality_prediction_agreement"])
                for item in file_results
                if item["quality_prediction_count"] == item["direct_window_count"]
            ),
        },
        "files": file_results,
        "protected_inputs": {
            "changed_paths": changed_paths,
            "files": after,
        },
        "safety_declarations": {
            "training_performed": False,
            "optimizer_created": False,
            "backward_called": False,
            "checkpoint_modified": False,
            "label_map_modified": False,
            "quality_calibration_modified": False,
            "source_data_modified": False,
            "ui_modified": False,
        },
        "limitations": [
            "This is a backend wiring and behavior acceptance, not a new training run.",
            "File-level correctness is an observation on the fixed CWRU recordings, not an industrial performance claim.",
            "CWRU is laboratory test-rig data; unknown faults, cross-dataset generalization, and live sensor behavior were not tested.",
        ],
        "outputs": {"json": str(json_path), "files_csv": str(csv_path)},
    }
    if not all(
        [
            bool(payload["protected_inputs"]["changed_paths"] == []),
            bool(payload["results"]["all_files_have_expected_window_count"]),
            bool(payload["results"]["all_rejected_windows_have_no_prediction"]),
            bool(payload["results"]["direct_and_quality_predictions_agree_when_all_windows_enter_model"]),
            all(item["quality_file_correct"] for item in file_results),
            all(item["direct_file_correct"] for item in file_results),
        ]
    ):
        raise BearingInferenceAcceptanceError("one or more backend acceptance checks failed")

    _write_outputs(json_path, csv_path, payload, file_rows)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_acceptance(
        contract_path=args.contract,
        run_name=args.run_name,
        device_name=args.device,
    )
    print(json.dumps({
        "passed": True,
        "run_name": result["run_name"],
        "output_json": result["outputs"]["json"],
        "output_files_csv": result["outputs"]["files_csv"],
        "file_count": result["results"]["file_count"],
        "quality_status_counts": result["results"]["quality_status_counts"],
        "test_file_accuracy": result["results"]["test_file_accuracy_against_manifest_labels"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
