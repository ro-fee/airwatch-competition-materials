"""Evaluate a frozen bearing CNN under deterministic in-memory perturbations.

This command never trains, updates, or saves model weights. Source CWRU files are
read-only inputs; all perturbations are applied to temporary tensor copies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

from airwatch.analysis import BearingPerturbation, apply_bearing_perturbation
from airwatch.data import CWRUBearingDataset

from .common import (
    TrainingConfigError,
    build_datasets,
    collect_environment,
    load_config,
    make_model,
    project_root_from_config,
    resolve_device,
    resolve_project_path,
    set_reproducible_seed,
)


DEFAULT_CONDITIONS: tuple[BearingPerturbation, ...] = (
    BearingPerturbation("clean", "clean", None, "Original fixed test windows"),
    BearingPerturbation("awgn_20db", "awgn_snr_db", 20.0, "Additive white Gaussian noise at 20 dB SNR"),
    BearingPerturbation("awgn_10db", "awgn_snr_db", 10.0, "Additive white Gaussian noise at 10 dB SNR"),
    BearingPerturbation("awgn_5db", "awgn_snr_db", 5.0, "Additive white Gaussian noise at 5 dB SNR"),
    BearingPerturbation("awgn_0db", "awgn_snr_db", 0.0, "Additive white Gaussian noise at 0 dB SNR"),
    BearingPerturbation("awgn_minus5db", "awgn_snr_db", -5.0, "Additive white Gaussian noise at -5 dB SNR"),
    BearingPerturbation("amplitude_0p5", "amplitude_scale", 0.5, "Half signal amplitude"),
    BearingPerturbation("amplitude_2p0", "amplitude_scale", 2.0, "Double signal amplitude"),
    BearingPerturbation("shift_128", "circular_shift_samples", 128, "Circular time shift by 128 samples"),
    BearingPerturbation("time_scale_0p95", "time_scale", 0.95, "Simple resampling approximation at scale 0.95"),
    BearingPerturbation("time_scale_1p05", "time_scale", 1.05, "Simple resampling approximation at scale 1.05"),
)

_OUTPUT_SUFFIXES = {
    "evidence": ".json",
    "conditions": "_conditions.csv",
    "predictions": "_predictions.csv",
    "failures": "_failures.csv",
    "noise_curve": "_noise_curve.png",
}


class PerturbedBearingDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Apply one deterministic condition to copies of a fixed base split."""

    def __init__(
        self,
        base: CWRUBearingDataset,
        condition: BearingPerturbation,
        *,
        seed: int,
    ) -> None:
        self.base = base
        self.condition = condition
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        window, target = self.base[index]
        perturbed = apply_bearing_perturbation(
            window,
            self.condition,
            seed=self.seed,
            sample_index=index,
        )
        return perturbed, int(target), index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _output_paths(evidence_dir: Path, run_name: str) -> dict[str, Path]:
    if not run_name or Path(run_name).name != run_name or any(
        char.isspace() or char in '<>:"/\\|?*' for char in run_name
    ):
        raise TrainingConfigError("run name must be one filename-safe component")
    return {
        key: evidence_dir / f"{run_name}{suffix}"
        for key, suffix in _OUTPUT_SUFFIXES.items()
    }


def _ensure_outputs_absent(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise TrainingConfigError(
            "refusing to overwrite robustness evidence: " + "; ".join(existing)
        )


def _make_loader(
    base: CWRUBearingDataset,
    condition: BearingPerturbation,
    *,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = PerturbedBearingDataset(base, condition, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def _evaluate_condition(
    model: nn.Module,
    dataset: CWRUBearingDataset,
    condition: BearingPerturbation,
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = _make_loader(
        dataset,
        condition,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_confidences: list[float] = []
    all_sample_indices: list[int] = []
    total_loss = 0.0
    total_count = 0

    model.eval()
    with torch.inference_mode():
        for inputs, targets, sample_indices in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            count = int(targets.shape[0])
            total_loss += float(loss.item()) * count
            total_count += count
            all_targets.extend(int(value) for value in targets.cpu().tolist())
            all_predictions.extend(int(value) for value in predictions.cpu().tolist())
            all_confidences.extend(float(value) for value in confidences.cpu().tolist())
            all_sample_indices.extend(int(value) for value in sample_indices.tolist())

    if total_count == 0:
        raise TrainingConfigError("cannot evaluate an empty test split")
    if all_sample_indices != list(range(len(dataset))):
        raise TrainingConfigError("test loader changed fixed sample order")

    target_array = np.asarray(all_targets, dtype=np.int64)
    prediction_array = np.asarray(all_predictions, dtype=np.int64)
    labels = list(range(dataset.num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        target_array,
        prediction_array,
        labels=labels,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "loss": total_loss / total_count,
        "accuracy": float(np.mean(target_array == prediction_array)),
        "macro_f1": float(
            f1_score(
                target_array,
                prediction_array,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "sample_count": total_count,
        "correct_count": int(np.sum(target_array == prediction_array)),
        "failure_count": int(np.sum(target_array != prediction_array)),
        "mean_confidence": float(np.mean(all_confidences)),
        "confusion_matrix": confusion_matrix(
            target_array, prediction_array, labels=labels
        ).tolist(),
        "per_class": {
            str(index): {
                "label": dataset.label_map[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }

    rows: list[dict[str, Any]] = []
    for sample_index, target, prediction, confidence in zip(
        all_sample_indices, all_targets, all_predictions, all_confidences
    ):
        metadata = dataset.sample_metadata(sample_index)
        rows.append(
            {
                "condition_id": condition.condition_id,
                "sample_index": sample_index,
                "source_path": str(metadata.path),
                "original_filename": metadata.original_filename,
                "split_group": metadata.split_group,
                "window_index": metadata.window_index,
                "window_start": metadata.window_index * metadata.step,
                "target": target,
                "target_label": dataset.label_map[target],
                "prediction": prediction,
                "predicted_label": dataset.label_map[prediction],
                "confidence": confidence,
                "correct": target == prediction,
            }
        )
    return metrics, rows


def _read_reference_predictions(path: Path) -> list[tuple[int, int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"clean reference predictions do not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        return [
            (int(row["sample_index"]), int(row["target"]), int(row["prediction"]))
            for row in rows
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingConfigError(f"invalid clean reference CSV: {path}: {exc}") from exc


def _assert_clean_matches_reference(
    clean_metrics: dict[str, Any],
    clean_rows: list[dict[str, Any]],
    reference_path: Path,
) -> dict[str, Any]:
    if clean_metrics["sample_count"] != 831:
        raise TrainingConfigError(
            f"clean sample count changed: expected 831, got {clean_metrics['sample_count']}"
        )
    if clean_metrics["accuracy"] != 1.0 or clean_metrics["macro_f1"] != 1.0:
        raise TrainingConfigError(
            "clean baseline no longer matches expected accuracy=1.0 and macro_f1=1.0"
        )
    actual = [
        (int(row["sample_index"]), int(row["target"]), int(row["prediction"]))
        for row in clean_rows
    ]
    reference = _read_reference_predictions(reference_path)
    if actual != reference:
        mismatch = next(
            (
                {"position": index, "actual": left, "reference": right}
                for index, (left, right) in enumerate(zip(actual, reference))
                if left != right
            ),
            {"actual_length": len(actual), "reference_length": len(reference)},
        )
        raise TrainingConfigError(
            "clean predictions differ from frozen reference: " + json.dumps(mismatch)
        )
    return {
        "passed": True,
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "matched_rows": len(actual),
        "expected_accuracy": 1.0,
        "expected_macro_f1": 1.0,
    }


def _reproducibility_check(
    first: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    second: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    passed = True
    for condition_id in first:
        first_metrics, first_rows = first[condition_id]
        second_metrics, second_rows = second[condition_id]
        first_core = [
            (row["sample_index"], row["target"], row["prediction"])
            for row in first_rows
        ]
        second_core = [
            (row["sample_index"], row["target"], row["prediction"])
            for row in second_rows
        ]
        prediction_match = first_core == second_core
        first_confidence = np.asarray(
            [row["confidence"] for row in first_rows], dtype=np.float64
        )
        second_confidence = np.asarray(
            [row["confidence"] for row in second_rows], dtype=np.float64
        )
        max_confidence_difference = float(
            np.max(np.abs(first_confidence - second_confidence))
        )
        metric_differences = {
            key: abs(float(first_metrics[key]) - float(second_metrics[key]))
            for key in ("loss", "accuracy", "macro_f1", "mean_confidence")
        }
        condition_passed = (
            prediction_match
            and max_confidence_difference == 0.0
            and max(metric_differences.values()) == 0.0
        )
        passed = passed and condition_passed
        details[condition_id] = {
            "passed": condition_passed,
            "prediction_match": prediction_match,
            "max_confidence_difference": max_confidence_difference,
            "metric_absolute_differences": metric_differences,
        }
    if not passed:
        raise TrainingConfigError("repeated robustness evaluation was not deterministic")
    return {"passed": True, "method": "complete second in-memory evaluation", "conditions": details}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_noise_curve(
    path: Path,
    conditions: tuple[BearingPerturbation, ...],
    results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> None:
    noise_conditions = [
        condition for condition in conditions if condition.kind == "awgn_snr_db"
    ]
    noise_conditions.sort(key=lambda item: float(item.value))
    x = [float(condition.value) for condition in noise_conditions]
    accuracy = [results[condition.condition_id][0]["accuracy"] for condition in noise_conditions]
    macro_f1 = [results[condition.condition_id][0]["macro_f1"] for condition in noise_conditions]
    clean_accuracy = results["clean"][0]["accuracy"]
    clean_f1 = results["clean"][0]["macro_f1"]

    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    axis.plot(x, accuracy, marker="o", linewidth=2, label="Accuracy")
    axis.plot(x, macro_f1, marker="s", linewidth=2, label="Macro-F1")
    axis.axhline(clean_accuracy, color="tab:blue", linestyle="--", alpha=0.4, label="Clean accuracy")
    if clean_f1 != clean_accuracy:
        axis.axhline(clean_f1, color="tab:orange", linestyle="--", alpha=0.4, label="Clean macro-F1")
    axis.set_xlabel("Signal-to-noise ratio (dB)")
    axis.set_ylabel("Score")
    axis.set_title("Frozen Bearing CNN - AWGN Robustness on CWRU Test Split")
    axis.set_ylim(0.0, 1.03)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _source_inventory(dataset: CWRUBearingDataset) -> list[dict[str, Any]]:
    inventory = []
    for path in dataset.source_files:
        matching = [
            dataset.sample_metadata(index)
            for index in range(len(dataset))
            if dataset.sample_metadata(index).path == path
        ]
        first = matching[0]
        inventory.append(
            {
                "path": str(path),
                "original_filename": first.original_filename,
                "class_name": first.class_name,
                "split_group": first.split_group,
                "window_count": len(matching),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def run_robustness_evaluation(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    run_name: str,
    reference_predictions_path: str | Path,
    conditions: tuple[BearingPerturbation, ...] = DEFAULT_CONDITIONS,
) -> dict[str, Any]:
    project_root = project_root_from_config(config)
    evidence_dir = resolve_project_path(project_root, config["output"]["evidence_dir"])
    output_paths = _output_paths(evidence_dir, run_name)
    _ensure_outputs_absent(output_paths)

    checkpoint = resolve_project_path(project_root, checkpoint_path)
    reference_predictions = resolve_project_path(project_root, reference_predictions_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if len({condition.condition_id for condition in conditions}) != len(conditions):
        raise TrainingConfigError("robustness condition IDs must be unique")
    if not conditions or conditions[0].condition_id != "clean":
        raise TrainingConfigError("the first robustness condition must be clean")

    seed = int(config["training"]["seed"])
    set_reproducible_seed(seed)
    device = resolve_device(str(config["training"]["device"]))
    dataset = build_datasets(config, project_root)["test"]
    checkpoint_hash_before = _sha256(checkpoint)

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TrainingConfigError(f"unsupported checkpoint format: {checkpoint}")
    model = make_model(config).to(device)
    model.load_state_dict(payload["model_state_dict"])

    evaluation_args = {
        "device": device,
        "seed": seed,
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
    }
    first_results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for condition in conditions:
        print(f"[pass 1/2] evaluating {condition.condition_id}", flush=True)
        first_results[condition.condition_id] = _evaluate_condition(
            model, dataset, condition, **evaluation_args
        )

    clean_consistency = _assert_clean_matches_reference(
        first_results["clean"][0], first_results["clean"][1], reference_predictions
    )

    second_results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for condition in conditions:
        print(f"[pass 2/2] verifying {condition.condition_id}", flush=True)
        second_results[condition.condition_id] = _evaluate_condition(
            model, dataset, condition, **evaluation_args
        )
    reproducibility = _reproducibility_check(first_results, second_results)

    clean_metrics = first_results["clean"][0]
    condition_summaries: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    metrics_by_condition: dict[str, Any] = {}
    for condition in conditions:
        metrics, rows = first_results[condition.condition_id]
        metrics["accuracy_drop_from_clean"] = clean_metrics["accuracy"] - metrics["accuracy"]
        metrics["macro_f1_drop_from_clean"] = clean_metrics["macro_f1"] - metrics["macro_f1"]
        metrics_by_condition[condition.condition_id] = {
            "condition": asdict(condition),
            "metrics": metrics,
        }
        condition_summaries.append(
            {
                "condition_id": condition.condition_id,
                "kind": condition.kind,
                "value": condition.value,
                "description": condition.description,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mean_confidence": metrics["mean_confidence"],
                "correct_count": metrics["correct_count"],
                "failure_count": metrics["failure_count"],
                "accuracy_drop_from_clean": metrics["accuracy_drop_from_clean"],
                "macro_f1_drop_from_clean": metrics["macro_f1_drop_from_clean"],
            }
        )
        all_predictions.extend(rows)
        all_failures.extend(row for row in rows if not row["correct"])

    checkpoint_hash_after = _sha256(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise TrainingConfigError("checkpoint hash changed during read-only evaluation")

    environment = collect_environment(device)
    environment.update(
        {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        }
    )
    label_map_path = resolve_project_path(project_root, config["dataset"]["label_map"])
    result: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "run_name": run_name,
        "purpose": "read-only pressure test of one frozen bearing CNN checkpoint",
        "training_performed": False,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoint_saved": False,
        "checkpoint_modified": False,
        "ui_modified": False,
        "source_data_modified": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256_before": checkpoint_hash_before,
            "sha256_after": checkpoint_hash_after,
            "hash_unchanged": checkpoint_hash_before == checkpoint_hash_after,
            "model_name": payload.get("model_name"),
            "training_epoch": payload.get("epoch"),
        },
        "config_path": config["_config_path"],
        "seed": seed,
        "split": "test",
        "dataset": {
            "name": config["dataset"]["name"],
            "manifest": str(dataset.manifest_path),
            "manifest_sha256": _sha256(dataset.manifest_path),
            "label_map_path": str(label_map_path),
            "label_map_sha256": _sha256(label_map_path),
            "label_map": dataset.label_map,
            "window_count": len(dataset),
            "window_size": dataset.window_size,
            "step": dataset.step,
            "normalization": dataset.normalization,
            "source_files": _source_inventory(dataset),
        },
        "clean_consistency_check": clean_consistency,
        "reproducibility_check": reproducibility,
        "conditions": metrics_by_condition,
        "environment": environment,
        "limitations": [
            "CWRU is a laboratory bearing dataset; these results do not establish industrial field accuracy.",
            "Only the fixed audited CWRU test split is evaluated.",
            "AWGN is a controlled mathematical disturbance and does not cover all real sensor noise.",
            "Time scaling is a simple resampling approximation, not a physical rotating-machine simulation.",
            "No unknown-fault/open-set or cross-dataset generalization is tested here.",
        ],
        "output_files": {key: str(path) for key, path in output_paths.items()},
    }

    # All checks above must pass before any official evidence file is written.
    output_paths["evidence"].parent.mkdir(parents=True, exist_ok=True)
    _write_noise_curve(output_paths["noise_curve"], conditions, first_results)
    _write_csv(
        output_paths["conditions"],
        condition_summaries,
        list(condition_summaries[0].keys()),
    )
    prediction_fields = list(all_predictions[0].keys())
    _write_csv(output_paths["predictions"], all_predictions, prediction_fields)
    _write_csv(output_paths["failures"], all_failures, prediction_fields)
    _write_json(output_paths["evidence"], result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/bearing_cnn_baseline.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/bearing/bearing_cnn_baseline_best.pt"),
    )
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        default=Path("artifacts/evidence/bearing/bearing_cnn_baseline_test_predictions.csv"),
    )
    parser.add_argument(
        "--run-name",
        default="bearing_cnn_baseline_robustness_20260903",
    )
    args = parser.parse_args()
    result = run_robustness_evaluation(
        load_config(args.config),
        args.checkpoint,
        run_name=args.run_name,
        reference_predictions_path=args.reference_predictions,
    )
    summary = {
        "run_name": result["run_name"],
        "checkpoint_unchanged": result["checkpoint"]["hash_unchanged"],
        "clean_consistency": result["clean_consistency_check"]["passed"],
        "reproducible": result["reproducibility_check"]["passed"],
        "conditions": {
            key: value["metrics"]
            for key, value in result["conditions"].items()
        },
        "output_files": result["output_files"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
