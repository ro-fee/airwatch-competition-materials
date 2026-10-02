"""Audit preprocessing order for a frozen bearing fault classifier.

This module compares the historical robustness-test order (normalize, then
perturb) with the deployment-like order (perturb raw data, then normalize).
It is deliberately read-only with respect to source data, UI files, and model
checkpoints.  It never creates an optimizer, calls backward, trains, or saves
weights.
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

import numpy as np
import sklearn
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

from airwatch.analysis import BearingPerturbation, apply_bearing_perturbation
from airwatch.data import CWRUBearingDataset, normalize_windows

from .common import (
    TrainingConfigError,
    collect_environment,
    load_config,
    make_model,
    project_root_from_config,
    resolve_device,
    resolve_project_path,
    set_reproducible_seed,
)


NORMALIZE_THEN_PERTURB = "normalize_then_perturb"
PERTURB_THEN_NORMALIZE = "perturb_then_normalize"
ORDERS = (NORMALIZE_THEN_PERTURB, PERTURB_THEN_NORMALIZE)

DEFAULT_CONDITIONS: tuple[BearingPerturbation, ...] = (
    BearingPerturbation("clean", "clean", None, "Original fixed test windows"),
    BearingPerturbation("amplitude_0p5", "amplitude_scale", 0.5, "Half signal amplitude"),
    BearingPerturbation("amplitude_2p0", "amplitude_scale", 2.0, "Double signal amplitude"),
    BearingPerturbation("awgn_20db", "awgn_snr_db", 20.0, "Additive white Gaussian noise at 20 dB SNR"),
    BearingPerturbation("awgn_10db", "awgn_snr_db", 10.0, "Additive white Gaussian noise at 10 dB SNR"),
    BearingPerturbation("awgn_5db", "awgn_snr_db", 5.0, "Additive white Gaussian noise at 5 dB SNR"),
    BearingPerturbation("awgn_0db", "awgn_snr_db", 0.0, "Additive white Gaussian noise at 0 dB SNR"),
    BearingPerturbation("awgn_minus5db", "awgn_snr_db", -5.0, "Additive white Gaussian noise at -5 dB SNR"),
)

_OUTPUT_SUFFIXES = {
    "evidence": ".json",
    "conditions": "_conditions.csv",
    "predictions": "_predictions.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _output_paths(evidence_dir: Path, run_name: str) -> dict[str, Path]:
    """Return the three official outputs after validating a safe run name."""
    if not run_name or Path(run_name).name != run_name or any(
        char.isspace() or char in '<>:"/\\|?*' for char in run_name
    ):
        raise TrainingConfigError("run name must be one filename-safe component")
    return {
        key: Path(evidence_dir) / f"{run_name}{suffix}"
        for key, suffix in _OUTPUT_SUFFIXES.items()
    }


def _ensure_outputs_absent(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise TrainingConfigError(
            "refusing to overwrite preprocessing-order audit evidence: "
            + "; ".join(existing)
        )


def _window_zscore(window: torch.Tensor) -> torch.Tensor:
    normalized = normalize_windows(
        window.detach().cpu().numpy(),
        normalization="window_zscore",
    )
    result = torch.from_numpy(np.array(normalized, dtype=np.float32, copy=True))
    return result.to(device=window.device, dtype=torch.float32)


def apply_preprocess_order(
    raw_window: torch.Tensor,
    condition: BearingPerturbation,
    *,
    order: str,
    normalization: str,
    seed: int,
    sample_index: int,
) -> torch.Tensor:
    """Build one model input without modifying the supplied raw window."""
    if normalization != "window_zscore":
        raise TrainingConfigError(
            "preprocessing-order audit currently requires normalization='window_zscore'"
        )
    if order not in ORDERS:
        raise TrainingConfigError(f"unsupported preprocessing order: {order!r}")
    if not isinstance(raw_window, torch.Tensor):
        raise TypeError("raw_window must be a torch.Tensor")

    raw_copy = raw_window.detach().clone().to(dtype=torch.float32)
    if order == NORMALIZE_THEN_PERTURB:
        normalized = _window_zscore(raw_copy)
        result = apply_bearing_perturbation(
            normalized,
            condition,
            seed=seed,
            sample_index=sample_index,
        )
    else:
        perturbed = apply_bearing_perturbation(
            raw_copy,
            condition,
            seed=seed,
            sample_index=sample_index,
        )
        result = _window_zscore(perturbed)
    return result.detach().clone().to(device=raw_window.device, dtype=torch.float32)


class _PairedPreprocessDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]]
):
    """Create both audited inputs from each fixed raw test window."""

    def __init__(
        self,
        base: CWRUBearingDataset,
        condition: BearingPerturbation,
        *,
        normalization: str,
        seed: int,
    ) -> None:
        self.base = base
        self.condition = condition
        self.normalization = normalization
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        raw_window, target = self.base[index]
        clean_input = apply_preprocess_order(
            raw_window,
            DEFAULT_CONDITIONS[0],
            order=PERTURB_THEN_NORMALIZE,
            normalization=self.normalization,
            seed=self.seed,
            sample_index=index,
        )
        historical_input = apply_preprocess_order(
            raw_window,
            self.condition,
            order=NORMALIZE_THEN_PERTURB,
            normalization=self.normalization,
            seed=self.seed,
            sample_index=index,
        )
        deployment_like_input = apply_preprocess_order(
            raw_window,
            self.condition,
            order=PERTURB_THEN_NORMALIZE,
            normalization=self.normalization,
            seed=self.seed,
            sample_index=index,
        )
        return historical_input, deployment_like_input, clean_input, int(target), index


def _make_loader(
    dataset: CWRUBearingDataset,
    condition: BearingPerturbation,
    *,
    normalization: str,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    paired = _PairedPreprocessDataset(
        dataset,
        condition,
        normalization=normalization,
        seed=seed,
    )
    return DataLoader(
        paired,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def _metrics(
    *,
    total_loss: float,
    targets: list[int],
    predictions: list[int],
    confidences: list[float],
    label_map: dict[int, str],
) -> dict[str, Any]:
    if not targets:
        raise TrainingConfigError("cannot evaluate an empty test split")
    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    labels = list(range(len(label_map)))
    precision, recall, f1, support = precision_recall_fscore_support(
        target_array,
        prediction_array,
        labels=labels,
        zero_division=0,
    )
    correct_count = int(np.sum(target_array == prediction_array))
    return {
        "loss": total_loss / len(targets),
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
        "mean_confidence": float(np.mean(np.asarray(confidences, dtype=np.float64))),
        "sample_count": len(targets),
        "correct_count": correct_count,
        "failure_count": len(targets) - correct_count,
        "confusion_matrix": confusion_matrix(
            target_array, prediction_array, labels=labels
        ).tolist(),
        "per_class": {
            str(index): {
                "class_name": label_map[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }


def _evaluate_condition(
    model: nn.Module,
    dataset: CWRUBearingDataset,
    condition: BearingPerturbation,
    *,
    normalization: str,
    device: torch.device,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = _make_loader(
        dataset,
        condition,
        normalization=normalization,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    targets: list[int] = []
    sample_indices: list[int] = []
    order_predictions = {order: [] for order in ORDERS}
    order_confidences = {order: [] for order in ORDERS}
    order_losses = {order: 0.0 for order in ORDERS}
    rows: list[dict[str, Any]] = []

    pair_abs_sum = 0.0
    pair_element_count = 0
    pair_abs_max = 0.0
    clean_abs_sum = {order: 0.0 for order in ORDERS}
    clean_abs_max = {order: 0.0 for order in ORDERS}

    model.eval()
    with torch.inference_mode():
        for historical, deployment_like, clean_input, batch_targets, batch_indices in loader:
            batch_targets_device = batch_targets.to(device)
            input_by_order = {
                NORMALIZE_THEN_PERTURB: historical.to(device),
                PERTURB_THEN_NORMALIZE: deployment_like.to(device),
            }
            batch_results: dict[str, tuple[list[int], list[float]]] = {}
            for order in ORDERS:
                logits = model(input_by_order[order])
                order_losses[order] += float(
                    criterion(logits, batch_targets_device).item()
                )
                probabilities = torch.softmax(logits, dim=1)
                confidence_tensor, prediction_tensor = probabilities.max(dim=1)
                batch_predictions = [
                    int(value) for value in prediction_tensor.detach().cpu().tolist()
                ]
                batch_confidences = [
                    float(value) for value in confidence_tensor.detach().cpu().tolist()
                ]
                order_predictions[order].extend(batch_predictions)
                order_confidences[order].extend(batch_confidences)
                batch_results[order] = (batch_predictions, batch_confidences)

            pair_difference = torch.abs(historical - deployment_like)
            historical_clean_difference = torch.abs(historical - clean_input)
            deployment_clean_difference = torch.abs(deployment_like - clean_input)
            pair_abs_sum += float(pair_difference.sum().item())
            pair_element_count += int(pair_difference.numel())
            pair_abs_max = max(pair_abs_max, float(pair_difference.max().item()))
            clean_abs_sum[NORMALIZE_THEN_PERTURB] += float(
                historical_clean_difference.sum().item()
            )
            clean_abs_sum[PERTURB_THEN_NORMALIZE] += float(
                deployment_clean_difference.sum().item()
            )
            clean_abs_max[NORMALIZE_THEN_PERTURB] = max(
                clean_abs_max[NORMALIZE_THEN_PERTURB],
                float(historical_clean_difference.max().item()),
            )
            clean_abs_max[PERTURB_THEN_NORMALIZE] = max(
                clean_abs_max[PERTURB_THEN_NORMALIZE],
                float(deployment_clean_difference.max().item()),
            )

            batch_target_values = [int(value) for value in batch_targets.tolist()]
            batch_index_values = [int(value) for value in batch_indices.tolist()]
            targets.extend(batch_target_values)
            sample_indices.extend(batch_index_values)
            for position, (sample_index, target) in enumerate(
                zip(batch_index_values, batch_target_values)
            ):
                metadata = dataset.sample_metadata(sample_index)
                historical_prediction = batch_results[NORMALIZE_THEN_PERTURB][0][position]
                deployment_prediction = batch_results[PERTURB_THEN_NORMALIZE][0][position]
                historical_confidence = batch_results[NORMALIZE_THEN_PERTURB][1][position]
                deployment_confidence = batch_results[PERTURB_THEN_NORMALIZE][1][position]
                sample_pair_difference = pair_difference[position]
                sample_historical_clean = historical_clean_difference[position]
                sample_deployment_clean = deployment_clean_difference[position]
                rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "kind": condition.kind,
                        "value": condition.value,
                        "description": condition.description,
                        "sample_index": sample_index,
                        "target": target,
                        "target_label": dataset.label_map[target],
                        "source_path": str(metadata.path),
                        "class_name": metadata.class_name,
                        "split_group": metadata.split_group,
                        "original_filename": metadata.original_filename,
                        "window_index": metadata.window_index,
                        "window_start": metadata.window_index * metadata.step,
                        "normalize_then_perturb_prediction": historical_prediction,
                        "normalize_then_perturb_predicted_label": dataset.label_map[historical_prediction],
                        "normalize_then_perturb_confidence": historical_confidence,
                        "normalize_then_perturb_correct": target == historical_prediction,
                        "perturb_then_normalize_prediction": deployment_prediction,
                        "perturb_then_normalize_predicted_label": dataset.label_map[deployment_prediction],
                        "perturb_then_normalize_confidence": deployment_confidence,
                        "perturb_then_normalize_correct": target == deployment_prediction,
                        "predictions_match": historical_prediction == deployment_prediction,
                        "orders_input_mean_absolute_difference": float(
                            sample_pair_difference.mean().item()
                        ),
                        "orders_input_max_absolute_difference": float(
                            sample_pair_difference.max().item()
                        ),
                        "normalize_then_perturb_vs_clean_mean_absolute_difference": float(
                            sample_historical_clean.mean().item()
                        ),
                        "normalize_then_perturb_vs_clean_max_absolute_difference": float(
                            sample_historical_clean.max().item()
                        ),
                        "perturb_then_normalize_vs_clean_mean_absolute_difference": float(
                            sample_deployment_clean.mean().item()
                        ),
                        "perturb_then_normalize_vs_clean_max_absolute_difference": float(
                            sample_deployment_clean.max().item()
                        ),
                    }
                )

    if sample_indices != list(range(len(dataset))):
        raise TrainingConfigError("test loader changed fixed sample order")

    metrics_by_order = {
        order: _metrics(
            total_loss=order_losses[order],
            targets=targets,
            predictions=order_predictions[order],
            confidences=order_confidences[order],
            label_map=dataset.label_map,
        )
        for order in ORDERS
    }
    disagreements = sum(
        left != right
        for left, right in zip(
            order_predictions[NORMALIZE_THEN_PERTURB],
            order_predictions[PERTURB_THEN_NORMALIZE],
        )
    )
    comparison = {
        "prediction_disagreement_count": disagreements,
        "prediction_disagreement_rate": disagreements / len(dataset),
        "model_input_mean_absolute_difference": pair_abs_sum / pair_element_count,
        "model_input_max_absolute_difference": pair_abs_max,
        "relative_to_clean": {
            order: {
                "model_input_mean_absolute_difference": clean_abs_sum[order]
                / pair_element_count,
                "model_input_max_absolute_difference": clean_abs_max[order],
            }
            for order in ORDERS
        },
    }
    return {
        "condition": asdict(condition),
        "orders": metrics_by_order,
        "comparison": comparison,
    }, rows


def _source_inventory(dataset: CWRUBearingDataset) -> list[dict[str, Any]]:
    by_path: dict[Path, list[Any]] = {}
    for index in range(len(dataset)):
        metadata = dataset.sample_metadata(index)
        by_path.setdefault(metadata.path, []).append(metadata)
    return [
        {
            "path": str(path),
            "original_filename": samples[0].original_filename,
            "class_name": samples[0].class_name,
            "split_group": samples[0].split_group,
            "window_count": len(samples),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path, samples in by_path.items()
    ]


def _integrity_snapshot(
    checkpoint: Path,
    manifest: Path,
    label_map: Path,
    source_files: Iterable[Path],
) -> dict[str, str]:
    paths = [checkpoint, manifest, label_map, *source_files]
    return {str(path): _sha256(path) for path in paths}


def _assert_integrity_unchanged(
    before: dict[str, str], after: dict[str, str]
) -> dict[str, Any]:
    changed = {
        path: {"before": digest, "after": after.get(path)}
        for path, digest in before.items()
        if after.get(path) != digest
    }
    if changed:
        raise TrainingConfigError(
            "audit input files changed during evaluation: "
            + json.dumps(changed, ensure_ascii=False)
        )
    return {
        "passed": True,
        "checked_file_count": len(before),
        "files": {
            path: {"sha256_before": digest, "sha256_after": after[path]}
            for path, digest in before.items()
        },
    }


def _assert_reproducible(
    first: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    second: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for condition_id in first:
        first_summary, first_rows = first[condition_id]
        second_summary, second_rows = second[condition_id]
        summaries_match = first_summary == second_summary
        rows_match = first_rows == second_rows
        details[condition_id] = {
            "passed": summaries_match and rows_match,
            "metrics_predictions_confidences_and_input_differences_match": (
                summaries_match and rows_match
            ),
        }
    if not all(item["passed"] for item in details.values()):
        raise TrainingConfigError(
            "repeated preprocessing-order audit was not deterministic"
        )
    return {
        "passed": True,
        "method": "complete second in-memory evaluation",
        "conditions": details,
    }


def _checkpoint_compatibility(
    config: dict[str, Any], checkpoint_payload: dict[str, Any]
) -> dict[str, Any]:
    checkpoint_config = checkpoint_payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise TrainingConfigError("checkpoint does not contain a configuration object")
    expected = {
        "normalization": config["dataset"]["normalization"],
        "window_size": int(config["dataset"]["window_size"]),
        "step": int(config["dataset"]["step"]),
        "num_classes": int(config["dataset"]["num_classes"]),
    }
    checkpoint_dataset = checkpoint_config.get("dataset", {})
    actual = {
        "normalization": checkpoint_dataset.get("normalization"),
        "window_size": checkpoint_dataset.get("window_size"),
        "step": checkpoint_dataset.get("step"),
        "num_classes": checkpoint_dataset.get("num_classes"),
    }
    if actual != expected:
        raise TrainingConfigError(
            "checkpoint preprocessing/model contract differs from audit config: "
            + json.dumps({"expected": expected, "actual": actual}, ensure_ascii=False)
        )
    return {"passed": True, "expected": expected, "checkpoint": actual}


def _condition_rows(
    conditions: tuple[BearingPerturbation, ...],
    results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        summary = results[condition.condition_id][0]
        for order in ORDERS:
            metrics = summary["orders"][order]
            comparison = summary["comparison"]
            rows.append(
                {
                    "condition_id": condition.condition_id,
                    "kind": condition.kind,
                    "value": condition.value,
                    "description": condition.description,
                    "order": order,
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "mean_confidence": metrics["mean_confidence"],
                    "correct_count": metrics["correct_count"],
                    "failure_count": metrics["failure_count"],
                    "accuracy_drop_from_order_clean": metrics[
                        "accuracy_drop_from_order_clean"
                    ],
                    "macro_f1_drop_from_order_clean": metrics[
                        "macro_f1_drop_from_order_clean"
                    ],
                    "prediction_disagreement_count_between_orders": comparison[
                        "prediction_disagreement_count"
                    ],
                    "prediction_disagreement_rate_between_orders": comparison[
                        "prediction_disagreement_rate"
                    ],
                    "model_input_mean_absolute_difference_between_orders": comparison[
                        "model_input_mean_absolute_difference"
                    ],
                    "model_input_max_absolute_difference_between_orders": comparison[
                        "model_input_max_absolute_difference"
                    ],
                    "model_input_mean_absolute_difference_from_clean": comparison[
                        "relative_to_clean"
                    ][order]["model_input_mean_absolute_difference"],
                    "model_input_max_absolute_difference_from_clean": comparison[
                        "relative_to_clean"
                    ][order]["model_input_max_absolute_difference"],
                }
            )
    return rows


def _write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_outputs_atomically(
    output_paths: dict[str, Path],
    result: dict[str, Any],
    condition_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    _ensure_outputs_absent(output_paths)
    output_directory = output_paths["evidence"].parent
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".preprocess-audit-", dir=output_directory
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        staged = {
            key: temporary_root / path.name for key, path in output_paths.items()
        }
        _write_json(staged["evidence"], result)
        _write_csv(
            staged["conditions"], condition_rows, list(condition_rows[0].keys())
        )
        _write_csv(
            staged["predictions"], prediction_rows, list(prediction_rows[0].keys())
        )
        _ensure_outputs_absent(output_paths)
        for key in ("conditions", "predictions", "evidence"):
            os.replace(staged[key], output_paths[key])


def run_preprocess_order_audit(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    run_name: str,
    evidence_dir: str | Path | None = None,
    conditions: tuple[BearingPerturbation, ...] = DEFAULT_CONDITIONS,
) -> dict[str, Any]:
    project_root = project_root_from_config(config)
    configured_normalization = str(config["dataset"]["normalization"])
    if configured_normalization != "window_zscore":
        raise TrainingConfigError(
            "preprocessing-order audit requires a window_zscore experiment config"
        )
    resolved_evidence_dir = resolve_project_path(
        project_root,
        evidence_dir
        if evidence_dir is not None
        else "artifacts/evidence/bearing/preprocess_audit",
    )
    output_paths = _output_paths(resolved_evidence_dir, run_name)
    _ensure_outputs_absent(output_paths)

    if not conditions or conditions[0].condition_id != "clean":
        raise TrainingConfigError("the first audit condition must be clean")
    condition_ids = [condition.condition_id for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise TrainingConfigError("audit condition IDs must be unique")

    checkpoint = resolve_project_path(project_root, checkpoint_path)
    manifest = resolve_project_path(project_root, config["dataset"]["manifest"])
    label_map_path = resolve_project_path(project_root, config["dataset"]["label_map"])
    for required_path in (checkpoint, manifest, label_map_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"required audit input does not exist: {required_path}")

    seed = int(config["training"]["seed"])
    set_reproducible_seed(seed)
    device = resolve_device(str(config["training"]["device"]))
    dataset = CWRUBearingDataset(
        manifest,
        split="test",
        label_map_path=label_map_path,
        normalization="none",
        project_root=project_root,
        verify_window_counts=True,
    )
    if dataset.window_size != int(config["dataset"]["window_size"]):
        raise TrainingConfigError("test dataset window size differs from config")
    if dataset.step != int(config["dataset"]["step"]):
        raise TrainingConfigError("test dataset step differs from config")
    if dataset.num_classes != int(config["dataset"]["num_classes"]):
        raise TrainingConfigError("test dataset class count differs from config")

    integrity_before = _integrity_snapshot(
        checkpoint, manifest, label_map_path, dataset.source_files
    )
    checkpoint_payload = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    if (
        not isinstance(checkpoint_payload, dict)
        or "model_state_dict" not in checkpoint_payload
    ):
        raise TrainingConfigError(f"unsupported checkpoint format: {checkpoint}")
    compatibility = _checkpoint_compatibility(config, checkpoint_payload)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])

    evaluation_arguments = {
        "normalization": configured_normalization,
        "device": device,
        "seed": seed,
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
    }
    first_results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for condition in conditions:
        print(f"[pass 1/2] auditing {condition.condition_id}", flush=True)
        first_results[condition.condition_id] = _evaluate_condition(
            model, dataset, condition, **evaluation_arguments
        )

    second_results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for condition in conditions:
        print(f"[pass 2/2] verifying {condition.condition_id}", flush=True)
        second_results[condition.condition_id] = _evaluate_condition(
            model, dataset, condition, **evaluation_arguments
        )
    reproducibility = _assert_reproducible(first_results, second_results)

    for order in ORDERS:
        clean_metrics = first_results["clean"][0]["orders"][order]
        for condition in conditions:
            metrics = first_results[condition.condition_id][0]["orders"][order]
            metrics["accuracy_drop_from_order_clean"] = (
                clean_metrics["accuracy"] - metrics["accuracy"]
            )
            metrics["macro_f1_drop_from_order_clean"] = (
                clean_metrics["macro_f1"] - metrics["macro_f1"]
            )
    clean_comparison = first_results["clean"][0]["comparison"]
    if (
        clean_comparison["prediction_disagreement_count"] != 0
        or clean_comparison["model_input_max_absolute_difference"] != 0.0
    ):
        raise TrainingConfigError(
            "clean model inputs differ between preprocessing orders"
        )

    integrity_after_snapshot = _integrity_snapshot(
        checkpoint, manifest, label_map_path, dataset.source_files
    )
    integrity = _assert_integrity_unchanged(
        integrity_before, integrity_after_snapshot
    )
    source_inventory = _source_inventory(dataset)
    environment = collect_environment(device)
    environment.update(
        {
            "python_executable": __import__("sys").executable,
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
        }
    )

    metrics_by_condition = {
        condition.condition_id: first_results[condition.condition_id][0]
        for condition in conditions
    }
    prediction_rows = [
        row
        for condition in conditions
        for row in first_results[condition.condition_id][1]
    ]
    condition_rows = _condition_rows(conditions, first_results)
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_name": run_name,
        "audit_date": "2026-09-03",
        "purpose": (
            "read-only comparison of normalize-then-perturb versus "
            "perturb-then-window-zscore for one frozen bearing CNN"
        ),
        "training_performed": False,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoint_saved": False,
        "checkpoint_modified": False,
        "ui_modified": False,
        "source_data_modified": False,
        "preprocessing_orders": {
            NORMALIZE_THEN_PERTURB: (
                "raw window -> window_zscore -> perturbation -> frozen model"
            ),
            PERTURB_THEN_NORMALIZE: (
                "raw window -> perturbation -> window_zscore -> frozen model"
            ),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256_before": integrity_before[str(checkpoint)],
            "sha256_after": integrity_after_snapshot[str(checkpoint)],
            "hash_unchanged": True,
            "model_name": checkpoint_payload.get("model_name"),
            "training_epoch": checkpoint_payload.get("epoch"),
            "stored_validation_metrics": checkpoint_payload.get(
                "validation_metrics"
            ),
            "compatibility_check": compatibility,
        },
        "config_path": config["_config_path"],
        "seed": seed,
        "split": "test",
        "dataset": {
            "name": config["dataset"]["name"],
            "status": "laboratory test-rig dataset",
            "manifest": str(manifest),
            "manifest_sha256": integrity_before[str(manifest)],
            "label_map_path": str(label_map_path),
            "label_map_sha256": integrity_before[str(label_map_path)],
            "label_map": dataset.label_map,
            "window_count": len(dataset),
            "window_size": dataset.window_size,
            "step": dataset.step,
            "loader_normalization": dataset.normalization,
            "model_input_normalization": configured_normalization,
            "source_files": source_inventory,
        },
        "integrity_check": integrity,
        "reproducibility_check": reproducibility,
        "conditions": metrics_by_condition,
        "environment": environment,
        "interpretation_rules": [
            "Positive non-zero pure gain before window z-score should be almost cancelled by normalization.",
            "Gain applied after window z-score changes model-input scale and is not a faithful sensor-gain test.",
            "AWGN conclusions must be taken from measured paired results rather than assumed from theory.",
        ],
        "limitations": [
            "CWRU is a laboratory bearing test-rig dataset; these results do not establish field or industrial deployment accuracy.",
            "Perturbations are offline mathematical approximations, not a physical sensor or rotating-machine simulation.",
            "Amplitude scaling does not model clipping, quantization, sensor frequency response, mounting differences, or other nonlinear effects.",
            "Only the fixed audited CWRU test split is evaluated.",
            "Unknown faults, open-set rejection, and cross-dataset generalization are not tested.",
        ],
        "output_files": {key: str(path) for key, path in output_paths.items()},
    }

    _write_outputs_atomically(
        output_paths, result, condition_rows, prediction_rows
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/bearing_cnn_window_zscore.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/checkpoints/bearing/window_zscore/"
            "bearing_cnn_window_zscore_20260903_best.pt"
        ),
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("artifacts/evidence/bearing/preprocess_audit"),
    )
    parser.add_argument(
        "--run-name",
        default="bearing_window_zscore_preprocess_order_audit_20260903_v1",
    )
    args = parser.parse_args()
    result = run_preprocess_order_audit(
        load_config(args.config),
        args.checkpoint,
        run_name=args.run_name,
        evidence_dir=args.evidence_dir,
    )
    summary = {
        "run_name": result["run_name"],
        "training_performed": result["training_performed"],
        "checkpoint_unchanged": result["checkpoint"]["hash_unchanged"],
        "source_inputs_unchanged": result["integrity_check"]["passed"],
        "reproducible": result["reproducibility_check"]["passed"],
        "conditions": {
            condition_id: {
                "orders": condition_result["orders"],
                "comparison": condition_result["comparison"],
            }
            for condition_id, condition_result in result["conditions"].items()
        },
        "output_files": result["output_files"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
