"""Train and evaluate DroneRF baselines under an explicit input contract."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from airwatch.analysis import ControlledAWGNDataset, DeterministicNoiseAugmentedDataset
from airwatch.data import DroneRFWindowDataset
from airwatch.models import (
    DroneRFCNN,
    DroneRFDualBranch,
    DroneRFResNet18,
    DroneRFSpectralCNN,
    DroneRFTCN,
)
from training.common import (
    TrainingConfigError,
    collect_environment,
    ensure_output_paths_available,
    load_config,
    make_loader,
    project_root_from_config,
    resolve_device,
    resolve_output_paths,
    resolve_project_path,
    set_reproducible_seed,
    write_json,
)


def build_uav_datasets(config: dict[str, Any], project_root: Path) -> dict[str, DroneRFWindowDataset]:
    dataset_config = config["dataset"]
    if dataset_config.get("name") != "DroneRF" or dataset_config.get("target") != "drone_type":
        raise TrainingConfigError("首个无人机基线仅支持 DroneRF drone_type 任务")
    root = resolve_project_path(project_root, dataset_config["root"])
    datasets = {
        split: DroneRFWindowDataset(
            root,
            split=split,
            normalization=str(dataset_config["normalization"]),
            band_selection=str(dataset_config.get("band_selection", "both")),
            cache_recordings=int(dataset_config.get("cache_recordings", 8)),
        )
        for split in ("train", "validation", "test")
    }
    expected_classes = int(dataset_config["num_classes"])
    expected_window = int(dataset_config["window_size"])
    expected_channels = int(dataset_config.get("input_channels", 2))
    expected_semantics = dataset_config.get("input_semantics")
    for split, dataset in datasets.items():
        if (
            dataset.num_classes != expected_classes
            or dataset.window_size != expected_window
            or dataset.channels != expected_channels
        ):
            raise TrainingConfigError(f"{split} 数据结构与配置不一致")
        if expected_semantics is not None and dataset.input_semantics != expected_semantics:
            raise TrainingConfigError(f"{split} 输入语义与配置不一致")
        if dataset.data_identity != datasets["train"].data_identity:
            raise TrainingConfigError("训练、验证和测试没有使用同一数据版本")
    recording_sets = {split: dataset.recording_ids for split, dataset in datasets.items()}
    if (
        recording_sets["train"] & recording_sets["validation"]
        or recording_sets["train"] & recording_sets["test"]
        or recording_sets["validation"] & recording_sets["test"]
    ):
        raise TrainingConfigError("检测到 recording_id 跨集合泄漏")
    return datasets


def make_uav_model(config: dict[str, Any]) -> nn.Module:
    model_name = config.get("model", {}).get("name")
    in_channels = int(config["dataset"].get("input_channels", 2))
    model_types = {
        "DroneRFCNN": DroneRFCNN,
        "DroneRFResNet18": DroneRFResNet18,
        "DroneRFTCN": DroneRFTCN,
        "DroneRFSpectralCNN": DroneRFSpectralCNN,
    }
    if model_name == "DroneRFDualBranch":
        if in_channels != 2:
            raise TrainingConfigError("DroneRFDualBranch 当前仅支持已验证的双频段输入")
        model_config = config["model"]
        return DroneRFDualBranch(
            num_classes=int(config["dataset"]["num_classes"]),
            fusion_mode=str(model_config["fusion_mode"]),
            presence_head=bool(model_config.get("presence_head", False)),
        )
    try:
        model_type = model_types[model_name]
    except KeyError as exc:
        raise TrainingConfigError(f"不支持的无人机基线模型：{model_name}") from exc
    if model_name == "DroneRFSpectralCNN":
        if in_channels != 2:
            raise TrainingConfigError("DroneRFSpectralCNN 当前仅支持已验证的双频段输入")
        return model_type(num_classes=int(config["dataset"]["num_classes"]))
    return model_type(
        num_classes=int(config["dataset"]["num_classes"]),
        in_channels=in_channels,
    )


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    label_map: dict[str, int],
) -> dict[str, Any]:
    labels = list(range(len(label_map)))
    names = {index: name for name, index in label_map.items()}
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean(f1)),
        "sample_count": int(targets.size),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "per_class": {
            names[index]: {
                "label_index": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }


def evaluate_uav_loader(
    model: nn.Module,
    loader: DataLoader,
    dataset: DroneRFWindowDataset,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate windows and mean-probability recording aggregation."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    indices: list[int] = []
    probabilities: list[np.ndarray] = []
    presence_targets: list[int] = []
    presence_predictions: list[int] = []
    presence_probabilities: list[np.ndarray] = []
    fusion_weights: list[float] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for inputs, batch_targets, batch_indices in loader:
            inputs = inputs.to(device, non_blocking=False)
            batch_targets = batch_targets.to(device)
            outputs = model(inputs)
            logits = outputs["type_logits"] if isinstance(outputs, dict) else outputs
            total_loss += float(criterion(logits, batch_targets).item()) * int(batch_targets.shape[0])
            batch_probabilities = torch.softmax(logits, dim=1)
            targets.extend(int(value) for value in batch_targets.cpu().tolist())
            predictions.extend(int(value) for value in batch_probabilities.argmax(dim=1).cpu().tolist())
            indices.extend(int(value) for value in batch_indices.tolist())
            probabilities.extend(batch_probabilities.cpu().numpy())
            if isinstance(outputs, dict):
                batch_presence_targets = (batch_targets != 0).long()
                if "presence_logits" in outputs:
                    batch_presence_probabilities = torch.softmax(outputs["presence_logits"], dim=1)
                    presence_targets.extend(int(value) for value in batch_presence_targets.cpu().tolist())
                    presence_predictions.extend(
                        int(value) for value in batch_presence_probabilities.argmax(dim=1).cpu().tolist()
                    )
                    presence_probabilities.extend(batch_presence_probabilities.cpu().numpy())
                if "fusion_weight_time" in outputs:
                    fusion_weights.extend(
                        float(value) for value in outputs["fusion_weight_time"].detach().cpu().flatten().tolist()
                    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not targets:
        raise TrainingConfigError("不能评估空数据集")

    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    window_metrics = _classification_metrics(
        target_array, prediction_array, label_map=dataset.label_map
    )
    window_metrics["loss"] = total_loss / len(targets)
    window_metrics["evaluation_seconds"] = elapsed
    window_metrics["windows_per_second"] = len(targets) / elapsed

    grouped_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_targets: dict[str, set[int]] = defaultdict(set)
    result_rows: list[dict[str, Any]] = []
    names = {index: name for name, index in dataset.label_map.items()}
    for row_number, (dataset_index, target, prediction, probability) in enumerate(
        zip(indices, targets, predictions, probability_array)
    ):
        sample = dataset.sample_metadata(dataset_index)
        grouped_probabilities[sample.recording_id].append(probability)
        grouped_targets[sample.recording_id].add(target)
        result = {
            "sample_id": sample.sample_id,
            "recording_id": sample.recording_id,
            "target": target,
            "target_name": names[target],
            "prediction": prediction,
            "prediction_name": names[prediction],
        }
        for name, label_index in dataset.label_map.items():
            result[f"prob_{name}"] = float(probability[label_index])
        if presence_probabilities:
            result["prob_drone_present"] = float(presence_probabilities[row_number][1])
            result["presence_prediction"] = presence_predictions[row_number]
        if fusion_weights:
            result["fusion_weight_time"] = fusion_weights[row_number]
        result_rows.append(result)
    recording_targets: list[int] = []
    recording_predictions: list[int] = []
    for recording_id in sorted(grouped_probabilities):
        if len(grouped_targets[recording_id]) != 1:
            raise TrainingConfigError(f"同一录制出现多个目标标签：{recording_id}")
        recording_targets.append(next(iter(grouped_targets[recording_id])))
        mean_probability = np.mean(grouped_probabilities[recording_id], axis=0)
        recording_predictions.append(int(np.argmax(mean_probability)))
    recording_metrics = _classification_metrics(
        np.asarray(recording_targets, dtype=np.int64),
        np.asarray(recording_predictions, dtype=np.int64),
        label_map=dataset.label_map,
    )
    metrics = {
        "window": window_metrics,
        "recording": recording_metrics,
    }
    if presence_targets:
        metrics["presence_window"] = _classification_metrics(
            np.asarray(presence_targets, dtype=np.int64),
            np.asarray(presence_predictions, dtype=np.int64),
            label_map={"background": 0, "drone_present": 1},
        )
        grouped_presence: dict[str, list[np.ndarray]] = defaultdict(list)
        grouped_presence_targets: dict[str, set[int]] = defaultdict(set)
        for dataset_index, target, probability in zip(
            indices, presence_targets, presence_probabilities
        ):
            recording_id = dataset.sample_metadata(dataset_index).recording_id
            grouped_presence[recording_id].append(probability)
            grouped_presence_targets[recording_id].add(target)
        recording_presence_targets, recording_presence_predictions = [], []
        for recording_id in sorted(grouped_presence):
            recording_presence_targets.append(next(iter(grouped_presence_targets[recording_id])))
            recording_presence_predictions.append(
                int(np.argmax(np.mean(grouped_presence[recording_id], axis=0)))
            )
        metrics["presence_recording"] = _classification_metrics(
            np.asarray(recording_presence_targets, dtype=np.int64),
            np.asarray(recording_presence_predictions, dtype=np.int64),
            label_map={"background": 0, "drone_present": 1},
        )
    if fusion_weights:
        metrics["fusion_weight_time"] = {
            "mean": float(np.mean(fusion_weights)),
            "standard_deviation": float(np.std(fusion_weights)),
            "minimum": float(np.min(fusion_weights)),
            "maximum": float(np.max(fusion_weights)),
        }
    return metrics, result_rows


def _class_weights(dataset: DroneRFWindowDataset, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    counts = Counter(sample.label_index for sample in dataset.samples)
    total = len(dataset)
    weights = [total / (dataset.num_classes * counts[index]) for index in range(dataset.num_classes)]
    names = {index: name for name, index in dataset.label_map.items()}
    return (
        torch.tensor(weights, dtype=torch.float32, device=device),
        {names[index]: float(weights[index]) for index in range(dataset.num_classes)},
    )


def _write_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_dataset_for_epoch(
    dataset: DroneRFWindowDataset, config: dict[str, Any], epoch: int
) -> DroneRFWindowDataset | DeterministicNoiseAugmentedDataset:
    augmentation = config.get("training_augmentation")
    if augmentation is None:
        return dataset
    if augmentation.get("name") != "deterministic_mixed_awgn_after_normalization":
        raise TrainingConfigError("不支持的无人机训练增强方式")
    try:
        return DeterministicNoiseAugmentedDataset(
            dataset,
            snr_db_levels=augmentation["snr_db_levels"],
            noise_probability=float(augmentation["noise_probability"]),
            base_seed=int(augmentation["noise_seed"]),
            epoch=epoch,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingConfigError(f"无人机训练增强配置无效：{exc}") from exc


def _evaluate_validation_selection(
    model: nn.Module,
    dataset: DroneRFWindowDataset,
    config: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, Any], float, float, bool]:
    robustness = config.get("validation_robustness")
    clean_loader = make_loader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, seed=seed,
    )
    clean_metrics, _ = evaluate_uav_loader(model, clean_loader, dataset, device=device)
    if robustness is None:
        return (
            clean_metrics,
            float(clean_metrics["recording"]["macro_f1"]),
            float(clean_metrics["window"]["loss"]),
            True,
        )
    if robustness.get("selection_metric") != "mean_noisy_recording_macro_f1":
        raise TrainingConfigError("不支持的鲁棒性验证选择指标")
    levels = robustness.get("snr_db_levels")
    if not isinstance(levels, list) or not levels:
        raise TrainingConfigError("鲁棒性验证必须声明非空 snr_db_levels")
    conditions = [{"condition": "clean", "snr_db": None, "metrics": clean_metrics}]
    noisy_scores = []
    for snr_db in levels:
        noisy_dataset = ControlledAWGNDataset(
            dataset, snr_db=float(snr_db), base_seed=int(robustness["noise_seed"])
        )
        loader = make_loader(
            noisy_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, seed=seed,
        )
        metrics, _ = evaluate_uav_loader(model, loader, noisy_dataset, device=device)
        conditions.append({
            "condition": f"{float(snr_db):g}_db", "snr_db": float(snr_db),
            "metrics": metrics,
        })
        noisy_scores.append(float(metrics["recording"]["macro_f1"]))
    selection_score = float(np.mean(noisy_scores))
    clean_score = float(clean_metrics["recording"]["macro_f1"])
    guardrail_minimum = float(robustness["clean_guardrail_min_recording_macro_f1"])
    guardrail_passed = clean_score >= guardrail_minimum
    return ({
        "conditions": conditions,
        "selection": {
            "metric": robustness["selection_metric"],
            "score": selection_score,
            "clean_recording_macro_f1": clean_score,
            "clean_guardrail_minimum": guardrail_minimum,
            "clean_guardrail_passed": guardrail_passed,
        },
    }, selection_score, float(clean_metrics["window"]["loss"]), guardrail_passed)


def train_uav(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    run_name: str | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    project_root = project_root_from_config(config)
    paths = resolve_output_paths(config, project_root, run_name)
    training_config = config["training"]
    seed = int(training_config["seed"])
    set_reproducible_seed(seed)
    device = resolve_device(str(training_config["device"]))
    datasets = build_uav_datasets(config, project_root)
    model = make_uav_model(config).to(device)
    model_name = type(model).__name__
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    class_weights, class_weight_map = _class_weights(datasets["train"], device)
    presence_counts = Counter(0 if sample.label_index == 0 else 1 for sample in datasets["train"].samples)
    presence_class_weights = torch.tensor(
        [len(datasets["train"]) / (2 * presence_counts[index]) for index in (0, 1)],
        dtype=torch.float32,
        device=device,
    )
    presence_loss_weight = float(config.get("model", {}).get("presence_loss_weight", 0.0))
    _training_dataset_for_epoch(datasets["train"], config, 1)
    run = paths["checkpoint"].stem.removesuffix("_best")
    last_checkpoint = paths["checkpoint"].with_name(f"{run}_last.pt")
    resolved_config_path = paths["training_evidence"].with_name(f"{run}_config.json")
    summary: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "run_name": run,
        "status": "dry_run" if dry_run else "running",
        "config_path": config["_config_path"],
        "environment": collect_environment(device),
        "seed": seed,
        "dataset": {
            "train_windows": len(datasets["train"]),
            "validation_windows": len(datasets["validation"]),
            "test_windows": len(datasets["test"]),
            "train_recordings": len(datasets["train"].recording_ids),
            "validation_recordings": len(datasets["validation"].recording_ids),
            "test_recordings": len(datasets["test"].recording_ids),
            "window_size": datasets["train"].window_size,
            "channels": datasets["train"].channels,
            "normalization": datasets["train"].normalization,
            "label_map": datasets["train"].label_map,
            "data_identity": datasets["train"].data_identity,
            "known_limitation": datasets["train"].metadata["known_limitation"],
        },
        "model": {
            "name": model_name,
            "input_shape": [datasets["train"].channels, datasets["train"].window_size],
            "num_classes": datasets["train"].num_classes,
            "parameter_count": parameter_count,
        },
        "class_weights": class_weight_map,
        "auxiliary_presence": {
            "enabled": bool(config.get("model", {}).get("presence_head", False)),
            "loss_weight": presence_loss_weight,
            "class_weights": {
                "background": float(presence_class_weights[0].item()),
                "drone_present": float(presence_class_weights[1].item()),
            },
        },
        "training_augmentation": config.get("training_augmentation"),
        "validation_robustness": config.get("validation_robustness"),
        "selection_rule": (
            "maximum mean noisy validation recording macro_f1 among checkpoints meeting "
            "the frozen clean validation guardrail; no test evaluation"
            if config.get("validation_robustness") else
            "maximum validation recording macro_f1; test unseen until frozen evaluation"
        ),
    }
    if dry_run:
        summary["planned_outputs"] = {
            **{key: str(value) for key, value in paths.items()},
            "last_checkpoint": str(last_checkpoint),
            "resolved_config": str(resolved_config_path),
        }
        return summary

    ensure_output_paths_available(paths)
    if last_checkpoint.exists():
        raise TrainingConfigError(f"拒绝覆盖已有恢复点：{last_checkpoint}")
    if resolved_config_path.exists():
        raise TrainingConfigError(f"拒绝覆盖已有生效配置：{resolved_config_path}")
    resolved_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    resolved_config["output"] = dict(resolved_config["output"])
    resolved_config["output"]["run_name"] = run
    write_json(resolved_config_path, resolved_config)
    summary["resolved_config_path"] = str(resolved_config_path)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    presence_criterion = nn.CrossEntropyLoss(weight=presence_class_weights)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = -1.0
    best_loss = float("inf")
    best_guardrail_passed = False
    stale_epochs = 0
    if resume_checkpoint is not None:
        resume_path = resolve_project_path(project_root, resume_checkpoint)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if payload.get("data_identity") != datasets["train"].data_identity:
            raise TrainingConfigError("恢复点的数据身份与当前 DroneRF 数据不一致")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload.get("history", []))
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload.get("best_score", -1.0))
        best_loss = float(payload.get("best_loss", float("inf")))
        stale_epochs = int(payload.get("stale_epochs", 0))
        best_guardrail_passed = bool(payload.get("best_guardrail_passed", False))

    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    if start_epoch > epochs:
        raise TrainingConfigError("恢复点已达到或超过配置的总轮数")
    for epoch in range(start_epoch, epochs + 1):
        epoch_train_dataset = _training_dataset_for_epoch(datasets["train"], config, epoch)
        train_loader = make_loader(
            epoch_train_dataset, batch_size=int(training_config["batch_size"]),
            shuffle=True, num_workers=int(training_config["num_workers"]), seed=seed + epoch,
        )
        model.train()
        total_loss = 0.0
        total_count = 0
        correct = 0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            logits = outputs["type_logits"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, targets)
            if isinstance(outputs, dict) and "presence_logits" in outputs:
                if presence_loss_weight <= 0:
                    raise TrainingConfigError("presence_head 已启用但 presence_loss_weight 不是正数")
                loss = loss + presence_loss_weight * presence_criterion(
                    outputs["presence_logits"], (targets != 0).long()
                )
            loss.backward()
            optimizer.step()
            count = int(targets.shape[0])
            total_loss += float(loss.item()) * count
            total_count += count
            correct += int((logits.argmax(dim=1) == targets).sum().item())
        validation_metrics, score, validation_loss, guardrail_passed = _evaluate_validation_selection(
            model, datasets["validation"], config, device=device,
            batch_size=int(training_config["batch_size"]),
            num_workers=int(training_config["num_workers"]), seed=seed,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / total_count,
            "train_accuracy": correct / total_count,
            "validation": validation_metrics,
        }
        history.append(epoch_result)
        improved = (
            (guardrail_passed and not best_guardrail_passed)
            or (
                guardrail_passed == best_guardrail_passed
                and (score > best_score or (score == best_score and validation_loss < best_loss))
            )
        )
        if improved:
            best_score, best_loss, stale_epochs = score, validation_loss, 0
            best_guardrail_passed = guardrail_passed
            paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_name": model_name,
                "epoch": epoch,
                "best_score": best_score,
                "best_loss": best_loss,
                "stale_epochs": stale_epochs,
                "best_guardrail_passed": best_guardrail_passed,
                "history": history,
                "config": config,
                "label_map": datasets["train"].label_map,
                "data_identity": datasets["train"].data_identity,
            }, paths["checkpoint"])
        else:
            stale_epochs += 1
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": model_name,
            "epoch": epoch,
            "best_score": best_score,
            "best_loss": best_loss,
            "stale_epochs": stale_epochs,
            "best_guardrail_passed": best_guardrail_passed,
            "history": history,
            "config": config,
            "label_map": datasets["train"].label_map,
            "data_identity": datasets["train"].data_identity,
        }, last_checkpoint)
        print(
            f"epoch={epoch} train_loss={epoch_result['train_loss']:.6f} "
            f"val_selection_score={score:.6f} val_clean_loss={validation_loss:.6f} "
            f"clean_guardrail={'pass' if guardrail_passed else 'fail'}",
            flush=True,
        )
        if stale_epochs >= patience:
            break
    summary.update({
        "status": "completed",
        "checkpoint_path": str(paths["checkpoint"]),
        "last_checkpoint_path": str(last_checkpoint),
        "best_validation_window_loss": best_loss,
        "best_validation_clean_guardrail_passed": best_guardrail_passed,
        "history": history,
        "completed_epochs": len(history),
    })
    if config.get("validation_robustness"):
        summary["best_validation_robustness_mean_recording_macro_f1"] = best_score
    else:
        summary["best_validation_recording_macro_f1"] = best_score
    write_json(paths["training_evidence"], summary)
    return summary


def evaluate_uav(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    run_name: str | None = None,
) -> dict[str, Any]:
    project_root = project_root_from_config(config)
    paths = resolve_output_paths(config, project_root, run_name)
    ensure_output_paths_available({
        "test_evidence": paths["test_evidence"],
        "test_predictions": paths["test_predictions"],
    })
    device = resolve_device(str(config["training"]["device"]))
    datasets = build_uav_datasets(config, project_root)
    model = make_uav_model(config).to(device)
    model_name = type(model).__name__
    checkpoint = resolve_project_path(project_root, checkpoint_path)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("model_name") != model_name:
        raise TrainingConfigError("检查点模型类型不匹配")
    if payload.get("data_identity") != datasets["test"].data_identity:
        raise TrainingConfigError("检查点数据身份与固定测试集不一致")
    if payload.get("label_map") != datasets["test"].label_map:
        raise TrainingConfigError("检查点标签映射与测试集不一致")
    model.load_state_dict(payload["model_state_dict"])
    loader = make_loader(
        datasets["test"], batch_size=int(config["training"]["batch_size"]),
        shuffle=False, num_workers=int(config["training"]["num_workers"]),
        seed=int(config["training"]["seed"]),
    )
    metrics, rows = evaluate_uav_loader(model, loader, datasets["test"], device=device)
    result = {
        "experiment_name": config["experiment_name"],
        "run_name": paths["test_evidence"].stem.removesuffix("_test"),
        "status": "completed",
        "split": "test",
        "checkpoint_path": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "config_path": config["_config_path"],
        "environment": collect_environment(device),
        "data_identity": datasets["test"].data_identity,
        "label_map": datasets["test"].label_map,
        "metrics": metrics,
        "known_limitation": datasets["test"].metadata["known_limitation"],
    }
    write_json(paths["test_evidence"], result)
    _write_prediction_rows(paths["test_predictions"], rows)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--run-name")
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument("--resume", type=Path)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "train":
        result = train_uav(
            config, dry_run=args.dry_run, run_name=args.run_name,
            resume_checkpoint=args.resume,
        )
    else:
        result = evaluate_uav(config, args.checkpoint, run_name=args.run_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
