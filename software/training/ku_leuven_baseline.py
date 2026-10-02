"""Train KU Leuven known-source baselines without opening test or unknown data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

from airwatch.data import KULeuvenMaterializedDataset
from airwatch.inference import aggregate_recording_probabilities
from airwatch.models import DroneRFResNet18, DroneRFTCN
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
from training.ku_leuven_checkpoint_selection import (
    SELECTION_RULE_DESCRIPTION,
    is_checkpoint_selection_improvement,
    next_stale_epochs,
    summarize_checkpoint_selection,
    validate_checkpoint_selection,
)


EXPECTED_WINDOWS_PER_RECORDING = 32


def build_ku_leuven_datasets(
    config: dict[str, Any],
    project_root: Path,
    *,
    dataset_root_override: str | Path | None = None,
) -> dict[str, KULeuvenMaterializedDataset]:
    dataset_config = config["dataset"]
    if dataset_config.get("name") != "KU Leuven Drone RF Dataset":
        raise TrainingConfigError("该入口只支持 KU Leuven Drone RF Dataset")
    if dataset_config.get("target") != "known_source_classification":
        raise TrainingConfigError("目标必须是 known_source_classification")
    root = (
        Path(dataset_root_override).resolve()
        if dataset_root_override is not None
        else resolve_project_path(project_root, dataset_config["root"])
    )
    datasets = {
        split: KULeuvenMaterializedDataset(root, split=split, verify_hashes=True)
        for split in ("train", "validation")
    }
    try:
        expected = {
            "num_classes": 3,
            "input_channels": 2,
            "window_size": 4096,
            "windows_per_recording": EXPECTED_WINDOWS_PER_RECORDING,
            "preprocessing_id": "ku-leuven-iq-dc-rms-v1",
        }
        for key, value in expected.items():
            if dataset_config.get(key) != value:
                raise TrainingConfigError(f"数据配置 {key} 必须为 {value!r}")
        if datasets["train"].data_identity != datasets["validation"].data_identity:
            raise TrainingConfigError("训练与验证数据身份不一致")
        if datasets["train"].recording_ids & datasets["validation"].recording_ids:
            raise TrainingConfigError("训练与验证 recording_id 存在泄漏")
        for split, dataset in datasets.items():
            counts = Counter(sample.recording_id for sample in dataset.samples)
            if set(counts.values()) != {EXPECTED_WINDOWS_PER_RECORDING}:
                raise TrainingConfigError(f"{split} 录制窗口数与冻结协议不一致")
        return datasets
    except Exception:
        for dataset in datasets.values():
            dataset.close()
        raise


def make_ku_leuven_model(config: dict[str, Any]) -> nn.Module:
    name = config.get("model", {}).get("name")
    model_types = {
        "DroneRFTCN": DroneRFTCN,
        "DroneRFResNet18": DroneRFResNet18,
    }
    try:
        model_type = model_types[name]
    except KeyError as exc:
        raise TrainingConfigError(f"不支持的 KU Leuven 基线模型：{name}") from exc
    return model_type(num_classes=3, in_channels=2)


def _metrics(targets: np.ndarray, predictions: np.ndarray, label_map: dict[str, int]) -> dict[str, Any]:
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
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }


def evaluate_validation(
    model: nn.Module,
    dataset: KULeuvenMaterializedDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, Any]:
    loader = make_loader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    recording_ids: list[str] = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_targets, indices in loader:
            inputs = inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(inputs)
            total_loss += float(criterion(logits, batch_targets).item()) * len(batch_targets)
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
            targets.extend(int(value) for value in batch_targets.cpu().tolist())
            recording_ids.extend(
                dataset.sample_metadata(int(index)).recording_id for index in indices.tolist()
            )
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if target_array.size == 0:
        raise TrainingConfigError("验证集为空")
    window_predictions = probability_array.argmax(axis=1)
    order, recording_probabilities, recording_targets = aggregate_recording_probabilities(
        probability_array,
        recording_ids,
        target_array,
        expected_windows_per_recording=EXPECTED_WINDOWS_PER_RECORDING,
    )
    recording_predictions = recording_probabilities.argmax(axis=1)
    return {
        "loss": total_loss / target_array.size,
        "window": _metrics(target_array, window_predictions, dataset.label_map),
        "recording": _metrics(recording_targets, recording_predictions, dataset.label_map),
        "recording_aggregation": {
            "protocol": "mean-window-softmax-probability-v1",
            "windows_per_recording": EXPECTED_WINDOWS_PER_RECORDING,
            "recording_count": len(order),
            "confidence_definition": "maximum_of_mean_class_probability",
        },
    }


def _class_weights(dataset: KULeuvenMaterializedDataset, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    counts = Counter(sample.label_index for sample in dataset.samples)
    total = len(dataset)
    weights = [total / (dataset.num_classes * counts[index]) for index in range(dataset.num_classes)]
    names = {index: name for name, index in dataset.label_map.items()}
    return (
        torch.tensor(weights, dtype=torch.float32, device=device),
        {names[index]: float(weights[index]) for index in range(dataset.num_classes)},
    )


def train_ku_leuven(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    run_name: str | None = None,
    dataset_root_override: str | Path | None = None,
) -> dict[str, Any]:
    project_root = project_root_from_config(config)
    paths = resolve_output_paths(config, project_root, run_name)
    training = config["training"]
    augmentation = config.get("augmentation")
    checkpoint_selection = validate_checkpoint_selection(
        config.get("checkpoint_selection")
    )
    if augmentation is not None:
        from training.ku_leuven_augmentation import PROTOCOLS
        if (
            set(augmentation) != {"protocol"}
            or augmentation.get("protocol") not in PROTOCOLS
        ):
            raise TrainingConfigError("unsupported IQ augmentation protocol")
    aggregation = config.get("aggregation", {})
    if (
        aggregation.get("protocol") != "mean-window-softmax-probability-v1"
        or aggregation.get("windows_per_recording") != EXPECTED_WINDOWS_PER_RECORDING
        or aggregation.get("open_set_target_known_acceptance") != 0.95
        or aggregation.get("threshold_selection_split") != "validation"
    ):
        raise TrainingConfigError("聚合或开放集阈值规则与冻结协议不一致")
    seed = int(training["seed"])
    set_reproducible_seed(seed)
    device = resolve_device(str(training["device"]))
    datasets = build_ku_leuven_datasets(
        config, project_root, dataset_root_override=dataset_root_override
    )
    try:
        model = make_ku_leuven_model(config).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        class_weights, class_weight_map = _class_weights(datasets["train"], device)
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
                "test_windows_opened": False,
                "unknown_windows_opened": False,
                "train_recordings": len(datasets["train"].recording_ids),
                "validation_recordings": len(datasets["validation"].recording_ids),
                "window_size": datasets["train"].window_size,
                "channels": datasets["train"].channels,
                "label_map": datasets["train"].label_map,
                "data_identity": datasets["train"].data_identity,
            },
            "model": {
                "name": type(model).__name__,
                "input_shape": [2, 4096],
                "num_classes": 3,
                "parameter_count": parameter_count,
            },
            "class_weights": class_weight_map,
            "selection_rule": (
                SELECTION_RULE_DESCRIPTION
                if checkpoint_selection is not None
                else "maximum validation recording macro_f1; tie-break minimum validation window loss; test and unknown unopened"
            ),
            "aggregation_protocol": "mean-window-softmax-probability-v1",
        }
        if dry_run:
            summary["planned_outputs"] = {
                "checkpoint": str(paths["checkpoint"]),
                "last_checkpoint": str(last_checkpoint),
                "training_evidence": str(paths["training_evidence"]),
                "resolved_config": str(resolved_config_path),
            }
            return summary

        ensure_output_paths_available(paths)
        if last_checkpoint.exists() or resolved_config_path.exists():
            raise TrainingConfigError("拒绝覆盖已有 KU Leuven 实验产物")
        resolved_config = {key: value for key, value in config.items() if not key.startswith("_")}
        resolved_config["output"] = dict(resolved_config["output"])
        resolved_config["output"]["run_name"] = run
        write_json(resolved_config_path, resolved_config)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        epochs = int(training["epochs"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        amp_enabled = bool(training.get("amp", False)) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        history = []
        best_score = -1.0
        best_loss = float("inf")
        best_checkpoint_selection = None
        stale_epochs = 0
        patience = int(training["early_stopping_patience"])
        for epoch in range(1, epochs + 1):
            epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
            training_dataset = datasets["train"]
            if augmentation is not None:
                from training.ku_leuven_augmentation import AugmentedIQTrainingDataset
                training_dataset = AugmentedIQTrainingDataset(
                    training_dataset,
                    seed=seed,
                    epoch=epoch,
                    protocol=augmentation["protocol"],
                )
            loader = make_loader(
                training_dataset,
                batch_size=int(training["batch_size"]),
                shuffle=True,
                num_workers=int(training["num_workers"]),
                seed=seed + epoch,
            )
            model.train()
            total_loss = 0.0
            total_count = 0
            correct = 0
            for inputs, targets, _ in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                count = len(targets)
                total_loss += float(loss.item()) * count
                total_count += count
                correct += int((logits.argmax(dim=1) == targets).sum().item())
            validation = evaluate_validation(
                model,
                datasets["validation"],
                device=device,
                batch_size=int(training["batch_size"]),
                num_workers=int(training["num_workers"]),
                seed=seed,
            )
            score = float(validation["recording"]["macro_f1"])
            validation_loss = float(validation["loss"])
            result = {
                "epoch": epoch,
                "learning_rate": epoch_learning_rate,
                "train_loss": total_loss / total_count,
                "train_accuracy": correct / total_count,
                "validation": validation,
            }
            selection_summary = None
            if checkpoint_selection is not None:
                from airwatch.analysis.iq_perturbations import (
                    DeterministicIQPerturbationDataset,
                )

                stress_validations = []
                for repeat_seed in checkpoint_selection["repeat_seeds"]:
                    stress_dataset = DeterministicIQPerturbationDataset(
                        datasets["validation"],
                        perturbation={
                            "kind": "awgn",
                            "snr_db": checkpoint_selection["awgn_snr_db"],
                        },
                        base_seed=repeat_seed,
                    )
                    stress_validations.append(
                        evaluate_validation(
                            model,
                            stress_dataset,
                            device=device,
                            batch_size=int(training["batch_size"]),
                            num_workers=int(training["num_workers"]),
                            seed=seed,
                        )
                    )
                selection_summary = summarize_checkpoint_selection(
                    validation, stress_validations, checkpoint_selection
                )
                result["checkpoint_selection"] = selection_summary
            history.append(result)
            scheduler.step()
            improved = (
                is_checkpoint_selection_improvement(
                    selection_summary, best_checkpoint_selection
                )
                if selection_summary is not None
                else score > best_score
                or (score == best_score and validation_loss < best_loss)
            )
            if improved:
                best_score, best_loss, stale_epochs = score, validation_loss, 0
                best_checkpoint_selection = selection_summary
                paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_name": type(model).__name__,
                    "epoch": epoch,
                    "label_map": datasets["train"].label_map,
                    "data_identity": datasets["train"].data_identity,
                    "aggregation_protocol": "mean-window-softmax-probability-v1",
                    "best_validation_recording_macro_f1": best_score,
                    "checkpoint_selection": best_checkpoint_selection,
                    "config": resolved_config,
                }, paths["checkpoint"])
            else:
                stale_epochs = (
                    next_stale_epochs(
                        stale_epochs,
                        improved=False,
                        has_eligible_checkpoint=best_checkpoint_selection is not None,
                    )
                    if checkpoint_selection is not None
                    else stale_epochs + 1
                )
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_name": type(model).__name__,
                "epoch": epoch,
                "label_map": datasets["train"].label_map,
                "data_identity": datasets["train"].data_identity,
                "checkpoint_selection": selection_summary,
                "history": history,
            }, last_checkpoint)
            stress_text = ""
            if selection_summary is not None:
                stress_text = (
                    " minus5_worst="
                    f"{selection_summary['worst_repeat_recording_macro_f1']:.6f}"
                    " minus5_mean="
                    f"{selection_summary['mean_repeat_recording_macro_f1']:.6f}"
                    f" eligible={selection_summary['eligible']}"
                )
            print(
                f"epoch={epoch} train_loss={result['train_loss']:.6f} "
                f"val_recording_macro_f1={score:.6f} val_loss={validation_loss:.6f}"
                f"{stress_text}",
                flush=True,
            )
            if stale_epochs >= patience:
                break
        summary.update({
            "status": "completed_engineering_smoke" if bool(training.get("engineering_smoke")) else "completed_training_validation_only",
            "checkpoint_path": str(paths["checkpoint"]),
            "last_checkpoint_path": str(last_checkpoint),
            "resolved_config_path": str(resolved_config_path),
            "best_validation_recording_macro_f1": best_score,
            "best_validation_window_loss": best_loss,
            "completed_epochs": len(history),
            "history": history,
            "validation_model_selection_evidence_allowed": not bool(training.get("engineering_smoke")),
            "competition_test_claim_allowed": False,
            "test_data_used": False,
            "unknown_data_used": False,
        })
        if checkpoint_selection is not None:
            if best_checkpoint_selection is None:
                raise TrainingConfigError(
                    "训练结束时没有检查点通过 clean Macro-F1 守门线"
                )
            summary["best_checkpoint_selection"] = best_checkpoint_selection
        write_json(paths["training_evidence"], summary)
        return summary
    finally:
        for dataset in datasets.values():
            dataset.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = train_ku_leuven(
        load_config(args.config),
        dry_run=args.dry_run,
        run_name=args.run_name,
        dataset_root_override=args.dataset_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_WINDOWS_PER_RECORDING",
    "build_ku_leuven_datasets",
    "evaluate_validation",
    "make_ku_leuven_model",
    "train_ku_leuven",
]
