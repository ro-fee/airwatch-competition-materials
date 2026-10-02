"""Train the new CWRU one-dimensional CNN baseline.

Run from the project root with ``python -m training.train``.  This module is
not imported by the desktop UI and never writes to the historical ``models/``
directory.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .bearing_augmentation import AugmentedBearingDataset, TrainingAugmentationConfig
from .common import (
    build_datasets,
    collect_environment,
    evaluate_loader,
    load_config,
    make_loader,
    make_model,
    ensure_output_paths_available,
    project_root_from_config,
    resolve_device,
    resolve_output_paths,
    set_reproducible_seed,
    write_json,
)


def train(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    run_name: str | None = None,
) -> dict[str, Any]:
    project_root = project_root_from_config(config)
    output_paths = resolve_output_paths(config, project_root, run_name)
    training_config = config["training"]
    seed = int(training_config["seed"])
    set_reproducible_seed(seed)
    device = resolve_device(str(training_config["device"]))
    augmentation_config = training_config.get("augmentation")
    augmentation_enabled = bool(
        isinstance(augmentation_config, dict)
        and augmentation_config.get("enabled", False)
    )
    datasets = build_datasets(
        config,
        project_root,
        train_normalization="none" if augmentation_enabled else None,
    )
    train_dataset = datasets["train"]
    parsed_augmentation: TrainingAugmentationConfig | None = None
    if augmentation_enabled:
        if not isinstance(augmentation_config, dict):
            raise ValueError("training.augmentation must be an object")
        parsed_augmentation = TrainingAugmentationConfig(
            clean_probability=float(augmentation_config["clean_probability"]),
            awgn_snr_db=tuple(
                float(value) for value in augmentation_config["awgn_snr_db"]
            ),
            seed=seed,
        )
        train_dataset = AugmentedBearingDataset(
            datasets["train"],
            config=parsed_augmentation,
            normalization=str(config["dataset"]["normalization"]),
        )
    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["num_workers"])
    train_loader = make_loader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, seed=seed,
    )
    validation_loader = make_loader(
        datasets["validation"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, seed=seed,
    )
    model = make_model(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    resolved_run_name = output_paths["checkpoint"].stem.removesuffix("_best")
    summary: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "run_name": resolved_run_name,
        "status": "dry_run" if dry_run else "completed",
        "config_path": config["_config_path"],
        "environment": collect_environment(device),
        "dataset": {
            "train_windows": len(datasets["train"]),
            "validation_windows": len(datasets["validation"]),
            "test_windows": len(datasets["test"]),
            "window_size": datasets["train"].window_size,
            "step": datasets["train"].step,
            "label_map": datasets["train"].label_map,
        },
        "model": {
            "name": "BearingCNN",
            "input_shape": [1, datasets["train"].window_size],
            "num_classes": datasets["train"].num_classes,
            "parameter_count": parameter_count,
        },
        "seed": seed,
        "device": str(device),
    }
    summary["training_input"] = {
        "augmentation_enabled": augmentation_enabled,
        "augmentation": (
            {
                "clean_probability": parsed_augmentation.clean_probability,
                "awgn_snr_db": list(parsed_augmentation.awgn_snr_db),
                "seed": parsed_augmentation.seed,
                "normalization_after_augmentation": str(
                    config["dataset"]["normalization"]
                ),
            }
            if parsed_augmentation is not None
            else None
        ),
        "validation_is_augmented": False,
        "test_is_augmented": False,
    }
    if dry_run:
        summary["planned_outputs"] = {
            key: str(path) for key, path in output_paths.items()
        }
        return summary

    # Refuse before training starts so an accidental rerun cannot overwrite
    # the frozen baseline checkpoint or any evidence files.
    ensure_output_paths_available(output_paths)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["epochs"])
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")
    checkpoint_path = output_paths["checkpoint"]
    evidence_dir = output_paths["training_evidence"].parent
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        if parsed_augmentation is not None:
            train_dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_count = 0
        correct = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            count = int(targets.shape[0])
            total_loss += float(loss.item()) * count
            total_count += count
            correct += int((logits.argmax(dim=1) == targets).sum().item())
        validation_metrics, _ = evaluate_loader(
            model, validation_loader, device=device, num_classes=datasets["train"].num_classes
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / total_count,
            "train_accuracy": correct / total_count,
            "validation": validation_metrics,
        }
        history.append(epoch_result)
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "BearingCNN",
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "config": config,
                },
                checkpoint_path,
            )

    summary["status"] = "completed"
    summary["checkpoint_path"] = str(checkpoint_path)
    summary["history"] = history
    write_json(output_paths["training_evidence"], summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/bearing_cnn_baseline.json"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate data/config/model wiring without training or writing weights",
    )
    parser.add_argument(
        "--run-name",
        help="safe filename stem for a new run; existing outputs are never overwritten",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = train(config, dry_run=args.dry_run, run_name=args.run_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

