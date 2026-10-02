"""Evaluate a frozen bearing checkpoint on the fixed test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import (
    build_datasets,
    collect_environment,
    ensure_output_paths_available,
    evaluate_loader,
    load_config,
    make_loader,
    make_model,
    project_root_from_config,
    resolve_device,
    resolve_output_paths,
    resolve_project_path,
    write_json,
    write_predictions_csv,
)


def evaluate(
    config: dict,
    checkpoint_path: str | Path,
    *,
    run_name: str | None = None,
) -> dict:
    project_root = project_root_from_config(config)
    output_paths = resolve_output_paths(config, project_root, run_name)
    device = resolve_device(str(config["training"]["device"]))
    datasets = build_datasets(config, project_root)
    loader = make_loader(
        datasets["test"],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        seed=int(config["training"]["seed"]),
    )
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = resolve_project_path(project_root, checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    # Evaluation reuses the already-created checkpoint and training evidence.
    # Only the two test artifacts are new outputs and must be protected.
    ensure_output_paths_available({
        'test_evidence': output_paths['test_evidence'],
        'test_predictions': output_paths['test_predictions'],
    })
    model = make_model(config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"unsupported checkpoint format: {checkpoint}")
    model.load_state_dict(payload["model_state_dict"])
    metrics, predictions = evaluate_loader(
        model,
        loader,
        device=device,
        num_classes=datasets["test"].num_classes,
    )
    result = {
        "experiment_name": config["experiment_name"],
        "run_name": output_paths["test_evidence"].stem.removesuffix("_test"),
        "checkpoint_path": str(checkpoint),
        "config_path": config["_config_path"],
        "split": "test",
        "metrics": metrics,
        "environment": collect_environment(device),
        "dataset": {
            "manifest": str(datasets["test"].manifest_path),
            "window_count": len(datasets["test"]),
            "window_size": datasets["test"].window_size,
            "step": datasets["test"].step,
            "label_map": datasets["test"].label_map,
        },
    }
    write_json(output_paths["test_evidence"], result)
    write_predictions_csv(output_paths["test_predictions"], predictions)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/bearing_cnn_baseline.json"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        help="safe filename stem for new evidence; existing outputs are never overwritten",
    )
    args = parser.parse_args()
    result = evaluate(load_config(args.config), args.checkpoint, run_name=args.run_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
