"""Calibrate frozen UAV open-set thresholds from known validation recordings only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from airwatch.data import DroneRFWindowDataset
from airwatch.data.data_provenance import sha256_file
from airwatch.inference.uav_open_set import (
    aggregate_recording_scores,
    calibrate_known_acceptance,
    energy_knownness,
    fit_class_prototypes,
    maximum_softmax_knownness,
    prototype_knownness,
)
from training.common import (
    TrainingConfigError,
    collect_environment,
    load_config,
    make_loader,
    resolve_device,
    resolve_project_path,
    set_reproducible_seed,
    write_json,
)
from training.uav_baseline import make_uav_model


def _read_protocol(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError(f"无法读取开放集校准协议 {path}：{exc}") from exc
    required = {
        "protocol", "frozen_on", "model_config", "checkpoint", "prototype_split",
        "calibration_split", "target_known_acceptance", "methods", "evaluation", "output",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise TrainingConfigError(f"开放集校准协议缺少字段：{', '.join(missing)}")
    if payload["prototype_split"] != "train" or payload["calibration_split"] != "validation":
        raise TrainingConfigError("开放集原型只能使用 train，阈值只能使用 validation")
    method_names = [item.get("name") for item in payload["methods"]]
    if method_names != ["maximum_softmax_probability", "negative_energy", "prototype_cosine"]:
        raise TrainingConfigError("首版协议必须按预注册顺序比较 MSP、Energy 和余弦原型")
    return payload


def _extract_outputs(
    model: torch.nn.Module,
    dataset: DroneRFWindowDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if not callable(getattr(model, "forward_features", None)) or not callable(
        getattr(model, "classifier", None)
    ):
        raise TrainingConfigError("开放集校准需要模型提供 forward_features 和 classifier")
    loader = make_loader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed
    )
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    recording_ids: list[str] = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_targets, indices in loader:
            inputs = inputs.to(device)
            batch_features = model.forward_features(inputs)
            batch_logits = model.classifier(batch_features)
            features.append(batch_features.cpu().numpy())
            logits.append(batch_logits.cpu().numpy())
            targets.append(batch_targets.numpy())
            recording_ids.extend(
                dataset.sample_metadata(int(index)).recording_id for index in indices.tolist()
            )
    if not features:
        raise TrainingConfigError(f"{dataset.split} 数据集为空，无法校准")
    return (
        np.concatenate(features),
        np.concatenate(logits),
        np.concatenate(targets),
        tuple(recording_ids),
    )


def _summary(scores: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(scores)),
        "maximum": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "standard_deviation": float(np.std(scores)),
    }


def calibrate(protocol_path: str | Path) -> dict[str, Any]:
    protocol_file = Path(protocol_path).resolve()
    protocol = _read_protocol(protocol_file)
    project_root = Path(__file__).resolve().parents[1]
    model_config_path = resolve_project_path(project_root, protocol["model_config"])
    checkpoint_path = resolve_project_path(project_root, protocol["checkpoint"])
    output_json = resolve_project_path(project_root, protocol["output"]["json"])
    output_csv = resolve_project_path(project_root, protocol["output"]["scores_csv"])
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("拒绝覆盖已有开放集校准证据")

    config = load_config(model_config_path)
    dataset_config = config["dataset"]
    dataset_root = resolve_project_path(project_root, dataset_config["root"])
    common_dataset_args = {
        "normalization": str(dataset_config["normalization"]),
        "band_selection": str(dataset_config.get("band_selection", "both")),
        "cache_recordings": int(dataset_config.get("cache_recordings", 8)),
    }
    train_dataset = DroneRFWindowDataset(dataset_root, split="train", **common_dataset_args)
    validation_dataset = DroneRFWindowDataset(
        dataset_root, split="validation", **common_dataset_args
    )
    if train_dataset.recording_ids & validation_dataset.recording_ids:
        raise TrainingConfigError("训练与验证录制存在交集，拒绝开放集校准")
    if train_dataset.data_identity != validation_dataset.data_identity:
        raise TrainingConfigError("训练与验证数据身份不一致")

    evaluation = protocol["evaluation"]
    seed = int(evaluation["seed"])
    set_reproducible_seed(seed)
    device = resolve_device(str(evaluation["device"]))
    model = make_uav_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != type(model).__name__:
        raise TrainingConfigError("开放集检查点模型类型不匹配")
    if checkpoint.get("data_identity") != train_dataset.data_identity:
        raise TrainingConfigError("开放集检查点与校准数据身份不一致")
    if checkpoint.get("label_map") != train_dataset.label_map:
        raise TrainingConfigError("开放集检查点与校准标签映射不一致")
    model.load_state_dict(checkpoint["model_state_dict"])

    extract_args = {
        "device": device,
        "batch_size": int(evaluation["batch_size"]),
        "num_workers": int(evaluation["num_workers"]),
        "seed": seed,
    }
    train_features, _, train_targets, _ = _extract_outputs(
        model, train_dataset, **extract_args
    )
    validation_features, validation_logits, _, validation_recordings = _extract_outputs(
        model, validation_dataset, **extract_args
    )
    prototypes = fit_class_prototypes(train_features, train_targets)

    method_window_scores = {
        "maximum_softmax_probability": maximum_softmax_knownness(validation_logits),
        "negative_energy": energy_knownness(
            validation_logits, temperature=float(protocol["methods"][1]["temperature"])
        ),
        "prototype_cosine": prototype_knownness(validation_features, prototypes),
    }
    rows: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    target = float(protocol["target_known_acceptance"])
    for method_spec in protocol["methods"]:
        method = str(method_spec["name"])
        recording_order, recording_scores = aggregate_recording_scores(
            method_window_scores[method], validation_recordings
        )
        calibration = calibrate_known_acceptance(
            recording_scores, method=method, target_known_acceptance=target
        )
        accepted = recording_scores >= calibration.threshold
        result: dict[str, Any] = {
            "calibration": calibration.to_dict(),
            "validation_recording_score_summary": _summary(recording_scores),
        }
        if method == "negative_energy":
            result["temperature"] = float(method_spec["temperature"])
        if method == "prototype_cosine":
            result["prototype_source"] = "train_windows_only"
            result["prototype_labels"] = list(prototypes.labels)
            result["prototype_vectors"] = prototypes.vectors.tolist()
        methods.append(result)
        rows.extend({
            "method": method,
            "recording_id": recording_id,
            "knownness_score": float(score),
            "threshold": calibration.threshold,
            "accepted_as_known": bool(is_known),
        } for recording_id, score, is_known in zip(recording_order, recording_scores, accepted))

    report = {
        "schema_version": "1.0",
        "status": "calibrated_known_only",
        "protocol": protocol["protocol"],
        "protocol_path": str(protocol_file),
        "protocol_sha256": sha256_file(protocol_file),
        "protocol_frozen_on": protocol["frozen_on"],
        "environment": collect_environment(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_config_path": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "data_identity": train_dataset.data_identity,
        "label_map": train_dataset.label_map,
        "prototype_split": "train",
        "prototype_window_count": len(train_dataset),
        "calibration_split": "validation",
        "calibration_window_count": len(validation_dataset),
        "calibration_recording_count": len(validation_dataset.recording_ids),
        "test_data_used": False,
        "unknown_data_used": False,
        "methods": methods,
        "known_limitation": (
            "Only known-validation acceptance thresholds are calibrated. No unknown-source "
            "AUROC, FPR95, or rejection rate may be claimed until an independently audited "
            "unknown dataset is evaluated without changing these thresholds."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(output_csv)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    result = calibrate(args.protocol)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
