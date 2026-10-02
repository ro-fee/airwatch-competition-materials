"""Evaluate frozen local KU Leuven TCN checkpoints on known validation stressors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

from airwatch.analysis.iq_perturbations import (
    DeterministicIQPerturbationDataset,
    IQPerturbationSpec,
    MULTIPATH_PROFILES,
)
from airwatch.data import KULeuvenMaterializedDataset
from airwatch.data.data_provenance import sha256_file
from airwatch.inference import aggregate_recording_probabilities
from airwatch.models import DroneRFTCN
from training.common import (
    TrainingConfigError,
    collect_environment,
    load_config,
    make_loader,
    resolve_device,
    resolve_project_path,
    set_reproducible_seed,
)


EXPECTED_PROTOCOL = "ku-leuven-tcn-local-known-validation-robustness-v1"
EXPECTED_SEEDS = [20260909, 20260910, 20260911]
EXPECTED_REPEAT_SEEDS = [2026090901, 2026090902, 2026090903]
EXPECTED_AWGN_LEVELS = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
EXPECTED_CFO_LEVELS = [-100000, -50000, -25000, 25000, 50000, 100000]
EXPECTED_WINDOWS_PER_RECORDING = 32
ENSEMBLE_ID = "tcn_three_seed_probability_ensemble"


@dataclass(frozen=True)
class EnsembleMember:
    predictor_id: str
    seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    model: nn.Module
    source: dict[str, Any]


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError(f"无法读取{description}：{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingConfigError(f"{description}必须是 JSON 对象：{path}")
    return payload


def _verify_hash(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{description}不存在：{path}")
    actual = sha256_file(path)
    if actual != expected:
        raise TrainingConfigError(
            f"{description} SHA-256 偏离冻结协议：{path}; expected={expected}; actual={actual}"
        )
    return actual


def _validate_multipath_profiles(protocol: dict[str, Any]) -> None:
    registered = protocol["perturbations"]["multipath"]["profiles"]
    if set(registered) != set(MULTIPATH_PROFILES):
        raise TrainingConfigError("协议与实现的多径 profile 集合不一致")
    for name, expected in registered.items():
        implemented_taps = MULTIPATH_PROFILES[name]["taps"]
        delays = [int(item["delay_samples"]) for item in implemented_taps]
        amplitudes = [float(item["relative_amplitude_db"]) for item in implemented_taps]
        if delays != expected["delay_samples"] or not np.allclose(
            amplitudes,
            expected["relative_amplitude_db"],
            rtol=0.0,
            atol=1e-6,
        ):
            raise TrainingConfigError(f"多径 profile {name} 的实现偏离冻结协议")


def load_protocol(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    protocol = _read_json(source, "KU Leuven 鲁棒性协议")
    if protocol.get("protocol") != EXPECTED_PROTOCOL:
        raise TrainingConfigError("KU Leuven 鲁棒性协议名称不匹配")
    if protocol.get("split") != "validation":
        raise TrainingConfigError("鲁棒性开发只允许 validation 划分")
    access = protocol.get("data_access", {})
    required_false = (
        "test_data_allowed",
        "unknown_data_allowed",
        "open_set_threshold_calibration_allowed",
        "competition_test_claim_allowed",
        "a800_protocol_replaced",
    )
    if access.get("development_validation_only") is not True or any(
        access.get(name) is not False for name in required_false
    ):
        raise TrainingConfigError("协议的数据封存或声明边界不满足要求")
    dataset = protocol.get("dataset", {})
    if (
        dataset.get("sample_rate_hz") != 100_000_000
        or dataset.get("window_samples") != 4096
        or dataset.get("windows_per_recording") != EXPECTED_WINDOWS_PER_RECORDING
        or dataset.get("expected_validation_windows") != 768
        or dataset.get("expected_validation_recordings") != 24
        or dataset.get("preprocessing_id") != "ku-leuven-iq-dc-rms-v1"
        or dataset.get("post_perturbation_preprocessing_id")
        != "ku-leuven-iq-dc-rms-v1"
    ):
        raise TrainingConfigError("协议的数据或预处理契约不匹配")
    models = protocol.get("models", [])
    if [item.get("seed") for item in models] != EXPECTED_SEEDS:
        raise TrainingConfigError("协议必须按顺序包含三个冻结 TCN 种子")
    if any(item.get("model") != "DroneRFTCN" for item in models):
        raise TrainingConfigError("协议只允许三个 DroneRFTCN 检查点")
    if len({item.get("id") for item in models}) != 3:
        raise TrainingConfigError("协议中的 TCN predictor id 必须唯一")
    ensemble = protocol.get("ensemble", {})
    if (
        ensemble.get("id") != ENSEMBLE_ID
        or ensemble.get("method")
        != "arithmetic_mean_of_window_softmax_probabilities"
        or ensemble.get("logit_averaging_allowed") is not False
        or ensemble.get("majority_voting_allowed") is not False
    ):
        raise TrainingConfigError("三种子集成协议不匹配")
    perturbations = protocol.get("perturbations", {})
    if (
        perturbations.get("single_factor_only") is not True
        or perturbations.get("include_clean_reference") is not True
        or perturbations.get("awgn", {}).get("snr_db_levels")
        != EXPECTED_AWGN_LEVELS
        or perturbations.get("awgn", {}).get("repeat_seeds")
        != EXPECTED_REPEAT_SEEDS
        or perturbations.get("awgn", {}).get("common_random_numbers_across_snr")
        is not True
        or perturbations.get("carrier_frequency_offset", {}).get(
            "offset_hz_levels"
        )
        != EXPECTED_CFO_LEVELS
        or perturbations.get("carrier_frequency_offset", {}).get("sample_rate_hz")
        != 100_000_000
        or perturbations.get("multipath", {}).get("repeat_seeds")
        != EXPECTED_REPEAT_SEEDS
    ):
        raise TrainingConfigError("冻结扰动条件不匹配")
    statistics = protocol.get("statistics", {})
    if (
        statistics.get("primary_unit") != "recording"
        or statistics.get("primary_metric") != "recording_macro_f1"
        or statistics.get("bootstrap_replicates") != 2000
    ):
        raise TrainingConfigError("冻结统计协议不匹配")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("num_workers") != 0:
        raise TrainingConfigError("Windows 本机评估必须使用 num_workers=0")
    outputs = protocol.get("outputs", {})
    if set(outputs) != {
        "summary_json",
        "metrics_csv",
        "window_predictions_csv",
        "recording_predictions_csv",
        "curve_png",
        "curve_svg",
    } or len(set(outputs.values())) != len(outputs):
        raise TrainingConfigError("协议输出集合缺失或路径重复")
    _validate_multipath_profiles(protocol)
    protocol["_protocol_path"] = str(source)
    return protocol


def _condition_slug(value: int | float) -> str:
    rendered = f"{value:g}"
    return rendered.replace("-", "minus_").replace(".", "p")


def expand_conditions(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    perturbations = protocol["perturbations"]
    conditions: list[dict[str, Any]] = [
        {
            "condition_id": "clean",
            "family": "clean",
            "value": "clean",
            "repeat_seed": None,
            "spec": None,
        }
    ]
    for repeat_seed in perturbations["awgn"]["repeat_seeds"]:
        repeat_index = perturbations["awgn"]["repeat_seeds"].index(repeat_seed) + 1
        for snr_db in perturbations["awgn"]["snr_db_levels"]:
            conditions.append(
                {
                    "condition_id": f"awgn_{_condition_slug(snr_db)}db_r{repeat_index}",
                    "family": "awgn",
                    "value": float(snr_db),
                    "repeat_seed": int(repeat_seed),
                    "spec": IQPerturbationSpec(kind="awgn", snr_db=snr_db),
                }
            )
    cfo = perturbations["carrier_frequency_offset"]
    for offset_hz in cfo["offset_hz_levels"]:
        conditions.append(
            {
                "condition_id": f"cfo_{_condition_slug(offset_hz)}hz",
                "family": "carrier_frequency_offset",
                "value": float(offset_hz),
                "repeat_seed": None,
                "spec": IQPerturbationSpec(
                    kind="carrier_frequency_offset",
                    offset_hz=offset_hz,
                    sample_rate_hz=cfo["sample_rate_hz"],
                ),
            }
        )
    multipath = perturbations["multipath"]
    for repeat_seed in multipath["repeat_seeds"]:
        repeat_index = multipath["repeat_seeds"].index(repeat_seed) + 1
        for profile in multipath["profiles"]:
            conditions.append(
                {
                    "condition_id": f"multipath_{profile}_r{repeat_index}",
                    "family": "multipath",
                    "value": profile,
                    "repeat_seed": int(repeat_seed),
                    "spec": IQPerturbationSpec(kind="multipath", profile=profile),
                }
            )
    identifiers = [item["condition_id"] for item in conditions]
    if len(identifiers) != 40 or len(set(identifiers)) != len(identifiers):
        raise TrainingConfigError("展开后的冻结条件必须是 40 个唯一单因素条件")
    return conditions


def ensemble_probabilities(member_probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(member_probabilities, dtype=np.float64)
    if matrix.ndim != 3 or matrix.shape[0] != 3 or matrix.shape[1] < 1:
        raise TrainingConfigError("集成输入必须为 [3, samples, classes]")
    if (
        not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or not np.allclose(matrix.sum(axis=2), 1.0, atol=1e-6)
    ):
        raise TrainingConfigError("集成输入必须是有限且归一的 softmax 概率")
    result = matrix.mean(axis=0)
    result /= result.sum(axis=1, keepdims=True)
    return result


def _classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_map: dict[str, int],
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or probabilities.shape != (targets.size, len(label_map)):
        raise TrainingConfigError("分类指标的标签与概率形状不匹配")
    if targets.size < 1 or not np.isfinite(probabilities).all():
        raise TrainingConfigError("分类指标输入为空或包含非有限值")
    predictions = probabilities.argmax(axis=1)
    labels = list(range(len(label_map)))
    names = {index: name for name, index in label_map.items()}
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    clipped = np.clip(probabilities[np.arange(targets.size), targets], 1e-12, 1.0)
    one_hot = np.eye(len(label_map), dtype=np.float64)[targets]
    confidence = probabilities.max(axis=1)
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean(f1)),
        "sample_count": int(targets.size),
        "negative_log_likelihood": float(-np.mean(np.log(clipped))),
        "brier_score": float(np.mean(np.square(probabilities - one_hot).sum(axis=1))),
        "confidence": {
            "mean": float(np.mean(confidence)),
            "minimum": float(np.min(confidence)),
            "p05": float(np.quantile(confidence, 0.05)),
            "median": float(np.median(confidence)),
        },
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=labels
        ).tolist(),
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


def evaluate_probability_sets(
    probabilities_by_predictor: dict[str, np.ndarray],
    targets: np.ndarray,
    recording_ids: list[str] | tuple[str, ...],
    label_map: dict[str, int],
    *,
    expected_windows_per_recording: int = EXPECTED_WINDOWS_PER_RECORDING,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(probabilities_by_predictor) != 3:
        raise TrainingConfigError("概率评估必须包含恰好三个 TCN 成员")
    ordered_ids = tuple(probabilities_by_predictor)
    stacked = np.stack([probabilities_by_predictor[name] for name in ordered_ids])
    all_probabilities = {
        **probabilities_by_predictor,
        ENSEMBLE_ID: ensemble_probabilities(stacked),
    }
    metrics: dict[str, Any] = {}
    cache: dict[str, Any] = {}
    for predictor_id, probabilities in all_probabilities.items():
        order, recording_probabilities, recording_targets = (
            aggregate_recording_probabilities(
                probabilities,
                recording_ids,
                targets,
                expected_windows_per_recording=expected_windows_per_recording,
            )
        )
        metrics[predictor_id] = {
            "window": _classification_metrics(targets, probabilities, label_map),
            "recording": _classification_metrics(
                recording_targets, recording_probabilities, label_map
            ),
        }
        cache[predictor_id] = {
            "window_probabilities": probabilities,
            "recording_ids": order,
            "recording_probabilities": recording_probabilities,
            "recording_targets": recording_targets,
            "recording_predictions": recording_probabilities.argmax(axis=1),
        }
    return metrics, cache


def _stratified_bootstrap_indices(
    targets: np.ndarray, *, seed: int, replicates: int
) -> np.ndarray:
    if replicates < 100:
        raise TrainingConfigError("bootstrap_replicates 必须至少为 100")
    targets = np.asarray(targets, dtype=np.int64)
    strata = [np.flatnonzero(targets == label) for label in sorted(set(targets.tolist()))]
    if not strata or any(indices.size == 0 for indices in strata):
        raise TrainingConfigError("bootstrap 类别分层不能为空")
    rng = np.random.default_rng(seed)
    return np.asarray(
        [
            np.concatenate(
                [rng.choice(indices, size=indices.size, replace=True) for indices in strata]
            )
            for _ in range(replicates)
        ],
        dtype=np.int64,
    )


def _macro_f1(targets: np.ndarray, predictions: np.ndarray, classes: int) -> float:
    scores = []
    for label in range(classes):
        true_positive = np.sum((targets == label) & (predictions == label))
        false_positive = np.sum((targets != label) & (predictions == label))
        false_negative = np.sum((targets == label) & (predictions != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def paired_bootstrap_clean_delta(
    targets: np.ndarray,
    condition_predictions: np.ndarray,
    clean_predictions: np.ndarray,
    bootstrap_indices: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, Any]:
    if not (
        targets.shape == condition_predictions.shape == clean_predictions.shape
        and bootstrap_indices.ndim == 2
        and bootstrap_indices.shape[1] == targets.size
    ):
        raise TrainingConfigError("配对 bootstrap 输入形状不匹配")
    differences = np.asarray(
        [
            _macro_f1(
                targets[indices], condition_predictions[indices], num_classes
            )
            - _macro_f1(targets[indices], clean_predictions[indices], num_classes)
            for indices in bootstrap_indices
        ],
        dtype=np.float64,
    )
    point = _macro_f1(targets, condition_predictions, num_classes) - _macro_f1(
        targets, clean_predictions, num_classes
    )
    return {
        "metric": "ensemble_recording_macro_f1_condition_minus_clean",
        "point": float(point),
        "ci95": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
        "interpretation": (
            "descriptive paired uncertainty conditional on the fixed validation recordings, "
            "checkpoints, and perturbation repeat"
        ),
    }


def _load_members(
    protocol: dict[str, Any],
    *,
    project_root: Path,
    dataset: KULeuvenMaterializedDataset,
    device: torch.device,
) -> tuple[EnsembleMember, ...]:
    members = []
    for specification in protocol["models"]:
        evidence_path = resolve_project_path(project_root, specification["training_evidence"])
        config_path = resolve_project_path(project_root, specification["resolved_config"])
        checkpoint_path = resolve_project_path(project_root, specification["checkpoint"])
        _verify_hash(
            evidence_path,
            specification["training_evidence_sha256"],
            f"{specification['id']} 训练证据",
        )
        _verify_hash(
            config_path,
            specification["resolved_config_sha256"],
            f"{specification['id']} 解析配置",
        )
        checkpoint_sha256 = _verify_hash(
            checkpoint_path,
            specification["checkpoint_sha256"],
            f"{specification['id']} 检查点",
        )
        evidence = _read_json(evidence_path, "训练证据")
        resolved_config = _read_json(config_path, "解析训练配置")
        seed = int(specification["seed"])
        if (
            evidence.get("status") != "completed_training_validation_only"
            or evidence.get("model", {}).get("name") != "DroneRFTCN"
            or int(evidence.get("seed", -1)) != seed
            or evidence.get("test_data_used") is not False
            or evidence.get("unknown_data_used") is not False
            or evidence.get("competition_test_claim_allowed") is not False
            or evidence.get("dataset", {}).get("data_identity") != dataset.data_identity
            or resolved_config.get("training", {}).get("seed") != seed
            or resolved_config.get("model", {}).get("name") != "DroneRFTCN"
        ):
            raise TrainingConfigError(
                f"{specification['id']} 训练证据或解析配置不满足冻结边界"
            )
        if Path(evidence["checkpoint_path"]).resolve() != checkpoint_path:
            raise TrainingConfigError(f"{specification['id']} 训练证据指向其他检查点")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("model_name") != "DroneRFTCN"
            or checkpoint.get("label_map") != dataset.label_map
            or checkpoint.get("data_identity") != dataset.data_identity
            or checkpoint.get("aggregation_protocol")
            != "mean-window-softmax-probability-v1"
            or checkpoint.get("config", {}).get("training", {}).get("seed") != seed
        ):
            raise TrainingConfigError(f"{specification['id']} 检查点身份不匹配")
        model = DroneRFTCN(num_classes=3, in_channels=2).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        members.append(
            EnsembleMember(
                predictor_id=specification["id"],
                seed=seed,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                model=model,
                source={
                    **specification,
                    "training_evidence_path": str(evidence_path),
                    "resolved_config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                },
            )
        )
    return tuple(members)


def _evaluate_condition(
    members: tuple[EnsembleMember, ...],
    dataset: Any,
    base_dataset: KULeuvenMaterializedDataset,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        seed=seed,
    )
    buffers: dict[str, list[np.ndarray]] = {
        member.predictor_id: [] for member in members
    }
    targets: list[int] = []
    indices: list[int] = []
    recording_ids: list[str] = []
    with torch.inference_mode():
        for inputs, batch_targets, batch_indices in loader:
            inputs = inputs.to(device)
            for member in members:
                logits = member.model(inputs)
                buffers[member.predictor_id].append(
                    torch.softmax(logits, dim=1).cpu().numpy()
                )
            batch_index_values = [int(value) for value in batch_indices.tolist()]
            targets.extend(int(value) for value in batch_targets.tolist())
            indices.extend(batch_index_values)
            recording_ids.extend(
                base_dataset.sample_metadata(index).recording_id
                for index in batch_index_values
            )
    if indices != list(range(len(base_dataset))):
        raise TrainingConfigError("验证 DataLoader 索引缺失、重复或顺序偏离")
    probabilities = {
        predictor_id: np.concatenate(chunks, axis=0).astype(np.float64)
        for predictor_id, chunks in buffers.items()
    }
    metrics, cache = evaluate_probability_sets(
        probabilities,
        np.asarray(targets, dtype=np.int64),
        recording_ids,
        base_dataset.label_map,
    )
    return metrics, cache, indices


def _prediction_rows(
    condition: dict[str, Any],
    cache: dict[str, Any],
    base_dataset: KULeuvenMaterializedDataset,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    window_rows: list[dict[str, Any]] = []
    recording_rows: list[dict[str, Any]] = []
    class_names = [
        name for name, _ in sorted(base_dataset.label_map.items(), key=lambda item: item[1])
    ]
    for predictor_id, values in cache.items():
        window_probabilities = values["window_probabilities"]
        for index, probability in enumerate(window_probabilities):
            metadata = base_dataset.sample_metadata(index)
            prediction = int(np.argmax(probability))
            window_rows.append(
                {
                    "condition_id": condition["condition_id"],
                    "family": condition["family"],
                    "value": condition["value"],
                    "repeat_seed": condition["repeat_seed"],
                    "predictor_id": predictor_id,
                    "data_index": index,
                    "window_id": metadata.window_id,
                    "recording_id": metadata.recording_id,
                    "target": metadata.label_index,
                    "prediction": prediction,
                    "confidence": float(probability[prediction]),
                    **{
                        f"prob_{name}": float(probability[class_index])
                        for class_index, name in enumerate(class_names)
                    },
                }
            )
        for recording_id, target, probability in zip(
            values["recording_ids"],
            values["recording_targets"],
            values["recording_probabilities"],
        ):
            prediction = int(np.argmax(probability))
            recording_rows.append(
                {
                    "condition_id": condition["condition_id"],
                    "family": condition["family"],
                    "value": condition["value"],
                    "repeat_seed": condition["repeat_seed"],
                    "predictor_id": predictor_id,
                    "recording_id": recording_id,
                    "target": int(target),
                    "prediction": prediction,
                    "confidence": float(probability[prediction]),
                    **{
                        f"prob_{name}": float(probability[class_index])
                        for class_index, name in enumerate(class_names)
                    },
                }
            )
    return window_rows, recording_rows


def _sample_summary(values: list[float]) -> dict[str, Any]:
    if not values or any(not isfinite(value) for value in values):
        raise TrainingConfigError("无法汇总空或非有限指标")
    return {
        "mean": fmean(values),
        "sample_standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "observation_count": len(values),
    }


def summarize_condition_groups(
    results: list[dict[str, Any]], member_ids: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        key = (result["family"], str(result["value"]))
        grouped.setdefault(key, []).append(result)
    clean = next(result for result in results if result["family"] == "clean")
    clean_member_mean = fmean(
        clean["metrics"][member_id]["recording"]["macro_f1"]
        for member_id in member_ids
    )
    clean_ensemble = clean["metrics"][ENSEMBLE_ID]["recording"]["macro_f1"]
    summaries = []
    for (family, _), group in grouped.items():
        value = group[0]["value"]
        member_values = [
            item["metrics"][member_id]["recording"]["macro_f1"]
            for item in group
            for member_id in member_ids
        ]
        ensemble_values = [
            item["metrics"][ENSEMBLE_ID]["recording"]["macro_f1"]
            for item in group
        ]
        summaries.append(
            {
                "family": family,
                "value": value,
                "member_seed_repeat_recording_macro_f1": _sample_summary(member_values),
                "ensemble_repeat_recording_macro_f1": _sample_summary(ensemble_values),
                "member_mean_delta_from_clean": fmean(member_values)
                - clean_member_mean,
                "ensemble_mean_delta_from_clean": fmean(ensemble_values)
                - clean_ensemble,
            }
        )
    family_order = {"clean": 0, "awgn": 1, "carrier_frequency_offset": 2, "multipath": 3}
    return sorted(
        summaries,
        key=lambda item: (family_order[item["family"]], str(item["value"])),
    )


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise TrainingConfigError(f"拒绝写入空 CSV：{path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _metric_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for predictor_id, metrics in result["metrics"].items():
            rows.append(
                {
                    "condition_id": result["condition_id"],
                    "family": result["family"],
                    "value": result["value"],
                    "repeat_seed": result["repeat_seed"],
                    "predictor_id": predictor_id,
                    "window_accuracy": metrics["window"]["accuracy"],
                    "window_macro_f1": metrics["window"]["macro_f1"],
                    "recording_accuracy": metrics["recording"]["accuracy"],
                    "recording_macro_f1": metrics["recording"]["macro_f1"],
                    "recording_negative_log_likelihood": metrics["recording"][
                        "negative_log_likelihood"
                    ],
                    "recording_brier_score": metrics["recording"]["brier_score"],
                    "recording_mean_confidence": metrics["recording"]["confidence"][
                        "mean"
                    ],
                    "recording_macro_f1_delta_from_clean": metrics[
                        "recording_macro_f1_delta_from_clean"
                    ],
                    "window_macro_f1_delta_from_clean": metrics[
                        "window_macro_f1_delta_from_clean"
                    ],
                }
            )
    return rows


def _plot_curves(
    path_png: Path,
    path_svg: Path,
    groups: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    clean = next(item for item in groups if item["family"] == "clean")
    clean_member = clean["member_seed_repeat_recording_macro_f1"]["mean"]
    clean_ensemble = clean["ensemble_repeat_recording_macro_f1"]["mean"]

    awgn = sorted(
        (item for item in groups if item["family"] == "awgn"),
        key=lambda item: float(item["value"]),
    )
    x_awgn = np.asarray([item["value"] for item in awgn], dtype=float)
    member_awgn = np.asarray(
        [item["member_seed_repeat_recording_macro_f1"]["mean"] for item in awgn]
    )
    std_awgn = np.asarray(
        [
            item["member_seed_repeat_recording_macro_f1"][
                "sample_standard_deviation"
            ]
            for item in awgn
        ]
    )
    ensemble_awgn = np.asarray(
        [item["ensemble_repeat_recording_macro_f1"]["mean"] for item in awgn]
    )
    axes[0].plot(x_awgn, member_awgn, marker="o", label="3 seeds/repeats mean")
    axes[0].fill_between(
        x_awgn,
        np.clip(member_awgn - std_awgn, 0, 1),
        np.clip(member_awgn + std_awgn, 0, 1),
        alpha=0.18,
    )
    axes[0].plot(x_awgn, ensemble_awgn, marker="s", label="probability ensemble")
    axes[0].axhline(clean_ensemble, linestyle="--", color="#64748b", linewidth=1)
    axes[0].set(title="Added circular AWGN", xlabel="Relative SNR (dB)")

    cfo = sorted(
        (item for item in groups if item["family"] == "carrier_frequency_offset"),
        key=lambda item: float(item["value"]),
    )
    cfo_points = [
        *[item for item in cfo if float(item["value"]) < 0],
        {
            "value": 0.0,
            "member_seed_repeat_recording_macro_f1": {"mean": clean_member},
            "ensemble_repeat_recording_macro_f1": {"mean": clean_ensemble},
        },
        *[item for item in cfo if float(item["value"]) > 0],
    ]
    x_cfo = np.asarray([float(item["value"]) / 1000.0 for item in cfo_points])
    axes[1].plot(
        x_cfo,
        [item["member_seed_repeat_recording_macro_f1"]["mean"] for item in cfo_points],
        marker="o",
        label="3-seed mean",
    )
    axes[1].plot(
        x_cfo,
        [item["ensemble_repeat_recording_macro_f1"]["mean"] for item in cfo_points],
        marker="s",
        label="probability ensemble",
    )
    axes[1].set(title="Carrier-frequency offset", xlabel="Offset (kHz)")

    multipath = [
        clean,
        *sorted(
            (item for item in groups if item["family"] == "multipath"),
            key=lambda item: str(item["value"]),
        ),
    ]
    labels = ["clean" if item["family"] == "clean" else str(item["value"]) for item in multipath]
    positions = np.arange(len(labels))
    axes[2].plot(
        positions,
        [item["member_seed_repeat_recording_macro_f1"]["mean"] for item in multipath],
        marker="o",
        label="3 seeds/repeats mean",
    )
    axes[2].plot(
        positions,
        [item["ensemble_repeat_recording_macro_f1"]["mean"] for item in multipath],
        marker="s",
        label="probability ensemble",
    )
    axes[2].set_xticks(positions, labels)
    axes[2].set(title="Window-local multipath", xlabel="Profile")

    for axis in axes:
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Recording Macro-F1")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("KU Leuven known-validation synthetic robustness diagnostics")
    for path in (path_png, path_svg):
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        figure.savefig(temporary, dpi=180)
        temporary.replace(path)
    plt.close(figure)


def run(protocol_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    protocol_file = Path(protocol["_protocol_path"])
    project_root = protocol_file.parents[2]
    outputs = {
        name: resolve_project_path(project_root, value)
        for name, value in protocol["outputs"].items()
    }
    if not dry_run:
        existing = [str(path) for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError("拒绝覆盖已有 KU Leuven 鲁棒性证据：" + "; ".join(existing))
        for path in outputs.values():
            path.parent.mkdir(parents=True, exist_ok=True)

    dataset_spec = protocol["dataset"]
    metadata_path = Path(dataset_spec["metadata"]).resolve()
    _verify_hash(metadata_path, dataset_spec["metadata_sha256"], "物化数据元数据")
    if metadata_path.parent != Path(dataset_spec["root"]).resolve():
        raise TrainingConfigError("数据根目录与 metadata 路径不一致")
    base_config_path = resolve_project_path(project_root, dataset_spec["base_config"])
    _verify_hash(base_config_path, dataset_spec["base_config_sha256"], "基础训练配置")
    base_config = load_config(base_config_path)
    if Path(base_config["dataset"]["root"]).resolve() != metadata_path.parent:
        raise TrainingConfigError("基础训练配置指向其他物化数据")
    comparison_path = resolve_project_path(
        project_root, protocol["source_comparison"]["path"]
    )
    _verify_hash(
        comparison_path,
        protocol["source_comparison"]["sha256"],
        "本机三种子复现汇总",
    )
    comparison = _read_json(comparison_path, "本机三种子复现汇总")
    if (
        comparison.get("test_data_used") is not False
        or comparison.get("unknown_data_used") is not False
        or comparison.get("a800_protocol_replaced") is not False
    ):
        raise TrainingConfigError("源复现汇总不满足封存边界")

    device = resolve_device(str(protocol["evaluation"]["device"]))
    set_reproducible_seed(int(protocol["statistics"]["bootstrap_seed"]))
    dataset: KULeuvenMaterializedDataset | None = None
    try:
        dataset = KULeuvenMaterializedDataset(
            metadata_path.parent, split="validation", verify_hashes=True
        )
        if (
            len(dataset) != dataset_spec["expected_validation_windows"]
            or len(dataset.recording_ids)
            != dataset_spec["expected_validation_recordings"]
            or dataset.data_identity != comparison.get("data_identity")
        ):
            raise TrainingConfigError("验证数据数量或身份偏离源复现汇总")
        counts: dict[str, int] = {}
        for sample in dataset.samples:
            counts[sample.recording_id] = counts.get(sample.recording_id, 0) + 1
        if set(counts.values()) != {EXPECTED_WINDOWS_PER_RECORDING}:
            raise TrainingConfigError("验证录制未保持每成员 32 窗口")
        members = _load_members(
            protocol, project_root=project_root, dataset=dataset, device=device
        )
        conditions = expand_conditions(protocol)
        if dry_run:
            return {
                "status": "dry_run",
                "protocol_path": str(protocol_file),
                "protocol_sha256": sha256_file(protocol_file),
                "device": str(device),
                "validation_windows": len(dataset),
                "validation_recordings": len(dataset.recording_ids),
                "member_ids": [member.predictor_id for member in members],
                "condition_count": len(conditions),
                "test_data_used": False,
                "unknown_data_used": False,
            }

        results: list[dict[str, Any]] = []
        all_window_rows: list[dict[str, Any]] = []
        all_recording_rows: list[dict[str, Any]] = []
        caches: dict[str, dict[str, Any]] = {}
        for condition in conditions:
            viewed_dataset = dataset
            if condition["spec"] is not None:
                viewed_dataset = DeterministicIQPerturbationDataset(
                    dataset,
                    perturbation=condition["spec"],
                    base_seed=(
                        condition["repeat_seed"]
                        if condition["repeat_seed"] is not None
                        else protocol["statistics"]["bootstrap_seed"]
                    ),
                )
            metrics, cache, _ = _evaluate_condition(
                members,
                viewed_dataset,
                dataset,
                device=device,
                batch_size=int(protocol["evaluation"]["batch_size"]),
                seed=int(protocol["statistics"]["bootstrap_seed"]),
            )
            window_rows, recording_rows = _prediction_rows(condition, cache, dataset)
            result = {
                key: value for key, value in condition.items() if key != "spec"
            }
            result["perturbation"] = (
                None if condition["spec"] is None else condition["spec"].as_dict()
            )
            result["metrics"] = metrics
            results.append(result)
            caches[condition["condition_id"]] = cache
            all_window_rows.extend(window_rows)
            all_recording_rows.extend(recording_rows)
            ensemble_score = metrics[ENSEMBLE_ID]["recording"]["macro_f1"]
            print(
                f"condition={condition['condition_id']} ensemble_recording_macro_f1={ensemble_score:.6f}",
                flush=True,
            )

        clean_result = next(item for item in results if item["condition_id"] == "clean")
        clean_cache = caches["clean"]
        clean_targets = clean_cache[ENSEMBLE_ID]["recording_targets"]
        bootstrap_indices = _stratified_bootstrap_indices(
            clean_targets,
            seed=int(protocol["statistics"]["bootstrap_seed"]),
            replicates=int(protocol["statistics"]["bootstrap_replicates"]),
        )
        predictor_ids = [member.predictor_id for member in members] + [ENSEMBLE_ID]
        for result in results:
            cache = caches[result["condition_id"]]
            if cache[ENSEMBLE_ID]["recording_ids"] != clean_cache[ENSEMBLE_ID][
                "recording_ids"
            ] or not np.array_equal(
                cache[ENSEMBLE_ID]["recording_targets"], clean_targets
            ):
                raise TrainingConfigError("条件间录制身份或标签不一致")
            for predictor_id in predictor_ids:
                result["metrics"][predictor_id][
                    "recording_macro_f1_delta_from_clean"
                ] = (
                    result["metrics"][predictor_id]["recording"]["macro_f1"]
                    - clean_result["metrics"][predictor_id]["recording"]["macro_f1"]
                )
                result["metrics"][predictor_id]["window_macro_f1_delta_from_clean"] = (
                    result["metrics"][predictor_id]["window"]["macro_f1"]
                    - clean_result["metrics"][predictor_id]["window"]["macro_f1"]
                )
            result["ensemble_recording_macro_f1_clean_delta_bootstrap"] = (
                paired_bootstrap_clean_delta(
                    clean_targets,
                    cache[ENSEMBLE_ID]["recording_predictions"],
                    clean_cache[ENSEMBLE_ID]["recording_predictions"],
                    bootstrap_indices,
                    num_classes=len(dataset.label_map),
                )
            )

        member_ids = [member.predictor_id for member in members]
        group_summaries = summarize_condition_groups(results, member_ids)
        low_awgn = [
            item
            for item in group_summaries
            if item["family"] == "awgn" and float(item["value"]) <= 0.0
        ]
        low_snr_summary = {
            "interval_db": [-20, 0],
            "member_seed_repeat_mean_recording_macro_f1": fmean(
                item["member_seed_repeat_recording_macro_f1"]["mean"]
                for item in low_awgn
            ),
            "ensemble_repeat_mean_recording_macro_f1": fmean(
                item["ensemble_repeat_recording_macro_f1"]["mean"]
                for item in low_awgn
            ),
        }
        _write_csv_atomic(outputs["metrics_csv"], _metric_rows(results))
        _write_csv_atomic(outputs["window_predictions_csv"], all_window_rows)
        _write_csv_atomic(outputs["recording_predictions_csv"], all_recording_rows)
        _plot_curves(outputs["curve_png"], outputs["curve_svg"], group_summaries)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report = {
            "schema_version": "1.0",
            "artifact_type": "ku_leuven_tcn_known_validation_robustness",
            "status": "completed_development_validation_only",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_path": str(protocol_file),
            "protocol_sha256": sha256_file(protocol_file),
            "environment": collect_environment(device),
            "dataset": {
                "metadata_path": str(metadata_path),
                "metadata_sha256": dataset_spec["metadata_sha256"],
                "data_identity": dataset.data_identity,
                "validation_windows": len(dataset),
                "validation_recordings": len(dataset.recording_ids),
                "validation_artifacts": dataset.metadata["artifacts"]["validation"],
            },
            "sources": [member.source for member in members],
            "ensemble": protocol["ensemble"],
            "conditions": results,
            "condition_group_summaries": group_summaries,
            "low_snr_summary": low_snr_summary,
            "statistics": protocol["statistics"],
            "perturbation_implementation": {
                "multipath_profiles": MULTIPATH_PROFILES,
                "post_perturbation_preprocessing_id": dataset_spec[
                    "post_perturbation_preprocessing_id"
                ],
            },
            "artifacts": {
                name: {"path": str(outputs[name]), "sha256": sha256_file(outputs[name])}
                for name in (
                    "metrics_csv",
                    "window_predictions_csv",
                    "recording_predictions_csv",
                    "curve_png",
                    "curve_svg",
                )
            },
            "development_validation_only": True,
            "open_set_threshold_calibrated": False,
            "competition_test_claim_allowed": False,
            "test_data_used": False,
            "unknown_data_used": False,
            "a800_protocol_replaced": False,
            "interpretation_boundaries": protocol["interpretation_boundaries"],
        }
        temporary = outputs["summary_json"].with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(outputs["summary_json"])
        return report
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if dataset is not None:
            dataset.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.protocol, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "status": report["status"],
                "protocol_sha256": report["protocol_sha256"],
                "condition_count": report.get("condition_count", len(report.get("conditions", []))),
                "low_snr_summary": report.get("low_snr_summary"),
                "test_data_used": report["test_data_used"],
                "unknown_data_used": report["unknown_data_used"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENSEMBLE_ID",
    "EnsembleMember",
    "ensemble_probabilities",
    "evaluate_probability_sets",
    "expand_conditions",
    "load_protocol",
    "paired_bootstrap_clean_delta",
    "run",
    "summarize_condition_groups",
]
