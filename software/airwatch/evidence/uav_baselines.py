"""Build a source-backed comparison from frozen UAV baseline evidence."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from airwatch.data.data_provenance import sha256_file
from scipy.stats import t as student_t


class UAVBaselineEvidenceError(ValueError):
    """Raised when baseline evidence is missing or not directly comparable."""


def build_uav_baseline_comparison(
    test_evidence_paths: Iterable[str | Path],
    *,
    output_json: str | Path,
    output_csv: str | Path,
) -> dict:
    paths = [Path(path).resolve() for path in test_evidence_paths]
    if len(paths) < 2:
        raise UAVBaselineEvidenceError("至少需要两个基线结果才能生成对比")
    destination_json = Path(output_json)
    destination_csv = Path(output_csv)
    if destination_json.exists() or destination_csv.exists():
        raise FileExistsError("拒绝覆盖已有无人机基线对比证据")

    payloads = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UAVBaselineEvidenceError(f"无法读取测试证据 {path}：{exc}") from exc
        if payload.get("status") != "completed" or payload.get("split") != "test":
            raise UAVBaselineEvidenceError(f"不是已完成的固定测试结果：{path}")
        payloads.append((path, payload))

    identities = {json.dumps(item["data_identity"], sort_keys=True) for _, item in payloads}
    label_maps = {json.dumps(item["label_map"], sort_keys=True) for _, item in payloads}
    if len(identities) != 1 or len(label_maps) != 1:
        raise UAVBaselineEvidenceError("基线使用了不同数据身份或标签映射，禁止横向对比")

    rows = []
    sources = []
    for evidence_path, payload in payloads:
        checkpoint = Path(payload["checkpoint_path"])
        if not checkpoint.is_file():
            raise UAVBaselineEvidenceError(f"测试检查点不存在：{checkpoint}")
        training_path = evidence_path.with_name(
            evidence_path.name.removesuffix("_test.json") + "_training.json"
        )
        try:
            training = json.loads(training_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UAVBaselineEvidenceError(f"无法读取训练证据 {training_path}：{exc}") from exc
        if training.get("dataset", {}).get("data_identity") != payload["data_identity"]:
            raise UAVBaselineEvidenceError(f"训练/测试数据身份不一致：{evidence_path}")
        metrics = payload["metrics"]
        row = {
            "model": training["model"]["name"],
            "seed": training["seed"],
            "checkpoint_epoch": payload["checkpoint_epoch"],
            "parameter_count": training["model"]["parameter_count"],
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "window_accuracy": metrics["window"]["accuracy"],
            "window_macro_f1": metrics["window"]["macro_f1"],
            "recording_accuracy": metrics["recording"]["accuracy"],
            "recording_macro_f1": metrics["recording"]["macro_f1"],
            "evaluation_windows_per_second": metrics["window"]["windows_per_second"],
        }
        rows.append(row)
        sources.append({
            "model": row["model"],
            "training_evidence": str(training_path),
            "training_evidence_sha256": sha256_file(training_path),
            "test_evidence": str(evidence_path),
            "test_evidence_sha256": sha256_file(evidence_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        })
    ranked = sorted(rows, key=lambda row: (-row["recording_macro_f1"], row["parameter_count"]))
    for rank, row in enumerate(ranked, start=1):
        row["recording_macro_f1_rank"] = rank
    report = {
        "schema_version": "1.0",
        "dataset": "DroneRF",
        "task": "four-class drone type including background",
        "selection_metric": "test recording_macro_f1 (reporting only; never used for model tuning)",
        "data_identity": payloads[0][1]["data_identity"],
        "label_map": payloads[0][1]["label_map"],
        "results": ranked,
        "sources": sources,
        "known_limitation": payloads[0][1]["known_limitation"],
        "throughput_scope": "end-to-end fixed-test DataLoader normalization plus batched inference; not single-sample CPU latency",
    }
    destination_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = destination_json.with_suffix(destination_json.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_json.replace(destination_json)
    temporary_csv = destination_csv.with_suffix(destination_csv.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    temporary_csv.replace(destination_csv)
    return report


def _metric_summary(values_by_seed: dict[int, float]) -> dict:
    ordered = [values_by_seed[seed] for seed in sorted(values_by_seed)]
    count = len(ordered)
    mean = statistics.fmean(ordered)
    standard_deviation = statistics.stdev(ordered) if count > 1 else 0.0
    if count > 1:
        margin = float(student_t.ppf(0.975, df=count - 1)) * standard_deviation / math.sqrt(count)
        low, high = max(0.0, mean - margin), min(1.0, mean + margin)
    else:
        low = high = mean
    return {
        "values_by_seed": {str(seed): values_by_seed[seed] for seed in sorted(values_by_seed)},
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "student_t_95_ci": [low, high],
        "run_count": count,
    }


def build_uav_multiseed_summary(
    test_evidence_paths: Iterable[str | Path],
    *,
    output_json: str | Path,
    output_csv: str | Path,
) -> dict:
    """Aggregate fixed-protocol runs without selecting a favorable seed."""
    paths = [Path(path).resolve() for path in test_evidence_paths]
    destination_json, destination_csv = Path(output_json), Path(output_csv)
    if destination_json.exists() or destination_csv.exists():
        raise FileExistsError("拒绝覆盖已有无人机多种子证据")
    grouped: dict[str, list[dict]] = defaultdict(list)
    identities, label_maps, sources = set(), set(), []
    for evidence_path in paths:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or payload.get("split") != "test":
            raise UAVBaselineEvidenceError(f"不是已完成的固定测试结果：{evidence_path}")
        identities.add(json.dumps(payload["data_identity"], sort_keys=True))
        label_maps.add(json.dumps(payload["label_map"], sort_keys=True))
        training_path = evidence_path.with_name(
            evidence_path.name.removesuffix("_test.json") + "_training.json"
        )
        training = json.loads(training_path.read_text(encoding="utf-8"))
        if training.get("dataset", {}).get("data_identity") != payload["data_identity"]:
            raise UAVBaselineEvidenceError(f"训练/测试数据身份不一致：{evidence_path}")
        checkpoint = Path(payload["checkpoint_path"])
        row = {
            "model": training["model"]["name"],
            "seed": int(training["seed"]),
            "parameter_count": int(training["model"]["parameter_count"]),
            "window_accuracy": float(payload["metrics"]["window"]["accuracy"]),
            "window_macro_f1": float(payload["metrics"]["window"]["macro_f1"]),
            "recording_accuracy": float(payload["metrics"]["recording"]["accuracy"]),
            "recording_macro_f1": float(payload["metrics"]["recording"]["macro_f1"]),
        }
        grouped[row["model"]].append(row)
        sources.append({
            "model": row["model"], "seed": row["seed"],
            "training_evidence": str(training_path),
            "training_evidence_sha256": sha256_file(training_path),
            "test_evidence": str(evidence_path),
            "test_evidence_sha256": sha256_file(evidence_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        })
    if len(identities) != 1 or len(label_maps) != 1:
        raise UAVBaselineEvidenceError("多种子运行的数据身份或标签映射不一致")
    seed_sets = {tuple(sorted(row["seed"] for row in rows)) for rows in grouped.values()}
    if len(seed_sets) != 1:
        raise UAVBaselineEvidenceError("各模型没有使用完全相同的随机种子集合")
    if any(len({row["seed"] for row in rows}) != len(rows) for rows in grouped.values()):
        raise UAVBaselineEvidenceError("同一模型包含重复随机种子")

    models = []
    flat_rows = []
    metric_names = (
        "window_accuracy", "window_macro_f1", "recording_accuracy", "recording_macro_f1"
    )
    for model, rows in sorted(grouped.items()):
        if len({row["parameter_count"] for row in rows}) != 1:
            raise UAVBaselineEvidenceError(f"{model} 在不同种子下参数量不一致")
        metrics = {
            metric: _metric_summary({row["seed"]: row[metric] for row in rows})
            for metric in metric_names
        }
        model_result = {
            "model": model,
            "parameter_count": rows[0]["parameter_count"],
            "seeds": list(next(iter(seed_sets))),
            "metrics": metrics,
        }
        models.append(model_result)
    models.sort(key=lambda item: -item["metrics"]["recording_macro_f1"]["mean"])
    for rank, item in enumerate(models, start=1):
        item["mean_recording_macro_f1_rank"] = rank
        flat = {
            "model": item["model"],
            "parameter_count": item["parameter_count"],
            "seed_count": len(item["seeds"]),
            "seeds": ";".join(str(seed) for seed in item["seeds"]),
            "mean_recording_macro_f1_rank": rank,
        }
        for metric in metric_names:
            summary = item["metrics"][metric]
            flat[f"{metric}_mean"] = summary["mean"]
            flat[f"{metric}_sample_std"] = summary["sample_standard_deviation"]
            flat[f"{metric}_ci95_low"] = summary["student_t_95_ci"][0]
            flat[f"{metric}_ci95_high"] = summary["student_t_95_ci"][1]
        flat_rows.append(flat)
    first_payload = json.loads(paths[0].read_text(encoding="utf-8"))
    report = {
        "schema_version": "1.0",
        "dataset": "DroneRF",
        "task": "four-class drone type including background",
        "seeds": list(next(iter(seed_sets))),
        "interval_method": "two-sided 95% Student-t interval across random seeds; descriptive only for n=3",
        "ranking_scope": "reporting only; test-set ranking is not used for architecture or threshold selection",
        "data_identity": first_payload["data_identity"],
        "label_map": first_payload["label_map"],
        "models": models,
        "sources": sources,
        "known_limitation": first_payload["known_limitation"],
    }
    destination_json.parent.mkdir(parents=True, exist_ok=True)
    temp_json = destination_json.with_suffix(destination_json.suffix + ".tmp")
    temp_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_json.replace(destination_json)
    temp_csv = destination_csv.with_suffix(destination_csv.suffix + ".tmp")
    with temp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader(); writer.writerows(flat_rows)
    temp_csv.replace(destination_csv)
    return report


def _ablation_signature(item: dict) -> tuple[str, str | None, bool, float]:
    return (
        str(item.get("model", item.get("name"))),
        item.get("fusion_mode"),
        bool(item.get("presence_head", False)),
        float(item.get("presence_loss_weight", 0.0)),
    )


def build_uav_ablation_summary(
    protocol_path: str | Path,
    test_evidence_paths: Iterable[str | Path],
    *,
    output_json: str | Path,
    output_csv: str | Path,
) -> dict:
    """Validate and summarize every predeclared dual-branch ablation variant."""
    protocol_path = Path(protocol_path).resolve()
    destination_json, destination_csv = Path(output_json), Path(output_csv)
    if destination_json.exists() or destination_csv.exists():
        raise FileExistsError("拒绝覆盖已有无人机消融汇总证据")
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UAVBaselineEvidenceError(f"无法读取冻结消融协议：{exc}") from exc

    expected = {
        _ablation_signature(variant): variant["id"] for variant in protocol["variants"]
    }
    if len(expected) != len(protocol["variants"]):
        raise UAVBaselineEvidenceError("冻结协议包含无法区分的重复变体")
    paths = [Path(path).resolve() for path in test_evidence_paths]
    if len(paths) != len(expected):
        raise UAVBaselineEvidenceError("测试证据数量与冻结协议不一致")

    rows, sources, identities, label_maps, seen_variants = [], [], set(), set(), set()
    for evidence_path in paths:
        try:
            test = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UAVBaselineEvidenceError(f"无法读取测试证据 {evidence_path}：{exc}") from exc
        if test.get("status") != "completed" or test.get("split") != "test":
            raise UAVBaselineEvidenceError(f"不是已完成的固定测试结果：{evidence_path}")
        training_path = evidence_path.with_name(
            evidence_path.name.removesuffix("_test.json") + "_training.json"
        )
        try:
            training = json.loads(training_path.read_text(encoding="utf-8"))
            resolved_config_path = Path(training["resolved_config_path"])
            config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise UAVBaselineEvidenceError(f"训练证据或固化配置不完整：{training_path}：{exc}") from exc

        if training.get("dataset", {}).get("data_identity") != test.get("data_identity"):
            raise UAVBaselineEvidenceError(f"训练/测试数据身份不一致：{evidence_path}")
        identities.add(json.dumps(test["data_identity"], sort_keys=True))
        label_maps.add(json.dumps(test["label_map"], sort_keys=True))
        protocol_reference = config.get("experiment_protocol", "").replace("\\", "/")
        if not protocol_reference.endswith(protocol_path.name):
            raise UAVBaselineEvidenceError(f"固化配置没有引用本消融协议：{resolved_config_path}")

        model_config = config.get("model", {})
        signature = _ablation_signature(model_config)
        if signature not in expected:
            raise UAVBaselineEvidenceError(f"存在未预注册变体：{resolved_config_path}")
        variant_id = expected[signature]
        if variant_id in seen_variants:
            raise UAVBaselineEvidenceError(f"变体重复：{variant_id}")
        seen_variants.add(variant_id)

        fixed = protocol["fixed_training"]
        actual_fixed = {
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "early_stopping_patience": config["training"]["early_stopping_patience"],
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "normalization": config["dataset"]["normalization"],
        }
        if actual_fixed != fixed or int(training["seed"]) != int(protocol["seed"]):
            raise UAVBaselineEvidenceError(f"变体偏离冻结训练参数：{resolved_config_path}")

        checkpoint = Path(test["checkpoint_path"])
        if not checkpoint.is_file():
            raise UAVBaselineEvidenceError(f"测试检查点不存在：{checkpoint}")
        metrics = test["metrics"]
        row = {
            "variant_id": variant_id,
            "experiment_name": test["experiment_name"],
            "model": training["model"]["name"],
            "fusion_mode": model_config.get("fusion_mode", "none"),
            "presence_head": bool(model_config.get("presence_head", False)),
            "presence_loss_weight": float(model_config.get("presence_loss_weight", 0.0)),
            "seed": int(training["seed"]),
            "checkpoint_epoch": int(test["checkpoint_epoch"]),
            "parameter_count": int(training["model"]["parameter_count"]),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "validation_recording_macro_f1": float(
                training["best_validation_recording_macro_f1"]
            ),
            "test_window_accuracy": float(metrics["window"]["accuracy"]),
            "test_window_macro_f1": float(metrics["window"]["macro_f1"]),
            "test_recording_accuracy": float(metrics["recording"]["accuracy"]),
            "test_recording_macro_f1": float(metrics["recording"]["macro_f1"]),
            "test_presence_recording_macro_f1": (
                float(metrics["presence_recording"]["macro_f1"])
                if "presence_recording" in metrics else None
            ),
            "test_fusion_weight_time_mean": (
                float(metrics["fusion_weight_time"]["mean"])
                if "fusion_weight_time" in metrics else None
            ),
            "test_fusion_weight_time_standard_deviation": (
                float(metrics["fusion_weight_time"]["standard_deviation"])
                if "fusion_weight_time" in metrics else None
            ),
        }
        rows.append(row)
        sources.append({
            "variant_id": variant_id,
            "resolved_config": str(resolved_config_path),
            "resolved_config_sha256": sha256_file(resolved_config_path),
            "training_evidence": str(training_path),
            "training_evidence_sha256": sha256_file(training_path),
            "test_evidence": str(evidence_path),
            "test_evidence_sha256": sha256_file(evidence_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        })

    if seen_variants != set(expected.values()):
        raise UAVBaselineEvidenceError("没有覆盖冻结协议中的全部变体")
    if len(identities) != 1 or len(label_maps) != 1:
        raise UAVBaselineEvidenceError("消融变体的数据身份或标签映射不一致")
    selected = max(rows, key=lambda row: row["validation_recording_macro_f1"])
    protocol_order = {item["id"]: index for index, item in enumerate(protocol["variants"])}
    rows.sort(key=lambda row: protocol_order[row["variant_id"]])
    report = {
        "schema_version": "1.0",
        "dataset": "DroneRF",
        "task": "four-class drone type including background",
        "protocol": protocol["protocol"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_frozen_on": protocol["frozen_on"],
        "seed": protocol["seed"],
        "selection_rule": protocol["selection_metric"],
        "selected_variant_by_validation": selected["variant_id"],
        "test_usage": protocol["test_usage"],
        "test_ranking_scope": "reporting only; test metrics did not alter the frozen variants",
        "data_identity": json.loads(next(iter(identities))),
        "label_map": json.loads(next(iter(label_maps))),
        "results": rows,
        "sources": sources,
        "known_limitation": test["known_limitation"],
    }
    destination_json.parent.mkdir(parents=True, exist_ok=True)
    temp_json = destination_json.with_suffix(destination_json.suffix + ".tmp")
    temp_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_json.replace(destination_json)
    temp_csv = destination_csv.with_suffix(destination_csv.suffix + ".tmp")
    with temp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp_csv.replace(destination_csv)
    return report


__all__ = [
    "UAVBaselineEvidenceError",
    "build_uav_baseline_comparison",
    "build_uav_multiseed_summary",
    "build_uav_ablation_summary",
]
