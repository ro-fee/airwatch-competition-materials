"""Evaluate frozen DroneRF checkpoints under a predeclared added-noise protocol."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from airwatch.analysis import ControlledAWGNDataset
from airwatch.data.data_provenance import sha256_file
from training.common import (
    TrainingConfigError,
    collect_environment,
    load_config,
    make_loader,
    project_root_from_config,
    resolve_device,
)
from training.uav_baseline import build_uav_datasets, evaluate_uav_loader, make_uav_model


DEFAULT_PROTOCOL = Path("training/configs/dronerf_added_noise_robustness_protocol_v1.json")


def _macro_f1(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    scores = []
    for label in range(num_classes):
        true_positive = np.sum((targets == label) & (predictions == label))
        false_positive = np.sum((targets != label) & (predictions == label))
        false_negative = np.sum((targets == label) & (predictions != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _recording_predictions(
    rows: list[dict[str, Any]], label_map: dict[str, int]
) -> list[dict[str, Any]]:
    grouped_probabilities: dict[str, list[list[float]]] = defaultdict(list)
    grouped_targets: dict[str, set[int]] = defaultdict(set)
    ordered_labels = sorted(label_map.items(), key=lambda item: item[1])
    for row in rows:
        recording_id = str(row["recording_id"])
        grouped_targets[recording_id].add(int(row["target"]))
        grouped_probabilities[recording_id].append(
            [float(row[f"prob_{name}"]) for name, _ in ordered_labels]
        )
    results = []
    for recording_id in sorted(grouped_probabilities):
        if len(grouped_targets[recording_id]) != 1:
            raise TrainingConfigError(f"同一录制出现多个标签：{recording_id}")
        probability = np.mean(grouped_probabilities[recording_id], axis=0)
        results.append({
            "recording_id": recording_id,
            "target": next(iter(grouped_targets[recording_id])),
            "prediction": int(np.argmax(probability)),
        })
    return results


def _stratified_bootstrap_indices(
    targets: np.ndarray, *, seed: int, replicates: int
) -> np.ndarray:
    if replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(targets == label) for label in sorted(set(targets.tolist()))]
    if any(indices.size == 0 for indices in class_indices):
        raise ValueError("bootstrap target strata cannot be empty")
    samples = []
    for _ in range(replicates):
        samples.append(np.concatenate([
            rng.choice(indices, size=indices.size, replace=True) for indices in class_indices
        ]))
    return np.asarray(samples, dtype=np.int64)


def _paired_bootstrap(
    targets: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    *,
    seed: int,
    replicates: int,
    num_classes: int,
) -> dict[str, Any]:
    samples = _stratified_bootstrap_indices(targets, seed=seed, replicates=replicates)
    distributions = {
        model_id: np.asarray([
            _macro_f1(targets[index], predictions[index], num_classes)
            for index in samples
        ])
        for model_id, predictions in predictions_by_model.items()
    }
    intervals = {
        model_id: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for model_id, values in distributions.items()
    }
    model_ids = list(predictions_by_model)
    if len(model_ids) != 2:
        raise ValueError("paired comparison currently requires exactly two models")
    reference, candidate = model_ids
    difference = distributions[candidate] - distributions[reference]
    difference_ci = [float(value) for value in np.quantile(difference, [0.025, 0.975])]
    return {
        "model_ci95": intervals,
        "difference": {
            "candidate": candidate,
            "reference": reference,
            "metric": "recording_macro_f1",
            "point": _macro_f1(targets, predictions_by_model[candidate], num_classes)
            - _macro_f1(targets, predictions_by_model[reference], num_classes),
            "ci95": difference_ci,
            "direction_supported_95pct": (
                "candidate_better" if difference_ci[0] > 0
                else "candidate_worse" if difference_ci[1] < 0
                else "inconclusive"
            ),
        },
        "distributions": distributions,
    }


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _plot_curve(path_png: Path, path_svg: Path, results: list[dict[str, Any]], model_ids: list[str]) -> None:
    numeric = [item for item in results if item["snr_db"] is not None]
    clean = next(item for item in results if item["snr_db"] is None)
    colors = {model_ids[0]: "#1677ff", model_ids[1]: "#f97316"}
    figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for model_id in model_ids:
        x = np.asarray([item["snr_db"] for item in numeric], dtype=float)
        y = np.asarray([item["models"][model_id]["recording_macro_f1"] for item in numeric])
        low = np.asarray([item["models"][model_id]["recording_macro_f1_ci95"][0] for item in numeric])
        high = np.asarray([item["models"][model_id]["recording_macro_f1_ci95"][1] for item in numeric])
        axis.plot(x, y, marker="o", linewidth=2, label=model_id, color=colors[model_id])
        axis.fill_between(x, low, high, alpha=0.14, color=colors[model_id])
        axis.axhline(
            clean["models"][model_id]["recording_macro_f1"], linestyle="--",
            linewidth=1, alpha=0.65, color=colors[model_id],
        )
    axis.set(
        xlabel="Signal-to-added-noise ratio (dB)",
        ylabel="Recording-level Macro-F1",
        title="DroneRF controlled added-noise robustness (frozen checkpoints)",
        xlim=(-20.5, 20.5), ylim=(0, 1.02),
    )
    axis.set_xticks([item["snr_db"] for item in numeric])
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")
    for path in (path_png, path_svg):
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        figure.savefig(temporary, dpi=180)
        temporary.replace(path)
    plt.close(figure)


def run(protocol_path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    project_root = protocol_path.parents[2]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    outputs = {
        name: (project_root / relative).resolve()
        for name, relative in protocol["outputs"].items()
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("拒绝覆盖已有鲁棒性证据：" + "; ".join(existing))
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    data_source = (project_root / protocol["data_identity_source"]).resolve()
    if sha256_file(data_source) != protocol["data_identity_source_sha256"]:
        raise TrainingConfigError("数据身份源哈希偏离冻结协议")
    device = resolve_device(protocol["evaluation"]["device"])
    model_entries: list[tuple[dict[str, Any], dict[str, Any], torch.nn.Module]] = []
    test_dataset = None
    data_identity = None
    label_map = None
    for model_spec in protocol["models"]:
        config_path = (project_root / model_spec["config"]).resolve()
        checkpoint_path = (project_root / model_spec["checkpoint"]).resolve()
        if sha256_file(config_path) != model_spec["config_sha256"]:
            raise TrainingConfigError(f"配置哈希偏离冻结协议：{model_spec['id']}")
        if sha256_file(checkpoint_path) != model_spec["checkpoint_sha256"]:
            raise TrainingConfigError(f"检查点哈希偏离冻结协议：{model_spec['id']}")
        config = load_config(config_path)
        if project_root_from_config(config) != project_root:
            raise TrainingConfigError("模型配置不位于本项目 training/configs")
        datasets = build_uav_datasets(config, project_root)
        dataset = datasets[protocol["split"]]
        model = make_uav_model(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("model_name") != type(model).__name__:
            raise TrainingConfigError(f"模型类型不匹配：{model_spec['id']}")
        if checkpoint.get("data_identity") != dataset.data_identity:
            raise TrainingConfigError(f"检查点数据身份不一致：{model_spec['id']}")
        if checkpoint.get("label_map") != dataset.label_map:
            raise TrainingConfigError(f"检查点标签映射不一致：{model_spec['id']}")
        model.load_state_dict(checkpoint["model_state_dict"])
        if data_identity is None:
            test_dataset, data_identity, label_map = dataset, dataset.data_identity, dataset.label_map
        elif dataset.data_identity != data_identity or dataset.label_map != label_map:
            raise TrainingConfigError("两个模型没有使用同一固定测试集")
        model_entries.append((model_spec, config, model))

    assert test_dataset is not None and label_map is not None and data_identity is not None
    perturbation = protocol["perturbation"]
    conditions: list[float | None] = list(perturbation["snr_db_levels"])
    if perturbation["include_clean_reference"]:
        conditions.append(None)
    all_results, flat_rows, prediction_rows = [], [], []
    recording_cache: dict[float | None, dict[str, list[dict[str, Any]]]] = {}
    for condition_index, snr_db in enumerate(conditions):
        condition_label = "clean" if snr_db is None else f"{snr_db:g}_db"
        condition_result: dict[str, Any] = {
            "condition": condition_label, "snr_db": snr_db, "models": {},
        }
        recording_cache[snr_db] = {}
        for model_spec, config, model in model_entries:
            evaluated_dataset = test_dataset if snr_db is None else ControlledAWGNDataset(
                test_dataset, snr_db=snr_db, base_seed=perturbation["noise_seed"]
            )
            loader = make_loader(
                evaluated_dataset, batch_size=protocol["evaluation"]["batch_size"],
                shuffle=False, num_workers=protocol["evaluation"]["num_workers"],
                seed=perturbation["noise_seed"],
            )
            metrics, rows = evaluate_uav_loader(
                model, loader, evaluated_dataset, device=device
            )
            recording_cache[snr_db][model_spec["id"]] = _recording_predictions(rows, label_map)
            condition_result["models"][model_spec["id"]] = {
                "window_accuracy": metrics["window"]["accuracy"],
                "window_macro_f1": metrics["window"]["macro_f1"],
                "recording_accuracy": metrics["recording"]["accuracy"],
                "recording_macro_f1": metrics["recording"]["macro_f1"],
                "fusion_weight_time": metrics.get("fusion_weight_time"),
            }
            for row in rows:
                prediction_rows.append({
                    "condition": condition_label, "snr_db": "" if snr_db is None else snr_db,
                    "model_id": model_spec["id"], **row,
                })
        reference_records = recording_cache[snr_db][protocol["models"][0]["id"]]
        ids = [row["recording_id"] for row in reference_records]
        targets = np.asarray([row["target"] for row in reference_records], dtype=np.int64)
        predictions_by_model = {}
        for model_spec in protocol["models"]:
            records = recording_cache[snr_db][model_spec["id"]]
            if [row["recording_id"] for row in records] != ids:
                raise TrainingConfigError("配对 bootstrap 的 recording_id 顺序不一致")
            predictions_by_model[model_spec["id"]] = np.asarray(
                [row["prediction"] for row in records], dtype=np.int64
            )
        bootstrap = _paired_bootstrap(
            targets, predictions_by_model,
            seed=protocol["statistics"]["bootstrap_seed"] + condition_index,
            replicates=protocol["statistics"]["bootstrap_replicates"],
            num_classes=len(label_map),
        )
        for model_id, interval in bootstrap["model_ci95"].items():
            condition_result["models"][model_id]["recording_macro_f1_ci95"] = interval
        condition_result["paired_difference"] = bootstrap["difference"]
        all_results.append(condition_result)
        for model_spec in protocol["models"]:
            model_id = model_spec["id"]
            metrics = condition_result["models"][model_id]
            flat_rows.append({
                "condition": condition_label, "snr_db": "" if snr_db is None else snr_db,
                "model_id": model_id,
                "window_accuracy": metrics["window_accuracy"],
                "window_macro_f1": metrics["window_macro_f1"],
                "recording_accuracy": metrics["recording_accuracy"],
                "recording_macro_f1": metrics["recording_macro_f1"],
                "recording_macro_f1_ci95_low": metrics["recording_macro_f1_ci95"][0],
                "recording_macro_f1_ci95_high": metrics["recording_macro_f1_ci95"][1],
                "candidate_minus_reference_recording_macro_f1": bootstrap["difference"]["point"],
                "difference_ci95_low": bootstrap["difference"]["ci95"][0],
                "difference_ci95_high": bootstrap["difference"]["ci95"][1],
                "difference_direction_95pct": bootstrap["difference"]["direction_supported_95pct"],
                "fusion_weight_time_mean": (
                    metrics["fusion_weight_time"]["mean"]
                    if metrics["fusion_weight_time"] else ""
                ),
            })
        print(
            f"condition={condition_label} " + " ".join(
                f"{model_id}={condition_result['models'][model_id]['recording_macro_f1']:.6f}"
                for model_id in predictions_by_model
            ) + f" delta={bootstrap['difference']['point']:.6f}",
            flush=True,
        )

    low_min, low_max = protocol["statistics"]["low_snr_interval_db"]
    low_results = [item for item in all_results if item["snr_db"] is not None and low_min <= item["snr_db"] <= low_max]
    low_summary = {}
    for model_spec in protocol["models"]:
        model_id = model_spec["id"]
        low_summary[model_id] = {
            "mean_recording_macro_f1_across_predeclared_levels": float(np.mean([
                item["models"][model_id]["recording_macro_f1"] for item in low_results
            ]))
        }
    candidate, reference = protocol["models"][1]["id"], protocol["models"][0]["id"]
    low_summary["candidate_minus_reference"] = (
        low_summary[candidate]["mean_recording_macro_f1_across_predeclared_levels"]
        - low_summary[reference]["mean_recording_macro_f1_across_predeclared_levels"]
    )

    _write_csv_atomic(outputs["metrics_csv"], flat_rows)
    _write_csv_atomic(outputs["predictions_csv"], prediction_rows)
    _plot_curve(
        outputs["curve_png"], outputs["curve_svg"], all_results,
        [item["id"] for item in protocol["models"]],
    )
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "protocol": protocol["protocol"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_frozen_on": protocol["frozen_on"],
        "environment": collect_environment(device),
        "data_identity": data_identity,
        "label_map": label_map,
        "window_count_per_condition": len(test_dataset),
        "recording_count_per_condition": len(test_dataset.recording_ids),
        "perturbation": perturbation,
        "statistics": protocol["statistics"],
        "results": all_results,
        "low_snr_summary": low_summary,
        "sources": [{
            **model_spec,
            "config_path": str((project_root / model_spec["config"]).resolve()),
            "checkpoint_path": str((project_root / model_spec["checkpoint"]).resolve()),
        } for model_spec in protocol["models"]],
        "artifacts": {
            "metrics_csv": str(outputs["metrics_csv"]),
            "metrics_csv_sha256": sha256_file(outputs["metrics_csv"]),
            "predictions_csv": str(outputs["predictions_csv"]),
            "predictions_csv_sha256": sha256_file(outputs["predictions_csv"]),
            "curve_png": str(outputs["curve_png"]),
            "curve_png_sha256": sha256_file(outputs["curve_png"]),
            "curve_svg": str(outputs["curve_svg"]),
            "curve_svg_sha256": sha256_file(outputs["curve_svg"]),
        },
        "interpretation_boundary": (
            "Injected dB values describe normalized captured-window power relative to newly added AWGN. "
            "They are not calibrated physical acquisition SNR and do not establish cross-source deployment performance."
        ),
        "selection_boundary": protocol["test_usage"],
    }
    temporary_json = outputs["summary_json"].with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_json.replace(outputs["summary_json"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    report = run(args.protocol)
    print(json.dumps({
        "status": report["status"],
        "protocol_sha256": report["protocol_sha256"],
        "low_snr_summary": report["low_snr_summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
