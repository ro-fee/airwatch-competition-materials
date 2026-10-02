"""Derive per-class failure evidence from frozen UAV robustness predictions."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from airwatch.data.data_provenance import sha256_file


class UAVRobustnessEvidenceError(ValueError):
    """Raised when robustness predictions cannot support a valid comparison."""


def _metrics(targets: np.ndarray, predictions: np.ndarray, label_map: dict[str, int]) -> dict:
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
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]), "support": int(support[index]),
            }
            for index in labels
        },
    }


def build_uav_robustness_failure_analysis(
    summary_path: str | Path,
    *,
    output_json: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    output_json, output_csv = Path(output_json), Path(output_csv)
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("拒绝覆盖已有鲁棒性失败分析")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise UAVRobustnessEvidenceError("鲁棒性主证据尚未完成")
    prediction_path = Path(summary["artifacts"]["predictions_csv"])
    if sha256_file(prediction_path) != summary["artifacts"]["predictions_csv_sha256"]:
        raise UAVRobustnessEvidenceError("逐窗口预测文件哈希与主证据不一致")
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    label_map = {name: int(index) for name, index in summary["label_map"].items()}
    probability_fields = [
        f"prob_{name}" for name, _ in sorted(label_map.items(), key=lambda item: item[1])
    ]
    expected_conditions = [item["condition"] for item in summary["results"]]
    expected_models = [item["id"] for item in summary["sources"]]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["model_id"])].append(row)
    expected_keys = {
        (condition, model) for condition in expected_conditions for model in expected_models
    }
    if set(grouped) != expected_keys:
        raise UAVRobustnessEvidenceError("预测条件或模型集合与主证据不一致")

    analyses, flat_rows = [], []
    for condition in expected_conditions:
        for model_id in expected_models:
            group = grouped[(condition, model_id)]
            if len(group) != summary["window_count_per_condition"]:
                raise UAVRobustnessEvidenceError(f"窗口数量不一致：{condition}/{model_id}")
            window_targets = np.asarray([int(row["target"]) for row in group], dtype=np.int64)
            window_predictions = np.asarray([int(row["prediction"]) for row in group], dtype=np.int64)
            window_metrics = _metrics(window_targets, window_predictions, label_map)
            recording_probabilities: dict[str, list[list[float]]] = defaultdict(list)
            recording_targets: dict[str, set[int]] = defaultdict(set)
            for row in group:
                recording_probabilities[row["recording_id"]].append(
                    [float(row[field]) for field in probability_fields]
                )
                recording_targets[row["recording_id"]].add(int(row["target"]))
            ordered_recordings = sorted(recording_probabilities)
            if len(ordered_recordings) != summary["recording_count_per_condition"]:
                raise UAVRobustnessEvidenceError(f"录音数量不一致：{condition}/{model_id}")
            if any(len(recording_targets[item]) != 1 for item in ordered_recordings):
                raise UAVRobustnessEvidenceError("同一录音包含多个目标标签")
            rec_targets = np.asarray([
                next(iter(recording_targets[item])) for item in ordered_recordings
            ], dtype=np.int64)
            rec_predictions = np.asarray([
                int(np.argmax(np.mean(recording_probabilities[item], axis=0)))
                for item in ordered_recordings
            ], dtype=np.int64)
            recording_metrics = _metrics(rec_targets, rec_predictions, label_map)
            counts = Counter(window_predictions.tolist())
            dominant_index, dominant_count = counts.most_common(1)[0]
            names = {index: name for name, index in label_map.items()}
            analysis = {
                "condition": condition,
                "model_id": model_id,
                "window": window_metrics,
                "recording": recording_metrics,
                "window_prediction_collapse": {
                    "dominant_label": names[dominant_index],
                    "dominant_fraction": dominant_count / len(window_predictions),
                },
            }
            analyses.append(analysis)
            for unit, metrics in (("window", window_metrics), ("recording", recording_metrics)):
                for class_name, class_metrics in metrics["per_class"].items():
                    flat_rows.append({
                        "condition": condition, "model_id": model_id, "unit": unit,
                        "class_name": class_name, **class_metrics,
                        "dominant_window_prediction": names[dominant_index],
                        "dominant_window_prediction_fraction": dominant_count / len(window_predictions),
                    })
    report = {
        "schema_version": "1.0",
        "source_summary": str(summary_path),
        "source_summary_sha256": sha256_file(summary_path),
        "source_predictions": str(prediction_path),
        "source_predictions_sha256": sha256_file(prediction_path),
        "data_identity": summary["data_identity"],
        "label_map": label_map,
        "analyses": analyses,
        "interpretation_boundary": summary["interpretation_boundary"],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temp_json = output_json.with_suffix(output_json.suffix + ".tmp")
    temp_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_json.replace(output_json)
    temp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader(); writer.writerows(flat_rows)
    temp_csv.replace(output_csv)
    return report


__all__ = ["UAVRobustnessEvidenceError", "build_uav_robustness_failure_analysis"]
