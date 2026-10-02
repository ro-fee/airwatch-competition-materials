"""Read-only compatibility checks for external bearing-model references.

This module compares the public CNN-Transformer reference contract recorded in
``docs/网上轴承模型对照记录.md`` with the local, audited CWRU contract.  It
never downloads code or weights, trains a model, changes the UI, or touches
historical checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CompatibilityCheckError(ValueError):
    """Raised when a local manifest or label map is not safe to compare."""


@dataclass(frozen=True)
class ExternalReferenceSpec:
    """Facts recorded from the reviewed public repository, not local metrics."""

    name: str = "CNN-Transformer-for-Bearing-Fault-Diagnosis"
    source_url: str = (
        "https://github.com/GFAlpha/CNN-Transformer-for-Bearing-Fault-Diagnosis"
    )
    branch: str = "main"
    commit: str = "d980f387702ba99cf72cdd0505db13d5c95011e9"
    window_size: int = 1024
    step: int = 1024
    class_order: tuple[str, ...] = (
        "normal",
        "ball",
        "inner_race",
        "outer_race",
    )
    channel_description: str = (
        "one-dimensional vibration input; the reviewed preprocessing does not"
        " establish a fixed DE channel contract"
    )
    normalization: str = "not fixed by the reviewed repository contract"
    split_strategy: str = "window-level random split"
    weights_available: bool = False
    license_status: str = "clear public LICENSE file not confirmed"


@dataclass(frozen=True)
class CheckResult:
    """One explainable comparison result."""

    status: str
    local: Any
    external: Any
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise CompatibilityCheckError(f"manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise CompatibilityCheckError("manifest contains no rows")
    required = {
        "dataset",
        "class_name",
        "label_index",
        "load_hp",
        "sample_rate_hz",
        "sensor_key",
        "window_size",
        "step",
        "window_count",
        "split",
        "split_group",
        "local_path",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise CompatibilityCheckError(f"manifest missing required columns: {missing}")
    return rows


def _read_label_order(label_map_path: Path) -> tuple[str, ...]:
    if not label_map_path.is_file():
        raise CompatibilityCheckError(f"label map does not exist: {label_map_path}")
    try:
        payload = json.loads(label_map_path.read_text(encoding="utf-8"))
        labels = payload["labels"]
        indexed = {int(index): str(name) for index, name in labels.items()}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CompatibilityCheckError(f"invalid label map: {label_map_path}: {exc}") from exc
    if not indexed or set(indexed) != set(range(len(indexed))):
        raise CompatibilityCheckError("label map indices must be contiguous from zero")
    return tuple(indexed[index] for index in range(len(indexed)))


def _unique_int(rows: list[dict[str, str]], column: str) -> int | None:
    values = {int(row[column]) for row in rows}
    return values.pop() if len(values) == 1 else None


def _local_contract(rows: list[dict[str, str]], label_order: tuple[str, ...]) -> dict[str, Any]:
    datasets = sorted({row["dataset"] for row in rows})
    sample_rates = sorted({int(row["sample_rate_hz"]) for row in rows})
    sensors = sorted({row["sensor_key"] for row in rows})
    window_sizes = sorted({int(row["window_size"]) for row in rows})
    steps = sorted({int(row["step"]) for row in rows})
    split_groups = {
        split: sorted({row["split_group"] for row in rows if row["split"] == split})
        for split in ("train", "validation", "test")
    }
    window_counts = {
        split: sum(int(row["window_count"]) for row in rows if row["split"] == split)
        for split in ("train", "validation", "test")
    }
    class_names = sorted({row["class_name"] for row in rows})
    return {
        "dataset": datasets,
        "record_count": len(rows),
        "source_file_count": len({row["local_path"] for row in rows}),
        "class_order": list(label_order),
        "class_names": class_names,
        "sample_rate_hz": sample_rates,
        "sensor_keys": sensors,
        "window_sizes": window_sizes,
        "steps": steps,
        "normalization": "none (from the local training contract)",
        "split_strategy": "record/load-grouped manifest split",
        "split_groups": split_groups,
        "window_counts": window_counts,
        "all_sensor_keys_are_de": all(key.endswith("_DE_time") for key in sensors),
        "unique_label_indices": sorted({int(row["label_index"]) for row in rows}),
    }


def _compare(
    *,
    local: dict[str, Any],
    external: ExternalReferenceSpec,
    local_label_order: tuple[str, ...],
) -> dict[str, Any]:
    checks = {
        "window_size": CheckResult(
            "compatible" if local["window_sizes"] == [external.window_size] else "different",
            local["window_sizes"],
            external.window_size,
            "两边都按 1024 点窗口描述；这只说明输入长度相同，不代表权重可直接使用。",
        ),
        "step": CheckResult(
            "compatible" if local["steps"] == [external.step] else "different",
            local["steps"],
            external.step,
            "两边都按 1024 点步长描述。",
        ),
        "label_set": CheckResult(
            "compatible"
            if set(local_label_order) == set(external.class_order)
            else "different",
            list(local_label_order),
            list(external.class_order),
            "类别集合相同，但还必须检查数字标签的排列顺序。",
        ),
        "label_order": CheckResult(
            "different"
            if local_label_order != external.class_order
            else "compatible",
            list(local_label_order),
            list(external.class_order),
            "数字标签顺序不同；直接套用外部输出会把故障类别认错。",
        ),
        "channel": CheckResult(
            "requires_adapter",
            sorted(local["sensor_keys"]),
            external.channel_description,
            "本项目明确固定并记录 DE 通道；外部预处理没有形成同样明确的通道选择约定。",
        ),
        "normalization": CheckResult(
            "unknown",
            local["normalization"],
            external.normalization,
            "外部仓库的归一化契约未确认，不能假设与本项目一致。",
        ),
        "split": CheckResult(
            "not_comparable_as_is",
            local["split_strategy"],
            external.split_strategy,
            "外部窗口级随机切分可能让相邻窗口跨集合；本项目按原始记录/负载分组，不能直接拿结果做公平比较。",
        ),
        "weights": CheckResult(
            "unavailable",
            False,
            external.weights_available,
            "仓库文件树和 Release 未确认公开深度学习 checkpoint，因此当前没有可加载权重。",
        ),
        "license": CheckResult(
            "needs_review",
            "local project code only",
            external.license_status,
            "许可证文本未确认前不复制外部代码、权重或数据。",
        ),
    }
    can_directly_integrate = all(
        checks[key].status == "compatible" for key in ("window_size", "step", "label_set")
    ) and checks["label_order"].status == "compatible" and external.weights_available
    return {
        "reference": asdict(external),
        "local_contract": local,
        "checks": {name: check.to_dict() for name, check in checks.items()},
        "can_directly_integrate": can_directly_integrate,
        "decision": "external_reference_only",
        "weights_available": False,
        "trained": False,
        "ui_integrated": False,
        "historical_weights_modified": False,
        "summary": "不能直接接入；保留为结构和流程复现参考。",
    }


def build_cnn_transformer_compatibility_report(
    manifest_path: str | Path,
    label_map_path: str | Path,
    *,
    checked_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, read-only comparison report from local records."""
    manifest = Path(manifest_path).resolve()
    label_map = Path(label_map_path).resolve()
    rows = _read_rows(manifest)
    label_order = _read_label_order(label_map)
    local = _local_contract(rows, label_order)
    report = _compare(
        local=local,
        external=ExternalReferenceSpec(),
        local_label_order=label_order,
    )
    report["checked_at"] = checked_at or datetime.now(timezone.utc).isoformat()
    report["inputs"] = {
        "manifest": str(manifest),
        "label_map": str(label_map),
    }
    return report


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a JSON evidence artifact without touching model or UI files."""
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--label-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_cnn_transformer_compatibility_report(args.manifest, args.label_map)
    output = write_report(report, args.output)
    print(json.dumps({"status": "ok", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
