"""Run the frozen TCN/ResNet18 three-seed validation matrix on an A800."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.common import TrainingConfigError, load_config, resolve_project_path
from training.ku_leuven_baseline import train_ku_leuven


def load_matrix(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError(f"无法读取 A800 实验矩阵：{exc}") from exc
    if payload.get("protocol") != "ku-leuven-known-source-a800-model-selection-v1":
        raise TrainingConfigError("A800 实验矩阵协议不匹配")
    if payload.get("seeds") != [20260909, 20260910, 20260911]:
        raise TrainingConfigError("A800 实验必须使用冻结的三个种子")
    if [item.get("model") for item in payload.get("candidates", [])] != [
        "DroneRFTCN", "DroneRFResNet18"
    ]:
        raise TrainingConfigError("A800 候选模型集合或顺序不匹配")
    if payload.get("selection", {}).get("test_data_allowed") is not False:
        raise TrainingConfigError("模型选择期间必须封存测试集")
    if payload.get("selection", {}).get("unknown_data_allowed") is not False:
        raise TrainingConfigError("模型选择期间必须封存未知类")
    payload["_matrix_path"] = str(source)
    return payload


def run_matrix(
    matrix: dict[str, Any],
    *,
    dataset_root: str | Path,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[1]
    results = []
    for candidate in matrix["candidates"]:
        config_path = resolve_project_path(project_root, candidate["config"])
        for seed in matrix["seeds"]:
            config = load_config(config_path)
            config["training"] = dict(config["training"])
            config["training"]["seed"] = int(seed)
            config["output"] = dict(config["output"])
            run_name = f"{candidate['run_prefix']}_seed{seed}_v1"
            config["output"]["run_name"] = run_name
            result = train_ku_leuven(
                config,
                dry_run=dry_run,
                run_name=run_name,
                dataset_root_override=dataset_root,
            )
            results.append({
                "model": candidate["model"],
                "seed": seed,
                "run_name": run_name,
                "status": result["status"],
                "training_evidence": result.get("planned_outputs", {}).get("training_evidence")
                or str(resolve_project_path(project_root, config["output"]["evidence_dir"]) / f"{run_name}_training.json"),
            })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    results = run_matrix(
        load_matrix(args.matrix), dataset_root=args.dataset_root, dry_run=args.dry_run
    )
    print(json.dumps({"runs": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_matrix", "run_matrix"]
