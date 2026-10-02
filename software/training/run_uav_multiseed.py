"""Run predeclared DroneRF baselines at additional fixed random seeds."""
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

from training.common import load_config
from training.uav_baseline import evaluate_uav, train_uav


DEFAULT_CONFIGS = (
    Path("training/configs/dronerf_cnn_type_baseline_v1.json"),
    Path("training/configs/dronerf_resnet18_type_baseline_v1.json"),
    Path("training/configs/dronerf_tcn_type_baseline_v1.json"),
)


def config_for_seed(base_config: dict, seed: int) -> dict:
    if seed < 0:
        raise ValueError("随机种子必须为非负整数")
    config = deepcopy(base_config)
    config["training"]["seed"] = seed
    original = str(config["output"]["run_name"])
    replaced, count = re.subn(r"seed\d+", f"seed{seed}", original, count=1)
    if count != 1:
        raise ValueError(f"run_name 缺少 seed<数字> 段：{original}")
    config["output"]["run_name"] = replaced
    return config


def run(config_paths: list[Path], seeds: list[int]) -> list[dict]:
    results = []
    for config_path in config_paths:
        base = load_config(config_path)
        for seed in seeds:
            config = config_for_seed(base, seed)
            run_name = config["output"]["run_name"]
            print(f"START model={config['model']['name']} seed={seed} run={run_name}", flush=True)
            training = train_uav(config)
            evaluation = evaluate_uav(config, training["checkpoint_path"])
            row = {
                "model": training["model"]["name"],
                "seed": seed,
                "run_name": run_name,
                "checkpoint_epoch": evaluation["checkpoint_epoch"],
                "window_macro_f1": evaluation["metrics"]["window"]["macro_f1"],
                "recording_macro_f1": evaluation["metrics"]["recording"]["macro_f1"],
            }
            results.append(row)
            print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=list(DEFAULT_CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)
    results = run(args.configs, args.seeds)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
