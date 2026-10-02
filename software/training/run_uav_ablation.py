"""Execute the frozen DroneRF dual-branch ablation protocol once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from airwatch.data.data_provenance import sha256_file
from airwatch.evidence.uav_baselines import build_uav_ablation_summary
from training.common import load_config
from training.uav_baseline import evaluate_uav, train_uav


DEFAULT_PROTOCOL = Path("training/configs/dronerf_dual_branch_ablation_protocol_v1.json")
DEFAULT_CONFIGS = (
    Path("training/configs/dronerf_spectral_type_ablation_v1.json"),
    Path("training/configs/dronerf_dual_concat_type_ablation_v1.json"),
    Path("training/configs/dronerf_dual_adaptive_type_ablation_v1.json"),
    Path("training/configs/dronerf_dual_adaptive_presence_ablation_v1.json"),
)
DEFAULT_SUMMARY_JSON = Path(
    "artifacts/evidence/uav/dronerf_dual_branch_ablation_seed20260908_v1_summary.json"
)
DEFAULT_SUMMARY_CSV = Path(
    "artifacts/evidence/uav/dronerf_dual_branch_ablation_seed20260908_v1_summary.csv"
)


def _variant_signature(item: dict) -> tuple[str, str | None, bool, float]:
    return (
        str(item.get("model", item.get("name"))),
        item.get("fusion_mode"),
        bool(item.get("presence_head", False)),
        float(item.get("presence_loss_weight", 0.0)),
    )


def _validate_frozen_configs(protocol: dict, protocol_path: Path, configs: list[dict]) -> None:
    expected = {_variant_signature(item) for item in protocol["variants"]}
    actual = {_variant_signature(config["model"]) for config in configs}
    if len(configs) != len(expected) or actual != expected:
        raise ValueError("配置集合与冻结消融变体不一致")
    fixed = protocol["fixed_training"]
    for config in configs:
        if config.get("experiment_protocol") != str(protocol_path).replace("\\", "/"):
            raise ValueError(f"配置没有引用冻结协议：{config['experiment_name']}")
        actual_fixed = {
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "early_stopping_patience": config["training"]["early_stopping_patience"],
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "normalization": config["dataset"]["normalization"],
        }
        if actual_fixed != fixed or config["training"]["seed"] != protocol["seed"]:
            raise ValueError(f"配置偏离冻结训练参数：{config['experiment_name']}")


def run(protocol_path: Path, config_paths: list[Path]) -> list[dict]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    configs = [load_config(config_path) for config_path in config_paths]
    _validate_frozen_configs(protocol, protocol_path, configs)
    results = []
    for config in configs:
        print(f"START {config['experiment_name']}", flush=True)
        training = train_uav(config)
        evaluation = evaluate_uav(config, training["checkpoint_path"])
        row = {
            "experiment_name": config["experiment_name"],
            "model": training["model"]["name"],
            "checkpoint_epoch": evaluation["checkpoint_epoch"],
            "validation_recording_macro_f1": training["best_validation_recording_macro_f1"],
            "test_window_macro_f1": evaluation["metrics"]["window"]["macro_f1"],
            "test_recording_macro_f1": evaluation["metrics"]["recording"]["macro_f1"],
            "protocol_sha256": sha256_file(protocol_path),
        }
        results.append(row)
        print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--configs", type=Path, nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    args = parser.parse_args(argv)
    results = run(args.protocol, args.configs)
    configs = [load_config(path) for path in args.configs]
    test_paths = [
        Path(config["output"]["evidence_dir"]) / f"{config['output']['run_name']}_test.json"
        for config in configs
    ]
    build_uav_ablation_summary(
        args.protocol, test_paths,
        output_json=args.summary_json, output_csv=args.summary_csv,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
