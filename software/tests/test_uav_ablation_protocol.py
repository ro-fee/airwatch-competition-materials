import json
import unittest
from pathlib import Path

from training.common import load_config
from training.run_uav_ablation import (
    DEFAULT_CONFIGS,
    DEFAULT_PROTOCOL,
    _validate_frozen_configs,
)


ROOT = Path(__file__).resolve().parents[1]


class UAVAblationProtocolTests(unittest.TestCase):
    def test_protocol_and_configs_predeclare_all_variants(self):
        protocol = json.loads((ROOT / DEFAULT_PROTOCOL).read_text(encoding="utf-8"))
        self.assertEqual(protocol["seed"], 20260908)
        self.assertEqual([item["id"] for item in protocol["variants"]], [
            "spectral_only", "dual_concat", "dual_adaptive", "dual_adaptive_presence"
        ])
        configs = [load_config(ROOT / path) for path in DEFAULT_CONFIGS]
        self.assertEqual(len(configs), 4)
        self.assertTrue(all(config["training"]["seed"] == 20260908 for config in configs))
        self.assertTrue(all(
            config["experiment_protocol"].endswith(DEFAULT_PROTOCOL.as_posix())
            for config in configs
        ))
        _validate_frozen_configs(protocol, DEFAULT_PROTOCOL, configs)

    def test_rejects_training_parameter_drift_before_training(self):
        protocol = json.loads((ROOT / DEFAULT_PROTOCOL).read_text(encoding="utf-8"))
        configs = [load_config(ROOT / path) for path in DEFAULT_CONFIGS]
        configs[0]["training"]["batch_size"] += 1
        with self.assertRaisesRegex(ValueError, "偏离冻结训练参数"):
            _validate_frozen_configs(protocol, DEFAULT_PROTOCOL, configs)


if __name__ == "__main__":
    unittest.main()
