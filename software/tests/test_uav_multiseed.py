import unittest

from training.run_uav_multiseed import config_for_seed


class UAVMultiseedTests(unittest.TestCase):
    def test_seed_override_changes_seed_and_immutable_run_name(self):
        base = {
            "training": {"seed": 1},
            "output": {"run_name": "dronerf_cnn_type_seed1_v1"},
        }
        changed = config_for_seed(base, 20260909)
        self.assertEqual(changed["training"]["seed"], 20260909)
        self.assertEqual(changed["output"]["run_name"], "dronerf_cnn_type_seed20260909_v1")
        self.assertEqual(base["training"]["seed"], 1)

    def test_requires_explicit_seed_token_in_run_name(self):
        with self.assertRaises(ValueError):
            config_for_seed({"training": {"seed": 1}, "output": {"run_name": "run"}}, 2)


if __name__ == "__main__":
    unittest.main()
