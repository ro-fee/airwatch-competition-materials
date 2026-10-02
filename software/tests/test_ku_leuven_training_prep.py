import unittest
from pathlib import Path

from training.common import load_config
from training.ku_leuven_baseline import build_ku_leuven_datasets, train_ku_leuven


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training" / "configs" / "ku_leuven_tcn_local_smoke_v2.json"
DATA = Path(r"D:\AirWatch_Datasets\ku_leuven_drone_rf\prepared\known-iq-v1")


@unittest.skipUnless((DATA / "dataset-metadata.json").is_file(), "external audited KU Leuven data unavailable")
class KULeuvenTrainingPreparationTests(unittest.TestCase):
    def test_dry_run_binds_real_data_model_and_does_not_open_test(self):
        result = train_ku_leuven(load_config(CONFIG), dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["dataset"]["train_windows"], 3680)
        self.assertEqual(result["dataset"]["validation_windows"], 768)
        self.assertFalse(result["dataset"]["test_windows_opened"])
        self.assertFalse(result["dataset"]["unknown_windows_opened"])
        self.assertEqual(result["model"]["input_shape"], [2, 4096])
        self.assertEqual(result["model"]["num_classes"], 3)

    def test_train_and_validation_recordings_are_disjoint(self):
        datasets = build_ku_leuven_datasets(load_config(CONFIG), ROOT)
        try:
            self.assertFalse(datasets["train"].recording_ids & datasets["validation"].recording_ids)
            self.assertEqual(datasets["train"].data_identity, datasets["validation"].data_identity)
        finally:
            for dataset in datasets.values():
                dataset.close()


if __name__ == "__main__":
    unittest.main()
