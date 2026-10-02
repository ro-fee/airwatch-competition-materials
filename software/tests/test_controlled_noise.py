import unittest

import torch

from airwatch.analysis.controlled_noise import (
    ControlledAWGNDataset,
    ControlledNoiseError,
    DeterministicNoiseAugmentedDataset,
    add_awgn_at_relative_snr,
    sample_noise_seed,
)


class _TinyDataset:
    label_map = {"background": 0, "drone": 1}
    recording_ids = {"r0"}

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.ones(2, 1024), 1, index

    def sample_metadata(self, index):
        return {"index": index}


class ControlledNoiseTests(unittest.TestCase):
    def test_empirical_added_noise_snr_matches_request_per_channel(self):
        signal = torch.stack((torch.linspace(-2, 2, 4096), torch.linspace(1, -1, 4096)))
        noisy = add_awgn_at_relative_snr(signal, snr_db=-5.0, seed=17)
        noise = noisy - signal
        measured = 10 * torch.log10(
            signal.square().mean(dim=1) / noise.square().mean(dim=1)
        )
        torch.testing.assert_close(measured, torch.full((2,), -5.0), atol=2e-5, rtol=0)

    def test_noise_is_deterministic_per_sample_and_condition(self):
        signal = torch.ones(2, 256)
        first = add_awgn_at_relative_snr(
            signal, snr_db=0, seed=sample_noise_seed(3, 4, 0)
        )
        repeated = add_awgn_at_relative_snr(
            signal, snr_db=0, seed=sample_noise_seed(3, 4, 0)
        )
        other = add_awgn_at_relative_snr(
            signal, snr_db=0, seed=sample_noise_seed(3, 5, 0)
        )
        torch.testing.assert_close(first, repeated)
        self.assertFalse(torch.equal(first, other))

    def test_dataset_view_preserves_index_and_metadata(self):
        view = ControlledAWGNDataset(_TinyDataset(), snr_db=10, base_seed=9)
        values, target, index = view[0]
        self.assertEqual((target, index), (1, 0))
        self.assertEqual(tuple(values.shape), (2, 1024))
        self.assertEqual(view.sample_metadata(0), {"index": 0})
        self.assertEqual(view.recording_ids, {"r0"})

    def test_rejects_zero_power_signal(self):
        with self.assertRaisesRegex(ControlledNoiseError, "zero power"):
            add_awgn_at_relative_snr(torch.zeros(2, 32), snr_db=0, seed=1)

    def test_training_view_is_reproducible_for_one_epoch(self):
        first = DeterministicNoiseAugmentedDataset(
            _TinyDataset(), snr_db_levels=[0, 10], noise_probability=1,
            base_seed=5, epoch=2,
        )[0][0]
        repeated = DeterministicNoiseAugmentedDataset(
            _TinyDataset(), snr_db_levels=[0, 10], noise_probability=1,
            base_seed=5, epoch=2,
        )[0][0]
        other_epoch = DeterministicNoiseAugmentedDataset(
            _TinyDataset(), snr_db_levels=[0, 10], noise_probability=1,
            base_seed=5, epoch=3,
        )[0][0]
        torch.testing.assert_close(first, repeated)
        self.assertFalse(torch.equal(first, other_epoch))

    def test_training_view_can_be_frozen_clean(self):
        clean = _TinyDataset()[0][0]
        augmented = DeterministicNoiseAugmentedDataset(
            _TinyDataset(), snr_db_levels=[0], noise_probability=0,
            base_seed=5, epoch=1,
        )[0][0]
        torch.testing.assert_close(clean, augmented)


if __name__ == "__main__":
    unittest.main()
