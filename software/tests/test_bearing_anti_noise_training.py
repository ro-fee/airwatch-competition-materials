from __future__ import annotations

import unittest

import numpy as np
import torch

from training.bearing_augmentation import (
    AugmentedBearingDataset,
    TrainingAugmentationConfig,
    apply_training_augmentation,
)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.samples = [
            (torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).unsqueeze(0), 0),
            (torch.linspace(1.0, -1.0, 16, dtype=torch.float32).unsqueeze(0), 1),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        signal, target = self.samples[index]
        return signal.clone(), target


class TestBearingAntiNoiseTraining(unittest.TestCase):
    def test_training_augmentation_is_deterministic_and_does_not_mutate_input(self):
        signal = torch.linspace(-1.0, 1.0, 1024, dtype=torch.float32).unsqueeze(0)
        original = signal.clone()
        config = TrainingAugmentationConfig(
            clean_probability=0.0,
            awgn_snr_db=(5.0,),
            seed=20260904,
        )
        first = apply_training_augmentation(signal, config, epoch=1, sample_index=3)
        second = apply_training_augmentation(signal, config, epoch=1, sample_index=3)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, signal))
        self.assertTrue(torch.equal(signal, original))

    def test_wrapper_normalizes_augmented_train_windows_only(self):
        base = _TinyDataset()
        config = TrainingAugmentationConfig(
            clean_probability=0.0,
            awgn_snr_db=(5.0,),
            seed=20260904,
        )
        dataset = AugmentedBearingDataset(
            base,
            config=config,
            normalization="window_zscore",
        )
        signal, target = dataset[0]
        self.assertEqual(tuple(signal.shape), (1, 16))
        self.assertIn(target, (0, 1))
        self.assertAlmostEqual(float(signal.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(signal.std(unbiased=False)), 1.0, places=5)

    def test_clean_probability_can_leave_some_windows_unchanged(self):
        signal = torch.linspace(-1.0, 1.0, 1024, dtype=torch.float32).unsqueeze(0)
        config = TrainingAugmentationConfig(
            clean_probability=1.0,
            awgn_snr_db=(5.0,),
            seed=20260904,
        )
        changed = apply_training_augmentation(signal, config, epoch=1, sample_index=0)
        self.assertTrue(torch.equal(changed, signal))


if __name__ == "__main__":
    unittest.main()
