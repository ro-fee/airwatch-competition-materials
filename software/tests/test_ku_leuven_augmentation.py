import unittest
from types import SimpleNamespace
import torch
from training.ku_leuven_augmentation import (
    AugmentedIQTrainingDataset,
    PROTOCOL_ID,
    PROTOCOLS,
    STRONG_NOISE_PROTOCOL_ID,
    WEIGHTED_WORST_NOISE_PROTOCOL_ID,
)


class TinyTrain:
    split = 'train'
    def __len__(self): return 30
    def __getitem__(self, i):
        n = torch.arange(256, dtype=torch.float32)
        return torch.stack((torch.cos(n/7), torch.sin(n/7))), 0, i
    def sample_metadata(self, i):
        return SimpleNamespace(window_id=f'w{i}', recording_id='r', start_sample=i*256)


class AugmentationTests(unittest.TestCase):
    def test_frozen_protocol_definitions(self):
        self.assertEqual(PROTOCOLS[PROTOCOL_ID]["awgn_snr_db"], (0, 5, 10, 15, 20))
        self.assertEqual(
            PROTOCOLS[STRONG_NOISE_PROTOCOL_ID]["awgn_snr_db"],
            (-5, 0, 5, 10, 15, 20),
        )
        self.assertEqual(
            PROTOCOLS[STRONG_NOISE_PROTOCOL_ID]["multipath_profiles"],
            ("mild", "severe"),
        )

    def test_training_only(self):
        data = TinyTrain()
        data.split = 'validation'
        with self.assertRaises(ValueError):
            AugmentedIQTrainingDataset(data, seed=1, epoch=1)

    def test_deterministic_epoch_specific_and_source_immutable(self):
        data = TinyTrain()
        first = AugmentedIQTrainingDataset(data, seed=1, epoch=1)
        same = AugmentedIQTrainingDataset(data, seed=1, epoch=1)
        other = AugmentedIQTrainingDataset(data, seed=1, epoch=2)
        changed = 0
        for i in range(len(data)):
            before = data[i][0].clone()
            torch.testing.assert_close(first[i][0], same[i][0])
            self.assertTrue(torch.isfinite(first[i][0]).all())
            torch.testing.assert_close(before, data[i][0])
            changed += not torch.equal(first[i][0], other[i][0])
        self.assertGreater(changed, 0)

    def test_strong_noise_protocol_is_deterministic_and_includes_minus_five_db(self):
        data = TinyTrain()
        first = AugmentedIQTrainingDataset(
            data, seed=9, epoch=3, protocol=STRONG_NOISE_PROTOCOL_ID
        )
        same = AugmentedIQTrainingDataset(
            data, seed=9, epoch=3, protocol=STRONG_NOISE_PROTOCOL_ID
        )
        self.assertEqual(first.awgn_snr_db, (-5, 0, 5, 10, 15, 20))
        for index in range(len(data)):
            torch.testing.assert_close(first[index][0], same[index][0])

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported IQ augmentation"):
            AugmentedIQTrainingDataset(
                TinyTrain(), seed=1, epoch=1, protocol="not-a-real-protocol"
            )

    def test_weighted_protocol_assigns_half_of_awgn_bins_to_minus_five(self):
        definition = PROTOCOLS[WEIGHTED_WORST_NOISE_PROTOCOL_ID]
        self.assertEqual(definition["awgn_weights"], (5, 1, 1, 1, 1, 1))
        dataset = AugmentedIQTrainingDataset(
            TinyTrain(),
            seed=9,
            epoch=3,
            protocol=WEIGHTED_WORST_NOISE_PROTOCOL_ID,
        )
        self.assertEqual(len(dataset.awgn_sampling_levels), 10)
        self.assertEqual(dataset.awgn_sampling_levels.count(-5), 5)
        self.assertEqual(dataset.awgn_sampling_levels.count(0), 1)
