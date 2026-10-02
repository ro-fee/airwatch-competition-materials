import unittest

import torch

from airwatch.models import BearingCNN


class TestBearingCNN(unittest.TestCase):
    def test_forward_returns_four_class_logits(self):
        model = BearingCNN(num_classes=4)
        outputs = model(torch.randn(2, 1, 1024))
        self.assertEqual(tuple(outputs.shape), (2, 4))
        self.assertTrue(torch.isfinite(outputs).all().item())

    def test_rejects_wrong_channel_count(self):
        model = BearingCNN(num_classes=4)
        with self.assertRaisesRegex(ValueError, "input channel"):
            model(torch.randn(1, 2, 1024))

    def test_rejects_non_batched_input(self):
        model = BearingCNN(num_classes=4)
        with self.assertRaisesRegex(ValueError, "expects"):
            model(torch.randn(1, 1024))


if __name__ == "__main__":
    unittest.main()
