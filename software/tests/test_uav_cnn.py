import unittest

import torch

from airwatch.models import DroneRFCNN, DroneRFResNet18, DroneRFTCN


class DroneRFCNNTests(unittest.TestCase):
    def test_forward_returns_four_class_logits_and_features(self):
        model = DroneRFCNN(num_classes=4)
        inputs = torch.randn(3, 2, 4096)
        self.assertEqual(tuple(model(inputs).shape), (3, 4))
        self.assertEqual(tuple(model.forward_features(inputs).shape), (3, 256))

    def test_rejects_wrong_shape_or_channel_count(self):
        model = DroneRFCNN()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4096))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 4096))

    def test_resnet_and_tcn_share_two_band_contract(self):
        inputs = torch.randn(2, 2, 4096)
        for model_type in (DroneRFResNet18, DroneRFTCN):
            with self.subTest(model=model_type.__name__):
                model = model_type(num_classes=4)
                self.assertEqual(tuple(model(inputs).shape), (2, 4))
                self.assertEqual(model.forward_features(inputs).shape[0], 2)
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 1, 4096))

    def test_baselines_accept_explicit_single_band_contract(self):
        inputs = torch.randn(2, 1, 4096)
        for model_type in (DroneRFCNN, DroneRFResNet18, DroneRFTCN):
            with self.subTest(model=model_type.__name__):
                model = model_type(num_classes=4, in_channels=1)
                self.assertEqual(tuple(model(inputs).shape), (2, 4))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 2, 4096))


if __name__ == "__main__":
    unittest.main()
