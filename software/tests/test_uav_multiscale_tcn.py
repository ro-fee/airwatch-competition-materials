"""CPU checks for the isolated A800 V2 model contract."""
import importlib
import unittest

import torch


class MultiScaleTCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model_module(self):
        spec = importlib.util.find_spec("airwatch.models.uav_multiscale_tcn")
        self.assertIsNotNone(spec, "A800 multi-scale model has not been implemented")
        return importlib.import_module("airwatch.models.uav_multiscale_tcn")

    def test_full_window_features_logits_and_backward(self):
        model = self.model_module().MultiScaleRFTCN()
        signal = torch.randn(2, 2, 4096, generator=torch.Generator().manual_seed(7))
        features = model.forward_features(signal)
        self.assertEqual(tuple(features.shape), (2, 256))
        logits = model(signal)
        self.assertEqual(tuple(logits.shape), (2, 3))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2]))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_temporal_blocks_do_not_read_future_samples_during_inference(self):
        # BN training statistics depend on a whole batch; causality is an
        # inference convolution property with frozen BN, and excludes the stem.
        model = self.model_module().MultiScaleRFTCN().eval()
        original = torch.randn(1, 32, 96, generator=torch.Generator().manual_seed(9))
        changed = original.clone()
        changed[..., 64:] += 100
        with torch.no_grad():
            first = model.temporal(original)
            second = model.temporal(changed)
        torch.testing.assert_close(first[..., :16], second[..., :16], atol=0, rtol=0)
        self.assertGreater((first[..., 16:] - second[..., 16:]).abs().sum().item(), 0)

    def test_factory_uses_frozen_tcn_and_supports_checkpoint_round_trip(self):
        module = self.model_module()
        from airwatch.models.uav_baselines import DroneRFTCN
        self.assertIsInstance(module.build_a800_model("tcn"), DroneRFTCN)
        model = module.build_a800_model("multiscale_tcn").eval()
        restored = module.build_a800_model("multiscale_tcn").eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        inputs = torch.randn(1, 2, 64)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), restored(inputs), atol=0, rtol=0)

    def test_rejects_wrong_channels_classes_and_unknown_model(self):
        module = self.model_module()
        for constructor in (
            lambda: module.MultiScaleRFTCN(in_channels=1),
            lambda: module.MultiScaleRFTCN(num_classes=1),
            lambda: module.MultiScaleRFTCN(num_classes=True),
            lambda: module.build_a800_model("unregistered"),
        ):
            with self.subTest(constructor=constructor), self.assertRaises(ValueError):
                constructor()
        with self.assertRaises(ValueError):
            module.MultiScaleRFTCN()(torch.randn(2, 1, 128))


if __name__ == "__main__":
    unittest.main()
