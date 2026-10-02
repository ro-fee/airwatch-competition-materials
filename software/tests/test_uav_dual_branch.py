import unittest

import torch

from airwatch.models import DroneRFDualBranch, DroneRFSpectralBackbone, DroneRFSpectralCNN


class UAVFusionModelTests(unittest.TestCase):
    def test_spectral_branch_has_expected_contract(self):
        model = DroneRFSpectralCNN(num_classes=4)
        inputs = torch.randn(2, 2, 4096)
        self.assertEqual(tuple(model(inputs).shape), (2, 4))
        spectrum = model.backbone.spectrogram(inputs)
        self.assertEqual(spectrum.shape[:3], (2, 2, 129))
        self.assertTrue(torch.isfinite(spectrum).all().item())

    def test_adaptive_fusion_returns_bounded_gate_and_presence_logits(self):
        model = DroneRFDualBranch(num_classes=4, fusion_mode="adaptive", presence_head=True)
        outputs = model(torch.randn(2, 2, 4096))
        self.assertEqual(tuple(outputs["type_logits"].shape), (2, 4))
        self.assertEqual(tuple(outputs["presence_logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["fusion_weight_time"].shape), (2, 1))
        self.assertTrue(((outputs["fusion_weight_time"] >= 0) &
                         (outputs["fusion_weight_time"] <= 1)).all().item())

    def test_concat_ablation_uses_fixed_half_weight_for_reporting(self):
        model = DroneRFDualBranch(fusion_mode="concat", presence_head=False)
        outputs = model(torch.randn(1, 2, 4096))
        self.assertNotIn("presence_logits", outputs)
        self.assertEqual(float(outputs["fusion_weight_time"].item()), 0.5)

    def test_spectral_branch_rejects_invalid_input(self):
        branch = DroneRFSpectralBackbone()
        with self.assertRaises(ValueError):
            branch(torch.randn(1, 1, 4096))


if __name__ == "__main__":
    unittest.main()
