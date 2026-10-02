"""Numerical and recovery invariants of A800 paired-view training."""
from collections import Counter
import importlib
import math
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _LogitModel(nn.Module):
    """A real linear classifier with an observable BN-style batch boundary."""
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))
        self.batch_sizes = []

    def forward(self, inputs):
        self.batch_sizes.append(len(inputs))
        return inputs * self.gain


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("training.a800_objective"),
                             "A800 objective has not been implemented")
        return importlib.import_module("training.a800_objective")

    def test_zero_coefficient_is_mean_weighted_ce_and_one_forward(self):
        model = _LogitModel()
        a = torch.tensor([[2., -1., 0.], [-1., 1., 2.]])
        b = torch.tensor([[0., -1., 2.], [2., 1., -1.]])
        labels = torch.tensor([0, 2])
        weights = torch.tensor([.5, 1., 1.5])
        actual, diagnostics = self.module().paired_objective(
            model, a, b, labels, class_weights=weights)
        expected = (F.cross_entropy(a, labels, weight=weights)
                    + F.cross_entropy(b, labels, weight=weights)) / 2
        torch.testing.assert_close(actual, expected)
        self.assertEqual(model.batch_sizes, [4])
        self.assertEqual(set(diagnostics), {"ce", "js", "total"})
        self.assertTrue(all(not v.requires_grad for v in diagnostics.values()))

    def test_identical_distributions_have_zero_js(self):
        logits = torch.tensor([[3., -2., 0.], [0., 1., 2.]])
        _, diagnostics = self.module().paired_objective(
            _LogitModel(), logits, logits, torch.tensor([0, 1]), js_weight=.1)
        self.assertAlmostEqual(diagnostics["js"].item(), 0., places=7)

    def test_js_matches_hand_computed_value_and_is_view_symmetric(self):
        a = torch.log(torch.tensor([[.8, .2], [.2, .8]]))
        b = a.flip(1)
        labels = torch.tensor([0, 1])
        loss1, d1 = self.module().paired_objective(_LogitModel(), a, b, labels, .1)
        loss2, d2 = self.module().paired_objective(_LogitModel(), b, a, labels, .1)
        expected_js = .8 * math.log(1.6) + .2 * math.log(.4)
        self.assertAlmostEqual(d1["js"].item(), expected_js, places=6)
        torch.testing.assert_close(loss1, loss2)
        torch.testing.assert_close(d1["js"], d2["js"])
        torch.testing.assert_close(loss1, d1["ce"] + .1 * d1["js"])

    def test_class_weights_do_not_weight_js(self):
        a = torch.tensor([[2., -1.], [0., 2.]])
        b = a.flip(1)
        _, unweighted = self.module().paired_objective(
            _LogitModel(), a, b, torch.tensor([0, 1]), .1)
        _, weighted = self.module().paired_objective(
            _LogitModel(), a, b, torch.tensor([0, 1]), .1,
            class_weights=torch.tensor([.2, 1.8]))
        torch.testing.assert_close(unweighted["js"], weighted["js"], atol=0, rtol=0)

    def test_js_backpropagates_into_both_views(self):
        module = self.module()
        a = torch.tensor([[.5, -1., 2.]], requires_grad=True)
        b = torch.tensor([[2., .5, -1.]], requires_grad=True)
        base, _ = module.paired_objective(_LogitModel(), a, b, torch.tensor([0]), 0)
        base_grads = torch.autograd.grad(base, (a, b))
        total, _ = module.paired_objective(_LogitModel(), a, b, torch.tensor([0]), .1)
        grads = torch.autograd.grad(total, (a, b))
        for gradient, base_gradient in zip(grads, base_grads):
            self.assertGreater((gradient - base_gradient).abs().sum().item(), 1e-6)

    def test_extreme_bfloat16_logits_use_finite_fp32_loss_and_gradients(self):
        a = torch.tensor([[10000., -10000., 0.]], dtype=torch.bfloat16,
                         requires_grad=True)
        b = -a.detach().clone().requires_grad_(True)
        b.retain_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, diagnostics = self.module().paired_objective(
                _LogitModel(), a, b, torch.tensor([1]), .1)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertAlmostEqual(diagnostics["js"].item(), math.log(2), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all().item())
        self.assertTrue(torch.isfinite(b.grad).all().item())

    def test_rejects_malformed_pairs_weights_labels_and_nonfinite_logits(self):
        module = self.module()
        a = torch.ones(2, 3)
        labels = torch.tensor([0, 1])
        calls = (
            lambda: module.paired_objective(_LogitModel(), a, a[:1], labels),
            lambda: module.paired_objective(_LogitModel(), a, a, labels, -.1),
            lambda: module.paired_objective(_LogitModel(), a, a, labels, float("nan")),
            lambda: module.paired_objective(_LogitModel(), a, a, labels,
                                           class_weights=torch.tensor([1., 0., 1.])),
            lambda: module.paired_objective(_LogitModel(), a, a, torch.tensor([0, 3])),
            lambda: module.paired_objective(_LogitModel(), a * float("nan"), a, labels),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


class PairedAugmentationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("training.a800_objective"),
                             "A800 paired augmentation has not been implemented")
        return importlib.import_module("training.a800_objective")

    @staticmethod
    def inputs(count=12):
        from airwatch.analysis.iq_perturbations import normalize_complex_iq_window
        generator = torch.Generator().manual_seed(13)
        return torch.stack([normalize_complex_iq_window(torch.randn(2, 128,
                            generator=generator)) for _ in range(count)])

    def test_resume_batch_partition_and_global_random_state_are_independent(self):
        module = self.module()
        batch = self.inputs()
        original = batch.clone()
        ids = [f"window-{i}" for i in range(len(batch))]
        positions = list(range(41, 53))
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        views = module.make_paired_views(batch, ids, run_seed=20260909,
                                        logical_epoch=3, draw_positions=positions)
        restored = module.make_paired_views(batch[5:], ids[5:], run_seed=20260909,
                                           logical_epoch=3, draw_positions=positions[5:])
        for full, resumed in zip(views, restored):
            torch.testing.assert_close(full[5:], resumed, atol=0, rtol=0)
            self.assertNotEqual(full.data_ptr(), batch.data_ptr())
            self.assertTrue(torch.isfinite(full).all().item())
        torch.testing.assert_close(batch, original, atol=0, rtol=0)
        self.assertFalse(torch.equal(views[0], views[1]))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], current_numpy[0])
        np.testing.assert_array_equal(numpy_state[1], current_numpy[1])
        self.assertEqual(numpy_state[2:], current_numpy[2:])

    def test_every_identity_component_changes_augmentation_seed(self):
        recipe = self.module().paired_view_recipe
        identity = dict(run_seed=5, logical_epoch=2, draw_position=3,
                        window_id="window-4", view_id=0)
        baseline = recipe(**identity)["seed"]
        for field, replacement in dict(run_seed=6, logical_epoch=3, draw_position=4,
                                       window_id="window-5", view_id=1).items():
            with self.subTest(field=field):
                self.assertNotEqual(recipe(**(identity | {field: replacement}))["seed"],
                                    baseline)

    def test_frozen_v4_branch_and_awgn_distribution(self):
        counts = Counter()
        recipe = self.module().paired_view_recipe
        for position in range(12000):
            result = recipe(run_seed=20260909, logical_epoch=0, draw_position=position,
                            window_id="repeated-anchor", view_id=position % 2)
            counts[result["kind"]] += 1
            if result["kind"] == "awgn":
                counts[result["snr_db"]] += 1
            elif result["kind"] == "multipath":
                counts[result["profile"]] += 1
        # Deterministic, broad acceptance intervals catch uniform SNR weighting
        # and wrong branch probabilities without requiring a particular hash.
        self.assertTrue(.46 < counts["clean"] / 12000 < .54)
        self.assertTrue(.21 < counts["awgn"] / 12000 < .29)
        self.assertTrue(.21 < counts["multipath"] / 12000 < .29)
        self.assertTrue(.44 < counts[-5] / counts["awgn"] < .56)
        for level in (0, 5, 10, 15, 20):
            self.assertTrue(.065 < counts[level] / counts["awgn"] < .135)
        self.assertTrue(.44 < counts["mild"] / counts["multipath"] < .56)

    def test_views_apply_auditable_recipe_and_frozen_postprocessing(self):
        from airwatch.analysis.iq_perturbations import apply_iq_perturbation
        module = self.module()
        batch = self.inputs(24)
        views = module.make_paired_views(batch, ["anchor"] * 24, run_seed=11,
                                        logical_epoch=2, draw_positions=range(24))
        kinds = set()
        for view_id, values in enumerate(views):
            for position, actual in enumerate(values):
                recipe = module.paired_view_recipe(run_seed=11, logical_epoch=2,
                    draw_position=position, window_id="anchor", view_id=view_id)
                kinds.add(recipe["kind"])
                if recipe["kind"] == "clean":
                    expected = batch[position]
                else:
                    spec = {key: value for key, value in recipe.items() if key != "seed"}
                    expected = apply_iq_perturbation(batch[position], spec,
                        seed=recipe["seed"], channel_seed=recipe["seed"])
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(actual.mean(1), torch.zeros(2), atol=1e-6, rtol=0)
                self.assertAlmostEqual(actual.square().sum(0).mean().item(), 1., places=5)
        self.assertEqual(kinds, {"clean", "awgn", "multipath"})

    def test_rejects_unrecoverable_or_invalid_identity_and_invalid_inputs(self):
        module = self.module()
        inputs = self.inputs(2)
        defaults = dict(run_seed=1, logical_epoch=0, draw_positions=[0, 1])
        for batch, ids, overrides in (
            (inputs, ["one"], {}),
            (inputs, ["", "two"], {}),
            (inputs, ["one", "two"], {"draw_positions": [0, 0]}),
            (inputs, ["one", "two"], {"logical_epoch": -1}),
            (inputs, ["one", "two"], {"run_seed": True}),
            (inputs.double(), ["one", "two"], {}),
            (inputs * float("nan"), ["one", "two"], {}),
            (inputs[:, :1], ["one", "two"], {}),
        ):
            with self.subTest(overrides=overrides, ids=ids), self.assertRaises(ValueError):
                module.make_paired_views(batch, ids, **(defaults | overrides))


if __name__ == "__main__":
    unittest.main()
