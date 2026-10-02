import json
import unittest

import torch

from airwatch.analysis.iq_perturbations import (
    DeterministicIQPerturbationDataset,
    IQPerturbationError,
    IQPerturbationSpec,
    MULTIPATH_PROFILES,
    add_complex_awgn,
    apply_carrier_frequency_offset,
    apply_multipath_channel,
    complex_signal_power,
    multipath_profile_taps,
    recording_iq_perturbation_seed,
    window_iq_perturbation_seed,
)


class _TinyIQDataset:
    label_map = {"frysky": 0, "spektrum_dx4e": 1}
    recording_ids = frozenset({"recording-7", "recording-11"})

    def __init__(self):
        n = torch.arange(256, dtype=torch.float64)
        self.signal = torch.stack((torch.cos(n / 11), torch.sin(n / 11)))

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.signal.clone(), index, 7 + 4 * index

    def sample_metadata(self, index):
        return {
            "recording_id": f"recording-{7 + 4 * index}",
            "start_sample": index * 256,
            "window_id": f"window-{7 + 4 * index}",
        }


class IQPerturbationTests(unittest.TestCase):
    def test_complex_awgn_matches_requested_total_complex_snr(self):
        n = torch.arange(4096, dtype=torch.float64)
        signal = torch.stack((2.0 * torch.cos(n / 17), 0.7 * torch.sin(n / 9)))
        noisy = add_complex_awgn(signal, snr_db=-3.25, seed=123)
        measured_snr = 10.0 * torch.log10(
            complex_signal_power(signal) / complex_signal_power(noisy - signal)
        )
        torch.testing.assert_close(
            measured_snr, torch.tensor(-3.25, dtype=torch.float64), atol=1e-10, rtol=0
        )

    def test_awgn_is_deterministic_and_window_seeded_in_dataset(self):
        base = _TinyIQDataset()
        spec = {"kind": "awgn", "snr_db": 5}
        first = DeterministicIQPerturbationDataset(
            base, perturbation=spec, base_seed=19
        )
        repeated = DeterministicIQPerturbationDataset(
            base, perturbation=spec, base_seed=19
        )
        other_seed = DeterministicIQPerturbationDataset(
            base, perturbation=spec, base_seed=20
        )
        torch.testing.assert_close(first[0][0], repeated[0][0])
        self.assertFalse(torch.equal(first[0][0], first[1][0]))
        self.assertFalse(torch.equal(first[0][0], other_seed[0][0]))

    def test_awgn_noise_direction_is_independent_of_snr(self):
        signal = _TinyIQDataset().signal
        seed = window_iq_perturbation_seed(2026090901, "window-7")
        low = add_complex_awgn(signal, snr_db=-5, seed=seed) - signal
        high = add_complex_awgn(signal, snr_db=15, seed=seed) - signal
        torch.testing.assert_close(
            low / torch.linalg.vector_norm(low),
            high / torch.linalg.vector_norm(high),
            atol=1e-12,
            rtol=1e-12,
        )
        component_power = low.square().mean(dim=1)
        self.assertTrue(bool(torch.all(component_power > 0)))

    def test_cfo_preserves_magnitude_and_is_reversible(self):
        n = torch.arange(2048, dtype=torch.float64)
        signal = torch.stack((1.7 * torch.cos(n / 13), 0.8 * torch.sin(n / 7)))
        shifted = apply_carrier_frequency_offset(
            signal, offset_hz=125_000, sample_rate_hz=100_000_000, start_sample=8192
        )
        recovered = apply_carrier_frequency_offset(
            shifted,
            offset_hz=-125_000,
            sample_rate_hz=100_000_000,
            start_sample=8192,
        )
        torch.testing.assert_close(
            shifted.square().sum(dim=0),
            signal.square().sum(dim=0),
            atol=2e-14,
            rtol=1e-14,
        )
        torch.testing.assert_close(recovered, signal, atol=2e-14, rtol=1e-14)

    def test_cfo_absolute_phase_matches_unsliced_recording(self):
        signal = torch.stack(
            (
                torch.linspace(-1.0, 1.0, 400, dtype=torch.float64),
                torch.linspace(0.5, -0.5, 400, dtype=torch.float64),
            )
        )
        whole = apply_carrier_frequency_offset(
            signal, offset_hz=730_000, sample_rate_hz=100_000_000, start_sample=5000
        )
        first = apply_carrier_frequency_offset(
            signal[:, :173],
            offset_hz=730_000,
            sample_rate_hz=100_000_000,
            start_sample=5000,
        )
        second = apply_carrier_frequency_offset(
            signal[:, 173:],
            offset_hz=730_000,
            sample_rate_hz=100_000_000,
            start_sample=5173,
        )
        torch.testing.assert_close(torch.cat((first, second), dim=1), whole)

    def test_multipath_profiles_are_unit_energy_and_keep_window_length(self):
        serialized = json.loads(json.dumps(MULTIPATH_PROFILES))
        self.assertEqual(set(serialized), {"mild", "severe"})
        self.assertEqual(
            serialized["mild"]["boundary_model"],
            "independent_window_causal_zero_padding",
        )
        self.assertEqual(
            [tap["delay_samples"] for tap in serialized["mild"]["taps"]],
            [0, 5, 12],
        )
        self.assertEqual(
            [tap["relative_amplitude_db"] for tap in serialized["mild"]["taps"]],
            [0.0, -6.0, -12.0],
        )
        self.assertEqual(
            [tap["delay_samples"] for tap in serialized["severe"]["taps"]],
            [0, 8, 24, 64],
        )
        self.assertEqual(
            [tap["relative_amplitude_db"] for tap in serialized["severe"]["taps"]],
            [0.0, -3.0, -6.0, -9.0],
        )
        impulse = torch.zeros(2, 96, dtype=torch.float64)
        impulse[0, 0] = 1.0
        for profile in ("mild", "severe"):
            taps = multipath_profile_taps(profile)
            registered = serialized[profile]["taps"]
            nonzero_delays = torch.nonzero(taps.abs() > 0, as_tuple=False).flatten().tolist()
            self.assertEqual(
                nonzero_delays, [tap["delay_samples"] for tap in registered]
            )
            self.assertTrue(bool(torch.all(taps.real[nonzero_delays] > 0).item()))
            torch.testing.assert_close(
                taps.imag, torch.zeros_like(taps.imag), atol=0, rtol=0
            )
            measured_relative_db = [
                20.0 * torch.log10(taps[delay].abs() / taps[0].abs()).item()
                for delay in nonzero_delays
            ]
            for measured, tap_metadata in zip(measured_relative_db, registered):
                self.assertAlmostEqual(
                    measured, tap_metadata["relative_amplitude_db"], places=12
                )
            torch.testing.assert_close(
                taps.abs().square().sum(),
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-14,
                rtol=1e-14,
            )
            output = apply_multipath_channel(impulse, profile=profile)
            self.assertEqual(output.shape, impulse.shape)
            response = torch.complex(output[0], output[1])
            torch.testing.assert_close(response[: len(taps)], taps)
            torch.testing.assert_close(
                complex_signal_power(output) * output.shape[1],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-14,
                rtol=1e-14,
            )

    def test_seeded_multipath_phase_is_recording_stable(self):
        seed_a = recording_iq_perturbation_seed(31, "same-recording", "severe")
        seed_b = recording_iq_perturbation_seed(31, "same-recording", "severe")
        seed_c = recording_iq_perturbation_seed(31, "other-recording", "severe")
        self.assertEqual(seed_a, seed_b)
        taps_a = multipath_profile_taps("severe", seed=seed_a)
        taps_b = multipath_profile_taps("severe", seed=seed_b)
        taps_c = multipath_profile_taps("severe", seed=seed_c)
        torch.testing.assert_close(taps_a, taps_b)
        self.assertFalse(torch.equal(taps_a, taps_c))
        self.assertEqual(taps_a[0].imag.item(), 0.0)
        self.assertGreater(taps_a[0].real.item(), 0.0)
        torch.testing.assert_close(
            taps_a.abs(), multipath_profile_taps("severe").abs()
        )
        torch.testing.assert_close(
            taps_a.abs().square().sum(), torch.tensor(1.0, dtype=torch.float64)
        )

    def test_json_compatible_spec_and_dataset_contract_passthrough(self):
        spec = IQPerturbationSpec.from_mapping(
            {
                "kind": "carrier_frequency_offset",
                "offset_hz": 10_000,
                "sample_rate_hz": 100_000_000,
            }
        )
        self.assertEqual(spec.as_dict()["kind"], "carrier_frequency_offset")
        base = _TinyIQDataset()
        view = DeterministicIQPerturbationDataset(
            base, perturbation=spec, base_seed=5
        )
        values, target, dataset_index = view[1]
        self.assertEqual(values.shape, base.signal.shape)
        self.assertEqual((target, dataset_index), (1, 11))
        self.assertIs(view.label_map, base.label_map)
        self.assertIs(view.recording_ids, base.recording_ids)
        self.assertEqual(
            view.sample_metadata(1),
            {
                "recording_id": "recording-11",
                "start_sample": 256,
                "window_id": "window-11",
            },
        )
        torch.testing.assert_close(
            values.mean(dim=1),
            torch.zeros(2, dtype=torch.float64),
            atol=1e-15,
            rtol=0,
        )
        torch.testing.assert_close(
            complex_signal_power(values), torch.tensor(1.0, dtype=torch.float64)
        )

    def test_rejects_invalid_signal_and_parameters(self):
        valid = torch.ones(2, 32)
        invalid_cases = (
            lambda: add_complex_awgn(torch.ones(3, 32), snr_db=0, seed=1),
            lambda: add_complex_awgn(valid.to(torch.int64), snr_db=0, seed=1),
            lambda: add_complex_awgn(
                torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), snr_db=0, seed=1
            ),
            lambda: add_complex_awgn(torch.zeros(2, 32), snr_db=0, seed=1),
            lambda: add_complex_awgn(valid, snr_db=float("inf"), seed=1),
            lambda: apply_carrier_frequency_offset(
                valid, offset_hz=10, sample_rate_hz=0
            ),
            lambda: apply_multipath_channel(valid, profile="invented"),
            lambda: IQPerturbationSpec(kind="awgn", snr_db=0, profile="mild"),
            lambda: IQPerturbationSpec.from_mapping({"kind": "awgn", "unknown": 1}),
        )
        for operation in invalid_cases:
            with self.subTest(operation=operation), self.assertRaises(IQPerturbationError):
                operation()


if __name__ == "__main__":
    unittest.main()
