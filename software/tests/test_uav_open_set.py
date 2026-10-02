import unittest

import numpy as np

from airwatch.inference.uav_open_set import (
    UAVOpenSetError,
    aggregate_recording_probabilities,
    aggregate_recording_scores,
    calibrate_known_acceptance,
    decide_known,
    energy_knownness,
    fit_class_prototypes,
    maximum_softmax_knownness,
    prototype_knownness,
)


class UAVOpenSetTests(unittest.TestCase):
    def test_softmax_and_energy_are_higher_for_peaked_known_logits(self):
        logits = np.asarray([[8.0, 0.0], [0.1, 0.0]])
        self.assertGreater(maximum_softmax_knownness(logits)[0], maximum_softmax_knownness(logits)[1])
        self.assertGreater(energy_knownness(logits)[0], energy_knownness(logits)[1])

    def test_prototype_scores_support_cosine_and_squared_euclidean(self):
        training = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
        prototypes = fit_class_prototypes(training, np.asarray([0, 0, 1, 1]))
        samples = np.asarray([[0.9, 0.1], [-1.0, -1.0]])
        cosine = prototype_knownness(samples, prototypes)
        euclidean = prototype_knownness(samples, prototypes, metric="squared_euclidean")
        self.assertEqual(prototypes.labels, (0, 1))
        self.assertGreater(cosine[0], cosine[1])
        self.assertGreater(euclidean[0], euclidean[1])

    def test_recording_aggregation_prevents_window_count_weighting(self):
        ids, scores = aggregate_recording_scores([0.1, 0.3, 0.9], ["a", "a", "b"])
        self.assertEqual(ids, ("a", "b"))
        np.testing.assert_allclose(scores, [0.2, 0.9])

    def test_probability_aggregation_enforces_target_and_window_contract(self):
        ids, probabilities, targets = aggregate_recording_probabilities(
            [[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.2, 0.8]],
            ["a", "a", "b", "b"],
            [0, 0, 1, 1],
            expected_windows_per_recording=2,
        )
        self.assertEqual(ids, ("a", "b"))
        np.testing.assert_allclose(probabilities, [[0.7, 0.3], [0.15, 0.85]])
        np.testing.assert_array_equal(targets, [0, 1])
        with self.assertRaises(UAVOpenSetError):
            aggregate_recording_probabilities(
                [[0.8, 0.2], [0.6, 0.4]], ["a", "a"], [0, 1]
            )
        with self.assertRaises(UAVOpenSetError):
            aggregate_recording_probabilities(
                [[0.8, 0.2], [0.6, 0.4]], ["a", "a"], [0, 0],
                expected_windows_per_recording=3,
            )

    def test_calibration_meets_target_without_unknown_samples(self):
        scores = np.arange(20, dtype=np.float64)
        calibration = calibrate_known_acceptance(
            scores, method="maximum_softmax_probability", target_known_acceptance=0.95
        )
        self.assertEqual(calibration.threshold, 1.0)
        self.assertEqual(calibration.observed_known_acceptance, 0.95)
        np.testing.assert_array_equal(
            decide_known([0.99, 1.0, 2.0], calibration), [False, True, True]
        )

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(UAVOpenSetError):
            maximum_softmax_knownness([[float("nan"), 0.0]])
        with self.assertRaises(UAVOpenSetError):
            energy_knownness([[1.0, 0.0]], temperature=0)
        with self.assertRaises(UAVOpenSetError):
            fit_class_prototypes([[1.0, 2.0]], [0, 1])
        with self.assertRaises(UAVOpenSetError):
            calibrate_known_acceptance([0.5], method="msp", target_known_acceptance=0)


if __name__ == "__main__":
    unittest.main()
