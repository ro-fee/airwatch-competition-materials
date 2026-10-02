import unittest
from training.diagnose_ku_leuven_noise_errors import summarize


class NoiseErrorTests(unittest.TestCase):
    def test_confident_errors_and_recall(self):
        rows = [dict(target=0, prediction=1, confidence=0.95),
                dict(target=1, prediction=1, confidence=0.8),
                dict(target=2, prediction=2, confidence=0.7)]
        result = summarize(rows)
        self.assertEqual(result['errors_with_confidence_at_least_0_9'], 1)
        self.assertEqual(result['mean_error_confidence'], 0.95)
        self.assertEqual(result['per_class_recall']['frysky'], 0)
        self.assertEqual(result['per_class_recall']['dji_mini2_rc'], 1)

    def test_no_errors_has_no_error_confidence(self):
        self.assertIsNone(summarize([dict(target=0, prediction=0, confidence=0.6)])['mean_error_confidence'])
