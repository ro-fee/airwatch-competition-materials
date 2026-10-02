import unittest
from airwatch.workflows.uav_contract import (
    QualityStatus, RecognitionStatus, UAVInputInfo, UAVRecognitionResult)

class UAVContractTests(unittest.TestCase):
    def setUp(self):
        self.info = UAVInputInfo("demo.dat", 1e6, 2048, 2)

    def test_completed_known_result_can_publish(self):
        result = UAVRecognitionResult(self.info, QualityStatus.ACCEPTED, "信号质量良好",
            RecognitionStatus.COMPLETED, "目标A", "known", .91, "uav-v0", .2)
        self.assertTrue(result.can_publish_prediction)

    def test_rejected_quality_cannot_publish_prediction(self):
        result = UAVRecognitionResult(self.info, QualityStatus.REJECTED, "信号质量不合格", RecognitionStatus.COMPLETED)
        self.assertFalse(result.can_publish_prediction)

    def test_unknown_is_explicit(self):
        result = UAVRecognitionResult(self.info, QualityStatus.CAUTION, "信号质量一般",
            RecognitionStatus.COMPLETED, "未知信号", "unknown", None)
        self.assertTrue(result.can_publish_prediction)

    def test_invalid_result_is_rejected(self):
        with self.assertRaises(ValueError):
            UAVRecognitionResult(self.info, QualityStatus.REJECTED, "不合格", RecognitionStatus.COMPLETED, "目标A")
        with self.assertRaises(ValueError):
            UAVRecognitionResult(self.info, QualityStatus.ACCEPTED, "良好", RecognitionStatus.PROCESSING, "目标A")
        with self.assertRaises(ValueError):
            UAVRecognitionResult(self.info, QualityStatus.ACCEPTED, "良好", RecognitionStatus.COMPLETED, "目标A", "known", 1.2)
        with self.assertRaises(ValueError):
            UAVRecognitionResult(
                self.info,
                QualityStatus.ACCEPTED,
                "良好",
                RecognitionStatus.COMPLETED,
                window_count=0,
            )
        with self.assertRaises(ValueError):
            UAVRecognitionResult(
                self.info,
                QualityStatus.ACCEPTED,
                "良好",
                RecognitionStatus.COMPLETED,
                limitations=["not-a-tuple"],
            )

if __name__ == "__main__": unittest.main()
