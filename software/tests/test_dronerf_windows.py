import unittest

import numpy as np

from airwatch.data.dronerf_windows import (
    DroneRFWindowSpec,
    DroneRFWindowingError,
    _sample_csv_chunks,
    stratified_window_offsets,
)


class DroneRFWindowTests(unittest.TestCase):
    def test_offsets_are_stable_sorted_and_cover_strata(self):
        spec = DroneRFWindowSpec(window_length=16, windows_per_recording=4,
                                 expected_sample_count=100, seed=7)
        first = stratified_window_offsets("recording-a", spec)
        self.assertEqual(first, stratified_window_offsets("recording-a", spec))
        self.assertNotEqual(first, stratified_window_offsets("recording-b", spec))
        self.assertEqual(tuple(sorted(first)), first)
        self.assertEqual(len(set(first)), 4)
        self.assertTrue(all(0 <= item <= 84 for item in first))

    def test_stream_sampler_handles_tokens_split_across_chunks(self):
        values = np.arange(30, dtype=np.float32)
        payload = ",".join(str(float(item)) for item in values).encode("ascii")
        chunks = [payload[:7], payload[7:19], payload[19:41], payload[41:]]
        windows, count = _sample_csv_chunks(
            chunks, (2, 17), 5, expected_sample_count=30
        )
        self.assertEqual(count, 30)
        np.testing.assert_allclose(windows[0], values[2:7])
        np.testing.assert_allclose(windows[1], values[17:22])

    def test_stream_sampler_supports_trailing_comma(self):
        windows, count = _sample_csv_chunks(
            [b"1,2,", b"3,4,"], (1,), 2, expected_sample_count=4
        )
        self.assertEqual(count, 4)
        np.testing.assert_allclose(windows, [[2, 3]])

    def test_stream_sampler_rejects_wrong_length_and_nonfinite_window(self):
        with self.assertRaises(DroneRFWindowingError):
            _sample_csv_chunks([b"1,2,3"], (0,), 2, expected_sample_count=4)
        with self.assertRaises(DroneRFWindowingError):
            _sample_csv_chunks([b"1,nan,3"], (0,), 2, expected_sample_count=3)

    def test_spec_rejects_impossible_values(self):
        with self.assertRaises(DroneRFWindowingError):
            DroneRFWindowSpec(window_length=8)
        with self.assertRaises(DroneRFWindowingError):
            DroneRFWindowSpec(window_length=32, expected_sample_count=16)


if __name__ == "__main__":
    unittest.main()
