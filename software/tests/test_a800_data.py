import unittest
from airwatch.data.ku_leuven_training_v2 import dense_offsets
from airwatch.data.vti_audit import safe_member_name

class DensePlanTests(unittest.TestCase):
    def test_original_windows_preserved_and_new_windows_disjoint(self):
        old = [i * 300000 + 91 for i in range(32)]
        starts = dense_offsets('example', 10000000, old)
        self.assertEqual(len(starts), 256)
        self.assertTrue(set(old).issubset(starts))
        self.assertEqual(starts, dense_offsets('example', 10000000, old))
        self.assertTrue(all(b >= a + 4096 for a, b in zip(starts, starts[1:])))
        self.assertNotEqual(starts, dense_offsets('other', 10000000, old))

    def test_reject_bad_original_count_or_overlap(self):
        for old in ([0], list(range(32))):
            with self.assertRaises(ValueError):
                dense_offsets('example', 10000000, old)

    def test_safe_archive_names(self):
        for name in ('../escape', '/root', 'C:/escape', 'a/../../b', 'a\\b', 'a:stream'):
            self.assertFalse(safe_member_name(name))
        self.assertTrue(safe_member_name('raw/drone_01.mat'))

if __name__ == '__main__':
    unittest.main()
