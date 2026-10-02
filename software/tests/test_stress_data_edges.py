"""Real malformed files and configured-budget boundaries, not model accuracy."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from airwatch.data.home_input import load_home_signal
from airwatch.data.recognition_input import load_recognition_files
from airwatch.general_recognition_contract import TASKS


class StressDataEdgesTests(unittest.TestCase):
    def test_raw_files_reject_partial_scalar_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            for task,spec in TASKS.items():
                with self.subTest(task=task):
                    p=Path(directory)/'truncated.dat'
                    good=np.ones(spec.window_size*spec.channels,dtype=spec.sample_dtype).tobytes()
                    p.write_bytes(good+b'\x01')
                    with self.assertRaisesRegex(ValueError,'字节|完整|截断'):
                        load_recognition_files(task,[p])

    def test_home_rejects_truncated_npy_and_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'signal.npy'
            np.save(p,np.arange(1024.))
            data=p.read_bytes(); p.write_bytes(data[:-1])
            with self.assertRaises(ValueError): load_home_signal(p)
            p.write_bytes(data)
            np.testing.assert_array_equal(load_home_signal(p),np.arange(1024.))

    def test_coding_window_budget_before_at_and_after(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'budget.dat'
            for count,accepted in ((2047*512,True),(2048*512,True),(2049*512,False)):
                with self.subTest(count=count):
                    np.ones(count,dtype=np.int16).tofile(p)
                    if accepted:
                        result=load_recognition_files('信号编码识别',[p])
                        self.assertEqual(result.arrays[0].shape,(1,count))
                    else:
                        with self.assertRaisesRegex(ValueError,'窗口上限'):
                            load_recognition_files('信号编码识别',[p])


if __name__=='__main__': unittest.main()
