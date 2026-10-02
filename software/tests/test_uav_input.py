import tempfile, unittest
from pathlib import Path
import numpy as np
from airwatch.data.uav_input import UAVFileSpec, load_uav_signal

class UAVInputTests(unittest.TestCase):
    def test_npy_real_and_complex_are_owned_float32(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.npy"; np.save(p, np.array([1,2,3], dtype=np.float64))
            signal=load_uav_signal(p)
            self.assertEqual(signal.samples.dtype, np.float32); self.assertEqual(signal.sample_count,3)
            self.assertFalse(signal.samples.flags.writeable)
            q=Path(d)/"iq.npy"; np.save(q, np.array([1+2j,3+4j]))
            iq=load_uav_signal(q)
            self.assertEqual(iq.samples.shape,(2,2)); np.testing.assert_allclose(iq.samples[:,0],[1,2])
    def test_npy_channel_last_and_windowed_iq_share_one_canonical_shape(self):
        with tempfile.TemporaryDirectory() as d:
            channel_last=Path(d)/"channel-last.npy"
            np.save(channel_last,np.arange(16,dtype=np.float32).reshape(8,2))
            actual=load_uav_signal(channel_last)
            self.assertEqual(actual.samples.shape,(2,8))
            windowed=Path(d)/"windowed.npy"
            source=np.arange(32,dtype=np.float32).reshape(2,2,8)
            np.save(windowed,source)
            prepared=load_uav_signal(windowed)
            self.assertEqual(prepared.samples.shape,(2,16))
            np.testing.assert_array_equal(
                prepared.samples,
                source.transpose(1,0,2).reshape(2,16),
            )
    def test_dat_requires_explicit_spec_and_supports_interleaved_iq(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.dat"; np.array([1,10,2,20],dtype=np.int16).tofile(p)
            with self.assertRaises(ValueError): load_uav_signal(p)
            s=load_uav_signal(p,UAVFileSpec("int16",2,"interleaved_iq",1e6))
            self.assertEqual(s.samples.tolist(), [[1,2],[10,20]])
            self.assertEqual(s.sample_rate_hz,1e6)
    def test_rejects_bad_shape_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bad.npy"; np.save(p,np.ones((3,4,5)))
            with self.assertRaises(ValueError): load_uav_signal(p)
            q=Path(d)/"nan.npy"; np.save(q,np.array([np.nan]))
            with self.assertRaises(ValueError): load_uav_signal(q)
    def test_spec_validation(self):
        with self.assertRaises(ValueError): UAVFileSpec("int16",2,"single")
        with self.assertRaises(ValueError): UAVFileSpec("int16",1,"single",0)

if __name__ == "__main__": unittest.main()
