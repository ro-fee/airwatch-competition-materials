import unittest
import numpy as np
from airwatch.analysis.uav_quality import UAVQualityStatus, assess_uav_signal
class UAVQualityTests(unittest.TestCase):
 def test_good_signal_is_accepted(self):
  r=assess_uav_signal(np.sin(np.arange(128)/5)); self.assertEqual(r.status,UAVQualityStatus.ACCEPTED); self.assertTrue(r.has_signal_variation)
 def test_zero_short_and_nonfinite_are_rejected(self):
  for x in (np.zeros(128),np.ones(4),np.array([1.,np.nan])):
   self.assertEqual(assess_uav_signal(x).status,UAVQualityStatus.REJECTED)
 def test_full_scale_clipping_is_caution(self):
  x=np.r_[np.ones(20),np.linspace(-.5,.5,108)]; self.assertEqual(assess_uav_signal(x).status,UAVQualityStatus.CAUTION)
 def test_arbitrary_physical_units_are_not_called_clipped(self):
  r=assess_uav_signal(np.linspace(-100,100,128)); self.assertEqual(r.clipped_fraction,0.)
if __name__=='__main__': unittest.main()
