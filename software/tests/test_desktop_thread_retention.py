"""Native Windows OpenMP pools must not accumulate with short-lived Qt jobs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(os.name=='nt' and importlib.util.find_spec('psutil'), 'Windows native-thread probe needs psutil')
class DesktopThreadRetentionTests(unittest.TestCase):
    def test_repeated_desktop_model_loads_do_not_retain_a_pool_per_job(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,str(root/'tools/probe_thread_growth.py'),
                               '--desktop','--mode','load','--rounds','12'],
                              cwd=root,capture_output=True,text=True,timeout=60,check=True)
        rows=json.loads(result.stdout)['rows']
        growth=rows[-1]['threads']-rows[2]['threads']
        self.assertLessEqual(growth,3,f'Native thread retention across equivalent idle jobs: {rows}')
        self.assertLessEqual(rows[-1]['handles']-rows[2]['handles'],8)


if __name__=='__main__': unittest.main()
