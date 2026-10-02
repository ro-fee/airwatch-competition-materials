"""A deliberately stopped soak must not produce a successful process exit."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path

class SoakDriverExitTests(unittest.TestCase):
    def test_resource_gate_restarts_import_before_curve_assertion(self):
        root=Path(__file__).resolve().parents[1]
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); inputs=parent/'inputs'; inputs.mkdir(); out=parent/'run'
            t=np.arange(16384)/48000
            iq=np.exp(-2j*np.pi*6000*t)
            for name, data in {'tone':np.sin(2*np.pi*6000*t), 'negative_iq':iq,
                               'iq_rows':np.array([iq.real,iq.imag]), 'nan':np.array([np.nan])}.items():
                np.save(inputs/(name+'.npy'),data)
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen')
            result=subprocess.run([sys.executable,str(root/'tools/ui_stress_session.py'),
                '--output',str(out),'--seconds','4','--cycles','1',
                '--simulate-resource-gate-step','1'],cwd=root,env=env,capture_output=True,text=True,timeout=60)
            summary=json.loads((out/'summary.json').read_text(encoding='utf8'))
            events=[json.loads(line) for line in (out/'operations.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual(summary['status'],'PASS',summary)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(any(e['event']=='resource_cycle_rewind' for e in events))

    def test_stop_before_acceptance_exits_nonzero(self):
        root=Path(__file__).resolve().parents[1]
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);(parent/'inputs').mkdir();out=parent/'run';out.mkdir()
            (out/'STOP').write_text('intentional early stop',encoding='utf8')
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen')
            result=subprocess.run([sys.executable,str(root/'tools/ui_stress_session.py'),'--output',str(out),'--seconds','60','--cycles','1'],cwd=root,env=env,capture_output=True,text=True,timeout=40)
            summary=json.loads((out/'summary.json').read_text(encoding='utf8'))
            self.assertEqual(summary['status'],'FAIL')
            self.assertNotEqual(result.returncode,0,'failed soak incorrectly signalled success to CI')

if __name__=='__main__':unittest.main()
