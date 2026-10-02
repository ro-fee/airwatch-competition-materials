"""Required portable CPU checks: missing real data or skipped tests is a failure."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import unittest
import numpy as np
import torch
from airwatch.data.ku_leuven_dataset import KULeuvenMaterializedDataset
from airwatch.data.ku_leuven_training_v2 import A800Dataset
from training.a800_common import atomic_json
from training.a800_data import FROZEN_SPLIT_SHA256, read_csv
from training.a800_environment import verify_environment
from training.build_a800_bundle import REQUIRED_TESTS, sha256, verify_integrity

def check_real_data(root):
    import hashlib
    split_path=root/'manifests/known-split-v1/known-split-assignments.csv'
    if hashlib.sha256(split_path.read_bytes()).hexdigest()!=FROZEN_SPLIT_SHA256:
        raise ValueError('Frozen split changed')
    assignments={row['recording_id']:row for row in read_csv(split_path)}
    sets={}
    for name,split,cls,count in [('known-iq-v1','train',KULeuvenMaterializedDataset,3680),('known-iq-v1','validation',KULeuvenMaterializedDataset,768),('dense-train-v2','train',A800Dataset,29440),('dense-train-v2','validation',A800Dataset,768)]:
        with cls(root/'data/prepared'/name,split=split) as dataset:
            if len(dataset)!=count:
                raise ValueError('Required real dataset count mismatch')
            for sample in dataset.samples:
                assignment=assignments[sample.recording_id]
                if assignment['split']!=split or any(str(getattr(sample,key))!=str(assignment[key]) for key in ('archive_id','label_index','member_index','member_path','member_sha256')):
                    raise ValueError('Real dataset recording provenance mismatch')
            if split=='train':
                sets[name]=dataset.recording_ids
            tensor,label,index=dataset[0]
            if tensor.shape!=(2,4096) or not torch.isfinite(tensor).all():
                raise ValueError('Real data smoke failed')
    if sets['known-iq-v1']!=sets['dense-train-v2']:
        raise ValueError('Dense recording membership changed')
    sealed=json.loads((root/'manifests/sealed-evaluation.json').read_text(encoding='utf-8'))
    if len(sealed['known_test'])!=24 or len(sealed['unknown'])!=204:
        raise ValueError('Sealed test/unknown metadata inventory mismatch')
    if set(x['recording_id'] for x in sealed['known_test']+sealed['unknown']) & sets['known-iq-v1']:
        raise ValueError('Sealed metadata overlaps training')
    return {'train_members':115,'train_windows_v1':3680,'train_windows_v2':29440,'validation_windows':768,'sealed_known_test_members':24,'sealed_unknown_members':204,'sealed_signal_samples_read':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--require-linux',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    import airwatch
    if not Path(airwatch.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Independent check imported code outside bundle root')
    verify_environment(root,args.require_linux)
    if args.require_linux and (platform.system()!='Linux' or platform.machine()!='x86_64' or sys.version_info[:3]!=(3,10,21)):
        raise RuntimeError('Wrong server platform/Python')
    torch.set_num_threads(4)
    if args.require_linux and torch.__version__!='2.14.0+cu126':
        raise RuntimeError('Wrong pinned PyTorch build')
    tests=sorted(path.stem for path in (root/'tests').glob('test_a800*.py'))+['test_uav_multiscale_tcn','test_uav_a800_contract']
    required=REQUIRED_TESTS
    if not required.issubset(tests):
        raise RuntimeError('Required test file missing')
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.'+name for name in tests])
    if suite.countTestCases()<=0:
        raise RuntimeError('Zero required tests discovered')
    verify_integrity(root)
    checked_index=sha256(root/'bundle-files.json')
    inventory=Counter()
    class InventoryResult(unittest.TextTestResult):
        def startTest(self,test):
            module=test.__class__.__module__.split('.')[-1]
            inventory[module]+=1
            super().startTest(test)
    result=unittest.TextTestRunner(verbosity=1,resultclass=InventoryResult).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun<=0 or not required.issubset(inventory) or any(inventory[name]<=0 for name in required):
        raise RuntimeError('Required tests failed or were skipped')
    data=check_real_data(root)
    packages={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','h5py','scikit-learn','matplotlib')}
    artifact={'status':'passed','platform':platform.platform(),'python':sys.version,'python_executable':sys.executable,
              'packages':packages,'unit_tests':result.testsRun,'skipped_tests':len(result.skipped),'data':data,
              'executed_test_modules':dict(inventory),'bundle_index_sha256':checked_index,
              'cuda_training_verified':False,'claim_scope':'CPU preparation check only'}
    atomic_json(root/'outputs/evidence/environment/cpu-checks.json',artifact)
    print(json.dumps(artifact,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        if '--root' in sys.argv:
            failure_root=Path(sys.argv[sys.argv.index('--root')+1])
            atomic_json(failure_root/'outputs/evidence/environment/cpu-checks-failed.json',{'status':'failed','error_type':type(error).__name__,'message':str(error)})
        raise
