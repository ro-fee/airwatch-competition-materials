"""Build and verify a portable immutable upload payload (stdlib only)."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath,PureWindowsPath
import shutil
import tarfile

REQUIRED_TESTS={'test_a800_data','test_a800_data_integrity','test_a800_bundle','test_a800_bundle_review',
                'test_a800_runner','test_a800_allocation','test_a800_evaluation','test_a800_delivery','test_a800_objective',
                'test_uav_multiscale_tcn','test_uav_a800_contract'}
REQUIRED_CODE=['training/a800_runner.py','training/a800_common.py','training/a800_evaluation.py','training/a800_delivery.py',
               'training/a800_cpu_checks.py','training/a800_environment.py','training/build_a800_bundle.py',
               'training/configs/a800_v2/matrix.json','airwatch/models/uav_multiscale_tcn.py',
               'airwatch/data/ku_leuven_training_v2.py','airwatch/inference/uav_a800_contract.py']
REQUIRED_SCRIPTS=['scripts/'+name for name in ('bootstrap_server.sh','run_cpu_checks.sh','run_allocated_gpu.sh','export_results.sh')]

def require_payload(root):
    required=REQUIRED_CODE+REQUIRED_SCRIPTS+['tests/'+name+'.py' for name in REQUIRED_TESTS]
    required+=['START_HERE.md','MODEL_TRAINING_PLAN.md','AGENTS.md','environment/requirements-linux-lock.txt',
               'environment/linux-resolve.json','environment/candidate-status.json','manifests/sealed-evaluation.json',
               'manifests/vti-audit.json','manifests/known-split-v1/known-split-assignments.csv']
    required+=['data/raw/'+name for name in ('Frysky.zip','Spektrum_DX4e.zip','mini2RC.zip','SJRC_pro.zip','NineEagles.zip','Q205.zip','wltoys.zip','VTI_DroneSET.7z')]
    for dataset in ('known-iq-v1','dense-train-v2'):
        required.append(f'data/prepared/{dataset}/dataset-metadata.json')
        for split in ('train','validation'):
            required += [f'data/prepared/{dataset}/{split}/{name}' for name in ('data.npy','labels.npy','windows.csv')]
    missing=[name for name in required if not (root/name).is_file()]
    if missing:
        raise ValueError('Required bundle files missing: '+', '.join(missing))

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024**2),b''):
            digest.update(chunk)
    return digest.hexdigest()

def json_write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def payload_files(root):
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if path.is_symlink():
            raise ValueError('Symbolic links are not supported in immutable bundle payload: '+str(relative))
        if not path.is_file() or path.is_symlink() or '__pycache__' in relative.parts or relative.parts[0]=='outputs' or path.name.endswith(('.pyc','.tmp')) or relative.as_posix() in {'bundle-files.json','SHA256SUMS'}:
            continue
        yield path

def write_integrity(root,status):
    root=Path(root)
    json_write(root/'bundle-status.json',status)
    records=[{'path':path.relative_to(root).as_posix(),'size_bytes':path.stat().st_size,'sha256':sha256(path)} for path in payload_files(root)]
    payload={'schema_version':'2.0','scope':'immutable upload payload; outputs excluded','files':records,
             'total_bytes':sum(row['size_bytes'] for row in records)}
    json_write(root/'bundle-files.json',payload)
    (root/'SHA256SUMS').write_text(sha256(root/'bundle-files.json')+'  bundle-files.json\n',encoding='ascii')
    return payload

def verify_integrity(root):
    root=Path(root).resolve()
    expected=(root/'SHA256SUMS').read_text(encoding='ascii').strip()
    if expected!=sha256(root/'bundle-files.json')+'  bundle-files.json':
        raise ValueError('Bundle index checksum mismatch')
    index=json.loads((root/'bundle-files.json').read_text(encoding='utf-8'))
    seen=set()
    for item in index['files']:
        name=item['path'];relative=PurePosixPath(name)
        path=(root/name).resolve()
        if not name or name in seen or relative.is_absolute() or PureWindowsPath(name).drive or '..' in relative.parts or '\\' in name or not path.is_relative_to(root):
            raise ValueError('Unsafe or duplicated bundle member')
        seen.add(name)
        if not path.is_file() or path.stat().st_size!=item['size_bytes'] or sha256(path)!=item['sha256']:
            raise ValueError('Bundle payload mismatch: '+name)
    actual={p.relative_to(root).as_posix() for p in payload_files(root)}
    if seen!=actual:
        raise ValueError('Unregistered immutable payload files')
    return {'ok':True,'files':len(seen),'total_bytes':index['total_bytes']}

def copy_file(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)

def stage(workspace,root,environment,refresh_sources=False):
    workspace=Path(workspace).resolve();root=Path(root).resolve()
    if root==workspace or root.is_relative_to(workspace):
        raise ValueError('Independent bundle directory required')
    status_path=root/'bundle-status.json'
    if status_path.exists() and json.loads(status_path.read_text(encoding='utf-8')).get('ready'):
        raise ValueError('Do not stage over a published upload bundle')
    if (root/'outputs/runs').exists():
        raise ValueError('Do not stage over a directory containing training runs')
    # Default repeated staging is idempotent only: changed source requires an
    # explicit preparation refresh and another independent check/index.
    for name in ('airwatch','training'):
        for source in (workspace/name).rglob('*'):
            target=root/source.relative_to(workspace)
            if source.is_file() and source.suffix in {'.py','.json'} and '__pycache__' not in source.parts and target.is_file() and sha256(source)!=sha256(target) and not refresh_sources:
                raise FileExistsError('Staged source differs; use explicit preparation refresh: '+str(target))
    required=REQUIRED_CODE+REQUIRED_SCRIPTS+['tests/'+name+'.py' for name in REQUIRED_TESTS]
    required+=['docs/superpowers/plans/2026-09-29-a800-server-training-plan.md','docs/A800_ClaudeCode执行指南.md','training/A800_SERVER_AGENTS.md']
    missing=[name for name in required if not (workspace/name).is_file()]
    missing += [str(Path(environment)/name) for name in ('requirements-linux-lock.txt','linux-resolve.json','candidate-status.json') if not (Path(environment)/name).is_file()]
    if missing:
        raise ValueError('Staging inputs incomplete: '+', '.join(missing))
    for name in ('airwatch','training'):
        for source in (workspace/name).rglob('*'):
            if source.is_file() and source.suffix in {'.py','.json'} and '__pycache__' not in source.parts:
                # Qt pages/workflows and unrelated old training runners are unnecessary for execution.
                if name=='airwatch' and any(part in {'ui','workflows'} for part in source.relative_to(workspace/name).parts):
                    continue
                copy_file(source,root/source.relative_to(workspace))
    for source in (workspace/'tests').glob('*.py'):
        if source.name.startswith('test_a800') or source.name in {'__init__.py','test_uav_multiscale_tcn.py','test_uav_a800_contract.py'}:
            copy_file(source,root/'tests'/source.name)
    for name in ('bootstrap_server.sh','run_cpu_checks.sh','run_allocated_gpu.sh','export_results.sh'):
        source=workspace/'scripts'/name
        if source.is_file():
            copy_file(source,root/'scripts'/name)
    for source in Path(environment).glob('*'):
        if source.is_file() and source.suffix in {'.txt','.json'}:
            copy_file(source,root/'environment'/source.name)
    doc_map={'docs/superpowers/plans/2026-09-29-a800-server-training-plan.md':'MODEL_TRAINING_PLAN.md',
             'docs/A800_ClaudeCode执行指南.md':'START_HERE.md','training/A800_IMPLEMENTATION_CONTRACT.md':'docs/implementation-record.md',
             'training/A800_SERVER_AGENTS.md':'AGENTS.md',
             'docs/项目说明.md':'docs/项目说明.md','docs/参赛改造方案.md':'docs/参赛改造方案.md',
             'docs/开发与验收规范.md':'docs/开发与验收规范.md'}
    for src,dst in doc_map.items():
        if (workspace/src).is_file():
            copy_file(workspace/src,root/dst)
    evidence_dir=workspace/'artifacts/evidence/uav/ku_leuven/local_development'
    for source in evidence_dir.glob('*.json'):
        copy_file(source,root/source.relative_to(workspace))
    for source in (workspace/'artifacts/checkpoints/uav/ku_leuven/local_development').glob('*v4_best.pt'):
        copy_file(source,root/source.relative_to(workspace))
    for source in (workspace/'datasets/uav/ku_leuven_drone_rf').glob('*.json'):
        copy_file(source,root/'provenance/ku-leuven'/source.name)
    for source in (workspace/'datasets/uav/vti_droneset').glob('*.json'):
        copy_file(source,root/'provenance/vti'/source.name)
    return {'stage':'source and metadata copied','root':str(root)}

def finalize(root):
    root=Path(root)
    require_payload(root)
    verify_integrity(root)
    checks=json.loads((root/'outputs/evidence/environment/cpu-checks.json').read_text(encoding='utf-8'))
    environment=json.loads((root/'environment/candidate-status.json').read_text(encoding='utf-8'))
    inventory=checks.get('executed_test_modules',{})
    if checks['status']!='passed' or checks['skipped_tests']!=0 or checks.get('unit_tests',0)<=0 or not REQUIRED_TESTS.issubset(inventory) or any(inventory[name]<=0 for name in REQUIRED_TESTS) or not environment['linux_markers_explicitly_evaluated']:
        raise ValueError('Required independent preparation evidence incomplete')
    if checks.get('bundle_index_sha256')!=sha256(root/'bundle-files.json'):
        raise ValueError('Independent CPU evidence is not bound to this payload index')
    failure=root/'outputs/evidence/environment/cpu-checks-failed.json'
    if failure.exists() and failure.stat().st_mtime_ns >= (root/'outputs/evidence/environment/cpu-checks.json').stat().st_mtime_ns:
        raise ValueError('Latest CPU check failed; do not reuse an older success')
    copy_file(root/'outputs/evidence/environment/cpu-checks.json',root/'preparation-evidence/independent-cpu-checks.json')
    status={'schema_version':'2.0','status':'ready_for_upload','ready':True,'ready_for_upload':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'local_preparation_verified':True,'independent_cpu_tests':checks['unit_tests'],'server_linux_cuda_verified':False,
            'server_bootstrap_required':True,'actual_gpu_allocation_required':True,'training_started':False,
            'vti_external_metrics':'blocked_incompatible_representation','metrics_claim':'No A800 training or test result has been generated locally.'}
    return write_integrity(root,status)

def package(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError('Archive must be a new file outside the bundle directory')
    verify_integrity(root)
    status=json.loads((root/'bundle-status.json').read_text(encoding='utf-8'))
    if not status.get('ready'):
        raise ValueError('Bundle is not ready')
    paths=list(payload_files(root))+[root/'bundle-files.json',root/'SHA256SUMS']
    output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'w') as archive:
        for path in paths:
            archive.add(path,arcname=(Path('airwatch-a800')/path.relative_to(root)).as_posix(),recursive=False)
    digest=sha256(output)
    output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n',encoding='ascii')
    return {'archive':str(output),'size_bytes':output.stat().st_size,'sha256':digest,'payload_files':len(paths)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ('stage','index','verify','finalize','package'):
        entry=sub.add_parser(cmd);entry.add_argument('--root',type=Path,required=True)
        if cmd=='stage':
            entry.add_argument('--workspace',type=Path,required=True);entry.add_argument('--environment',type=Path,required=True)
            entry.add_argument('--refresh-sources',action='store_true')
        if cmd=='package':
            entry.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='stage':result=stage(args.workspace,args.root,args.environment,args.refresh_sources)
    elif args.command=='index':result=write_integrity(args.root,{'schema_version':'2.0','status':'preparing','ready':False})
    elif args.command=='verify':result=verify_integrity(args.root)
    elif args.command=='finalize':result=finalize(args.root)
    else:result=package(args.root,args.output)
    if 'files' in result and isinstance(result['files'],list):
        result={'files':len(result['files']),'total_bytes':result['total_bytes']}
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
