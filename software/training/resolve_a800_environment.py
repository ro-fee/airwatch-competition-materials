"""Resolve Linux wheel closure from Windows, explicitly evaluating Linux markers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT_REQUIREMENTS=['torch==2.14.0+cu126','numpy==2.2.6','scipy==1.15.3','h5py==3.16.0','scikit-learn==1.7.2','matplotlib==3.10.9']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--finish-existing',action='store_true',help='Validate and finish the saved completed resolution without choosing newer package versions')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    environment=default_environment()
    environment.update(os_name='posix',sys_platform='linux',platform_system='Linux',platform_machine='x86_64',python_version='3.10',python_full_version='3.10.21',implementation_name='cpython',platform_release='not supplied by server',platform_version='not supplied by server')
    requirements=set(ROOT_REQUIREMENTS)
    extras={}
    if args.finish_existing:
        requirements.update((args.output/'resolver-input.txt').read_text(encoding='utf-8').splitlines())
    for iteration in range(12):
        for text in requirements:
            req=Requirement(text)
            extras.setdefault(canonicalize_name(req.name),set()).update(req.extras)
        input_path=args.output/'resolver-input.txt'
        input_path.write_text('\n'.join(sorted(requirements))+'\n',encoding='utf-8')
        report_path=args.output/'linux-resolve.json'
        command=[sys.executable,'-m','pip','install','--dry-run','--ignore-installed','--only-binary=:all:',
                 '--python-version','3.10','--implementation','cp','--abi','cp310','--report',str(report_path),
                 '--extra-index-url','https://download.pytorch.org/whl/cu126/','-r',str(input_path)]
        # Ubuntu 22.04 glibc is 2.35: include every compatible PEP600 tag,
        # because cross-platform pip does not expand 2.28 to e.g. 2.27.
        for platform in [f'manylinux_2_{version}_x86_64' for version in range(35,4,-1)]+['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64','linux_x86_64']:
            command += ['--platform',platform]
        if not args.finish_existing:
            subprocess.run(command,check=True)
        report=json.loads(report_path.read_text(encoding='utf-8'))
        expanded=set(requirements)
        for item in report['install']:
            name=canonicalize_name(item['metadata']['name'])
            for text in item['metadata'].get('requires_dist',[]):
                req=Requirement(text)
                if req.marker is None or any(req.marker.evaluate({**environment,'extra':extra}) for extra in {''}|extras.get(name,set())):
                    expanded.add(str(req).split(';')[0].strip())
        if expanded==requirements:
            break
        requirements=expanded
    else:
        raise RuntimeError('Linux dependency closure did not converge')
    installed={canonicalize_name(item['metadata']['name']):item for item in report['install']}
    for text in requirements:
        req=Requirement(text);item=installed.get(canonicalize_name(req.name))
        if item is None or not req.specifier.contains(item['metadata']['version'],prereleases=True):
            raise ValueError(f'Incomplete Linux dependency closure: {req.name}')
    lines=['# Candidate Linux x86_64 CPython 3.10 closure. Verified metadata/hashes; server execution pending.']
    for name,item in sorted(installed.items()):
        info=item['download_info'];sha=info.get('archive_info',{}).get('hashes',{}).get('sha256')
        if not sha:
            # Some index pages omit digest fragments. Hash the actual wheel,
            # retaining its exact URL rather than inventing an index checksum.
            digest=hashlib.sha256();size=0
            with urllib.request.urlopen(info['url'],timeout=90) as handle:
                while chunk:=handle.read(1024**2):
                    size+=len(chunk)
                    if size>128*1024**2:
                        raise RuntimeError('Unhashed wheel exceeds bounded metadata repair size')
                    digest.update(chunk)
            sha=digest.hexdigest()
            info.setdefault('archive_info',{}).setdefault('hashes',{})['sha256']=sha
            info['digest_provenance']='downloaded exact wheel and computed SHA-256'
        lines.append(f'{name} @ {info["url"]} --hash=sha256:{sha}')
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (args.output/'requirements-linux-lock.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (args.output/'requirements-training.txt').write_text('\n'.join(ROOT_REQUIREMENTS)+'\n',encoding='utf-8')
    summary={'status':'candidate_linux_metadata_closure_verified','target_environment':environment,'packages':len(installed),
             'server_runtime_verified':False,'linux_markers_explicitly_evaluated':True,'iterations':iteration+1}
    (args.output/'candidate-status.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
