"""Verify the installed Linux environment against the immutable wheel resolution."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

PREFIX=Path('/root/lyh_workspace/envs/airwatch-a800')

def verify_environment(root,require_linux=False):
    root=Path(root)
    if require_linux:
        if platform.system()!='Linux' or platform.machine()!='x86_64' or sys.version_info[:3]!=(3,10,21) or Path(sys.prefix).resolve()!=PREFIX.resolve():
            raise RuntimeError('Use the dedicated approved Linux prefix with Python 3.10.21')
    report=json.loads((root/'environment/linux-resolve.json').read_text(encoding='utf-8'))
    actual={}
    if require_linux:
        for item in report['install']:
            name=item['metadata']['name'];expected=item['metadata']['version']
            version=importlib.metadata.version(name)
            if version!=expected:
                raise RuntimeError(f'Pinned package drift: {name}: {version} != {expected}; existing environment left unchanged')
            actual[name]=version
    return {'platform':platform.platform(),'python':sys.version,'prefix':sys.prefix,'packages':actual,'linux_environment_verified':require_linux}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--require-linux',action='store_true')
    args=parser.parse_args()
    result=verify_environment(args.root,args.require_linux)
    output=args.root/'outputs/evidence/environment/installed-environment.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
