"""Bounded VTI archive inventory; password never serialized or accepted in argv."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path, PurePosixPath
import shutil
from .data_provenance import sha256_file

def safe_member_name(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and ':' not in name and '\\' not in name

def archive_inventory(path, password):
    import py7zr
    with py7zr.SevenZipFile(path, mode='r', password=password) as archive:
        rows = [{'name':item.filename, 'size_bytes':item.uncompressed, 'is_directory':item.is_directory} for item in archive.list()]
        if any(not safe_member_name(row['name']) for row in rows):
            raise ValueError('Unsafe archive member')
        return rows

def audit(path: Path, output: Path, password: str) -> dict:
    import py7zr
    output.mkdir(parents=True, exist_ok=True)
    rows = archive_inventory(path,password)
    needed = sum(row['size_bytes'] for row in rows if not row['is_directory'])
    if shutil.disk_usage(output).free < needed + 2*1024**3:
        raise ValueError('Insufficient bounded extraction space')
    extracted = output / 'outer-members'
    extracted.mkdir(exist_ok=True)
    with py7zr.SevenZipFile(path, mode='r', password=password) as archive:
        archive.extractall(path=extracted)
    inner = []
    for row in rows:
        if row['is_directory']:
            continue
        member = extracted / row['name']
        entry = {**row, 'sha256':sha256_file(member)}
        if member.suffix == '.7z':
            entry['members'] = archive_inventory(member,password)
        inner.append(entry)
    report = {'schema_version':'2.0','status':'password_verified_outer_crc_complete',
              'generated_at_utc':datetime.now(timezone.utc).isoformat(),
              'source_sha256':sha256_file(path),'source_bytes':path.stat().st_size,
              'tool':{'name':'py7zr','version':py7zr.__version__},
              'members':inner,'model_predictions_run':False,'training_eligible':False,
              'compatibility':{'status':'pending_content_inspection','reason':'Nested archive metadata alone does not establish raw complex IQ, sample rate, labels or representation compatibility.'}}
    (output/'vti-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    return report

def inspect_content(output: Path, password: str) -> dict:
    """Inspect every PP/DNN header and one raw header per band, without predictions."""
    import py7zr
    import scipy.io
    import h5py
    report=json.loads((output/'vti-audit.json').read_text(encoding='utf-8'))
    content=[]
    for entry in report['members']:
        package=output/'outer-members'/entry['name']
        if sha256_file(package)!=entry['sha256']:
            raise ValueError('VTI package changed')
        files=[row for row in entry['members'] if not row['is_directory']]
        targets=files[:1] if entry['name'].startswith('00_RAW') else files
        required=sum(row['size_bytes'] for row in targets)
        if shutil.disk_usage(output).free < required+2*1024**3:
            raise ValueError('Insufficient content inspection space')
        with py7zr.SevenZipFile(package,mode='r',password=password) as archive:
            archive.extract(path=output/'content-samples',targets=[x['name'] for x in targets])
        for row in targets:
            path=output/'content-samples'/row['name']
            inspected={'path':row['name'],'size_bytes':path.stat().st_size,'sha256':sha256_file(path)}
            if path.suffix=='.mat':
                if h5py.is_hdf5(path):
                    variables=[]
                    with h5py.File(path,'r') as handle:
                        def visit(name,obj):
                            if isinstance(obj,h5py.Dataset):
                                variables.append({'name':name,'shape':list(obj.shape),'dtype':str(obj.dtype)})
                        handle.visititems(visit)
                    inspected.update(format='HDF5',variables=variables)
                else:
                    inspected.update(format='MAT',variables=[{'name':name,'shape':list(shape),'class':kind} for name,shape,kind in scipy.io.whosmat(path)])
                    scalar_names=[v['name'] for v in inspected['variables'] if v['shape']==[1,1] and v['class']!='struct']
                    values=scipy.io.loadmat(path,variable_names=scalar_names)
                    inspected['scalar_metadata']={name:float(values[name].reshape(-1)[0]) for name in scalar_names}
            elif path.suffix=='.csv':
                with path.open(encoding='utf-8-sig',newline='') as handle:
                    first=handle.readline()
                    tail=deque(maxlen=3)
                    row_count=1
                    for line in handle:
                        tail.append(line)
                        row_count+=1
                inspected.update(format='CSV',rows=row_count,first_row_columns=first.count(',')+1,
                                 first_row_is_numeric=all(_is_number(v) for v in first.strip().split(',')[:64]),
                                 last_three_row_unique_values=[sorted(set(line.strip().split(',')))[:32] for line in tail],
                                 label_row_semantics='requires author metadata; not inferred from model predictions')
            content.append(inspected)
        print(f'Inspected metadata: {entry["name"]}',flush=True)
    report.update(content_inspection=content,raw_content_scope='one member per band; all raw filenames inventoried',
                  aggregate_content_scope='all PP MATLAB headers and all DNN CSV row/column counts',
                  declared_sample_rate_hz=150000000,model_sample_rate_hz=100000000)
    report['compatibility']={'status':'blocked_incompatible_representation','compatible':False,
        'reason':'The frozen model requires 100 MS/s single-band complex IQ. VTI declares 150 MS/s and different device/multi-drone labels; no independently specified and validated resampling/label contract is available. PP/DNN matrices cannot be assumed equivalent IQ. Audit is complete, model metrics are conditional and not produced.',
        'sample_rate_source':'Mendeley Data VTI_DroneSET V1 official description, DOI 10.17632/s6tgnnp5n2.1',
        'model_predictions_run':False}
    (output/'vti-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report

def _is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inspect-content',action='store_true')
    args=parser.parse_args()
    password=getpass.getpass('VTI archive password (not logged): ')
    try:
        report=inspect_content(args.output,password) if args.inspect_content else audit(args.archive,args.output,password)
    except Exception:
        print('VTI audit failed; no password or exception payload logged.',flush=True)
        raise SystemExit(1)
    finally:
        password=None
    print(json.dumps({'status':report['status'],'member_count':len(report['members'])}),flush=True)

if __name__=='__main__':
    main()
