"""Prepare dense known data and sealed metadata; contains no model selection."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import zipfile
import numpy as np
from airwatch.data.data_provenance import sha256_file
from airwatch.data.ku_leuven_training_v2 import dense_offsets, A800Dataset
from airwatch.data.ku_leuven_dataset import KULeuvenMaterializedDataset
from airwatch.data.ku_leuven_preprocessing import PREPROCESSING_ID, preprocess_iq_window
from airwatch.data.materialize_ku_leuven import materialize_known_windows
from airwatch.data.matlab_v73_iq import read_mat_v73_iq_windows_fileobj, inspect_mat_v73_iq_fileobj
from airwatch.data.ku_leuven_manifest import KULeuvenArchiveSpec, KULeuvenWindowSpec, build_manifest_bundle, write_manifest_bundle

KNOWN = [('frysky-v1','Frysky.zip','frysky-manifest-v1'),
         ('spektrum-dx4e-v1','Spektrum_DX4e.zip','spektrum-dx4e-manifest-v1'),
         ('mini2-rc-v1','mini2RC.zip','mini2-rc-manifest-v1')]
UNKNOWN = [('sjrc-pro-v1','SJRC_pro.zip','sjrc-pro-manifest-v1'),
           ('nine-eagles-v1','NineEagles.zip','nine-eagles-manifest-v1'),
           ('q205-v1','Q205.zip','q205-manifest-v1'),
           ('wltoys-v1','wltoys.zip','wltoys-manifest-v1')]
FROZEN_SPLIT_SHA256='bb6363366ea7ba5a8908eea4e048c817cbb28126f0d402547f458d036c8428ce'
MAX_SEALED_MEMBER_BYTES=64*1024**2

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))

def write_json(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def verify_manifest_bundle(folder: Path):
    index=json.loads((folder/'bundle-index.json').read_text(encoding='utf-8'))
    for spec in index['artifacts'].values():
        path=(folder/spec['path']).resolve()
        if not path.is_relative_to(folder.resolve()) or path.stat().st_size!=spec['size_bytes'] or sha256_file(path)!=spec['sha256']:
            raise ValueError('Frozen manifest bundle mismatch')

def verify_known_inputs(manifests: Path, raw: Path, old: Path):
    assignments=manifests/'known-split-v1'/'known-split-assignments.csv'
    if sha256_file(assignments)!=FROZEN_SPLIT_SHA256:
        raise ValueError('Frozen split checksum mismatch; no raw members may be read')
    verify_manifest_bundle(manifests/'known-split-v1')
    old_meta=json.loads((old/'dataset-metadata.json').read_text(encoding='utf-8'))
    if old_meta['source_split_assignments']['sha256']!=FROZEN_SPLIT_SHA256:
        raise ValueError('V1 data split mismatch')
    plan_hashes={row['sha256'] for row in old_meta['source_window_plans']}
    evidence=[]
    for archive_id,filename,folder in KNOWN:
        verify_manifest_bundle(manifests/folder)
        if sha256_file(manifests/folder/'window-plan.csv') not in plan_hashes:
            raise ValueError('Original window plan changed')
        verified=json.loads((manifests/folder/'manifest-verification.json').read_text(encoding='utf-8'))
        expected=verified['archive'];path=raw/filename
        if expected['archive_id']!=archive_id or path.stat().st_size!=expected['size_bytes'] or sha256_file(path)!=expected['sha256']:
            raise ValueError('Frozen raw archive mismatch')
        evidence.append({'name':filename,'sha256':expected['sha256'],'size_bytes':expected['size_bytes']})
    for split in ('train','validation'):
        with KULeuvenMaterializedDataset(old,split=split):
            pass
    return evidence

def prepare_dense(manifests: Path, raw: Path, old: Path, output: Path):
    verified_inputs=verify_known_inputs(manifests,raw,old)
    assignments=manifests/'known-split-v1'/'known-split-assignments.csv'
    split_rows=read_csv(assignments)
    train_ids={row['recording_id'] for row in split_rows if row['split']=='train'}
    if len(train_ids)!=115 or len(split_rows)!=193:
        raise ValueError('Unexpected frozen split inventory')
    if output.exists():
        raise FileExistsError(output)
    plan_root=manifests/'dense-train-v2'
    plan_root.mkdir(exist_ok=True)
    sources=[]
    for archive_id,filename,folder in KNOWN:
        recordings={r['recording_id']:r for r in read_csv(manifests/folder/'recording-manifest.csv')}
        grouped=defaultdict(list)
        for row in read_csv(manifests/folder/'window-plan.csv'):
            grouped[row['recording_id']].append(row)
        dense=[]
        for rid, rows in grouped.items():
            if rid not in train_ids:
                continue
            original={int(row['start_sample']):row for row in rows}
            for ordinal,start in enumerate(dense_offsets(rid,int(recordings[rid]['sample_count']),list(original))):
                row=dict(original.get(start,rows[0]))
                if start not in original:
                    row['window_id']=f'{rid}:dense-v2:{start}'
                row.update(start_sample=start,end_sample_exclusive=start+4096,split='train')
                dense.append(row)
        path=plan_root/(archive_id+'-window-plan.csv')
        with path.open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(dense[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(dense)
        sources.append((archive_id,raw/filename,path))
    materialize_known_windows(split_assignments=assignments,sources=sources,output_dir=output,splits=('train',))
    meta=json.loads((output/'dataset-metadata.json').read_text(encoding='utf-8'))
    old_meta=json.loads((old/'dataset-metadata.json').read_text(encoding='utf-8'))
    shutil.copytree(old/'validation',output/'validation')
    meta.update(artifact_type='ku_leuven_a800_v2',schema_version='2.0',dense_protocol='dense-v2|20260929',windows_per_train_recording=256)
    meta['artifacts']['validation']=old_meta['artifacts']['validation']
    meta['split_counts']['validation']=old_meta['split_counts']['validation']
    meta['source_base_metadata_sha256']=sha256_file(old/'dataset-metadata.json')
    write_json(output/'dataset-metadata.json',meta)
    with A800Dataset(output) as new, KULeuvenMaterializedDataset(old,split='train') as base:
        indexed={sample.window_id:i for i,sample in enumerate(new.samples)}
        for i,sample in enumerate(base.samples):
            if not np.array_equal(base[i][0].numpy(),new[indexed[sample.window_id]][0].numpy()):
                raise ValueError('Original clean training window was changed')
        report={'ok':True,'training_windows':len(new),'training_recordings':len(new.recording_ids),
                'original_windows_bitwise_preserved':len(base),'validation_files_bitwise_preserved':True,
                'test_unknown_guard_used':False,'source_archives':verified_inputs,'metadata_sha256':sha256_file(output/'dataset-metadata.json')}
    write_json(plan_root/'dense-verification.json',report)
    return report

def build_unknown_manifest(archive: Path, archive_id: str, output: Path):
    with zipfile.ZipFile(archive) as handle:
        names=[x.filename for x in handle.infolist() if not x.is_dir()]
    first=names[0]
    match=re.fullmatch(r'(.*?)(\d+)(\.mat)',first)
    if not match:
        raise ValueError('Unexpected official member naming')
    pattern=re.escape(match[1])+r'(?P<index>\d+)'+re.escape(match[3])
    spec=KULeuvenArchiveSpec(archive_id=archive_id,member_pattern=pattern,device_group=archive.stem,
                            device_label=archive.stem+' unknown RF source',expected_recording_count=len(names),
                            expected_archive_bytes=archive.stat().st_size,expected_archive_sha256=sha256_file(archive))
    bundle=build_manifest_bundle(archive,spec,KULeuvenWindowSpec())
    write_manifest_bundle(bundle,output)
    return bundle['verification']

def sealed_manifest(root: Path):
    manifests=root/'manifests'
    split_file=manifests/'known-split-v1'/'known-split-assignments.csv'
    if sha256_file(split_file)!=FROZEN_SPLIT_SHA256:
        raise ValueError('Frozen sealed split changed')
    verify_manifest_bundle(manifests/'known-split-v1')
    assignments={r['recording_id']:r for r in read_csv(manifests/'known-split-v1'/'known-split-assignments.csv')}
    payload={'schema_version':'2.0','known_test':[],'unknown':[],'sources':[],'manifest_files':[]}
    for name in ('known-split-assignments.csv','known-split-verification.json','bundle-index.json'):
        path=manifests/'known-split-v1'/name
        payload['manifest_files'].append({'path':path.relative_to(root).as_posix(),'sha256':sha256_file(path),'size_bytes':path.stat().st_size})
    for archive_id,filename,folder in KNOWN+UNKNOWN:
        verify_manifest_bundle(manifests/folder)
        archive=root/'data'/'raw'/filename
        payload['sources'].append({'path':archive.relative_to(root).as_posix(),'size_bytes':archive.stat().st_size,'sha256':sha256_file(archive)})
        records={r['recording_id']:r for r in read_csv(manifests/folder/'recording-manifest.csv')}
        windows=defaultdict(list)
        for r in read_csv(manifests/folder/'window-plan.csv'):
            windows[r['recording_id']].append(r)
        for rid,record in records.items():
            known=(archive_id,filename,folder) in KNOWN
            if known and assignments[rid]['split']!='test':
                continue
            frozen_rows=sorted(windows[rid],key=lambda row:int(row['start_sample']))
            starts=[int(row['start_sample']) for row in frozen_rows]
            if len(starts)!=32:
                raise ValueError('Expected 32 frozen evaluation windows')
            entry={'recording_id':rid,'archive_id':archive_id,'member_path':record['member_path'],
                   'member_sha256':record['member_sha256'],'member_size_bytes':int(record['member_size_bytes']),
                   'label_index':int(assignments[rid]['label_index']) if known else -1,
                   'split':'test' if known else 'unknown','archive_path':archive.relative_to(root).as_posix(),
                   'window_starts':starts,'window_ids':[row['window_id'] for row in frozen_rows],
                   'sample_count':int(record['sample_count']),'sample_rate_hz':float(record['sample_rate_hz']),'preprocessing_id':PREPROCESSING_ID}
            payload['known_test' if known else 'unknown'].append(entry)
        for filename2 in ('recording-manifest.csv','window-plan.csv','bundle-index.json'):
            path=manifests/folder/filename2
            payload['manifest_files'].append({'path':path.relative_to(root).as_posix(),'sha256':sha256_file(path),'size_bytes':path.stat().st_size})
    if len(payload['known_test'])!=24:
        raise ValueError('Expected 24 frozen test members')
    write_json(manifests/'sealed-evaluation.json',payload)
    return {'known_test_members':len(payload['known_test']),'unknown_members':len(payload['unknown'])}

def load_sealed_member(root: Path, member: dict) -> np.ndarray:
    """Evaluator must verify its freeze before calling this bounded raw reader."""
    root=Path(root).resolve()
    path=(root/member['archive_path']).resolve()
    if not path.is_relative_to(root) or member['split'] not in {'test','unknown'} or len(member['window_starts'])!=32 or member['preprocessing_id']!=PREPROCESSING_ID or member['sample_rate_hz']!=100000000:
        raise ValueError('Sealed member contract mismatch')
    starts=member['window_starts']
    if any(isinstance(x,bool) or not isinstance(x,int) for x in starts) or starts[0]<0 or any(b<a+4096 for a,b in zip(starts,starts[1:])) or starts[-1]+4096>member.get('sample_count',10000000):
        raise ValueError('Invalid sealed window offsets')
    with zipfile.ZipFile(path) as archive:
        # BytesIO avoids repeated ZIP seek/decompression in HDF5 and binds the exact member.
        info=archive.getinfo(member['member_path'])
        if not 0 < info.file_size <= MAX_SEALED_MEMBER_BYTES:
            raise ValueError('Sealed member exceeds bounded reader size')
        if 'member_size_bytes' in member and info.file_size!=member['member_size_bytes']:
            raise ValueError('Sealed member declared size mismatch')
        with archive.open(info) as source:
            data=source.read(MAX_SEALED_MEMBER_BYTES+1)
        if len(data)!=info.file_size or len(data)>MAX_SEALED_MEMBER_BYTES:
            raise ValueError('Invalid sealed member length')
    if hashlib.sha256(data).hexdigest()!=member['member_sha256']:
        raise ValueError('Sealed member checksum mismatch')
    metadata=inspect_mat_v73_iq_fileobj(io.BytesIO(data),source_name=member['member_path'],sample_rate_hz=100000000)
    if metadata.sample_count!=member.get('sample_count',10000000):
        raise ValueError('Sealed member observed sample count mismatch')
    windows=read_mat_v73_iq_windows_fileobj(io.BytesIO(data),source_name=member['member_path'],start_samples=member['window_starts'],window_size=4096,sample_rate_hz=100000000)
    return np.stack([preprocess_iq_window(window.samples) for window in windows])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    dense=sub.add_parser('dense')
    for arg in ('manifests','raw','old','output'):
        dense.add_argument('--'+arg,type=Path,required=True)
    unknown=sub.add_parser('unknown')
    unknown.add_argument('--archive',type=Path,required=True)
    unknown.add_argument('--archive-id',required=True)
    unknown.add_argument('--output',type=Path,required=True)
    sealed=sub.add_parser('sealed');sealed.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='dense':
        result=prepare_dense(args.manifests,args.raw,args.old,args.output)
    elif args.command=='unknown':
        result=build_unknown_manifest(args.archive,args.archive_id,args.output)
    else:
        result=sealed_manifest(args.root)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
