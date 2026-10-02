"""Download the three missing official KU Leuven archives with integrity checks."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = {
    'NineEagles.zip': (154427, 1491713902, 'd08bc6bf7cfa9d267d6f637f58c384a5'),
    'Q205.zip': (154428, 1531075910, '3506e1e507b6575f6d741e93effdb3f5'),
    'wltoys.zip': (154421, 1513957160, 'fb2847bfe3696b835c900fc253ef3f89'),
}

def fetch_one(root: Path, name: str) -> dict:
    file_id, expected_size, expected_md5 = SOURCES[name]
    url = f'https://rdr.kuleuven.be/api/access/datafile/{file_id}'
    target = root / name
    part = root / (name + '.part')
    if not target.exists():
        offset = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            append = offset > 0 and response.status == 206
            with part.open('ab' if append else 'wb') as output:
                total = offset if append else 0
                announced = total // (128 * 1024**2)
                while chunk := response.read(4 * 1024**2):
                    output.write(chunk)
                    total += len(chunk)
                    if total // (128 * 1024**2) > announced:
                        announced = total // (128 * 1024**2)
                        print(f'{name}: {total}/{expected_size} bytes', flush=True)
    actual = target if target.exists() else part
    md5, sha = hashlib.md5(), hashlib.sha256()
    with actual.open('rb') as handle:
        while chunk := handle.read(8 * 1024**2):
            md5.update(chunk)
            sha.update(chunk)
    if actual.stat().st_size != expected_size or md5.hexdigest() != expected_md5:
        raise ValueError(f'Official size/MD5 mismatch: {name}')
    if actual == part:
        part.rename(target)
    result = {'name': name, 'source_url': url, 'size_bytes': expected_size,
              'official_md5': expected_md5, 'sha256': sha.hexdigest(), 'verified': True}
    (root / (name + '.provenance.json')).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'{name}: verified', flush=True)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda name: fetch_one(args.output, name), SOURCES))

if __name__ == '__main__':
    main()
