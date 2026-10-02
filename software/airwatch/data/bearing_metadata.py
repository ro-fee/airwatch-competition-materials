"""Read the audited per-path channel and sampling metadata shared by UI workflows."""
import csv
from pathlib import Path


def find_bearing_metadata(manifest_path, project_root, input_path):
    target = Path(input_path).resolve()
    try:
        with Path(manifest_path).open('r', encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                local = row.get('local_path', '')
                if local and (Path(project_root) / local).resolve() == target:
                    return row
    except (OSError, UnicodeDecodeError):
        return None
    return None


def metadata_sensor_key(row):
    keys = [key.strip() for key in (row or {}).get('sensor_key', '').split('|') if key.strip()]
    return keys[0] if len(keys) == 1 else None
