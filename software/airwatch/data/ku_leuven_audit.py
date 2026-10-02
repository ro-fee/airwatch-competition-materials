"""Audit KU Leuven Drone RF Dataverse metadata before any large download."""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


DATASET_DOI = "doi:10.48804/HZRVNZ"
DATASET_VERSION = "1.0"
EXPECTED_LICENSE = "CC-BY-NC-4.0"
DATAVERSE_API_URL = (
    "https://rdr.kuleuven.be/api/datasets/:persistentId/"
    "?persistentId=doi:10.48804/HZRVNZ"
)
DATAFILE_ACCESS_URL = "https://rdr.kuleuven.be/api/access/datafile/{file_id}"
DOCUMENTATION_FILES = {"README.txt", "get_spectrogram2.m", "signal_visualize.m"}


class KULeuvenAuditError(ValueError):
    """Raised when official metadata is incomplete or contradicts the protocol."""


@dataclass(frozen=True)
class DataverseFile:
    file_id: int
    name: str
    size_bytes: int
    md5: str
    restricted: bool
    content_type: str
    access_url: str


def _citation_field(version: dict[str, Any], type_name: str) -> Any:
    fields = version.get("metadataBlocks", {}).get("citation", {}).get("fields", [])
    for field in fields:
        if field.get("typeName") == type_name:
            return field.get("value")
    return None


def normalize_official_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one live Dataverse API response."""
    if payload.get("status") != "OK":
        raise KULeuvenAuditError("Dataverse API did not return status=OK")
    version = payload.get("data", {}).get("latestVersion")
    if not isinstance(version, dict):
        raise KULeuvenAuditError("Dataverse response lacks latestVersion")
    version_text = f"{version.get('versionNumber')}.{version.get('versionMinorNumber')}"
    if version_text != DATASET_VERSION:
        raise KULeuvenAuditError(
            f"dataset version changed from frozen {DATASET_VERSION} to {version_text}"
        )
    license_name = version.get("license", {}).get("name")
    if license_name != EXPECTED_LICENSE:
        raise KULeuvenAuditError(
            f"dataset license changed from {EXPECTED_LICENSE} to {license_name!r}"
        )
    title = _citation_field(version, "title")
    if title != "Drone RF Dataset":
        raise KULeuvenAuditError(f"unexpected dataset title: {title!r}")

    records: list[DataverseFile] = []
    for item in version.get("files", []):
        data_file = item.get("dataFile", {})
        try:
            file_id = int(data_file["id"])
            name = str(item["label"])
            size = int(data_file["filesize"])
            md5 = str(data_file["md5"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise KULeuvenAuditError("file inventory contains an incomplete record") from exc
        if file_id <= 0 or not name or size <= 0 or len(md5) != 32:
            raise KULeuvenAuditError(f"invalid file inventory record: {name!r}")
        records.append(DataverseFile(
            file_id=file_id,
            name=name,
            size_bytes=size,
            md5=md5,
            restricted=bool(data_file.get("restricted", False)),
            content_type=str(data_file.get("contentType", "")),
            access_url=DATAFILE_ACCESS_URL.format(file_id=file_id),
        ))
    if len(records) != 19:
        raise KULeuvenAuditError(f"expected 19 files for V1, found {len(records)}")
    if len({record.file_id for record in records}) != len(records) or len(
        {record.name for record in records}
    ) != len(records):
        raise KULeuvenAuditError("file ids and names must be unique")
    if any(record.restricted for record in records):
        raise KULeuvenAuditError("frozen V1 unexpectedly contains restricted files")
    if {record.name for record in records if record.name in DOCUMENTATION_FILES} != DOCUMENTATION_FILES:
        raise KULeuvenAuditError("official documentation files are incomplete")

    ordered = sorted(records, key=lambda record: record.name.casefold())
    return {
        "schema_version": "1.0",
        "dataset": "KU Leuven Drone RF Dataset",
        "dataset_doi": DATASET_DOI,
        "dataset_version": DATASET_VERSION,
        "title": title,
        "publication_date": version.get("releaseTime"),
        "license": license_name,
        "license_verified": True,
        "source_url": "https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/HZRVNZ",
        "api_url": DATAVERSE_API_URL,
        "file_count": len(ordered),
        "total_size_bytes": sum(record.size_bytes for record in ordered),
        "all_files_public": True,
        "files": [asdict(record) for record in ordered],
        "official_description": {
            "sample_rate_samples_per_second": 100_000_000,
            "center_frequency_hz": 2_440_000_000,
            "receiver": "Ettus Research USRP X310",
            "environment": "semi-anechoic chamber",
            "stored_format": "MATLAB v7.3 complex signal vectors inside ZIP archives",
        },
        "compatibility_gate": {
            "existing_dronerf_tcn_direct_input_compatible": False,
            "existing_input_semantics": (
                "two real time-domain amplitude channels from the lower and upper halves "
                "of the 2.4 GHz band"
            ),
            "candidate_input_semantics": "single 2.44 GHz complex I/Q recording",
            "reason": (
                "Both tensors may have two numeric rows, but I/Q components are not interchangeable "
                "with DroneRF low/high-band channels. Direct evaluation would be semantically invalid."
            ),
            "allowed_next_use": (
                "audit one archive, preserve recording boundaries, and design a common single-band "
                "I/Q or spectrogram contract before training or external evaluation"
            ),
        },
        "training_eligible": False,
        "training_blocker": (
            "No large archive has been downloaded or content-audited, and the existing DroneRF "
            "two-band TCN input contract is incompatible with this dataset's complex-IQ semantics."
        ),
    }


def fetch_official_metadata(*, timeout_seconds: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        DATAVERSE_API_URL,
        headers={"User-Agent": "AirWatch-Dataset-Audit/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except (OSError, json.JSONDecodeError) as exc:
        raise KULeuvenAuditError(f"cannot fetch official Dataverse metadata: {exc}") from exc
    if not isinstance(payload, dict):
        raise KULeuvenAuditError("Dataverse API response must be a JSON object")
    return payload


def _download_verified(file: DataverseFile, destination: Path, *, timeout_seconds: float) -> dict:
    request = urllib.request.Request(
        file.access_url,
        headers={"User-Agent": "AirWatch-Dataset-Audit/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content = response.read()
    except OSError as exc:
        raise KULeuvenAuditError(f"cannot download documentation file {file.name}: {exc}") from exc
    actual_md5 = hashlib.md5(content).hexdigest()  # noqa: S324 - verifies Dataverse's published MD5
    if len(content) != file.size_bytes or actual_md5 != file.md5:
        raise KULeuvenAuditError(f"documentation checksum mismatch: {file.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return {
        "name": file.name,
        "file_id": file.file_id,
        "size_bytes": len(content),
        "official_md5": file.md5,
        "actual_md5": actual_md5,
        "verified": True,
        "local_path": destination.as_posix(),
    }


def create_audit_bundle(
    output_root: str | Path,
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Fetch V1 metadata and its three tiny documentation files, never data archives."""
    root = Path(output_root)
    destinations = {
        "api": root / "official-api-response.json",
        "inventory": root / "file-inventory.json",
        "log": root / "download-log.json",
    }
    documentation_root = root / "source-documentation"
    if any(path.exists() for path in destinations.values()) or documentation_root.exists():
        raise FileExistsError("refusing to overwrite an existing KU Leuven audit bundle")
    payload = fetch_official_metadata(timeout_seconds=timeout_seconds)
    normalized = normalize_official_metadata(payload)
    records = [DataverseFile(**item) for item in normalized["files"]]
    documentation = []
    for record in records:
        if record.name in DOCUMENTATION_FILES:
            documentation.append(_download_verified(
                record,
                documentation_root / record.name,
                timeout_seconds=timeout_seconds,
            ))
    root.mkdir(parents=True, exist_ok=True)
    destinations["api"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    destinations["inventory"].write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log = {
        "schema_version": "1.0",
        "dataset": normalized["dataset"],
        "dataset_version": normalized["dataset_version"],
        "dataset_doi": normalized["dataset_doi"],
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_metadata_saved": True,
        "documentation": sorted(documentation, key=lambda item: item["name"].casefold()),
        "large_archives_downloaded": False,
        "training_eligible": False,
        "training_blocker": normalized["training_blocker"],
    }
    destinations["log"].write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"inventory": normalized, "download_log": log}


def audit_downloaded_archive(
    archive_path: str | Path,
    official_inventory: dict[str, Any],
    *,
    expected_name: str,
) -> dict[str, Any]:
    """Verify one downloaded ZIP and list members without extracting it."""
    path = Path(archive_path).resolve()
    if not path.is_file():
        raise KULeuvenAuditError(f"downloaded archive does not exist: {path}")
    matches = [item for item in official_inventory.get("files", []) if item.get("name") == expected_name]
    if len(matches) != 1:
        raise KULeuvenAuditError(f"official inventory does not uniquely identify {expected_name!r}")
    official = matches[0]
    expected_size = int(official["size_bytes"])
    if path.stat().st_size != expected_size:
        raise KULeuvenAuditError(
            f"archive size mismatch: expected {expected_size}, found {path.stat().st_size}"
        )
    md5_digest = hashlib.md5(usedforsecurity=False)
    sha256_digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            md5_digest.update(chunk)
            sha256_digest.update(chunk)
    actual_md5 = md5_digest.hexdigest()
    if actual_md5 != official["md5"]:
        raise KULeuvenAuditError(
            f"archive MD5 mismatch: expected {official['md5']}, found {actual_md5}"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise KULeuvenAuditError(f"invalid ZIP archive: {exc}") from exc
    files = []
    for member in members:
        member_path = PurePosixPath(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise KULeuvenAuditError(f"unsafe ZIP member path: {member.filename!r}")
        if member.is_dir():
            continue
        files.append({
            "name": member.filename,
            "size_bytes": int(member.file_size),
            "compressed_bytes": int(member.compress_size),
            "crc32": f"{member.CRC:08x}",
        })
    if not files or any(not item["name"].lower().endswith(".mat") for item in files):
        raise KULeuvenAuditError("selected archive must contain only MAT recording files")
    return {
        "schema_version": "1.0",
        "dataset": official_inventory.get("dataset"),
        "dataset_version": official_inventory.get("dataset_version"),
        "dataset_doi": official_inventory.get("dataset_doi"),
        "archive_name": expected_name,
        "archive_path": str(path),
        "archive_size_bytes": path.stat().st_size,
        "official_md5": official["md5"],
        "actual_md5": actual_md5,
        "sha256": sha256_digest.hexdigest(),
        "checksum_verified": True,
        "member_count": len(files),
        "mat_member_count": len(files),
        "total_uncompressed_bytes": sum(item["size_bytes"] for item in files),
        "members": files,
        "content_extracted": False,
        "training_eligible": False,
        "training_blocker": (
            "ZIP integrity and member names are verified, but MAT variables and recording "
            "semantics require a separate content audit before any training use."
        ),
    }


def write_archive_audit(
    archive_path: str | Path,
    inventory_path: str | Path,
    output_path: str | Path,
    *,
    expected_name: str,
) -> dict[str, Any]:
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite an existing archive audit")
    try:
        inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KULeuvenAuditError(f"cannot read official file inventory: {exc}") from exc
    report = audit_downloaded_archive(
        archive_path, inventory, expected_name=expected_name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report


__all__ = [
    "DATASET_DOI",
    "DATASET_VERSION",
    "DATAVERSE_API_URL",
    "DataverseFile",
    "KULeuvenAuditError",
    "create_audit_bundle",
    "audit_downloaded_archive",
    "fetch_official_metadata",
    "normalize_official_metadata",
    "write_archive_audit",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--archive-name")
    parser.add_argument("--archive-audit-output", type=Path)
    args = parser.parse_args()
    if args.output_root and not any(
        (args.archive, args.inventory, args.archive_name, args.archive_audit_output)
    ):
        result = create_audit_bundle(args.output_root)
        summary = {
            "dataset": result["inventory"]["dataset"],
            "version": result["inventory"]["dataset_version"],
            "file_count": result["inventory"]["file_count"],
            "total_size_bytes": result["inventory"]["total_size_bytes"],
            "large_archives_downloaded": result["download_log"]["large_archives_downloaded"],
            "training_eligible": result["download_log"]["training_eligible"],
        }
    elif all((args.archive, args.inventory, args.archive_name, args.archive_audit_output)) and not args.output_root:
        report = write_archive_audit(
            args.archive,
            args.inventory,
            args.archive_audit_output,
            expected_name=args.archive_name,
        )
        summary = {
            "archive": report["archive_name"],
            "checksum_verified": report["checksum_verified"],
            "mat_member_count": report["mat_member_count"],
            "content_extracted": report["content_extracted"],
            "training_eligible": report["training_eligible"],
        }
    else:
        parser.error(
            "use --output-root alone, or provide --archive, --inventory, "
            "--archive-name and --archive-audit-output together"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
