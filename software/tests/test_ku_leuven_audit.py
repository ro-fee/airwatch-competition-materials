import unittest
import hashlib
import tempfile
import zipfile
from pathlib import Path

from airwatch.data.ku_leuven_audit import (
    EXPECTED_LICENSE,
    KULeuvenAuditError,
    audit_downloaded_archive,
    normalize_official_metadata,
)


def _payload():
    documentation = [
        (1, "README.txt", 10),
        (2, "get_spectrogram2.m", 11),
        (3, "signal_visualize.m", 12),
    ]
    archives = [(index, f"device_{index}.zip", 100 + index) for index in range(4, 20)]
    files = []
    for file_id, name, size in documentation + archives:
        files.append({
            "label": name,
            "dataFile": {
                "id": file_id,
                "filesize": size,
                "md5": f"{file_id:032x}",
                "restricted": False,
                "contentType": "application/zip" if name.endswith(".zip") else "text/plain",
            },
        })
    return {
        "status": "OK",
        "data": {"latestVersion": {
            "versionNumber": 1,
            "versionMinorNumber": 0,
            "releaseTime": "2024-01-16T09:08:34Z",
            "license": {"name": EXPECTED_LICENSE},
            "metadataBlocks": {"citation": {"fields": [
                {"typeName": "title", "value": "Drone RF Dataset"},
            ]}},
            "files": files,
        }},
    }


class KULeuvenAuditTests(unittest.TestCase):
    def test_normalizes_frozen_public_inventory_and_blocks_direct_tcn_use(self):
        result = normalize_official_metadata(_payload())
        self.assertEqual(result["dataset_version"], "1.0")
        self.assertEqual(result["file_count"], 19)
        self.assertEqual(result["total_size_bytes"], sum(range(104, 120)) + 33)
        self.assertTrue(result["all_files_public"])
        self.assertFalse(result["compatibility_gate"]["existing_dronerf_tcn_direct_input_compatible"])
        self.assertFalse(result["training_eligible"])

    def test_fails_closed_when_version_or_license_changes(self):
        changed = _payload()
        changed["data"]["latestVersion"]["versionNumber"] = 2
        with self.assertRaises(KULeuvenAuditError):
            normalize_official_metadata(changed)
        changed = _payload()
        changed["data"]["latestVersion"]["license"]["name"] = "NOASSERTION"
        with self.assertRaises(KULeuvenAuditError):
            normalize_official_metadata(changed)

    def test_fails_closed_for_restricted_or_incomplete_files(self):
        restricted = _payload()
        restricted["data"]["latestVersion"]["files"][0]["dataFile"]["restricted"] = True
        with self.assertRaises(KULeuvenAuditError):
            normalize_official_metadata(restricted)
        incomplete = _payload()
        incomplete["data"]["latestVersion"]["files"].pop()
        with self.assertRaises(KULeuvenAuditError):
            normalize_official_metadata(incomplete)

    def test_archive_audit_verifies_checksum_and_lists_mat_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "sample.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("device/recording_0.mat", b"mat bytes")
            content = archive_path.read_bytes()
            inventory = {
                "dataset": "test", "dataset_version": "1.0", "dataset_doi": "doi:test",
                "files": [{
                    "name": "sample.zip", "size_bytes": len(content),
                    "md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
                }],
            }
            report = audit_downloaded_archive(
                archive_path, inventory, expected_name="sample.zip"
            )
            self.assertTrue(report["checksum_verified"])
            self.assertEqual(report["mat_member_count"], 1)
            self.assertFalse(report["content_extracted"])
            self.assertFalse(report["training_eligible"])


if __name__ == "__main__":
    unittest.main()
