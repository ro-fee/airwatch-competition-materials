import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
import zipfile

import h5py
import numpy as np

from airwatch.data.data_provenance import sha256_file
from airwatch.data.ku_leuven_manifest import (
    ARCHIVE_SPECS,
    FRYSKY_SPEC,
    KULeuvenArchiveSpec,
    KULeuvenManifestError,
    KULeuvenWindowSpec,
    MINI2_RC_SPEC,
    SPEKTRUM_DX4E_SPEC,
    build_bundle_index,
    build_manifest_bundle,
    inspect_zip_recordings,
    stratified_iq_window_offsets,
    write_bundle_index,
    write_manifest_bundle,
)


class KULeuvenManifestTests(unittest.TestCase):
    def test_fixed_archive_profiles_bind_known_and_unknown_sources(self):
        self.assertIs(ARCHIVE_SPECS["frysky"], FRYSKY_SPEC)
        self.assertEqual(FRYSKY_SPEC.expected_recording_count, 51)
        self.assertEqual(
            FRYSKY_SPEC.member_pattern,
            r"Frysky/Frysky_(?P<index>\d+)\.mat",
        )
        self.assertEqual(len(FRYSKY_SPEC.expected_archive_sha256), 64)
        self.assertIs(ARCHIVE_SPECS["spektrum-dx4e"], SPEKTRUM_DX4E_SPEC)
        self.assertEqual(SPEKTRUM_DX4E_SPEC.expected_recording_count, 71)
        self.assertEqual(
            SPEKTRUM_DX4E_SPEC.member_pattern,
            r"Spektrum_DX4e/DX4e_(?P<index>\d+)\.mat",
        )
        self.assertEqual(len(SPEKTRUM_DX4E_SPEC.expected_archive_sha256), 64)
        self.assertIs(ARCHIVE_SPECS["mini2-rc"], MINI2_RC_SPEC)
        self.assertEqual(MINI2_RC_SPEC.expected_recording_count, 71)
        self.assertEqual(
            MINI2_RC_SPEC.member_pattern,
            r"mini2RC/mini2_(?P<index>\d+)\.mat",
        )
        self.assertEqual(len(MINI2_RC_SPEC.expected_archive_sha256), 64)

    @staticmethod
    def _mat_bytes(sample_count: int = 160) -> bytes:
        output = io.BytesIO()
        dtype = np.dtype([("real", "<f8"), ("imag", "<f8")])
        values = np.zeros((1, sample_count), dtype=dtype)
        values["real"] = np.arange(sample_count)
        values["imag"] = -np.arange(sample_count)
        with h5py.File(output, "w") as handle:
            dataset = handle.create_dataset("uhd_samps", data=values)
            dataset.attrs["MATLAB_class"] = np.bytes_("double")
        return output.getvalue()

    @staticmethod
    def _spec(count: int) -> KULeuvenArchiveSpec:
        return KULeuvenArchiveSpec(
            archive_id="test-v1",
            member_pattern=r"group/sample__(?P<index>\d+)\.mat",
            device_group="group",
            device_label="test device",
            expected_recording_count=count,
            sample_rate_hz=80,
            center_frequency_hz=2_440_000_000,
        )

    def _archive(self, root: Path, count: int = 3) -> Path:
        archive = root / "recordings.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for index in reversed(range(count)):
                handle.writestr(f"group/sample__{index}.mat", self._mat_bytes())
        return archive

    def test_inspects_all_recordings_inside_zip_and_hashes_members(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self._archive(Path(directory))
            recordings = inspect_zip_recordings(archive, self._spec(3))
            self.assertEqual([row.member_index for row in recordings], [0, 1, 2])
            self.assertEqual({row.sample_count for row in recordings}, {160})
            self.assertEqual({row.source_shape for row in recordings}, {"1x160"})
            self.assertTrue(all(len(row.member_sha256) == 64 for row in recordings))
            self.assertTrue(all(row.split == "unassigned" for row in recordings))

    def test_offsets_are_stable_contained_and_nonoverlapping(self):
        spec = KULeuvenWindowSpec(window_length=16, windows_per_recording=5, seed=7)
        first = stratified_iq_window_offsets("recording-a", 160, spec)
        self.assertEqual(first, stratified_iq_window_offsets("recording-a", 160, spec))
        self.assertNotEqual(first, stratified_iq_window_offsets("recording-b", 160, spec))
        self.assertEqual(len(first), 5)
        self.assertTrue(all(right >= left + 16 for left, right in zip(first, first[1:])))
        self.assertGreaterEqual(first[0], 0)
        self.assertLessEqual(first[-1] + 16, 160)

    def test_bundle_writes_parseable_evidence_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            bundle = build_manifest_bundle(
                archive,
                self._spec(3),
                KULeuvenWindowSpec(window_length=16, windows_per_recording=5),
            )
            self.assertTrue(bundle["verification"]["ok"])
            self.assertFalse(bundle["verification"]["training_eligible"])
            self.assertEqual(len(bundle["windows"]), 15)
            outputs = write_manifest_bundle(bundle, root / "evidence")
            with outputs["recordings"].open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            with outputs["windows"].open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 15)
            report = json.loads(outputs["verification"].read_text(encoding="utf-8"))
            self.assertEqual(report["archive"]["sha256"], sha256_file(archive))
            index = json.loads(outputs["index"].read_text(encoding="utf-8"))
            self.assertEqual(
                index["artifacts"]["recording_manifest"]["sha256"],
                sha256_file(outputs["recordings"]),
            )
            with self.assertRaises(FileExistsError):
                write_manifest_bundle(bundle, root / "evidence")

    def test_standalone_bundle_index_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name in ("recording.csv", "windows.csv", "verification.json"):
                path = root / name
                path.write_text(name, encoding="utf-8")
                paths.append(path)
            index = build_bundle_index(*paths)
            self.assertEqual(len(index["interpretation_limits"]), 2)
            output = root / "bundle-index.json"
            self.assertEqual(write_bundle_index(index, output), output)
            with self.assertRaises(FileExistsError):
                write_bundle_index(index, output)

    def test_rejects_gaps_unexpected_members_and_too_short_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gap = root / "gap.zip"
            with zipfile.ZipFile(gap, "w") as handle:
                handle.writestr("group/sample__0.mat", self._mat_bytes())
                handle.writestr("group/sample__2.mat", self._mat_bytes())
            with self.assertRaises(KULeuvenManifestError):
                inspect_zip_recordings(gap, self._spec(2))

            extra = root / "extra.zip"
            with zipfile.ZipFile(extra, "w") as handle:
                handle.writestr("group/sample__0.mat", self._mat_bytes())
                handle.writestr("README.txt", b"unexpected")
            with self.assertRaises(KULeuvenManifestError):
                inspect_zip_recordings(extra, self._spec(1))

            with self.assertRaises(KULeuvenManifestError):
                stratified_iq_window_offsets(
                    "short", 31, KULeuvenWindowSpec(window_length=16, windows_per_recording=2)
                )


if __name__ == "__main__":
    unittest.main()
