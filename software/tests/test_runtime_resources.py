"""Tests for the contract-driven packaged runtime resource closure."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from airwatch.inference.bearing_contract import load_frozen_bearing_runtime_contract
from airwatch.inference.general_models import TASKS
from airwatch.inference.ku_leuven import load_ku_leuven_runtime_contract
from airwatch.runtime_resources import (
    BEARING_RUNTIME_CONTRACT,
    GENERAL_TASK_CHECKPOINTS,
    GENERAL_TASK_SLOT_IDS,
    HISTORICAL_GENERATOR_SLOT_ID,
    HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID,
    KU_LEUVEN_SOFTWARE_CONTRACT,
    MODEL_SLOTS,
    RuntimeResourceError,
    build_runtime_resource_manifest,
    general_task_slot_id,
    model_slot,
    model_slot_resource_path,
    pyinstaller_datas,
    validate_runtime_resource_manifest,
    write_runtime_resource_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


class RuntimeResourceManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_runtime_resource_manifest(PROJECT_ROOT)

    def test_general_task_registry_uses_packaged_checkpoint_registry(self) -> None:
        self.assertEqual(
            {task: spec.checkpoint for task, spec in TASKS.items()},
            GENERAL_TASK_CHECKPOINTS,
        )
        self.assertEqual(set(GENERAL_TASK_SLOT_IDS), set(TASKS))
        self.assertEqual(
            general_task_slot_id("信号个体识别"),
            "historical.individual",
        )

    def test_manifest_is_a_minimal_verified_resource_closure(self) -> None:
        paths = {entry["path"] for entry in self.manifest["resources"]}
        self.assertIn("assets/airwatch.ico", paths)
        self.assertIn(BEARING_RUNTIME_CONTRACT, paths)
        self.assertIn(KU_LEUVEN_SOFTWARE_CONTRACT, paths)
        self.assertIn("models/999G_plus.ckpt", paths)
        self.assertIn("models/newdata_TCN1122.pkl", paths)
        self.assertIn("datasets/bearing/cwru/label-map.json", paths)
        self.assertIn("datasets/bearing/cwru/split-manifest.csv", paths)
        self.assertFalse(any(path.startswith("datasets/bearing/cwru/raw/") for path in paths))
        self.assertFalse(any("failed_attempt" in path for path in paths))
        self.assertFalse(any("superseded" in path for path in paths))

        old_bulk_bytes = sum(
            path.stat().st_size
            for root_name in ("models", "artifacts")
            for path in (PROJECT_ROOT / root_name).rglob("*")
            if path.is_file()
        )
        self.assertLess(self.manifest["summary"]["total_bytes"], old_bulk_bytes)

        direct_available = {
            slot.resource
            for slot in MODEL_SLOTS
            if slot.status == "available"
            and slot.slot_id not in {"bearing.diagnosis", "uav.known_source"}
        }
        self.assertTrue(direct_available <= paths)

    def test_historical_generation_and_comparison_use_distinct_model_slots(self) -> None:
        generator = model_slot(HISTORICAL_GENERATOR_SLOT_ID)
        reference = model_slot(HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID)
        self.assertEqual(generator.resource, "models/999G_plus.ckpt")
        self.assertEqual(reference.resource, "models/newdata_TCN1122.pkl")
        self.assertEqual(reference.release_tier, "historical_demo")
        self.assertNotEqual(reference.slot_id, "edge.student")
        self.assertTrue(model_slot_resource_path(generator.slot_id).is_file())
        self.assertTrue(model_slot_resource_path(reference.slot_id).is_file())

    def test_reserved_model_slots_ship_no_placeholder_weights(self) -> None:
        reserved = [slot for slot in MODEL_SLOTS if slot.status == "reserved"]
        self.assertEqual(
            {slot.slot_id for slot in reserved},
            {"uav.multitask", "uav.open_set", "edge.student"},
        )
        self.assertTrue(all(slot.resource is None for slot in reserved))
        self.assertTrue(all(slot.release_tier == "not_available" for slot in reserved))
        self.assertEqual(model_slot("uav.open_set").status, "reserved")
        with self.assertRaises(KeyError):
            model_slot("uav.does_not_exist")

    def test_staged_closure_validates_and_loads_active_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            staged_root = Path(directory)
            for entry in self.manifest["resources"]:
                relative = Path(entry["path"])
                destination = staged_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(PROJECT_ROOT / relative, destination)
            manifest_path = staged_root / "runtime-resources.json"
            manifest_path.write_text(
                json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8"
            )

            result = validate_runtime_resource_manifest(staged_root, manifest_path)
            self.assertEqual(
                result,
                {
                    "resource_files": self.manifest["summary"]["resource_files"],
                    "total_bytes": self.manifest["summary"]["total_bytes"],
                },
            )

            bearing = load_frozen_bearing_runtime_contract(
                staged_root / BEARING_RUNTIME_CONTRACT,
                project_root=staged_root,
            )
            uav = load_ku_leuven_runtime_contract(
                staged_root / KU_LEUVEN_SOFTWARE_CONTRACT,
                project_root=staged_root,
            )
            self.assertEqual(bearing.project_root, staged_root.resolve())
            self.assertEqual(uav.project_root, staged_root.resolve())
            self.assertEqual(uav.open_set_status, "not_evaluated")

    def test_validation_rejects_a_missing_declared_resource(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeResourceError, "missing"):
                validate_runtime_resource_manifest(directory, self.manifest)

    def test_pyinstaller_datas_are_file_level_and_include_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "runtime-resources.json"
            datas = pyinstaller_datas(PROJECT_ROOT, manifest_path)
        self.assertEqual(len(datas), self.manifest["summary"]["resource_files"] + 1)
        self.assertEqual(datas[-1][1], ".")
        self.assertTrue(all(Path(source).is_file() for source, _ in datas[:-1]))

    def test_unchanged_runtime_manifest_preserves_file_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "runtime-resources.json"
            write_runtime_resource_manifest(PROJECT_ROOT, manifest_path)
            stable_timestamp_ns = 1_700_000_000_000_000_000
            os.utime(
                manifest_path,
                ns=(stable_timestamp_ns, stable_timestamp_ns),
            )

            write_runtime_resource_manifest(PROJECT_ROOT, manifest_path)

            self.assertEqual(manifest_path.stat().st_mtime_ns, stable_timestamp_ns)

    def test_desktop_specs_do_not_collect_offline_airwatch_modules(self) -> None:
        for name in ("airwatch_onedir.spec", "airwatch_debug.spec"):
            source = (PROJECT_ROOT / "packaging" / name).read_text(encoding="utf-8")
            self.assertNotIn("collect_submodules", source, name)
            self.assertIn("pyinstaller_datas", source, name)
            for optional_package in (
                "pandas",
                "torchvision",
                "h5py",
                "pytest",
                "fsspec",
                "sympy",
                "cv2",
            ):
                self.assertIn(f'"{optional_package}"', source, name)

    def test_onedir_analysis_excludes_are_stable_for_pyinstaller_cache(self) -> None:
        captured = {}

        class AnalysisDouble:
            def __init__(self, *_args, excludes, binaries, datas, **_kwargs):
                captured["excludes"] = tuple(excludes)
                self.pure = []
                self.scripts = []
                self.binaries = list(binaries)
                self.datas = list(datas)

        namespace = {
            "SPECPATH": str(PROJECT_ROOT),
            "Analysis": AnalysisDouble,
            "PYZ": lambda *_args, **_kwargs: object(),
            "EXE": lambda *_args, **_kwargs: object(),
            "COLLECT": lambda *_args, **_kwargs: object(),
        }
        spec_path = PROJECT_ROOT / "packaging" / "airwatch_onedir.spec"
        with mock.patch(
            "airwatch.runtime_resources.pyinstaller_datas", return_value=[]
        ):
            exec(
                compile(spec_path.read_text(encoding="utf-8"), spec_path, "exec"),
                namespace,
            )

        # PyInstaller appends __main__ in-place when it is absent, then stores
        # the mutated list in Analysis-00.toc. Supplying it up front keeps the
        # next run's inputs equal and prevents a false "excludes changed".
        self.assertIn("__main__", captured["excludes"])

    def test_maintained_desktop_entry_does_not_import_opencv(self) -> None:
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("import cv2", source)
        self.assertNotIn("from cv2", source)

    def test_release_validator_rejects_retired_heavy_dependencies(self) -> None:
        source = (
            PROJECT_ROOT / "packaging" / "validate_portable_release.ps1"
        ).read_text(encoding="utf-8")
        for package in ("cv2", "pandas", "torchvision", "h5py", "tensorboard"):
            self.assertIn(f"'{package}'", source)

    def test_recognition_input_import_does_not_load_offline_data_modules(self) -> None:
        script = (
            "import json, sys; import airwatch.data.recognition_input; "
            "print(json.dumps({'airwatch_data': sorted(m for m in sys.modules "
            "if m.startswith('airwatch.data')), 'heavy': {name: name in sys.modules "
            "for name in ('torch','scipy','pandas','h5py','torchvision')}}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(
            payload["airwatch_data"],
            ["airwatch.data", "airwatch.data.recognition_input"],
        )
        self.assertFalse(any(payload["heavy"].values()), payload)

    @unittest.skipUnless(POWERSHELL, "PowerShell is required for release validation")
    def test_portable_release_metadata_and_validator_use_runtime_manifest(self) -> None:
        build_root = PROJECT_ROOT / "build"
        build_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="release-validation-", dir=build_root
        ) as directory:
            package = Path(directory) / "AirWatch"
            internal = package / "_internal"
            internal.mkdir(parents=True)
            manifest_path = internal / "runtime-resources.json"
            manifest = write_runtime_resource_manifest(PROJECT_ROOT, manifest_path)
            for entry in manifest["resources"]:
                relative = Path(entry["path"])
                source = PROJECT_ROOT / relative
                destination = internal / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            (package / "AirWatch.exe").write_bytes(b"test executable")

            preparation = subprocess.run(
                [
                    str(POWERSHELL),
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(PROJECT_ROOT / "packaging/prepare_portable_release.ps1"),
                    "-Package",
                    str(package),
                    "-Version",
                    "test",
                ],
                check=False,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                preparation.returncode,
                0,
                preparation.stdout + "\n" + preparation.stderr,
            )
            validation = subprocess.run(
                [
                    str(POWERSHELL),
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(PROJECT_ROOT / "packaging/validate_portable_release.ps1"),
                    "-Package",
                    str(package),
                ],
                check=False,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                validation.returncode,
                0,
                validation.stdout + "\n" + validation.stderr,
            )

            release = json.loads(
                (package / "release.json").read_text(encoding="utf-8-sig")
            )
            self.assertEqual(release["version"], "test")
            self.assertEqual(
                release["runtime_resource_bytes"],
                manifest["summary"]["total_bytes"],
            )
            self.assertEqual(
                release["capabilities"]["uav.known_source"]["status"],
                "available",
            )
            self.assertEqual(
                release["capabilities"]["uav.open_set"]["status"],
                "reserved",
            )
            self.assertIn("PASS", validation.stdout)


if __name__ == "__main__":
    unittest.main()
