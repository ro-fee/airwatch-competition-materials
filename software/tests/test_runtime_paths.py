"""Tests for source and packaged-runtime path resolution."""

from __future__ import annotations

import os
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

from airwatch import runtime_paths
from airwatch.inference.bearing_contract import (
    DEFAULT_FROZEN_BEARING_CONTRACT,
    load_frozen_bearing_runtime_contract,
)


class RuntimePathTests(unittest.TestCase):
    def test_source_resource_root_is_project_root(self) -> None:
        expected = Path(__file__).resolve().parents[1]
        self.assertEqual(runtime_paths.resource_root(frozen=False), expected)

    def test_frozen_resource_root_prefers_meipass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                runtime_paths.resource_root(frozen=True, meipass=directory),
                Path(directory).resolve(),
            )

    def test_frozen_resource_root_falls_back_to_executable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "AirWatch.exe"
            with patch.object(runtime_paths.sys, "executable", str(executable)):
                self.assertEqual(
                    runtime_paths.resource_root(frozen=True, meipass=None),
                    Path(directory).resolve(),
                )

    def test_explicit_user_data_directory_wins_in_source_and_frozen_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {runtime_paths.DATA_ROOT_ENV: directory},
                clear=False,
            ):
                self.assertEqual(
                    runtime_paths.writable_root(frozen=False),
                    Path(directory).resolve(),
                )
                self.assertEqual(
                    runtime_paths.writable_root(frozen=True),
                    Path(directory).resolve(),
                )

    def test_frozen_default_writable_root_uses_local_app_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "LOCALAPPDATA": directory,
                    runtime_paths.DATA_ROOT_ENV: "",
                },
                clear=False,
            ):
                self.assertEqual(
                    runtime_paths.writable_root(frozen=True),
                    (Path(directory) / runtime_paths.APP_DATA_NAME).resolve(),
                )

    def test_unsafe_relative_parts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                runtime_paths, "resource_root", return_value=Path(directory)
            ):
                with self.assertRaises(ValueError):
                    runtime_paths.resource_path("..", "escape")
                with self.assertRaises(ValueError):
                    runtime_paths.resource_path(Path("C:/absolute"))

    def test_ensure_user_directories_creates_expected_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {runtime_paths.DATA_ROOT_ENV: directory},
                clear=False,
            ):
                paths = runtime_paths.ensure_user_directories()

            self.assertEqual(
                set(paths),
                {"root", "output", "generation", "comparison", "logs", "cache"},
            )
            for path in paths.values():
                self.assertTrue(path.is_dir(), path)
            self.assertEqual(paths["output"], Path(directory).resolve() / "result")
            self.assertEqual(
                paths["generation"],
                Path(directory).resolve() / "result" / "generation",
            )
            self.assertEqual(
                paths["comparison"],
                Path(directory).resolve() / "result" / "comparison",
            )

    def test_frozen_bearing_runtime_contract_loads_from_packaged_tree(self) -> None:
        source_root = runtime_paths.resource_root(frozen=False)
        contract_payload = json.loads(
            DEFAULT_FROZEN_BEARING_CONTRACT.read_text(encoding="utf-8")
        )
        runtime_roles = {
            "checkpoint",
            "label_map",
            "quality_calibration",
            "model_definition",
            "quality_gate_implementation",
            "preprocessing_implementation",
            "inference_implementation",
            "dataset_manifest",
        }

        with tempfile.TemporaryDirectory() as directory:
            packaged_root = Path(directory)
            contract_relative = DEFAULT_FROZEN_BEARING_CONTRACT.relative_to(source_root)
            files_to_copy = [contract_relative]
            files_to_copy.extend(
                Path(record["path"])
                for record in contract_payload["integrity"]["files"]
                if record.get("role") in runtime_roles
            )
            for relative in files_to_copy:
                destination = packaged_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / relative, destination)

            runtime = load_frozen_bearing_runtime_contract(
                packaged_root / contract_relative,
                project_root=packaged_root,
            )

            self.assertEqual(runtime.project_root, packaged_root.resolve())
            self.assertEqual(
                runtime.checkpoint_path,
                packaged_root
                / "artifacts/checkpoints/bearing/anti_noise/"
                / "bearing_cnn_anti_noise_20260904_best.pt",
            )
            self.assertEqual(runtime.normalization, "window_zscore")
            self.assertEqual(runtime.quality_gate_version, "V2")

    def test_app_icon_is_bundled_resource(self) -> None:
        self.assertTrue(runtime_paths.resource_path("assets", "airwatch.ico").is_file())


if __name__ == "__main__":
    unittest.main()
