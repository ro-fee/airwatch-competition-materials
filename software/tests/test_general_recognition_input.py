"""Tests for the historical-recognition task contract and raw-file decoder."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from airwatch.data.recognition_input import (
    discover_recognition_files,
    load_recognition_files,
)
from airwatch.general_recognition_contract import TASKS, task_spec
from airwatch.inference.general_models import TASKS as MODEL_TASKS
from airwatch.runtime_resources import GENERAL_TASK_CHECKPOINTS


class GeneralRecognitionInputTests(unittest.TestCase):
    def _write_task_file(self, root: Path, task_name: str, name: str = "sample.dat") -> Path:
        spec = task_spec(task_name)
        folder = root / spec.classes[0]
        folder.mkdir(parents=True, exist_ok=True)
        count = spec.window_size * spec.channels
        dtype = np.dtype(spec.sample_dtype)
        values = np.arange(count, dtype=dtype)
        path = folder / name
        values.tofile(path)
        return path

    def test_one_contract_drives_model_loader_and_packaging(self) -> None:
        self.assertIs(TASKS, MODEL_TASKS)
        self.assertEqual(
            GENERAL_TASK_CHECKPOINTS,
            {name: spec.checkpoint for name, spec in TASKS.items()},
        )
        self.assertEqual(set(TASKS), {
            "信号个体识别", "信号调制识别", "信号通联识别",
            "信号业务识别", "信号编码识别",
        })

    def test_all_task_formats_decode_to_owned_channel_first_float32(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task_name, spec in TASKS.items():
                path = self._write_task_file(root, task_name, f"{spec.demo_subdir}.dat")
                loaded = load_recognition_files(task_name, [path])
                array = loaded.arrays[0]
                self.assertEqual(array.shape, (spec.channels, spec.window_size), task_name)
                self.assertEqual(array.dtype, np.float32, task_name)
                self.assertFalse(array.flags.writeable, task_name)
                self.assertEqual(loaded.filenames, (str(path.resolve()),))
                self.assertEqual(loaded.inferred_directory_labels, (spec.classes[0],))
                snapshot = loaded.snapshot(2)
                self.assertEqual(snapshot.repeat_count, 2)
                self.assertEqual(snapshot.window_size, spec.window_size)
                self.assertEqual(snapshot.labels, ())

    def test_iq_layouts_are_explicit_and_not_interchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interleaved = self._write_task_file(root, "信号个体识别", "interleaved.dat")
            iq = load_recognition_files("信号个体识别", [interleaved]).arrays[0]
            self.assertEqual(tuple(iq[:, :3].ravel()), (0, 2, 4, 1, 3, 5))

            channel_major = self._write_task_file(root, "信号调制识别", "planar.dat")
            planar = load_recognition_files("信号调制识别", [channel_major]).arrays[0]
            window = task_spec("信号调制识别").window_size
            self.assertEqual(tuple(planar[0, :3]), (0, 1, 2))
            self.assertEqual(tuple(planar[1, :3]), (window, window + 1, window + 2))

    def test_interleaved_iq_rejects_odd_scalar_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odd.dat"
            np.arange(2049, dtype=np.float32).tofile(path)
            with self.assertRaisesRegex(ValueError, "偶数"):
                load_recognition_files("信号个体识别", [path])

    def test_batch_discovery_is_deterministic_and_keeps_full_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "class-a"
            second = root / "class-b"
            first.mkdir(); second.mkdir()
            for path in (first / "z.dat", first / "A.dat", second / "m.dat"):
                path.write_bytes(b"x")
            (first / "nested").mkdir()
            discovered = discover_recognition_files([first, second])
            self.assertEqual(
                discovered,
                ((first / "A.dat").resolve(), (first / "z.dat").resolve(), (second / "m.dat").resolve()),
            )
            self.assertTrue(all(path.is_absolute() for path in discovered))

    def test_file_shorter_than_task_window_is_rejected_at_load_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.dat"
            np.zeros(100, dtype=np.int16).tofile(path)
            with self.assertRaisesRegex(ValueError, "少于任务窗长"):
                load_recognition_files("信号编码识别", [path])

    def test_filename_snr_text_is_not_promoted_to_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("signal.dat", "signal=10.dat"):
                with self.subTest(name=name):
                    path = self._write_task_file(root, "信号编码识别", name)
                    loaded = load_recognition_files("信号编码识别", [path])
                    snapshot = loaded.snapshot(1)
                    self.assertEqual(snapshot.labels, ())
                    self.assertEqual(snapshot.data.shape, (1, task_spec("信号编码识别").window_size))


if __name__ == "__main__":
    unittest.main()
