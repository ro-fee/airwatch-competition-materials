"""PyTorch dataset for hash-bound KU Leuven materialized IQ windows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_provenance import sha256_file
from .ku_leuven_preprocessing import INPUT_WINDOW_SAMPLES, PREPROCESSING_ID


KU_LEUVEN_KNOWN_LABELS = {
    "frysky": 0,
    "spektrum_dx4e": 1,
    "dji_mini2_rc": 2,
}


class KULeuvenDatasetError(ValueError):
    """Raised when a materialized dataset violates its evidence contract."""


@dataclass(frozen=True)
class KULeuvenWindowSample:
    data_index: int
    window_id: str
    recording_id: str
    archive_id: str
    label_index: int
    member_index: int
    member_path: str
    member_sha256: str
    split: str
    start_sample: int
    end_sample_exclusive: int
    preprocessing_id: str


class KULeuvenMaterializedDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Read immutable train/validation arrays produced by the data module."""

    def __init__(self, root: str | Path, *, split: str, verify_hashes: bool = True) -> None:
        self.root = Path(root)
        metadata_path = self.root / "dataset-metadata.json"
        try:
            self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KULeuvenDatasetError(f"无法读取物化数据元数据：{exc}") from exc
        if self.metadata.get("artifact_type") != "ku_leuven_materialized_known_train_validation":
            raise KULeuvenDatasetError("物化数据类型不匹配")
        if self.metadata.get("preprocessing_id") != PREPROCESSING_ID:
            raise KULeuvenDatasetError("预处理版本与当前代码不一致")
        if split not in self.metadata.get("split_counts", {}):
            raise KULeuvenDatasetError(f"物化数据不包含划分：{split}")
        if split not in {"train", "validation"}:
            raise KULeuvenDatasetError("该数据接口不开放 test 或 unknown")
        self.split = split
        split_dir = self.root / split
        self.data_path = split_dir / "data.npy"
        self.labels_path = split_dir / "labels.npy"
        self.windows_path = split_dir / "windows.csv"
        artifact_specs = self.metadata.get("artifacts", {}).get(split, {})
        for name, path in (
            ("data.npy", self.data_path),
            ("labels.npy", self.labels_path),
            ("windows.csv", self.windows_path),
        ):
            spec = artifact_specs.get(name)
            if not isinstance(spec, dict) or not path.is_file():
                raise KULeuvenDatasetError(f"{split}/{name} 缺失或未登记")
            if int(spec.get("size_bytes", -1)) != path.stat().st_size:
                raise KULeuvenDatasetError(f"{split}/{name} 字节数不一致")
            if verify_hashes and spec.get("sha256") != sha256_file(path):
                raise KULeuvenDatasetError(f"{split}/{name} SHA-256 不一致")

        self._data = np.load(self.data_path, mmap_mode="r")
        self._labels = np.load(self.labels_path, mmap_mode="r")
        expected_count = int(self.metadata["split_counts"][split])
        if self._data.shape != (expected_count, 2, INPUT_WINDOW_SAMPLES):
            raise KULeuvenDatasetError(f"{split} data.npy 形状不一致")
        if self._data.dtype != np.float32:
            raise KULeuvenDatasetError(f"{split} data.npy 必须是 float32")
        if self._labels.shape != (expected_count,) or self._labels.dtype != np.int64:
            raise KULeuvenDatasetError(f"{split} labels.npy 形状或类型不一致")
        if not np.isfinite(self._data).all():
            raise KULeuvenDatasetError(f"{split} data.npy 包含非有限值")
        if np.any(self._labels < 0) or np.any(self._labels >= len(KU_LEUVEN_KNOWN_LABELS)):
            raise KULeuvenDatasetError(f"{split} 标签超出预注册范围")

        with self.windows_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_count:
            raise KULeuvenDatasetError(f"{split} 窗口追溯行数不一致")
        self.samples = tuple(
            KULeuvenWindowSample(
                data_index=int(row["data_index"]),
                window_id=row["window_id"],
                recording_id=row["recording_id"],
                archive_id=row["archive_id"],
                label_index=int(row["label_index"]),
                member_index=int(row["member_index"]),
                member_path=row["member_path"],
                member_sha256=row["member_sha256"],
                split=row["split"],
                start_sample=int(row["start_sample"]),
                end_sample_exclusive=int(row["end_sample_exclusive"]),
                preprocessing_id=row["preprocessing_id"],
            )
            for row in rows
        )
        if [sample.data_index for sample in self.samples] != list(range(expected_count)):
            raise KULeuvenDatasetError(f"{split} data_index 必须连续且与数组行一致")
        for sample in self.samples:
            if (
                sample.split != split
                or sample.preprocessing_id != PREPROCESSING_ID
                or sample.label_index != int(self._labels[sample.data_index])
                or sample.end_sample_exclusive - sample.start_sample != INPUT_WINDOW_SAMPLES
                or len(sample.member_sha256) != 64
            ):
                raise KULeuvenDatasetError(f"窗口追溯契约不一致：{sample.window_id}")
        self.recording_ids = frozenset(sample.recording_id for sample in self.samples)
        self.label_map = dict(KU_LEUVEN_KNOWN_LABELS)
        self.num_classes = len(self.label_map)
        self.channels = 2
        self.window_size = INPUT_WINDOW_SAMPLES
        self.data_identity = {
            "artifact_type": self.metadata["artifact_type"],
            "preprocessing_id": PREPROCESSING_ID,
            "source_split_assignments_sha256": self.metadata["source_split_assignments"]["sha256"],
            "source_window_plan_sha256": [
                item["sha256"] for item in self.metadata["source_window_plans"]
            ],
            "label_map": self.label_map,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if not 0 <= normalized < len(self):
            raise IndexError(index)
        values = np.array(self._data[normalized], dtype=np.float32, order="C", copy=True)
        return torch.from_numpy(values), int(self._labels[normalized]), normalized

    def sample_metadata(self, index: int) -> KULeuvenWindowSample:
        return self.samples[index]

    def close(self) -> None:
        """Release NumPy memory maps deterministically, especially on Windows."""
        for name in ("_data", "_labels"):
            array = getattr(self, name, None)
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None and not memory_map.closed:
                memory_map.close()

    def __enter__(self) -> "KULeuvenMaterializedDataset":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter timing is nondeterministic
        self.close()


__all__ = [
    "KU_LEUVEN_KNOWN_LABELS",
    "KULeuvenDatasetError",
    "KULeuvenMaterializedDataset",
    "KULeuvenWindowSample",
]
