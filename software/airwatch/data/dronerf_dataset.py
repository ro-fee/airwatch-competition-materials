"""PyTorch dataset for verified DroneRF windows with explicit band selection."""
from __future__ import annotations

import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_provenance import sha256_file


class DroneRFDatasetError(ValueError):
    """Raised when materialized DroneRF evidence and samples disagree."""


DRONE_TYPE_LABELS = {
    "background": 0,
    "parrot_bebop": 1,
    "parrot_ar": 2,
    "dji_phantom_3": 3,
}


_BAND_SELECTIONS = {
    "both": {
        "source_channel_indices": (0, 1),
        "input_semantics": "lower_and_upper_halves_of_2_4ghz_real_time_domain_amplitude",
    },
    "low_2_4ghz": {
        "source_channel_indices": (0,),
        "input_semantics": "lower_half_of_2_4ghz_real_time_domain_amplitude",
    },
}


@dataclass(frozen=True)
class DroneRFWindowSample:
    sample_id: str
    recording_id: str
    split: str
    code: str
    drone_type: str
    label_index: int
    window_index: int
    start_sample: int
    artifact_path: str


class DroneRFWindowDataset(Dataset):
    """Read one fixed split and expose an explicit subset of its RF bands."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        normalization: str = "per_channel_window_zscore",
        band_selection: str = "both",
        cache_recordings: int = 8,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        if split not in {"train", "validation", "test"}:
            raise DroneRFDatasetError(f"非法数据集合：{split}")
        if normalization not in {"none", "per_channel_window_zscore"}:
            raise DroneRFDatasetError(f"不支持的归一化：{normalization}")
        if band_selection not in _BAND_SELECTIONS:
            raise DroneRFDatasetError(f"不支持的频段选择：{band_selection}")
        if cache_recordings < 1:
            raise DroneRFDatasetError("录制缓存数量至少为 1")
        self.split = split
        self.normalization = normalization
        self.band_selection = band_selection
        self.cache_recordings = cache_recordings
        self.label_map = dict(DRONE_TYPE_LABELS)
        self.num_classes = len(self.label_map)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

        try:
            self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
            self.verification = json.loads((self.root / "verification.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DroneRFDatasetError(f"无法读取 DroneRF 数据证据：{exc}") from exc
        if not self.verification.get("ok") or self.verification.get("findings"):
            raise DroneRFDatasetError("DroneRF 落盘验证未通过，拒绝训练")
        if self.metadata.get("dataset") != "DroneRF" or self.metadata.get("dataset_version") != "1":
            raise DroneRFDatasetError("仅支持已验证的 DroneRF V1 数据")
        window_spec = self.metadata.get("window_spec", {})
        self.window_size = int(window_spec.get("window_length", 0))
        band_contract = _BAND_SELECTIONS[band_selection]
        self.source_channel_indices = tuple(band_contract["source_channel_indices"])
        self.input_semantics = str(band_contract["input_semantics"])
        self.channels = len(self.source_channel_indices)
        self.preprocessing_version = str(window_spec.get("preprocessing_version", ""))
        if self.window_size < 1 or not self.preprocessing_version:
            raise DroneRFDatasetError("metadata.json 缺少窗口或预处理版本")

        manifest_path = self.root / str(self.metadata.get("window_manifest", "window-manifest.csv"))
        try:
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                source_rows = list(csv.DictReader(handle))
        except OSError as exc:
            raise DroneRFDatasetError(f"无法读取窗口清单：{exc}") from exc
        all_sample_ids = [row.get("sample_id", "") for row in source_rows]
        if len(set(all_sample_ids)) != len(all_sample_ids):
            raise DroneRFDatasetError("窗口清单包含重复 sample_id")
        self.samples: list[DroneRFWindowSample] = []
        for row in source_rows:
            if row.get("split") != split:
                continue
            drone_type = row.get("drone_type", "")
            if drone_type not in self.label_map:
                raise DroneRFDatasetError(f"未知无人机类型标签：{drone_type}")
            if int(row["window_length"]) != self.window_size or int(row["channels"]) != 2:
                raise DroneRFDatasetError(f"窗口结构与元数据不一致：{row['sample_id']}")
            if row.get("preprocessing_version") != self.preprocessing_version:
                raise DroneRFDatasetError(f"预处理版本不一致：{row['sample_id']}")
            artifact = (self.root / row["artifact_path"]).resolve()
            if not artifact.is_relative_to(self.root) or not artifact.is_file():
                raise DroneRFDatasetError(f"窗口产物路径非法或不存在：{row['artifact_path']}")
            self.samples.append(DroneRFWindowSample(
                sample_id=row["sample_id"],
                recording_id=row["recording_id"],
                split=split,
                code=row["code"],
                drone_type=drone_type,
                label_index=self.label_map[drone_type],
                window_index=int(row["window_index"]),
                start_sample=int(row["start_sample"]),
                artifact_path=row["artifact_path"],
            ))
        if not self.samples:
            raise DroneRFDatasetError(f"DroneRF {split} 集为空")
        expected = int(self.metadata["counts_by_split"][split])
        if len(self.samples) != expected:
            raise DroneRFDatasetError(
                f"{split} 窗口数不一致：期望 {expected}，实际 {len(self.samples)}"
            )
        self.manifest_path = manifest_path
        self.data_identity = {
            "dataset": "DroneRF",
            "dataset_version": "1",
            "preprocessing_version": self.preprocessing_version,
            "metadata_sha256": sha256_file(self.root / "metadata.json"),
            "verification_sha256": sha256_file(self.root / "verification.json"),
            "window_manifest_sha256": sha256_file(manifest_path),
            "source_recording_manifest_sha256": self.metadata["source_recording_manifest_sha256"],
            "source_split_manifest_sha256": self.metadata["source_split_manifest_sha256"],
            "split_protocol": self.metadata["split_protocol"],
            "split_seed": self.metadata["split_seed"],
        }
        if self.band_selection != "both":
            self.data_identity.update({
                "band_selection": self.band_selection,
                "source_channel_indices": list(self.source_channel_indices),
                "input_channels": self.channels,
                "input_semantics": self.input_semantics,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def _recording_array(self, relative_path: str) -> np.ndarray:
        if relative_path in self._cache:
            array = self._cache.pop(relative_path)
            self._cache[relative_path] = array
            return array
        path = self.root / relative_path
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.dtype != np.float32 or array.ndim != 3 or array.shape[1:] != (2, self.window_size):
            raise DroneRFDatasetError(f"窗口产物类型或形状错误：{relative_path}")
        self._cache[relative_path] = array
        while len(self._cache) > self.cache_recordings:
            self._cache.popitem(last=False)
        return array

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        sample = self.samples[index]
        recording = self._recording_array(sample.artifact_path)
        if not 0 <= sample.window_index < recording.shape[0]:
            raise DroneRFDatasetError(f"窗口索引越界：{sample.sample_id}")
        values = np.array(
            recording[sample.window_index, self.source_channel_indices, :],
            dtype=np.float32,
            copy=True,
        )
        if self.normalization == "per_channel_window_zscore":
            means = values.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
            standard_deviations = values.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
            if np.any(standard_deviations < 1e-6):
                raise DroneRFDatasetError(f"窗口缺少幅值变化：{sample.sample_id}")
            values = (values - means) / standard_deviations
        if not np.isfinite(values).all():
            raise DroneRFDatasetError(f"窗口包含非有限值：{sample.sample_id}")
        return torch.from_numpy(values), sample.label_index, index

    def sample_metadata(self, index: int) -> DroneRFWindowSample:
        return self.samples[index]

    @property
    def recording_ids(self) -> set[str]:
        return {sample.recording_id for sample in self.samples}


__all__ = [
    "DRONE_TYPE_LABELS",
    "DroneRFDatasetError",
    "DroneRFWindowDataset",
    "DroneRFWindowSample",
]
