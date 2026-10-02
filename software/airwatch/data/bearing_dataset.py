"""Dataset boundary for deterministic CWRU bearing experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .cwru import (
    CWRUDataError,
    Normalization,
    load_cwru_signal,
    normalize_windows,
    slice_signal,
)

BearingSplit = Literal["train", "validation", "test"]
_REQUIRED_COLUMNS = {
    "dataset",
    "local_path",
    "original_filename",
    "class_name",
    "label_index",
    "load_hp",
    "rpm",
    "sample_rate_hz",
    "sensor_key",
    "window_size",
    "step",
    "window_count",
    "split",
    "split_group",
}
_VALID_SPLITS = {"train", "validation", "test"}


class BearingDatasetError(ValueError):
    """Raised when the audited bearing manifest cannot be used safely."""


@dataclass(frozen=True)
class BearingSample:
    """One deterministic window reference, without copying signal samples."""

    path: Path
    sensor_key: str
    label_index: int
    class_name: str
    split: BearingSplit
    window_index: int
    window_size: int
    step: int
    split_group: str
    original_filename: str


class CWRUBearingDataset(Dataset[tuple[torch.Tensor, int]]):
    """Read fixed windows from the audited CWRU split manifest.

    The manifest is one row per original recording.  Each row expands to a
    deterministic list of complete windows; no random split, padding, or
    augmentation is performed here.  Source ``.mat`` files remain untouched.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: BearingSplit,
        label_map_path: str | Path | None = None,
        normalization: Normalization = "none",
        project_root: str | Path | None = None,
        verify_window_counts: bool = True,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise BearingDatasetError(f"unsupported split: {split!r}")
        if normalization not in {"none", "zscore", "window_zscore"}:
            raise BearingDatasetError(f"unsupported normalization: {normalization!r}")

        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise BearingDatasetError(f"manifest does not exist: {self.manifest_path}")
        self.split = split
        self.normalization = normalization
        self.project_root = self._resolve_project_root(project_root)
        self._rows = self._read_manifest()
        self.label_map = self._read_label_map(label_map_path)
        self._validate_rows()
        self._samples = self._expand_samples(verify_window_counts=verify_window_counts)
        self._signal_cache: dict[tuple[Path, str], np.ndarray] = {}

    def _resolve_project_root(self, project_root: str | Path | None) -> Path:
        if project_root is not None:
            return Path(project_root).resolve()
        # split-manifest.csv is expected at <project>/datasets/bearing/cwru/.
        if len(self.manifest_path.parents) >= 4:
            return self.manifest_path.parents[3]
        return self.manifest_path.parent

    def _read_manifest(self) -> list[dict[str, str]]:
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = _REQUIRED_COLUMNS - columns
            if missing:
                raise BearingDatasetError(
                    f"manifest missing required columns: {sorted(missing)}"
                )
            rows = [dict(row) for row in reader]
        if not rows:
            raise BearingDatasetError("manifest contains no rows")
        return rows

    def _read_label_map(self, label_map_path: str | Path | None) -> dict[int, str]:
        path = (
            Path(label_map_path).resolve()
            if label_map_path is not None
            else self.manifest_path.parent / "label-map.json"
        )
        if not path.is_file():
            raise BearingDatasetError(f"label map does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            labels = payload["labels"]
            mapping = {int(index): str(name) for index, name in labels.items()}
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BearingDatasetError(f"invalid label map: {path}: {exc}") from exc
        if not mapping:
            raise BearingDatasetError("label map is empty")
        return mapping

    def _validate_rows(self) -> None:
        selected = [row for row in self._rows if row["split"] == self.split]
        if not selected:
            raise BearingDatasetError(f"manifest contains no rows for split={self.split!r}")

        expected_window_size: int | None = None
        expected_step: int | None = None
        seen_paths: set[str] = set()
        for row in selected:
            try:
                label_index = int(row["label_index"])
                window_size = int(row["window_size"])
                step = int(row["step"])
                window_count = int(row["window_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise BearingDatasetError(f"invalid numeric manifest row: {row}") from exc
            if row["dataset"] != "CWRU":
                raise BearingDatasetError(f"unexpected dataset: {row['dataset']!r}")
            if row["split"] not in _VALID_SPLITS:
                raise BearingDatasetError(f"unexpected split: {row['split']!r}")
            if label_index not in self.label_map:
                raise BearingDatasetError(f"label {label_index} is absent from label map")
            if window_size <= 0 or step <= 0 or window_count < 0:
                raise BearingDatasetError(f"invalid window settings: {row}")
            if not row["sensor_key"].endswith("_DE_time"):
                raise BearingDatasetError(f"sensor is not a DE channel: {row['sensor_key']!r}")
            expected_window_size = window_size if expected_window_size is None else expected_window_size
            expected_step = step if expected_step is None else expected_step
            if window_size != expected_window_size or step != expected_step:
                raise BearingDatasetError(
                    "mixed window_size/step values are not supported in one dataset"
                )
            split_group = row["split_group"]
            if not split_group:
                raise BearingDatasetError(f"missing split_group: {row}")
            local_path = row["local_path"]
            if local_path in seen_paths:
                raise BearingDatasetError(
                    f"duplicate source recording within {self.split}: {local_path!r}"
                )
            seen_paths.add(local_path)

        # A split group may have several class-specific recordings in one split,
        # but it must never be shared by different splits.
        paths_by_split: dict[str, set[str]] = {name: set() for name in _VALID_SPLITS}
        groups_by_split: dict[str, set[str]] = {name: set() for name in _VALID_SPLITS}
        for row in self._rows:
            paths_by_split[row["split"]].add(row["local_path"])
            groups_by_split[row["split"]].add(row["split_group"])
        for left in _VALID_SPLITS:
            for right in _VALID_SPLITS - {left}:
                path_overlap = paths_by_split[left] & paths_by_split[right]
                if path_overlap:
                    raise BearingDatasetError(
                        f"source recording appears in multiple splits: {sorted(path_overlap)}"
                    )
                group_overlap = groups_by_split[left] & groups_by_split[right]
                if group_overlap:
                    raise BearingDatasetError(
                        f"split_group appears in multiple splits: {sorted(group_overlap)}"
                    )

    def _resolve_row_path(self, local_path: str) -> Path:
        candidate = Path(local_path)
        if candidate.is_absolute():
            path = candidate
        else:
            path = self.project_root / candidate
            if not path.is_file():
                path = self.manifest_path.parent / candidate
        return path.resolve()

    def _expand_samples(self, *, verify_window_counts: bool) -> list[BearingSample]:
        samples: list[BearingSample] = []
        for row in self._rows:
            if row["split"] != self.split:
                continue
            path = self._resolve_row_path(row["local_path"])
            if not path.is_file():
                raise BearingDatasetError(f"source MAT file does not exist: {path}")
            window_size = int(row["window_size"])
            step = int(row["step"])
            expected_count = int(row["window_count"])
            if verify_window_counts:
                try:
                    record = load_cwru_signal(
                        path,
                        sensor_key=row["sensor_key"],
                        normalization=(
                            "none"
                            if self.normalization == "window_zscore"
                            else self.normalization
                        ),
                    )
                    actual_count = slice_signal(
                        record.signal, window_size=window_size, step=step
                    ).shape[0]
                except CWRUDataError as exc:
                    raise BearingDatasetError(f"cannot audit {path}: {exc}") from exc
                if actual_count != expected_count:
                    raise BearingDatasetError(
                        f"window_count mismatch for {path.name}: "
                        f"manifest={expected_count}, actual={actual_count}"
                    )
            for window_index in range(expected_count):
                samples.append(
                    BearingSample(
                        path=path,
                        sensor_key=row["sensor_key"],
                        label_index=int(row["label_index"]),
                        class_name=row["class_name"],
                        split=self.split,
                        window_index=window_index,
                        window_size=window_size,
                        step=step,
                        split_group=row["split_group"],
                        original_filename=row["original_filename"],
                    )
                )
        if not samples:
            raise BearingDatasetError(f"split={self.split!r} expands to zero windows")
        return samples

    def _get_signal(self, sample: BearingSample) -> np.ndarray:
        key = (sample.path, sample.sensor_key)
        if key not in self._signal_cache:
            record = load_cwru_signal(
                sample.path,
                sensor_key=sample.sensor_key,
                normalization=(
                    "none"
                    if self.normalization == "window_zscore"
                    else self.normalization
                ),
            )
            self._signal_cache[key] = record.signal
        return self._signal_cache[key]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self._samples[index]
        signal = self._get_signal(sample)
        start = sample.window_index * sample.step
        end = start + sample.window_size
        window = signal[start:end]
        if window.shape != (sample.window_size,):
            raise BearingDatasetError(
                f"incomplete window at index={index}: shape={window.shape}"
            )
        if self.normalization == "window_zscore":
            try:
                window = normalize_windows(
                    np.asarray(window, dtype=np.float32)[np.newaxis, :],
                    normalization="window_zscore",
                )[0]
            except CWRUDataError as exc:
                raise BearingDatasetError(
                    f"cannot normalize window at index={index}: {exc}"
                ) from exc
        # Copy into writable contiguous memory before handing it to PyTorch.
        tensor = torch.from_numpy(np.array(window, dtype=np.float32, copy=True)).unsqueeze(0)
        return tensor, sample.label_index

    def sample_metadata(self, index: int) -> BearingSample:
        """Return provenance for one dataset item without loading its samples."""
        return self._samples[index]

    @property
    def window_size(self) -> int:
        return self._samples[0].window_size

    @property
    def step(self) -> int:
        return self._samples[0].step

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    @property
    def source_files(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(sample.path for sample in self._samples))

