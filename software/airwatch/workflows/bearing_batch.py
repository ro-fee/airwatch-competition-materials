"""Batch diagnosis and evaluation for the frozen bearing checkpoint.

This module is a backend-only orchestration layer.  It reads a previously
validated manifest, reuses one already-loaded diagnosis workflow for every
recording, and computes metrics from the returned predictions.  It does not
train, alter source files, modify the UI, or write checkpoint files.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from airwatch.inference.adapter import BearingInferenceResult

from .bearing_diagnosis import BearingDiagnosisWorkflow


class BearingBatchError(ValueError):
    """Raised when a batch manifest or batch request is invalid."""


_REQUIRED_COLUMNS = {
    "local_path",
    "original_filename",
    "class_name",
    "sensor_key",
}


@dataclass(frozen=True)
class BearingBatchFileResult:
    """Evaluation result for one manifest row / one selected sensor channel."""

    manifest_row: int
    source_path: str
    original_filename: str
    class_name: str
    label_index: int
    split: str | None
    split_group: str | None
    load_hp: str | None
    sensor_key: str
    window_count: int
    predicted_label: str
    predicted_class: int
    mean_confidence: float
    file_correct: bool
    window_correct: int
    window_accuracy: float
    class_counts: dict[str, int]
    windows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["windows"] = [dict(window) for window in self.windows]
        payload["class_counts"] = dict(self.class_counts)
        return payload


@dataclass(frozen=True)
class BearingBatchResult:
    """JSON-friendly aggregate evidence for one manifest evaluation."""

    generated_at: str
    checkpoint_path: str
    manifest_path: str
    project_root: str
    device: str
    window_size: int
    step: int
    label_map: dict[str, str]
    split_filter: str | None
    file_count: int
    window_count: int
    file_accuracy: float
    window_accuracy: float
    macro_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: dict[str, dict[str, float | int]]
    files: tuple[BearingBatchFileResult, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label_map"] = dict(self.label_map)
        payload["confusion_matrix"] = [list(row) for row in self.confusion_matrix]
        payload["per_class"] = {
            str(label): dict(values) for label, values in self.per_class.items()
        }
        payload["files"] = [item.to_dict() for item in self.files]
        payload["limitations"] = list(self.limitations)
        return payload


class BearingBatchDiagnosisWorkflow:
    """Run one frozen predictor over all rows in a validated manifest.

    The workflow loads the checkpoint once in ``from_checkpoint`` and reuses
    that workflow for every file.  The default project root is the repository
    root containing this ``airwatch`` package, so manifest paths such as
    ``datasets/bearing/cwru/raw/...`` resolve consistently on Windows.
    """

    def __init__(
        self,
        workflow: BearingDiagnosisWorkflow,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        self.workflow = workflow
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        label_map_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "BearingBatchDiagnosisWorkflow":
        """Create one reusable batch workflow around a frozen checkpoint."""
        return cls(
            BearingDiagnosisWorkflow.from_checkpoint(
                checkpoint_path,
                device=device,
                label_map_path=label_map_path,
            ),
            project_root=project_root,
        )

    def run_manifest(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        max_files: int | None = None,
        output_path: str | Path | None = None,
    ) -> BearingBatchResult:
        """Evaluate manifest rows and optionally write a new JSON evidence file.

        ``split`` filters by the manifest's split column.  A row with multiple
        sensor keys (``A|B``) is rejected rather than silently guessing a
        channel; use the already-resolved ``split-manifest.csv`` when it is
        available.
        """
        path = Path(manifest_path).resolve()
        rows = self._read_manifest(path, split=split, max_files=max_files)
        if not rows:
            filter_text = f" for split={split!r}" if split is not None else ""
            raise BearingBatchError(f"manifest contains no rows{filter_text}: {path}")

        file_results: list[BearingBatchFileResult] = []
        for row_number, row in rows:
            source_path = self._resolve_source_path(row["local_path"])
            sensor_key = self._single_sensor_key(row["sensor_key"], row_number)
            result = self.workflow.run_file(source_path, sensor_key=sensor_key)
            file_results.append(
                self._file_result(
                    row_number=row_number,
                    row=row,
                    source_path=source_path,
                    sensor_key=sensor_key,
                    result=result,
                )
            )

        batch_result = self._aggregate(
            file_results,
            manifest_path=path,
            split_filter=split,
        )
        if output_path is not None:
            self.write_json(batch_result, output_path)
        return batch_result

    def run_rows(
        self,
        rows: Iterable[Mapping[str, str]],
        *,
        manifest_path: str | Path = "<in-memory>",
        split: str | None = None,
    ) -> BearingBatchResult:
        """Evaluate already-loaded rows; useful for callers and unit tests."""
        selected: list[tuple[int, dict[str, str]]] = []
        for index, raw_row in enumerate(rows, start=2):
            row = {str(key): str(value or "") for key, value in raw_row.items()}
            row = self._prepare_row(row, index)
            if split is None or row.get("split") == split:
                selected.append((index, row))
        if not selected:
            raise BearingBatchError("no rows selected")

        file_results: list[BearingBatchFileResult] = []
        for row_number, row in selected:
            source_path = self._resolve_source_path(row["local_path"])
            sensor_key = self._single_sensor_key(row["sensor_key"], row_number)
            result = self.workflow.run_file(source_path, sensor_key=sensor_key)
            file_results.append(
                self._file_result(
                    row_number=row_number,
                    row=row,
                    source_path=source_path,
                    sensor_key=sensor_key,
                    result=result,
                )
            )
        return self._aggregate(
            file_results,
            manifest_path=Path(manifest_path).resolve()
            if str(manifest_path) != "<in-memory>"
            else Path(manifest_path),
            split_filter=split,
        )

    @staticmethod
    def write_json(result: BearingBatchResult, output_path: str | Path) -> Path:
        """Atomically write batch evidence without partially-written JSON."""
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            dir=destination.parent,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            Path(temporary_name).replace(destination)
        except BaseException:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            finally:
                raise
        return destination

    @staticmethod
    def render_json(result: BearingBatchResult) -> str:
        """Render evidence as readable JSON without writing a file."""
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

    def _read_manifest(
        self,
        path: Path,
        *,
        split: str | None,
        max_files: int | None,
    ) -> list[tuple[int, dict[str, str]]]:
        if not path.is_file():
            raise BearingBatchError(f"manifest does not exist: {path}")
        if max_files is not None and (
            not isinstance(max_files, int) or isinstance(max_files, bool) or max_files <= 0
        ):
            raise BearingBatchError("max_files must be a positive integer")

        selected: list[tuple[int, dict[str, str]]] = []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or ())
                missing = sorted(_REQUIRED_COLUMNS - fieldnames)
                if missing:
                    raise BearingBatchError(
                        f"manifest is missing required columns: {', '.join(missing)}"
                    )
                for row_number, raw_row in enumerate(reader, start=2):
                    row = {
                        str(key): str(value or "") for key, value in raw_row.items()
                    }
                    row = self._prepare_row(row, row_number)
                    if split is not None and row.get("split") != split:
                        continue
                    selected.append((row_number, row))
                    if max_files is not None and len(selected) >= max_files:
                        break
        except UnicodeDecodeError as exc:
            raise BearingBatchError(f"manifest is not UTF-8 text: {path}") from exc
        return selected

    def _prepare_row(
        self,
        row: Mapping[str, str],
        row_number: int,
    ) -> dict[str, str]:
        """Validate a row and resolve its true class against the checkpoint.

        The original source manifest predates the derived ``label_index`` column,
        so that field is optional at the input boundary.  When it is absent, the
        class name is converted through the frozen checkpoint label map.  When it
        is present, both values are checked against the same map so an incorrect
        hand-edited label cannot silently affect the reported metrics.
        """
        normalized = {str(key): str(value or "") for key, value in row.items()}
        self._validate_row(normalized, row_number)

        predictor = self.workflow.adapter.predictor
        label_map = predictor.label_map
        class_name = normalized["class_name"].strip()
        normalized["class_name"] = class_name

        raw_label_index = normalized.get("label_index", "").strip()
        if raw_label_index:
            try:
                label_index = int(raw_label_index)
            except ValueError as exc:
                raise BearingBatchError(
                    f"manifest row {row_number} has invalid label_index: "
                    f"{raw_label_index!r}"
                ) from exc
            if label_index < 0:
                raise BearingBatchError(
                    f"manifest row {row_number} has negative label_index"
                )
            expected_class = label_map.get(label_index)
            if expected_class is None:
                raise BearingBatchError(
                    f"manifest row {row_number} label_index {label_index} "
                    "is outside checkpoint class range"
                )
            if str(expected_class) != class_name:
                raise BearingBatchError(
                    f"manifest row {row_number} class_name/label_index mismatch: "
                    f"class_name={class_name!r}, label_index={label_index} "
                    f"maps to {expected_class!r}"
                )
        else:
            matches = [
                int(index)
                for index, label in label_map.items()
                if str(label) == class_name
            ]
            if len(matches) != 1:
                raise BearingBatchError(
                    f"manifest row {row_number} class_name {class_name!r} "
                    "does not resolve to exactly one checkpoint label"
                )
            label_index = matches[0]

        normalized["label_index"] = str(label_index)
        return normalized

    @staticmethod
    def _validate_row(row: Mapping[str, str], row_number: int) -> None:
        missing = sorted(
            column for column in _REQUIRED_COLUMNS if not row.get(column, "").strip()
        )
        if missing:
            raise BearingBatchError(
                f"manifest row {row_number} is missing values: {', '.join(missing)}"
            )

    @staticmethod
    def _single_sensor_key(value: str, row_number: int) -> str:
        keys = [item.strip() for item in value.split("|") if item.strip()]
        if len(keys) != 1:
            raise BearingBatchError(
                f"manifest row {row_number} lists {len(keys)} sensor keys; "
                "batch evaluation refuses to guess. Use split-manifest.csv or "
                "provide one verified sensor_key."
            )
        return keys[0]

    def _resolve_source_path(self, local_path: str) -> Path:
        candidate = Path(local_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    @staticmethod
    def _file_result(
        *,
        row_number: int,
        row: Mapping[str, str],
        source_path: Path,
        sensor_key: str,
        result: BearingInferenceResult,
    ) -> BearingBatchFileResult:
        true_class = int(row["label_index"])
        windows = tuple(dict(window) for window in result.windows)
        window_correct = sum(
            int(window["predicted_class"]) == true_class for window in windows
        )
        return BearingBatchFileResult(
            manifest_row=row_number,
            source_path=str(source_path),
            original_filename=row["original_filename"],
            class_name=row["class_name"],
            label_index=true_class,
            split=row.get("split") or None,
            split_group=row.get("split_group") or None,
            load_hp=row.get("load_hp") or None,
            sensor_key=sensor_key,
            window_count=result.window_count,
            predicted_label=result.predicted_label,
            predicted_class=result.predicted_class,
            mean_confidence=result.mean_confidence,
            file_correct=result.predicted_class == true_class,
            window_correct=window_correct,
            window_accuracy=(window_correct / result.window_count)
            if result.window_count
            else 0.0,
            class_counts=dict(result.class_counts),
            windows=windows,
        )

    def _aggregate(
        self,
        files: list[BearingBatchFileResult],
        *,
        manifest_path: Path,
        split_filter: str | None,
    ) -> BearingBatchResult:
        predictor = self.workflow.adapter.predictor
        label_map = {str(index): label for index, label in predictor.label_map.items()}
        class_count = len(label_map)
        confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
        for item in files:
            if not 0 <= item.label_index < class_count:
                raise BearingBatchError(
                    f"label_index {item.label_index} is outside checkpoint class range"
                )
            if not 0 <= item.predicted_class < class_count:
                raise BearingBatchError(
                    f"predicted_class {item.predicted_class} is outside checkpoint class range"
                )
            confusion[item.label_index][item.predicted_class] += 1

        file_count = len(files)
        window_count = sum(item.window_count for item in files)
        file_correct = sum(item.file_correct for item in files)
        window_correct = sum(item.window_correct for item in files)
        per_class: dict[str, dict[str, float | int]] = {}
        f1_values: list[float] = []
        for class_index in range(class_count):
            true_positive = confusion[class_index][class_index]
            support = sum(confusion[class_index])
            predicted_total = sum(row[class_index] for row in confusion)
            precision = true_positive / predicted_total if predicted_total else 0.0
            recall = true_positive / support if support else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            f1_values.append(f1)
            per_class[str(class_index)] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }

        return BearingBatchResult(
            generated_at=datetime.now(timezone.utc).isoformat(),
            checkpoint_path=str(Path(predictor.checkpoint_path).resolve()),
            manifest_path=str(manifest_path),
            project_root=str(self.project_root),
            device=str(predictor.device),
            window_size=predictor.window_size,
            step=predictor.step,
            label_map=label_map,
            split_filter=split_filter,
            file_count=file_count,
            window_count=window_count,
            file_accuracy=file_correct / file_count if file_count else 0.0,
            window_accuracy=window_correct / window_count if window_count else 0.0,
            macro_f1=sum(f1_values) / len(f1_values) if f1_values else 0.0,
            confusion_matrix=tuple(tuple(row) for row in confusion),
            per_class=per_class,
            files=tuple(files),
            limitations=(
                "结果来自冻结的 BearingCNN checkpoint，未在本批次训练或调参。",
                "CWRU 是实验室公开数据；结果不代表真实现场、未知设备或低信噪比性能。",
                "文件级指标按多数窗口投票结果统计；窗口级指标按每个窗口统计。",
            ),
        )


__all__ = [
    "BearingBatchDiagnosisWorkflow",
    "BearingBatchError",
    "BearingBatchFileResult",
    "BearingBatchResult",
]

