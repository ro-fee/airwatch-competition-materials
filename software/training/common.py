"""Shared helpers for reproducible bearing training and evaluation."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

from airwatch.data import CWRUBearingDataset
from airwatch.models import BearingCNN


class TrainingConfigError(ValueError):
    """Raised when a training configuration is incomplete or inconsistent."""


_OUTPUT_KEYS = (
    "checkpoint",
    "training_evidence",
    "test_evidence",
    "test_predictions",
)


def resolve_run_name(config: dict[str, Any], run_name: str | None = None) -> str:
    """Resolve and validate the immutable name of one experiment run.

    A run name becomes part of every output filename.  It is deliberately
    restricted to a single safe path component so a command cannot escape the
    configured output directories or accidentally target an unrelated file.
    """
    if run_name is not None:
        candidate = run_name
    else:
        candidate = config.get("output", {}).get("run_name") or config.get(
            "experiment_name"
        )
    if not isinstance(candidate, str) or not candidate.strip():
        raise TrainingConfigError(
            "run name must be a non-empty filename-safe string"
        )
    candidate = candidate.strip()
    if candidate in {".", ".."} or Path(candidate).name != candidate:
        raise TrainingConfigError(
            f"invalid run name {candidate!r}: use one filename-safe name without path separators"
        )
    if any(char in candidate for char in ("/", "\\")):
        raise TrainingConfigError(
            f"invalid run name {candidate!r}: path separators are not allowed"
        )
    if any(char.isspace() for char in candidate):
        raise TrainingConfigError(
            f"invalid run name {candidate!r}: whitespace is not allowed"
        )
    if any(char in candidate for char in '<>:"|?*'):
        raise TrainingConfigError(
            f"invalid run name {candidate!r}: reserved filename characters are not allowed"
        )
    return candidate


def resolve_output_paths(
    config: dict[str, Any],
    project_root: Path,
    run_name: str | None = None,
) -> dict[str, Path]:
    """Return all files a run would create without touching the filesystem."""
    resolved_name = resolve_run_name(config, run_name)
    checkpoint_dir = resolve_project_path(project_root, config["output"]["checkpoint_dir"])
    evidence_dir = resolve_project_path(project_root, config["output"]["evidence_dir"])
    return {
        "checkpoint": checkpoint_dir / f"{resolved_name}_best.pt",
        "training_evidence": evidence_dir / f"{resolved_name}_training.json",
        "test_evidence": evidence_dir / f"{resolved_name}_test.json",
        "test_predictions": evidence_dir / f"{resolved_name}_test_predictions.csv",
    }


def ensure_output_paths_available(paths: dict[str, Path]) -> None:
    """Refuse a run if any planned output already exists.

    This is intentionally a fail-closed policy: new experiments must use a new
    run name rather than overwrite a checkpoint or evidence artifact.
    """
    existing = [
        f"{key}={path}"
        for key in _OUTPUT_KEYS
        if (path := paths.get(key)) is not None and path.exists()
    ]
    if existing:
        details = "; ".join(existing)
        raise TrainingConfigError(
            "refusing to overwrite existing experiment outputs: "
            f"{details}. Choose a new --run-name."
        )


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingConfigError("training config must be a JSON object")
    for section in ("dataset", "training", "output"):
        if not isinstance(payload.get(section), dict):
            raise TrainingConfigError(f"missing config section: {section}")
    payload["_config_path"] = str(path)
    return payload


def project_root_from_config(config: dict[str, Any]) -> Path:
    config_path = Path(config["_config_path"])
    # Config lives at <project>/training/configs/<name>.json.
    return config_path.parents[2]


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def build_datasets(
    config: dict[str, Any],
    project_root: Path,
    *,
    train_normalization: str | None = None,
) -> dict[str, CWRUBearingDataset]:
    """Build the audited splits, optionally overriding train preprocessing.

    The override exists for training-time augmentation: the train windows must
    be loaded in their raw finite form, augmented, and then window-normalized.
    Validation and test always retain the normalization declared by the config.
    The default path is unchanged for historical baseline runs.
    """
    dataset_config = config["dataset"]
    manifest_path = resolve_project_path(project_root, dataset_config["manifest"])
    label_map_path = resolve_project_path(project_root, dataset_config["label_map"])
    datasets = {
        split: CWRUBearingDataset(
            manifest_path,
            split=split,
            label_map_path=label_map_path,
            normalization=(
                train_normalization
                if split == "train" and train_normalization is not None
                else dataset_config["normalization"]
            ),
            project_root=project_root,
            verify_window_counts=True,
        )
        for split in ("train", "validation", "test")
    }
    expected_window_size = int(dataset_config["window_size"])
    expected_step = int(dataset_config["step"])
    expected_classes = int(dataset_config["num_classes"])
    for split, dataset in datasets.items():
        if dataset.window_size != expected_window_size or dataset.step != expected_step:
            raise TrainingConfigError(
                f"{split} dataset does not match configured window settings"
            )
        if dataset.num_classes != expected_classes:
            raise TrainingConfigError(
                f"{split} dataset has {dataset.num_classes} classes, "
                f"config expects {expected_classes}"
            )
    return datasets


def resolve_device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise TrainingConfigError("config requested CUDA, but CUDA is unavailable")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise TrainingConfigError(f"unsupported device: {name!r}")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms make the experiment easier to reproduce.  The
    # training configuration intentionally uses num_workers=0 for this slice.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loader(
    dataset: Dataset[tuple[torch.Tensor, int]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
    )


def make_model(config: dict[str, Any]) -> BearingCNN:
    return BearingCNN(num_classes=int(config["dataset"]["num_classes"]), in_channels=1)


def collect_environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        result["cuda_version"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(0)
    return result


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_size = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            all_targets.extend(int(value) for value in targets.cpu().tolist())
            all_predictions.extend(int(value) for value in logits.argmax(dim=1).cpu().tolist())

    if total_count == 0:
        raise TrainingConfigError("cannot evaluate an empty dataset")
    target_array = np.asarray(all_targets, dtype=np.int64)
    prediction_array = np.asarray(all_predictions, dtype=np.int64)
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        target_array,
        prediction_array,
        labels=labels,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "loss": total_loss / total_count,
        "accuracy": float(np.mean(target_array == prediction_array)),
        "macro_f1": float(f1_score(target_array, prediction_array, labels=labels, average="macro", zero_division=0)),
        "sample_count": total_count,
        "confusion_matrix": confusion_matrix(
            target_array, prediction_array, labels=labels
        ).tolist(),
        "per_class": {
            str(index): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }
    predictions = [
        {"target": target, "prediction": prediction}
        for target, prediction in zip(all_targets, all_predictions)
    ]
    return metrics, predictions


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_predictions_csv(path: Path, predictions: list[dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "target", "prediction"])
        writer.writeheader()
        for index, row in enumerate(predictions):
            writer.writerow({"sample_index": index, **row})
