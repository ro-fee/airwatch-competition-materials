"""Read-only audit of the maintained historical-recognition workflow.

Run with the dedicated project Python and ``--output <writable JSON path>``.
Synthetic input verifies decoding/model plumbing only; it is never an accuracy
evaluation and never supplies ground-truth labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from airwatch.data.recognition_input import RecognitionSnapshot
from airwatch.general_recognition_contract import TASKS
from airwatch.inference.general_models import load_general_model
from airwatch.workflows.general_recognition import predict_one


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(output: Path) -> None:
    observations = {
        "scope": (
            "Maintained Data/Inference/Workflow audit using synthetic signals; "
            "not an accuracy evaluation"
        ),
        "python": sys.executable,
        "python_version": sys.version,
        "torch": torch.__version__,
        "device": "cpu",
        "tasks": [],
    }
    hashes_before: dict[str, str] = {}

    for task_name, spec in TASKS.items():
        sample_count = spec.window_size * 2
        phase = np.arange(sample_count, dtype=np.float32) * np.float32(0.073)
        signal = np.stack(
            [np.sin(phase + channel * 0.1) for channel in range(spec.channels)]
        ).astype(np.float32)
        snapshot = RecognitionSnapshot(
            task_name=task_name,
            filenames=(f"synthetic/{spec.demo_subdir}/unlabelled.dat",),
            data=signal,
            labels=(),
            window_size=spec.window_size,
        )

        model, checkpoint = load_general_model(task_name, device="cpu")
        checkpoint = Path(checkpoint).resolve()
        relative_checkpoint = str(checkpoint.relative_to(ROOT))
        hashes_before[relative_checkpoint] = _sha256(checkpoint)
        progress: list[tuple[int, int]] = []
        result = predict_one(
            (model, checkpoint),
            snapshot,
            threading.Event(),
            lambda done, total: progress.append((done, total)),
            batch_size=2,
        )
        observations["tasks"].append(
            {
                "task": task_name,
                "model": result.model_name,
                "checkpoint": relative_checkpoint,
                "classes": list(spec.classes),
                "window_size": spec.window_size,
                "channels": spec.channels,
                "window_count": len(result.window_predictions[0]),
                "feature_shape": list(result.features.shape),
                "prediction_count": len(result.predicted_labels),
                "has_ground_truth": result.has_ground_truth,
                "discarded_samples": list(result.discarded_samples),
                "progress_final": list(progress[-1]) if progress else None,
            }
        )

    hashes_after = {name: _sha256(ROOT / name) for name in hashes_before}
    observations["weights_unchanged"] = hashes_before == hashes_after
    observations["weights_sha256"] = hashes_after

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(observations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output)
    if not observations["weights_unchanged"]:
        raise RuntimeError("Read-only weight integrity failed")
    if any(item["has_ground_truth"] for item in observations["tasks"]):
        raise RuntimeError("Synthetic audit input must remain unlabelled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
