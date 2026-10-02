"""Packaged-runtime model loading and finite forward-pass smoke checks.

This module verifies deployment plumbing only.  It never evaluates accuracy,
changes checkpoints, reads training data, or publishes model metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any


def _finite_shape(tensor: object, *, name: str) -> list[int]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} did not return a tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} returned non-finite values")
    return list(tensor.shape)


def run_runtime_model_smoke() -> dict[str, Any]:
    """Load every shipped model role on CPU and return a JSON-safe report."""

    import numpy as np
    import torch

    from airwatch.general_recognition_contract import TASKS
    from airwatch.inference.bearing import BearingPredictor
    from airwatch.inference.bearing_contract import load_frozen_bearing_runtime_contract
    from airwatch.inference.general_models import load_general_model
    from airwatch.inference.ku_leuven import (
        KULeuvenKnownSourcePredictor,
        load_ku_leuven_runtime_contract,
    )
    from airwatch.models.uav_baselines import DroneRFTCN
    from airwatch.runtime_paths import resource_path, resource_root
    from airwatch.runtime_resources import (
        KU_LEUVEN_SOFTWARE_CONTRACT,
        build_runtime_resource_manifest,
        validate_runtime_resource_manifest,
    )
    from airwatch.data.historical_comparison import (
        default_historical_comparison_manifest,
    )
    from airwatch.workflows.ku_leuven_recognition import KULeuvenRecognitionWorkflow
    from airwatch.workflows.historical_generation import (
        GenerationRequest,
        HistoricalGenerationWorkflow,
    )
    from airwatch.workflows.historical_model_comparison import (
        ComparisonRunRequest,
        HistoricalModelComparisonWorkflow,
    )
    from airwatch.workflows.uav_intake import prepare_uav_input

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    root = resource_root()
    manifest_path = resource_path("runtime-resources.json")
    manifest = (
        manifest_path
        if manifest_path.is_file()
        else build_runtime_resource_manifest(root)
    )
    manifest_result = validate_runtime_resource_manifest(
        root,
        manifest,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "runtime loading and finite forward passes; not an accuracy evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "runtime_resources": manifest_result,
        "models": {},
    }

    with torch.inference_mode():
        for task_name, spec in TASKS.items():
            model, checkpoint = load_general_model(task_name, "cpu")
            _, logits = model(
                torch.zeros(1, spec.channels, spec.window_size, dtype=torch.float32)
            )
            report["models"][f"historical:{task_name}"] = {
                "checkpoint": checkpoint.name,
                "logits_shape": _finite_shape(logits, name=task_name),
            }

        with TemporaryDirectory(prefix="airwatch-generation-smoke-") as directory:
            generation = HistoricalGenerationWorkflow(device="cpu").run(
                GenerationRequest(Path(directory), "32PSK", 2, batch_size=2),
                Event(),
            )
            generated = torch.from_numpy(np.array(generation.batches[0], copy=True))
            if not (generation.session_directory / "generation-session.json").is_file():
                raise FileNotFoundError("historical generation session manifest is missing")
        report["models"]["historical:generator"] = {
            "checkpoint": generation.checkpoint_name,
            "output_shape": _finite_shape(generated, name="historical generator"),
            "workflow_atomic_session": True,
        }

        with TemporaryDirectory(prefix="airwatch-comparison-smoke-") as directory:
            comparison = HistoricalModelComparisonWorkflow(
                device="cpu", batch_size=32
            ).run(
                ComparisonRunRequest(
                    default_historical_comparison_manifest(),
                    Path(directory) / "evidence",
                ),
                Event(),
            )
            comparison_payload = json.loads(
                comparison.evidence_file.read_text(encoding="utf-8")
            )
        report["models"]["historical:individual_reference"] = {
            "checkpoint": comparison.reference.checkpoint_name,
            "workflow_total_windows": comparison.total_windows,
            "workflow_atomic_evidence": True,
            "workflow_performance_status": comparison.performance_status,
            "workflow_accuracy_published": comparison.accuracy is not None,
            "workflow_evidence_schema": comparison_payload["schema_version"],
        }

        bearing_runtime = load_frozen_bearing_runtime_contract()
        bearing = BearingPredictor(
            bearing_runtime.checkpoint_path,
            device="cpu",
            label_map_path=bearing_runtime.label_map_path,
        )
        bearing_logits = bearing.model(
            torch.zeros(1, 1, bearing.window_size, dtype=torch.float32)
        )
        report["models"]["bearing:diagnosis"] = {
            "checkpoint": bearing_runtime.checkpoint_path.name,
            "logits_shape": _finite_shape(bearing_logits, name="bearing diagnosis"),
        }

        uav_contract = load_ku_leuven_runtime_contract(
            resource_path(KU_LEUVEN_SOFTWARE_CONTRACT),
            project_root=root,
        )
        uav = KULeuvenKnownSourcePredictor(uav_contract, device="cpu", batch_size=1)
        if not isinstance(uav.model, DroneRFTCN):
            raise TypeError("KU Leuven runtime did not construct DroneRFTCN")
        uav_logits = uav.model(
            torch.zeros(
                1,
                uav_contract.input_channels,
                uav_contract.window_samples,
                dtype=torch.float32,
            )
        )
        with TemporaryDirectory(prefix="airwatch-runtime-smoke-") as directory:
            input_path = Path(directory) / "uav-known-source-input.npy"
            sample_index = np.arange(
                uav_contract.windows_per_recording * uav_contract.window_samples,
                dtype=np.float32,
            )
            np.save(
                input_path,
                np.stack(
                    (
                        np.sin(sample_index / 17.0),
                        np.cos(sample_index / 19.0),
                    )
                ),
            )
            prepared = prepare_uav_input(
                input_path,
                declared_sample_rate_hz=uav_contract.sample_rate_hz,
            )
            workflow_result = KULeuvenRecognitionWorkflow(uav).as_background_backend(
                prepared,
                Event(),
                lambda _current, _total: None,
            )
        if workflow_result.window_count != uav_contract.windows_per_recording:
            raise ValueError("UAV workflow returned an unexpected window count")
        report["models"]["uav:known_source"] = {
            "checkpoint": uav_contract.checkpoint_path.name,
            "logits_shape": _finite_shape(uav_logits, name="UAV known source"),
            "open_set_status": uav_contract.open_set_status,
            "workflow_window_count": workflow_result.window_count,
            "workflow_contract_id": workflow_result.contract_id,
            "workflow_quality_status": workflow_result.quality_status.value,
        }

    return report


def write_runtime_model_smoke_report(output_path: str | Path) -> dict[str, Any]:
    """Run the smoke check and atomically write its machine-readable report."""

    report = run_runtime_model_smoke()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return report


__all__ = ["run_runtime_model_smoke", "write_runtime_model_smoke_report"]
