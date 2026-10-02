"""Generate reproducible evidence for one bounded MATLAB v7.3 IQ window read."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from .data_provenance import sha256_file
from .matlab_v73_iq import inspect_mat_v73_iq, read_mat_v73_iq_window


def build_window_read_acceptance(
    path: str | Path,
    *,
    start_sample: int,
    window_size: int,
    sample_rate_hz: float,
    dataset_name: str = "uhd_samps",
) -> dict[str, Any]:
    """Compute source-bound acceptance evidence; this is not model evaluation."""
    source = Path(path)
    metadata = inspect_mat_v73_iq(
        source, dataset_name=dataset_name, sample_rate_hz=sample_rate_hz
    )
    window = read_mat_v73_iq_window(
        source,
        start_sample=start_sample,
        window_size=window_size,
        dataset_name=dataset_name,
        sample_rate_hz=sample_rate_hz,
    )
    samples64 = window.samples.astype(np.float64, copy=False)
    import h5py

    return {
        "schema_version": "1.0",
        "artifact_type": "matlab_v73_iq_bounded_window_read_acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": source.as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
        "input_contract": {
            "dataset_name": metadata.dataset_name,
            "source_shape": list(metadata.source_shape),
            "source_dtype": metadata.source_dtype,
            "matlab_class": metadata.matlab_class,
            "orientation": metadata.orientation,
            "sample_rate_hz": metadata.sample_rate_hz,
            "total_sample_count": metadata.sample_count,
            "recording_duration_seconds": metadata.duration_seconds,
        },
        "bounded_read": {
            "start_sample": window.start_sample,
            "end_sample_exclusive": window.end_sample,
            "window_size": window.sample_count,
            "window_duration_seconds": window.duration_seconds,
            "output_shape": list(window.samples.shape),
            "output_dtype": str(window.samples.dtype),
            "output_bytes": window.samples.nbytes,
            "writeable": bool(window.samples.flags.writeable),
            "all_finite": bool(np.isfinite(window.samples).all()),
            "nonzero_fraction": float(np.count_nonzero(window.samples) / window.samples.size),
            "rms_i": float(np.sqrt(np.mean(np.square(samples64[0])))),
            "rms_q": float(np.sqrt(np.mean(np.square(samples64[1])))),
        },
        "passed": True,
        "scope": "format and bounded-read acceptance only; no recognition metric was computed",
        "training_eligible": False,
        "training_blocker": (
            "A recording manifest, leakage-audited split, common single-band IQ representation, "
            "and compatible trained model are still required."
        ),
    }


def write_window_read_acceptance(report: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite existing window-read acceptance evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--start-sample", type=int, required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--sample-rate-hz", type=float, required=True)
    parser.add_argument("--dataset-name", default="uhd_samps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_window_read_acceptance(
        args.path,
        start_sample=args.start_sample,
        window_size=args.window_size,
        sample_rate_hz=args.sample_rate_hz,
        dataset_name=args.dataset_name,
    )
    written = write_window_read_acceptance(report, args.output)
    print(json.dumps({"output": str(written), "passed": report["passed"]}))


if __name__ == "__main__":
    main()
