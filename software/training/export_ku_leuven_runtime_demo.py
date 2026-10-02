"""Export one traceable validation recording for local desktop-flow rehearsal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from airwatch.data import KULeuvenMaterializedDataset
from airwatch.data.data_provenance import sha256_file
from training.common import write_json


def export_demo(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    recording_id: str | None = None,
) -> dict[str, object]:
    output = Path(output_dir).resolve()
    dataset = KULeuvenMaterializedDataset(
        dataset_root, split="validation", verify_hashes=True
    )
    try:
        selected_id = recording_id or dataset.samples[0].recording_id
        indices = [
            index
            for index, sample in enumerate(dataset.samples)
            if sample.recording_id == selected_id
        ]
        if len(indices) != 32:
            raise ValueError(
                f"本机演示严格要求一个完整的 32 窗口成员，实际为 {len(indices)}"
            )
        safe_id = selected_id.replace(":", "_").replace("/", "_").replace("\\", "_")
        array_path = output / f"{safe_id}.npy"
        metadata_path = output / f"{safe_id}.json"
        if array_path.exists() or metadata_path.exists():
            raise FileExistsError("拒绝覆盖已有本机演示产物")
        output.mkdir(parents=True, exist_ok=True)
        windows = np.stack([dataset[index][0].numpy() for index in indices])
        continuous_iq = np.concatenate(windows, axis=-1)
        np.save(array_path, continuous_iq, allow_pickle=False)
        first = dataset.sample_metadata(indices[0])
        payload = {
            "schema_version": "1.0",
            "artifact_type": "ku_leuven_local_runtime_demo_recording",
            "purpose": "Local desktop-flow rehearsal only; not performance evidence.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_split": "validation",
            "recording_id": selected_id,
            "archive_id": first.archive_id,
            "label_index": first.label_index,
            "label": next(
                label for label, index in dataset.label_map.items() if index == first.label_index
            ),
            "member_path": first.member_path,
            "member_sha256": first.member_sha256,
            "window_ids": [dataset.sample_metadata(index).window_id for index in indices],
            "preprocessing_id": first.preprocessing_id,
            "array": {
                "path": str(array_path),
                "shape": list(continuous_iq.shape),
                "dtype": str(continuous_iq.dtype),
                "layout": "channel_first_concatenated_windows",
                "window_count": len(indices),
                "window_samples": int(windows.shape[-1]),
                "size_bytes": array_path.stat().st_size,
                "sha256": sha256_file(array_path),
            },
            "data_identity": dataset.data_identity,
            "test_data_used": False,
            "unknown_data_used": False,
            "competition_metric_claim_allowed": False,
            "redistribution_allowed_by_this_artifact": False,
        }
        write_json(metadata_path, payload)
        return {**payload, "metadata_path": str(metadata_path)}
    finally:
        dataset.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recording-id")
    args = parser.parse_args(argv)
    result = export_demo(
        args.dataset_root, args.output_dir, recording_id=args.recording_id
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
