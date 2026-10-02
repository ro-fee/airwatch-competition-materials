"""Read-only runtime boundary for the KU Leuven known-source TCN candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
import torch

from airwatch.data.ku_leuven_preprocessing import (
    INPUT_SAMPLE_RATE_HZ,
    INPUT_WINDOW_SAMPLES,
    PREPROCESSING_ID,
    preprocess_iq_window,
)
from airwatch.models import DroneRFTCN
from airwatch.runtime_paths import resource_path, resource_root

from .uav_open_set import aggregate_single_recording_probabilities


DEFAULT_KU_LEUVEN_DEVELOPMENT_CONTRACT = resource_path(
    "artifacts",
    "evidence",
    "uav",
    "ku_leuven",
    "local_development",
    "ku_leuven_tcn_strong_noise_seed20260910_v2_runtime_contract.json",
)


class KULeuvenInferenceError(ValueError):
    """Raised when the frozen runtime contract or IQ input is unusable."""


@dataclass(frozen=True)
class KULeuvenRuntimeContract:
    contract_path: Path
    project_root: Path
    contract_id: str
    release_tier: str
    model_id: str
    model_version: str
    model_name: str
    training_seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    label_map: dict[str, int]
    display_labels: dict[str, str]
    preprocessing_id: str
    sample_rate_hz: int
    input_channels: int
    window_samples: int
    windows_per_recording: int
    aggregation_protocol: str
    open_set_status: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_path"] = str(self.contract_path)
        payload["project_root"] = str(self.project_root)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["limitations"] = list(self.limitations)
        return payload


@dataclass(frozen=True)
class KULeuvenWindowPrediction:
    window_index: int
    predicted_class: int
    predicted_label: str
    display_label: str
    confidence: float
    probabilities: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KULeuvenRecordingPrediction:
    contract_id: str
    release_tier: str
    model_id: str
    model_version: str
    model_sha256: str
    device: str
    preprocessing_id: str
    aggregation_protocol: str
    window_count: int
    predicted_class: int
    predicted_label: str
    display_label: str
    confidence: float
    probabilities: tuple[float, ...]
    open_set_status: str
    known_unknown: None
    elapsed_seconds: float
    windows: tuple[KULeuvenWindowPrediction, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["windows"] = [window.to_dict() for window in self.windows]
        payload["limitations"] = list(self.limitations)
        return payload


_RUNTIME_ROLES = {
    "checkpoint",
    "resolved_config",
    "model_definition",
    "preprocessing_implementation",
    "inference_implementation",
}


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KULeuvenInferenceError(f"{name} 必须是 JSON 对象")
    return value


def _resolve_bundled_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise KULeuvenInferenceError("运行契约包含无效资源路径")
    relative = Path(value)
    if relative.is_absolute():
        raise KULeuvenInferenceError("运行契约只能使用包内相对路径")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise KULeuvenInferenceError("运行契约资源路径越界") from exc
    return resolved


def load_ku_leuven_runtime_contract(
    contract_path: str | Path = DEFAULT_KU_LEUVEN_DEVELOPMENT_CONTRACT,
    *,
    project_root: str | Path | None = None,
) -> KULeuvenRuntimeContract:
    """Verify the development runtime release without importing training code."""
    path = Path(contract_path).resolve()
    root = resource_root() if project_root is None else Path(project_root).resolve()
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise KULeuvenInferenceError(f"无法读取无人机模型运行契约：{path}") from exc
    except json.JSONDecodeError as exc:
        raise KULeuvenInferenceError("无人机模型运行契约不是有效 JSON") from exc
    contract = _mapping(contract, "contract")
    if contract.get("schema_version") != 1 or contract.get("status") != "development_frozen":
        raise KULeuvenInferenceError("仅支持 development_frozen V1 运行契约")
    expected_digest = contract.get("contract_digest_sha256")
    body = dict(contract)
    body.pop("contract_digest_sha256", None)
    actual_digest = hashlib.sha256(_canonical_json(body)).hexdigest()
    if not isinstance(expected_digest, str) or actual_digest != expected_digest.lower():
        raise KULeuvenInferenceError("无人机模型运行契约摘要不一致")

    integrity = _mapping(contract.get("integrity"), "integrity")
    records = integrity.get("files")
    if not isinstance(records, list):
        raise KULeuvenInferenceError("integrity.files 必须是列表")
    runtime_files: dict[str, Path] = {}
    runtime_hashes: dict[str, str] = {}
    runtime_relative_paths: dict[str, str] = {}
    for raw in records:
        record = _mapping(raw, "integrity file")
        role = record.get("role")
        if role not in _RUNTIME_ROLES:
            continue
        if role in runtime_files:
            raise KULeuvenInferenceError(f"运行资源角色重复：{role}")
        resource = _resolve_bundled_path(root, record.get("path"))
        if not resource.is_file():
            raise KULeuvenInferenceError(f"无人机模型运行资源缺失：{resource}")
        expected_size = record.get("size_bytes")
        expected_hash = record.get("sha256")
        if resource.stat().st_size != expected_size or _sha256(resource) != expected_hash:
            raise KULeuvenInferenceError(f"无人机模型运行资源完整性失败：{role}")
        runtime_files[str(role)] = resource
        runtime_hashes[str(role)] = str(expected_hash)
        runtime_relative_paths[str(role)] = str(record.get("path"))
    missing = sorted(_RUNTIME_ROLES - set(runtime_files))
    if missing:
        raise KULeuvenInferenceError("无人机运行资源登记不完整：" + ", ".join(missing))

    model = _mapping(contract.get("model"), "model")
    input_contract = _mapping(contract.get("input_contract"), "input_contract")
    aggregation = _mapping(contract.get("aggregation"), "aggregation")
    open_set = _mapping(contract.get("open_set"), "open_set")
    label_map = dict(_mapping(input_contract.get("label_map"), "label_map"))
    display_labels = dict(_mapping(input_contract.get("display_labels"), "display_labels"))
    if (
        model.get("name") != "DroneRFTCN"
        or model.get("checkpoint") != runtime_relative_paths["checkpoint"]
        or model.get("checkpoint_sha256") != runtime_hashes["checkpoint"]
        or input_contract.get("preprocessing_id") != PREPROCESSING_ID
        or int(input_contract.get("sample_rate_hz", -1)) != INPUT_SAMPLE_RATE_HZ
        or int(input_contract.get("channels", -1)) != 2
        or int(input_contract.get("window_samples", -1)) != INPUT_WINDOW_SAMPLES
        or aggregation.get("protocol") != "mean-window-softmax-probability-v1"
        or int(aggregation.get("windows_per_recording", -1)) != 32
        or open_set.get("status") != "not_evaluated"
        or open_set.get("threshold") is not None
    ):
        raise KULeuvenInferenceError("无人机运行契约与当前代码能力不兼容")
    if label_map != {"frysky": 0, "spektrum_dx4e": 1, "dji_mini2_rc": 2}:
        raise KULeuvenInferenceError("无人机运行契约标签映射不匹配")
    if set(display_labels) != set(label_map):
        raise KULeuvenInferenceError("无人机运行契约显示标签不完整")
    limitations = contract.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise KULeuvenInferenceError("开发版运行契约必须声明限制")

    return KULeuvenRuntimeContract(
        contract_path=path,
        project_root=root,
        contract_id=str(contract["contract_id"]),
        release_tier=str(contract["release_tier"]),
        model_id=str(model["id"]),
        model_version=str(model["version"]),
        model_name=str(model["name"]),
        training_seed=int(model["training_seed"]),
        checkpoint_path=runtime_files["checkpoint"],
        checkpoint_sha256=runtime_hashes["checkpoint"],
        label_map={str(key): int(value) for key, value in label_map.items()},
        display_labels={str(key): str(value) for key, value in display_labels.items()},
        preprocessing_id=str(input_contract["preprocessing_id"]),
        sample_rate_hz=int(input_contract["sample_rate_hz"]),
        input_channels=int(input_contract["channels"]),
        window_samples=int(input_contract["window_samples"]),
        windows_per_recording=int(aggregation["windows_per_recording"]),
        aggregation_protocol=str(aggregation["protocol"]),
        open_set_status=str(open_set["status"]),
        limitations=tuple(str(value) for value in limitations),
    )


class KULeuvenKnownSourcePredictor:
    """Load one verified TCN once and predict IQ windows or one recording."""

    def __init__(
        self,
        contract: KULeuvenRuntimeContract,
        *,
        device: str | torch.device = "cpu",
        batch_size: int = 64,
    ) -> None:
        if not isinstance(contract, KULeuvenRuntimeContract):
            raise TypeError("contract 必须是 KULeuvenRuntimeContract")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise KULeuvenInferenceError("batch_size 必须是正整数")
        self.contract = contract
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        try:
            checkpoint = torch.load(
                contract.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise KULeuvenInferenceError(f"无法加载无人机模型检查点：{exc}") from exc
        if not isinstance(checkpoint, dict):
            raise KULeuvenInferenceError("无人机检查点必须是字典")
        if (
            checkpoint.get("model_name") != contract.model_name
            or checkpoint.get("label_map") != contract.label_map
            or checkpoint.get("aggregation_protocol") != contract.aggregation_protocol
            or checkpoint.get("data_identity", {}).get("preprocessing_id")
            != contract.preprocessing_id
            or checkpoint.get("config", {}).get("training", {}).get("seed")
            != contract.training_seed
        ):
            raise KULeuvenInferenceError("无人机检查点身份与运行契约不一致")
        self.index_to_label = {
            index: label for label, index in contract.label_map.items()
        }
        self.model = DroneRFTCN(num_classes=len(contract.label_map), in_channels=2)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise KULeuvenInferenceError(f"无人机模型权重与结构不匹配：{exc}") from exc
        self.model.to(self.device).eval()

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise KULeuvenInferenceError("已请求 CUDA，但当前环境不可用")
        if requested.type not in {"cpu", "cuda"}:
            raise KULeuvenInferenceError("推理设备只支持 cpu、cuda 或 auto")
        return requested

    def _canonical_windows(self, samples: object) -> np.ndarray:
        array = np.asarray(samples)
        if array.dtype.kind == "c" and array.ndim == 1:
            array = np.stack((array.real, array.imag), axis=0)
        if array.dtype.kind not in "fiu":
            raise KULeuvenInferenceError("IQ 输入必须是实数 I/Q 数组或一维复数数组")
        if array.ndim == 2:
            if array.shape[1] == 2 and array.shape[0] != 2:
                array = array.T
            if array.shape[0] != 2 or array.shape[1] % self.contract.window_samples:
                raise KULeuvenInferenceError(
                    f"连续 IQ 输入必须为 [2,N] 或 [N,2]，且 N 是 {self.contract.window_samples} 的整数倍"
                )
            array = array.reshape(2, -1, self.contract.window_samples).transpose(1, 0, 2)
        if array.ndim != 3 or array.shape[1:] != (
            self.contract.input_channels,
            self.contract.window_samples,
        ):
            raise KULeuvenInferenceError(
                f"窗口 IQ 输入必须为 [W,2,{self.contract.window_samples}]"
            )
        if array.shape[0] < 1 or not np.isfinite(array).all():
            raise KULeuvenInferenceError("IQ 输入为空或包含 NaN/无穷值")
        return array

    def predict_windows(
        self,
        samples: object,
        *,
        cancelled: Any | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[KULeuvenWindowPrediction, ...]:
        windows = self._canonical_windows(samples)
        total = int(windows.shape[0])
        rows: list[KULeuvenWindowPrediction] = []
        with torch.inference_mode():
            for start in range(0, total, self.batch_size):
                if cancelled is not None and cancelled.is_set():
                    raise InterruptedError
                stop = min(start + self.batch_size, total)
                normalized = np.stack(
                    [preprocess_iq_window(windows[index]) for index in range(start, stop)]
                )
                inputs = torch.from_numpy(normalized).to(self.device)
                probabilities = torch.softmax(self.model(inputs), dim=1).cpu().numpy()
                for offset, probability in enumerate(probabilities):
                    class_index = int(np.argmax(probability))
                    label = self.index_to_label[class_index]
                    rows.append(
                        KULeuvenWindowPrediction(
                            window_index=start + offset,
                            predicted_class=class_index,
                            predicted_label=label,
                            display_label=self.contract.display_labels[label],
                            confidence=float(probability[class_index]),
                            probabilities=tuple(float(value) for value in probability),
                        )
                    )
                if progress is not None:
                    progress(stop, total)
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError
        return tuple(rows)

    def predict_recording(
        self,
        samples: object,
        *,
        cancelled: Any | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> KULeuvenRecordingPrediction:
        started = perf_counter()
        windows = self._canonical_windows(samples)
        if windows.shape[0] != self.contract.windows_per_recording:
            raise KULeuvenInferenceError(
                "成员级识别严格要求 "
                f"{self.contract.windows_per_recording} 个窗口，实际为 {windows.shape[0]}"
            )
        rows = self.predict_windows(windows, cancelled=cancelled, progress=progress)
        probabilities = aggregate_single_recording_probabilities(
            [row.probabilities for row in rows],
            expected_window_count=self.contract.windows_per_recording,
        )
        class_index = int(np.argmax(probabilities))
        label = self.index_to_label[class_index]
        return KULeuvenRecordingPrediction(
            contract_id=self.contract.contract_id,
            release_tier=self.contract.release_tier,
            model_id=self.contract.model_id,
            model_version=self.contract.model_version,
            model_sha256=self.contract.checkpoint_sha256,
            device=str(self.device),
            preprocessing_id=self.contract.preprocessing_id,
            aggregation_protocol=self.contract.aggregation_protocol,
            window_count=len(rows),
            predicted_class=class_index,
            predicted_label=label,
            display_label=self.contract.display_labels[label],
            confidence=float(probabilities[class_index]),
            probabilities=tuple(float(value) for value in probabilities),
            open_set_status=self.contract.open_set_status,
            known_unknown=None,
            elapsed_seconds=perf_counter() - started,
            windows=rows,
            limitations=self.contract.limitations,
        )


__all__ = [
    "DEFAULT_KU_LEUVEN_DEVELOPMENT_CONTRACT",
    "KULeuvenInferenceError",
    "KULeuvenKnownSourcePredictor",
    "KULeuvenRecordingPrediction",
    "KULeuvenRuntimeContract",
    "KULeuvenWindowPrediction",
    "load_ku_leuven_runtime_contract",
]
