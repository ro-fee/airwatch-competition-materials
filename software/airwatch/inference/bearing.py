"""Frozen-checkpoint inference for the CWRU bearing CNN baseline.

This module intentionally remains independent of Qt and of the historical
``models/`` directory.  It loads a checkpoint, applies the preprocessing
contract recorded in that checkpoint, and returns window-level predictions.
No source data or checkpoint is modified.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from airwatch.analysis import (
    BearingQualityThresholds,
    BearingQualityThresholdsV2,
    assess_bearing_window,
    assess_bearing_window_v2,
)
from airwatch.data.cwru import (
    CWRUDataError,
    load_cwru_signal,
    normalize_windows,
    slice_signal,
)
from airwatch.models import BearingCNN


class BearingInferenceError(ValueError):
    """Raised when a bearing checkpoint or inference input is invalid."""


class InsufficientSignalError(BearingInferenceError):
    """Raised when a signal cannot provide one complete inference window."""


@dataclass(frozen=True)
class WindowPrediction:
    """JSON-friendly prediction for one complete signal window."""

    window_index: int
    predicted_label: str
    predicted_class: int
    confidence: float
    probabilities: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for JSON serialization."""
        payload = asdict(self)
        payload["probabilities"] = list(self.probabilities)
        return payload


@dataclass(frozen=True)
class QualityAwareWindowPrediction:
    """Quality-gate result and optional model prediction for one raw window."""

    window_index: int
    accepted: bool
    reason_code: str
    message: str
    quality_score: float
    quality_features: dict[str, float]
    prediction: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for JSON serialization."""
        return asdict(self)


@dataclass(frozen=True)
class QualityAwareWindowPredictionV2:
    """Three-state quality result and optional prediction for one raw window."""

    window_index: int
    status: str
    accepted: bool
    reason_code: str
    message: str
    quality_score: float
    quality_features: dict[str, float]
    prediction: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for strict JSON serialization."""
        return asdict(self)


class BearingPredictor:
    """Run deterministic inference with a frozen ``BearingCNN`` checkpoint.

    The checkpoint is the model contract.  Its embedded config supplies the
    window size, step, normalization mode, class count, and label-map path.
    The default device is CPU so this interface is safe for desktop runtime;
    callers may explicitly select CUDA when it is available.
    """

    _EXPECTED_MODEL_NAME = "BearingCNN"
    _VALID_NORMALIZATIONS = {"none", "zscore", "window_zscore"}

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        label_map_path: str | Path | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise BearingInferenceError(
                f"checkpoint does not exist: {self.checkpoint_path}"
            )

        self.device = self._resolve_device(device)
        try:
            payload = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise BearingInferenceError(
                f"failed to load checkpoint {self.checkpoint_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise BearingInferenceError("checkpoint must contain a dictionary")
        self.checkpoint = payload
        self.config = self._validate_checkpoint(payload)
        dataset_config = self.config["dataset"]

        self.window_size = int(dataset_config["window_size"])
        self.step = int(dataset_config["step"])
        self.normalization = str(dataset_config["normalization"])
        self.num_classes = int(dataset_config["num_classes"])
        self.label_map = self._read_label_map(label_map_path)

        if len(self.label_map) != self.num_classes:
            raise BearingInferenceError(
                "label map class count does not match checkpoint config: "
                f"map={len(self.label_map)}, config={self.num_classes}"
            )
        expected_indices = set(range(self.num_classes))
        if set(self.label_map) != expected_indices:
            raise BearingInferenceError(
                "label map indices must be contiguous from 0 to num_classes-1"
            )

        self.model = BearingCNN(num_classes=self.num_classes, in_channels=1)
        try:
            self.model.load_state_dict(payload["model_state_dict"], strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise BearingInferenceError(
                f"checkpoint weights do not match BearingCNN: {exc}"
            ) from exc
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise BearingInferenceError("CUDA was requested but is unavailable")
        if requested.type not in {"cpu", "cuda"}:
            raise BearingInferenceError(
                f"unsupported inference device: {requested.type!r}; use cpu or cuda"
            )
        return requested

    @classmethod
    def _validate_checkpoint(cls, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("model_name") != cls._EXPECTED_MODEL_NAME:
            raise BearingInferenceError(
                "unsupported checkpoint model_name: "
                f"{payload.get('model_name')!r}; expected {cls._EXPECTED_MODEL_NAME!r}"
            )
        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise BearingInferenceError("checkpoint has no model_state_dict")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise BearingInferenceError("checkpoint has no embedded config")
        dataset = config.get("dataset")
        if not isinstance(dataset, dict):
            raise BearingInferenceError("checkpoint config has no dataset section")
        required = ("window_size", "step", "normalization", "num_classes", "label_map")
        missing = [key for key in required if key not in dataset]
        if missing:
            raise BearingInferenceError(
                f"checkpoint config missing dataset fields: {missing}"
            )
        try:
            window_size = int(dataset["window_size"])
            step = int(dataset["step"])
            num_classes = int(dataset["num_classes"])
        except (TypeError, ValueError) as exc:
            raise BearingInferenceError(
                "checkpoint dataset window_size, step, and num_classes must be integers"
            ) from exc
        if window_size <= 0 or step <= 0 or num_classes < 2:
            raise BearingInferenceError(
                "checkpoint dataset has invalid window_size, step, or num_classes"
            )
        if dataset["normalization"] not in cls._VALID_NORMALIZATIONS:
            raise BearingInferenceError(
                f"unsupported checkpoint normalization: {dataset['normalization']!r}"
            )
        if not isinstance(dataset["label_map"], (str, Path)):
            raise BearingInferenceError("checkpoint dataset.label_map must be a path")
        return config

    def _label_map_candidates(self, override: str | Path | None) -> Iterable[Path]:
        if override is not None:
            yield Path(override).resolve()
            return

        configured = Path(str(self.config["dataset"]["label_map"]))
        if configured.is_absolute():
            yield configured.resolve()
            return

        config_path_value = self.config.get("_config_path")
        if config_path_value:
            config_path = Path(str(config_path_value))
            if config_path.is_absolute():
                # training/configs/<file>.json -> project root is parents[2]
                yield (config_path.parents[2] / configured).resolve()
        # Useful when a checkpoint is copied together with a project tree.
        for parent in (self.checkpoint_path.parent, *self.checkpoint_path.parents):
            yield (parent / configured).resolve()

    def _read_label_map(self, override: str | Path | None) -> dict[int, str]:
        candidates = list(dict.fromkeys(self._label_map_candidates(override)))
        for path in candidates:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                labels = payload["labels"]
                mapping = {int(index): str(name) for index, name in labels.items()}
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise BearingInferenceError(f"invalid label map {path}: {exc}") from exc
            if not mapping:
                raise BearingInferenceError(f"label map is empty: {path}")
            self.label_map_path = path
            return mapping
        searched = ", ".join(str(path) for path in candidates)
        raise BearingInferenceError(
            "label map does not exist; searched: " + (searched or "<no candidate path>")
        )

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        if self.normalization == "none":
            return signal.astype(np.float32, copy=True)
        mean = float(signal.mean())
        std = float(signal.std())
        if std == 0.0:
            raise BearingInferenceError("cannot z-score a constant signal")
        return ((signal - mean) / std).astype(np.float32, copy=False)

    @staticmethod
    def _validate_input(signal: object) -> np.ndarray:
        array = np.asarray(signal)
        if array.ndim != 1:
            raise BearingInferenceError(
                f"bearing inference expects a one-dimensional signal, got shape={array.shape}"
            )
        if np.iscomplexobj(array):
            raise BearingInferenceError("bearing inference expects a real-valued signal")
        if not np.issubdtype(array.dtype, np.number):
            raise BearingInferenceError(
                f"bearing inference expects numeric samples, got dtype={array.dtype}"
            )
        try:
            converted = np.asarray(array, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise BearingInferenceError("signal cannot be converted to float32") from exc
        if not np.isfinite(converted).all():
            raise BearingInferenceError("signal contains NaN or infinite values")
        return converted

    @staticmethod
    def _validate_quality_input(signal: object) -> np.ndarray:
        """Validate structure but leave NaN/Inf for per-window rejection."""
        array = np.asarray(signal)
        if array.ndim != 1:
            raise BearingInferenceError(
                f"bearing inference expects a one-dimensional signal, got shape={array.shape}"
            )
        if np.iscomplexobj(array):
            raise BearingInferenceError("bearing inference expects a real-valued signal")
        if not np.issubdtype(array.dtype, np.number):
            raise BearingInferenceError(
                f"bearing inference expects numeric samples, got dtype={array.dtype}"
            )
        try:
            return np.asarray(array, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise BearingInferenceError("signal cannot be converted to float32") from exc

    def _raw_windows(self, signal: np.ndarray) -> list[np.ndarray]:
        """Slice complete windows without applying finite-value validation."""
        if signal.size < self.window_size:
            raise InsufficientSignalError(
                f"signal has {signal.size} samples, but one window requires "
                f"{self.window_size} samples"
            )
        return [
            signal[start : start + self.window_size]
            for start in range(0, signal.size - self.window_size + 1, self.step)
        ]

    def _predict_windows(
        self,
        windows: np.ndarray,
        *,
        window_indices: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Run one model batch and preserve caller-supplied source indices."""
        if windows.ndim != 2 or windows.shape[1] != self.window_size:
            raise BearingInferenceError(
                "preprocessed windows must have shape "
                f"[N, {self.window_size}], got {windows.shape}"
            )
        if windows.shape[0] == 0:
            return []
        if window_indices is None:
            window_indices = list(range(windows.shape[0]))
        if len(window_indices) != windows.shape[0]:
            raise BearingInferenceError("window index count does not match window batch")

        inputs = torch.from_numpy(
            np.array(windows, dtype=np.float32, copy=True)
        ).unsqueeze(1)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(inputs), dim=1)
            confidences, predictions = probabilities.max(dim=1)
        return [
            WindowPrediction(
                window_index=int(source_index),
                predicted_label=self.label_map[int(predicted_class)],
                predicted_class=int(predicted_class),
                confidence=float(confidence),
                probabilities=tuple(float(value) for value in row),
            ).to_dict()
            for source_index, predicted_class, confidence, row in zip(
                window_indices,
                predictions.cpu().tolist(),
                confidences.cpu().tolist(),
                probabilities.cpu().tolist(),
            )
        ]

    def predict_signal(self, signal: object) -> list[dict[str, Any]]:
        """Predict every complete, non-overlapping window in one signal."""
        validated = self._validate_input(signal)
        if validated.size < self.window_size:
            raise InsufficientSignalError(
                f"signal has {validated.size} samples, but one window requires "
                f"{self.window_size} samples"
            )
        if self.normalization == "window_zscore":
            raw_windows = slice_signal(
                validated,
                window_size=self.window_size,
                step=self.step,
            )
            try:
                windows = normalize_windows(
                    raw_windows,
                    normalization="window_zscore",
                )
            except CWRUDataError as exc:
                raise BearingInferenceError(str(exc)) from exc
        else:
            normalized = self._normalize(validated)
            windows = slice_signal(
                normalized,
                window_size=self.window_size,
                step=self.step,
            )
        if windows.shape[0] == 0:
            raise InsufficientSignalError(
                f"signal produced zero complete windows with window_size={self.window_size} "
                f"and step={self.step}"
            )

        inputs = torch.from_numpy(np.array(windows, dtype=np.float32, copy=True)).unsqueeze(1)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
        predictions_cpu = predictions.cpu().tolist()
        confidence_cpu = confidences.cpu().tolist()
        probabilities_cpu = probabilities.cpu().tolist()
        return [
            WindowPrediction(
                window_index=index,
                predicted_label=self.label_map[int(predicted_class)],
                predicted_class=int(predicted_class),
                confidence=float(confidence),
                probabilities=tuple(float(value) for value in row),
            ).to_dict()
            for index, (predicted_class, confidence, row) in enumerate(
                zip(predictions_cpu, confidence_cpu, probabilities_cpu)
            )
        ]

    def predict_file(
        self,
        path: str | Path,
        *,
        sensor_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load a CWRU MAT file and return its window-level predictions."""
        try:
            record = load_cwru_signal(
                path,
                sensor_key=sensor_key,
                normalization="none",
            )
        except CWRUDataError as exc:
            raise BearingInferenceError(str(exc)) from exc
        return self.predict_signal(record.signal)

    def predict_signal_with_quality(
        self,
        signal: object,
        *,
        thresholds: BearingQualityThresholds | None = None,
    ) -> list[dict[str, Any]]:
        """Quality-check raw windows, then infer only on accepted windows.

        Quality is assessed before checkpoint preprocessing. A rejected window
        has ``prediction=None`` so callers cannot present a fault class for
        unusable input.
        """
        validated = self._validate_quality_input(signal)
        raw_windows = self._raw_windows(validated)
        quality_results = [
            assess_bearing_window(window, thresholds=thresholds)
            for window in raw_windows
        ]
        accepted_indices = [
            index for index, result in enumerate(quality_results) if result.accepted
        ]

        predictions_by_index: dict[int, dict[str, Any]] = {}
        if accepted_indices:
            accepted_raw = np.stack(
                [raw_windows[index] for index in accepted_indices], axis=0
            ).astype(np.float32, copy=False)
            if self.normalization == "window_zscore":
                try:
                    processed = normalize_windows(
                        accepted_raw, normalization="window_zscore"
                    )
                except CWRUDataError as exc:
                    raise BearingInferenceError(str(exc)) from exc
            elif self.normalization == "zscore":
                finite_signal = validated[np.isfinite(validated)]
                mean = float(finite_signal.mean())
                std = float(finite_signal.std())
                if std == 0.0:
                    raise BearingInferenceError("cannot z-score a constant signal")
                processed = ((accepted_raw - mean) / std).astype(
                    np.float32, copy=False
                )
            else:
                processed = accepted_raw.astype(np.float32, copy=True)

            predictions_by_index = {
                prediction["window_index"]: prediction
                for prediction in self._predict_windows(
                    processed, window_indices=accepted_indices
                )
            }

        return [
            QualityAwareWindowPrediction(
                window_index=index,
                accepted=result.accepted,
                reason_code=result.reason_code,
                message=result.message,
                quality_score=float(result.quality_score),
                quality_features={
                    str(name): float(value) for name, value in result.features.items()
                },
                prediction=predictions_by_index.get(index),
            ).to_dict()
            for index, result in enumerate(quality_results)
        ]

    def predict_file_with_quality(
        self,
        path: str | Path,
        *,
        sensor_key: str | None = None,
        thresholds: BearingQualityThresholds | None = None,
    ) -> list[dict[str, Any]]:
        """Load a CWRU MAT file and apply quality-aware window inference."""
        try:
            record = load_cwru_signal(
                path, sensor_key=sensor_key, normalization="none"
            )
        except CWRUDataError as exc:
            raise BearingInferenceError(str(exc)) from exc
        return self.predict_signal_with_quality(record.signal, thresholds=thresholds)

    def predict_signal_with_quality_v2(
        self,
        signal: object,
        *,
        thresholds: BearingQualityThresholdsV2 | None = None,
    ) -> list[dict[str, Any]]:
        """Apply the three-state quality gate before frozen-model inference.

        ``accepted`` and ``caution`` windows are inferred. ``rejected`` windows
        never enter the model and therefore always return ``prediction=None``.
        Quality is measured on raw windows before checkpoint preprocessing.
        """
        validated = self._validate_quality_input(signal)
        raw_windows = self._raw_windows(validated)
        quality_results = [
            assess_bearing_window_v2(window, thresholds=thresholds)
            for window in raw_windows
        ]
        infer_indices = [
            index for index, result in enumerate(quality_results) if result.accepted
        ]

        predictions_by_index: dict[int, dict[str, Any]] = {}
        if infer_indices:
            infer_raw = np.stack(
                [raw_windows[index] for index in infer_indices], axis=0
            ).astype(np.float32, copy=False)
            if self.normalization == "window_zscore":
                try:
                    processed = normalize_windows(
                        infer_raw, normalization="window_zscore"
                    )
                except CWRUDataError as exc:
                    raise BearingInferenceError(str(exc)) from exc
            elif self.normalization == "zscore":
                finite_signal = validated[np.isfinite(validated)]
                mean = float(finite_signal.mean())
                std = float(finite_signal.std())
                if std == 0.0:
                    raise BearingInferenceError("cannot z-score a constant signal")
                processed = ((infer_raw - mean) / std).astype(
                    np.float32, copy=False
                )
            else:
                processed = infer_raw.astype(np.float32, copy=True)

            predictions_by_index = {
                prediction["window_index"]: prediction
                for prediction in self._predict_windows(
                    processed, window_indices=infer_indices
                )
            }

        return [
            QualityAwareWindowPredictionV2(
                window_index=index,
                status=result.status,
                accepted=result.accepted,
                reason_code=result.reason_code,
                message=result.message,
                quality_score=float(result.quality_score),
                quality_features={
                    str(name): float(value) for name, value in result.features.items()
                },
                prediction=predictions_by_index.get(index),
            ).to_dict()
            for index, result in enumerate(quality_results)
        ]

    def predict_file_with_quality_v2(
        self,
        path: str | Path,
        *,
        sensor_key: str | None = None,
        thresholds: BearingQualityThresholdsV2 | None = None,
    ) -> list[dict[str, Any]]:
        """Load a CWRU MAT file and apply the three-state quality gate."""
        try:
            record = load_cwru_signal(
                path, sensor_key=sensor_key, normalization="none"
            )
        except CWRUDataError as exc:
            raise BearingInferenceError(str(exc)) from exc
        return self.predict_signal_with_quality_v2(
            record.signal, thresholds=thresholds
        )

    def _predict_preprocessed_signal(self, signal: object) -> list[dict[str, Any]]:
        validated = self._validate_input(signal)
        if validated.size < self.window_size:
            raise InsufficientSignalError(
                f"signal has {validated.size} samples, but one window requires "
                f"{self.window_size} samples"
            )
        windows = slice_signal(validated, window_size=self.window_size, step=self.step)
        return self._predict_windows(windows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/bearing/bearing_cnn_baseline_best.pt"),
    )
    parser.add_argument("--input", type=Path, required=True, help="CWRU .mat file")
    parser.add_argument("--sensor-key", default=None, help="explicit *_DE_time key")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional output path; omitted means print JSON to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    predictor = BearingPredictor(args.checkpoint, device=args.device)
    predictions = predictor.predict_file(args.input, sensor_key=args.sensor_key)
    result = {
        "checkpoint_path": str(predictor.checkpoint_path),
        "input_path": str(Path(args.input).resolve()),
        "sensor_key": args.sensor_key,
        "device": str(predictor.device),
        "window_size": predictor.window_size,
        "step": predictor.step,
        "normalization": predictor.normalization,
        "label_map": predictor.label_map,
        "window_count": len(predictions),
        "predictions": predictions,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output_json is None:
        print(rendered, end="")
    else:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
