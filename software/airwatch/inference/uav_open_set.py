"""Open-set scoring and known-only calibration for frozen UAV models.

Every score in this module follows one convention: larger values mean that a
sample looks more like the known training classes.  Thresholds are calibrated
from known validation recordings only; unknown evaluation data must never be
passed to :func:`calibrate_known_acceptance`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class UAVOpenSetError(ValueError):
    """Raised when open-set inputs or calibration settings are invalid."""


def _finite_matrix(values: object, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise UAVOpenSetError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise UAVOpenSetError(f"{name} contains non-finite values")
    return matrix


def _finite_scores(values: object, *, name: str = "scores") -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1:
        raise UAVOpenSetError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(scores).all():
        raise UAVOpenSetError(f"{name} contains non-finite values")
    return scores


def maximum_softmax_knownness(logits: object) -> np.ndarray:
    """Return maximum softmax probability, with higher values meaning known."""
    matrix = _finite_matrix(logits, name="logits")
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.max(axis=1)


def energy_knownness(logits: object, *, temperature: float = 1.0) -> np.ndarray:
    """Return negative energy (T * logsumexp(logits / T)) as knownness."""
    matrix = _finite_matrix(logits, name="logits")
    if not np.isfinite(temperature) or temperature <= 0:
        raise UAVOpenSetError("temperature must be a positive finite number")
    scaled = matrix / float(temperature)
    maxima = np.max(scaled, axis=1, keepdims=True)
    logsumexp = maxima[:, 0] + np.log(np.exp(scaled - maxima).sum(axis=1))
    return float(temperature) * logsumexp


@dataclass(frozen=True)
class ClassPrototypes:
    """Class-centroid features fitted exclusively from the training split."""

    labels: tuple[int, ...]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        vectors = _finite_matrix(self.vectors, name="prototype vectors").copy()
        if len(self.labels) != vectors.shape[0] or len(set(self.labels)) != len(self.labels):
            raise UAVOpenSetError("prototype labels must uniquely match prototype rows")
        vectors.setflags(write=False)
        object.__setattr__(self, "vectors", vectors)


def fit_class_prototypes(features: object, targets: object) -> ClassPrototypes:
    matrix = _finite_matrix(features, name="features")
    labels = np.asarray(targets)
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise UAVOpenSetError("targets must contain one label per feature row")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.isfinite(labels.astype(np.float64)).all() or np.any(labels != np.floor(labels)):
            raise UAVOpenSetError("targets must contain finite integer labels")
    labels = labels.astype(np.int64)
    unique = tuple(int(value) for value in np.unique(labels))
    vectors = np.vstack([matrix[labels == label].mean(axis=0) for label in unique])
    return ClassPrototypes(unique, vectors)


def prototype_knownness(
    features: object,
    prototypes: ClassPrototypes,
    *,
    metric: str = "cosine",
) -> np.ndarray:
    """Score features by their nearest training prototype."""
    matrix = _finite_matrix(features, name="features")
    if matrix.shape[1] != prototypes.vectors.shape[1]:
        raise UAVOpenSetError("feature and prototype dimensions differ")
    if metric == "cosine":
        feature_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        prototype_norms = np.linalg.norm(prototypes.vectors, axis=1, keepdims=True)
        if np.any(feature_norms == 0) or np.any(prototype_norms == 0):
            raise UAVOpenSetError("cosine prototype scoring does not accept zero vectors")
        similarities = (matrix / feature_norms) @ (prototypes.vectors / prototype_norms).T
        return similarities.max(axis=1)
    if metric == "squared_euclidean":
        distances = ((matrix[:, None, :] - prototypes.vectors[None, :, :]) ** 2).sum(axis=2)
        return -distances.min(axis=1)
    raise UAVOpenSetError(f"unsupported prototype metric: {metric!r}")


def aggregate_recording_scores(
    scores: object,
    recording_ids: Sequence[str] | Iterable[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Mean window scores per recording, preserving first-seen recording order."""
    vector = _finite_scores(scores)
    identifiers = tuple(str(value) for value in recording_ids)
    if len(identifiers) != vector.size or any(not value for value in identifiers):
        raise UAVOpenSetError("recording_ids must contain one non-empty id per score")
    order: list[str] = []
    grouped: dict[str, list[float]] = {}
    for identifier, score in zip(identifiers, vector):
        if identifier not in grouped:
            order.append(identifier)
            grouped[identifier] = []
        grouped[identifier].append(float(score))
    aggregated = np.asarray([np.mean(grouped[item]) for item in order], dtype=np.float64)
    return tuple(order), aggregated


def aggregate_recording_probabilities(
    probabilities: object,
    recording_ids: Sequence[str] | Iterable[str],
    targets: object,
    *,
    expected_windows_per_recording: int | None = None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Mean class probabilities per recording and enforce one target per group."""
    matrix = _finite_matrix(probabilities, name="probabilities")
    if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise UAVOpenSetError("probability rows must be non-negative and sum to one")
    identifiers = tuple(str(value) for value in recording_ids)
    labels = np.asarray(targets)
    if len(identifiers) != matrix.shape[0] or any(not value for value in identifiers):
        raise UAVOpenSetError("recording_ids must contain one non-empty id per row")
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise UAVOpenSetError("targets must contain one label per probability row")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.isfinite(labels.astype(np.float64)).all() or np.any(labels != np.floor(labels)):
            raise UAVOpenSetError("targets must contain finite integer labels")
    labels = labels.astype(np.int64)
    if expected_windows_per_recording is not None and expected_windows_per_recording < 1:
        raise UAVOpenSetError("expected_windows_per_recording must be positive")

    order: list[str] = []
    grouped_probabilities: dict[str, list[np.ndarray]] = {}
    grouped_targets: dict[str, set[int]] = {}
    for identifier, probability, label in zip(identifiers, matrix, labels):
        if identifier not in grouped_probabilities:
            order.append(identifier)
            grouped_probabilities[identifier] = []
            grouped_targets[identifier] = set()
        grouped_probabilities[identifier].append(probability)
        grouped_targets[identifier].add(int(label))
    if any(len(values) != 1 for values in grouped_targets.values()):
        raise UAVOpenSetError("one recording contains multiple target labels")
    if expected_windows_per_recording is not None and any(
        len(values) != expected_windows_per_recording
        for values in grouped_probabilities.values()
    ):
        raise UAVOpenSetError("recording window count differs from frozen protocol")
    means = np.vstack([
        np.mean(grouped_probabilities[identifier], axis=0) for identifier in order
    ])
    recording_targets = np.asarray(
        [next(iter(grouped_targets[identifier])) for identifier in order], dtype=np.int64
    )
    return tuple(order), means, recording_targets


def aggregate_single_recording_probabilities(
    probabilities: object,
    *,
    expected_window_count: int | None = None,
) -> np.ndarray:
    """Mean window probabilities for one unlabeled runtime recording.

    Evaluation uses :func:`aggregate_recording_probabilities` because targets
    are available there.  Desktop inference has no target label, so this
    companion keeps the same probability validation and averaging rule without
    inventing a target merely to call the evaluation API.
    """
    matrix = _finite_matrix(probabilities, name="probabilities")
    if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise UAVOpenSetError("probability rows must be non-negative and sum to one")
    if expected_window_count is not None:
        if expected_window_count < 1:
            raise UAVOpenSetError("expected_window_count must be positive")
        if matrix.shape[0] != expected_window_count:
            raise UAVOpenSetError("recording window count differs from frozen protocol")
    return matrix.mean(axis=0, dtype=np.float64)


@dataclass(frozen=True)
class OpenSetCalibration:
    """An immutable validation-derived known/unknown decision threshold."""

    method: str
    threshold: float
    target_known_acceptance: float
    observed_known_acceptance: float
    recording_count: int
    score_direction: str = "higher_is_more_known"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "target_known_acceptance": self.target_known_acceptance,
            "observed_known_acceptance": self.observed_known_acceptance,
            "recording_count": self.recording_count,
            "score_direction": self.score_direction,
        }


def calibrate_known_acceptance(
    known_validation_scores: object,
    *,
    method: str,
    target_known_acceptance: float = 0.95,
) -> OpenSetCalibration:
    """Choose a conservative threshold using known validation scores only."""
    scores = _finite_scores(known_validation_scores, name="known validation scores")
    if not method.strip():
        raise UAVOpenSetError("method must be non-empty")
    if not np.isfinite(target_known_acceptance) or not 0 < target_known_acceptance <= 1:
        raise UAVOpenSetError("target_known_acceptance must be in (0, 1]")
    ordered = np.sort(scores)
    maximum_rejections = int(np.floor((1.0 - float(target_known_acceptance)) * scores.size))
    threshold = float(ordered[min(maximum_rejections, scores.size - 1)])
    observed = float(np.mean(scores >= threshold))
    return OpenSetCalibration(
        method=method,
        threshold=threshold,
        target_known_acceptance=float(target_known_acceptance),
        observed_known_acceptance=observed,
        recording_count=int(scores.size),
    )


def decide_known(scores: object, calibration: OpenSetCalibration) -> np.ndarray:
    """Apply a frozen calibration without re-estimating its threshold."""
    vector = _finite_scores(scores)
    return vector >= calibration.threshold


__all__ = [
    "ClassPrototypes",
    "OpenSetCalibration",
    "UAVOpenSetError",
    "aggregate_recording_scores",
    "aggregate_recording_probabilities",
    "aggregate_single_recording_probabilities",
    "calibrate_known_acceptance",
    "decide_known",
    "energy_knownness",
    "fit_class_prototypes",
    "maximum_softmax_knownness",
    "prototype_knownness",
]
