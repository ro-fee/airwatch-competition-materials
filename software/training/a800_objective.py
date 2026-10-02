"""Stateless V4-distribution view generation and the A800 V2 paired objective.

Augmentation consumes only caller-provided training tensors. The training data
access gate belongs to the runner; this module never opens a dataset. Branch
probabilities and channel profiles reuse V4, while the paired draw identities
are explicitly versioned rather than claiming to reproduce old V4 training.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from airwatch.analysis.iq_perturbations import apply_iq_perturbation


PAIRED_AUGMENTATION_ID = "a800-v2-paired-v4-mixture-sha256-v1"
_AWGN_SAMPLING_LEVELS = (-5, -5, -5, -5, -5, 0, 5, 10, 15, 20)
_MULTIPATH_PROFILES = ("mild", "severe")


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def paired_view_recipe(
    *, run_seed: int, logical_epoch: int, draw_position: int,
    window_id: str, view_id: int,
) -> dict[str, int | str]:
    """Return an auditable perturbation recipe without consuming any RNG state.

    Epoch and draw position are zero based. The UTF-8 JSON array containing
    protocol, run seed, epoch, draw position, window ID, and view ID is hashed
    with SHA-256. Its first eight big-endian bytes choose a 50/25/25 branch and
    weighted level; the next eight bytes, reduced modulo 2**63-1, seed noise or
    channel phases. Both members of a pair are independent draws, so two clean
    views are a valid outcome. Repeated anchors have different draw positions.
    """
    run_seed = _nonnegative_integer(run_seed, "run_seed")
    logical_epoch = _nonnegative_integer(logical_epoch, "logical_epoch")
    draw_position = _nonnegative_integer(draw_position, "draw_position")
    view_id = _nonnegative_integer(view_id, "view_id")
    if view_id not in (0, 1):
        raise ValueError("view_id must be 0 or 1")
    if not isinstance(window_id, str) or not window_id.strip():
        raise ValueError("window_id must be a nonempty string")
    identity = [PAIRED_AUGMENTATION_ID, run_seed, logical_epoch,
                draw_position, window_id, view_id]
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).digest()
    choice = int.from_bytes(digest[:8], "big")
    seed = int.from_bytes(digest[8:16], "big") % (2**63 - 1)
    branch = choice % 4
    if branch < 2:
        return {"kind": "clean", "seed": seed}
    if branch == 2:
        return {"kind": "awgn", "seed": seed,
                "snr_db": _AWGN_SAMPLING_LEVELS[(choice // 4) % 10]}
    return {"kind": "multipath", "seed": seed,
            "profile": _MULTIPATH_PROFILES[(choice // 4) % 2]}


def make_paired_views(
    batch: torch.Tensor, window_ids: Sequence[str], *, run_seed: int,
    logical_epoch: int, draw_positions: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce CPU float32 views from [N, 2, L] training anchors, without mutation.

    All non-clean effects use the frozen IQ implementation and its mandatory
    post-effect DC removal / complex RMS normalization. Clean inputs are copied
    exactly. Batch partition, worker count, and global RNG state cannot alter a
    view's identity. Call before transfer to the training device.
    """
    if (not isinstance(batch, torch.Tensor) or batch.ndim != 3
            or batch.shape[0] < 1 or batch.shape[1] != 2 or batch.shape[2] < 2):
        raise ValueError("paired augmentation expects nonempty [N, 2, L>=2] IQ tensors")
    if batch.device.type != "cpu" or batch.dtype != torch.float32:
        raise ValueError("paired augmentation requires CPU float32 input")
    if not torch.isfinite(batch).all().item():
        raise ValueError("paired augmentation requires finite input")
    if isinstance(window_ids, (str, bytes)):
        raise ValueError("window_ids must be a sequence with one ID per anchor")
    ids = list(window_ids)
    positions = list(draw_positions)
    if len(ids) != len(batch) or len(positions) != len(batch):
        raise ValueError("window_ids and draw_positions must match the anchor batch")
    positions = [_nonnegative_integer(value, "draw_position") for value in positions]
    if len(set(positions)) != len(positions):
        raise ValueError("draw_positions must uniquely identify draws within the epoch")
    views = []
    for view_id in (0, 1):
        samples = []
        for sample, window_id, position in zip(batch, ids, positions):
            recipe = paired_view_recipe(run_seed=run_seed, logical_epoch=logical_epoch,
                                        draw_position=position, window_id=window_id,
                                        view_id=view_id)
            if recipe["kind"] == "clean":
                samples.append(sample.clone())
            else:
                seed = recipe.pop("seed")
                samples.append(apply_iq_perturbation(sample, recipe, seed=seed,
                                                     channel_seed=seed))
        views.append(torch.stack(samples))
    return views[0], views[1]


def paired_objective(
    model: nn.Module, view1: torch.Tensor, view2: torch.Tensor,
    labels: torch.Tensor, js_weight: float = 0.0, *,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One joined forward, mean weighted CE, and symmetric FP32 JS loss.

    The caller derives positive class weights exclusively from train counts,
    normalizes them to mean one, and supplies the preregistered JS schedule.
    Weights affect CE only. Diagnostics are detached scalar tensors; the first
    return retains gradients into both views and the model.
    """
    if (not isinstance(view1, torch.Tensor) or not isinstance(view2, torch.Tensor)
            or view1.ndim < 2 or view1.shape != view2.shape or len(view1) < 1
            or view1.device != view2.device or view1.dtype != view2.dtype):
        raise ValueError("paired views must have equal nonempty shapes, devices, and dtypes")
    if (isinstance(js_weight, bool) or not isinstance(js_weight, Real)
            or not math.isfinite(js_weight) or js_weight < 0):
        raise ValueError("js_weight must be finite and nonnegative")
    if (not isinstance(labels, torch.Tensor) or labels.shape != (len(view1),)
            or labels.dtype != torch.long or labels.device != view1.device):
        raise ValueError("labels must be int64 [N] on the paired views' device")
    if not torch.isfinite(view1).all().item() or not torch.isfinite(view2).all().item():
        raise ValueError("paired views must contain finite values")
    logits = model(torch.cat((view1, view2), dim=0))
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or logits.shape[0] != 2 * len(view1) or logits.shape[1] < 2
            or not logits.is_floating_point() or not torch.isfinite(logits).all().item()):
        raise ValueError("model must return finite logits [2*N, num_classes>=2]")
    if labels.min().item() < 0 or labels.max().item() >= logits.shape[1]:
        raise ValueError("labels must index the model's known classes")
    weights = None
    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=torch.float32, device=logits.device)
        if (weights.shape != (logits.shape[1],) or not torch.isfinite(weights).all().item()
                or not (weights > 0).all().item()):
            raise ValueError("class_weights must have one positive finite value per class")
    # Explicitly leave autocast for all probability and logarithm operations.
    # Clamping log arguments, rather than probabilities themselves, retains the
    # exact distribution and avoids log(0) in highly confident predictions.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        first, second = logits.float().chunk(2, dim=0)
        ce = (F.cross_entropy(first, labels, weight=weights)
              + F.cross_entropy(second, labels, weight=weights)) * .5
        p = F.softmax(first, dim=1)
        q = F.softmax(second, dim=1)
        mean = (p + q) * .5
        log_mean = mean.clamp_min(1e-7).log()
        js = ((p * (p.clamp_min(1e-7).log() - log_mean)).sum(dim=1)
              + (q * (q.clamp_min(1e-7).log() - log_mean)).sum(dim=1)).mean() * .5
        total = ce + float(js_weight) * js
    if not torch.isfinite(total).item():
        raise ValueError("paired objective produced nonfinite loss")
    return total, {"ce": ce.detach(), "js": js.detach(), "total": total.detach()}


__all__ = ["PAIRED_AUGMENTATION_ID", "paired_view_recipe", "make_paired_views",
           "paired_objective"]
