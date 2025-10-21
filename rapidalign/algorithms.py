"""Baseline geometric utilities retained for compatibility.

Only the pairwise-distance spectral loss is provided now that rigid
alignment kernels have been removed from the CUDA extension. The helper
accepts optional per-graph batching so existing callers do not need to
change their invocation patterns.
"""
from __future__ import annotations

from typing import List, Optional

import torch

Tensor = torch.Tensor

__all__ = ["pairwise_distance_loss"]


def _split_by_batch(points: Tensor, batch: Optional[Tensor]) -> List[Tensor]:
    if batch is None:
        if points.ndim != 2:
            raise ValueError("points must have shape (N, d)")
        return [points]

    if points.ndim != 2 or batch.ndim != 1:
        raise ValueError("points must be (N, d) and batch must be (N,)")
    if points.shape[0] != batch.shape[0]:
        raise ValueError("points and batch must have the same length")

    groups: List[Tensor] = []
    for batch_id in torch.unique(batch, sorted=True):
        mask = batch == batch_id
        groups.append(points[mask])
    return groups


def _pairwise_distance_vector(points: Tensor) -> Tensor:
    if points.shape[0] < 2:
        return torch.zeros(0, device=points.device, dtype=points.dtype)
    diff = points.unsqueeze(0) - points.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)
    tri_idx = torch.triu_indices(points.shape[0], points.shape[0], offset=1)
    vec = dist[tri_idx[0], tri_idx[1]]
    return torch.sort(vec).values


def _pad_vector(vec: Tensor, length: int) -> Tensor:
    if vec.shape[0] == length:
        return vec
    pad = length - vec.shape[0]
    return torch.nn.functional.pad(vec, (0, pad))


def pairwise_distance_loss(
    src: Tensor,
    tgt: Tensor,
    src_batch: Optional[Tensor] = None,
    tgt_batch: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Baseline loss that compares pairwise distance spectra.

    The loss is invariant to rigid transforms and supports variable point
    counts. Distances are sorted and padded before computing an MSE on each
    graph; the losses are then aggregated using the requested reduction.
    """
    src_groups = _split_by_batch(src, src_batch)
    tgt_groups = _split_by_batch(tgt, tgt_batch)
    if len(src_groups) != len(tgt_groups):
        raise ValueError("src and tgt batch counts must match")

    losses: List[Tensor] = []
    for src_pts, tgt_pts in zip(src_groups, tgt_groups):
        if src_pts.size(-1) != tgt_pts.size(-1):
            raise ValueError("point dimensionality must match per graph")
        src_vec = _pairwise_distance_vector(src_pts)
        tgt_vec = _pairwise_distance_vector(tgt_pts)
        max_len = max(src_vec.shape[0], tgt_vec.shape[0])
        src_padded = _pad_vector(src_vec, max_len)
        tgt_padded = _pad_vector(tgt_vec, max_len)
        losses.append(torch.nn.functional.mse_loss(src_padded, tgt_padded, reduction="mean"))

    stacked = torch.stack(losses)
    if reduction == "mean":
        return stacked.mean()
    if reduction == "sum":
        return stacked.sum()
    if reduction == "none":
        return stacked
    raise ValueError(f"Unsupported reduction: {reduction}")
