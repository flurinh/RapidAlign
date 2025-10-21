"""Lightweight alignment and loss utilities for RapidAlign."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import os
import torch

try:  # pragma: no cover - extension optional during bootstrap
    from . import _cuda
except ImportError:  # pragma: no cover
    _cuda = None

USE_CUDA_BACKEND = os.getenv("RAPIDALIGN_USE_CUDA_BACKEND", "0") == "1" and _cuda is not None

Tensor = torch.Tensor

_EPS = 1e-8


def _normalize_weights(weights: Optional[Tensor], n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if weights is None:
        return torch.full((n,), 1.0 / max(n, 1), device=device, dtype=dtype)
    weights = weights.to(device=device, dtype=dtype)
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return weights / total


def _split_by_batch(points: Tensor, batch: Optional[Tensor], weights: Optional[Tensor] = None) -> List[Tuple[Tensor, Tensor]]:
    if batch is None:
        w = _normalize_weights(weights, points.shape[0], points.device, points.dtype)
        return [(points, w)]

    if points.ndim != 2 or batch.ndim != 1:
        raise ValueError("points must be (N,3) and batch must be (N,)")
    if points.shape[0] != batch.shape[0]:
        raise ValueError("points and batch must have the same length")

    if weights is not None and weights.shape[0] != points.shape[0]:
        raise ValueError("weights must match points length")

    groups: List[Tuple[Tensor, Tensor]] = []
    for batch_id in torch.unique(batch, sorted=True):
        mask = batch == batch_id
        pts = points[mask]
        wts = weights[mask] if weights is not None else None
        norm_wts = _normalize_weights(wts, pts.shape[0], points.device, points.dtype)
        groups.append((pts, norm_wts))
    return groups


@dataclass
class AlignmentResult:
    aligned: Tensor
    rotations: Tensor
    translations: Tensor


def procrustes_align(
    src: Tensor,
    tgt: Tensor,
    src_batch: Optional[Tensor] = None,
    tgt_batch: Optional[Tensor] = None,
) -> AlignmentResult:
    """Rigid Procrustes alignment with optional batching.

    Args:
        src: Source points to be aligned, shape (N, 3).
        tgt: Target points, same shape as src.
        src_batch: Optional batch assignment for src points.
        tgt_batch: Optional batch assignment for tgt points.

    Returns:
        AlignmentResult with aligned source points (same shape as src),
        rotation matrices, and translation vectors per graph.
    """
    if src.ndim != 2 or tgt.ndim != 2 or src.size(-1) != 3 or tgt.size(-1) != 3:
        raise ValueError("src and tgt must be shaped (N, 3)")

    if USE_CUDA_BACKEND and src.is_cuda and tgt.is_cuda:
        aligned, rotations, translations = _cuda.procrustes_align_cuda(
            src, tgt, src_batch, tgt_batch
        )
        return AlignmentResult(aligned, rotations, translations)

    src_groups = _split_by_batch(src, src_batch)
    tgt_groups = _split_by_batch(tgt, tgt_batch)
    if len(src_groups) != len(tgt_groups):
        raise ValueError("src and tgt batch counts must match")

    aligned_parts: List[Tensor] = []
    rotations: List[Tensor] = []
    translations: List[Tensor] = []
    src_offsets = []

    if src_batch is None:
        src_offsets = [torch.arange(src.shape[0], device=src.device)]
    else:
        src_offsets = [torch.where(src_batch == bid)[0] for bid in torch.unique(src_batch, sorted=True)]

    for (src_pts, _), (tgt_pts, _) in zip(src_groups, tgt_groups):
        if src_pts.shape[0] != tgt_pts.shape[0]:
            raise ValueError("Procrustes requires equal point counts per graph")

        src_mean = src_pts.mean(dim=0, keepdim=True)
        tgt_mean = tgt_pts.mean(dim=0, keepdim=True)

        src_centered = src_pts - src_mean
        tgt_centered = tgt_pts - tgt_mean

        cov = src_centered.transpose(0, 1) @ tgt_centered
        U, _, Vh = torch.linalg.svd(cov)
        R = Vh.transpose(0, 1) @ U.transpose(0, 1)
        if torch.det(R) < 0:
            Vh[-1, :] *= -1
            R = Vh.transpose(0, 1) @ U.transpose(0, 1)
        t = tgt_mean.squeeze(0) - (src_mean.squeeze(0) @ R)
        aligned = src_pts @ R + t

        aligned_parts.append(aligned)
        rotations.append(R)
        translations.append(t)

    aligned_full = torch.zeros_like(src)
    for idxs, aligned in zip(src_offsets, aligned_parts):
        aligned_full[idxs] = aligned

    return AlignmentResult(
        aligned=aligned_full,
        rotations=torch.stack(rotations),
        translations=torch.stack(translations),
    )


def pairwise_distance_loss(
    src: Tensor,
    tgt: Tensor,
    src_batch: Optional[Tensor] = None,
    tgt_batch: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Baseline loss that compares pairwise distance spectra.

    The loss is invariant to rigid transforms and supports variable point counts.
    Distances are sorted and padded before computing an MSE.
    """
    src_groups = _split_by_batch(src, src_batch)
    tgt_groups = _split_by_batch(tgt, tgt_batch)
    if len(src_groups) != len(tgt_groups):
        raise ValueError("src and tgt batch counts must match")

    losses: List[Tensor] = []
    for (src_pts, _), (tgt_pts, _) in zip(src_groups, tgt_groups):
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
    raise ValueError("Unsupported reduction: {reduction}")


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


def se3_kernel_loss(
    src: Tensor,
    tgt: Tensor,
    src_batch: Optional[Tensor] = None,
    tgt_batch: Optional[Tensor] = None,
    sigma: float = 0.3,
    iterations: int = 3,
    detach_pose: bool = False,
) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
    """Differentiable SE(3)-aware kernel correlation loss.

    Args:
        src: Source points (N, 3).
        tgt: Target points (M, 3).
        src_batch: Optional batch indices for src.
        tgt_batch: Optional batch indices for tgt.
        sigma: Gaussian bandwidth for correspondence weights.
        iterations: Number of MM refinements per batch element.
        detach_pose: If True, does not backprop through pose estimation.

    Returns:
        loss: Scalar tensor
        rotations: list of rotation matrices per batch
        translations: list of translation vectors per batch
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if USE_CUDA_BACKEND and src.is_cuda and tgt.is_cuda:
        loss, rotations, translations = _cuda.se3_kernel_loss_cuda(
            src, tgt, src_batch, tgt_batch, float(sigma), int(iterations)
        )
        rot_list = [rotations[i] for i in range(rotations.shape[0])]
        trn_list = [translations[i] for i in range(translations.shape[0])]
        return loss.squeeze(), rot_list, trn_list

    src_groups = _split_by_batch(src, src_batch)
    tgt_groups = _split_by_batch(tgt, tgt_batch)
    if len(src_groups) != len(tgt_groups):
        raise ValueError("src and tgt batch counts must match")

    losses: List[Tensor] = []
    rotations: List[Tensor] = []
    translations: List[Tensor] = []

    for (src_pts, src_w), (tgt_pts, tgt_w) in zip(src_groups, tgt_groups):
        R = torch.eye(3, device=src_pts.device, dtype=src_pts.dtype)
        t = torch.zeros(3, device=src_pts.device, dtype=src_pts.dtype)
        loss_val, R, t = _kernel_mm(src_pts, tgt_pts, src_w, tgt_w, R, t, sigma, iterations)
        losses.append(loss_val)
        rotations.append(R.detach() if detach_pose else R)
        translations.append(t.detach() if detach_pose else t)

    return torch.stack(losses).mean(), rotations, translations


def _kernel_mm(
    src: Tensor,
    tgt: Tensor,
    src_w: Tensor,
    tgt_w: Tensor,
    R_init: Tensor,
    t_init: Tensor,
    sigma: float,
    iterations: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    R = R_init
    t = t_init
    kappa = None
    for _ in range(max(iterations, 1)):
        src_transformed = src @ R + t
        diff = tgt.unsqueeze(1) - src_transformed.unsqueeze(0)
        dist2 = diff.pow(2).sum(-1)
        kernel = torch.exp(-dist2 / (2.0 * sigma ** 2)).clamp(min=_EPS)
        hat_w = (tgt_w.unsqueeze(1) * src_w.unsqueeze(0)) * kernel
        kappa = hat_w.sum()
        if kappa.item() <= _EPS:
            break
        w = hat_w / kappa

        alpha = w.sum(dim=1)
        beta = w.sum(dim=0)
        x_bar = (alpha.unsqueeze(1) * tgt).sum(dim=0)
        y_bar = (beta.unsqueeze(1) * src).sum(dim=0)

        tgt_centered = tgt - x_bar
        src_centered = src - y_bar

        S = src_centered.transpose(0, 1) @ (w.transpose(0, 1) @ tgt_centered)
        U, _, Vh = torch.linalg.svd(S)
        R_new = Vh.transpose(0, 1) @ U.transpose(0, 1)
        if torch.det(R_new) < 0:
            Vh[-1, :] *= -1
            R_new = Vh.transpose(0, 1) @ U.transpose(0, 1)
        t_new = x_bar - (y_bar @ R_new)
        R, t = R_new, t_new

    loss = -torch.log(kappa + _EPS)
    return loss, R, t

__all__ = [
    "AlignmentResult",
    "procrustes_align",
    "pairwise_distance_loss",
    "se3_kernel_loss",
]
