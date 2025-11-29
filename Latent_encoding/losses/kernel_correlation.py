"""SE(3)-invariant, correspondence-free kernel losses (global + local)."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch_geometric.utils import to_dense_batch

Tensor = torch.Tensor

_DEFAULT_GLOBAL_CFG = dict(n_bins=32, r_max=None, gamma=None, normalize=True)
_DEFAULT_LOCAL_CFG = dict(
    num_bins=16,
    r_max=None,
    gamma=None,
    radius=None,
    k_max=None,
    tau=1.0,
    normalize=True,
)


def _center_cloud(x: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.unsqueeze(-1).float()
    counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1e-8)
    centers = (x * mask_f).sum(dim=1, keepdim=True) / counts
    return x - centers


def _prepare_dense(pos: Tensor, batch: Tensor, center: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x, mask = to_dense_batch(pos, batch)
    if center:
        x = _center_cloud(x, mask)
    dist = torch.cdist(x, x)
    mask_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
    return x, mask, dist, mask_pair


def _auto_rmax(dist_a: Tensor, mask_a: Tensor, dist_b: Tensor, mask_b: Tensor, eps: float) -> float:
    val = 0.0
    if mask_a.any():
        val = max(val, dist_a[mask_a].max().item())
    if mask_b.any():
        val = max(val, dist_b[mask_b].max().item())
    if not math.isfinite(val) or val <= eps:
        return 1.0
    return val


def _histogram_features(
    dist: Tensor,
    mask_pair: Tensor,
    bin_centers: Tensor,
    gamma: float,
) -> Tensor:
    diff = dist.unsqueeze(-1) - bin_centers.view(1, 1, 1, -1)
    feats = torch.exp(-0.5 * (diff / gamma) ** 2)
    feats = feats * mask_pair.unsqueeze(-1).float()
    feats = feats.sum(dim=(1, 2))
    return feats


def global_distance_kernel_loss_pyg(
    pos_pred: Tensor,
    pos_true: Tensor,
    batch_true: Tensor,
    *,
    batch_pred: Optional[Tensor] = None,
    center: bool = True,
    n_bins: int = 32,
    r_max: Optional[float] = None,
    gamma: Optional[float] = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Global distance histogram kernel loss."""

    if batch_pred is None:
        batch_pred = batch_true

    _, _, dist_true, mask_true_pair = _prepare_dense(pos_true, batch_true, center)
    _, _, dist_pred, mask_pred_pair = _prepare_dense(pos_pred, batch_pred, center)

    eye = torch.eye(dist_true.size(1), device=dist_true.device, dtype=torch.bool).unsqueeze(0)
    mask_true_pair = mask_true_pair & eye.logical_not()
    mask_pred_pair = mask_pred_pair & eye.logical_not()

    max_val = r_max if r_max is not None else _auto_rmax(dist_true, mask_true_pair, dist_pred, mask_pred_pair, eps)
    bin_centers = torch.linspace(0.0, max_val, n_bins, device=dist_true.device)
    gamma_val = gamma if gamma is not None else max(max_val / max(n_bins * 2, 1), eps)

    feats_true = _histogram_features(dist_true, mask_true_pair, bin_centers, gamma_val)
    feats_pred = _histogram_features(dist_pred, mask_pred_pair, bin_centers, gamma_val)

    if normalize:
        feats_true = feats_true / feats_true.norm(dim=1, keepdim=True).clamp_min(eps)
        feats_pred = feats_pred / feats_pred.norm(dim=1, keepdim=True).clamp_min(eps)

    sim = (feats_true * feats_pred).sum(dim=1)
    return -sim.mean()


def _build_local_neighbor_mask(
    mask_nodes: Tensor,
    dist: Tensor,
    radius: Optional[float],
    k_max: Optional[int],
) -> Tensor:
    B, N, _ = dist.shape
    device = dist.device
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    mask_pair = mask_nodes.unsqueeze(2) & mask_nodes.unsqueeze(1)
    mask_pair = mask_pair & eye.logical_not()
    if radius is not None:
        mask_pair = mask_pair & (dist <= radius)
    if k_max is not None and k_max > 0:
        k = min(k_max, N - 1)
        if k > 0:
            huge = torch.full_like(dist, float("inf"))
            dist_masked = torch.where(mask_pair, dist, huge)
            idx = dist_masked.argsort(dim=2)
            top_idx = idx[:, :, :k]
            new_mask = torch.zeros_like(mask_pair)
            arange_b = torch.arange(B, device=device)[:, None, None]
            arange_i = torch.arange(N, device=device)[None, :, None]
            new_mask[arange_b, arange_i, top_idx] = True
            mask_pair = new_mask & mask_pair
    return mask_pair


def _local_signatures(
    dist: Tensor,
    neighbor_mask: Tensor,
    bin_centers: Tensor,
    gamma: float,
) -> Tuple[Tensor, Tensor]:
    diff = dist.unsqueeze(-1) - bin_centers.view(1, 1, 1, -1)
    feats = torch.exp(-0.5 * (diff / gamma) ** 2)
    feats = feats * neighbor_mask.unsqueeze(-1).float()
    feats = feats.sum(dim=2)
    counts = neighbor_mask.sum(dim=2, keepdim=True).clamp_min(1)
    feats = feats / counts
    node_mask = neighbor_mask.sum(dim=2) > 0
    return feats, node_mask


def _feature_kernel(
    feats_a: Tensor,
    mask_a: Tensor,
    feats_b: Tensor,
    mask_b: Tensor,
    tau: float,
    normalize: bool,
    eps: float,
) -> Tensor:
    diff = feats_a.unsqueeze(2) - feats_b.unsqueeze(1)
    d2 = diff.pow(2).sum(dim=-1)
    sim_mat = torch.exp(-d2 / (2.0 * tau**2))
    valid = mask_a.unsqueeze(2) & mask_b.unsqueeze(1)
    sim = (sim_mat * valid.float()).sum(dim=(1, 2))
    counts = valid.sum(dim=(1, 2)).clamp_min(1)
    sim = sim / counts
    if not normalize:
        return sim

    def _self_sim(feats: Tensor, mask: Tensor) -> Tensor:
        diff_self = feats.unsqueeze(2) - feats.unsqueeze(1)
        d2_self = diff_self.pow(2).sum(dim=-1)
        sim_self = torch.exp(-d2_self / (2.0 * tau**2))
        valid_self = mask.unsqueeze(2) & mask.unsqueeze(1)
        sim_self = (sim_self * valid_self.float()).sum(dim=(1, 2))
        counts_self = valid_self.sum(dim=(1, 2)).clamp_min(1)
        sim_self = sim_self / counts_self
        return sim_self

    self_a = _self_sim(feats_a, mask_a)
    self_b = _self_sim(feats_b, mask_b)
    denom = (self_a * self_b).clamp_min(eps).sqrt()
    return sim / denom


def local_distance_kernel_loss_pyg(
    pos_pred: Tensor,
    pos_true: Tensor,
    batch_true: Tensor,
    *,
    batch_pred: Optional[Tensor] = None,
    center: bool = True,
    num_bins: int = 16,
    r_max: Optional[float] = None,
    gamma: Optional[float] = None,
    radius: Optional[float] = None,
    k_max: Optional[int] = None,
    tau: float = 1.0,
    normalize: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Local distance-signature kernel loss."""

    if batch_pred is None:
        batch_pred = batch_true

    _, mask_true, dist_true, _ = _prepare_dense(pos_true, batch_true, center)
    _, mask_pred, dist_pred, _ = _prepare_dense(pos_pred, batch_pred, center)

    neighbors_true = _build_local_neighbor_mask(mask_true, dist_true, radius, k_max)
    neighbors_pred = _build_local_neighbor_mask(mask_pred, dist_pred, radius, k_max)

    max_val = r_max if r_max is not None else _auto_rmax(dist_true, neighbors_true, dist_pred, neighbors_pred, eps)
    bin_centers = torch.linspace(0.0, max_val, num_bins, device=dist_true.device)
    gamma_val = gamma if gamma is not None else max(max_val / max(num_bins * 2, 1), eps)

    feats_true, nodes_true = _local_signatures(dist_true, neighbors_true, bin_centers, gamma_val)
    feats_pred, nodes_pred = _local_signatures(dist_pred, neighbors_pred, bin_centers, gamma_val)

    sim = _feature_kernel(feats_true, nodes_true, feats_pred, nodes_pred, tau=tau, normalize=normalize, eps=eps)
    return -sim.mean()


def kernel_correlation_loss_pyg(
    pos_pred: Tensor,
    pos_true: Tensor,
    batch_true: Tensor,
    *,
    batch_pred: Optional[Tensor] = None,
    center: bool = True,
    lambda_global: float = 1.0,
    lambda_local: float = 1.0,
    global_config: Optional[Dict] = None,
    local_config: Optional[Dict] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Combined global + local kernel correlation loss."""

    if lambda_global <= 0 and lambda_local <= 0:
        raise ValueError("At least one of lambda_global or lambda_local must be positive.")

    total = 0.0
    weight = 0.0

    if lambda_global > 0:
        cfg = {**_DEFAULT_GLOBAL_CFG, **(global_config or {})}
        loss_g = global_distance_kernel_loss_pyg(
            pos_pred,
            pos_true,
            batch_true,
            batch_pred=batch_pred,
            center=center,
            n_bins=cfg["n_bins"],
            r_max=cfg["r_max"],
            gamma=cfg["gamma"],
            normalize=cfg["normalize"],
            eps=eps,
        )
        total += lambda_global * loss_g
        weight += lambda_global

    if lambda_local > 0:
        cfg_l = {**_DEFAULT_LOCAL_CFG, **(local_config or {})}
        loss_l = local_distance_kernel_loss_pyg(
            pos_pred,
            pos_true,
            batch_true,
            batch_pred=batch_pred,
            center=center,
            num_bins=cfg_l["num_bins"],
            r_max=cfg_l["r_max"],
            gamma=cfg_l["gamma"],
            radius=cfg_l["radius"],
            k_max=cfg_l["k_max"],
            tau=cfg_l["tau"],
            normalize=cfg_l["normalize"],
            eps=eps,
        )
        total += lambda_local * loss_l
        weight += lambda_local

    return total / weight


__all__ = [
    "kernel_correlation_loss_pyg",
    "global_distance_kernel_loss_pyg",
    "local_distance_kernel_loss_pyg",
]
