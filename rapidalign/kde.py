"""Kernel-based, correspondence-free similarity (MMD) utilities.

This baseline uses pure PyTorch ops (CPU/GPU) and is fully differentiable.
It provides a zero-valued loss when two weighted point sets are identical.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

Tensor = torch.Tensor

try:  # pragma: no cover - optional CUDA extension
    from . import _cuda
except ImportError:  # pragma: no cover
    _cuda = None


def _normalize_weights(w: Optional[Tensor], n: int, device, dtype) -> Tensor:
    if w is None:
        return torch.full((n,), 1.0 / max(n, 1), device=device, dtype=dtype)
    w = w.to(device=device, dtype=dtype)
    s = w.sum()
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    return w / s


def _pairwise_sqdist(a: Tensor, b: Tensor) -> Tensor:
    # a: (N,d), b: (M,d)
    a2 = (a * a).sum(dim=1, keepdim=True)  # (N,1)
    b2 = (b * b).sum(dim=1, keepdim=True).T  # (1,M)
    # Clamp for numerical stability
    d2 = (a2 + b2 - 2.0 * a @ b.T).clamp_min(0.0)
    return d2


def kde_mmd_loss(
    x: Tensor,
    y: Tensor,
    x_w: Optional[Tensor] = None,
    y_w: Optional[Tensor] = None,
    sigma: float = 0.2,
    center: bool = True,
    cosine: bool = False,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute RKHS distance (MMD^2) or cosine-normalized similarity loss.

    Args:
        x: Points (N,3)
        y: Points (M,3)
        x_w: Optional weights (N,) normalized if provided
        y_w: Optional weights (M,)
        sigma: Gaussian bandwidth (>0)
        center: If True, subtract weighted centroids for translation invariance
        cosine: If True, return cosine-style loss (1 - Kxy/sqrt(Kxx*Kyy)) instead of MMD^2
        eps: Numerical epsilon

    Returns:
        loss, Kxx, Kyy, Kxy (scalars)
    """
    if x.ndim != 2 or y.ndim != 2 or x.size(-1) != y.size(-1):
        raise ValueError("x and y must be (N,d) and (M,d) with same d")

    device = x.device
    dtype = x.dtype
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)
    if (sigma <= 0).any():
        raise ValueError("sigma must be positive")
    x_w = _normalize_weights(x_w, x.shape[0], device, dtype)
    y_w = _normalize_weights(y_w, y.shape[0], device, dtype)

    x_c = x
    y_c = y
    if center:
        mu_x = (x_w.unsqueeze(1) * x).sum(dim=0, keepdim=True)
        mu_y = (y_w.unsqueeze(1) * y).sum(dim=0, keepdim=True)
        x_c = x - mu_x
        y_c = y - mu_y

    denom = 2.0 * (sigma ** 2)

    # Kxx
    d2_xx = _pairwise_sqdist(x_c, x_c)
    Kxx_mat = torch.exp(-d2_xx / denom)
    wxx = x_w.unsqueeze(1) * x_w.unsqueeze(0)
    Kxx = (wxx * Kxx_mat).sum()

    # Kyy
    d2_yy = _pairwise_sqdist(y_c, y_c)
    Kyy_mat = torch.exp(-d2_yy / denom)
    wyy = y_w.unsqueeze(1) * y_w.unsqueeze(0)
    Kyy = (wyy * Kyy_mat).sum()

    # Kxy
    d2_xy = _pairwise_sqdist(x_c, y_c)
    Kxy_mat = torch.exp(-d2_xy / denom)
    wxy = x_w.unsqueeze(1) * y_w.unsqueeze(0)
    Kxy = (wxy * Kxy_mat).sum()

    if cosine:
        denom_c = (Kxx * Kyy).clamp_min(eps).sqrt()
        loss = 1.0 - (Kxy / denom_c)
    else:
        loss = Kxx + Kyy - 2.0 * Kxy

    return loss, Kxx, Kyy, Kxy


__all__ = ["kde_mmd_loss", "kde_mmd_loss_dense", "pyg_kde_mmd_loss"]


def _prepare_batch_weights(weights: Optional[Tensor], mask: Tensor, dtype, device) -> Tensor:
    if weights is None:
        w = mask.to(dtype)
    else:
        w = weights.to(device=device, dtype=dtype) * mask.to(dtype)
    sums = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return w / sums


def _ensure_ptr(
    ptr: Optional[Tensor],
    batch: Optional[Tensor],
    total_nodes: int,
    device,
) -> Tuple[Tensor, int]:
    if ptr is not None:
        ptr = ptr.to(device=device, dtype=torch.long).contiguous()
        if ptr.dim() != 1:
            raise ValueError("ptr must be 1-D")
        if ptr.numel() == 0:
            raise ValueError("ptr must have at least one element")
        if ptr[0].item() != 0:
            raise ValueError("ptr must start at 0")
        if ptr[-1].item() != total_nodes:
            raise ValueError("ptr[-1] must equal number of nodes")
        return ptr, ptr.numel() - 1

    if batch is None:
        raise ValueError("either ptr or batch must be provided")

    batch = batch.to(device=device, dtype=torch.long).contiguous()
    if batch.numel() != total_nodes:
        raise ValueError("batch length must match number of nodes")
    if batch.numel() > 1 and not torch.all(batch[:-1] <= batch[1:]):
        raise ValueError("batch indices must be sorted for contiguous blocks")

    num_graphs = 0 if batch.numel() == 0 else (int(batch[-1].item()) + 1)
    ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    if num_graphs > 0:
        counts = torch.bincount(batch, minlength=num_graphs)
        ptr[1:] = counts.cumsum(0)
    return ptr, num_graphs


def _segment_ids_from_ptr(ptr: Tensor) -> Tensor:
    if ptr.numel() <= 1:
        return ptr.new_empty((0,), dtype=torch.long)
    counts = (ptr[1:] - ptr[:-1]).to(torch.long)
    if counts.numel() == 0 or counts.sum() == 0:
        return ptr.new_empty((0,), dtype=torch.long)
    segments = torch.arange(ptr.numel() - 1, device=ptr.device, dtype=torch.long)
    return torch.repeat_interleave(segments, counts)


def _normalize_segment_weights(
    weights: Optional[Tensor],
    ptr: Tensor,
    segment_ids: Tensor,
    num_segments: int,
    device,
    dtype,
) -> Tensor:
    total_nodes = int(ptr[-1].item())
    if total_nodes == 0:
        if weights is not None and weights.numel() != 0:
            raise ValueError("weights must be empty when there are no nodes")
        return torch.empty((0,), device=device, dtype=dtype)

    if weights is None:
        counts = (ptr[1:] - ptr[:-1]).to(dtype=dtype)
        per_segment = torch.zeros((num_segments,), device=device, dtype=dtype)
        nonzero = counts > 0
        per_segment[nonzero] = 1.0 / counts[nonzero]
        return per_segment.index_select(0, segment_ids)

    weights = weights.to(device=device, dtype=dtype)
    if weights.numel() != total_nodes:
        raise ValueError("weights length must equal number of nodes")

    sums = torch.zeros((num_segments,), device=device, dtype=dtype)
    sums.index_add_(0, segment_ids, weights)
    norm = sums.index_select(0, segment_ids)
    if torch.any(norm <= 0):
        raise ValueError("weights must sum to a positive value per segment")
    return weights / norm


def _center_segments(pos: Tensor, weights: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    if pos.numel() == 0 or segment_ids.numel() == 0:
        return pos
    dim = pos.size(1)
    centroids = torch.zeros((num_segments, dim), device=pos.device, dtype=pos.dtype)
    centroids.index_add_(0, segment_ids, weights.unsqueeze(1) * pos)
    return pos - centroids.index_select(0, segment_ids)


def kde_mmd_loss_dense(
    x: Tensor,
    y: Tensor,
    mask_x: Tensor,
    mask_y: Tensor,
    x_w: Optional[Tensor] = None,
    y_w: Optional[Tensor] = None,
    sigma: float = 0.2,
    center: bool = True,
    cosine: bool = False,
    eps: float = 1e-8,
):
    """Batched KDE/MMD loss on dense padded tensors.

    Args:
        x, y: (B, Nmax, 3) float tensors (GPU or CPU)
        mask_x, mask_y: (B, Nmax) bool tensors indicating valid nodes
        x_w, y_w: optional weights (B, Nmax). If None, uniform over valid nodes.
    Returns:
        loss, Kxx, Kyy, Kxy (each shape (B,))
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if x.ndim != 3 or y.ndim != 3 or x.size(0) != y.size(0) or x.size(2) != y.size(2):
        raise ValueError("x and y must have shape (B, Nmax, D)")

    device = x.device
    dtype = x.dtype
    mask_x = mask_x.to(device=device, dtype=torch.bool)
    mask_y = mask_y.to(device=device, dtype=torch.bool)
    x_w = _prepare_batch_weights(x_w, mask_x, dtype, device)
    y_w = _prepare_batch_weights(y_w, mask_y, dtype, device)

    x_proc = x
    y_proc = y
    if center:
        mu_x = (x_w.unsqueeze(-1) * x).sum(dim=1, keepdim=True)
        mu_y = (y_w.unsqueeze(-1) * y).sum(dim=1, keepdim=True)
        x_proc = (x - mu_x) * mask_x.unsqueeze(-1)
        y_proc = (y - mu_y) * mask_y.unsqueeze(-1)
    else:
        x_proc = x * mask_x.unsqueeze(-1)
        y_proc = y * mask_y.unsqueeze(-1)

    if _cuda is not None and x_proc.is_cuda:
        Kxx, Kyy, Kxy = _cuda.kde_mmd_forward(
            x_proc.contiguous().to(torch.float32),
            y_proc.contiguous().to(torch.float32),
            x_w.contiguous().to(torch.float64),
            y_w.contiguous().to(torch.float64),
            mask_x.contiguous(),
            mask_y.contiguous(),
            float(sigma),
        )
        if cosine:
            loss = 1.0 - (Kxy / (Kxx * Kyy).clamp_min(eps).sqrt())
        else:
            loss = Kxx + Kyy - 2.0 * Kxy
        return loss, Kxx, Kyy, Kxy

    # CPU fallback: compute graph-by-graph
    B = x.size(0)
    losses = []
    kxx_list = []
    kyy_list = []
    kxy_list = []
    for b in range(B):
        mask_q = mask_x[b]
        mask_t = mask_y[b]
        xb = x_proc[b][mask_q]
        yb = y_proc[b][mask_t]
        wb_x = x_w[b][mask_q]
        wb_y = y_w[b][mask_t]
        loss_b, kxx_b, kyy_b, kxy_b = kde_mmd_loss(
            xb, yb, wb_x, wb_y, sigma=sigma, center=False, cosine=cosine, eps=eps)
        losses.append(loss_b)
        kxx_list.append(kxx_b)
        kyy_list.append(kyy_b)
        kxy_list.append(kxy_b)

    return (torch.stack(losses),
            torch.stack(kxx_list),
            torch.stack(kyy_list),
            torch.stack(kxy_list))


def pyg_kde_mmd_loss(
    src_pos: Tensor,
    tgt_pos: Tensor,
    *,
    src_batch: Optional[Tensor] = None,
    tgt_batch: Optional[Tensor] = None,
    src_ptr: Optional[Tensor] = None,
    tgt_ptr: Optional[Tensor] = None,
    src_w: Optional[Tensor] = None,
    tgt_w: Optional[Tensor] = None,
    sigma: float = 0.2,
    center: bool = True,
    cosine: bool = False,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """KDE/MMD loss for ragged PyG-style batches."""

    if src_pos.ndim != 2 or tgt_pos.ndim != 2:
        raise ValueError("src_pos and tgt_pos must be 2-D")
    if src_pos.size(1) != tgt_pos.size(1):
        raise ValueError("source and target must share feature dimension")
    if src_pos.device != tgt_pos.device:
        raise ValueError("source and target must be on the same device")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    device = src_pos.device
    dtype = src_pos.dtype

    src_ptr, num_src = _ensure_ptr(src_ptr, src_batch, src_pos.size(0), device)
    tgt_ptr, num_tgt = _ensure_ptr(tgt_ptr, tgt_batch, tgt_pos.size(0), device)
    if num_src != num_tgt:
        raise ValueError("source and target batches must have the same length")

    num_pairs = num_src
    if num_pairs == 0:
        zero = torch.zeros((0,), device=device, dtype=torch.float64)
        return zero, zero, zero, zero

    src_segments = _segment_ids_from_ptr(src_ptr)
    tgt_segments = _segment_ids_from_ptr(tgt_ptr)
    if src_segments.numel() != src_pos.size(0) or tgt_segments.numel() != tgt_pos.size(0):
        raise ValueError("ptr offsets must match node stacks")

    src_weights = _normalize_segment_weights(src_w, src_ptr, src_segments, num_pairs, device, dtype)
    tgt_weights = _normalize_segment_weights(tgt_w, tgt_ptr, tgt_segments, num_pairs, device, dtype)

    src_proc = src_pos
    tgt_proc = tgt_pos
    if center:
        src_proc = _center_segments(src_proc, src_weights, src_segments, num_pairs)
        tgt_proc = _center_segments(tgt_proc, tgt_weights, tgt_segments, num_pairs)

    sigma_tensor = torch.as_tensor(sigma, device=device, dtype=dtype)
    inv_denom = 2.0 * sigma_tensor ** 2

    if _cuda is not None and src_proc.is_cuda:
        Kxx, Kyy, Kxy = _cuda.kde_mmd_forward_segmented(
            src_proc.contiguous().to(torch.float32),
            tgt_proc.contiguous().to(torch.float32),
            src_weights.contiguous().to(torch.float64),
            tgt_weights.contiguous().to(torch.float64),
            src_ptr.contiguous(),
            tgt_ptr.contiguous(),
            float(sigma_tensor.item()),
        )
        if cosine:
            loss = 1.0 - (Kxy / (Kxx * Kyy).clamp_min(eps).sqrt())
        else:
            loss = Kxx + Kyy - 2.0 * Kxy
        return (
            loss.to(torch.float64),
            Kxx.to(torch.float64),
            Kyy.to(torch.float64),
            Kxy.to(torch.float64),
        )

    losses = []
    kxx_list = []
    kyy_list = []
    kxy_list = []
    for idx in range(num_pairs):
        s0 = int(src_ptr[idx].item())
        s1 = int(src_ptr[idx + 1].item())
        t0 = int(tgt_ptr[idx].item())
        t1 = int(tgt_ptr[idx + 1].item())

        xs = src_proc[s0:s1]
        ys = tgt_proc[t0:t1]
        wx = src_weights[s0:s1]
        wy = tgt_weights[t0:t1]

        if xs.numel() == 0:
            kxx_val = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            d2_xx = _pairwise_sqdist(xs, xs)
            kxx_val = (wx.unsqueeze(1) * wx.unsqueeze(0) * torch.exp(-d2_xx / inv_denom)).sum()

        if ys.numel() == 0:
            kyy_val = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            d2_yy = _pairwise_sqdist(ys, ys)
            kyy_val = (wy.unsqueeze(1) * wy.unsqueeze(0) * torch.exp(-d2_yy / inv_denom)).sum()

        if xs.numel() == 0 or ys.numel() == 0:
            kxy_val = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            d2_xy = _pairwise_sqdist(xs, ys)
            kxy_val = (wx.unsqueeze(1) * wy.unsqueeze(0) * torch.exp(-d2_xy / inv_denom)).sum()

        if cosine:
            denom_c = (kxx_val * kyy_val).clamp_min(eps).sqrt()
            loss_val = 1.0 - (kxy_val / denom_c)
        else:
            loss_val = kxx_val + kyy_val - 2.0 * kxy_val

        losses.append(loss_val)
        kxx_list.append(kxx_val)
        kyy_list.append(kyy_val)
        kxy_list.append(kxy_val)

    return (
        torch.stack(losses).to(device=device, dtype=torch.float64),
        torch.stack(kxx_list).to(device=device, dtype=torch.float64),
        torch.stack(kyy_list).to(device=device, dtype=torch.float64),
        torch.stack(kxy_list).to(device=device, dtype=torch.float64),
    )
