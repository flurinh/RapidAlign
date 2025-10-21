"""Analytic KDE/MMD forward & backward for CPU verification.

These helpers implement the closed-form gradients of the Gaussian kernel
similarity used by ``pyg_kde_mmd_loss``. They are pure PyTorch (CPU) and
serve as a trusted reference for finite-difference tests and future CUDA
autograd implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

Tensor = torch.Tensor


@dataclass
class KDEForwardResult:
    loss: Tensor
    Kxx: Tensor
    Kyy: Tensor
    Kxy: Tensor
    Gxx: Tensor
    Gyy: Tensor
    Gxy: Tensor
    diff_xx: Tensor
    diff_yy: Tensor
    diff_xy: Tensor


def _gaussian_kernel(x: Tensor, y: Tensor, sigma: float) -> Tuple[Tensor, Tensor]:
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, D)
    sqdist = (diff ** 2).sum(dim=2)
    G = torch.exp(-sqdist / (2.0 * sigma * sigma))
    return G, diff


def kde_mmd_forward_cpu(
    x: Tensor,
    y: Tensor,
    w_x: Tensor,
    w_y: Tensor,
    sigma: float,
) -> KDEForwardResult:
    if x.dtype != torch.float64 or y.dtype != torch.float64:
        raise ValueError("kde_mmd_forward_cpu expects float64 tensors for stability")

    Gxx, diff_xx = _gaussian_kernel(x, x, sigma)
    Gyy, diff_yy = _gaussian_kernel(y, y, sigma)
    Gxy, diff_xy = _gaussian_kernel(x, y, sigma)

    coeff_xx = w_x.unsqueeze(1) * w_x.unsqueeze(0) * Gxx
    coeff_yy = w_y.unsqueeze(1) * w_y.unsqueeze(0) * Gyy
    coeff_xy = w_x.unsqueeze(1) * w_y.unsqueeze(0) * Gxy

    Kxx = coeff_xx.sum()
    Kyy = coeff_yy.sum()
    Kxy = coeff_xy.sum()
    loss = Kxx + Kyy - 2.0 * Kxy

    return KDEForwardResult(
        loss=loss,
        Kxx=Kxx,
        Kyy=Kyy,
        Kxy=Kxy,
        Gxx=Gxx,
        Gyy=Gyy,
        Gxy=Gxy,
        diff_xx=diff_xx,
        diff_yy=diff_yy,
        diff_xy=diff_xy,
    )


def kde_mmd_backward_cpu(
    x: Tensor,
    y: Tensor,
    w_x: Tensor,
    w_y: Tensor,
    sigma: float,
) -> Dict[str, Tensor]:
    """Analytic gradients of KDE/MMD loss w.r.t. inputs."""
    if x.dtype != torch.float64 or y.dtype != torch.float64:
        raise ValueError("kde_mmd_backward_cpu expects float64 tensors")

    fwd = kde_mmd_forward_cpu(x, y, w_x, w_y, sigma)

    inv_sigma2 = 1.0 / (sigma * sigma)

    coeff_xx = w_x.unsqueeze(1) * w_x.unsqueeze(0) * fwd.Gxx
    coeff_yy = w_y.unsqueeze(1) * w_y.unsqueeze(0) * fwd.Gyy
    coeff_xy = w_x.unsqueeze(1) * w_y.unsqueeze(0) * fwd.Gxy

    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(y)

    # self terms (x vs x)
    contrib_xx = coeff_xx.unsqueeze(2) * fwd.diff_xx
    grad_x -= (2.0 * inv_sigma2) * contrib_xx.sum(dim=1)

    contrib_yy = coeff_yy.unsqueeze(2) * fwd.diff_yy
    grad_y -= (2.0 * inv_sigma2) * contrib_yy.sum(dim=1)

    # cross terms (x vs y)
    contrib_xy = coeff_xy.unsqueeze(2) * fwd.diff_xy
    grad_x += (2.0 * inv_sigma2) * contrib_xy.sum(dim=1)
    grad_y += (2.0 * inv_sigma2) * contrib_xy.transpose(0, 1).neg().sum(dim=1)

    # gradients w.r.t. weights
    grad_wx = 2.0 * (fwd.Gxx @ w_x) - 2.0 * (fwd.Gxy @ w_y)
    grad_wy = 2.0 * (fwd.Gyy @ w_y) - 2.0 * (fwd.Gxy.t() @ w_x)

    # sigma gradient
    sq_xx = (fwd.diff_xx ** 2).sum(dim=2)
    sq_yy = (fwd.diff_yy ** 2).sum(dim=2)
    sq_xy = (fwd.diff_xy ** 2).sum(dim=2)

    inv_sigma3 = 1.0 / (sigma * sigma * sigma)
    dKxx_dsigma = (coeff_xx * sq_xx).sum() * inv_sigma3
    dKyy_dsigma = (coeff_yy * sq_yy).sum() * inv_sigma3
    dKxy_dsigma = (coeff_xy * sq_xy).sum() * inv_sigma3
    grad_sigma = dKxx_dsigma + dKyy_dsigma - 2.0 * dKxy_dsigma

    return {
        "x": grad_x,
        "y": grad_y,
        "w_x": grad_wx,
        "w_y": grad_wy,
        "sigma": torch.as_tensor(grad_sigma, dtype=torch.float64),
        "forward": fwd,
    }


def kde_mmd_backward_ragged_cpu(
    x: Tensor,
    y: Tensor,
    w_x: Tensor,
    w_y: Tensor,
    ptr_x: Tensor,
    ptr_y: Tensor,
    sigma: float,
) -> Dict[str, Tensor]:
    if not (x.dtype == y.dtype == w_x.dtype == w_y.dtype == torch.float64):
        raise ValueError("ragged backward expects float64 tensors")
    if ptr_x.numel() != ptr_y.numel():
        raise ValueError("ptr_x and ptr_y must have the same length")

    num_pairs = ptr_x.numel() - 1
    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(y)
    grad_wx = torch.zeros_like(w_x)
    grad_wy = torch.zeros_like(w_y)
    grad_sigma_per_pair = torch.zeros(num_pairs, dtype=torch.float64)

    losses = torch.zeros(num_pairs, dtype=torch.float64)
    Kxx = torch.zeros_like(losses)
    Kyy = torch.zeros_like(losses)
    Kxy = torch.zeros_like(losses)

    for b in range(num_pairs):
        xs = slice(ptr_x[b].item(), ptr_x[b + 1].item())
        ys = slice(ptr_y[b].item(), ptr_y[b + 1].item())
        sub = kde_mmd_backward_cpu(
            x[xs],
            y[ys],
            w_x[xs],
            w_y[ys],
            sigma,
        )
        grad_x[xs] = sub["x"]
        grad_y[ys] = sub["y"]
        grad_wx[xs] = sub["w_x"]
        grad_wy[ys] = sub["w_y"]
        grad_sigma_per_pair[b] = sub["sigma"]

        losses[b] = sub["forward"].loss
        Kxx[b] = sub["forward"].Kxx
        Kyy[b] = sub["forward"].Kyy
        Kxy[b] = sub["forward"].Kxy

    return {
        "x": grad_x,
        "y": grad_y,
        "w_x": grad_wx,
        "w_y": grad_wy,
        "sigma_per_pair": grad_sigma_per_pair,
        "sigma_total": grad_sigma_per_pair.sum(),
        "loss": losses,
        "Kxx": Kxx,
        "Kyy": Kyy,
        "Kxy": Kxy,
    }


__all__ = [
    "kde_mmd_forward_cpu",
    "kde_mmd_backward_cpu",
    "kde_mmd_backward_ragged_cpu",
    "KDEForwardResult",
]
