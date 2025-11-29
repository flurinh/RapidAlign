"""Alignment utilities (random SE(3) + Kabsch)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

Tensor = torch.Tensor


def apply_random_se3(points: Tensor, seed: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Apply a random rotation + translation to points (CPU numpy-based)."""
    device = points.device
    rng = np.random.default_rng(int(seed.item()) if isinstance(seed, Tensor) else None)
    pts = points.detach().cpu().numpy()
    rand = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(rand)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    translation = rng.standard_normal(3) * 0.5
    rotated = pts @ q.T + translation
    return (
        torch.from_numpy(rotated).to(points),
        torch.from_numpy(q).to(points),
        torch.from_numpy(translation).to(points),
    )


def kabsch_align(src: Tensor, tgt: Tensor) -> Tensor:
    """Align src to tgt using Kabsch algorithm (numpy backend)."""
    src_np = src.detach().cpu().numpy()
    tgt_np = tgt.detach().cpu().numpy()
    src_center = src_np.mean(axis=0, keepdims=True)
    tgt_center = tgt_np.mean(axis=0, keepdims=True)
    src_c = src_np - src_center
    tgt_c = tgt_np - tgt_center
    cov = tgt_c.T @ src_c
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    aligned = (src_c @ R.T) + tgt_center
    return torch.from_numpy(aligned).to(src)
