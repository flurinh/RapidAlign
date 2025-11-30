# Latent_encoding/data/noise.py
from __future__ import annotations

"""Synthetic graph noise generation for loss evaluation and loss-network pretraining.

All noise is applied in coordinate space only; node features and edges are preserved.
The API is batch-aware and works directly with PyG Data objects.

We also return an SE(3)-invariant severity measure based on normalized node and
pairwise distance distortions.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data

NoiseType = str  # e.g. "rigid", "global_gaussian", "local_gaussian", "shear", ...


@dataclass
class NoiseConfig:
    """
    Configuration for sampling synthetic noise on graphs.
    """
    types: Tuple[NoiseType, ...] = (
        "rigid",
        "global_gaussian",
        "local_gaussian",
        "anisotropic_scale",
        "drift",
        "shear",
        "bend",
        "overlap",
    )

    # Gaussian / scale
    sigma_global: float = 0.1
    sigma_local: float = 0.2
    local_fraction: float = 0.3
    scale_std: float = 0.25

    # Rigid
    max_rotation_deg: float = 180.0
    translation_std: float = 0.0
    include_rigid_in_supervision: bool = False

    # Structural
    drift_max: float = 0.5
    drift_fraction: float = 0.3
    shear_max: float = 0.5
    bend_max_deg: float = 45.0
    overlap_fraction: float = 0.1
    overlap_jitter_std: float = 0.0

    # Supervision
    alpha_node: float = 0.5
    beta_edge: float = 0.5

    # [NEW] Training Stability Control
    # If True: Use Root-Mean-Square (Linear) severity.
    #   - Small errors (0.01) -> 0.01 Loss (Good signal)
    #   - Large errors (10.0) -> 3.16 Loss (Compressed)
    # If False: Use Mean-Squared-Error (Quadratic) severity.
    #   - Small errors (0.01) -> 0.0001 Loss (Vanishing signal)
    #   - Large errors (10.0) -> 100.0 Loss (Exploding signal)
    use_robust_metric: bool = True

    # Misc
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------

def _get_rng(cfg: NoiseConfig, rng: Optional[random.Random]) -> random.Random:
    """Return an RNG, optionally seeded from cfg."""
    if rng is not None:
        return rng
    if cfg.seed is not None:
        return random.Random(cfg.seed)
    return random


# ---------------------------------------------------------------------------
# Core SE(3)-invariant distortion metric for supervision
# ---------------------------------------------------------------------------

def _node_edge_severity(
        pos: Tensor,
        pos_noisy: Tensor,
        *,
        alpha_node: float,
        beta_edge: float,
        use_robust_metric: bool = True,
        eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    SE(3)-invariant severity calculation.

    Returns (L_true, L_node, L_edge).
    """
    assert pos.shape == pos_noisy.shape
    N = pos.size(0)
    if N <= 1:
        return 0.0, 0.0, 0.0

    with torch.no_grad():
        d0 = torch.cdist(pos, pos)  # [N,N]
        d1 = torch.cdist(pos_noisy, pos_noisy)

        triu = torch.triu(torch.ones_like(d0, dtype=torch.bool), diagonal=1)
        d0_vec = d0[triu]
        d1_vec = d1[triu]

        if d0_vec.numel() == 0:
            return 0.0, 0.0, 0.0

        # Characteristic Scale (Mean Squared Distance)
        s2 = (d0_vec.pow(2).mean()).clamp_min(eps)

        # 1. Node Component (Normalized MSE)
        dx = pos_noisy - pos
        E_node = dx.pow(2).sum(dim=-1).mean()
        L_node_mse = (E_node / s2).item()

        # 2. Edge Component (Normalized MSE)
        diff = d1_vec - d0_vec
        E_edge = diff.pow(2).mean()
        L_edge_mse = (E_edge / s2).item()

        # 3. Combine
        if use_robust_metric:
            # Linear (RMS) scaling: Sqrt recovers "Distance" from "Energy"
            L_node = math.sqrt(L_node_mse)
            L_edge = math.sqrt(L_edge_mse)
        else:
            # Quadratic (MSE) scaling
            L_node = L_node_mse
            L_edge = L_edge_mse

        L_true = alpha_node * L_node + beta_edge * L_edge

    return float(L_true), float(L_node), float(L_edge)


# ---------------------------------------------------------------------------
# Basic geometric helpers
# ---------------------------------------------------------------------------

def _sample_rotation(angle_deg_max: float, device: torch.device, rng: random.Random) -> Tensor:
    """Sample a random 3×3 rotation matrix."""
    if angle_deg_max <= 0.0:
        return torch.eye(3, device=device)

    u = rng.random()
    angle = math.radians(u * angle_deg_max)
    angle_t = torch.tensor(angle, device=device, dtype=torch.float32)

    axis = torch.randn(3, device=device)
    axis = axis / axis.norm().clamp_min(1e-8)
    x, y, z = axis

    K = torch.tensor(
        [[0.0, -z, y],
         [z, 0.0, -x],
         [-y, x, 0.0]],
        device=device,
        dtype=torch.float32,
    )

    I = torch.eye(3, device=device)
    R = I + torch.sin(angle_t) * K + (1.0 - torch.cos(angle_t)) * (K @ K)
    return R


def _random_unit_vector(device: torch.device, rng: random.Random) -> Tensor:
    v = torch.randn(3, device=device)
    v = v / v.norm().clamp_min(1e-8)
    return v


def _orthonormal_basis(device: torch.device, rng: random.Random) -> Tuple[Tensor, Tensor, Tensor]:
    """Return an orthonormal basis (k,u,v)."""
    k = _random_unit_vector(device, rng)
    u = _random_unit_vector(device, rng)
    u = u - (u @ k) * k
    u = u / u.norm().clamp_min(1e-8)
    v = torch.cross(k, u, dim=0)
    v = v / v.norm().clamp_min(1e-8)
    return k, u, v


# ---------------------------------------------------------------------------
# Individual noise families
# ---------------------------------------------------------------------------

def _apply_rigid(pos: Tensor, cfg: NoiseConfig, rng: random.Random) -> Tuple[Tensor, float, Dict]:
    device = pos.device
    R = _sample_rotation(cfg.max_rotation_deg, device=device, rng=rng)
    out = pos @ R.T
    if cfg.translation_std > 0.0:
        t = torch.randn(1, 3, device=device) * cfg.translation_std
        out = out + t

    L_true, L_node, L_edge = _node_edge_severity(
        pos,
        out,
        alpha_node=cfg.alpha_node,
        beta_edge=cfg.beta_edge,
        use_robust_metric=cfg.use_robust_metric,
    )
    severity = L_true if cfg.include_rigid_in_supervision else 0.0

    meta = {
        "noise_type": "rigid",
        "L_true": L_true,
        "L_node": L_node,
        "L_edge": L_edge,
    }
    return out, float(severity), meta


def _apply_global_gaussian(pos: Tensor, cfg: NoiseConfig, rng: random.Random) -> Tuple[Tensor, float, Dict]:
    if cfg.sigma_global <= 0.0 or pos.numel() == 0:
        return pos, 0.0, {
            "noise_type": "global_gaussian",
            "sigma": 0.0,
            "L_true": 0.0, "L_node": 0.0, "L_edge": 0.0,
        }

    u = rng.random()
    sigma = u * cfg.sigma_global
    noise = torch.randn_like(pos) * sigma
    out = pos + noise

    L_true, L_node, L_edge = _node_edge_severity(
        pos,
        out,
        alpha_node=cfg.alpha_node,
        beta_edge=cfg.beta_edge,
        use_robust_metric=cfg.use_robust_metric,
    )
    meta = {
        "noise_type": "global_gaussian",
        "sigma": sigma,
        "L_true": L_true, "L_node": L_node, "L_edge": L_edge,
    }
    return out, float(L_true), meta


def _apply_local_gaussian(pos: Tensor, cfg: NoiseConfig, rng: random.Random) -> Tuple[Tensor, float, Dict]:
    N = pos.size(0)
    if cfg.sigma_local <= 0.0 or N == 0:
        return pos, 0.0, {
            "noise_type": "local_gaussian",
            "sigma": 0.0, "fraction": 0.0, "num_noisy_nodes": 0,
            "L_true": 0.0, "L_node": 0.0, "L_edge": 0.0,
        }

    frac = max(min(cfg.local_fraction, 1.0), 0.0)
    k = max(1, int(round(frac * N)))

    u = rng.random()
    sigma = u * cfg.sigma_local

    device = pos.device
    out = pos.clone()
    idx = torch.randperm(N, device=device)[:k]
    out[idx] = out[idx] + torch.randn(k, 3, device=device) * sigma

    L_true, L_node, L_edge = _node_edge_severity(
        pos,
        out,
        alpha_node=cfg.alpha_node,
        beta_edge=cfg.beta_edge,
        use_robust_metric=cfg.use_robust_metric,
    )
    meta = {
        "noise_type": "local_gaussian",
        "sigma": sigma,
        "fraction": k / max(N, 1),
        "num_noisy_nodes": int(k),
        "L_true": L_true, "L_node": L_node, "L_edge": L_edge,
    }
    return out, float(L_true), meta


def _apply_anisotropic_scale(pos: Tensor, cfg: NoiseConfig, rng: random.Random) -> Tuple[Tensor, float, Dict]:
    if cfg.scale_std <= 0.0:
        return pos, 0.0, {
            "noise_type": "anisotropic_scale",
            "scale": 1.0,
            "L_true": 0.0, "L_node": 0.0, "L_edge": 0.0,
        }

    u = rng.random()
    delta = u * cfg.scale_std
    sign = -1.0 if rng.random() < 0.5 else 1.0
    scale = 1.0 + sign * delta
    out = pos * scale

    L_true, L_node, L_edge = _node_edge_severity(
        pos,
        out,
        alpha_node=cfg.alpha_node,
        beta_edge=cfg.beta_edge,
        use_robust_metric=cfg.use_robust_metric,
    )
    meta = {
        "noise_type": "anisotropic_scale",
        "scale": scale,
        "L_true": L_true, "L_node": L_node, "L_edge": L_edge,
    }
    return out, float(L_true), meta


def _apply_drift(pos: Tensor, cfg: NoiseConfig, rng: random.Random) -> Tuple[Tensor, float, Dict]:
    N = pos.size(0)
    if cfg.drift_max <= 0.0 or N == 0:
        return pos, 0.0, {
            "noise_type": "drift",
            "fraction": 0.0, "num_drift_nodes": 0, "shift_norm": 0.0,
            "L_true": 0.0, "L_node": 0.0, "L_edge": 0.0,
        }

    frac = max(min(cfg.drift_fraction, 1.0), 0.0)
    k = max(1, int(round(frac * N)))

    device = pos.device
    amp = rng.random() * cfg.drift_max
    direction = _random_unit_vector(device, rng)
    shift = amp * direction

    out = pos.clone()
    idx = torch.randperm(N, device=device)[:k]
    out[idx] = out[idx] + shift

    L_true, L_node, L_edge = _node_edge_severity(
        pos,
        out,
        alpha_node=cfg.alpha_node,
        beta_edge=cfg.beta_edge,
        use_robust_metric=cfg.use_robust_metric,
    )
    meta = {
        "noise_type": "drift",
        "fraction": k / max(N, 1),
        "num_drift_nodes": int(k),
        "shift_norm": float(amp),
        "L_true": L_true, "L_node": L_node, "L_edge": L_edge,
    }
    return out, float(L_true), meta


def _apply_shear(pos: Tensor, cfg: NoiseConfig, rng: random.Random) -> Tuple[Tensor, float, Dict]:
    N = pos.size(0)
    if cfg.shear_max <= 0.0 or N == 0:
        return pos, 0.0, {
            "noise_type": "shear",
            "k": 0.0,
            "L_true": 0.0, "L_node": 0.0, "L_edge": 0.0,
        }

    device = pos.device
    k = (2.0 * rng.random() - 1.0) * cfg.shear_max
    normal = _random_unit_vector(device, rng)
    center = pos.mean(dim=0, keepdim=True)
    offset = -(center @ normal.unsqueeze(-1)).squeeze(-1)

    dir_vec = _random_unit_vector(device, rng)
    dir_vec = dir_vec - (dir_vec @ normal) * normal
    dir_vec = dir_vec / dir_vec.norm().clamp_min(1e-8)

    d = (pos @ normal) + offset
    out = pos + k * d.unsqueeze(-1) * dir_vec.unsqueeze(0)

    L_true, L_node, L_edge = _node_edge_severity(
        pos,
        out,
        alpha_node=cfg.alpha_node,
        beta_edge=cfg.beta_edge,
        use_robust_metric=cfg.use_robust_metric,
    )
    meta = {
        "noise_type": "shear",
        "k": k,
        "L_true": L_true, "L_node": L_node, "L_edge": L_edge,
    }
    return out, float(L_true), meta


def _apply_bend(pos: Tensor, cfg: NoiseConfig, rng: random.Random) -> Tuple[Tensor, float, Dict]:
    N = pos.size(0)
    if cfg.bend_max_deg <= 0.0 or N == 0:
        return pos, 0.0, {
            "noise_type": "bend",
            "max_angle_deg": 0.0,
            "L_true": 0.0, "L_node": 0.0, "L_edge": 0.0,
        }

    device = pos.device
    k, u, v = _orthonormal_basis(device, rng)
    center = pos.mean(dim=0, keepdim=True)
    rel = pos - center

    z = (rel @ k)
    x = (rel @ u)
    y = (rel @ v)

    z_max = z.abs().max().clamp_min(1e-6)
    max_angle_rad = math.radians(cfg.bend_max_deg)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    max_angle_rad = sign * max_angle_rad

    theta = max_angle_rad * (z / z_max)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    new_z = z * cos_t - x * sin_t
    new_x = z * sin_t + x * cos_t
    new_y = y

    out = center + new_z.unsqueeze(-1) * k + new_x.unsqueeze(-1) * u + new_y.unsqueeze(-1) * v

    L_true, L_node, L_edge = _node_edge_severity(
        pos,
        out,
        alpha_node=cfg.alpha_node,
        beta_edge=cfg.beta_edge,
        use_robust_metric=cfg.use_robust_metric,
    )
    meta = {
        "noise_type": "bend",
        "max_angle_deg": cfg.bend_max_deg * sign,
        "L_true": L_true, "L_node": L_node, "L_edge": L_edge,
    }
    return out, float(L_true), meta


def _apply_overlap(pos: Tensor, cfg: NoiseConfig, rng: random.Random) -> Tuple[Tensor, float, Dict]:
    N = pos.size(0)
    if N <= 1 or cfg.overlap_fraction <= 0.0:
        return pos, 0.0, {
            "noise_type": "overlap",
            "fraction": 0.0, "num_overlap_nodes": 0,
            "L_true": 0.0, "L_node": 0.0, "L_edge": 0.0,
        }

    frac = max(min(cfg.overlap_fraction, 1.0), 0.0)
    k = max(1, int(round(frac * N)))

    device = pos.device
    out = pos.clone()
    idx_targets = torch.randperm(N, device=device)[:k]

    for j in idx_targets.tolist():
        if N == 1:
            continue
        donor = j
        while donor == j:
            donor = rng.randrange(0, N)
        out[j] = pos[donor]

    if cfg.overlap_jitter_std > 0.0:
        out[idx_targets] = out[idx_targets] + torch.randn_like(out[idx_targets]) * cfg.overlap_jitter_std

    L_true, L_node, L_edge = _node_edge_severity(
        pos,
        out,
        alpha_node=cfg.alpha_node,
        beta_edge=cfg.beta_edge,
        use_robust_metric=cfg.use_robust_metric,
    )
    meta = {
        "noise_type": "overlap",
        "fraction": k / max(N, 1),
        "num_overlap_nodes": int(k),
        "L_true": L_true, "L_node": L_node, "L_edge": L_edge,
    }
    return out, float(L_true), meta


# ---------------------------------------------------------------------------
# Dispatch + batched API
# ---------------------------------------------------------------------------

NOISE_DISPATCH = {
    "rigid": _apply_rigid,
    "global_gaussian": _apply_global_gaussian,
    "local_gaussian": _apply_local_gaussian,
    "anisotropic_scale": _apply_anisotropic_scale,
    "drift": _apply_drift,
    "shear": _apply_shear,
    "bend": _apply_bend,
    "overlap": _apply_overlap,
}


def noisify_batch(
        batch_data: Data,
        cfg: NoiseConfig,
        rng: Optional[random.Random] = None,
) -> Tuple[Data, Tensor, List[Dict]]:
    """
    Apply per-graph noise to a batched PyG Data object.
    """
    rng = _get_rng(cfg, rng)
    pos = batch_data.pos
    batch = batch_data.batch
    device = pos.device

    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    pos_noisy = pos.clone()
    severities = torch.zeros(B, dtype=torch.float32, device=device)
    metas: List[Dict] = []

    if B == 0:
        noisy_data = batch_data.clone()
        return noisy_data, severities, metas

    available_types: Sequence[NoiseType] = cfg.types

    for b in range(B):
        mask = batch == b
        pos_b = pos[mask]
        if pos_b.numel() == 0:
            metas.append({"noise_type": "empty", "L_true": 0.0, "L_node": 0.0, "L_edge": 0.0})
            continue

        noise_type = rng.choice(available_types)
        apply_fn = NOISE_DISPATCH.get(noise_type)
        if apply_fn is None:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        pos_b_noisy, sev_b, meta_b = apply_fn(pos_b, cfg, rng)
        pos_noisy[mask] = pos_b_noisy
        severities[b] = float(sev_b)
        meta_b.setdefault("noise_type", noise_type)
        metas.append(meta_b)

    noisy_data = batch_data.clone()
    noisy_data.pos = pos_noisy
    return noisy_data, severities, metas


__all__ = ["NoiseConfig", "noisify_batch", "NOISE_DISPATCH"]