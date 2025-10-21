"""Learned multi-scale KDE/MMD kernel for graphs."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .kde import kde_mmd_loss

Tensor = torch.Tensor


def _full_cdist(a: Tensor, b: Tensor) -> Tensor:
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True).T
    d2 = (a2 + b2 - 2.0 * (a @ b.T)).clamp_min(0.0)
    return d2.sqrt()


class NodeWeightNet(nn.Module):
    """Simple MLP to produce positive node weights for a point cloud."""

    def __init__(self, in_feats: int = 2, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, pos: Tensor) -> Tensor:
        logits = self.mlp(pos)
        weights = F.softplus(logits) + 1e-6
        weights = weights / weights.sum()
        return weights.squeeze(-1)


class LearnedKernel(nn.Module):
    """Multi-scale KDE/MMD kernel with learnable weights and anisotropy.

    Args:
        sigmas: initial bandwidths
        learn_scale: learn anisotropic per-axis scale via diagonal matrix
        use_weight_net: learn node weights from coordinates
        gated: learn a gate that mixes the per-sigma losses based on pair statistics
    """

    def __init__(
        self,
        sigmas: Sequence[float] = (0.05, 0.1, 0.2),
        learn_scale: bool = True,
        use_weight_net: bool = True,
        gated: bool = True,
        coord_dim: int = 2,
    ):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.log(torch.tensor(sigmas, dtype=torch.float32)))
        self.use_weight_net = use_weight_net
        self.learn_scale = learn_scale
        self.coord_dim = coord_dim

        if use_weight_net:
            self.weight_net = NodeWeightNet(in_feats=coord_dim)
        if learn_scale:
            self.log_scale = nn.Parameter(torch.zeros(coord_dim))

        hidden = 32
        self.gate = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(sigmas))
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def _scale_points(self, pts: Tensor) -> Tensor:
        if not self.learn_scale:
            return pts
        scale = torch.exp(self.log_scale)
        return pts * scale

    def _weights(self, pts: Tensor) -> Tensor:
        if not self.use_weight_net:
            return torch.full((pts.size(0),), 1.0 / pts.size(0), device=pts.device, dtype=pts.dtype)
        return self.weight_net(pts)

    def forward(self, base_points: Tensor, target_points: Tensor) -> Tensor:
        base = self._scale_points(base_points)
        target = self._scale_points(target_points)

        w_base = self._weights(base)
        w_target = self._weights(target)

        per_band_losses = []
        for sigma in torch.exp(self.log_sigmas):
            loss, _, _, _ = kde_mmd_loss(base, target, w_base, w_target, sigma=sigma, center=True)
            per_band_losses.append(loss)
        per_band_losses = torch.stack(per_band_losses)

        dist_mat = _full_cdist(base, target)
        mean_dist = dist_mat.mean()
        var_dist = dist_mat.var()
        size_diff = abs(float(base.size(0) - target.size(0))) / (base.size(0) + 1e-6)
        weight_var = w_base.var() + w_target.var()

        stats = torch.tensor([mean_dist, var_dist, size_diff, weight_var],
                             device=base.device, dtype=base.dtype)

        mix_logits = self.gate(stats.float())
        weights = F.softplus(mix_logits)
        weights = weights / weights.sum()
        loss = (weights * per_band_losses).sum() + self.bias.squeeze()
        return loss


class SingleSigmaKernel(nn.Module):
    """Single-band KDE/MMD kernel with optional node weights and anisotropy."""

    def __init__(
        self,
        init_sigma: float = 0.1,
        learn_scale: bool = True,
        use_weight_net: bool = False,
        coord_dim: int = 2,
    ):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma), dtype=torch.float32))
        self.learn_scale = learn_scale
        self.use_weight_net = use_weight_net
        self.coord_dim = coord_dim

        if learn_scale:
            self.log_scale = nn.Parameter(torch.zeros(coord_dim))
        if use_weight_net:
            self.weight_net = NodeWeightNet(in_feats=coord_dim)

    def _scale_points(self, pts: Tensor) -> Tensor:
        if not self.learn_scale:
            return pts
        return pts * torch.exp(self.log_scale)

    def _weights(self, pts: Tensor) -> Tensor:
        if not self.use_weight_net:
            return torch.full((pts.size(0),), 1.0 / pts.size(0), device=pts.device, dtype=pts.dtype)
        return self.weight_net(pts)

    def forward(self, base_points: Tensor, target_points: Tensor) -> Tensor:
        base = self._scale_points(base_points)
        target = self._scale_points(target_points)

        w_base = self._weights(base)
        w_target = self._weights(target)

        sigma = torch.exp(self.log_sigma)
        loss, _, _, _ = kde_mmd_loss(base, target, w_base, w_target, sigma=sigma, center=True)
        return loss


class StructuredKernel(nn.Module):
    """Single-band KDE/MMD kernel with structural node features and descriptors."""

    def __init__(
        self,
        init_sigma: float = 0.1,
        learn_scale: bool = True,
        feature_dim: int = 5,
        descriptor_dim: int = 4,
    ):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma), dtype=torch.float32))
        self.learn_scale = learn_scale
        if learn_scale:
            self.log_scale = nn.Parameter(torch.zeros(2))

        self.weight_net = NodeWeightNet(in_feats=feature_dim)
        self.descriptor_gate = nn.Sequential(
            nn.Linear(descriptor_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def _scale_points(self, pts: Tensor) -> Tensor:
        if not self.learn_scale:
            return pts
        return pts * torch.exp(self.log_scale)

    def forward(self, base: dict, target: dict) -> Tensor:
        base_pos = self._scale_points(base['pos'])
        target_pos = self._scale_points(target['pos'])

        sigma = torch.exp(self.log_sigma)
        base_w = self.weight_net(base['feat'])
        target_w = self.weight_net(target['feat'])

        loss, _, _, _ = kde_mmd_loss(base_pos, target_pos, base_w, target_w, sigma=sigma, center=True)

        desc_diff = torch.abs(base['desc'] - target['desc'])
        scale = torch.nn.functional.softplus(self.descriptor_gate(desc_diff)) + 1e-6
        return loss * scale.squeeze() + self.bias.squeeze()


__all__ = ["LearnedKernel", "NodeWeightNet", "SingleSigmaKernel", "StructuredKernel"]
