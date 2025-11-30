# Latent_encoding/losses.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Sequence

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

Tensor = torch.Tensor

# ---------------------------
# Utilities
# ---------------------------

def _center_cloud(x: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.unsqueeze(-1).float()
    counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1e-8)
    centers = (x * mask_f).sum(dim=1, keepdim=True) / counts
    return x - centers

def _prepare_dense(pos: Tensor, batch: Tensor, center: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x, mask = to_dense_batch(pos, batch)
    if center:
        x = _center_cloud(x, mask)
    dist = torch.cdist(x, x)  # [B, N, N]
    mask_pair = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N] boolean
    return x, mask, dist, mask_pair

def _auto_rmax(dist_a: Tensor, mask_a: Tensor, dist_b: Tensor, mask_b: Tensor, eps: float) -> float:
    val = 0.0
    if mask_a.any():
        val = max(val, dist_a[mask_a].max().detach().item())
    if mask_b.any():
        val = max(val, dist_b[mask_b].max().detach().item())
    if not math.isfinite(val) or val <= eps:
        return 1.0
    return val


# ---------------------------
# (A) Your original binned KC loss (kept for compatibility)
# ---------------------------

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
    return feats  # [B, n_bins]

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
    if batch_pred is None:
        batch_pred = batch_true
    _, _, dist_true, mask_true_pair = _prepare_dense(pos_true, batch_true, center)
    _, _, dist_pred, mask_pred_pair = _prepare_dense(pos_pred, batch_pred, center)

    eye = torch.eye(dist_true.size(1), device=dist_true.device, dtype=torch.bool).unsqueeze(0)
    mask_true_pair = mask_true_pair & ~eye
    mask_pred_pair = mask_pred_pair & ~eye

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
    mask_pair = mask_pair & ~eye
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
    return feats, node_mask  # [B,N,bins], [B,N]

def _feature_kernel(
    feats_a: Tensor,
    mask_a: Tensor,
    feats_b: Tensor,
    mask_b: Tensor,
    tau: float,
    normalize: bool,
    eps: float,
) -> Tensor:
    diff = feats_a.unsqueeze(2) - feats_b.unsqueeze(1)  # [B,N,N,bins]
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
    _DEFAULT_GLOBAL_CFG = dict(n_bins=32, r_max=None, gamma=None, normalize=True)
    _DEFAULT_LOCAL_CFG = dict(num_bins=16, r_max=None, gamma=None, radius=None, k_max=None, tau=1.0, normalize=True)
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

# ---------------------------
# (B) Unbinned RFF losses
# ---------------------------

class RFF1D(nn.Module):
    """Random Fourier features for 1D RBF kernel k(r, r') with lengthscale sigma."""
    def __init__(self, num_features: int = 64, sigma: float = 1.0, seed: int = 0):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        w = torch.randn(num_features, generator=g) / max(sigma, 1e-8)  # N(0, 1/sigma^2)
        b = 2 * math.pi * torch.rand(num_features, generator=g)
        self.register_buffer("w", w)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / num_features)

    def forward(self, r: Tensor) -> Tensor:
        # r: arbitrary shape (...), returns (..., num_features)
        proj = r.unsqueeze(-1) * self.w  # (..., M)
        return self.scale * torch.cos(proj + self.b)

def _rff_sigma_auto(dist: Tensor, mask: Tensor) -> float:
    vals = dist[mask]
    if vals.numel() == 0:
        return 1.0
    med = torch.median(vals)
    return float(med.item()) if math.isfinite(float(med)) and med > 1e-8 else 1.0

def global_rff_mmd_loss(
    pos_pred: Tensor,
    pos_true: Tensor,
    batch_true: Tensor,
    *,
    batch_pred: Optional[Tensor] = None,
    center: bool = True,
    num_features: int = 64,
    sigma: Optional[float] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Unbinned RFF MMD over pairwise distance distributions."""
    if batch_pred is None:
        batch_pred = batch_true
    _, _, d_true, m_true = _prepare_dense(pos_true, batch_true, center)
    _, _, d_pred, m_pred = _prepare_dense(pos_pred, batch_pred, center)

    B, N, _ = d_true.shape
    eye = torch.eye(N, device=d_true.device, dtype=torch.bool).unsqueeze(0)
    m_true = m_true & ~eye
    m_pred = m_pred & ~eye

    sig = sigma if sigma is not None else _rff_sigma_auto(d_true, m_true)
    rff = RFF1D(num_features=num_features, sigma=sig).to(d_true.device)

    # Lower triangle to avoid double counting (i<j)
    tril = torch.tril(torch.ones(N, N, device=d_true.device, dtype=torch.bool), diagonal=-1).unsqueeze(0)
    m_true = m_true & tril
    m_pred = m_pred & tril

    def _embed(d, m):
        feats = rff(d[m])  # [#pairs, M]
        # Build per-graph means
        # Compute counts per graph:
        idx_b = torch.arange(B, device=d.device).repeat_interleave(N*N)
        # But constructing idx_b like this is heavy; slice per-graph instead:
        means = []
        start = 0
        for b in range(B):
            mask_b = m[b]
            nb = int(mask_b.sum().item())
            if nb == 0:
                means.append(torch.zeros(rff.w.numel(), device=d.device))
            else:
                fb = rff(d[b][mask_b])  # [nb, M]
                means.append(fb.mean(dim=0))
        return torch.stack(means, dim=0)  # [B, M]

    mu_t = _embed(d_true, m_true)
    mu_p = _embed(d_pred, m_pred)
    mmd2 = ((mu_t - mu_p) ** 2).sum(dim=1)
    return mmd2.mean()

def local_rff_mmd_loss(
    pos_pred: Tensor,
    pos_true: Tensor,
    batch_true: Tensor,
    *,
    batch_pred: Optional[Tensor] = None,
    center: bool = True,
    num_features: int = 32,
    sigma: Optional[float] = None,
    radius: Optional[float] = None,
    k_max: Optional[int] = None,
    tau: float = 1.0,
    normalize: bool = True,
) -> Tensor:
    """Unbinned local node-signature RFF + MMD across nodes (correspondence-free)."""
    if batch_pred is None:
        batch_pred = batch_true
    x_t, mask_t, d_t, _ = _prepare_dense(pos_true, batch_true, center)
    x_p, mask_p, d_p, _ = _prepare_dense(pos_pred, batch_pred, center)

    # Build neighbors
    nbr_t = _build_local_neighbor_mask(mask_t, d_t, radius, k_max)
    nbr_p = _build_local_neighbor_mask(mask_p, d_p, radius, k_max)

    # RFF bandwidth from pooled neighbor distances
    sig = sigma if sigma is not None else _rff_sigma_auto(d_t, nbr_t)
    rff = RFF1D(num_features=num_features, sigma=sig).to(d_t.device)

    def _node_sig(d: Tensor, nbr: Tensor) -> Tuple[Tensor, Tensor]:
        # d: [B,N,N], nbr: [B,N,N]
        phi = rff(d.unsqueeze(-1).squeeze(-1))  # [B,N,N,M] using broadcasting
        phi = phi * nbr.unsqueeze(-1).float()
        sigs = phi.sum(dim=2)
        counts = nbr.sum(dim=2, keepdim=True).clamp_min(1)
        sigs = sigs / counts  # [B,N,M]
        node_mask = nbr.sum(dim=2) > 0  # [B,N]
        return sigs, node_mask

    s_t, m_t = _node_sig(d_t, nbr_t)
    s_p, m_p = _node_sig(d_p, nbr_p)

    # Kernel in feature space (RBF with width tau)
    diff = s_t.unsqueeze(2) - s_p.unsqueeze(1)  # [B,N,N,M]
    d2 = diff.pow(2).sum(dim=-1)               # [B,N,N]
    sim = torch.exp(-d2 / (2.0 * (tau**2)))
    valid = m_t.unsqueeze(2) & m_p.unsqueeze(1)
    num = (sim * valid.float()).sum(dim=(1, 2))
    den = valid.sum(dim=(1, 2)).clamp_min(1)
    sims = num / den  # [B]
    if not normalize:
        return -sims.mean()

    # Self-norms
    def _self(s, m):
        d2s = (s.unsqueeze(2) - s.unsqueeze(1)).pow(2).sum(dim=-1)
        sims = torch.exp(-d2s / (2.0 * (tau**2)))
        v = m.unsqueeze(2) & m.unsqueeze(1)
        num = (sims * v.float()).sum(dim=(1, 2))
        den = v.sum(dim=(1, 2)).clamp_min(1)
        return num / den

    denom = (_self(s_t, m_t) * _self(s_p, m_p)).clamp_min(1e-12).sqrt()
    return -(sims / denom).mean()

# ---------------------------
# (C) Irreps (spherical harmonics) power spectrum moments
# ---------------------------

@dataclass
class IrrepConfig:
    L_max: int = 2               # up to Y_2
    num_radial: int = 2          # multi-scale radials
    r_max: Optional[float] = None
    radial_gamma: Optional[float] = None  # RBF width per scale (auto if None)
    k_max: Optional[int] = None  # optional top-k neighbors

def _sph_harm(unit_vecs: Tensor, L_max: int) -> Dict[int, Tensor]:
    """
    Real spherical harmonics for directions.
    unit_vecs: [B, N, N, 3] with last dim unit-normalized, nan-safe.
    Returns dict L-> [B,N,N,(2L+1)]
    """
    # Use a simple torch implementation (approx) to avoid extra deps;
    # For high accuracy, swap to cuequivariance_torch.SphericalHarmonics.
    x, y, z = unit_vecs[..., 0], unit_vecs[..., 1], unit_vecs[..., 2]
    # theta = arccos(z), phi = atan2(y,x)
    theta = torch.acos(z.clamp(-1, 1))
    phi = torch.atan2(y, x)
    out = {}
    # L=0
    if L_max >= 0:
        Y0 = 0.5**0.5 * torch.ones_like(theta)[..., None]  # real Y_00
        out[0] = Y0
    if L_max >= 1:
        # real Y_1m (m=-1,0,1)
        Y10 = torch.sqrt(torch.tensor(3.0/(4*math.pi), device=theta.device)) * torch.cos(theta)
        Y11 = torch.sqrt(torch.tensor(3.0/(4*math.pi), device=theta.device)) * torch.sin(theta) * torch.cos(phi)
        Y1m1= torch.sqrt(torch.tensor(3.0/(4*math.pi), device=theta.device)) * torch.sin(theta) * torch.sin(phi)
        out[1] = torch.stack([Y1m1, Y10, Y11], dim=-1)
    if L_max >= 2:
        # A compact real basis for L=2 (not full; acceptable for norms)
        c = torch.sqrt(torch.tensor(15.0/(4*math.pi), device=theta.device))
        s = torch.sqrt(torch.tensor(5.0/(16*math.pi), device=theta.device))
        Y2m2 = c * torch.sin(theta)**2 * torch.sin(2*phi) / 2
        Y2m1 = c * torch.sin(theta)*torch.cos(theta)*torch.sin(phi)
        Y20  = s * (3*torch.cos(theta)**2 - 1)
        Y21  = c * torch.sin(theta)*torch.cos(theta)*torch.cos(phi)
        Y22  = c * torch.sin(theta)**2 * torch.cos(2*phi) / 2
        out[2] = torch.stack([Y2m2, Y2m1, Y20, Y21, Y22], dim=-1)
    return out

def irrep_power_spectrum_loss(
    pos_pred: Tensor,
    pos_true: Tensor,
    batch_true: Tensor,
    *,
    batch_pred: Optional[Tensor] = None,
    center: bool = True,
    config: IrrepConfig = IrrepConfig(),
    tau: float = 1.0,
) -> Tensor:
    """Rotation-invariant L2 / MMD over per-node spherical-harmonics power spectra."""
    if batch_pred is None:
        batch_pred = batch_true
    x_t, mask_t, d_t, _ = _prepare_dense(pos_true, batch_true, center)
    x_p, mask_p, d_p, _ = _prepare_dense(pos_pred, batch_pred, center)

    B, N, _ = x_t.shape
    eye = torch.eye(N, device=x_t.device, dtype=torch.bool).unsqueeze(0)

    # Neighborhoods (radius or k)
    nbr_t = _build_local_neighbor_mask(mask_t, d_t, config.r_max, config.k_max)
    nbr_p = _build_local_neighbor_mask(mask_p, d_p, config.r_max, config.k_max)

    def _ps(x, d, nbr):
        # Unit directions
        row = x.unsqueeze(2).expand(-1, -1, N, -1)
        col = x.unsqueeze(1).expand(-1, N, -1, -1)
        vec = col - row  # [B,N,N,3]
        r = d.clamp_min(1e-8)
        u = vec / r.unsqueeze(-1)  # unit vectors
        sh = _sph_harm(u, config.L_max)  # dict L->[B,N,N,2L+1]

        # Radial windows
        # choose centers evenly within max radius of each batch
        if config.r_max is not None:
            rmax = config.r_max
        else:
            rmax = float(d[nbr].max()) if nbr.any() else 1.0
        S = config.num_radial
        centers = torch.linspace(0.0, rmax, S, device=x.device)
        gamma = config.radial_gamma or max(rmax / max(2*S,1), 1e-3)
        rad = torch.exp(-0.5 * ((d.unsqueeze(-1) - centers)**2) / (gamma**2))  # [B,N,N,S]

        # Accumulate per-node power spectra over neighbors and scales
        ps = []  # list of [B,N,S] per L
        for L, Y in sh.items():  # Y: [B,N,N,2L+1]
            # weighted multiplets sum over neighbors: sum_j rad * Y (mask)
            Yw = Y.unsqueeze(-1) * rad.unsqueeze(-2)  # [B,N,N,(2L+1),S]
            Yw = Yw * nbr.unsqueeze(-1).unsqueeze(-1).float()
            multiplet = Yw.sum(dim=2)  # [B,N,(2L+1),S]
            # invariant by L2 norm across m:
            ps_L = multiplet.pow(2).sum(dim=2).sqrt()  # [B,N,S]
            ps.append(ps_L)
        # concat over L: [B,N,S*(#L-terms)]
        return torch.cat(ps, dim=-1), (nbr.sum(dim=2) > 0)

    ps_t, m_t = _ps(x_t, d_t, nbr_t)
    ps_p, m_p = _ps(x_p, d_p, nbr_p)

    # Compare distributions via RBF kernel in feature space
    diff = ps_t.unsqueeze(2) - ps_p.unsqueeze(1)  # [B,N,N,D]
    d2 = diff.pow(2).sum(dim=-1)
    sim = torch.exp(-d2 / (2.0 * (tau**2)))
    valid = m_t.unsqueeze(2) & m_p.unsqueeze(1)
    score = (sim * valid.float()).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
    return -score.mean()

# ---------------------------
# (D) Combined & staged losses for iterative decoders
# ---------------------------

@dataclass
class LossWeights:
    w_global_binned: float = 0.0
    w_local_binned: float = 0.0
    w_global_rff: float = 1.0
    w_local_rff: float = 1.0
    w_irrep: float = 0.0

def combined_invariant_loss(
    pos_pred: Tensor,
    pos_true: Tensor,
    batch_true: Tensor,
    *,
    weights: LossWeights = LossWeights(),
    # configs
    kc_global_cfg: Optional[Dict] = None,
    kc_local_cfg: Optional[Dict] = None,
    rff_global_cfg: Optional[Dict] = None,
    rff_local_cfg: Optional[Dict] = None,
    irrep_cfg: Optional[IrrepConfig] = None,
    center: bool = True,
) -> Tensor:
    total = 0.0
    denom = 0.0
    if weights.w_global_binned > 0.0:
        total = total + weights.w_global_binned * global_distance_kernel_loss_pyg(
            pos_pred, pos_true, batch_true, center=center, **(kc_global_cfg or {})
        )
        denom += weights.w_global_binned
    if weights.w_local_binned > 0.0:
        total = total + weights.w_local_binned * local_distance_kernel_loss_pyg(
            pos_pred, pos_true, batch_true, center=center, **(kc_local_cfg or {})
        )
        denom += weights.w_local_binned
    if weights.w_global_rff > 0.0:
        total = total + weights.w_global_rff * global_rff_mmd_loss(
            pos_pred, pos_true, batch_true, center=center, **(rff_global_cfg or {})
        )
        denom += weights.w_global_rff
    if weights.w_local_rff > 0.0:
        total = total + weights.w_local_rff * local_rff_mmd_loss(
            pos_pred, pos_true, batch_true, center=center, **(rff_local_cfg or {})
        )
        denom += weights.w_local_rff
    if weights.w_irrep > 0.0:
        total = total + weights.w_irrep * irrep_power_spectrum_loss(
            pos_pred, pos_true, batch_true, center=center, config=(irrep_cfg or IrrepConfig())
        )
        denom += weights.w_irrep
    return total / max(denom, 1e-9)

def staged_loss_over_steps(
    seq_coords: Sequence[Tensor],  # list of [B,N,3] over T steps
    pos_true: Tensor,
    batch_true: Tensor,
    *,
    weights_schedule: Sequence[LossWeights],
    center: bool = True,
    loss_cfgs: Optional[Dict] = None,
) -> Tensor:
    """
    Apply a step-wise schedule: early weights emphasize global (coarse), later emphasize local (fine).
    """
    loss_cfgs = loss_cfgs or {}
    assert len(seq_coords) == len(weights_schedule)
    total = 0.0
    for coords, w in zip(seq_coords, weights_schedule):
        total = total + combined_invariant_loss(
            coords, pos_true, batch_true, weights=w, center=center, **loss_cfgs
        )
    return total / len(seq_coords)

__all__ = [
    "kernel_correlation_loss_pyg",
    "global_distance_kernel_loss_pyg",
    "local_distance_kernel_loss_pyg",
    "global_rff_mmd_loss",
    "local_rff_mmd_loss",
    "irrep_power_spectrum_loss",
    "LossWeights",
    "combined_invariant_loss",
    "staged_loss_over_steps",
]
