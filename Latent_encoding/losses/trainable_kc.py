# Latent_encoding/losses/trainable_kc.py
"""
Parameterized KC Loss (Fast Calibrated MMD).

Key features:
1. Analytic Kernel Embeddings: "Fingerprints" geometry using distributions of pairwise distances.
2. Multi-Scale: Captures both local (high freq) and global (low freq) structure.
3. Calibrated: A lightweight MLP maps kernel differences to exact geometric severity (L_true).
4. Fast: Pure matrix operations (no soft-chamfer or O(N^2) loops).

Optional: Can accept edge_index to compute local features from explicit topology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj


# ==============================================================================
# Exact severity (Ground Truth)
# ==============================================================================

def compute_exact_severity(
        pos_clean: Tensor,
        pos_noisy: Tensor,
        alpha_node: float = 0.5,
        beta_edge: float = 0.5,
        eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """Compute exact SE(3)-invariant severity given known correspondence."""
    N = pos_clean.size(0)
    if N <= 1:
        return 0.0, 0.0, 0.0

    with torch.no_grad():
        d_clean = torch.cdist(pos_clean, pos_clean)
        d_noisy = torch.cdist(pos_noisy, pos_noisy)

        triu = torch.triu(torch.ones_like(d_clean, dtype=torch.bool), diagonal=1)
        d_clean_vec = d_clean[triu]
        d_noisy_vec = d_noisy[triu]

        if d_clean_vec.numel() == 0:
            return 0.0, 0.0, 0.0

        s2 = d_clean_vec.pow(2).mean().clamp_min(eps)

        dx = pos_noisy - pos_clean
        E_node = dx.pow(2).sum(dim=-1).mean()
        L_node = (E_node / s2).item()

        diff = d_noisy_vec - d_clean_vec
        E_edge = diff.pow(2).mean()
        L_edge = (E_edge / s2).item()

        L_true = alpha_node * L_node + beta_edge * L_edge

    return float(L_true), float(L_node), float(L_edge)


def compute_exact_severity_batch(
        pos_clean: Tensor,
        pos_noisy: Tensor,
        batch: Tensor,
        alpha_node: float = 0.5,
        beta_edge: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Batched version for PyG graphs."""
    device = pos_clean.device
    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

    L_true = torch.zeros(B, device=device)
    L_node = torch.zeros(B, device=device)
    L_edge = torch.zeros(B, device=device)

    for b in range(B):
        mask = batch == b
        if mask.sum() > 0:
            lt, ln, le = compute_exact_severity(
                pos_clean[mask], pos_noisy[mask], alpha_node, beta_edge
            )
            L_true[b] = lt
            L_node[b] = ln
            L_edge[b] = le

    return L_true, L_node, L_edge


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class ParameterizedKCConfig:
    """
    Configuration for Fast Calibrated KC loss.
    """
    # Number of Gaussian scales to fingerprint the geometry
    num_scales: int = 32

    # Range of sigmas (distance scales) to capture
    # Small sigma = local neighbors, Large sigma = global shape
    sigma_range: Tuple[float, float] = (0.1, 10.0)

    # k-NN for local view (only used if edge_index not provided)
    k_neighbors: int = 16

    # MLP hidden dimension for calibration
    mlp_hidden: int = 64

    # Whether to learn the mixing weights/calibration
    learnable: bool = True

    # Whether to include local (edge-based) features
    use_local_features: bool = False


# ==============================================================================
# Main Loss Class
# ==============================================================================

class ParameterizedKCLoss(nn.Module):
    """
    Fast, Calibrated Kernel Correlation Loss.

    This replaces the slow 'estimator' network with an analytic Multi-Scale MMD
    approach followed by a small calibration MLP.

    Can optionally leverage decoder's edge_index for local features.
    """

    def __init__(self, config: ParameterizedKCConfig = ParameterizedKCConfig()):
        super().__init__()
        self.config = config

        # --- Analytic Kernel Basis ---
        # Log-spaced sigmas to cover both fine details and global structure
        sigmas = torch.exp(torch.linspace(
            math.log(config.sigma_range[0]),
            math.log(config.sigma_range[1]),
            config.num_scales
        ))
        self.register_buffer("sigmas", sigmas)

        # --- Feature dimension ---
        # Global features: num_scales
        # Local features (optional): num_scales
        self.feature_dim = config.num_scales
        if config.use_local_features:
            self.feature_dim += config.num_scales

        # --- Calibration MLP ---
        if config.learnable:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, config.mlp_hidden),
                nn.LayerNorm(config.mlp_hidden),
                nn.GELU(),
                nn.Linear(config.mlp_hidden, config.mlp_hidden),
                nn.GELU(),
                nn.Linear(config.mlp_hidden, 1),
                nn.Softplus()  # Severity must be positive
            )

            # Initialize last layer small for stable training
            nn.init.uniform_(self.mlp[-2].weight, -0.01, 0.01)
            if hasattr(self.mlp[-2], 'bias') and self.mlp[-2].bias is not None:
                nn.init.zeros_(self.mlp[-2].bias)
        else:
            self.mlp = None

    def _get_kernel_embedding(
            self,
            pos: Tensor,
            mask: Tensor,
            local_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Computes the 'Fingerprint' (Kernel Mean Embedding) of the geometry.

        Args:
            pos: [B, N_max, 3] Dense node positions
            mask: [B, N_max] Boolean mask
            local_mask: [B, N_max, N_max] Optional mask for local features (from edge_index)

        Returns:
            embedding: [B, num_scales] or [B, 2*num_scales] if local features enabled
        """
        # [B, N, N] Pairwise Euclidean Distances
        dist = torch.cdist(pos, pos)

        # Handle Masking
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)

        # Exclude self-loops
        B, N, _ = dist.shape
        eye = torch.eye(N, device=dist.device).unsqueeze(0)
        valid_pairs = mask_2d.float() * (1.0 - eye)

        # Reshape for broadcasting
        dist_expanded = dist.unsqueeze(-1)  # [B, N, N, 1]
        sigmas_expanded = self.sigmas.view(1, 1, 1, -1)  # [1, 1, 1, num_scales]

        # Apply Gaussian Kernels: exp(-d^2 / (2 * sigma^2))
        k_vals = torch.exp(-dist_expanded.pow(2) / (2 * sigmas_expanded.pow(2)))

        # Zero out invalid pairs
        k_vals_global = k_vals * valid_pairs.unsqueeze(-1)

        # Sum and normalize for global features
        sum_k = k_vals_global.sum(dim=(1, 2))  # [B, num_scales]
        num_valid = valid_pairs.sum(dim=(1, 2)).clamp_min(1.0).unsqueeze(-1)  # [B, 1]
        global_embedding = sum_k / num_valid  # [B, num_scales]

        if not self.config.use_local_features or local_mask is None:
            return global_embedding

        # Compute local features using provided mask
        k_vals_local = k_vals * local_mask.unsqueeze(-1) * valid_pairs.unsqueeze(-1)
        sum_k_local = k_vals_local.sum(dim=(1, 2))
        num_local = (local_mask * valid_pairs).sum(dim=(1, 2)).clamp_min(1.0).unsqueeze(-1)
        local_embedding = sum_k_local / num_local

        return torch.cat([global_embedding, local_embedding], dim=-1)

    def forward(
            self,
            pos_a: Tensor,
            pos_b: Tensor,
            batch: Tensor,
            edge_index_a: Optional[Tensor] = None,
            edge_index_b: Optional[Tensor] = None,
            return_components: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Predicts severity between pos_a and pos_b.

        Args:
            pos_a: [N_total, 3] Clean/reference positions
            pos_b: [N_total, 3] Noisy/predicted positions
            batch: [N_total] Batch indices
            edge_index_a: Optional [2, E] edges for graph A (for local features)
            edge_index_b: Optional [2, E] edges for graph B
            return_components: If True, also return feature details
        """
        # Convert to dense for fast vectorized operations
        x_a, mask_a = to_dense_batch(pos_a, batch)
        x_b, mask_b = to_dense_batch(pos_b, batch)

        B, N_max, _ = x_a.shape

        # Build local masks from edge_index if provided
        local_mask_a = None
        local_mask_b = None

        if self.config.use_local_features:
            if edge_index_a is not None:
                adj_a = to_dense_adj(edge_index_a, batch=batch, max_num_nodes=N_max)
                local_mask_a = adj_a.float()
            if edge_index_b is not None:
                adj_b = to_dense_adj(edge_index_b, batch=batch, max_num_nodes=N_max)
                local_mask_b = adj_b.float()

        # Compute Fingerprints
        emb_a = self._get_kernel_embedding(x_a, mask_a, local_mask_a)
        emb_b = self._get_kernel_embedding(x_b, mask_b, local_mask_b)

        # Compute Difference (MMD Squared)
        diff = (emb_a - emb_b).pow(2)  # [B, feature_dim]

        # Calibrate
        if self.mlp is not None:
            severity = self.mlp(diff).squeeze(-1)  # [B]
        else:
            severity = diff.mean(dim=1)

        if return_components:
            return severity, {
                "features": diff,
                "emb_a": emb_a,
                "emb_b": emb_b
            }

        return severity

    def training_loss(
            self,
            pos_clean: Tensor,
            pos_noisy: Tensor,
            batch: Tensor,
            edge_index: Optional[Tensor] = None,
            alpha_node: float = 0.5,
            beta_edge: float = 0.5,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute training loss to calibrate the MLP against exact severity.
        """
        # Compute Ground Truth (Exact)
        with torch.no_grad():
            L_true, L_node, L_edge = compute_exact_severity_batch(
                pos_clean, pos_noisy, batch, alpha_node, beta_edge
            )

        # Predict Severity
        pred_severity, components = self.forward(
            pos_clean, pos_noisy, batch,
            edge_index_a=edge_index,
            edge_index_b=edge_index,
            return_components=True
        )

        # Loss (MSE between prediction and truth)
        loss = F.mse_loss(pred_severity, L_true)

        metrics = {
            "loss": loss,
            "L_true_mae": (pred_severity - L_true).abs().mean(),
            "pred_mean": pred_severity.mean(),
            "true_mean": L_true.mean(),
        }

        return loss, metrics

    def as_loss(
            self,
            pos_pred: Tensor,
            pos_true: Tensor,
            batch: Tensor,
            edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        """Use as loss function (returns mean predicted severity)."""
        return self.forward(
            pos_pred, pos_true, batch,
            edge_index_a=edge_index,
            edge_index_b=edge_index,
        ).mean()


__all__ = [
    "ParameterizedKCLoss",
    "ParameterizedKCConfig",
    "compute_exact_severity",
    "compute_exact_severity_batch",
]