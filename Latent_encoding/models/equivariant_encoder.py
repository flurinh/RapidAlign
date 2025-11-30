# Latent_encoding/equivariant_encoder.py
"""Equivariant MPNN encoder with slot attention."""

from __future__ import annotations

from typing import List

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
from torch import nn

from .slots import SlotAttention


def _extract_invariants(features: torch.Tensor, irreps: cue.Irreps) -> torch.Tensor:
    """Extract rotation-invariant scalars from an irrep-structured feature tensor.

    - Scalar irreps (0e) are kept as-is.
    - Higher-order irreps (l>0) are collapsed by L2-norm over their (2l+1)-dimensional components,
      yielding one scalar per multiplicity.
    """
    invariants: List[torch.Tensor] = []
    offset = 0
    for mul, ir in irreps:
        block_dim = mul * ir.dim
        block = features[:, offset : offset + block_dim]
        offset += block_dim
        if ir.is_scalar():
            # scalar channels are already rotation-invariant
            invariants.append(block)
        else:
            # non-scalar channels: take norm over the (2l+1)-dimensional irrep
            reshaped = block.view(features.size(0), mul, ir.dim)
            invariants.append(reshaped.norm(dim=-1))
    return torch.cat(invariants, dim=1)


def _invariant_dim(irreps: cue.Irreps) -> int:
    """Compute the number of scalar invariants produced by _extract_invariants for given irreps."""
    dim = 0
    for mul, ir in irreps:
        if ir.is_scalar():
            dim += mul * ir.dim
        else:
            dim += mul
    return dim


class RadialBasisExpansion(nn.Module):
    """Gaussian radial basis function expansion for edge distances.

    Expands a scalar distance into a vector of Gaussian basis functions,
    providing richer geometric conditioning for the edge MLP.

    Args:
        num_basis: Number of Gaussian basis functions.
        cutoff: Maximum distance for the basis functions.
        trainable: If True, centers and widths are learnable parameters.
    """

    def __init__(
        self,
        num_basis: int = 20,
        cutoff: float = 5.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

        # Evenly spaced Gaussian centers from 0 to cutoff
        centers = torch.linspace(0.0, cutoff, num_basis)
        # Width chosen so Gaussians overlap reasonably
        widths = torch.full((num_basis,), cutoff / num_basis)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Expand distances to radial basis functions.

        Args:
            distances: Edge distances of shape [E] or [E, 1].

        Returns:
            RBF expansion of shape [E, num_basis].
        """
        if distances.dim() == 2:
            distances = distances.squeeze(-1)
        # Gaussian: exp(-((d - center) / width)^2)
        diff = distances.unsqueeze(-1) - self.centers  # [E, num_basis]
        return torch.exp(-((diff / self.widths) ** 2))


class EquivariantMPNNLayer(nn.Module):
    """Single FullyConnectedTensorProductConv with edge MLP weights.

    Args:
        in_irreps: Input irreducible representations.
        sh_irreps: Spherical harmonic irreps for edges.
        out_irreps: Output irreducible representations.
        mlp_hidden: Hidden dimension for edge MLP.
        edge_input_dim: Input dimension for edge MLP (1 for raw distance, or num_basis for RBF).
    """

    def __init__(
        self,
        in_irreps: cue.Irreps,
        sh_irreps: cue.Irreps,
        out_irreps: cue.Irreps,
        mlp_hidden: int = 64,
        edge_input_dim: int = 1,
    ) -> None:
        super().__init__()
        self.out_irreps = out_irreps
        self.conv = cuet.layers.FullyConnectedTensorProductConv(
            in_irreps,
            sh_irreps,
            out_irreps,
            batch_norm=True,
            layout=cue.ir_mul,
            mlp_channels=None,
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, self.conv.tp.weight_numel),
        )
        self.activation = nn.SiLU()

    def forward(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: Node features [N, in_irreps.dim].
            pos: Node positions [N, 3].
            edge_index: Edge indices [2, E].
            edge_sh: Spherical harmonics for edges [E, sh_irreps.dim].
            edge_embedding: Edge distance embedding [E, edge_input_dim] (raw dist or RBF).

        Returns:
            Updated node features [N, out_irreps.dim].
        """
        edge_emb = self.edge_mlp(edge_embedding)
        num_nodes = pos.size(0)
        graph = (edge_index, (num_nodes, num_nodes))
        out = self.conv(node_features, edge_sh, edge_emb, graph)
        return self.activation(out)


class EquivariantBackbone(nn.Module):
    """Stack of equivariant MPNN layers producing invariant features per layer.

    Irreps configuration:
      - scalar_width: multiplicity of l=0 (0e) channels.
      - vector_width: multiplicity of l=1 (1o) channels.
      - l2_width:    multiplicity of l=2 (2e) channels (higher-order, directional).
      - sh_lmax:     maximum spherical-harmonic degree used on edges (None -> inferred).

    Radial basis configuration:
      - use_rbf: If True, expand edge distances with Gaussian RBF before edge MLP.
      - rbf_num_basis: Number of Gaussian basis functions.
      - rbf_cutoff: Maximum distance for RBF.
      - rbf_trainable: If True, RBF centers/widths are learnable.
    """

    def __init__(
        self,
        in_node_dim: int,
        num_layers: int = 3,
        scalar_width: int = 4,
        vector_width: int = 4,
        l2_width: int = 0,
        sh_lmax: int | None = None,
        mlp_hidden: int = 64,
        # Radial basis expansion options
        use_rbf: bool = False,
        rbf_num_basis: int = 20,
        rbf_cutoff: float = 5.0,
        rbf_trainable: bool = False,
    ) -> None:
        super().__init__()
        # Irreps for node features
        self.scalar_width = scalar_width
        self.vector_width = vector_width
        self.l2_width = l2_width

        # Input: purely scalar node features
        self.in_irreps = cue.Irreps("O3", f"{scalar_width}x0e")

        # Hidden irreps: 0e + 1o (+ 2e if requested)
        hidden_pieces = [f"{scalar_width}x0e"]
        if vector_width > 0:
            hidden_pieces.append(f"{vector_width}x1o")
        if l2_width > 0:
            hidden_pieces.append(f"{l2_width}x2e")
        hidden_irreps_str = " + ".join(hidden_pieces)
        hidden_irreps = cue.Irreps("O3", hidden_irreps_str)

        # Spherical harmonics degrees for edges: up to L=1 by default, L=2 if l2_width>0,
        # or explicit sh_lmax if provided.
        if sh_lmax is not None:
            max_L = sh_lmax
        elif l2_width > 0:
            max_L = 2
        else:
            max_L = 1

        ls = list(range(max_L + 1))  # e.g. [0,1] or [0,1,2]
        self.sh_module = cuet.SphericalHarmonics(ls=ls, normalize=True)
        # Build matching irreps string for spherical harmonics
        sh_terms = []
        for l in ls:
            parity = "e" if (l % 2 == 0) else "o"
            sh_terms.append(f"{l}{parity}")
        sh_irreps_str = " + ".join(sh_terms)
        self.sh_irreps = cue.Irreps("O3", sh_irreps_str)

        # Radial basis expansion
        self.use_rbf = use_rbf
        if use_rbf:
            self.rbf = RadialBasisExpansion(
                num_basis=rbf_num_basis,
                cutoff=rbf_cutoff,
                trainable=rbf_trainable,
            )
            edge_input_dim = rbf_num_basis
        else:
            self.rbf = None
            edge_input_dim = 1

        self.layers = nn.ModuleList()
        self.input_proj = nn.Linear(in_node_dim, int(self.in_irreps.dim))
        current_irreps = self.in_irreps
        for _ in range(num_layers):
            layer = EquivariantMPNNLayer(
                in_irreps=current_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=hidden_irreps,
                mlp_hidden=mlp_hidden,
                edge_input_dim=edge_input_dim,
            )
            self.layers.append(layer)
            current_irreps = hidden_irreps

        self.output_irreps = current_irreps
        self.invariant_dim = _invariant_dim(hidden_irreps)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> List[torch.Tensor]:
        node_feats = self.input_proj(x)
        invariants: List[torch.Tensor] = []

        # Precompute edge geometry (shared across layers)
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)

        # Spherical harmonics for all requested L up to sh_lmax
        edge_sh = self.sh_module(edge_vec)

        # Edge embedding: either RBF expansion or raw distance
        if self.use_rbf:
            edge_embedding = self.rbf(edge_dist)
        else:
            edge_embedding = edge_dist

        for layer in self.layers:
            node_feats = layer(node_feats, pos, edge_index, edge_sh, edge_embedding)
            invariants.append(_extract_invariants(node_feats, layer.out_irreps))
        return invariants


class EquivariantGraphSlotEncoder(nn.Module):
    """Equivariant backbone followed by slot-attention updates per layer.

    Args:
        in_node_dim: Input node feature dimension.
        num_layers: Number of equivariant MPNN layers.
        num_slots: Number of latent slots per graph.
        slot_dim: Dimension of each slot.
        slot_heads: Number of attention heads in slot self-attention.
        scalar_width: Multiplicity of l=0 (scalar) irreps.
        vector_width: Multiplicity of l=1 (vector) irreps.
        l2_width: Multiplicity of l=2 irreps (0 to disable).
        sh_lmax: Maximum spherical harmonic degree (None = auto).
        mlp_hidden: Hidden dimension for edge MLPs.
        use_rbf: If True, use radial basis expansion for edges.
        rbf_num_basis: Number of Gaussian RBF basis functions.
        rbf_cutoff: Cutoff distance for RBF.
        rbf_trainable: If True, RBF parameters are learnable.
    """

    def __init__(
        self,
        in_node_dim: int,
        num_layers: int = 3,
        num_slots: int = 8,
        slot_dim: int = 128,
        slot_heads: int = 1,
        scalar_width: int = 4,
        vector_width: int = 4,
        l2_width: int = 0,
        sh_lmax: int | None = None,
        mlp_hidden: int = 64,
        # Radial basis options
        use_rbf: bool = False,
        rbf_num_basis: int = 20,
        rbf_cutoff: float = 5.0,
        rbf_trainable: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = EquivariantBackbone(
            in_node_dim=in_node_dim,
            num_layers=num_layers,
            scalar_width=scalar_width,
            vector_width=vector_width,
            l2_width=l2_width,
            sh_lmax=sh_lmax,
            mlp_hidden=mlp_hidden,
            use_rbf=use_rbf,
            rbf_num_basis=rbf_num_basis,
            rbf_cutoff=rbf_cutoff,
            rbf_trainable=rbf_trainable,
        )
        hidden_dim = self.backbone.invariant_dim
        self.slot_layers = nn.ModuleList(
            [
                SlotAttention(num_slots, hidden_dim, slot_dim, attn_heads=slot_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        invariants_per_layer = self.backbone(x, pos, edge_index)
        slots = None
        for invariants, slot_layer in zip(invariants_per_layer, self.slot_layers):
            slots = slot_layer(invariants, batch=batch, slots_prev=slots)
        return slots


__all__ = [
    "EquivariantGraphSlotEncoder",
    "RadialBasisExpansion",
]