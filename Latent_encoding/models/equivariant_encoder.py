"""Equivariant MPNN encoder with slot attention."""

from __future__ import annotations

from typing import List

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
from torch import nn

from .slots import SlotAttention


def _extract_invariants(features: torch.Tensor, irreps: cue.Irreps) -> torch.Tensor:
    invariants: List[torch.Tensor] = []
    offset = 0
    for mul, ir in irreps:
        block_dim = mul * ir.dim
        block = features[:, offset : offset + block_dim]
        offset += block_dim
        if ir.is_scalar():
            invariants.append(block)
        else:
            reshaped = block.view(features.size(0), mul, ir.dim)
            invariants.append(reshaped.norm(dim=-1))
    return torch.cat(invariants, dim=1)


def _invariant_dim(irreps: cue.Irreps) -> int:
    dim = 0
    for mul, ir in irreps:
        if ir.is_scalar():
            dim += mul * ir.dim
        else:
            dim += mul
    return dim


class EquivariantMPNNLayer(nn.Module):
    """Single FullyConnectedTensorProductConv with edge MLP weights."""

    def __init__(
        self,
        in_irreps: cue.Irreps,
        sh_irreps: cue.Irreps,
        out_irreps: cue.Irreps,
        mlp_hidden: int = 64,
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
            nn.Linear(1, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.conv.tp.weight_numel),
        )
        self.activation = nn.GELU()

    def forward(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)
        edge_emb = self.edge_mlp(edge_dist)
        num_nodes = pos.size(0)
        graph = (edge_index, (num_nodes, num_nodes))
        out = self.conv(node_features, edge_sh, edge_emb, graph)
        return self.activation(out)


class EquivariantBackbone(nn.Module):
    """Stack of equivariant MPNN layers producing invariant features per layer."""

    def __init__(
        self,
        in_node_dim: int,
        num_layers: int = 3,
        scalar_width: int = 4,
        vector_width: int = 4,
    ) -> None:
        super().__init__()
        self.in_irreps = cue.Irreps("O3", f"{scalar_width}x0e")
        hidden_irreps = cue.Irreps(
            "O3",
            f"{scalar_width}x0e + {vector_width}x1o",
        )
        self.layers = nn.ModuleList()
        self.sh_module = cuet.SphericalHarmonics(ls=[0, 1], normalize=True)
        self.input_proj = nn.Linear(in_node_dim, int(self.in_irreps.dim))
        current_irreps = self.in_irreps
        for _ in range(num_layers):
            layer = EquivariantMPNNLayer(
                in_irreps=current_irreps,
                sh_irreps=cue.Irreps("O3", "0e + 1o"),
                out_irreps=hidden_irreps,
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
        for layer in self.layers:
            row, col = edge_index
            edge_vec = pos[row] - pos[col]
            edge_sh = self.sh_module(edge_vec)
            node_feats = layer(node_feats, pos, edge_index, edge_sh)
            invariants.append(_extract_invariants(node_feats, layer.out_irreps))
        return invariants


class EquivariantGraphSlotEncoder(nn.Module):
    """Equivariant backbone followed by slot-attention updates per layer."""

    def __init__(
        self,
        in_node_dim: int,
        num_layers: int = 3,
        num_slots: int = 8,
        slot_dim: int = 128,
        slot_heads: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = EquivariantBackbone(
            in_node_dim=in_node_dim,
            num_layers=num_layers,
        )
        hidden_dim = self.backbone.invariant_dim
        self.slot_layers = nn.ModuleList(
            [SlotAttention(num_slots, hidden_dim, slot_dim, attn_heads=slot_heads) for _ in range(num_layers)]
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
]
