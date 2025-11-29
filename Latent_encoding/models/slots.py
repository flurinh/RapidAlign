"""Slot-attention module for pooling PyG node embeddings."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class SlotAttention(nn.Module):
    """Soft-attention pooling from node invariants to latent slots."""

    def __init__(
        self,
        num_slots: int,
        in_dim: int,
        slot_dim: int,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.slot_init = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.01)

        self.score_mlp = nn.Sequential(
            nn.Linear(slot_dim + in_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, 1),
        )
        self.node_proj = nn.Linear(in_dim, slot_dim)
        self.slot_self_attn = nn.MultiheadAttention(
            embed_dim=slot_dim,
            num_heads=attn_heads,
            batch_first=True,
        )

    def forward(
        self,
        node_invariants: torch.Tensor,
        batch: torch.Tensor,
        slots_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate node invariants into slots for each graph in the batch."""

        if node_invariants.numel() == 0:
            raise ValueError("node_invariants cannot be empty")

        B = int(batch.max().item()) + 1
        K, D = self.num_slots, self.slot_dim
        device = node_invariants.device

        if slots_prev is None:
            slots = self.slot_init.unsqueeze(0).expand(B, K, D).contiguous()
        else:
            slots = slots_prev

        pooled = []
        for graph_idx in range(B):
            mask = batch == graph_idx
            s_b = node_invariants[mask]
            if s_b.numel() == 0:
                pooled.append(slots[graph_idx])
                continue

            slot_state = slots[graph_idx]
            num_nodes = s_b.size(0)
            slot_expand = slot_state.unsqueeze(1).expand(K, num_nodes, D)
            node_expand = s_b.unsqueeze(0).expand(K, num_nodes, s_b.size(-1))
            concat = torch.cat([slot_expand, node_expand], dim=-1)

            scores = self.score_mlp(concat).squeeze(-1)
            attn = F.softmax(scores, dim=-1)
            projected = self.node_proj(s_b)
            slot_update = attn @ projected

            slot_update = slot_update.unsqueeze(0)
            slot_update, _ = self.slot_self_attn(slot_update, slot_update, slot_update)
            pooled.append(slot_update.squeeze(0))

        slots_out = torch.stack(pooled, dim=0).to(device)
        return slots_out


__all__ = ["SlotAttention"]
