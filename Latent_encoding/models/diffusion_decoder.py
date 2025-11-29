"""Diffusion decoder that maps latent slots to point clouds."""

from __future__ import annotations

import torch
from torch import nn


class DiffusionDecoder(nn.Module):
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        num_nodes: int,
        hidden_dim: int = 256,
        steps: int = 30,
        step_size: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_nodes = num_nodes
        self.steps = steps
        self.step_size = step_size

        self.slot_embed = nn.Sequential(
            nn.Linear(num_slots * slot_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_embed = nn.Embedding(num_nodes, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, slots: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        device = slots.device
        B = slots.size(0)
        steps = num_steps or self.steps

        slot_cond = self.slot_embed(slots.reshape(B, -1))  # [B, hidden]
        node_idx = torch.arange(self.num_nodes, device=device)
        node_feat = self.node_embed(node_idx)  # [N, hidden]
        node_feat = node_feat.unsqueeze(0).expand(B, -1, -1)  # [B, N, hidden]

        coords = torch.zeros(B, self.num_nodes, 3, device=device)
        for _ in range(steps):
            cond = torch.cat([node_feat, slot_cond.unsqueeze(1).expand_as(node_feat)], dim=-1)
            delta = self.mlp(cond)
            coords = coords + self.step_size * delta
        return coords


__all__ = ["DiffusionDecoder"]
