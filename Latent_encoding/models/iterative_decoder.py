"""Iterative slot-conditioned refinement decoder."""

from __future__ import annotations

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / max(half - 1, 1))
        )
        args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class IterativeDecoder(nn.Module):
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        num_nodes: int,
        state_dim: int = 128,
        hidden_dim: int = 256,
        timesteps: int = 30,
        step_size: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.timesteps = timesteps
        self.step_size = step_size

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.latent_embed = nn.Sequential(
            nn.Linear(num_slots * slot_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        input_dim = num_nodes * state_dim
        self.refiner = nn.Sequential(
            nn.Linear(input_dim + hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.node_template = nn.Parameter(torch.randn(num_nodes, state_dim))
        self.coord_head = nn.Linear(state_dim, 3)

    def _delta(self, state: torch.Tensor, slots: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = state.size(0)
        latent = self.latent_embed(slots.reshape(B, -1))
        time_emb = self.time_embed(t).view(B, -1)
        state_flat = state.reshape(B, -1)
        h = torch.cat([state_flat, latent, time_emb], dim=-1)
        delta = self.refiner(h)
        return delta.reshape(B, self.num_nodes, self.state_dim)

    def forward(self, slots: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        B = slots.size(0)
        steps = num_steps or self.timesteps
        state = self.node_template.unsqueeze(0).expand(B, -1, -1)
        for step in range(steps):
            t = torch.full((B,), step, device=slots.device, dtype=torch.long)
            state = state + self.step_size * self._delta(state, slots, t)
        coords = self.coord_head(state)
        return coords


__all__ = ["IterativeDecoder"]
