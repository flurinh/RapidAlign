"""Graph autoencoder composed of encoder + decoder."""

from __future__ import annotations

from torch import nn

from .diffusion_decoder import DiffusionDecoder
from .equivariant_encoder import EquivariantGraphSlotEncoder


class GraphAutoencoder(nn.Module):
    """Equivariant slot encoder + diffusion decoder."""

    def __init__(
        self,
        num_nodes: int,
        in_node_dim: int,
        num_layers: int = 3,
        num_slots: int = 8,
        slot_dim: int = 128,
        slot_attn_heads: int = 1,
        diffusion_hidden: int = 256,
        diffusion_steps: int = 30,
        diffusion_step_size: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = EquivariantGraphSlotEncoder(
            in_node_dim=in_node_dim,
            num_layers=num_layers,
            num_slots=num_slots,
            slot_dim=slot_dim,
            slot_heads=slot_attn_heads,
        )
        self.decoder = DiffusionDecoder(
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_nodes=num_nodes,
            hidden_dim=diffusion_hidden,
            steps=diffusion_steps,
            step_size=diffusion_step_size,
        )

    def forward(self, data, return_slots: bool = False):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        slots = self.encoder(x, pos, edge_index, batch=batch)
        recon = self.decoder(slots)
        if return_slots:
            return recon, slots
        return recon

    def encode(self, data) -> torch.Tensor:
        return self.encoder(data.x, data.pos, data.edge_index, data.batch)


__all__ = ["GraphAutoencoder"]
