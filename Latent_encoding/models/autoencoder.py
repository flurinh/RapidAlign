# Latent_encoding/models/autoencoder.py
"""Graph autoencoder with equivariant encoder and state-space decoder."""

from __future__ import annotations
from typing import Optional, List

import torch
from torch import nn

from .coordinate_decoder import StateSpaceDecoder
from .equivariant_encoder import EquivariantGraphSlotEncoder


class GraphAutoencoder(nn.Module):
    """
    Equivariant slot encoder + state-space iterative decoder with cross-attention.

    Encoder:
      - sees the *clean* sparse PyG graph G_0 = (x, pos, edge_index, batch)
      - outputs K slots per graph: Z = [B, K, slot_dim]

    Decoder:
      - operates on rich state vectors (positional encodings)
      - uses cross-attention: each node queries slots for structural context
      - uses learned xyz projection for spatial adjacency (kNN edges)
      - refines states iteratively, conditioned on slots Z
      - optionally incorporates noised coordinates into initial state

    Irreps configuration for the encoder:
      - scalar_width, vector_width, l2_width, sh_lmax
      - use_rbf, rbf_num_basis, rbf_cutoff, rbf_trainable

    Decoder configuration:
      - state_dim: dimension of node state vectors
      - decoder_mp_layers: message passing layers per step
      - decoder_attn_heads: attention heads for slot cross-attention
      - decoder_rbf_basis, decoder_rbf_cutoff: edge encoding
      - decoder_use_direction: include direction vectors in edge features
    """

    def __init__(
        self,
        num_nodes: int,
        in_node_dim: int,
        num_layers: int = 3,
        num_slots: int = 8,
        slot_dim: int = 128,
        slot_attn_heads: int = 1,
        # Decoder configuration
        decoder_state_dim: int = 128,
        decoder_hidden: int = 256,
        decoder_steps: int = 8,
        decoder_step_size: float = 0.5,
        decoder_knn_k: int = 8,
        decoder_mp_layers: int = 2,
        decoder_attn_heads: int = 4,
        decoder_rbf_basis: int = 20,
        decoder_rbf_cutoff: float = 5.0,
        decoder_use_direction: bool = True,
        decoder_init_std: float = 0.1,
        return_sequence: bool = False,
        return_attention: bool = False,
        # Encoder irreps configuration
        scalar_width: int = 4,
        vector_width: int = 4,
        l2_width: int = 0,
        sh_lmax: int | None = None,
        mlp_hidden: int = 64,
        # Encoder radial basis configuration
        use_rbf: bool = False,
        rbf_num_basis: int = 20,
        rbf_cutoff: float = 5.0,
        rbf_trainable: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EquivariantGraphSlotEncoder(
            in_node_dim=in_node_dim,
            num_layers=num_layers,
            num_slots=num_slots,
            slot_dim=slot_dim,
            slot_heads=slot_attn_heads,
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
        self.decoder = StateSpaceDecoder(
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_nodes=num_nodes,
            state_dim=decoder_state_dim,
            hidden_dim=decoder_hidden,
            num_mp_layers=decoder_mp_layers,
            num_attn_heads=decoder_attn_heads,
            steps=decoder_steps,
            step_size=decoder_step_size,
            knn_k=decoder_knn_k,
            rbf_basis=decoder_rbf_basis,
            rbf_cutoff=decoder_rbf_cutoff,
            use_direction=decoder_use_direction,
            init_std=decoder_init_std,
            return_sequence=return_sequence,
            return_attention=return_attention,
        )
        self.return_sequence = return_sequence
        self.return_attention = return_attention

    def forward(
        self,
        data,
        coords_init: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_slots: bool = False,
    ):
        """
        Args:
            data: PyG Data(batch) with x, pos, edge_index, batch for the *clean* graph G_0
            coords_init: [B, N, 3] optional starting coordinates (e.g., noised).
                         If provided, embedded and added to initial state.
            mask: [B, N] boolean mask of valid nodes (from to_dense_batch)
            num_steps: Optional override for number of refinement steps
            return_slots: If True, also return the slot latents

        Returns:
            coords: [B, N, 3] or list of [B, N, 3] if return_sequence
            slots: [B, K, slot_dim] (only if return_slots=True)
        """
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        slots = self.encoder(x, pos, edge_index, batch=batch)  # [B, K, slot_dim]

        out = self.decoder(
            slots,
            coords_init=coords_init,
            mask=mask,
            num_steps=num_steps,
        )

        if return_slots:
            return out, slots
        return out

    @torch.no_grad()
    def encode(self, data) -> torch.Tensor:
        """Encode a graph batch to slot latents."""
        return self.encoder(data.x, data.pos, data.edge_index, data.batch)


__all__ = ["GraphAutoencoder"]