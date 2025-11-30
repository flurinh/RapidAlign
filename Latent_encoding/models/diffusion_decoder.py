from __future__ import annotations

import math
from typing import Optional, List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous or discrete values."""

    def __init__(self, dim: int, max_value: float = 1000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-math.log(self.max_value) / max(half - 1, 1))
        )
        args = x.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions for distance encoding."""

    def __init__(
            self,
            num_basis: int = 20,
            cutoff: float = 5.0,
            trainable: bool = False,
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

        centers = torch.linspace(0.0, cutoff, num_basis)
        widths = torch.full((num_basis,), cutoff / num_basis)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    def forward(self, distances: Tensor) -> Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-((diff / self.widths) ** 2))


class EdgeEncoder(nn.Module):
    """Encode edges from distances and optional direction vectors."""

    def __init__(
            self,
            hidden_dim: int,
            rbf_basis: int = 20,
            rbf_cutoff: float = 5.0,
            use_direction: bool = True,
    ) -> None:
        super().__init__()
        self.use_direction = use_direction
        self.rbf = GaussianRBF(rbf_basis, rbf_cutoff, trainable=False)

        in_dim = rbf_basis + (3 if use_direction else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
            self,
            distances: Tensor,
            directions: Optional[Tensor] = None,
    ) -> Tensor:
        rbf_feats = self.rbf(distances)
        if self.use_direction and directions is not None:
            edge_feats = torch.cat([rbf_feats, directions], dim=-1)
        else:
            edge_feats = rbf_feats
        return self.mlp(edge_feats)


class SlotCrossAttention(nn.Module):
    """Cross-attention where nodes query slots for structural context.

    Each node attends to all K slots, learning to extract the structural
    information most relevant for its current state. This allows different
    nodes to focus on different structural aspects encoded in the slots.

    Args:
        node_dim: Dimension of node state vectors
        slot_dim: Dimension of slot vectors
        hidden_dim: Internal projection dimension
        num_heads: Number of attention heads
        dropout: Attention dropout rate
    """

    def __init__(
            self,
            node_dim: int,
            slot_dim: int,
            hidden_dim: int,
            num_heads: int = 4,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Node states → queries
        self.q_proj = nn.Linear(node_dim, hidden_dim)
        # Slots → keys and values
        self.k_proj = nn.Linear(slot_dim, hidden_dim)
        self.v_proj = nn.Linear(slot_dim, hidden_dim)
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, node_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            node_states: Tensor,
            slots: Tensor,
            batch_idx: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            node_states: [B*N, node_dim] flattened node states
            slots: [B, K, slot_dim] slot latents
            batch_idx: [B*N] batch index for each node (for proper attention masking)

        Returns:
            context: [B*N, node_dim] slot context for each node
            attn_weights: [B*N, K] attention weights (for interpretability)
        """
        B, K, _ = slots.shape
        N_total = node_states.size(0)
        N = N_total // B  # nodes per graph

        # Project queries from node states [B*N, hidden_dim]
        q = self.q_proj(node_states)

        # Project keys and values from slots [B, K, hidden_dim]
        k = self.k_proj(slots)
        v = self.v_proj(slots)

        # Reshape for multi-head attention
        # q: [B*N, num_heads, head_dim] -> need to align with slots per batch
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, K, head_dim]
        v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, K, head_dim]

        # Attention: [B, heads, N, K]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: [B, heads, N, head_dim]
        context = torch.matmul(attn_weights, v)

        # Reshape back: [B, N, hidden_dim]
        context = context.transpose(1, 2).contiguous().view(B, N, -1)

        # Output projection and flatten: [B*N, node_dim]
        context = self.out_proj(context).view(B * N, -1)

        # Return mean attention across heads for interpretability: [B*N, K]
        attn_for_viz = attn_weights.mean(dim=1).view(B * N, K)

        return context, attn_for_viz


class MessagePassingLayer(nn.Module):
    """Message passing layer with edge features and per-node context."""

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            hidden_dim: int,
            context_dim: int,
    ) -> None:
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(
            self,
            node_feats: Tensor,
            edge_index: Tensor,
            edge_feats: Tensor,
            context: Tensor,
    ) -> Tensor:
        """
        Args:
            node_feats: [N, node_dim]
            edge_index: [2, E]
            edge_feats: [E, edge_dim]
            context: [N, context_dim] per-node context from slot cross-attention

        Returns:
            [N, node_dim] updated node features
        """
        row, col = edge_index
        num_nodes = node_feats.size(0)

        src_feats = node_feats[row]
        src_context = context[row]
        msg_input = torch.cat([src_feats, edge_feats, src_context], dim=-1)
        messages = self.msg_mlp(msg_input)

        # Mean aggregation
        agg = torch.zeros(num_nodes, messages.size(-1), device=messages.device)
        count = torch.zeros(num_nodes, 1, device=messages.device)
        agg.index_add_(0, col, messages)
        count.index_add_(0, col, torch.ones_like(col, dtype=torch.float).unsqueeze(-1))
        agg = agg / count.clamp(min=1)

        update_input = torch.cat([node_feats, agg], dim=-1)
        return node_feats + self.update_mlp(update_input)


class StateSpaceDecoder(nn.Module):
    """State-space decoder with slot cross-attention.

    Instead of projecting slots to a global context, each node attends to
    the slots to get node-specific structural information. This allows:

    1. Slots to specialize in different structural aspects
    2. Nodes to query relevant structural info for their refinement
    3. Interpretable attention patterns showing what each node uses
    4. Faster convergence (fewer steps needed)

    Args:
        num_slots: Number of latent slots (K)
        slot_dim: Dimension of each slot
        num_nodes: Maximum number of nodes per graph
        state_dim: Dimension of node state vectors
        hidden_dim: Hidden dimension for MLPs
        num_mp_layers: Number of message passing layers per step
        num_attn_heads: Number of heads in slot cross-attention
        steps: Number of refinement steps
        step_size: Multiplier for state updates
        knn_k: Number of neighbors for dynamic graph
        rbf_basis: Number of Gaussian RBF basis functions
        rbf_cutoff: Cutoff for RBF
        use_direction: Whether to use direction vectors in edge encoding
        init_std: Std for state template initialization
        return_sequence: If True, return all intermediate xyz projections
        return_attention: If True, also return attention weights
    """

    def __init__(
            self,
            num_slots: int,
            slot_dim: int,
            num_nodes: int,
            state_dim: int = 128,
            hidden_dim: int = 256,
            num_mp_layers: int = 2,
            num_attn_heads: int = 4,
            steps: int = 8,
            step_size: float = 0.5,
            knn_k: int = 8,
            rbf_basis: int = 20,
            rbf_cutoff: float = 5.0,
            use_direction: bool = True,
            init_std: float = 0.1,
            return_sequence: bool = False,
            return_attention: bool = False,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.step_size = step_size
        self.knn_k = knn_k
        self.return_sequence = return_sequence
        self.return_attention = return_attention

        # Learnable state template [num_nodes, state_dim]
        self.state_template = nn.Parameter(torch.randn(num_nodes, state_dim) * init_std)

        # Geometry projection: state → xyz proxy (for edge construction)
        self.geometry_proj = nn.Linear(state_dim, 3)

        # Coordinate init embedding: if coords_init provided, embed into state space
        self.coord_embed = nn.Linear(3, state_dim)

        # Time embedding (added to node states before cross-attention)
        self.time_embed = SinusoidalEmbedding(state_dim)

        # Slot cross-attention: nodes query slots for structural context
        self.slot_cross_attn = SlotCrossAttention(
            node_dim=state_dim,
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attn_heads,
            dropout=0.0,
        )

        # Edge encoder
        self.edge_encoder = EdgeEncoder(
            hidden_dim=hidden_dim,
            rbf_basis=rbf_basis,
            rbf_cutoff=rbf_cutoff,
            use_direction=use_direction,
        )

        # Message passing layers
        # Context dim is now state_dim (from cross-attention) + state_dim (time)
        context_dim = state_dim + state_dim
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(
                node_dim=state_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                context_dim=context_dim,
            )
            for _ in range(num_mp_layers)
        ])

        # State update head
        self.state_delta_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Final coordinate projection
        self.coord_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Zero-initialize delta head for stable start
        nn.init.zeros_(self.state_delta_head[-1].weight)
        nn.init.zeros_(self.state_delta_head[-1].bias)

    def _build_knn_graph(
            self,
            xyz: Tensor,
            k: int,
            mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Build kNN graph from xyz coordinates."""
        B, N, _ = xyz.shape
        device = xyz.device

        dist_matrix = torch.cdist(xyz, xyz)

        if mask is not None:
            invalid = ~mask
            dist_matrix = dist_matrix.masked_fill(
                invalid.unsqueeze(1) | invalid.unsqueeze(2),
                float('inf')
            )
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        dist_matrix = dist_matrix.masked_fill(eye, float('inf'))

        k_actual = min(k, N - 1)
        if k_actual <= 0:
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, 3, device=device),
            )

        _, knn_idx = dist_matrix.topk(k_actual, dim=-1, largest=False)

        batch_offset = torch.arange(B, device=device).view(B, 1, 1) * N
        src = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k_actual)
        src = (src + batch_offset).reshape(-1)
        dst = (knn_idx + batch_offset).reshape(-1)

        if mask is not None:
            mask_flat = mask.reshape(-1)
            valid = mask_flat[src] & mask_flat[dst]
            src = src[valid]
            dst = dst[valid]

        edge_index = torch.stack([src, dst], dim=0)

        xyz_flat = xyz.reshape(-1, 3)
        edge_vec = xyz_flat[dst] - xyz_flat[src]
        distances = edge_vec.norm(dim=-1)
        directions = edge_vec / distances.unsqueeze(-1).clamp(min=1e-8)

        return edge_index, distances, directions

    def forward(
            self,
            slots: Tensor,
            coords_init: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            num_steps: Optional[int] = None,
    ) -> Tensor | Tuple[Tensor, List[Tensor]] | List[Tensor]:
        """
        Args:
            slots: [B, K, slot_dim] latent slots
            coords_init: [B, N, 3] optional starting coordinates
            mask: [B, N] boolean mask of valid nodes
            num_steps: optional override for number of refinement steps

        Returns:
            coords: [B, N, 3] final coordinates
            attn_weights: list of [B*N, K] per step (if return_attention=True)
            sequence: list of [B, N, 3] per step (if return_sequence=True)
        """
        B = slots.size(0)
        device = slots.device
        steps = num_steps if num_steps is not None else self.steps

        # Initialize state from template
        state = self.state_template.unsqueeze(0).expand(B, -1, -1).clone()

        # If coords_init provided, embed and add to initial state
        if coords_init is not None:
            coord_state = self.coord_embed(coords_init)
            state = state + coord_state

        sequence = []
        attention_weights = []

        for step in range(steps):
            # Time embedding [B, N, state_dim]
            t = torch.full((B,), step / max(steps - 1, 1), device=device)
            time_emb = self.time_embed(t)  # [B, state_dim]
            time_emb = time_emb.unsqueeze(1).expand(B, self.num_nodes, -1)  # [B, N, state_dim]

            # Project state → xyz proxy for edge construction
            xyz_proxy = self.geometry_proj(state)

            # Build dynamic kNN graph
            edge_index, distances, directions = self._build_knn_graph(
                xyz_proxy, self.knn_k, mask
            )

            if edge_index.size(1) == 0:
                if self.return_sequence:
                    sequence.append(self.coord_head(state))
                continue

            # Encode edges
            edge_feats = self.edge_encoder(distances, directions)

            # Flatten state for cross-attention and message passing
            state_flat = state.reshape(-1, self.state_dim)

            # Cross-attention: nodes query slots [B*N, state_dim], [B*N, K]
            slot_context, attn = self.slot_cross_attn(state_flat, slots)

            if self.return_attention:
                attention_weights.append(attn.detach())

            # Combine slot context with time embedding
            time_flat = time_emb.reshape(-1, self.state_dim)
            context = torch.cat([slot_context, time_flat], dim=-1)  # [B*N, 2*state_dim]

            # Message passing
            for mp_layer in self.mp_layers:
                state_flat = mp_layer(state_flat, edge_index, edge_feats, context)

            # Predict state delta
            delta = self.state_delta_head(state_flat)
            delta = delta.reshape(B, self.num_nodes, self.state_dim)

            # Apply mask
            if mask is not None:
                delta = delta * mask.unsqueeze(-1).float()

            # Update state
            state = state + self.step_size * delta

            if self.return_sequence:
                sequence.append(self.coord_head(state))

        # Final projection to coordinates
        coords = self.coord_head(state)

        if mask is not None:
            coords = coords * mask.unsqueeze(-1).float()

        # Return based on flags
        if self.return_sequence and self.return_attention:
            return sequence, attention_weights
        elif self.return_sequence:
            return sequence
        elif self.return_attention:
            return coords, attention_weights
        return coords


# Alias for backwards compatibility
CoordinateDecoder = StateSpaceDecoder

__all__ = ["StateSpaceDecoder", "CoordinateDecoder", "SlotCrossAttention", "SinusoidalEmbedding", "GaussianRBF"]