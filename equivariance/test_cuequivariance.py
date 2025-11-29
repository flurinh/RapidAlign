#!/usr/bin/env python3
"""Quick cuEquivariance sanity test."""

from __future__ import annotations

import torch

import cuequivariance as cue
import cuequivariance_torch as cuet


def main() -> None:
    torch.manual_seed(0)
    pos = torch.randn(16, 3)
    in_irreps = cue.Irreps("O3", "4x0e")
    features = torch.randn(16, in_irreps.dim)
    edge_index = torch.combinations(torch.arange(16), r=2).T

    sh = cuet.SphericalHarmonics(ls=[0, 1])
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_sh = sh(edge_vec)

    sh_irreps = cue.Irreps("O3", "0e + 1o")
    out_irreps = cue.Irreps("O3", "4x0e + 4x1o")
    layer = cuet.layers.FullyConnectedTensorProductConv(in_irreps, sh_irreps, out_irreps)
    edge_mlp = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, layer.tp.weight_numel),
    )
    edge_dist = edge_vec.norm(dim=-1, keepdim=True)
    edge_emb = edge_mlp(edge_dist)
    graph = (edge_index, (pos.size(0), pos.size(0)))
    out = layer(features, edge_sh, edge_emb, graph)
    print("cuEquivariance layer output shape:", out.shape)


if __name__ == "__main__":
    main()
