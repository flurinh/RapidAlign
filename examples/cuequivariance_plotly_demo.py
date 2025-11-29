#!/usr/bin/env python3
"""Visualize equivariant graph features with cuEquivariance and Plotly."""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Optional

import plotly.graph_objects as go
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet


@dataclass
class GraphData:
    pos: torch.Tensor
    edge_index: torch.Tensor
    node_features: Optional[torch.Tensor] = None

    @property
    def num_nodes(self) -> int:
        return int(self.pos.size(0))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-path",
        type=pathlib.Path,
        default=None,
        help="Optional path to a torch.save()-ed PyG Data object containing 'pos' and (optionally) 'edge_index'.",
    )
    parser.add_argument(
        "--html-out",
        type=pathlib.Path,
        default=pathlib.Path("outputs/cuequivariance_graph.html"),
        help="Destination for the interactive Plotly HTML file.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=64,
        help="Number of nodes to synthesize when no graph file is provided.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=6,
        help="Number of neighbors per node for the KNN graph.",
    )
    parser.add_argument(
        "--max-l",
        type=int,
        default=2,
        help="Maximum spherical harmonic degree for edge encodings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for demo graph generation.",
    )
    return parser.parse_args()


def load_graph(args: argparse.Namespace) -> GraphData:
    if args.graph_path is not None:
        obj = torch.load(args.graph_path)
        pos = getattr(obj, "pos", None)
        if pos is None:
            raise ValueError("Loaded graph object must expose a 'pos' tensor.")
        edge_index = getattr(obj, "edge_index", None)
    else:
        torch.manual_seed(args.seed)
        pos = torch.randn(args.num_nodes, 3)
        pos = pos + 0.1 * torch.randn_like(pos)
        edge_index = None

    pos = pos.detach().to(torch.float32)

    if edge_index is None:
        edge_index = build_knn_edges(pos, args.neighbors)
    edge_index = edge_index.to(torch.long)

    return GraphData(pos=pos, edge_index=edge_index)


def build_knn_edges(pos: torch.Tensor, k: int) -> torch.Tensor:
    num_nodes = pos.size(0)
    with torch.no_grad():
        dist = torch.cdist(pos, pos, p=2)
    knn = dist.topk(k + 1, largest=False).indices[:, 1:]
    src = torch.arange(num_nodes).repeat_interleave(k)
    dst = knn.reshape(-1)
    return torch.stack([src, dst], dim=0)


def spherical_irreps(ls: list[int]) -> cue.Irreps:
    terms = []
    for ell in ls:
        parity = "e" if ell % 2 == 0 else "o"
        terms.append(f"1x{ell}{parity}")
    return cue.Irreps("O3", " + ".join(terms))


def encode_graph(graph: GraphData, max_l: int) -> tuple[cue.Irreps, torch.Tensor, cue.Irreps, torch.Tensor]:
    src, dst = graph.edge_index
    rel_vecs = graph.pos[dst] - graph.pos[src]

    ls = list(range(max_l + 1))
    sh_encoder = cuet.SphericalHarmonics(ls, normalize=True)
    edge_sh = sh_encoder(rel_vecs)
    edge_irreps = spherical_irreps(ls)

    node_irreps = cue.Irreps("O3", "1x0e + 1x1o")
    scalars = graph.pos.norm(dim=1, keepdim=True)
    agg = torch.zeros(graph.num_nodes, 3, dtype=graph.pos.dtype)
    agg.index_add_(0, src, rel_vecs)
    deg = torch.bincount(src, minlength=graph.num_nodes).clamp_min(1).view(-1, 1)
    vectors = agg / deg
    features = torch.cat([scalars, vectors], dim=1)
    return node_irreps, features, edge_irreps, edge_sh


def extract_vector_block(features: torch.Tensor, irreps: cue.Irreps) -> torch.Tensor:
    chunks = []
    for (mul, ir), sl in zip(irreps, irreps.slices()):
        if getattr(ir, "l", None) != 1:
            continue
        block = features[:, sl].reshape(features.size(0), mul, ir.dim)
        chunks.append(block.mean(dim=1))
    if not chunks:
        return torch.zeros(features.size(0), 3, dtype=features.dtype)
    return torch.stack(chunks, dim=0).mean(dim=0)


def plot_graph(
    graph: GraphData,
    node_irreps: cue.Irreps,
    node_features: torch.Tensor,
    edge_irreps: cue.Irreps,
    edge_sh: torch.Tensor,
    html_path: pathlib.Path,
) -> None:
    pos = graph.pos.cpu()
    scalars = node_features[:, 0].cpu()
    vectors = extract_vector_block(node_features, node_irreps).cpu()
    edge_strength = edge_sh.pow(2).sum(dim=1).sqrt().cpu()
    deg = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes).clamp_min(1)
    edge_energy = torch.zeros(graph.num_nodes)
    edge_energy.index_add_(0, graph.edge_index[0], edge_strength)
    edge_energy = edge_energy / deg

    strength = vectors.norm(dim=-1)
    hover_text = [
        (
            f"node {idx}<br>scalar={float(s):.3f}"
            f"<br>|vector|={float(v):.3f}"
            f"<br>avg SH energy={float(e):.3f}"
        )
        for idx, (s, v, e) in enumerate(zip(scalars, strength, edge_energy))
    ]

    node_trace = go.Scatter3d(
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=scalars,
            colorscale="Viridis",
            colorbar=dict(title="l=0 scalar"),
            line=dict(width=0.5, color="black"),
        ),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
    )

    edge_x, edge_y, edge_z = [], [], []
    for u, v in graph.edge_index.t().tolist():
        edge_x += [pos[u, 0], pos[v, 0], None]
        edge_y += [pos[u, 1], pos[v, 1], None]
        edge_z += [pos[u, 2], pos[v, 2], None]
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="rgba(150,150,150,0.5)", width=1),
        hoverinfo="skip",
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.add_trace(
        go.Cone(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            u=vectors[:, 0],
            v=vectors[:, 1],
            w=vectors[:, 2],
            sizemode="absolute",
            sizeref=0.2,
            colorscale="Portland",
            showscale=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=(
            "Equivariant graph features: "
            f"node_irreps={node_irreps}, edge_irreps={edge_irreps}"
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        annotations=[
            dict(
                text=f"mean edge SH energy={edge_strength.mean():.3f}",
                x=0,
                y=1.05,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
        ],
    )

    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved Plotly visualization to {html_path}")


def main() -> None:
    args = parse_args()
    graph = load_graph(args)
    node_irreps, node_features, edge_irreps, edge_sh = encode_graph(graph, args.max_l)
    plot_graph(graph, node_irreps, node_features, edge_irreps, edge_sh, args.html_out)


if __name__ == "__main__":
    main()
