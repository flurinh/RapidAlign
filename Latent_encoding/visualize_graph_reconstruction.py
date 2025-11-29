"""Visualize autoencoder reconstructions with per-graph losses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from torch_geometric.loader import DataLoader

import sys

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    from Latent_encoding.data import QM9PointCloudDataset, SyntheticPointCloudDataset  # type: ignore
    from Latent_encoding.losses import kernel_correlation_loss_pyg  # type: ignore
    from Latent_encoding.models import GraphAutoencoder  # type: ignore
    from Latent_encoding.utils import apply_config, kabsch_align  # type: ignore
else:
    from .data import QM9PointCloudDataset, SyntheticPointCloudDataset
    from .losses import kernel_correlation_loss_pyg
    from .models import GraphAutoencoder
    from .utils import apply_config, kabsch_align


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON config file.")
    parser.add_argument("--checkpoint", type=Path, default=Path("Latent_encoding/best_ae.pt"), help="Path to AE weights.")
    parser.add_argument("--num-graphs", type=int, default=10, help="Number of synthetic graphs to visualize.")
    parser.add_argument("--num-nodes", type=int, default=16, help="Nodes per synthetic graph.")
    parser.add_argument("--device", type=str, default="cuda", help="torch device.")
    parser.add_argument("--output", type=Path, default=Path("Latent_encoding/ae_recon_vis.html"), help="Output HTML path.")
    parser.add_argument("--num-layers", type=int, default=3, help="Encoder layers (match training).")
    parser.add_argument("--num-slots", type=int, default=8, help="Latent slots (match training).")
    parser.add_argument("--slot-dim", type=int, default=128, help="Slot dimensionality (match training).")
    parser.add_argument("--slot-heads", type=int, default=1, help="Slot-attention heads (match training).")
    parser.add_argument("--diff-hidden", type=int, default=256, help="Decoder hidden size.")
    parser.add_argument("--diff-steps", type=int, default=30, help="Decoder refinement steps.")
    parser.add_argument("--diff-step-size", type=float, default=1.0, help="Step size.")
    parser.add_argument("--kc-lambda-global", type=float, default=1.0, help="Weight for global kernel term.")
    parser.add_argument("--kc-lambda-local", type=float, default=1.0, help="Weight for local kernel term.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic graphs.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=("synthetic", "qm9"),
        help="Dataset to visualize (can also be set via config).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "test"),
        help="Split to draw from when using the QM9 dataset.",
    )
    args = parser.parse_args()
    args = apply_config(args, parser)
    return args


def build_dataset(args: argparse.Namespace):
    if args.dataset == "synthetic":
        return SyntheticPointCloudDataset(
            num_graphs=args.num_graphs,
            num_nodes=args.num_nodes,
            num_node_features=getattr(args, "num_node_features", 1),
            seed=getattr(args, "seed", 0),
            feature_mode=getattr(args, "feature_mode", "ones"),
            min_num_nodes=getattr(args, "min_num_nodes", None),
            max_num_nodes=getattr(args, "max_num_nodes", None),
            avg_edge_length=getattr(args, "avg_edge_length", 1.0),
            min_degree=getattr(args, "min_degree", 1),
            max_degree=getattr(args, "max_degree", 6),
        )
    if args.dataset == "qm9":
        qm9_root = Path(getattr(args, "qm9_root", Path("Latent_encoding/data/qm9")))
        split_fracs = getattr(args, "qm9_split_fractions", (0.8, 0.1, 0.1))
        if isinstance(split_fracs, list):
            split_fracs = tuple(split_fracs)
        max_nodes = getattr(args, "qm9_max_nodes", None) or args.num_nodes
        return QM9PointCloudDataset(
            root=qm9_root,
            split=getattr(args, "split", "val"),
            limit=args.num_graphs,
            split_fractions=split_fracs,
            split_seed=getattr(args, "qm9_split_seed", 0),
            max_nodes=max_nodes,
            center=getattr(args, "qm9_center", True),
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    dataset = build_dataset(args)
    num_graphs = len(dataset)
    if num_graphs == 0:
        raise ValueError("No graphs available for visualization")
    print(f"Loaded {args.dataset} dataset with {num_graphs} graphs for visualization")
    loader = DataLoader(dataset, batch_size=num_graphs, shuffle=False)
    batch = next(iter(loader)).to(device)

    model = GraphAutoencoder(
        num_nodes=args.num_nodes,
        in_node_dim=dataset.num_node_features,
        num_layers=args.num_layers,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        slot_attn_heads=args.slot_heads,
        diffusion_hidden=args.diff_hidden,
        diffusion_steps=args.diff_steps,
        diffusion_step_size=args.diff_step_size,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(batch)

    losses = []
    aligned_preds = []
    true_positions = []
    for idx in range(num_graphs):
        mask = batch.batch == idx
        pos_true = batch.pos[mask]
        if pos_true.numel() == 0:
            raise ValueError(f"Graph {idx} has no nodes; cannot visualize")
        pos_pred = preds[idx][: pos_true.size(0)]
        loss = kernel_correlation_loss_pyg(
            pos_pred=pos_pred,
            pos_true=pos_true,
            batch_true=torch.zeros(pos_true.size(0), dtype=torch.long, device=pos_true.device),
            center=True,
            lambda_global=args.kc_lambda_global,
            lambda_local=args.kc_lambda_local,
        )
        losses.append(loss.item())
        aligned = kabsch_align(pos_pred, pos_true).cpu()
        aligned_preds.append(aligned.numpy())
        true_positions.append(pos_true.cpu().numpy())

    data_traces = []
    buttons = []
    for idx in range(num_graphs):
        true_pts = true_positions[idx]
        pred_pts = aligned_preds[idx]
        true_trace = go.Scatter3d(
            x=true_pts[:, 0],
            y=true_pts[:, 1],
            z=true_pts[:, 2],
            mode="markers",
            name=f"Graph {idx} true",
            marker=dict(size=4, color="blue"),
            visible=(idx == 0),
        )
        pred_trace = go.Scatter3d(
            x=pred_pts[:, 0],
            y=pred_pts[:, 1],
            z=pred_pts[:, 2],
            mode="markers",
            name=f"Graph {idx} pred (loss={losses[idx]:.4f})",
            marker=dict(size=3, color="red"),
            visible=(idx == 0),
        )
        data_traces.extend([true_trace, pred_trace])
        vis = [False] * (2 * num_graphs)
        vis[2 * idx] = True
        vis[2 * idx + 1] = True
        buttons.append(
            dict(label=f"Graph {idx}", method="update", args=[{"visible": vis}])
        )

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title="Aligned Autoencoder Reconstructions",
        scene=dict(aspectmode="data"),
        updatemenus=[dict(active=0, buttons=buttons, direction="down", x=1.15, xanchor="left", y=1.0)],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Saved visualization to {args.output}")
    for idx, loss in enumerate(losses):
        print(f"Graph {idx}: loss={loss:.6f}")


if __name__ == "__main__":
    main()
