"""Visualize autoencoder reconstructions with per-graph losses."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("Latent_encoding/best_ae.pt"),
        help="Path to AE weights.",
    )
    parser.add_argument("--num-graphs", type=int, default=10, help="Number of synthetic graphs to visualize.")
    parser.add_argument("--num-nodes", type=int, default=16, help="Nodes per synthetic graph.")
    parser.add_argument("--device", type=str, default="cuda", help="torch device.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Latent_encoding/ae_recon_vis.html"),
        help="Output HTML path.",
    )

    # Encoder / slots / decoder (kept in sync with train.py)
    parser.add_argument("--num-layers", type=int, default=3, help="Encoder layers.")
    parser.add_argument("--num-slots", type=int, default=8, help="Latent slots.")
    parser.add_argument("--slot-dim", type=int, default=128, help="Slot dimensionality.")
    parser.add_argument("--slot-heads", type=int, default=1, help="Slot-attention heads.")
    parser.add_argument("--diff-hidden", type=int, default=256, help="Decoder hidden size.")
    parser.add_argument("--diff-steps", type=int, default=30, help="Decoder refinement steps.")
    parser.add_argument("--diff-step-size", type=float, default=1.0, help="Diffusion step size.")

    # Irreps / encoder width (new in train.py)
    parser.add_argument(
        "--scalar-width",
        type=int,
        default=4,
        help="Number of scalar (0e) channels in the equivariant backbone.",
    )
    parser.add_argument(
        "--vector-width",
        type=int,
        default=4,
        help="Number of vector (1o) channels in the equivariant backbone.",
    )
    parser.add_argument(
        "--l2-width",
        type=int,
        default=0,
        help="Number of l=2 (2e) channels in the equivariant backbone. Set >0 to enable L=2.",
    )
    parser.add_argument(
        "--sh-lmax",
        type=int,
        default=None,
        help="Maximum spherical harmonic degree for edges (None => inferred from widths).",
    )

    # Kernel-correlation loss weights (full config picked up from JSON if present)
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


def make_loss_kwargs(args: argparse.Namespace) -> dict:
    """
    Mirror the loss configuration used in train.py when possible, but stay
    backwards-compatible if some fields are missing from the config.
    """
    loss_kwargs: dict = dict(
        center=getattr(args, "kc_center", True),
        lambda_global=getattr(args, "kc_lambda_global", 1.0),
        lambda_local=getattr(args, "kc_lambda_local", 1.0),
    )

    # Optional global kernel config
    global_keys = ("kc_bins", "kc_rmax", "kc_gamma", "kc_normalize")
    if all(hasattr(args, k) for k in global_keys):
        loss_kwargs["global_config"] = dict(
            n_bins=getattr(args, "kc_bins"),
            r_max=getattr(args, "kc_rmax"),
            gamma=getattr(args, "kc_gamma"),
            normalize=getattr(args, "kc_normalize"),
        )

    # Optional local kernel config
    local_keys = (
        "kc_local_bins",
        "kc_local_rmax",
        "kc_local_gamma",
        "kc_local_radius",
        "kc_local_k",
        "kc_local_tau",
        "kc_local_normalize",
    )
    if all(hasattr(args, k) for k in local_keys):
        loss_kwargs["local_config"] = dict(
            num_bins=getattr(args, "kc_local_bins"),
            r_max=getattr(args, "kc_local_rmax"),
            gamma=getattr(args, "kc_local_gamma"),
            radius=getattr(args, "kc_local_radius"),
            k_max=getattr(args, "kc_local_k"),
            tau=getattr(args, "kc_local_tau"),
            normalize=getattr(args, "kc_local_normalize"),
        )

    return loss_kwargs


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

    # Build model in a way that's compatible with both old and new GraphAutoencoder
    base_kwargs = dict(
        num_nodes=args.num_nodes,
        in_node_dim=dataset.num_node_features,
        num_layers=args.num_layers,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        slot_attn_heads=args.slot_heads,
        diffusion_hidden=args.diff_hidden,
        diffusion_steps=args.diff_steps,
        diffusion_step_size=args.diff_step_size,
    )

    init_params = inspect.signature(GraphAutoencoder).parameters
    for name in ("scalar_width", "vector_width", "l2_width", "sh_lmax"):
        if name in init_params:
            base_kwargs[name] = getattr(args, name)

    model = GraphAutoencoder(**base_kwargs).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Dense positions + mask, matching the training pipeline
    max_nodes = args.num_nodes
    pos_dense, mask = to_dense_batch(batch.pos, batch=batch.batch, max_num_nodes=max_nodes)
    loss_kwargs = make_loss_kwargs(args)

    with torch.no_grad():
        preds = model(
            batch,
            coords_init=None,
            mask=mask,
            num_steps=getattr(args, "diff_steps", None),
        )
        if isinstance(preds, list):
            preds = preds[-1]  # [B, N, 3]

    if preds.dim() != 3:
        raise ValueError(f"Expected decoder output of shape [B, N, 3], got {tuple(preds.shape)}")

    losses = []
    aligned_preds = []
    true_positions = []

    B = mask.size(0)
    for idx in range(B):
        node_mask = mask[idx]  # [N]
        pos_true = pos_dense[idx][node_mask]  # [Ni, 3]
        if pos_true.numel() == 0:
            raise ValueError(f"Graph {idx} has no nodes; cannot visualize")

        pos_pred = preds[idx][node_mask]  # [Ni, 3]

        loss = kernel_correlation_loss_pyg(
            pos_pred=pos_pred,
            pos_true=pos_true,
            batch_true=torch.zeros(pos_true.size(0), dtype=torch.long, device=pos_true.device),
            **loss_kwargs,
        )
        losses.append(loss.item())

        aligned = kabsch_align(pos_pred, pos_true).cpu()
        aligned_preds.append(aligned.numpy())
        true_positions.append(pos_true.cpu().numpy())

    # Build Plotly figure with a dropdown to switch graphs
    data_traces = []
    buttons = []
    for idx in range(B):
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

        vis = [False] * (2 * B)
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
