"""Training script for the slot-latent graph autoencoder."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

if __package__ in (None, ""):
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    from Latent_encoding.data import QM9PointCloudDataset, SyntheticPointCloudDataset  # type: ignore
    from Latent_encoding.losses import kernel_correlation_loss_pyg  # type: ignore
    from Latent_encoding.models import GraphAutoencoder  # type: ignore
    from Latent_encoding.utils import apply_config  # type: ignore
else:
    from .data import QM9PointCloudDataset, SyntheticPointCloudDataset
    from .losses import kernel_correlation_loss_pyg
    from .models import GraphAutoencoder
    from .utils import apply_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON config file.")
    parser.add_argument("--num-epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=("synthetic", "qm9"),
        help="Dataset to train on (can also be set via config).",
    )
    args = parser.parse_args()
    args = apply_config(args, parser)
    return args


def dense_positions(batch, max_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return to_dense_batch(batch.pos, batch=batch.batch, max_num_nodes=max_nodes)


def build_dataset(args: argparse.Namespace, split: str):
    if args.dataset == "synthetic":
        num_graphs = args.num_train if split == "train" else args.num_val
        seed = getattr(args, "seed", 0)
        dataset_seed = seed if split == "train" else 123
        return SyntheticPointCloudDataset(
            num_graphs=num_graphs,
            num_nodes=args.num_nodes,
            num_node_features=args.num_node_features,
            seed=dataset_seed,
            feature_mode=getattr(args, "feature_mode", "ones"),
            min_num_nodes=getattr(args, "min_num_nodes", None),
            max_num_nodes=getattr(args, "max_num_nodes", None),
            avg_edge_length=getattr(args, "avg_edge_length", 1.0),
            min_degree=getattr(args, "min_degree", 1),
            max_degree=getattr(args, "max_degree", 6),
        )
    if args.dataset == "qm9":
        limit = args.num_train if split == "train" else args.num_val
        qm9_root = Path(getattr(args, "qm9_root", Path("Latent_encoding/data/qm9")))
        split_fracs = getattr(args, "qm9_split_fractions", (0.8, 0.1, 0.1))
        if isinstance(split_fracs, list):
            split_fracs = tuple(split_fracs)
        max_nodes = getattr(args, "qm9_max_nodes", None) or args.num_nodes
        return QM9PointCloudDataset(
            root=qm9_root,
            split=split,
            limit=limit,
            split_fractions=split_fracs,
            split_seed=getattr(args, "qm9_split_seed", 0),
            max_nodes=max_nodes,
            center=getattr(args, "qm9_center", True),
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_kwargs: dict,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    max_nodes = model.decoder.num_nodes

    for graph_batch in loader:
        graph_batch = graph_batch.to(device)
        optimizer.zero_grad()
        _, mask = dense_positions(graph_batch, max_nodes=max_nodes)
        mask_flat = mask.reshape(-1)
        pos_hat = model(graph_batch)
        pos_pred_flat = pos_hat.reshape(-1, 3)[mask_flat]
        loss = kernel_correlation_loss_pyg(
            pos_pred=pos_pred_flat,
            pos_true=graph_batch.pos,
            batch_true=graph_batch.batch,
            **loss_kwargs,
        )
        loss.backward()
        optimizer.step()

        batch_size = int(graph_batch.batch.max().item()) + 1
        total_loss += float(loss.item()) * batch_size
        total_graphs += batch_size

    return total_loss / max(total_graphs, 1)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_kwargs: dict,
) -> float:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    max_nodes = model.decoder.num_nodes

    with torch.no_grad():
        for graph_batch in loader:
            graph_batch = graph_batch.to(device)
            _, mask = dense_positions(graph_batch, max_nodes=max_nodes)
            mask_flat = mask.reshape(-1)
            pos_hat = model(graph_batch)
            pos_pred_flat = pos_hat.reshape(-1, 3)[mask_flat]
            loss = kernel_correlation_loss_pyg(
                pos_pred=pos_pred_flat,
                pos_true=graph_batch.pos,
                batch_true=graph_batch.batch,
                **loss_kwargs,
            )
            batch_size = int(graph_batch.batch.max().item()) + 1
            total_loss += float(loss.item()) * batch_size
            total_graphs += batch_size

    return total_loss / max(total_graphs, 1)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training configuration:")
    for key in sorted(vars(args)):
        print(f"  {key}: {getattr(args, key)}")

    train_ds = build_dataset(args, split="train")
    val_ds = build_dataset(args, split="val")
    print(
        f"Loaded {args.dataset} dataset | train graphs: {len(train_ds)} | val graphs: {len(val_ds)}"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GraphAutoencoder(
        num_nodes=args.num_nodes,
        in_node_dim=train_ds.num_node_features,
        num_layers=args.num_layers,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        slot_attn_heads=args.slot_heads,
        diffusion_hidden=args.diff_hidden,
        diffusion_steps=args.diff_steps,
        diffusion_step_size=args.diff_step_size,
    ).to(device)
    print("Model architecture:\n", model)

    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_opt_state = copy.deepcopy(optimizer.state_dict())
    patience_counter = 0
    loss_kwargs = dict(
        center=args.kc_center,
        lambda_global=args.kc_lambda_global,
        lambda_local=args.kc_lambda_local,
        global_config=dict(
            n_bins=args.kc_bins,
            r_max=args.kc_rmax,
            gamma=args.kc_gamma,
            normalize=args.kc_normalize,
        ),
        local_config=dict(
            num_bins=args.kc_local_bins,
            r_max=args.kc_local_rmax,
            gamma=args.kc_local_gamma,
            radius=args.kc_local_radius,
            k_max=args.kc_local_k,
            tau=args.kc_local_tau,
            normalize=args.kc_local_normalize,
        ),
    )

    def schedule(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            return args.lr * epoch / args.warmup_epochs
        if args.num_epochs > args.warmup_epochs:
            progress = (epoch - args.warmup_epochs) / max(args.num_epochs - args.warmup_epochs, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * args.lr * (1 + math.cos(math.pi * progress))
        return args.lr

    for epoch in range(1, args.num_epochs + 1):
        lr = schedule(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_kwargs)
        val_loss = eval_epoch(model, val_loader, device, loss_kwargs)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_opt_state = copy.deepcopy(optimizer.state_dict())
            Path("Latent_encoding").mkdir(parents=True, exist_ok=True)
            torch.save(best_state, Path("Latent_encoding/best_ae.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("No improvement - reloading best model weights and optimizer state")
                model.load_state_dict(best_state)
                optimizer.load_state_dict(best_opt_state)
                patience_counter = 0

    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
