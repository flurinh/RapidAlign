# Latent_encoding/train.py
"""Training script for the slot-latent graph autoencoder (denoising-capable, with configurable irreps)."""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

if __package__ in (None, ""):
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    from Latent_encoding.data import QM9PointCloudDataset, SyntheticPointCloudDataset  # type: ignore
    from Latent_encoding.data.noise import NoiseConfig, noisify_batch  # type: ignore
    from Latent_encoding.losses import kernel_correlation_loss_pyg  # type: ignore
    from Latent_encoding.models import GraphAutoencoder  # type: ignore
    from Latent_encoding.utils import apply_config  # type: ignore
else:
    from .data import QM9PointCloudDataset, SyntheticPointCloudDataset
    from .data.noise import NoiseConfig, noisify_batch
    from .losses import kernel_correlation_loss_pyg
    from .models import GraphAutoencoder
    from .utils import apply_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON config file.")
    parser.add_argument("--num-epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (used for logs/checkpoints). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=("synthetic", "qm9"),
        help="Dataset to train on (can also be set via config).",
    )

    # --- Irreps / encoder width ---
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
    parser.add_argument(
        "--mlp-hidden",
        type=int,
        default=64,
        help="Hidden dimension for edge MLPs in the equivariant backbone.",
    )

    # --- Radial basis expansion (encoder) ---
    parser.add_argument(
        "--use-rbf",
        action="store_true",
        help="Use Gaussian radial basis expansion for edge distances.",
    )
    parser.add_argument(
        "--rbf-num-basis",
        type=int,
        default=20,
        help="Number of Gaussian basis functions for RBF expansion.",
    )
    parser.add_argument(
        "--rbf-cutoff",
        type=float,
        default=5.0,
        help="Cutoff distance for RBF expansion.",
    )
    parser.add_argument(
        "--rbf-trainable",
        action="store_true",
        help="Make RBF centers and widths learnable parameters.",
    )

    # --- Decoder configuration ---
    parser.add_argument(
        "--decoder-state-dim",
        type=int,
        default=128,
        help="Dimension of node state vectors in decoder.",
    )
    parser.add_argument(
        "--decoder-hidden",
        type=int,
        default=256,
        help="Hidden dimension for decoder MLPs.",
    )
    parser.add_argument(
        "--decoder-steps",
        type=int,
        default=8,
        help="Number of refinement steps in decoder.",
    )
    parser.add_argument(
        "--decoder-step-size",
        type=float,
        default=0.5,
        help="Step size multiplier for coordinate updates.",
    )
    parser.add_argument(
        "--decoder-knn-k",
        type=int,
        default=8,
        help="Number of neighbors for dynamic kNN graph in decoder.",
    )
    parser.add_argument(
        "--decoder-mp-layers",
        type=int,
        default=2,
        help="Number of message passing layers per refinement step.",
    )
    parser.add_argument(
        "--decoder-attn-heads",
        type=int,
        default=4,
        help="Number of attention heads in slot cross-attention.",
    )
    parser.add_argument(
        "--decoder-rbf-basis",
        type=int,
        default=20,
        help="Number of RBF basis functions for decoder edge encoding.",
    )
    parser.add_argument(
        "--decoder-rbf-cutoff",
        type=float,
        default=5.0,
        help="RBF cutoff for decoder edge encoding.",
    )
    parser.add_argument(
        "--decoder-use-direction",
        action="store_true",
        default=True,
        help="Use direction vectors in decoder edge encoding.",
    )
    parser.add_argument(
        "--decoder-init-std",
        type=float,
        default=1.0,
        help="Std for decoder coordinate template initialization.",
    )

    # --- Noise / denoising schedule flags (overridable via JSON config) ---
    parser.add_argument(
        "--use-noise",
        action="store_true",
        help="Enable synthetic coordinate noise and train the decoder as a denoiser.",
    )
    parser.add_argument(
        "--blind-batch-prob",
        type=float,
        default=0.1,
        help="Probability per batch to run blind Z-only decoding (coords_init=None).",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=2,
        help="Minimum number of refinement steps for low-noise batches.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum number of refinement steps for high-noise batches. "
             "If 0, defaults to decoder.diffusion_steps.",
    )

    args = parser.parse_args()
    args = apply_config(args, parser)
    return args


def dense_positions(batch, max_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batched PyG Data to dense [B, N, 3] + [B, N] mask.
    """
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
        noise_cfg: NoiseConfig | None = None,
        blind_batch_prob: float = 0.1,
        min_steps: int = 2,
        max_steps: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    max_nodes = model.decoder.num_nodes
    max_steps = max_steps or model.decoder.steps

    for graph_batch in loader:
        graph_batch = graph_batch.to(device)
        optimizer.zero_grad()

        # Dense mask for this batch (based on clean graph)
        pos_dense, mask = dense_positions(graph_batch, max_nodes=max_nodes)
        B = mask.size(0)

        use_blind = (noise_cfg is None) or (torch.rand(1).item() < blind_batch_prob)
        coords_init = None
        num_steps = max_steps

        if (noise_cfg is not None) and (not use_blind):
            # 1) Apply coordinate-only noise to the batch
            noisy_batch, severities, metas = noisify_batch(graph_batch, noise_cfg)
            noisy_batch = noisy_batch.to(device)

            # 2) Dense noised positions (use same max_nodes)
            pos_noisy_dense, mask_noisy = dense_positions(noisy_batch, max_nodes=max_nodes)

            # Sanity: masks should match; if not, fall back to noisy mask
            if not torch.equal(mask_noisy, mask):
                mask = mask_noisy

            coords_init = pos_noisy_dense  # this is G_t

            # 3) Map mean severity to [min_steps, max_steps]
            sev = severities.to(device)  # [B]
            sev_norm = (sev / (sev.max() + 1e-8)).clamp(0.0, 1.0)
            mean_level = float(sev_norm.mean().item())
            span = max(0, max_steps - min_steps)
            num_steps = int(min_steps + round(mean_level * span))
            num_steps = max(min_steps, min(num_steps, max_steps))

        # 4) Forward pass: encoder sees clean graph, decoder sees coords_init (or template)
        pos_hat = model(
            graph_batch,
            coords_init=coords_init,
            mask=mask,
            num_steps=num_steps,
        )

        # If the decoder returns a sequence, take last frame
        if isinstance(pos_hat, list):
            pos_hat = pos_hat[-1]  # [B, N, 3]

        if not torch.isfinite(pos_hat).all():
            print("Non‑finite coords detected: "
                  f"min={pos_hat.nan_to_num().min().item():.3e}, "
                  f"max={pos_hat.nan_to_num().max().item():.3e}")
            # Optionally: break or raise to see the first bad batch
            raise RuntimeError("NaNs in decoder output")

        # 5) Geometric loss on real nodes only
        mask_flat = mask.reshape(-1)
        pos_pred_flat = pos_hat.reshape(-1, 3)[mask_flat]

        loss = kernel_correlation_loss_pyg(
            pos_pred=pos_pred_flat,
            pos_true=graph_batch.pos,
            batch_true=graph_batch.batch,
            **loss_kwargs,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # to avoid nan
        optimizer.step()

        batch_size = B
        total_loss += float(loss.item()) * batch_size
        total_graphs += batch_size

    return total_loss / max(total_graphs, 1)


def eval_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        loss_kwargs: dict,
        noise_cfg: NoiseConfig | None = None,
        blind_batch_prob: float = 0.0,
        min_steps: int = 2,
        max_steps: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    max_nodes = model.decoder.num_nodes
    max_steps = max_steps or model.decoder.steps

    with torch.no_grad():
        for graph_batch in loader:
            graph_batch = graph_batch.to(device)

            pos_dense, mask = dense_positions(graph_batch, max_nodes=max_nodes)
            B = mask.size(0)

            use_blind = (noise_cfg is None) or (torch.rand(1).item() < blind_batch_prob)
            coords_init = None
            num_steps = max_steps

            if (noise_cfg is not None) and (not use_blind):
                noisy_batch, severities, metas = noisify_batch(graph_batch, noise_cfg)
                noisy_batch = noisy_batch.to(device)
                pos_noisy_dense, mask_noisy = dense_positions(noisy_batch, max_nodes=max_nodes)
                if not torch.equal(mask_noisy, mask):
                    mask = mask_noisy
                coords_init = pos_noisy_dense

                sev = severities.to(device)
                sev_norm = (sev / (sev.max() + 1e-8)).clamp(0.0, 1.0)
                mean_level = float(sev_norm.mean().item())
                span = max(0, max_steps - min_steps)
                num_steps = int(min_steps + round(mean_level * span))
                num_steps = max(min_steps, min(num_steps, max_steps))

            pos_hat = model(
                graph_batch,
                coords_init=coords_init,
                mask=mask,
                num_steps=num_steps,
            )
            if isinstance(pos_hat, list):
                pos_hat = pos_hat[-1]

            mask_flat = mask.reshape(-1)
            pos_pred_flat = pos_hat.reshape(-1, 3)[mask_flat]

            loss = kernel_correlation_loss_pyg(
                pos_pred=pos_pred_flat,
                pos_true=graph_batch.pos,
                batch_true=graph_batch.batch,
                **loss_kwargs,
            )
            batch_size = B
            total_loss += float(loss.item()) * batch_size
            total_graphs += batch_size

    return total_loss / max(total_graphs, 1)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.dataset}_{timestamp}"
    else:
        run_name = args.run_name

    # Setup directories
    log_dir = Path("runs/logs") / run_name
    weight_dir = Path("runs/weights") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"Run name: {run_name}")
    print(f"Logs: {log_dir}")
    print(f"Weights: {weight_dir}")
    print("Training configuration:")
    for key in sorted(vars(args)):
        print(f"  {key}: {getattr(args, key)}")

    # Log hyperparameters to TensorBoard
    hparams = {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
               for k, v in vars(args).items()}
    writer.add_text("config", str(hparams), 0)

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
        # Decoder config
        decoder_state_dim=args.decoder_state_dim,
        decoder_hidden=args.decoder_hidden,
        decoder_steps=args.decoder_steps,
        decoder_step_size=args.decoder_step_size,
        decoder_knn_k=args.decoder_knn_k,
        decoder_mp_layers=args.decoder_mp_layers,
        decoder_attn_heads=args.decoder_attn_heads,
        decoder_rbf_basis=args.decoder_rbf_basis,
        decoder_rbf_cutoff=args.decoder_rbf_cutoff,
        decoder_use_direction=args.decoder_use_direction,
        decoder_init_std=args.decoder_init_std,
        # Encoder irreps config
        scalar_width=args.scalar_width,
        vector_width=args.vector_width,
        l2_width=args.l2_width,
        sh_lmax=args.sh_lmax,
        mlp_hidden=args.mlp_hidden,
        # Encoder RBF config
        use_rbf=args.use_rbf,
        rbf_num_basis=args.rbf_num_basis,
        rbf_cutoff=args.rbf_cutoff,
        rbf_trainable=args.rbf_trainable,
    ).to(device)
    print("Model architecture:\n", model)

    # Log model summary
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,} | Trainable: {num_trainable:,}")
    writer.add_text("model/summary", f"Total params: {num_params:,}, Trainable: {num_trainable:,}", 0)

    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_epoch = 0
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

    # Noise config + schedule
    noise_cfg = NoiseConfig()
    use_noise = bool(getattr(args, "use_noise", False))
    blind_prob = float(getattr(args, "blind_batch_prob", 0.1))
    min_steps = int(getattr(args, "min_steps", 2))
    max_steps_flag = int(getattr(args, "max_steps", 0))
    max_steps = max_steps_flag if max_steps_flag > 0 else args.decoder_steps

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

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_kwargs,
            noise_cfg=noise_cfg if use_noise else None,
            blind_batch_prob=blind_prob,
            min_steps=min_steps,
            max_steps=max_steps,
        )
        val_loss = eval_epoch(
            model,
            val_loader,
            device,
            loss_kwargs,
            noise_cfg=noise_cfg if use_noise else None,
            blind_batch_prob=blind_prob,
            min_steps=min_steps,
            max_steps=max_steps,
        )

        # Log to TensorBoard
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalars("loss/compare", {"train": train_loss, "val": val_loss}, epoch)

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.6f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val": best_val,
            "config": vars(args),
        }

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint["best_val"] = best_val
            torch.save(checkpoint, weight_dir / "best.pt")
            print(f"  → New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("No improvement - reloading best model weights and optimizer state")
                best_ckpt = torch.load(weight_dir / "best.pt", weights_only=False)
                model.load_state_dict(best_ckpt["model_state_dict"])
                optimizer.load_state_dict(best_ckpt["optimizer_state_dict"])
                patience_counter = 0

        # Save latest checkpoint (for resuming)
        torch.save(checkpoint, weight_dir / "latest.pt")

    # Save final model
    final_checkpoint = {
        "epoch": args.num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "config": vars(args),
    }
    torch.save(final_checkpoint, weight_dir / "final.pt")

    # Log final metrics
    writer.add_hparams(
        hparam_dict={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_layers": args.num_layers,
            "num_slots": args.num_slots,
            "slot_dim": args.slot_dim,
            "decoder_steps": args.decoder_steps,
            "use_rbf": args.use_rbf,
            "use_noise": use_noise,
        },
        metric_dict={
            "hparam/best_val_loss": best_val,
            "hparam/best_epoch": best_epoch,
            "hparam/final_train_loss": train_loss,
            "hparam/final_val_loss": val_loss,
        },
    )

    writer.close()
    print(f"\nDone. Best val loss: {best_val:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {weight_dir}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"  → Run: tensorboard --logdir runs/logs")


if __name__ == "__main__":
    main()