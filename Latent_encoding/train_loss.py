#!/usr/bin/env python3
# Latent_encoding/train_loss.py
"""
Train the KC Loss MLP calibration network.

This trains the MLP inside ParameterizedKCLoss to map kernel embedding
differences to exact geometric severity.

Usage:
    python -m Latent_encoding.pregenerate_data --output kc_data.pt
    python -m Latent_encoding.train_loss --data kc_data.pt --output kc_pretrained.pt
"""

import argparse
import inspect
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split

# Handle imports
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from losses.trainable_kc import ParameterizedKCLoss, ParameterizedKCConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Data
    parser.add_argument("--data", type=Path, default=Path("kc_training_data.pt"))
    parser.add_argument("--output", type=Path, default=Path("kc_pretrained.pt"))

    # Training
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.1)

    # MLP Architecture
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--mlp-layers", type=int, default=22)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Loss
    parser.add_argument("--loss", type=str, default="smooth_l1",
                        choices=["mse", "smooth_l1", "huber"])

    # Hardware
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def compute_metrics(preds, targets):
    """Compute evaluation metrics."""
    with torch.no_grad():
        mae = (preds - targets).abs().mean().item()
        rmse = ((preds - targets).pow(2).mean().sqrt()).item()

        # Pearson correlation
        p_centered = preds - preds.mean()
        t_centered = targets - targets.mean()
        p_std = p_centered.std().clamp_min(1e-8)
        t_std = t_centered.std().clamp_min(1e-8)
        corr = ((p_centered * t_centered).mean() / (p_std * t_std)).item()

        # R² score
        ss_res = (targets - preds).pow(2).sum()
        ss_tot = (targets - targets.mean()).pow(2).sum().clamp_min(1e-8)
        r2 = (1 - ss_res / ss_tot).item()

    return {"mae": mae, "rmse": rmse, "corr": corr, "r2": r2}


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("KC Loss MLP Training")
    print("=" * 60)

    # 1. Load pregenerated data
    print(f"\nLoading data from {args.data}...")
    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    checkpoint = torch.load(args.data, weights_only=True)
    features = checkpoint["features"]
    targets = checkpoint["targets"]
    saved_config = checkpoint["config"]
    metadata = checkpoint.get("metadata", {})

    feature_dim = features.shape[1]

    print(f"Loaded {len(targets):,} samples")
    print(f"Feature dimension: {feature_dim}")
    print(f"  - num_scales: {saved_config['num_scales']}")
    print(f"  - sigma_range: {saved_config['sigma_range']}")
    print(f"  - use_local_features: {saved_config.get('use_local_features', False)}")
    print(f"Target stats: mean={targets.mean():.4f}, std={targets.std():.4f}")
    print(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")

    # 2. Create train/val split
    dataset = TensorDataset(features, targets)
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {train_size:,}, Val: {val_size:,}")

    # 3. Build ParameterizedKCLoss model
    # Detect which config fields are available
    config_fields = set(inspect.signature(ParameterizedKCConfig).parameters.keys())

    config_kwargs = {
        "num_scales": saved_config["num_scales"],
        "sigma_range": tuple(saved_config["sigma_range"]),
        "learnable": True,
    }

    # Add optional fields if supported
    if "use_local_features" in config_fields:
        config_kwargs["use_local_features"] = saved_config.get("use_local_features", False)
    if "mlp_hidden" in config_fields:
        config_kwargs["mlp_hidden"] = args.mlp_hidden
    if "mlp_layers" in config_fields:
        config_kwargs["mlp_layers"] = args.mlp_layers
    if "dropout" in config_fields:
        config_kwargs["dropout"] = args.dropout
    if "use_layer_norm" in config_fields:
        config_kwargs["use_layer_norm"] = True

    kc_config = ParameterizedKCConfig(**config_kwargs)
    model = ParameterizedKCLoss(kc_config).to(device)

    # Verify feature dimension
    if model.feature_dim != feature_dim:
        raise ValueError(
            f"Feature dimension mismatch! Model: {model.feature_dim}, Data: {feature_dim}"
        )

    num_params = sum(p.numel() for p in model.mlp.parameters())
    print(f"\nMLP Parameters: {num_params:,}")

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss function
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.HuberLoss()

    print(f"Loss function: {args.loss}")

    # 5. Training loop
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60)

    best_val_loss = float('inf')
    best_metrics = {}
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model.mlp(x).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mlp.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_samples += len(y)

        train_loss /= train_samples

        # Validate
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model.mlp(x).squeeze(-1)
                val_loss += criterion(pred, y).item() * len(y)
                val_samples += len(y)
                all_preds.append(pred)
                all_targets.append(y)

        val_loss /= val_samples
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **metrics, "lr": lr,
        })

        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.5f} | "
              f"Val: {val_loss:.5f} | "
              f"MAE: {metrics['mae']:.4f} | "
              f"Corr: {metrics['corr']:.4f} | "
              f"R²: {metrics['r2']:.4f} | "
              f"LR: {lr:.2e}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics.copy()
            patience_counter = 0

            save_config = {
                "num_scales": saved_config["num_scales"],
                "sigma_range": saved_config["sigma_range"],
                "use_local_features": saved_config.get("use_local_features", False),
                "mlp_hidden": args.mlp_hidden,
                "mlp_layers": args.mlp_layers,
                "dropout": args.dropout,
                "use_layer_norm": True,
                "feature_dim": feature_dim,
            }

            torch.save({
                "state_dict": model.state_dict(),
                "config": save_config,
                "metrics": {"val_loss": val_loss, **metrics},
                "epoch": epoch,
                "history": history,
            }, args.output)

            print(f"  -> Saved best model (val_loss: {val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # 6. Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Val Loss: {best_val_loss:.5f}")
    print(f"Best Metrics:")
    print(f"  MAE:  {best_metrics['mae']:.4f}")
    print(f"  RMSE: {best_metrics['rmse']:.4f}")
    print(f"  Corr: {best_metrics['corr']:.4f}")
    print(f"  R2:   {best_metrics['r2']:.4f}")
    print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    main()