"""Visualize segmented KDE behaviour across noise/masking levels.

The script:
  1. Builds a 2D base graph.
  2. Creates noisy / masked targets at several noise amplitudes.
  3. Evaluates ``pyg_kde_mmd_loss`` on the ragged batches.
  4. Plots the loss curve and example overlays of source vs. target nodes.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch, Data

from rapidalign import pyg_kde_mmd_loss


def generate_targets(base: Data, noise_levels: list[float], drop: int) -> list[Data]:
    torch.manual_seed(77)
    direction = torch.randn_like(base.pos)

    targets = []
    for noise in noise_levels:
        pos = base.pos + noise * direction
        if drop > 0 and pos.size(0) > drop:
            perm = torch.randperm(pos.size(0))
            pos = pos[perm[:-drop]]
        targets.append(Data(pos=pos))
    return targets


def compute_losses(src: Batch, tgt: Batch, sigma: float) -> torch.Tensor:
    loss, _, _, _ = pyg_kde_mmd_loss(
        src.pos,
        tgt.pos,
        src_batch=src.batch,
        tgt_batch=tgt.batch,
        sigma=sigma,
        center=True,
    )
    return loss.detach().cpu()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Segmented KDE visualisation")
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--noise-max", type=float, default=0.8)
    parser.add_argument("--drop", type=int, default=0)
    parser.add_argument("--save", type=str, default=None, help="Optional path to save figure")
    args = parser.parse_args(argv)

    torch.manual_seed(10)
    base = Data(pos=torch.randn(96, 2))

    noise_values = [args.noise_max * i / (args.levels - 1) for i in range(args.levels)]
    targets = generate_targets(base, noise_values, args.drop)

    batch_src = Batch.from_data_list([base for _ in noise_values])
    batch_tgt = Batch.from_data_list(targets)

    losses = compute_losses(batch_src, batch_tgt, sigma=args.sigma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(noise_values, losses, marker="o")
    axes[0].set_title("KDE loss vs. noise")
    axes[0].set_xlabel("Noise level")
    axes[0].set_ylabel("MMD^2 loss")

    colors = ["tab:blue", "tab:orange"]
    examples = [0, max(1, args.levels // 2), args.levels - 1]
    for idx, level in enumerate(examples):
        ax = axes[1]
        src_pos = base.pos.cpu()
        tgt_pos = targets[level].pos.cpu()
        ax.scatter(src_pos[:, 0], src_pos[:, 1], s=12, color=colors[0], alpha=0.5, label="source" if idx == 0 else "")
        ax.scatter(tgt_pos[:, 0], tgt_pos[:, 1], s=12, color=colors[1], alpha=0.5,
                   label=f"noise={noise_values[level]:.2f}" if idx == 0 else "")

    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("Example overlays")
    axes[1].legend(loc="upper right")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=200)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
