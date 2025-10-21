"""Visualize KDE/MMD similarity on 2D graphs with matplotlib.

Creates two figures:
1. KDE/MMD loss vs. noise level and node masking
2. Overlaid source/target clouds for different noise/masking levels
"""
import torch
import matplotlib.pyplot as plt

from torch_geometric.data import Data

from rapidalign import kde_mmd_loss


def make_graph(num_nodes: int, seed: int = 0) -> Data:
    torch.manual_seed(seed)
    pos = torch.randn(num_nodes, 2)
    return Data(pos=pos)


def make_noisy_graph(base: Data, noise_level: float, direction: torch.Tensor) -> Data:
    pos = base.pos + noise_level * direction[: base.num_nodes]
    return Data(pos=pos)


def remove_nodes(base: Data, num_remove: int, seed: int) -> Data:
    torch.manual_seed(seed)
    perm = torch.randperm(base.num_nodes)
    keep = perm[:-num_remove] if num_remove > 0 else perm
    return Data(pos=base.pos[keep])


def compute_loss(base: Data, target: Data, sigma: float) -> float:
    loss, _, _, _ = kde_mmd_loss(base.pos, target.pos, sigma=sigma, center=True)
    return loss.item()


def figure_loss_curves(base: Data, sigma: float):
    torch.manual_seed(0)
    direction = torch.randn_like(base.pos)

    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    noise_losses = []
    for level in noise_levels:
        noisy = make_noisy_graph(base, level, direction)
        noise_losses.append(compute_loss(base, noisy, sigma))

    mask_counts = [0, 10, 20, 40]
    mask_losses = []
    for drop in mask_counts:
        masked = remove_nodes(base, drop, seed=123 + drop)
        mask_losses.append(compute_loss(base, masked, sigma))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(noise_levels, noise_losses, marker='o')
    axes[0].set_title('Noise vs. loss')
    axes[0].set_xlabel('Noise level')
    axes[0].set_ylabel('MMD^2 loss')
    axes[0].grid(True)

    axes[1].plot(mask_counts, mask_losses, marker='o', color='tab:orange')
    axes[1].set_title('Node removal vs. loss')
    axes[1].set_xlabel('Nodes removed')
    axes[1].set_ylabel('MMD^2 loss')
    axes[1].grid(True)

    fig.tight_layout()
    return fig


def figure_overlays(base: Data, sigma: float):
    torch.manual_seed(1)
    direction = torch.randn_like(base.pos)
    noise_levels = [0.0, 0.1, 0.2]
    mask_counts = [0, 20, 40]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, level in enumerate(noise_levels):
        noisy = make_noisy_graph(base, level, direction)
        loss = compute_loss(base, noisy, sigma)
        ax = axes[0, idx]
        ax.scatter(base.pos[:, 0], base.pos[:, 1], s=15, label='Base')
        ax.scatter(noisy.pos[:, 0], noisy.pos[:, 1], s=15, label='Noisy')
        ax.set_title(f'Noise {level:.2f}, loss={loss:.4f}')
        ax.set_aspect('equal')
        ax.legend()

    for idx, drop in enumerate(mask_counts):
        masked = remove_nodes(base, drop, seed=200 + drop)
        loss = compute_loss(base, masked, sigma)
        ax = axes[1, idx]
        ax.scatter(base.pos[:, 0], base.pos[:, 1], s=15, label='Base')
        ax.scatter(masked.pos[:, 0], masked.pos[:, 1], s=15, label='Masked')
        ax.set_title(f'Remove {drop}, loss={loss:.4f}')
        ax.set_aspect('equal')
        ax.legend()

    fig.tight_layout()
    return fig


def main():
    base = make_graph(128, seed=42)
    sigma = 0.2

    fig1 = figure_loss_curves(base, sigma)
    fig2 = figure_overlays(base, sigma)

    fig1.suptitle('KDE/MMD loss vs. noise and node removal')
    fig2.suptitle('Source/target overlays with KDE/MMD loss')
    plt.show()


if __name__ == '__main__':
    main()
