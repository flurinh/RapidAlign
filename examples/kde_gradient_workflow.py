"""CPU-only KDE gradient workflow demo.

Runs the staged gradient checks (identical, noisy, ragged) using the
analytic CPU helpers and visualises the resulting gradients.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from rapidalign.kde_gradients import (
    kde_mmd_backward_cpu,
    kde_mmd_backward_ragged_cpu,
)


def _normalize_weights(w: torch.Tensor) -> torch.Tensor:
    return w / w.sum()


def plot_overlay(ax, src, tgt, grads, title, show_descent=True):
    src = src.detach().float()
    tgt = tgt.detach().float()
    grads = grads.detach().float()

    arrows = -grads if show_descent else grads
    arrow_label = "descent" if show_descent else "gradient"
    arrow_color = "tab:red"

    ax.scatter(src[:, 0], src[:, 1], color="tab:green", alpha=0.7, label="source")
    ax.scatter(tgt[:, 0], tgt[:, 1], color="tab:blue", alpha=0.7, label="target")
    ax.quiver(
        tgt[:, 0],
        tgt[:, 1],
        arrows[:, 0],
        arrows[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=arrow_color,
        alpha=0.75,
        width=0.004,
        label=arrow_label
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)


def stage1_identical(ax):
    torch.manual_seed(0)
    x = torch.randn(5, 2, dtype=torch.float64)
    w = torch.full((5,), 1.0 / 5, dtype=torch.float64)
    sigma = 0.3

    grads = kde_mmd_backward_cpu(x, x.clone(), w, w, sigma)
    plot_overlay(ax, x, x, grads["x"], "Stage 1: identical")
    ax.text(0.02, 0.92, f"‖grad‖={grads['x'].norm():.2e}", transform=ax.transAxes)


def stage2_noisy(ax):
    torch.manual_seed(1)
    base = torch.randn(8, 2, dtype=torch.float64)
    noise = base + 0.12 * torch.randn_like(base)
    w = torch.full((base.size(0),), 1.0 / base.size(0), dtype=torch.float64)
    sigma = 0.25

    grads = kde_mmd_backward_cpu(base, noise, w, w, sigma)
    plot_overlay(ax, base, noise, grads["y"], "Stage 2: noisy target")


def stage3_ragged(axs):
    torch.manual_seed(2)
    sizes_src = [5, 7]
    sizes_tgt = [6, 10]
    ptr_x = torch.tensor([0, sizes_src[0], sum(sizes_src)], dtype=torch.long)
    ptr_y = torch.tensor([0, sizes_tgt[0], sum(sizes_tgt)], dtype=torch.long)

    x_chunks = []
    y_chunks = []
    w_x_chunks = []
    w_y_chunks = []
    noise_levels = [0.1, 0.2]

    for n_src, n_tgt, noise in zip(sizes_src, sizes_tgt, noise_levels):
        base = torch.randn(n_src, 2, dtype=torch.float64)
        tgt = base.clone()
        tgt = tgt + noise * torch.randn_like(tgt)
        if n_tgt < n_src:
            tgt = tgt[:n_tgt]
        elif n_tgt > n_src:
            centroid = base.mean(dim=0, keepdim=True)
            extra = centroid + 0.3 * torch.randn(n_tgt - n_src, 2, dtype=torch.float64)
            tgt = torch.cat([tgt, extra], dim=0)

        x_chunks.append(base)
        y_chunks.append(tgt)
        w_x_chunks.append(_normalize_weights(torch.rand(base.size(0), dtype=torch.float64)))
        w_y_chunks.append(_normalize_weights(torch.rand(tgt.size(0), dtype=torch.float64)))

    x = torch.cat(x_chunks, dim=0)
    y = torch.cat(y_chunks, dim=0)
    w_x = torch.cat(w_x_chunks, dim=0)
    w_y = torch.cat(w_y_chunks, dim=0)
    sigma = 0.3

    grads = kde_mmd_backward_ragged_cpu(x, y, w_x, w_y, ptr_x, ptr_y, sigma)

    for b, ax in enumerate(axs):
        xs = slice(ptr_x[b].item(), ptr_x[b + 1].item())
        ys = slice(ptr_y[b].item(), ptr_y[b + 1].item())
        plot_overlay(
            ax,
            x[xs],
            y[ys],
            grads["y"][ys],
            f"Stage 3 pair {b}: Nx={xs.stop - xs.start}, Ny={ys.stop - ys.start}"
        )


def main():
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax31 = fig.add_subplot(gs[0, 2])
    ax32 = fig.add_subplot(gs[0, 3])

    stage1_identical(ax1)
    stage2_noisy(ax2)
    stage3_ragged([ax31, ax32])

    fig.suptitle("KDE gradient workflow (CPU) — red arrows indicate descent direction")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
