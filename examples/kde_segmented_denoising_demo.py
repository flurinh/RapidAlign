"""Diffusion-style denoising demo using the segmented KDE loss.

This script trains a lightweight MLP to denoise noisy point clouds generated
on the fly. The loss is the KDE/MMD similarity between the predicted clean
cloud and the ground-truth cloud, evaluated with the segmented autograd
wrapper so that batches with variable node counts are supported.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.data import Data, Batch
from pathlib import Path
import matplotlib.pyplot as plt

from rapidalign.autograd_kde import segmented_kde_loss


@dataclass
class DatasetConfig:
    batch_size: int = 8
    num_nodes: int = 64
    dim: int = 3
    noise_min: float = 0.05
    noise_max: float = 0.25


def sample_batch(cfg: DatasetConfig, device: torch.device, *, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)

    clean_graphs: list[Data] = []
    noisy_graphs: list[Data] = []
    weights: list[torch.Tensor] = []

    dtype = torch.float32 if device.type == "cuda" else torch.float64

    for _ in range(cfg.batch_size):
        n = cfg.num_nodes
        theta = torch.linspace(0, 2 * torch.pi, steps=n + 1, dtype=torch.float32)[:-1]
        phi = torch.linspace(0, torch.pi, steps=n + 1, dtype=torch.float32)[:-1]
        base = torch.stack([
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi)
        ], dim=1).to(dtype)

        noise_level = torch.empty(1).uniform_(cfg.noise_min, cfg.noise_max).item()
        noisy = base + noise_level * torch.randn_like(base)

        clean_graphs.append(Data(pos=base))
        noisy_graphs.append(Data(pos=noisy))
        weights.append(torch.full((n,), 1.0 / n, dtype=dtype))

    clean_batch = Batch.from_data_list(clean_graphs).to(device)
    noisy_batch = Batch.from_data_list(noisy_graphs).to(device)

    weights = torch.cat(weights).to(device)

    return noisy_batch, clean_batch, weights


class ResidualDenoiser(nn.Module):
    def __init__(self, dim: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, pos: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
        batch_ids = torch.repeat_interleave(torch.arange(ptr.numel() - 1, device=pos.device), ptr.diff())
        centroids = torch.zeros((ptr.numel() - 1, pos.size(1)), device=pos.device, dtype=pos.dtype)
        centroids.index_add_(0, batch_ids, pos)
        counts = ptr.diff().clamp_min(1).unsqueeze(1)
        centroids = centroids / counts
        centered = pos - centroids[batch_ids]
        delta = self.net(centered)
        return pos + delta


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Training on {device}")

    cfg = DatasetConfig(
        batch_size=args.batch,
        num_nodes=args.num_nodes,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
    )

    model = ResidualDenoiser(dim=cfg.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    sigma_val = args.sigma
    sigma = torch.tensor(sigma_val, device=device, dtype=torch.float64 if device.type == "cpu" else torch.float32, requires_grad=False)

    for step in range(1, args.steps + 1):
        noisy_batch, clean_batch, weights = sample_batch(cfg, device)

        pred = model(noisy_batch.pos, noisy_batch.ptr)

        loss_kde = segmented_kde_loss(
            pred,
            clean_batch.pos,
            weights,
            weights,
            noisy_batch.ptr,
            clean_batch.ptr,
            sigma,
        ).mean()

        mse = (pred - clean_batch.pos).pow(2).mean()
        loss = loss_kde + args.mse_weight * mse

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:04d} | loss={loss.item():.4f} | kde={loss_kde.item():.4f} | mse={mse.item():.4f}"
            )

    if args.visualize:
        visualize(model, cfg, device, Path(args.visualize), args.vis_samples)


def visualize(model: nn.Module, cfg: DatasetConfig, device: torch.device, outdir: Path, samples: int):
    outdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx in range(samples):
            noisy_batch, clean_batch, _ = sample_batch(cfg, device, seed=idx)
            pred = model(noisy_batch.pos, noisy_batch.ptr)

            noisy = noisy_batch.pos.cpu().numpy()
            clean = clean_batch.pos.cpu().numpy()
            denoised = pred.cpu().numpy()

            fig = plt.figure(figsize=(9, 3))
            for col, (title, pts) in enumerate((
                ("Noisy", noisy),
                ("Denoised", denoised),
                ("Clean", clean),
            )):
                ax = fig.add_subplot(1, 3, col + 1, projection="3d")
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, alpha=0.8)
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_box_aspect([1, 1, 1])

            fig.tight_layout()
            out_path = outdir / f"sample_{idx:02d}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Segmented KDE denoising demo")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--noise-min", type=float, default=0.05)
    parser.add_argument("--noise-max", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--visualize", type=str, default="")
    parser.add_argument("--vis-samples", type=int, default=3)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
