"""Fit KDE loss parameters to track synthetic noise levels."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data, Batch

from rapidalign.autograd_kde import segmented_kde_loss


@dataclass
class DatasetConfig:
    batch_size: int = 8
    num_nodes: int = 128
    dim: int = 3
    noise_min: float = 0.0
    noise_max: float = 0.3
    drop_prob_max: float = 0.2
    shape: str = "sphere"  # sphere | cube | cuboid


def sample_batch(cfg: DatasetConfig, device: torch.device, *, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)

    clean_list: list[Data] = []
    noisy_list: list[Data] = []
    weights: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    dtype = torch.float32 if device.type == "cuda" else torch.float64

    for _ in range(cfg.batch_size):
        n = cfg.num_nodes
        base = generate_base_shape(cfg.shape, n, dtype)

        noise_level = torch.empty(1).uniform_(cfg.noise_min, cfg.noise_max).item()
        noisy = base + noise_level * torch.randn_like(base)

        drop_prob = torch.empty(1).uniform_(0.0, cfg.drop_prob_max).item()
        if drop_prob > 0.0:
            keep = torch.randperm(n)
            k = max(int(n * (1.0 - drop_prob)), 4)
            keep = keep[:k]
            base = base[keep]
            noisy = noisy[keep]
            n = base.size(0)

        clean_list.append(Data(pos=base))
        noisy_list.append(Data(pos=noisy))
        weights.append(torch.full((n,), 1.0 / n, dtype=dtype))
        labels.append(torch.tensor(noise_level, dtype=dtype))

    clean_batch = Batch.from_data_list(clean_list).to(device)
    noisy_batch = Batch.from_data_list(noisy_list).to(device)
    weights = torch.cat(weights).to(device)
    labels = torch.stack(labels).to(device)

    return noisy_batch, clean_batch, weights, labels


def generate_base_shape(shape: str, n: int, dtype: torch.dtype) -> torch.Tensor:
    if shape == "sphere":
        theta = torch.linspace(0, 2 * torch.pi, steps=n + 1, dtype=torch.float32)[:-1]
        phi = torch.linspace(0, torch.pi, steps=n + 1, dtype=torch.float32)[:-1]
        base = torch.stack([
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi),
        ], dim=1)
        return base.to(dtype)

    if shape == "cube":
        side = 1.0
        coords = torch.rand(n, 3) * side - side / 2
        faces = torch.randint(0, 6, (n,))
        axis = faces // 2
        sign = faces % 2
        coords[torch.arange(n), axis] = torch.where(sign == 0, -side / 2, side / 2)
        return coords.to(dtype)

    if shape == "cuboid":
        min_side, max_side = 0.6, 1.4
        lengths = torch.empty(3).uniform_(min_side, max_side)
        coords = torch.rand(n, 3) * lengths - lengths / 2
        faces = torch.randint(0, 6, (n,))
        axis = faces // 2
        sign = faces % 2
        fixed = torch.where(sign == 0, -lengths[axis] / 2, lengths[axis] / 2)
        coords[torch.arange(n), axis] = fixed
        return coords.to(dtype)

    raise ValueError(f"Unknown shape '{shape}'")


class LearnableKDE(nn.Module):
    def __init__(self, init_sigmas: list[float]):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.log(torch.tensor(init_sigmas)))
        self.log_weights = nn.Parameter(torch.zeros(len(init_sigmas)))

    def forward(self, noisy_batch: Batch, clean_batch: Batch, weights: torch.Tensor) -> torch.Tensor:
        pos_noisy = noisy_batch.pos
        pos_clean = clean_batch.pos
        ptr = noisy_batch.ptr
        sigmas = torch.exp(self.log_sigmas).to(pos_noisy.dtype)
        mix = torch.softmax(self.log_weights, dim=0).to(pos_noisy.dtype)

        losses = 0.0
        for sigma, coeff in zip(sigmas, mix):
            loss = segmented_kde_loss(
                pos_noisy,
                pos_clean,
                weights,
                weights,
                ptr,
                clean_batch.ptr,
                sigma,
            )
            losses = losses + coeff * loss
        return losses


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    print(f"Training kernel parameters on {device}")

    cfg = DatasetConfig(
        batch_size=args.batch,
        num_nodes=args.num_nodes,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        drop_prob_max=args.max_drop,
        shape=args.shape,
    )

    model = LearnableKDE(init_sigmas=args.bands).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses_hist = []
    corr = None

    for step in range(1, args.steps + 1):
        noisy_batch, clean_batch, weights, labels = sample_batch(cfg, device)

        preds = model(noisy_batch, clean_batch, weights)
        preds = preds.to(dtype)
        loss = ((preds - labels) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses_hist.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            sigmas = torch.exp(model.log_sigmas).detach().cpu().numpy()
            mix = torch.softmax(model.log_weights, dim=0).detach().cpu().numpy()
            sigma_str = ", ".join(f"{s:.4f}" for s in sigmas)
            mix_str = ", ".join(f"{m:.3f}" for m in mix)
            print(f"step={step:04d} | train MSE={loss.item():.6f} | sigmas=[{sigma_str}] | weights=[{mix_str}]")

    # Evaluation
    preds_eval = []
    labels_eval = []
    with torch.no_grad():
        for _ in range(args.eval_samples):
            noisy_batch, clean_batch, weights, labels = sample_batch(cfg, device)
            preds = model(noisy_batch, clean_batch, weights)
            preds_eval.append(preds.cpu())
            labels_eval.append(labels.cpu())

    preds_eval = torch.cat(preds_eval)
    labels_eval = torch.cat(labels_eval)
    corr = torch.corrcoef(torch.stack([preds_eval, labels_eval]))[0, 1].item()
    print(f"Correlation (eval) = {corr:.4f}")

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "log_sigmas": model.log_sigmas.detach().cpu(),
            "log_weights": model.log_weights.detach().cpu(),
            "corr": corr,
        }, path)
        print(f"Saved kernel parameters to {path}")


def main():
    parser = argparse.ArgumentParser(description="Fit KDE kernel parameters")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--num-nodes", type=int, default=128)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.3)
    parser.add_argument("--max-drop", type=float, default=0.2)
    parser.add_argument("--bands", nargs="+", type=float, default=[0.15, 0.25, 0.35])
    parser.add_argument("--shape", type=str, default="sphere", choices=["sphere", "cube", "cuboid"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
