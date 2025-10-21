"""Train a lightweight segmented KDE loss to track synthetic noise levels.

We generate batches of variable-sized PyG graphs, perturb them with Gaussian
noise (and optional node dropout), then learn:

  * a tiny SE(3)-invariant node-weight head, and
  * a global bandwidth parameter ``sigma``

so that the KDE/MMD loss correlates with the injected noise magnitude.

This mirrors the segmented runtime we expect inside diffusion/flow-based
autoencoders while keeping the example lightweight.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_mean

from rapidalign import pyg_kde_mmd_loss


def make_graph(num_nodes: int, seed: int) -> Data:
    torch.manual_seed(seed)
    pos = torch.randn(num_nodes, 3)
    return Data(pos=pos)


@dataclass
class SampleSpec:
    num_nodes: int
    noise: float
    drop_frac: float


class SyntheticBatchGenerator:
    def __init__(self, *, min_nodes: int, max_nodes: int,
                 noise_max: float, drop_max: float, seed: int = 0) -> None:
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.noise_max = noise_max
        self.drop_max = drop_max
        self.rng = torch.Generator().manual_seed(seed)
        self.sample_seed = seed

    def _sample_spec(self) -> SampleSpec:
        frac = torch.rand((), generator=self.rng).item()
        num_nodes = int(round(self.min_nodes + frac * (self.max_nodes - self.min_nodes)))
        noise = frac * self.noise_max
        drop_frac = frac * self.drop_max
        return SampleSpec(num_nodes=num_nodes, noise=noise, drop_frac=drop_frac)

    def sample(self, batch_size: int) -> tuple[Batch, Batch, torch.Tensor]:
        src_graphs = []
        tgt_graphs = []
        labels = []

        for _ in range(batch_size):
            spec = self._sample_spec()
            self.sample_seed += 1
            base = make_graph(spec.num_nodes, seed=self.sample_seed)

            tgt_pos = base.pos.clone()
            if spec.noise > 0:
                tgt_pos = tgt_pos + spec.noise * torch.randn_like(tgt_pos)

            if spec.drop_frac > 0:
                drop = min(spec.num_nodes - 1, int(round(spec.drop_frac * spec.num_nodes)))
                if drop > 0:
                    perm = torch.randperm(spec.num_nodes, generator=self.rng)
                    tgt_pos = tgt_pos[perm[:-drop]]

            src_graphs.append(Data(pos=base.pos))
            tgt_graphs.append(Data(pos=tgt_pos))
            labels.append(torch.tensor(spec.noise, dtype=torch.float32))

        src_batch = Batch.from_data_list(src_graphs)
        tgt_batch = Batch.from_data_list(tgt_graphs)
        labels_t = torch.stack(labels)
        return src_batch, tgt_batch, labels_t


class InvariantWeightHead(nn.Module):
    """Per-node weight predictor using simple SE(3)-invariant features."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Center coordinates per graph (translation invariance)
        centroids = scatter_mean(pos, batch, dim=0)
        centered = pos - centroids[batch]

        r_sq = (centered * centered).sum(dim=1, keepdim=True)
        ones = torch.ones_like(r_sq)
        feat = torch.cat([r_sq, ones], dim=1)
        weights = torch.sigmoid(self.linear(feat)).squeeze(-1)
        return weights + 1e-4  # keep strictly positive


class SegmentedKDERegressor(nn.Module):
    def __init__(self, sigma: float = 0.2) -> None:
        super().__init__()
        self.weight_head = InvariantWeightHead()
        self.sigma = sigma

    def forward(self, src: Batch, tgt: Batch) -> torch.Tensor:
        sigma = self.sigma

        w_src = self.weight_head(src.pos, src.batch)
        w_tgt = self.weight_head(tgt.pos, tgt.batch)

        loss, _, _, _ = pyg_kde_mmd_loss(
            src.pos,
            tgt.pos,
            src_batch=src.batch,
            tgt_batch=tgt.batch,
            src_w=w_src,
            tgt_w=w_tgt,
            sigma=sigma,
            center=True,
        )
        return loss


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train segmented KDE regressor")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--min-nodes", type=int, default=32)
    parser.add_argument("--max-nodes", type=int, default=160)
    parser.add_argument("--noise-max", type=float, default=1.0)
    parser.add_argument("--drop-max", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--plot", action="store_true", help="Show training curves")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"],
                        help="Computation device (CUDA currently uses non-differentiable kernel)")
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        print("[warn] CUDA path is forward-only; switching to CPU for backprop")
        device = torch.device("cpu")
    torch.manual_seed(0)

    generator = SyntheticBatchGenerator(
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        noise_max=args.noise_max,
        drop_max=args.drop_max,
        seed=123,
    )

    model = SegmentedKDERegressor(sigma=args.sigma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []

    for step in range(1, args.steps + 1):
        src_batch, tgt_batch, labels = generator.sample(args.batch)
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        labels = labels.to(device)

        losses = model(src_batch, tgt_batch)
        preds = losses
        mse = ((preds - labels) ** 2).mean()

        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

        history.append(mse.item())
        if step % 200 == 0 or step == 1:
            print(f"Step {step:4d} | train MSE={mse.item():.6f}")

    # Evaluate correlation on held-out samples
    preds_eval = []
    labels_eval = []
    with torch.no_grad():
        for _ in range(256):
            src_batch, tgt_batch, label = generator.sample(batch_size=1)
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            pred = model(src_batch, tgt_batch)
            preds_eval.append(pred.item())
            labels_eval.append(label.item())

    preds_eval_t = torch.tensor(preds_eval)
    labels_eval_t = torch.tensor(labels_eval)
    corr = torch.corrcoef(torch.stack([preds_eval_t, labels_eval_t]))[0, 1].item()
    print(f"Correlation (held-out): {corr:.4f}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history)
        axes[0].set_title("Training MSE")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")

        axes[1].scatter(labels_eval, preds_eval, alpha=0.6)
        axes[1].set_title("Target vs predicted")
        axes[1].set_xlabel("Noise level")
        axes[1].set_ylabel("Predicted KDE loss")

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
