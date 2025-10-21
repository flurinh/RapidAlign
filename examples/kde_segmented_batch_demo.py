"""Batch demo: segmented KDE on variable-sized PyG graphs.

Generates a batch of random source/target graph pairs whose node counts
vary between ``--min-nodes`` and ``--max-nodes``. Target graphs receive
noise and random node drops proportional to their assigned difficulty.

The script evaluates ``pyg_kde_mmd_loss`` once on the stacked ragged
representation and prints per-pair losses alongside node statistics.
"""
from __future__ import annotations

import argparse

import torch
from torch_geometric.data import Batch, Data

from rapidalign import pyg_kde_mmd_loss


def make_pair(num_nodes: int, noise: float, drop_frac: float, seed: int) -> tuple[Data, Data]:
    torch.manual_seed(seed)
    src_pos = torch.randn(num_nodes, 3)

    tgt_pos = src_pos.clone()
    if noise > 0:
        tgt_pos = tgt_pos + noise * torch.randn_like(tgt_pos)

    if drop_frac > 0:
        drop = min(num_nodes - 1, int(round(drop_frac * num_nodes)))
        if drop > 0:
            perm = torch.randperm(num_nodes)
            tgt_pos = tgt_pos[perm[:-drop]]

    return Data(pos=src_pos), Data(pos=tgt_pos)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Segmented KDE batch demo")
    parser.add_argument("--pairs", type=int, default=6, help="Number of graph pairs")
    parser.add_argument("--min-nodes", type=int, default=32, help="Smallest graph size")
    parser.add_argument("--max-nodes", type=int, default=160, help="Largest graph size")
    parser.add_argument("--noise-max", type=float, default=0.8,
                        help="Maximum Gaussian noise amplitude")
    parser.add_argument("--drop-max", type=float, default=0.35,
                        help="Maximum fraction of nodes dropped")
    parser.add_argument("--sigma", type=float, default=0.25, help="KDE bandwidth")
    args = parser.parse_args(argv)

    torch.manual_seed(1337)

    pairs_src: list[Data] = []
    pairs_tgt: list[Data] = []
    stats = []

    for idx in range(args.pairs):
        frac = idx / max(1, args.pairs - 1)
        num_nodes = int(round(args.min_nodes + frac * (args.max_nodes - args.min_nodes)))
        noise = frac * args.noise_max
        drop_frac = frac * args.drop_max
        src, tgt = make_pair(num_nodes, noise, drop_frac, seed=400 + idx)
        pairs_src.append(src)
        pairs_tgt.append(tgt)
        stats.append((num_nodes, tgt.pos.size(0), noise, drop_frac))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_src = Batch.from_data_list(pairs_src).to(device)
    batch_tgt = Batch.from_data_list(pairs_tgt).to(device)

    loss, Kxx, Kyy, Kxy = pyg_kde_mmd_loss(
        batch_src.pos,
        batch_tgt.pos,
        src_batch=batch_src.batch,
        tgt_batch=batch_tgt.batch,
        sigma=args.sigma,
        center=True,
    )

    loss = loss.detach().cpu()
    Kxy = Kxy.detach().cpu()

    print(f"Using device: {device}")
    print("\nPair statistics (src_nodes -> tgt_nodes | noise | drop_frac | KDE loss | Kxy)")
    for idx, (ns, nt, noise, drop) in enumerate(stats):
        print(f"  #{idx:02d}: {ns:3d} -> {nt:3d} | {noise:5.2f} | {drop:5.2f} | "
              f"{loss[idx]:.6f} | {Kxy[idx]:.6f}")

    if torch.any(loss[1:] < loss[:-1] - 1e-6):
        print("[warn] losses not strictly monotone; stochastic node drops may cause this")


if __name__ == "__main__":
    main()
