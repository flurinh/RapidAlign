"""Segmented KDE/MMD demo on PyG ragged batches.

This example builds several source/target graph pairs with increasing
perturbations (noise and node dropout) and evaluates the KDE similarity
directly on the stacked PyG representation via ``pyg_kde_mmd_loss``.

The script also cross-checks the segmented output against the dense helper
to confirm numerical parity, while avoiding the O(B·N_max^2) padding during
the main computation.
"""
from __future__ import annotations

import argparse

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

from rapidalign import kde_mmd_loss_dense, pyg_kde_mmd_loss


def make_pair(base: Data, noise: float, drop: int, seed: int) -> tuple[Data, Data]:
    """Clone ``base`` and return (source, target) with desired perturbations."""
    src = Data(pos=base.pos.clone())

    torch.manual_seed(seed)
    perturbed = base.pos.clone()
    if noise > 0:
        perturbed = perturbed + noise * torch.randn_like(perturbed)
    if drop > 0:
        if drop >= perturbed.size(0):
            raise ValueError("drop must be smaller than the number of nodes")
        perm = torch.randperm(perturbed.size(0))
        perturbed = perturbed[perm[:-drop]]

    tgt = Data(pos=perturbed)
    return src, tgt


def build_pairs(specs: list[dict]) -> tuple[list[Data], list[Data]]:
    torch.manual_seed(2024)
    base = Data(pos=torch.randn(96, 3))

    sources: list[Data] = []
    targets: list[Data] = []

    for idx, spec in enumerate(specs):
        src, tgt = make_pair(base, spec["noise"], spec["drop"], seed=100 + idx)
        sources.append(src)
        targets.append(tgt)
    return sources, targets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Segmented KDE demo")
    parser.add_argument("--sigma", type=float, default=0.2, help="Gaussian bandwidth")
    parser.add_argument("--noise-max", type=float, default=0.2,
                        help="Maximum noise level (min is always 0.0)")
    parser.add_argument("--levels", type=int, default=5,
                        help="Number of noise/drop levels to evaluate")
    parser.add_argument("--max-drop", type=int, default=16,
                        help="Maximum number of nodes removed at highest noise")
    args = parser.parse_args(argv)

    levels = max(2, args.levels)
    noise_values = [args.noise_max * i / (levels - 1) for i in range(levels)]
    specs = []
    for i, noise in enumerate(noise_values):
        drop = int(round(args.max_drop * i / (levels - 1))) if args.max_drop > 0 else 0
        label = "clean" if i == 0 else f"noise={noise:.2f}"
        specs.append({"label": label, "noise": noise, "drop": drop})

    src_graphs, tgt_graphs = build_pairs(specs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_src = Batch.from_data_list(src_graphs).to(device)
    batch_tgt = Batch.from_data_list(tgt_graphs).to(device)

    loss_seg, Kxx_seg, Kyy_seg, Kxy_seg = pyg_kde_mmd_loss(
        batch_src.pos,
        batch_tgt.pos,
        src_batch=batch_src.batch,
        tgt_batch=batch_tgt.batch,
        sigma=args.sigma,
        center=True,
    )

    loss_seg_cpu = loss_seg.detach().cpu()
    print("\nSegmented KDE losses:")
    for spec, loss_val in zip(specs, loss_seg_cpu):
        print(f"  {spec['label']:>12s} (drop={spec['drop']:2d}) -> loss={loss_val:.6f}")

    if not torch.all(loss_seg_cpu[1:] >= loss_seg_cpu[:-1] - 1e-6):
        print("[warn] loss is not strictly monotonic; check noise/drop specifications")

    # Optional parity check with dense helper (ran on the same device)
    X_dense, mask_x = to_dense_batch(batch_src.pos.cpu(), batch_src.batch.cpu())
    Y_dense, mask_y = to_dense_batch(batch_tgt.pos.cpu(), batch_tgt.batch.cpu())

    loss_dense, _, _, _ = kde_mmd_loss_dense(
        X_dense.to(device),
        Y_dense.to(device),
        mask_x.to(device),
        mask_y.to(device),
        sigma=args.sigma,
        center=True,
    )

    max_delta = (loss_seg - loss_dense).abs().max().item()
    print(f"\nDense vs. segmented max |Δ| = {max_delta:.3e}")


if __name__ == "__main__":
    main()
