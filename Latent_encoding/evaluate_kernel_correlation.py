"""Benchmark kernel-correlation loss speed and sensitivity."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import torch

if __package__ in (None, ""):
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    from Latent_encoding.losses import kernel_correlation_loss_pyg  # type: ignore
else:
    from .losses import kernel_correlation_loss_pyg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda", help="torch device to test on.")
    parser.add_argument("--num-nodes", type=int, nargs="+", default=[16, 32, 64], help="List of node counts per graph.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 32, 64], help="List of PyG batch sizes to test.")
    parser.add_argument("--noise-std", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1], help="Gaussian noise levels to add to predicted clouds.")
    parser.add_argument("--bins", type=int, default=32, help="Number of global histogram bins.")
    parser.add_argument("--rmax", type=float, default=None, help="Optional max distance for global histograms.")
    parser.add_argument("--gamma", type=float, default=None, help="Optional RBF width for global histograms.")
    parser.add_argument("--lambda-global", type=float, default=1.0, help="Weight for the global loss term.")
    parser.add_argument("--lambda-local", type=float, default=1.0, help="Weight for the local loss term.")
    parser.add_argument("--local-bins", type=int, default=16, help="Number of bins for local signatures.")
    parser.add_argument("--local-rmax", type=float, default=None, help="Optional max distance for local signatures.")
    parser.add_argument("--local-gamma", type=float, default=None, help="Optional RBF width for local signatures.")
    parser.add_argument("--local-radius", type=float, default=None, help="Neighbor radius for local descriptors.")
    parser.add_argument("--local-k", type=int, default=None, help="Max neighbors per node for local descriptors.")
    parser.add_argument("--local-tau", type=float, default=1.0, help="Feature bandwidth for local kernel.")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repeats per configuration for timing.")
    parser.add_argument("--rotate", action="store_true", help="Apply random rotations to predicted clouds.")
    parser.add_argument("--translate", action="store_true", help="Apply random translations to predicted clouds.")
    parser.add_argument("--no-center", action="store_false", dest="center", help="Disable centering before kernel evaluation.")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", help="Disable normalization by auto-correlations.")
    parser.add_argument(
        "--no-local-normalize",
        action="store_false",
        dest="local_normalize",
        help="Disable normalization for the local kernel term.",
    )
    parser.set_defaults(center=True, normalize=True, local_normalize=True)
    return parser.parse_args()


def random_rotations(batch: int, device: torch.device) -> torch.Tensor:
    mats = []
    for _ in range(batch):
        q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
        mats.append(q)
    return torch.stack(mats, dim=0)


def apply_transform(
    pos: torch.Tensor,
    *,
    noise_std: float,
    rotate: bool,
    translate: bool,
) -> torch.Tensor:
    out = pos.clone()
    if rotate:
        R = random_rotations(out.size(0), out.device)
        out = torch.einsum("bij,bnj->bni", R, out)
    if translate:
        t = torch.randn(out.size(0), 1, 3, device=out.device)
        out = out + t
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    return out


def flatten_batch(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, N, _ = pos.shape
    batch = torch.arange(B, device=pos.device).repeat_interleave(N)
    return pos.reshape(-1, 3), batch


def benchmark_configuration(
    num_nodes: int,
    batch_size: int,
    noise_std: float,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    pos_true = torch.randn(batch_size, num_nodes, 3, device=device)
    pos_pred = apply_transform(
        pos_true,
        noise_std=noise_std,
        rotate=args.rotate,
        translate=args.translate,
    )
    pred_flat, batch_vec = flatten_batch(pos_pred)
    true_flat, _ = flatten_batch(pos_true)

    loss_kwargs = dict(
        center=args.center,
        lambda_global=args.lambda_global,
        lambda_local=args.lambda_local,
        global_config=dict(
            n_bins=args.bins,
            r_max=args.rmax,
            gamma=args.gamma,
            normalize=args.normalize,
        ),
        local_config=dict(
            num_bins=args.local_bins,
            r_max=args.local_rmax,
            gamma=args.local_gamma,
            radius=args.local_radius,
            k_max=args.local_k,
            tau=args.local_tau,
            normalize=args.local_normalize,
        ),
    )

    # Warm-up
    _ = kernel_correlation_loss_pyg(
        pos_pred=pred_flat,
        pos_true=true_flat,
        batch_true=batch_vec,
        **loss_kwargs,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    acc = 0.0
    for _ in range(args.repeat):
        loss = kernel_correlation_loss_pyg(
            pos_pred=pred_flat,
            pos_true=true_flat,
            batch_true=batch_vec,
            **loss_kwargs,
        )
        acc += loss.item()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) / max(args.repeat, 1)
    avg_loss = acc / max(args.repeat, 1)

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_mem = float("nan")

    return avg_loss, elapsed * 1e3, peak_mem


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"Evaluating on device: {device}")
    header = (
        f"{'nodes':>6} | {'batch':>6} | {'noise':>8} | {'loss':>10} | "
        f"{'time_ms':>9} | {'peak_mem(MB)':>14}"
    )
    print(header)
    print("-" * len(header))

    for num_nodes in args.num_nodes:
        for batch_size in args.batch_sizes:
            for noise in args.noise_std:
                loss, time_ms, peak_mem = benchmark_configuration(
                    num_nodes=num_nodes,
                    batch_size=batch_size,
                    noise_std=noise,
                    args=args,
                )
                print(
                    f"{num_nodes:6d} | {batch_size:6d} | {noise:8.3f} | "
                    f"{loss:10.6f} | {time_ms:9.3f} | {peak_mem:14.2f}"
                )


if __name__ == "__main__":
    main()
