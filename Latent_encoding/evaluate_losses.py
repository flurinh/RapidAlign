# Latent_encoding/evaluate_losses.py
"""Benchmark invariant losses on synthetic noisy graphs.

For each noise type, graph size, and strength level (none / low / medium),
we generate noisy graphs and measure how different loss functions respond,
alongside the ground-truth geometric severity from data.noise.

Outputs:
- Console table with mean ± std for each loss and severity, plus error metrics.
- CSV with full stats (means, stds, MAE, RMSE, correlation, timing, throughput).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    from Latent_encoding.data import SyntheticPointCloudDataset  # type: ignore
    from Latent_encoding.data.noise import (  # type: ignore
        NoiseConfig,
        noisify_batch,
        NOISE_DISPATCH,
    )
    from Latent_encoding.losses import (  # type: ignore
        kernel_correlation_loss_pyg,
        global_rff_mmd_loss,
        local_rff_mmd_loss,
        irrep_power_spectrum_loss,
    )
else:
    from .data import SyntheticPointCloudDataset
    from .data.noise import NoiseConfig, noisify_batch, NOISE_DISPATCH
    from .losses import (
        kernel_correlation_loss_pyg,
        global_rff_mmd_loss,
        local_rff_mmd_loss,
        irrep_power_spectrum_loss,
    )


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-graphs", type=int, default=256, help="Graphs per noise-type/level per graph size.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for sampling graphs.")
    p.add_argument("--device", type=str, default="cuda", help="Torch device.")
    p.add_argument("--levels", type=str, default="none,low,medium", help="Noise levels.")
    p.add_argument(
        "--noise-types",
        type=str,
        default="rigid,global_gaussian,local_gaussian,anisotropic_scale,drift,shear,bend,overlap",
        help="Comma-separated list of noise types to evaluate.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for dataset / noise.")
    p.add_argument(
        "--num-nodes",
        type=int,
        default=16,
        help="Fallback number of nodes per graph if --num-nodes-list is not provided.",
    )
    p.add_argument(
        "--num-nodes-list",
        type=str,
        default="",
        help="Comma-separated list of node counts to evaluate (e.g. '8,32,128,512'). "
             "If empty, uses --num-nodes only.",
    )
    p.add_argument(
        "--num-node-features",
        type=int,
        default=1,
        help="Node feature dim for synthetic dataset.",
    )
    p.add_argument(
        "--csv-output",
        type=str,
        default="Latent_encoding/loss_benchmark.csv",
        help="Path to CSV file to store full results.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra information per graph size / noise type / level.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loss registry (distance-like scalars)
# ---------------------------------------------------------------------------

def _kc_distance(pos_pred: torch.Tensor, pos_true: torch.Tensor, batch: torch.Tensor) -> float:
    """kernel_correlation_loss_pyg in [-1,0] -> distance in [0,1]."""
    loss = kernel_correlation_loss_pyg(
        pos_pred=pos_pred,
        pos_true=pos_true,
        batch_true=batch,
    )
    return float(1.0 + loss.item())


def _irrep_distance(pos_pred: torch.Tensor, pos_true: torch.Tensor, batch: torch.Tensor) -> float:
    """irrep_power_spectrum_loss in [-1,0] -> distance in [0,1]."""
    loss = irrep_power_spectrum_loss(
        pos_pred=pos_pred,
        pos_true=pos_true,
        batch_true=batch,
    )
    return float(1.0 + loss.item())


def _global_rff_distance(pos_pred: torch.Tensor, pos_true: torch.Tensor, batch: torch.Tensor) -> float:
    """global_rff_mmd_loss is already >= 0 and 0 at perfect match."""
    loss = global_rff_mmd_loss(
        pos_pred=pos_pred,
        pos_true=pos_true,
        batch_true=batch,
    )
    return float(loss.item())


def _local_rff_distance(pos_pred: torch.Tensor, pos_true: torch.Tensor, batch: torch.Tensor) -> float:
    """
    local_rff_mmd_loss returns a similarity-like score in [-1,0].
    Map to distance in [0,1].
    """
    loss = local_rff_mmd_loss(
        pos_pred=pos_pred,
        pos_true=pos_true,
        batch_true=batch,
    )
    return float(1.0 + loss.item())


LOSS_REGISTRY = {
    "kc_total": _kc_distance,
    "rff_global": _global_rff_distance,
    "rff_local": _local_rff_distance,
    "irrep_ps": _irrep_distance,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scaled_noise_config(base: NoiseConfig, level: str, noise_type: str) -> NoiseConfig:
    """Return a copy of base config scaled according to level."""
    cfg = NoiseConfig(types=(noise_type,))
    # copy scalars
    cfg.sigma_global = base.sigma_global
    cfg.sigma_local = base.sigma_local
    cfg.local_fraction = base.local_fraction
    cfg.scale_std = base.scale_std
    cfg.max_rotation_deg = base.max_rotation_deg
    cfg.translation_std = base.translation_std
    cfg.include_rigid_in_supervision = base.include_rigid_in_supervision
    cfg.drift_max = base.drift_max
    cfg.drift_fraction = base.drift_fraction
    cfg.shear_max = base.shear_max
    cfg.bend_max_deg = base.bend_max_deg
    cfg.overlap_fraction = base.overlap_fraction
    cfg.overlap_jitter_std = base.overlap_jitter_std
    cfg.alpha_node = base.alpha_node
    cfg.beta_edge = base.beta_edge
    cfg.seed = None  # caller controls RNG

    if level == "none":
        cfg.sigma_global = 0.0
        cfg.sigma_local = 0.0
        cfg.scale_std = 0.0
        cfg.drift_max = 0.0
        cfg.shear_max = 0.0
        cfg.bend_max_deg = 0.0
        cfg.overlap_fraction = 0.0
        return cfg

    if level == "low":
        scale = 0.5
    elif level == "medium":
        scale = 1.0
    else:
        raise ValueError(f"Unsupported level: {level}")

    cfg.sigma_global *= scale
    cfg.sigma_local *= scale
    cfg.scale_std *= scale
    cfg.drift_max *= scale
    cfg.shear_max *= scale
    cfg.bend_max_deg *= scale
    cfg.overlap_fraction = min(1.0, cfg.overlap_fraction * scale)
    return cfg


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _error_stats(sev: List[float], scores: List[float]) -> Tuple[float, float, float]:
    """Return (MAE, RMSE, Pearson r) between severity and scores."""
    n = min(len(sev), len(scores))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    sev_arr = sev[:n]
    sc_arr = scores[:n]
    diffs = [s - t for s, t in zip(sc_arr, sev_arr)]
    mae = sum(abs(d) for d in diffs) / n
    rmse = math.sqrt(sum(d * d for d in diffs) / n)

    if n < 2:
        return mae, rmse, float("nan")

    mean_sev = sum(sev_arr) / n
    mean_sc = sum(sc_arr) / n
    cov = sum((s - mean_sev) * (p - mean_sc) for s, p in zip(sev_arr, sc_arr)) / (n - 1)
    var_s = sum((s - mean_sev) ** 2 for s in sev_arr) / (n - 1)
    var_p = sum((p - mean_sc) ** 2 for p in sc_arr) / (n - 1)
    if var_s <= 0.0 or var_p <= 0.0:
        corr = float("nan")
    else:
        corr = cov / math.sqrt(var_s * var_p)
    return mae, rmse, corr


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Graph sizes to evaluate
    if args.num_nodes_list:
        node_sizes = sorted({
            int(x) for x in args.num_nodes_list.split(",") if x.strip()
        })
    else:
        node_sizes = [args.num_nodes]

    base_cfg = NoiseConfig()
    noise_types = [t.strip() for t in args.noise_types.split(",") if t.strip()]
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]

    # Validate noise types
    for noise_type in noise_types:
        if noise_type not in NOISE_DISPATCH:
            raise ValueError(f"Unknown noise type: {noise_type}")

    # Stats:
    # scores[(N, noise_type, level, loss_name)] -> list of values
    stats_scores: Dict[Tuple[int, str, str, str], List[float]] = defaultdict(list)
    # severity[(N, noise_type, level)] -> list of sev values
    stats_sev: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)
    # timing
    runtime_time: Dict[Tuple[int, str, str], float] = defaultdict(float)
    runtime_graphs: Dict[Tuple[int, str, str], int] = defaultdict(int)

    print("\n=== Evaluating losses over graph sizes ===\n")

    for num_nodes in tqdm(node_sizes, desc="Graph sizes", position=0):
        if args.verbose:
            print(f"\n>>> Graph size N = {num_nodes}")

        # Dataset of "clean" graphs for this N
        ds = SyntheticPointCloudDataset(
            num_graphs=args.num_graphs,
            num_nodes=num_nodes,
            num_node_features=args.num_node_features,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        for noise_type in tqdm(noise_types, desc=f"N={num_nodes} | noise types", position=1, leave=False):
            if args.verbose:
                print(f"\n  Noise type: {noise_type}")

            for level in levels:
                if args.verbose:
                    print(f"    Level: {level}")

                cfg = _scaled_noise_config(base_cfg, level, noise_type)
                rng = random.Random(args.seed + hash((num_nodes, noise_type, level)) % (2**31 - 1))

                key_time = (num_nodes, noise_type, level)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                graphs_processed = 0

                for batch in tqdm(
                    loader,
                    desc=f"{noise_type}:{level}",
                    position=2,
                    leave=False,
                ):
                    batch = batch.to(device)
                    if not hasattr(batch, "batch") or batch.batch is None:
                        batch.batch = torch.zeros(batch.pos.size(0), dtype=torch.long, device=device)

                    noisy_batch, severities, metas = noisify_batch(batch, cfg, rng=rng)

                    B = int(batch.batch.max().item()) + 1 if batch.batch.numel() > 0 else 0
                    graphs_processed += B

                    for b in range(B):
                        mask = batch.batch == b
                        pos_true = batch.pos[mask]
                        pos_pred = noisy_batch.pos[mask]
                        if pos_true.numel() == 0:
                            continue
                        batch_vec = torch.zeros(pos_true.size(0), dtype=torch.long, device=device)
                        sev = float(severities[b].item())
                        stats_sev[(num_nodes, noise_type, level)].append(sev)

                        for loss_name, fn in LOSS_REGISTRY.items():
                            score = fn(pos_pred, pos_true, batch_vec)
                            stats_scores[(num_nodes, noise_type, level, loss_name)].append(score)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                runtime_time[key_time] += (t1 - t0)
                runtime_graphs[key_time] += graphs_processed

    # -----------------------------------------------------------------------
    # Print leaderboard (mean ± std) and write CSV
    # -----------------------------------------------------------------------

    loss_names = list(LOSS_REGISTRY.keys())

    print("\n=== Loss leaderboard (mean ± std over graphs, plus MAE/RMSE/corr vs L_true) ===")
    header_cols = [
        "num_nodes",
        "noise_type",
        "level",
        "mean_severity",
        "std_severity",
    ]
    for ln in loss_names:
        header_cols.extend([
            f"mean_{ln}",
            f"std_{ln}",
            f"mae_{ln}",
            f"rmse_{ln}",
            f"corr_{ln}",
        ])
    header_cols.extend(["total_runtime_sec", "graphs_per_sec", "num_graphs"])
    print("\t".join(header_cols))

    rows: List[Dict[str, str]] = []

    for num_nodes in node_sizes:
        for noise_type in noise_types:
            for level in levels:
                sev_key = (num_nodes, noise_type, level)
                sev_vals = stats_sev.get(sev_key, [])
                if len(sev_vals) == 0:
                    continue

                mean_sev, std_sev = _mean_std(sev_vals)

                row_vals: List[str] = [
                    str(num_nodes),
                    noise_type,
                    level,
                    f"{mean_sev:.6f}",
                    f"{std_sev:.6f}",
                ]
                row_dict: Dict[str, str] = {
                    "num_nodes": str(num_nodes),
                    "noise_type": noise_type,
                    "level": level,
                    "mean_severity": f"{mean_sev:.6f}",
                    "std_severity": f"{std_sev:.6f}",
                }

                for ln in loss_names:
                    scores = stats_scores.get((num_nodes, noise_type, level, ln), [])
                    m, s = _mean_std(scores)
                    mae, rmse, corr = _error_stats(sev_vals, scores)
                    row_vals.extend([
                        f"{m:.6f}",
                        f"{s:.6f}",
                        f"{mae:.6f}",
                        f"{rmse:.6f}",
                        f"{corr:.4f}",
                    ])
                    row_dict[f"mean_{ln}"] = f"{m:.6f}"
                    row_dict[f"std_{ln}"] = f"{s:.6f}"
                    row_dict[f"mae_{ln}"] = f"{mae:.6f}"
                    row_dict[f"rmse_{ln}"] = f"{rmse:.6f}"
                    row_dict[f"corr_{ln}"] = f"{corr:.4f}"

                sev_key_time = (num_nodes, noise_type, level)
                t_total = runtime_time.get(sev_key_time, 0.0)
                n_graphs = runtime_graphs.get(sev_key_time, 0)
                gps = (n_graphs / t_total) if t_total > 0 and n_graphs > 0 else float("nan")

                row_vals.extend([f"{t_total:.6f}", f"{gps:.3f}", str(n_graphs)])
                row_dict["total_runtime_sec"] = f"{t_total:.6f}"
                row_dict["graphs_per_sec"] = f"{gps:.3f}"
                row_dict["num_graphs"] = str(n_graphs)

                print("\t".join(row_vals))
                rows.append(row_dict)

    # Write CSV
    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_fieldnames = header_cols
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for r in rows:
            for col in csv_fieldnames:
                r.setdefault(col, "")
            writer.writerow(r)

    print(f"\nSaved CSV results to: {csv_path.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
