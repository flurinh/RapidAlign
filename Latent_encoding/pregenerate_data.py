#!/usr/bin/env python3
# Latent_encoding/pregenerate_data.py
"""
Pregenerate training data for Fast KC Loss calibration.

This script:
1. Loads synthetic graphs.
2. Applies noise (calculating exact L_true).
3. Pre-computes the Kernel Mean Embeddings (fingerprints).
4. Saves a compact .pt file containing (embeddings_diff, targets).

Usage:
    python -m Latent_encoding.pregenerate_data --output kc_data_100k.pt --num-samples 100000
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

# Handle imports
if __package__ in (None, ""):
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from data.synthetic import SyntheticPointCloudDataset
from data.noise import NoiseConfig, noisify_batch
from losses.trainable_kc import ParameterizedKCLoss, ParameterizedKCConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("kc_training_data.pt"))
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)

    # Kernel Config (MUST MATCH what you intend to train)
    parser.add_argument("--num-scales", type=int, default=32)
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=10.0)

    # Graph config
    parser.add_argument("--min-nodes", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=64)

    # Noise Config
    parser.add_argument("--sigma-global", type=float, default=0.3)
    parser.add_argument("--sigma-local", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_kernel_embedding(encoder: ParameterizedKCLoss, pos_dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Extract kernel embedding, compatible with both old and new versions of ParameterizedKCLoss.

    Old version has: _get_kernel_embedding(pos, mask) -> [B, num_scales]
    New version has: _get_rich_embedding(pos, mask, edge_index, batch) -> [B, 2*num_scales + 3]
    """
    # Try the old API first
    if hasattr(encoder, '_get_kernel_embedding'):
        return encoder._get_kernel_embedding(pos_dense, mask)

    # Try the new API (without edge_index for global-only features)
    elif hasattr(encoder, '_get_rich_embedding'):
        # New version expects edge_index and batch, but we can pass None for kNN fallback
        # However, the signature is different - it expects sparse inputs
        # We need to handle this differently

        # For pregeneration, we'll compute a simplified embedding manually
        # using the same logic as _get_kernel_embedding
        return _compute_kernel_embedding_manual(encoder, pos_dense, mask)

    else:
        raise AttributeError(
            "ParameterizedKCLoss has neither '_get_kernel_embedding' nor '_get_rich_embedding'. "
            "Please check your trainable_kc.py version."
        )


def _compute_kernel_embedding_manual(encoder: ParameterizedKCLoss, pos: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
    """
    Manually compute kernel embedding (compatible with any version).

    Args:
        encoder: The KC loss module (to access sigmas buffer)
        pos: [B, N_max, 3] Dense node positions
        mask: [B, N_max] Boolean mask

    Returns:
        embedding: [B, num_scales]
    """
    # Get sigmas from encoder
    sigmas = encoder.sigmas  # [num_scales]

    # [B, N, N] Pairwise Euclidean Distances
    dist = torch.cdist(pos, pos)

    # Handle Masking
    # mask_2d[b, i, j] is True only if both nodes i and j are valid
    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)

    # Exclude self-loops (diagonal is always 0 distance)
    B, N, _ = dist.shape
    eye = torch.eye(N, device=dist.device).unsqueeze(0)
    valid_pairs = mask_2d.float() * (1.0 - eye)

    # Reshape for broadcasting against scales
    dist_expanded = dist.unsqueeze(-1)  # [B, N, N, 1]
    sigmas_expanded = sigmas.view(1, 1, 1, -1)  # [1, 1, 1, num_scales]

    # Apply Gaussian Kernels: exp(-d^2 / (2 * sigma^2))
    k_vals = torch.exp(-dist_expanded.pow(2) / (2 * sigmas_expanded.pow(2)))  # [B, N, N, num_scales]

    # Zero out invalid pairs
    k_vals = k_vals * valid_pairs.unsqueeze(-1)

    # Sum over all pairs to get the "spectrum" of distances
    sum_k = k_vals.sum(dim=(1, 2))  # [B, num_scales]

    # Normalize by number of valid pairs to be size-invariant
    num_valid = valid_pairs.sum(dim=(1, 2))  # [B]
    num_valid = num_valid.clamp_min(1.0).unsqueeze(-1)  # [B, 1]

    embedding = sum_k / num_valid  # [B, num_scales]

    return embedding


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Pregenerating {args.num_samples} samples on {device}...")

    # 1. Setup Feature Extractor (Fixed Kernel Basis)
    kc_config = ParameterizedKCConfig(
        num_scales=args.num_scales,
        sigma_range=(args.sigma_min, args.sigma_max),
        learnable=False
    )
    encoder = ParameterizedKCLoss(kc_config).to(device)
    encoder.eval()

    print(f"Kernel scales: {args.num_scales}")
    print(f"Sigma range: [{args.sigma_min}, {args.sigma_max}]")

    # 2. Setup Noise Generator
    noise_config = NoiseConfig(
        sigma_global=args.sigma_global,
        sigma_local=args.sigma_local,
        use_robust_metric=True  # Ensure we target linear scale
    )

    # 3. Data Source
    ds = SyntheticPointCloudDataset(
        num_graphs=args.num_samples,
        min_num_nodes=args.min_nodes,
        max_num_nodes=args.max_nodes
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_diffs = []
    all_targets = []

    print("Processing batches...")
    for batch in tqdm(loader):
        batch = batch.to(device)

        # A. Apply Noise & Get Targets
        noisy_batch, severities, _ = noisify_batch(batch, noise_config)

        # B. Convert to dense format
        x_clean, mask_clean = to_dense_batch(batch.pos, batch.batch)
        x_noisy, mask_noisy = to_dense_batch(noisy_batch.pos, noisy_batch.batch)

        # C. Compute Fingerprints (using compatible helper)
        emb_clean = get_kernel_embedding(encoder, x_clean, mask_clean)
        emb_noisy = get_kernel_embedding(encoder, x_noisy, mask_noisy)

        # D. Compute Input Feature for MLP (Squared Difference)
        diff = (emb_clean - emb_noisy).pow(2)  # [B, num_scales]

        all_diffs.append(diff.cpu())
        all_targets.append(severities.cpu())

    # 4. Save
    features = torch.cat(all_diffs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    data = {
        "features": features,  # [N, num_scales]
        "targets": targets,  # [N]
        "config": {
            "num_scales": args.num_scales,
            "sigma_range": (args.sigma_min, args.sigma_max),
            "noise_config": {
                "sigma_global": noise_config.sigma_global,
                "sigma_local": noise_config.sigma_local,
            }
        }
    }

    torch.save(data, args.output)
    print(f"\nSaved precomputed data to {args.output}")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Target stats: mean={targets.mean():.4f}, std={targets.std():.4f}, "
          f"range=[{targets.min():.4f}, {targets.max():.4f}]")


if __name__ == "__main__":
    main()