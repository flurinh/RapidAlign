# Autoencoder Architecture Blueprint

This document summarizes the proposed end-to-end graph autoencoder that uses
RapidAlign's segmented KDE/MMD loss as the structural similarity metric.

## High-Level Flow
- **Input**: Noisy/partial point cloud (with optional node features).
- **Encoder**: Hierarchical SE(3)-equivariant GNN (e.g. e3nn) produces per-node
  embeddings and a global latent vector `z`.
- **Latent Heads**:
  - `n̂_head(z)`: predicts node count (for inference). During training we feed
    the true `N` into the decoder.
  - Optional auxiliary heads (e.g. for KDE weights or σ scaling).
- **Decoder** (two stages conditioned on `z`):
  1. **Coarse diffusion**: transforms a noisy/prior point cloud toward the
     target geometry using time-conditioned equivariant denoisers.
  2. **Flow matching**: starting from diffusion output, an ODE-style velocity
     field refines the point cloud to high fidelity.
- **Losses**:
  - Diffusion noise prediction (or score matching).
  - Flow velocity supervision.
  - Segmented KDE/MMD loss at selected coarse and fine timesteps.
  - Size prediction penalty `|n̂ - N|` and optional latent regularizers.

## Encoder Details
- Stack equivariant message-passing layers (EGNN, SEGNN, or e3nn tensor
  products) across multiple resolutions to capture local and global structure.
- Pool node features (mean/max or attention) to obtain `z_global`.
- Retain node embeddings for downstream weighting or feature reconstruction.
- `n̂_head`: small MLP on `z_global` predicting log-node-count; clamp and round
  at inference.

## Decoder Stage 1: Diffusion
- Initialize a point cloud from `z` (e.g. latent grid decoded with an MLP or
  learned prior).
- Run a small diffusion chain `t ∈ [T_max, T_mid]`; each denoiser is
  conditioned on `(pos_t, t, z)` and produces residuals/velocities.
- Apply the segmented KDE loss on the denoised outputs at selected timesteps
  plus the usual diffusion objective.

## Decoder Stage 2: Flow Matching
- Use the coarse output as the starting state `pos_mid`.
- Learn a time-conditioned velocity field `fθ(pos_τ, τ, z)` that pushes the
  point cloud toward the clean target; integrate with a few Euler steps.
- Loss combines flow supervision with the KDE loss on intermediate/final states.

## Segmented KDE Loss Integration
- Use the ragged CUDA autograd path (`segmented_kde_loss`) to compare decoder
  outputs to the clean point set.
- Optionally attach a lightweight invariant head to generate per-node weights
  from encoder features; pass these into the KDE to emphasise important regions.
- Fit kernel parameters (σ bands, weight head) on synthetic noise/masking before
  training the autoencoder to ensure monotonic behaviour.

## Training Workflow
1. Sample clean graphs and generate noisy/masked versions at different levels.
2. Encode clean graph → `z`, node embeddings, size prediction.
3. Decode via diffusion + flow while conditioning on `z` (feed true `N` during
   training).
4. Aggregate losses (diffusion, flow, KDE, size, regularisers), backprop.
5. Validate with KDE sweeps, Chamfer/Procrustes metrics, and qualitative plots.

## Optional Extensions
- VAE-style latent regularisation for generative sampling.
- Multi-band KDE with learned σ schedule conditioned on timestep or latent.
- Endpoint feature reconstruction (node attributes, edge masks) supervised with
  auxiliary heads.
- CPU fallback kernels allow experimentation without GPU.

This blueprint keeps SE(3) equivariance end-to-end, uses the latent vector to
control both graph size and reconstruction, and relies on RapidAlign's KDE
loss to ensure the decoded structure matches the original geometry.
