# Slot-Latent Autoencoder Guide (v2)

This doc summarizes the synthetic setup, model components, and the SE(3)-invariant kernel loss that replaces the old L2 reconstruction objective.

## 1. Synthetic Dataset
- `data_synthetic.py` defines `SyntheticPointCloudDataset`.
- Graphs are fully connected, directed, and have a fixed node count `N` (start with 8 or 16).
- Node coordinates are sampled from `𝒩(0, I)` and re-centered per graph.
- Node features default to all ones; extend later with chemistry/colors as needed.
- This controlled playground keeps the focus on validating the latent pipeline before using real data.

### QM9 Molecule Dataset (New)
- `data/qm9.py` exposes `QM9PointCloudDataset`, a thin wrapper around PyG's `QM9` dataset.
- The loader auto-downloads QM9 to `Latent_encoding/data/qm9` (configurable via `qm9_root`).
- Deterministic splits (80/10/10 by default) are enforced via `qm9_split_seed` so train/val/test are reproducible.
- Positions are optionally centered per molecule (`qm9_center`) and graphs exceeding `qm9_max_nodes` are filtered to stay compatible with the decoder length (29 atoms by default).
- Use `python Latent_encoding/train.py --config Latent_encoding/config/qm9.json` for a ready-made setup with sensible batch sizes and kernel configs.

## 2. Model Architecture (Scalar Prototype)
Components remain the same as the initial design:
1. **NodeBackbone** – scalar `TransformerConv` stack using pairwise distances as edge features.
2. **SlotAttention** – pools invariant node features into `K` latent slots.
3. **GraphSlotEncoder** – runs the backbone and updates slots after each layer.
4. **PointCloudDecoder** – MLP mapping slots → positions.
5. **GraphAutoencoder** – wraps encoder + decoder.

The big change is the training loss: we drop L2 and use the distance-based kernel below.

## 3. SE(3)-Invariant, Correspondence-Free Kernel Loss
`Latent_encoding/losses/kernel_correlation.py` exposes:
```python
from Latent_encoding.losses.kernel_correlation import (
    kernel_correlation_loss_pyg,
    global_distance_kernel_loss_pyg,
    local_distance_kernel_loss_pyg,
)
```

### 3.1 How it Works
Inputs:
- `pos_true`, `pos_pred`: `[total_nodes, 3]` tensors (PyG-style).
- `batch_true` (and optional `batch_pred`).
- Optional configs for global/local descriptors.

Steps per batch graph:
1. Pack points into dense `[B, N, 3]` with `to_dense_batch`.
2. (Optional) center each graph → translation invariance.
3. Compute intra-graph pairwise distances `d_ij`.
4. Build two descriptors:
   - **Global**: RBF histogram over all `d_ij` (coarse shape).
   - **Local**: RBF histogram per node over neighbor distances (fine structure).
5. Define cosine-style kernels on those descriptors and average their similarities.
6. Return negative mean similarity (higher similarity → lower loss).

Because we only use distances:
- Rotation invariance holds automatically.
- Summing over all pairs/nodes makes it permutation- and correspondence-free.
- This mirrors CVO / EquivAlign ideas (point clouds as functions in an RKHS). 

Reference: arXiv:2407.20223

### 3.2 API We Use
```python
loss = kernel_correlation_loss_pyg(
    pos_pred,
    pos_true,
    batch_true,
    batch_pred=None,
    center=True,
    lambda_global=1.0,
    lambda_local=1.0,
    global_config=dict(n_bins=32, r_max=None, gamma=None, normalize=True),
    local_config=dict(num_bins=16, r_max=None, gamma=None,
                      radius=None, k_max=None, tau=1.0, normalize=True),
)
```
- `center=True`: translation invariance.
- `lambda_*`: weight global vs. local terms.
- `*_config`: tweak histogram resolution, neighbor radius/`k`, etc.

## 4. Training Loop Integration
PyG `DataLoader` yields a batch with:
- `batch.pos`: `[total_nodes, 3]`
- `batch.batch`: `[total_nodes]` graph IDs.

Current decoder outputs `[B, num_nodes, 3]`. Flatten predictions and build a matching batch vector:
```python
pos_true = batch.pos
batch_true = batch.batch
B = int(batch_true.max().item()) + 1
num_nodes = pos_true.size(0) // B
pos_hat = model(batch)                     # [B, num_nodes, 3]
pos_pred = pos_hat.reshape(-1, 3)
batch_pred = torch.arange(B, device=pos_true.device).repeat_interleave(num_nodes)
loss = kernel_correlation_loss_pyg(
    pos_pred=pos_pred,
    pos_true=pos_true,
    batch_true=batch_true,
    batch_pred=batch_pred,
    center=True,
    lambda_global=1.0,
    lambda_local=1.0,
)
```
This single scalar replaces the old L2, providing SE(3)-invariant supervision without correspondences.

> Tip: you can load hyperparameters from `Latent_encoding/config/*.json` via `--config`. For example,
> `python Latent_encoding/train.py --config Latent_encoding/config/baseline.json` seeds all flags from the config, and you can override any of them on the command line.

Training now includes a 10-epoch warmup plus cosine LR annealing (set `warmup_epochs` in the config) and early-recovery patience (default 30 epochs): if the validation loss doesn’t improve for `patience` epochs, the script reloads the best checkpoint and continues.

## 5. cuEquivariance Hooks (Next Step)
Once CUDA is wired back in, replace `NodeBackbone` with a cuEquivariance stack:
- Treat coordinates as `1o` irreps, scalar features as `0e`.
- Use `cuequivariance_torch` layers (tensor products, spherical harmonics) to build equivariant message passing.
- Project equivariant features back to scalars (norms or contractions) before SlotAttention.

You can prototype kernels under `equivariance/`—e.g., quick scripts to ensure imports/libraries work before modifying the autoencoder backbone.

## 6. Visualization / Debug Helpers
- `Latent_encoding/visualize_graph_reconstruction.py`: loads a trained AE, reconstructs either synthetic or QM9 molecules (`--dataset` flag), aligns predictions via Kabsch, and saves a Plotly HTML with per-graph losses.
- `Latent_encoding/make_alignment_demo.py`: sanity-check script generating `alignment_demo.html` for a single rotated/translated cloud.

Use these to eyeball reconstructions vs. loss values and to verify alignment before touching real datasets.
