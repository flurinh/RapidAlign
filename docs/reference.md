# API Reference

## Core Classes

### BatchedProcrustes

```python
class BatchedProcrustes(torch.nn.Module)
```

A PyTorch module for batched Procrustes alignment of point clouds with known correspondences.

#### Methods

**`__init__(self)`**

Initialize the Procrustes alignment module.

**`forward(self, src_points, tgt_points, batch_indices=None)`**

Aligns batches of source point clouds to target point clouds.

Parameters:
- `src_points` (`torch.Tensor`): Source points with shape [N, 3] or batched points [B, N, 3]
- `tgt_points` (`torch.Tensor`): Target points with shape [M, 3] or batched points [B, M, 3]
- `batch_indices` (`torch.Tensor`, optional): Batch indices for each point, shape [N] for src and [M] for tgt. Only needed if not using batched points format.

Returns:
- `aligned_points` (`torch.Tensor`): Aligned source points with same shape as src_points
- `transforms` (`tuple(torch.Tensor, torch.Tensor)`): Tuple containing:
  - `rotation_matrices` (`torch.Tensor`): Rotation matrices with shape [B, 3, 3]
  - `translation_vectors` (`torch.Tensor`): Translation vectors with shape [B, 3]

### BatchedICP

```python
class BatchedICP(torch.nn.Module)
```

A PyTorch module for batched Iterative Closest Point (ICP) alignment of point clouds without known correspondences.

#### Methods

**`__init__(self, max_iterations=20, convergence_threshold=1e-6, use_grid_acceleration=True, use_cuda_streams=True, grid_cell_size=0.2)`**

Initialize the ICP alignment module.

Parameters:
- `max_iterations` (`int`): Maximum number of ICP iterations
- `convergence_threshold` (`float`): Threshold to determine convergence (change in error)
- `use_grid_acceleration` (`bool`): Whether to use grid-based acceleration for nearest neighbor search
- `use_cuda_streams` (`bool`): Whether to use CUDA streams for concurrent batch processing
- `grid_cell_size` (`float`): Cell size for grid acceleration (only used if grid acceleration is enabled)

**`forward(self, src_points, tgt_points, batch_indices=None)`**

Aligns batches of source point clouds to target point clouds using ICP.

Parameters:
- `src_points` (`torch.Tensor`): Source points with shape [N, 3] or batched points [B, N, 3]
- `tgt_points` (`torch.Tensor`): Target points with shape [M, 3] or batched points [B, M, 3]
- `batch_indices` (`torch.Tensor`, optional): Batch indices for each point, shape [N] for src and [M] for tgt. Only needed if not using batched points format.

Returns:
- `aligned_points` (`torch.Tensor`): Aligned source points with same shape as src_points
- `transforms` (`tuple(torch.Tensor, torch.Tensor)`): Tuple containing:
  - `rotation_matrices` (`torch.Tensor`): Rotation matrices with shape [B, 3, 3]
  - `translation_vectors` (`torch.Tensor`): Translation vectors with shape [B, 3]

### BatchedChamferLoss

```python
class BatchedChamferLoss(torch.nn.Module)
```

A PyTorch module for computing the Chamfer distance between batches of point clouds.

#### Methods

**`__init__(self, reduction='mean', use_grid_acceleration=True, grid_cell_size=0.2)`**

Initialize the Chamfer distance loss module.

Parameters:
- `reduction` (`str`): How to reduce the loss: 'mean', 'sum', or 'none'
- `use_grid_acceleration` (`bool`): Whether to use grid-based acceleration for nearest neighbor search
- `grid_cell_size` (`float`): Cell size for grid acceleration (only used if grid acceleration is enabled)

**`forward(self, src_points, tgt_points, batch_indices=None, weights=None)`**

Computes Chamfer distance between batches of point clouds.

Parameters:
- `src_points` (`torch.Tensor`): Source points with shape [N, 3] or batched points [B, N, 3]
- `tgt_points` (`torch.Tensor`): Target points with shape [M, 3] or batched points [B, M, 3]
- `batch_indices` (`torch.Tensor`, optional): Batch indices for each point, shape [N] for src and [M] for tgt. Only needed if not using batched points format.
- `weights` (`torch.Tensor`, optional): Weights for each batch item, shape [B]

Returns:
- `loss` (`torch.Tensor`): Chamfer distance loss, scalar if reduction is 'mean' or 'sum', tensor of shape [B] if reduction is 'none'

## Low-Level Functions

### CUDA Extension Functions

**`procrustes_align(src_points, tgt_points, src_batch_idx, tgt_batch_idx)`**

Low-level CUDA function for Procrustes alignment.

**`icp_align(src_points, tgt_points, src_batch_idx, tgt_batch_idx, aligned_points, rotations, translations, max_iterations, convergence_threshold, use_grid_acceleration, use_cuda_streams, grid_cell_size)`**

Low-level CUDA function for ICP alignment.

**`chamfer_distance(src_points, tgt_points, src_batch_idx, tgt_batch_idx, distances, use_grid_acceleration, grid_cell_size)`**

Low-level CUDA function for computing Chamfer distance.

## Performance Optimization Parameters

### Grid Acceleration

The `grid_cell_size` parameter in ICP and Chamfer distance calculations controls the trade-off between speed and accuracy:

| Cell Size | Effect                                                    |
|-----------|------------------------------------------------------------|
| Smaller   | More accurate results, slower for sparse point clouds       |
| Larger    | Faster execution, may miss some nearest neighbors           |

Recommended values:
- For high accuracy: 0.05-0.1
- For speed: 0.2-0.5
- Balanced: 0.1-0.2

### CUDA Streams

The `use_cuda_streams` parameter enables concurrent execution of operations for different batch elements. This is most effective when:

- Processing large batches (8+ items)
- Each point cloud has a moderate number of points
- Your GPU has sufficient compute resources

When processing a single point cloud or very small batches, CUDA streams may provide minimal performance improvement.