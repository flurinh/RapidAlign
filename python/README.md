# RapidAlign: Fast Batch Point Cloud Alignment for PyTorch

A high-performance CUDA-accelerated library for batched point cloud alignment and registration algorithms. Designed specifically for integration with deep learning frameworks like PyTorch and PyTorch Geometric, this library enables efficient training of models that require spatial alignment of graph structures.

## Features

- **Batched Procrustes Analysis**: Efficiently compute rigid transformations for point clouds with known correspondences
- **Batched Iterative Closest Point (ICP)**: Align point clouds without known correspondences
- **Batched Chamfer Distance**: Fast computation of alignment error metrics
- **Grid-Based Acceleration**: Efficient nearest neighbor search using spatial partitioning
- **CUDA Streams**: Concurrent batch processing for improved GPU utilization
- **Deep Learning Integration**: Seamless integration with PyTorch and PyTorch Geometric
- **Variable Batch Sizes**: Support for point clouds with varying sizes in a single batch

## Installation

### Prerequisites

- CUDA Toolkit (10.2 or newer)
- PyTorch (1.7 or newer)
- C++ compiler compatible with your CUDA version

### Install from Source

```bash
git clone https://github.com/username/rapidalign.git
cd rapidalign/python
pip install -e .
```

Or directly from the Python directory:

```bash
pip install -e /path/to/repository/python
```

## Usage

### Basic Usage

```python
import torch
from rapidalign import BatchedProcrustes, BatchedICP, BatchedChamferLoss

# Create point clouds
src_points = torch.rand(1000, 3).cuda()  # Source points
tgt_points = torch.rand(1000, 3).cuda()  # Target points

# Align using Procrustes (for point clouds with known correspondences)
procrustes = BatchedProcrustes()
aligned_points, (rotations, translations) = procrustes(src_points, tgt_points)

# Align using ICP (for point clouds without known correspondences)
icp = BatchedICP(
    max_iterations=20, 
    convergence_threshold=1e-6,
    use_grid_acceleration=True,
    use_cuda_streams=True,
    grid_cell_size=0.2
)
aligned_points, (rotations, translations) = icp(src_points, tgt_points)

# Compute Chamfer distance
chamfer = BatchedChamferLoss(reduction='mean')
loss = chamfer(aligned_points, tgt_points)
```

### Batched Processing

```python
import torch
from rapidalign import BatchedICP, BatchedChamferLoss

# Create batched point clouds
batch_size = 4
src_points = torch.rand(batch_size * 1000, 3).cuda()  # Source points
tgt_points = torch.rand(batch_size * 1000, 3).cuda()  # Target points
batch_indices = torch.repeat_interleave(
    torch.arange(batch_size, device='cuda'), 
    torch.tensor([1000] * batch_size, device='cuda')
)

# Align batch using ICP with all optimizations
icp = BatchedICP(use_grid_acceleration=True, use_cuda_streams=True)
aligned_points, (rotations, translations) = icp(src_points, tgt_points, batch_indices)

# Compute error for each batch item
chamfer = BatchedChamferLoss(reduction='none')
errors = chamfer(aligned_points, tgt_points, batch_indices)
```

### Integration with PyTorch Geometric

```python
import torch
from torch_geometric.data import Data, Batch
from rapidalign import BatchedICP, BatchedChamferLoss

# Create PyG data objects
data1 = Data(
    x=torch.randn(10, 16),  # Node features
    pos=torch.randn(10, 3),  # Node positions
    edge_index=torch.randint(0, 10, (2, 20))  # Edges
)

data2 = Data(
    x=torch.randn(15, 16),
    pos=torch.randn(15, 3),
    edge_index=torch.randint(0, 15, (2, 30))
)

# Create a batch
batch = Batch.from_data_list([data1, data2])

# Extract positions and batch indices
pos = batch.pos.cuda()
batch_idx = batch.batch.cuda()

# Run alignment
target_pos = pos + torch.randn_like(pos) * 0.1
icp = BatchedICP()
aligned_pos, transforms = icp(pos, target_pos, batch_idx)

# Compute loss
chamfer_loss = BatchedChamferLoss()
loss = chamfer_loss(aligned_pos, target_pos, batch_idx)
```

## Optimization Methods

The library implements several performance optimizations that can be enabled or disabled:

### Grid-Based Acceleration

Grid-based acceleration partitions the 3D space into uniform cells to vastly speed up nearest neighbor search operations. This reduces the complexity from O(n²) to approximately O(n).

```python
# Enable grid-based acceleration
icp = BatchedICP(use_grid_acceleration=True, grid_cell_size=0.2)
```

The `grid_cell_size` parameter controls the trade-off between speed and accuracy - smaller cells provide more accurate results but may be slower.

### CUDA Streams

CUDA streams enable concurrent execution of operations for different batch elements. This significantly improves GPU utilization when processing multiple point clouds.

```python
# Enable CUDA streams for concurrent batch processing
icp = BatchedICP(use_cuda_streams=True)
```

### Combined Optimizations

For maximum performance, especially with large batches, you can enable both optimizations:

```python
# Enable all optimizations
icp = BatchedICP(
    use_grid_acceleration=True,
    use_cuda_streams=True,
    grid_cell_size=0.2
)
```

## Examples

See the `examples` directory for more usage examples:

- `examples/pyg_example.py`: Integration with PyTorch Geometric
- `examples/benchmark.py`: Performance benchmarking of different optimization methods

## Benchmark Results

Typical speedups with optimizations enabled:

| Optimization          | Small Batch (1) | Large Batch (16) |
|-----------------------|-----------------|------------------|
| Grid Acceleration     | 3-8x            | 3-8x             |
| CUDA Streams          | 1-1.2x          | 3-5x             |
| Combined              | 3-8x            | 9-30x            |

Performance varies based on point cloud size, batch size, and hardware.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{rapidalign,
  author = {Authors},
  title = {RapidAlign: Fast Batch Point Cloud Alignment for Graph Neural Networks},
  year = {2025},
  url = {https://github.com/username/rapidalign}
}
```