# RapidAlign

A high-performance CUDA-accelerated library for fast batch point cloud and graph alignment with deep learning integration.

## Overview

RapidAlign provides optimized implementations of point cloud alignment algorithms (Procrustes, ICP, Chamfer) designed specifically for:
- Batch processing of multiple point clouds simultaneously
- Integration with deep learning frameworks (PyTorch)
- Performance optimization through grid acceleration and CUDA streams
- Geometric Graph Neural Network applications

## Key Features

- **CUDA-Accelerated**: Leverages GPU compute for fast alignment operations
- **Batched Processing**: Handle multiple point clouds with variable sizes in a single operation
- **Grid-Based Acceleration**: Spatial partitioning for O(n) nearest neighbor search
- **CUDA Streams**: Concurrent processing for optimal GPU utilization
- **PyTorch Integration**: Seamless integration with PyTorch and PyTorch Geometric
- **Production-Ready**: Comprehensive benchmarking and testing tools

## Use Cases

- **Geometric Deep Learning**: Align graph structures in 3D space during GNN training
- **Point Cloud Registration**: Fast registration of multiple 3D scans
- **Robot Perception**: Real-time alignment for robotics applications
- **3D Data Processing**: Batch operations on large point cloud datasets

## Performance

RapidAlign delivers substantial speedups over traditional implementations:

| Optimization          | Small Batch (1) | Large Batch (16) |
|-----------------------|-----------------|------------------|
| Grid Acceleration     | 3-8x            | 3-8x             |
| CUDA Streams          | 1-1.2x          | 3-5x             |
| Combined              | 3-8x            | 9-30x            |

## Installation and Usage

See the [Python README](python/README.md) for installation and usage instructions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.