import os
import sys
import torch
import numpy as np
import warnings
import subprocess
import re

# Get current file directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Function to check if CUDA is available
def check_cuda_availability():
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available. RapidAlign requires CUDA to function. "
            "Please install a compatible CUDA version for your PyTorch installation."
        )
        return False
    return True

# Detect CUDA version
def get_cuda_version():
    """Detect CUDA version"""
    try:
        # Try torch's CUDA version first (most reliable)
        if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
            return torch.version.cuda
            
        # Fallback to nvidia-smi
        try:
            nvidia_smi_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            version_match = re.search(r'CUDA Version: (\d+\.\d+)', nvidia_smi_output)
            if version_match:
                return version_match.group(1)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    except Exception as e:
        warnings.warn(f"Error detecting CUDA version: {str(e)}")
    
    # Unable to detect
    return None

# Get architectures based on CUDA version
def get_cuda_arch_flags():
    cuda_version = get_cuda_version()
    
    # Default architectures
    arch_flags = [
        "-gencode=arch=compute_60,code=sm_60",  # Pascal
        "-gencode=arch=compute_70,code=sm_70",  # Volta
        "-gencode=arch=compute_75,code=sm_75",  # Turing
    ]
    
    if cuda_version:
        # Add Ampere for CUDA 11+
        if float(cuda_version.split('.')[0]) >= 11:
            arch_flags.append("-gencode=arch=compute_80,code=sm_80")  # Ampere
        
        # Add Hopper for CUDA 12+
        if float(cuda_version.split('.')[0]) >= 12:
            arch_flags.append("-gencode=arch=compute_89,code=sm_89")  # Hopper
            arch_flags.append("-gencode=arch=compute_90,code=sm_90")  # Ada/Blackwell/Next
    
    return arch_flags

# Try to load the CUDA extension
try:
    if check_cuda_availability():
        from torch.utils.cpp_extension import load
        
        print(f"Loading RapidAlign CUDA extension (CUDA version: {get_cuda_version()})")
        
        # Load the custom CUDA extension
        # This builds and loads the extension the first time it's imported
        cuda_ext = load(
            name="rapidalign",
            sources=[
                os.path.join(script_dir, "src/pybind.cpp"),
                os.path.join(script_dir, "src/cuda_impl.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"] + get_cuda_arch_flags(),
            include_dirs=[parent_dir],
            verbose=True
        )
    else:
        # Create a dummy module for CPU-only mode (limited functionality)
        import types
        cuda_ext = types.ModuleType("rapidalign_cpu_fallback")
        cuda_ext.__dict__["procrustes_align"] = lambda *args, **kwargs: None
        cuda_ext.__dict__["icp_align"] = lambda *args, **kwargs: None
        cuda_ext.__dict__["chamfer_distance"] = lambda *args, **kwargs: None
        
        warnings.warn(
            "RapidAlign is running in CPU-only mode with very limited functionality. "
            "Please install CUDA for full functionality."
        )
        
except Exception as e:
    print(f"Error loading RapidAlign CUDA extension: {str(e)}")
    print("Please ensure your CUDA and PyTorch installations are compatible.")
    raise

class BatchedProcrustes(torch.nn.Module):
    """
    Differentiable Batched Procrustes Alignment for PyTorch
    
    This module aligns batches of source point clouds to target point clouds
    using the Procrustes algorithm. Supports point clouds with varying sizes.
    
    Forward pass computes:
    1. Optimal rotation and translation between source and target
    2. Transformed source points
    
    Backward pass computes gradients for source and target points.
    """
    
    def __init__(self):
        super(BatchedProcrustes, self).__init__()
    
    def forward(self, src_points, tgt_points, batch_indices=None):
        """
        Aligns batches of source point clouds to target point clouds
        
        Parameters:
        -----------
        src_points : torch.Tensor 
            Source points with shape [N, 3] or batched points
        tgt_points : torch.Tensor
            Target points with shape [M, 3] or batched points
        batch_indices : torch.Tensor, optional
            Batch indices for each point, shape [N] for src and [M] for tgt
            
        Returns:
        --------
        aligned_points : torch.Tensor
            Aligned source points
        transforms : tuple(torch.Tensor, torch.Tensor)
            (rotation_matrices, translation_vectors)
        """
        # Ensure inputs are on the GPU and have correct dtype
        src_points = src_points.contiguous().cuda().float()
        tgt_points = tgt_points.contiguous().cuda().float()
        
        # If batch_indices not provided, assume all points belong to one batch
        if batch_indices is None:
            batch_size = src_points.shape[0] if len(src_points.shape) == 3 else 1
            if len(src_points.shape) == 3:  # Already batched [B, N, 3]
                src_batch_idx = torch.arange(batch_size, device=src_points.device).repeat_interleave(src_points.shape[1])
                tgt_batch_idx = torch.arange(batch_size, device=tgt_points.device).repeat_interleave(tgt_points.shape[1])
                src_points = src_points.reshape(-1, 3)
                tgt_points = tgt_points.reshape(-1, 3)
            else:  # Single batch
                src_batch_idx = torch.zeros(src_points.shape[0], device=src_points.device, dtype=torch.int64)
                tgt_batch_idx = torch.zeros(tgt_points.shape[0], device=tgt_points.device, dtype=torch.int64)
        else:
            src_batch_idx = batch_indices[0] if isinstance(batch_indices, (list, tuple)) else batch_indices
            tgt_batch_idx = batch_indices[1] if isinstance(batch_indices, (list, tuple)) else batch_indices
            
        # Call the CUDA extension
        aligned_points, rotations, translations = cuda_ext.procrustes_align(
            src_points, tgt_points, src_batch_idx, tgt_batch_idx
        )
        
        return aligned_points, (rotations, translations)


class BatchedICP(torch.nn.Module):
    """
    Differentiable Batched Iterative Closest Point for PyTorch
    
    This module aligns batches of source point clouds to target point clouds
    using the ICP algorithm. Supports point clouds with varying sizes.
    
    Implements various optimizations:
    - Grid acceleration for faster nearest neighbor search
    - CUDA streams for concurrent batch processing
    - Combined optimizations for maximum performance
    
    Parameters:
    -----------
    max_iterations : int
        Maximum number of ICP iterations
    convergence_threshold : float
        Threshold to determine convergence (change in error)
    use_grid_acceleration : bool
        Whether to use grid-based acceleration for nearest neighbor search
    use_cuda_streams : bool
        Whether to use CUDA streams for concurrent batch processing
    grid_cell_size : float
        Cell size for grid acceleration (only used if grid acceleration is enabled)
    """
    
    def __init__(self, max_iterations=20, convergence_threshold=1e-6, 
                 use_grid_acceleration=True, use_cuda_streams=True, grid_cell_size=0.2):
        super(BatchedICP, self).__init__()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_grid_acceleration = use_grid_acceleration
        self.use_cuda_streams = use_cuda_streams
        self.grid_cell_size = grid_cell_size
        
    def forward(self, src_points, tgt_points, batch_indices=None):
        """
        Aligns batches of source point clouds to target point clouds using ICP
        
        Parameters:
        -----------
        src_points : torch.Tensor 
            Source points with shape [N, 3] or batched points
        tgt_points : torch.Tensor
            Target points with shape [M, 3] or batched points
        batch_indices : torch.Tensor, optional
            Batch indices for each point, shape [N] for src and [M] for tgt
            
        Returns:
        --------
        aligned_points : torch.Tensor
            Aligned source points
        transforms : tuple(torch.Tensor, torch.Tensor)
            (rotation_matrices, translation_vectors)
        """
        # Ensure inputs are on the GPU and have correct dtype
        src_points = src_points.contiguous().cuda().float()
        tgt_points = tgt_points.contiguous().cuda().float()
        
        # If batch_indices not provided, assume all points belong to one batch
        if batch_indices is None:
            batch_size = src_points.shape[0] if len(src_points.shape) == 3 else 1
            if len(src_points.shape) == 3:  # Already batched [B, N, 3]
                src_batch_idx = torch.arange(batch_size, device=src_points.device).repeat_interleave(src_points.shape[1])
                tgt_batch_idx = torch.arange(batch_size, device=tgt_points.device).repeat_interleave(tgt_points.shape[1])
                src_points = src_points.reshape(-1, 3)
                tgt_points = tgt_points.reshape(-1, 3)
            else:  # Single batch
                src_batch_idx = torch.zeros(src_points.shape[0], device=src_points.device, dtype=torch.int64)
                tgt_batch_idx = torch.zeros(tgt_points.shape[0], device=tgt_points.device, dtype=torch.int64)
        else:
            src_batch_idx = batch_indices[0] if isinstance(batch_indices, (list, tuple)) else batch_indices
            tgt_batch_idx = batch_indices[1] if isinstance(batch_indices, (list, tuple)) else batch_indices
            
        # Determine number of batches
        batch_size = max(src_batch_idx.max().item(), tgt_batch_idx.max().item()) + 1
        
        # Allocate outputs
        aligned_points = torch.zeros_like(src_points)
        rotations = torch.zeros((batch_size, 3, 3), device=src_points.device, dtype=torch.float32)
        translations = torch.zeros((batch_size, 3), device=src_points.device, dtype=torch.float32)
        
        # Call the CUDA extension with optimization parameters
        cuda_ext.icp_align(
            src_points, tgt_points, src_batch_idx, tgt_batch_idx,
            aligned_points, rotations, translations,
            self.max_iterations, self.convergence_threshold,
            self.use_grid_acceleration, self.use_cuda_streams, self.grid_cell_size
        )
        
        return aligned_points, (rotations, translations)


class BatchedChamferLoss(torch.nn.Module):
    """
    Differentiable Batched Chamfer Distance Loss for PyTorch
    
    This module computes the Chamfer distance between batches of point clouds.
    Supports point clouds with varying sizes.
    
    Parameters:
    -----------
    reduction : str
        How to reduce the loss: 'mean', 'sum', or 'none'
    use_grid_acceleration : bool
        Whether to use grid-based acceleration for nearest neighbor search
    grid_cell_size : float
        Cell size for grid acceleration (only used if grid acceleration is enabled)
    """
    
    def __init__(self, reduction='mean', use_grid_acceleration=True, grid_cell_size=0.2):
        super(BatchedChamferLoss, self).__init__()
        self.reduction = reduction
        self.use_grid_acceleration = use_grid_acceleration
        self.grid_cell_size = grid_cell_size
        
    def forward(self, src_points, tgt_points, batch_indices=None, weights=None):
        """
        Computes Chamfer distance between batches of point clouds
        
        Parameters:
        -----------
        src_points : torch.Tensor 
            Source points with shape [N, 3] or batched points
        tgt_points : torch.Tensor
            Target points with shape [M, 3] or batched points
        batch_indices : torch.Tensor, optional
            Batch indices for each point, shape [N] for src and [M] for tgt
        weights : torch.Tensor, optional
            Weights for each batch item, shape [B]
            
        Returns:
        --------
        loss : torch.Tensor
            Chamfer distance loss
        """
        # Ensure inputs are on the GPU and have correct dtype
        src_points = src_points.contiguous().cuda().float()
        tgt_points = tgt_points.contiguous().cuda().float()
        
        # If batch_indices not provided, assume all points belong to one batch
        if batch_indices is None:
            batch_size = src_points.shape[0] if len(src_points.shape) == 3 else 1
            if len(src_points.shape) == 3:  # Already batched [B, N, 3]
                src_batch_idx = torch.arange(batch_size, device=src_points.device).repeat_interleave(src_points.shape[1])
                tgt_batch_idx = torch.arange(batch_size, device=tgt_points.device).repeat_interleave(tgt_points.shape[1])
                src_points = src_points.reshape(-1, 3)
                tgt_points = tgt_points.reshape(-1, 3)
            else:  # Single batch
                src_batch_idx = torch.zeros(src_points.shape[0], device=src_points.device, dtype=torch.int64)
                tgt_batch_idx = torch.zeros(tgt_points.shape[0], device=tgt_points.device, dtype=torch.int64)
        else:
            src_batch_idx = batch_indices[0] if isinstance(batch_indices, (list, tuple)) else batch_indices
            tgt_batch_idx = batch_indices[1] if isinstance(batch_indices, (list, tuple)) else batch_indices
        
        # Determine number of batches
        batch_size = max(src_batch_idx.max().item(), tgt_batch_idx.max().item()) + 1
        
        # Allocate outputs
        distances = torch.zeros(batch_size, device=src_points.device, dtype=torch.float32)
        
        # Call the CUDA extension with optimization parameters
        cuda_ext.chamfer_distance(
            src_points, tgt_points, src_batch_idx, tgt_batch_idx,
            distances, self.use_grid_acceleration, self.grid_cell_size
        )
        
        # Apply weights if provided
        if weights is not None:
            distances = distances * weights
            
        # Apply reduction
        if self.reduction == 'mean':
            return distances.mean()
        elif self.reduction == 'sum':
            return distances.sum()
        else:
            return distances


# Example usage with PyTorch Geometric
def example_usage_pyg():
    """Example of how to use the batch alignment with PyTorch Geometric"""
    try:
        import torch_geometric as pyg
        from torch_geometric.data import Data, Batch
        
        # Create example PyG data objects
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
        pos = batch.pos
        batch_idx = batch.batch
        
        # Run alignment (assuming some target positions)
        target_pos = pos + torch.randn_like(pos) * 0.1
        
        # Create alignment modules
        procrustes = BatchedProcrustes()
        chamfer_loss = BatchedChamferLoss()
        
        # Align
        aligned_pos, transforms = procrustes(pos, target_pos, batch_idx)
        
        # Compute loss
        loss = chamfer_loss(aligned_pos, target_pos, batch_idx)
        
        print(f"Alignment loss: {loss.item()}")
        
    except ImportError:
        print("PyTorch Geometric not installed. Install with: pip install torch-geometric")


if __name__ == "__main__":
    # Simple test
    src = torch.randn(1000, 3).cuda()
    tgt = src + torch.randn_like(src) * 0.1  # Add some noise
    
    # Create some batch indices to simulate a batch with varying sizes
    src_batch = torch.cat([torch.zeros(500, dtype=torch.int64), torch.ones(500, dtype=torch.int64)]).cuda()
    tgt_batch = torch.cat([torch.zeros(500, dtype=torch.int64), torch.ones(500, dtype=torch.int64)]).cuda()
    
    # Run alignment
    procrustes = BatchedProcrustes()
    aligned, (R, t) = procrustes(src, tgt, (src_batch, tgt_batch))
    
    # Compute Chamfer distance
    chamfer = BatchedChamferLoss()
    loss = chamfer(aligned, tgt, (src_batch, tgt_batch))
    
    print(f"Chamfer loss: {loss.item()}")
    
    # Example with PyG
    example_usage_pyg()