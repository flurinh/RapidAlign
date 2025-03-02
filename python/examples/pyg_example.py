"""
Batched Point Cloud Alignment Integration with PyTorch Geometric

This example demonstrates how to use the CUDA-accelerated point cloud alignment tools
in a PyTorch Geometric context for:
1. Point cloud registration 
2. Graph alignment in GNN training
3. Performance benchmarking with different optimization settings

Requirements:
- PyTorch
- PyTorch Geometric
- rapidalign (our CUDA extension)
- matplotlib (for visualization)
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import our alignment modules
from pytorch_bindings import BatchedProcrustes, BatchedICP, BatchedChamferLoss

try:
    import torch_geometric as pyg
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    print("PyTorch Geometric not found. Some examples will be unavailable.")
    print("Install with: pip install torch-geometric torch-scatter torch-sparse")
    HAS_PYG = False


def generate_random_point_cloud(num_points, noise_level=0.1):
    """Generate a random point cloud with specified number of points"""
    points = torch.randn(num_points, 3).cuda()
    return points


def generate_test_transformation(batch_size=1):
    """Generate random rigid transformations for testing"""
    # Generate random rotations (simplified, using z-axis rotation only for clarity)
    angles = torch.rand(batch_size) * 2 * np.pi
    rotations = torch.zeros(batch_size, 3, 3).cuda()
    
    for i in range(batch_size):
        angle = angles[i]
        rotations[i] = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ]).cuda()
    
    # Generate random translations
    translations = (torch.rand(batch_size, 3) * 2 - 1).cuda()
    
    return rotations, translations


def apply_transformations(points, batch_idx, rotations, translations):
    """Apply rigid transformations to point clouds"""
    transformed_points = torch.zeros_like(points)
    
    # Apply each transformation to its corresponding batch
    for i in range(rotations.shape[0]):
        batch_mask = (batch_idx == i)
        batch_points = points[batch_mask]
        
        # Apply rotation
        transformed_batch = torch.matmul(batch_points, rotations[i].T)
        
        # Apply translation
        transformed_batch = transformed_batch + translations[i]
        
        # Store result
        transformed_points[batch_mask] = transformed_batch
    
    return transformed_points


def plot_point_clouds(source, target, aligned, title="Point Cloud Alignment"):
    """Visualize source, target, and aligned point clouds"""
    fig = plt.figure(figsize=(12, 4))
    
    # Source point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(source[:, 0].cpu(), source[:, 1].cpu(), source[:, 2].cpu(), c='blue', s=10, alpha=0.6)
    ax1.set_title('Source Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Target point cloud
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(target[:, 0].cpu(), target[:, 1].cpu(), target[:, 2].cpu(), c='red', s=10, alpha=0.6)
    ax2.set_title('Target Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Aligned and target point clouds
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(target[:, 0].cpu(), target[:, 1].cpu(), target[:, 2].cpu(), c='red', s=10, alpha=0.3)
    ax3.scatter(aligned[:, 0].cpu(), aligned[:, 1].cpu(), aligned[:, 2].cpu(), c='green', s=10, alpha=0.6)
    ax3.set_title('Aligned (green) vs Target (red)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def benchmark_optimization_options(batch_sizes=[1, 4, 16], point_counts=[1000, 5000, 10000, 20000]):
    """Benchmark different optimization options for ICP alignment"""
    print("\n===== Benchmarking Optimization Options =====")
    
    # Define optimization configurations to test
    configs = [
        {"name": "Baseline (No Optimizations)", "grid": False, "streams": False},
        {"name": "Grid Acceleration Only", "grid": True, "streams": False},
        {"name": "CUDA Streams Only", "grid": False, "streams": True},
        {"name": "Grid + Streams", "grid": True, "streams": True}
    ]
    
    results = []
    
    # Run benchmarks for each configuration
    for batch_size in batch_sizes:
        for point_count in point_counts:
            print(f"\nBenchmarking batch_size={batch_size}, point_count={point_count}")
            
            # Generate data once for fair comparison
            src_clouds = []
            tgt_clouds = []
            batch_indices = []
            
            # Create batch data
            for i in range(batch_size):
                src = generate_random_point_cloud(point_count)
                src_clouds.append(src)
                
                # Generate a target by applying a random transformation
                R, t = generate_test_transformation(1)
                tgt = apply_transformations(src, torch.zeros(point_count, dtype=torch.int64).cuda(), R, t)
                tgt_clouds.append(tgt)
                
                # Add some noise to make it more realistic
                tgt_clouds[-1] += torch.randn_like(tgt) * 0.05
                
                batch_indices.append(torch.ones(point_count, dtype=torch.int64).cuda() * i)
            
            # Concatenate data
            src_all = torch.cat(src_clouds, dim=0)
            tgt_all = torch.cat(tgt_clouds, dim=0)
            batch_idx = torch.cat(batch_indices, dim=0)
            
            # Test each configuration
            for config in configs:
                # Create ICP module with current configuration
                icp = BatchedICP(
                    max_iterations=20,
                    convergence_threshold=1e-6,
                    use_grid_acceleration=config["grid"],
                    use_cuda_streams=config["streams"],
                    grid_cell_size=0.2
                )
                
                # Warm-up run
                _ = icp(src_all, tgt_all, batch_idx)
                
                # Timed run
                torch.cuda.synchronize()
                start_time = time.time()
                
                aligned_points, (R, t) = icp(src_all, tgt_all, batch_idx)
                
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                
                # Compute error
                chamfer = BatchedChamferLoss(reduction='mean')
                error = chamfer(aligned_points, tgt_all, batch_idx).item()
                
                print(f"  {config['name']}: {elapsed*1000:.2f} ms, Error: {error:.6f}")
                
                # Record results
                results.append({
                    "batch_size": batch_size,
                    "point_count": point_count,
                    "config": config["name"],
                    "grid": config["grid"],
                    "streams": config["streams"],
                    "time_ms": elapsed * 1000,
                    "error": error
                })
    
    # Display summary
    print("\n===== Optimization Benchmark Summary =====")
    
    # Find the best configuration for each batch size and point count
    for batch_size in batch_sizes:
        for point_count in point_counts:
            subset = [r for r in results if r["batch_size"] == batch_size and r["point_count"] == point_count]
            best = min(subset, key=lambda x: x["time_ms"])
            
            print(f"Batch {batch_size}, Points {point_count}: Best is {best['config']} "
                  f"({best['time_ms']:.2f} ms, {best['error']:.6f} error)")
    
    # Plot results if matplotlib is available
    try:
        # Create a figure for the speedup comparison
        plt.figure(figsize=(15, 10))
        
        # Filter for the largest point count
        max_points = max(point_counts)
        speedup_data = {}
        
        # Get baseline times for each batch size
        baseline_times = {}
        for batch_size in batch_sizes:
            for r in results:
                if (r["batch_size"] == batch_size and r["point_count"] == max_points and
                    r["config"] == "Baseline (No Optimizations)"):
                    baseline_times[batch_size] = r["time_ms"]
        
        # Compute speedups
        for config_name in [c["name"] for c in configs]:
            if config_name == "Baseline (No Optimizations)":
                continue
                
            speedups = []
            batch_sizes_with_data = []
            
            for batch_size in batch_sizes:
                for r in results:
                    if (r["batch_size"] == batch_size and r["point_count"] == max_points and
                        r["config"] == config_name):
                        speedup = baseline_times[batch_size] / r["time_ms"]
                        speedups.append(speedup)
                        batch_sizes_with_data.append(batch_size)
            
            speedup_data[config_name] = (batch_sizes_with_data, speedups)
        
        # Plot speedups
        for config_name, (batch_sizes, speedups) in speedup_data.items():
            plt.plot(batch_sizes, speedups, 'o-', linewidth=2, label=config_name)
        
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel('Batch Size')
        plt.ylabel('Speedup vs. Baseline')
        plt.title(f'Performance Speedup with Different Optimizations (Point Count = {max_points})')
        plt.grid(True)
        plt.legend()
        
        # Set axes
        plt.xscale('log')
        plt.xticks(batch_sizes)
        plt.xlim(min(batch_sizes)*0.8, max(batch_sizes)*1.2)
        
        plt.tight_layout()
        plt.savefig('optimization_speedup.png', dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error creating plot: {e}")
    
    return results


def example_single_alignment():
    """Example of aligning a single pair of point clouds"""
    print("\n===== Single Point Cloud Alignment Example =====")
    
    # Generate a source point cloud
    num_points = 1000
    source = generate_random_point_cloud(num_points)
    
    # Generate a target by applying a random transformation
    R_true, t_true = generate_test_transformation()
    target = apply_transformations(source, torch.zeros(num_points, dtype=torch.int64).cuda(), R_true, t_true)
    
    # Add some noise to make it more realistic
    target += torch.randn_like(target) * 0.05
    
    # Align using Procrustes (for point clouds with known correspondences)
    procrustes = BatchedProcrustes()
    aligned_proc, (R_proc, t_proc) = procrustes(source, target)
    
    # Compute Procrustes error
    chamfer = BatchedChamferLoss()
    proc_error = chamfer(aligned_proc, target).item()
    print(f"Procrustes alignment error: {proc_error:.6f}")
    
    # Align using ICP (for point clouds without known correspondences)
    icp = BatchedICP(use_grid_acceleration=True, use_cuda_streams=True)
    aligned_icp, (R_icp, t_icp) = icp(source, target)
    
    # Compute ICP error
    icp_error = chamfer(aligned_icp, target).item()
    print(f"ICP alignment error: {icp_error:.6f}")
    
    # Visualize
    plot_point_clouds(source, target, aligned_icp, "ICP Alignment Results")
    
    return proc_error, icp_error


def example_batch_alignment():
    """Example of aligning batches of point clouds"""
    print("\n===== Batch Point Cloud Alignment Example =====")
    
    # Create a batch of point clouds
    batch_size = 4
    points_per_cloud = 1000
    
    src_clouds = []
    tgt_clouds = []
    batch_indices = []
    
    # Generate each batch item
    for i in range(batch_size):
        # Source point cloud
        num_points = points_per_cloud + int(torch.randn(1).item() * 100)  # Vary point count slightly
        src = generate_random_point_cloud(num_points)
        src_clouds.append(src)
        
        # Generate target by applying a random transformation
        R, t = generate_test_transformation(1)
        tgt = apply_transformations(src, torch.zeros(num_points, dtype=torch.int64).cuda(), R, t)
        tgt_clouds.append(tgt)
        
        # Add some noise to make it more realistic
        tgt_clouds[-1] += torch.randn_like(tgt) * 0.05
        
        # Set batch indices for this cloud
        batch_indices.append(torch.ones(num_points, dtype=torch.int64).cuda() * i)
    
    # Concatenate all points
    src_all = torch.cat(src_clouds, dim=0)
    tgt_all = torch.cat(tgt_clouds, dim=0)
    batch_idx = torch.cat(batch_indices, dim=0)
    
    print(f"Created batch with {batch_size} point clouds, total points: {src_all.size(0)}")
    
    # Align the batch using ICP with all optimizations
    icp = BatchedICP(use_grid_acceleration=True, use_cuda_streams=True)
    aligned_points, (rotations, translations) = icp(src_all, tgt_all, batch_idx)
    
    # Compute error for each batch item
    chamfer = BatchedChamferLoss(reduction='none')
    errors = chamfer(aligned_points, tgt_all, batch_idx)
    
    for i in range(batch_size):
        print(f"Batch item {i}: Alignment error = {errors[i].item():.6f}")
    
    print(f"Mean error across batch: {errors.mean().item():.6f}")
    
    # Visualize one of the batch items
    batch_to_viz = 0
    batch_mask = (batch_idx == batch_to_viz)
    
    plot_point_clouds(
        src_all[batch_mask], 
        tgt_all[batch_mask], 
        aligned_points[batch_mask],
        f"Batch Item {batch_to_viz} Alignment"
    )
    
    return errors


class SpatialGNN(torch.nn.Module):
    """
    Example Graph Neural Network that incorporates point cloud alignment
    
    This GNN performs message passing on node features, then transforms
    node positions. The transformed positions are aligned with original
    positions using our CUDA-accelerated alignment.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SpatialGNN, self).__init__()
        
        # GNN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        
        # Spatial transformation layer
        self.transform = torch.nn.Linear(out_channels, 3)
        
        # Alignment module (using grid acceleration and CUDA streams)
        self.alignment = BatchedICP(
            max_iterations=10,
            convergence_threshold=1e-5,
            use_grid_acceleration=True,
            use_cuda_streams=True
        )
        
        # Chamfer loss for alignment
        self.loss_fn = BatchedChamferLoss(reduction='mean')
    
    def forward(self, x, edge_index, pos, batch=None):
        """
        Forward pass with spatial alignment
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge connectivity
        pos : torch.Tensor
            Node positions (3D)
        batch : torch.Tensor, optional
            Batch indices for each node
            
        Returns:
        --------
        x : torch.Tensor
            Updated node features
        aligned_pos : torch.Tensor
            Aligned node positions
        alignment_loss : torch.Tensor
            Chamfer distance between aligned and target positions
        """
        # If batch is None, assume a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.int64, device=x.device)
        
        # Message passing
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # Transform node positions based on features
        pos_offset = self.transform(x)
        transformed_pos = pos + pos_offset
        
        # Align transformed positions with original
        aligned_pos, _ = self.alignment(transformed_pos, pos, batch)
        
        # Compute alignment loss
        alignment_loss = self.loss_fn(aligned_pos, pos, batch)
        
        return x, aligned_pos, alignment_loss


def example_gnn_with_alignment():
    """Example of using alignment in a GNN for graph learning tasks"""
    if not HAS_PYG:
        print("PyTorch Geometric not available. Skipping GNN example.")
        return
    
    print("\n===== GNN with Point Cloud Alignment Example =====")
    
    # Create a simple graph dataset
    graphs = []
    num_graphs = 10
    
    for i in range(num_graphs):
        # Random number of nodes
        num_nodes = torch.randint(20, 50, (1,)).item()
        
        # Node features
        node_features = torch.randn(num_nodes, 16).cuda()
        
        # Node positions (3D)
        node_positions = torch.randn(num_nodes, 3).cuda()
        
        # Create random edges (with some structured pattern)
        edge_index = []
        for j in range(num_nodes):
            # Connect to a few random nodes
            num_connections = min(5, num_nodes - 1)
            connected_nodes = torch.randperm(num_nodes)[:num_connections]
            for node in connected_nodes:
                if node != j:  # Avoid self-loops
                    edge_index.append([j, node.item()])
        
        edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous().cuda()
        
        # Create PyG Data object
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            pos=node_positions
        )
        
        graphs.append(graph)
    
    # Create a batch of graphs
    batch = Batch.from_data_list(graphs)
    
    # Create a GNN model with alignment
    model = SpatialGNN(
        in_channels=16,
        hidden_channels=32,
        out_channels=32
    ).cuda()
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop (simplified)
    print("Training GNN with alignment...")
    model.train()
    
    for epoch in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        node_features, aligned_pos, alignment_loss = model(
            batch.x, batch.edge_index, batch.pos, batch.batch
        )
        
        # Define a total loss (alignment loss + some other task loss)
        # In a real scenario, you would add your task-specific loss
        total_loss = alignment_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Alignment Loss: {alignment_loss.item():.6f}")
    
    print("Training complete!")
    
    # Visualize one of the graphs before and after alignment
    graph_to_viz = 0
    graph_mask = (batch.batch == graph_to_viz)
    
    # Forward pass for visualization
    model.eval()
    with torch.no_grad():
        _, aligned_pos, _ = model(batch.x, batch.edge_index, batch.pos, batch.batch)
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    
    # Original graph
    ax1 = fig.add_subplot(121, projection='3d')
    pos = batch.pos[graph_mask].cpu().numpy()
    ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=50, alpha=0.8)
    
    # Plot edges
    edge_mask = (batch.batch[batch.edge_index[0]] == graph_to_viz) & (batch.batch[batch.edge_index[1]] == graph_to_viz)
    for i in range(edge_mask.sum()):
        idx = torch.where(edge_mask)[0][i]
        start = batch.edge_index[0, idx]
        end = batch.edge_index[1, idx]
        
        if batch.batch[start] == graph_to_viz and batch.batch[end] == graph_to_viz:
            start_pos = batch.pos[start].cpu().numpy()
            end_pos = batch.pos[end].cpu().numpy()
            ax1.plot([start_pos[0], end_pos[0]],
                     [start_pos[1], end_pos[1]],
                     [start_pos[2], end_pos[2]], 'k-', alpha=0.2)
    
    ax1.set_title('Original Graph')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Aligned graph
    ax2 = fig.add_subplot(122, projection='3d')
    aligned = aligned_pos[graph_mask].cpu().numpy()
    ax2.scatter(aligned[:, 0], aligned[:, 1], aligned[:, 2], c='green', s=50, alpha=0.8)
    
    # Plot edges
    for i in range(edge_mask.sum()):
        idx = torch.where(edge_mask)[0][i]
        start = batch.edge_index[0, idx]
        end = batch.edge_index[1, idx]
        
        if batch.batch[start] == graph_to_viz and batch.batch[end] == graph_to_viz:
            start_pos = aligned_pos[start].cpu().numpy()
            end_pos = aligned_pos[end].cpu().numpy()
            ax2.plot([start_pos[0], end_pos[0]],
                     [start_pos[1], end_pos[1]],
                     [start_pos[2], end_pos[2]], 'k-', alpha=0.2)
    
    ax2.set_title('Graph After Alignment')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.suptitle('Graph Alignment in GNN')
    plt.tight_layout()
    plt.savefig('gnn_alignment.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    print("=== CUDA-Accelerated Point Cloud Alignment Examples ===\n")
    
    # Run examples
    example_single_alignment()
    example_batch_alignment()
    
    # Run GNN example (if PyTorch Geometric is available)
    if HAS_PYG:
        example_gnn_with_alignment()
    
    # Run optimization benchmarks
    benchmark_optimization_options(
        batch_sizes=[1, 4, 8],
        point_counts=[1000, 5000, 10000]
    )
    
    print("\nAll examples completed successfully!")