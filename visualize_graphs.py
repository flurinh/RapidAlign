"""
Graph Visualization Script for Point Cloud Alignment

This script visualizes the results of the graph alignment algorithm
by plotting source, target, and aligned graphs from PLY files.

Requirements:
- matplotlib
- numpy
- open3d for reading PLY files

Usage:
python visualize_graphs.py

The script looks for PLY files in the current directory with names:
- cpu_src_graph.ply, cpu_tgt_graph.ply, cpu_aligned_graph.ply (for single alignment)
- cpu_src_0.ply, cpu_tgt_0.ply, cpu_aligned_0.ply, etc. (for batch alignment)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Check if Open3D is available, otherwise use a simpler PLY reader
try:
    import open3d as o3d
    has_open3d = True
except ImportError:
    has_open3d = False
    print("Open3D not found. Using simple PLY reader instead.")

def simple_read_ply(filename):
    """Simple PLY reader without using Open3D"""
    vertices = []
    edges = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    i = 0
    num_vertices = 0
    num_edges = 0
    while i < len(lines):
        if lines[i].startswith('element vertex'):
            num_vertices = int(lines[i].split()[-1])
        elif lines[i].startswith('element edge'):
            num_edges = int(lines[i].split()[-1])
        elif lines[i].startswith('end_header'):
            i += 1
            break
        i += 1
    
    # Read vertices
    for j in range(num_vertices):
        x, y, z = map(float, lines[i].split())
        vertices.append([x, y, z])
        i += 1
    
    # Read edges
    for j in range(num_edges):
        v1, v2 = map(int, lines[i].split())
        edges.append([v1, v2])
        i += 1
    
    return np.array(vertices), np.array(edges)

def read_ply(filename):
    """Read a PLY file and return vertices and edges"""
    if has_open3d:
        # Use Open3D to read PLY
        mesh = o3d.io.read_triangle_mesh(filename)
        vertices = np.asarray(mesh.vertices)
        
        # For edges, we need to parse the file ourselves
        edges = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        # Find start of edge data
        edge_start = 0
        num_vertices = 0
        num_edges = 0
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('element edge'):
                num_edges = int(line.split()[-1])
            elif line.startswith('end_header'):
                edge_start = i + 1 + num_vertices
                break
        
        # Read edge data
        for i in range(edge_start, edge_start + num_edges):
            v1, v2 = map(int, lines[i].split())
            edges.append([v1, v2])
        
        return vertices, np.array(edges)
    else:
        # Use simple PLY reader
        return simple_read_ply(filename)

def plot_graph(ax, vertices, edges, color='b', alpha=0.6, label=None):
    """Plot a 3D graph with vertices and edges"""
    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c=color, marker='o', s=30, alpha=alpha, label=label)
    
    # Plot edges
    for edge in edges:
        v1, v2 = edge
        xs = [vertices[v1, 0], vertices[v2, 0]]
        ys = [vertices[v1, 1], vertices[v2, 1]]
        zs = [vertices[v1, 2], vertices[v2, 2]]
        ax.plot(xs, ys, zs, c=color, alpha=0.4)

def visualize_single_alignment():
    """Visualize results of single graph alignment"""
    # Check if the files exist
    if not os.path.exists('cpu_src_graph.ply') or \
       not os.path.exists('cpu_tgt_graph.ply') or \
       not os.path.exists('cpu_aligned_graph.ply'):
        print("Single alignment files not found. Skipping visualization.")
        return
    
    # Read the PLY files
    src_vertices, src_edges = read_ply('cpu_src_graph.ply')
    tgt_vertices, tgt_edges = read_ply('cpu_tgt_graph.ply')
    aligned_vertices, aligned_edges = read_ply('cpu_aligned_graph.ply')
    
    # Create figure and subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Source graph
    ax1 = fig.add_subplot(131, projection='3d')
    plot_graph(ax1, src_vertices, src_edges, color='blue', label='Source')
    ax1.set_title('Source Graph')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Target graph
    ax2 = fig.add_subplot(132, projection='3d')
    plot_graph(ax2, tgt_vertices, tgt_edges, color='red', label='Target')
    ax2.set_title('Target Graph')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # Aligned + Target graph
    ax3 = fig.add_subplot(133, projection='3d')
    plot_graph(ax3, tgt_vertices, tgt_edges, color='red', alpha=0.3, label='Target')
    plot_graph(ax3, aligned_vertices, aligned_edges, color='green', alpha=0.6, label='Aligned')
    ax3.set_title('Aligned vs. Target')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    # Adjust layout and show
    plt.tight_layout()
    plt.savefig('alignment_result.png', dpi=300)
    print("Saved visualization to alignment_result.png")
    plt.close()

def visualize_batch_alignment():
    """Visualize results of batch graph alignment"""
    # Check how many batch files exist
    batch_size = 0
    while os.path.exists(f'cpu_src_{batch_size}.ply'):
        batch_size += 1
    
    if batch_size == 0:
        print("No batch alignment files found. Skipping batch visualization.")
        return
    
    print(f"Found {batch_size} batch items. Visualizing...")
    
    # Create a figure for each batch item
    for i in range(batch_size):
        # Read the PLY files
        src_vertices, src_edges = read_ply(f'cpu_src_{i}.ply')
        tgt_vertices, tgt_edges = read_ply(f'cpu_tgt_{i}.ply')
        aligned_vertices, aligned_edges = read_ply(f'cpu_aligned_{i}.ply')
        
        # Create figure and subplots
        fig = plt.figure(figsize=(18, 6))
        
        # Source graph
        ax1 = fig.add_subplot(131, projection='3d')
        plot_graph(ax1, src_vertices, src_edges, color='blue', label='Source')
        ax1.set_title(f'Source Graph {i}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # Target graph
        ax2 = fig.add_subplot(132, projection='3d')
        plot_graph(ax2, tgt_vertices, tgt_edges, color='red', label='Target')
        ax2.set_title(f'Target Graph {i}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        # Aligned + Target graph
        ax3 = fig.add_subplot(133, projection='3d')
        plot_graph(ax3, tgt_vertices, tgt_edges, color='red', alpha=0.3, label='Target')
        plot_graph(ax3, aligned_vertices, aligned_edges, color='green', alpha=0.6, label='Aligned')
        ax3.set_title(f'Aligned vs. Target {i}')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        # Adjust layout and show
        plt.tight_layout()
        plt.savefig(f'alignment_result_{i}.png', dpi=300)
        print(f"Saved visualization to alignment_result_{i}.png")
        plt.close()

if __name__ == "__main__":
    print("Visualizing graph alignment results...")
    
    # Visualize single alignment
    visualize_single_alignment()
    
    # Visualize batch alignment
    visualize_batch_alignment()
    
    print("Visualization complete!")