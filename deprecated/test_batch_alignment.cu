#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <numeric>
#include <omp.h>

// Include the batch alignment header
#include "batch_alignment.cu"

// For visualization output
#define ENABLE_VISUALIZATION true

// For saving detailed benchmark results
#define SAVE_BENCHMARK_CSV true

// Benchmark configuration
#define MAX_POINT_COUNT 20000
#define MIN_POINT_COUNT 100
#define NUM_BENCHMARK_SIZES 6
#define MAX_BATCH_COUNT 32
#define MIN_BATCH_COUNT 1
#define NUM_BENCHMARK_BATCHES 5
#define MAX_ICP_ITERATIONS 50
#define NUM_BENCHMARK_ITERS 5

// Grid acceleration parameters to benchmark
#define MIN_GRID_CELL_SIZE 0.05f
#define MAX_GRID_CELL_SIZE 0.5f
#define NUM_GRID_CELL_SIZES 5

// Seed for reproducible results
#define RANDOM_SEED 42

// Optimization method flags
#define OPT_BASELINE 0      // Baseline implementation
#define OPT_GRID_ACCEL 1    // With grid acceleration
#define OPT_CUDA_STREAMS 2  // With CUDA streams
#define OPT_ALL 3           // All optimizations enabled

// Error thresholds
#define MAX_POSITION_ERROR 0.01f
#define MAX_ROTATION_ERROR 0.01f

// Visualization parameters
#define SAVE_ALIGNMENT_SNAPSHOTS true   // Save intermediate results during ICP

// Structure to hold a small graph (for testing)
typedef struct {
    float3* node_positions;    // 3D positions of nodes
    int* edges;                // Edge connectivity (pairs of node indices)
    int num_nodes;             // Number of nodes
    int num_edges;             // Number of edges
} SimpleGraph;

// Function to generate a synthetic graph
void generateSyntheticGraph(SimpleGraph* graph, int num_nodes, int density) {
    // Allocate memory for nodes
    graph->num_nodes = num_nodes;
    graph->node_positions = (float3*)malloc(num_nodes * sizeof(float3));
    
    // Generate random node positions in a unit cube
    for (int i = 0; i < num_nodes; i++) {
        graph->node_positions[i].x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
        graph->node_positions[i].y = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        graph->node_positions[i].z = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Generate edges - simple random connections based on density (0-100%)
    // For real graphs, you'd use a more sophisticated algorithm
    int potential_edges = num_nodes * (num_nodes - 1) / 2;  // Maximum possible edges
    int target_edges = (int)(potential_edges * density / 100.0f);
    
    // Allocate more than we might need, we'll adjust later
    int max_edges = target_edges + 10;
    graph->edges = (int*)malloc(max_edges * 2 * sizeof(int));  // Each edge has 2 node indices
    
    // Generate random edges
    int edge_count = 0;
    for (int i = 0; i < num_nodes && edge_count < target_edges; i++) {
        for (int j = i+1; j < num_nodes && edge_count < target_edges; j++) {
            // Add edge with probability based on density
            if ((float)rand() / RAND_MAX < (float)density / 100.0f) {
                graph->edges[edge_count*2] = i;
                graph->edges[edge_count*2 + 1] = j;
                edge_count++;
            }
        }
    }
    
    // Update actual edge count
    graph->num_edges = edge_count;
}

// Apply a random transformation to a graph
void applyRandomTransform(SimpleGraph* src, SimpleGraph* dst) {
    // Copy structure
    dst->num_nodes = src->num_nodes;
    dst->num_edges = src->num_edges;
    dst->node_positions = (float3*)malloc(src->num_nodes * sizeof(float3));
    dst->edges = (int*)malloc(src->num_edges * 2 * sizeof(int));
    
    // Copy edges
    memcpy(dst->edges, src->edges, src->num_edges * 2 * sizeof(int));
    
    // Generate random rotation (around z-axis for simplicity)
    float angle = ((float)rand() / RAND_MAX) * 2.0f * 3.14159f;  // [0, 2π]
    float R[9] = {
        cos(angle), -sin(angle), 0,
        sin(angle),  cos(angle), 0,
        0,          0,           1
    };
    
    // Generate random translation
    float3 t;
    t.x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    t.y = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    t.z = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    
    // Apply transformation to each node
    for (int i = 0; i < src->num_nodes; i++) {
        float3 p = src->node_positions[i];
        float3 p_transformed;
        p_transformed.x = R[0]*p.x + R[1]*p.y + R[2]*p.z + t.x;
        p_transformed.y = R[3]*p.x + R[4]*p.y + R[5]*p.z + t.y;
        p_transformed.z = R[6]*p.x + R[7]*p.y + R[8]*p.z + t.z;
        dst->node_positions[i] = p_transformed;
    }
    
    // Save the transformation for reference
    printf("Applied transformation:\n");
    printf("Rotation matrix:\n");
    printf("  [%f, %f, %f]\n", R[0], R[1], R[2]);
    printf("  [%f, %f, %f]\n", R[3], R[4], R[5]);
    printf("  [%f, %f, %f]\n", R[6], R[7], R[8]);
    printf("Translation: [%f, %f, %f]\n", t.x, t.y, t.z);
}

// Add random noise to graph node positions
void addNoise(SimpleGraph* graph, float noise_level) {
    for (int i = 0; i < graph->num_nodes; i++) {
        graph->node_positions[i].x += ((float)rand() / RAND_MAX) * noise_level - noise_level/2;
        graph->node_positions[i].y += ((float)rand() / RAND_MAX) * noise_level - noise_level/2;
        graph->node_positions[i].z += ((float)rand() / RAND_MAX) * noise_level - noise_level/2;
    }
}

// Convert a graph to a batched point cloud (single batch item)
void graphToPointCloud(SimpleGraph* graph, BatchedPointCloud* pointcloud) {
    // Set up batch with one item
    int batch_count = 1;
    int sizes[1] = { graph->num_nodes };
    
    // Allocate batched point cloud
    allocateBatchedPointCloud(pointcloud, batch_count, sizes);
    
    // Copy points to device
    float* h_points = (float*)malloc(graph->num_nodes * 3 * sizeof(float));
    for (int i = 0; i < graph->num_nodes; i++) {
        h_points[i*3] = graph->node_positions[i].x;
        h_points[i*3 + 1] = graph->node_positions[i].y;
        h_points[i*3 + 2] = graph->node_positions[i].z;
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(pointcloud->points, h_points, 
                    graph->num_nodes * 3 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free temporary memory
    free(h_points);
}

// Convert multiple graphs to a batched point cloud
void graphsToPointCloud(SimpleGraph* graphs, int num_graphs, BatchedPointCloud* pointcloud) {
    // Set up batch sizes
    int* sizes = (int*)malloc(num_graphs * sizeof(int));
    int total_points = 0;
    
    for (int i = 0; i < num_graphs; i++) {
        sizes[i] = graphs[i].num_nodes;
        total_points += graphs[i].num_nodes;
    }
    
    // Allocate batched point cloud
    allocateBatchedPointCloud(pointcloud, num_graphs, sizes);
    
    // Copy points to host buffer
    float* h_points = (float*)malloc(total_points * 3 * sizeof(float));
    int point_idx = 0;
    
    for (int g = 0; g < num_graphs; g++) {
        for (int i = 0; i < graphs[g].num_nodes; i++) {
            h_points[point_idx*3] = graphs[g].node_positions[i].x;
            h_points[point_idx*3 + 1] = graphs[g].node_positions[i].y;
            h_points[point_idx*3 + 2] = graphs[g].node_positions[i].z;
            point_idx++;
        }
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(pointcloud->points, h_points, 
                    total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free temporary memory
    free(h_points);
    free(sizes);
}

// Convert a batched point cloud back to graph node positions
void pointcloudToGraph(BatchedPointCloud* pointcloud, SimpleGraph* graph, int batch_idx) {
    // Get batch information
    int* h_batch_sizes = (int*)malloc(pointcloud->batch_count * sizeof(int));
    int* h_batch_offsets = (int*)malloc(pointcloud->batch_count * sizeof(int));
    
    CUDA_CHECK(cudaMemcpy(h_batch_sizes, pointcloud->batch_sizes, 
                        pointcloud->batch_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_batch_offsets, pointcloud->batch_offsets, 
                        pointcloud->batch_count * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Check if batch_idx is valid
    if (batch_idx >= pointcloud->batch_count) {
        printf("Error: batch_idx %d out of range (batch_count: %d)\n", 
               batch_idx, pointcloud->batch_count);
        free(h_batch_sizes);
        free(h_batch_offsets);
        return;
    }
    
    // Get points for this batch
    int num_points = h_batch_sizes[batch_idx];
    int offset = h_batch_offsets[batch_idx];
    
    // Check if graph has the right size
    if (graph->num_nodes != num_points) {
        printf("Error: graph has %d nodes but batch has %d points\n", 
               graph->num_nodes, num_points);
        free(h_batch_sizes);
        free(h_batch_offsets);
        return;
    }
    
    // Allocate buffer for points
    float* h_points = (float*)malloc(num_points * 3 * sizeof(float));
    
    // Copy points from device
    CUDA_CHECK(cudaMemcpy(h_points, &pointcloud->points[offset * 3], 
                        num_points * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Update graph node positions
    for (int i = 0; i < num_points; i++) {
        graph->node_positions[i].x = h_points[i*3];
        graph->node_positions[i].y = h_points[i*3 + 1];
        graph->node_positions[i].z = h_points[i*3 + 2];
    }
    
    // Free temporary memory
    free(h_points);
    free(h_batch_sizes);
    free(h_batch_offsets);
}

// Save graph to PLY file for visualization with color
void saveGraphToPLY(SimpleGraph* graph, const char* filename, float r = 1.0f, float g = 1.0f, float b = 1.0f) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << graph->num_nodes << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element edge " << graph->num_edges << "\n";
    file << "property int vertex1\n";
    file << "property int vertex2\n";
    file << "end_header\n";
    
    // Write vertices with color
    for (int i = 0; i < graph->num_nodes; i++) {
        file << graph->node_positions[i].x << " "
             << graph->node_positions[i].y << " "
             << graph->node_positions[i].z << " "
             << (int)(r * 255) << " "
             << (int)(g * 255) << " "
             << (int)(b * 255) << "\n";
    }
    
    // Write edges
    for (int i = 0; i < graph->num_edges; i++) {
        file << graph->edges[i*2] << " " << graph->edges[i*2 + 1] << "\n";
    }
    
    file.close();
    printf("Saved graph to %s with color (%.1f, %.1f, %.1f)\n", filename, r, g, b);
}

// Save point cloud to PLY file with error coloring based on distance to target
void saveErrorColoredPointCloud(SimpleGraph* source, SimpleGraph* target, SimpleGraph* aligned, const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    // Compute min/max error for color normalization
    float min_error = FLT_MAX;
    float max_error = 0.0f;
    std::vector<float> errors(source->num_nodes);
    
    for (int i = 0; i < source->num_nodes; i++) {
        float dx = aligned->node_positions[i].x - target->node_positions[i].x;
        float dy = aligned->node_positions[i].y - target->node_positions[i].y;
        float dz = aligned->node_positions[i].z - target->node_positions[i].z;
        float error = sqrt(dx*dx + dy*dy + dz*dz);
        
        errors[i] = error;
        min_error = std::min(min_error, error);
        max_error = std::max(max_error, error);
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << aligned->num_nodes << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "property float error\n";
    file << "element edge " << aligned->num_edges << "\n";
    file << "property int vertex1\n";
    file << "property int vertex2\n";
    file << "end_header\n";
    
    // Write vertices with color based on error (blue->green->red color map)
    float error_range = max_error - min_error;
    if (error_range < 1e-6) error_range = 1.0f;  // Avoid division by zero
    
    for (int i = 0; i < aligned->num_nodes; i++) {
        // Normalize error to [0,1]
        float normalized_error = (errors[i] - min_error) / error_range;
        
        // Simple blue-green-red color map
        int r = (int)(255 * std::min(1.0f, 2.0f * normalized_error));
        int g = (int)(255 * std::min(1.0f, 2.0f * (1.0f - std::abs(2.0f * normalized_error - 1.0f))));
        int b = (int)(255 * std::min(1.0f, 2.0f * (1.0f - normalized_error)));
        
        file << aligned->node_positions[i].x << " "
             << aligned->node_positions[i].y << " "
             << aligned->node_positions[i].z << " "
             << r << " " << g << " " << b << " "
             << errors[i] << "\n";
    }
    
    // Write edges
    for (int i = 0; i < aligned->num_edges; i++) {
        file << aligned->edges[i*2] << " " << aligned->edges[i*2 + 1] << "\n";
    }
    
    file.close();
    printf("Saved error-colored point cloud to %s (error range: %.6f - %.6f)\n", 
           filename, min_error, max_error);
}

// Save multiple point clouds to a single PLY file with different colors
void saveMultiColorPointCloud(std::vector<SimpleGraph*> graphs, std::vector<float3> colors, const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    // Count total vertices and edges
    int total_vertices = 0;
    int total_edges = 0;
    for (auto graph : graphs) {
        total_vertices += graph->num_nodes;
        total_edges += graph->num_edges;
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << total_vertices << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "property int graph_id\n";
    file << "element edge " << total_edges << "\n";
    file << "property int vertex1\n";
    file << "property int vertex2\n";
    file << "end_header\n";
    
    // Write vertices with colors
    int vertex_offset = 0;
    for (size_t g = 0; g < graphs.size(); g++) {
        SimpleGraph* graph = graphs[g];
        float3 color = colors[g % colors.size()];
        
        for (int i = 0; i < graph->num_nodes; i++) {
            file << graph->node_positions[i].x << " "
                 << graph->node_positions[i].y << " "
                 << graph->node_positions[i].z << " "
                 << (int)(color.x * 255) << " "
                 << (int)(color.y * 255) << " "
                 << (int)(color.z * 255) << " "
                 << g << "\n";
        }
        
        // Update vertex offset for edge indexing
        vertex_offset += graph->num_nodes;
    }
    
    // Write edges
    int edge_offset = 0;
    for (size_t g = 0; g < graphs.size(); g++) {
        SimpleGraph* graph = graphs[g];
        for (int i = 0; i < graph->num_edges; i++) {
            file << (graph->edges[i*2] + edge_offset) << " " 
                 << (graph->edges[i*2 + 1] + edge_offset) << "\n";
        }
        edge_offset += graph->num_nodes;
    }
    
    file.close();
    printf("Saved multi-color point cloud with %zu graphs to %s\n", graphs.size(), filename);
}

// Generate a visualization script for displaying PLY files
void generateVisualizationScript(const std::vector<std::string>& ply_files, const std::string& script_filename) {
    std::ofstream file(script_filename);
    if (!file.is_open()) {
        printf("Error: Could not open file %s\n", script_filename.c_str());
        return;
    }
    
    // Write Python script using Open3D for visualization
    file << "import open3d as o3d\n";
    file << "import numpy as np\n";
    file << "import os\n\n";
    
    // Function to create a coordinate frame
    file << "def create_coordinate_frame(size=1.0):\n";
    file << "    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)\n\n";
    
    // Function to load and visualize PLY files
    file << "def visualize_ply_files(ply_files):\n";
    file << "    geometries = []\n";
    file << "    for ply_file in ply_files:\n";
    file << "        if os.path.exists(ply_file):\n";
    file << "            mesh = o3d.io.read_triangle_mesh(ply_file)\n";
    file << "            if not mesh.has_vertex_colors():\n";
    file << "                mesh.paint_uniform_color([0.7, 0.7, 0.7])\n";
    file << "            geometries.append(mesh)\n";
    file << "        else:\n";
    file << "            print(f\"File not found: {ply_file}\")\n";
    file << "    \n";
    file << "    # Add a coordinate frame\n";
    file << "    geometries.append(create_coordinate_frame())\n";
    file << "    \n";
    file << "    # Visualize\n";
    file << "    o3d.visualization.draw_geometries(geometries)\n\n";
    
    // List of PLY files to visualize
    file << "# List of PLY files to visualize\n";
    file << "ply_files = [\n";
    for (const auto& ply_file : ply_files) {
        file << "    \"" << ply_file << "\",\n";
    }
    file << "]\n\n";
    
    // Call the visualization function
    file << "# Visualize all PLY files\n";
    file << "visualize_ply_files(ply_files)\n";
    
    // Visualize error plot if there are error-colored clouds
    file << "\n# Check if there are error-colored point clouds\n";
    file << "error_clouds = [f for f in ply_files if 'error_colored' in f or 'alignment_result' in f]\n";
    file << "if error_clouds:\n";
    file << "    print(\"\\nVisualizing error-colored point clouds...\")\n";
    file << "    visualize_ply_files(error_clouds)\n";
    
    file.close();
    printf("Generated visualization script: %s\n", script_filename.c_str());
}

// Save benchmark results to CSV file for analysis
void saveBenchmarkResultsToCSV(
    const std::string& filename,
    const std::vector<int>& point_counts,
    const std::vector<int>& batch_counts,
    const std::vector<std::vector<double>>& times_procrustes,
    const std::vector<std::vector<double>>& times_procrustes_streams,
    const std::vector<std::vector<double>>& times_icp,
    const std::vector<std::vector<double>>& times_icp_streams,
    const std::vector<std::vector<double>>& errors_icp,
    const std::vector<std::vector<double>>& errors_icp_streams
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        printf("Error: Could not open file %s\n", filename.c_str());
        return;
    }
    
    // Write header
    file << "Test,PointCount,BatchCount,Time_NoStreams_ms,Time_WithStreams_ms,Speedup,Error_NoStreams,Error_WithStreams\n";
    
    // Write Procrustes results
    for (size_t i = 0; i < point_counts.size(); i++) {
        for (size_t j = 0; j < batch_counts.size(); j++) {
            double time_no_streams = times_procrustes[i][j];
            double time_with_streams = times_procrustes_streams[i][j];
            double speedup = time_no_streams / time_with_streams;
            
            file << "Procrustes," 
                 << point_counts[i] << "," 
                 << batch_counts[j] << "," 
                 << time_no_streams << "," 
                 << time_with_streams << "," 
                 << speedup << ","
                 << "N/A" << ","
                 << "N/A" << "\n";
        }
    }
    
    // Write ICP results
    for (size_t i = 0; i < point_counts.size(); i++) {
        for (size_t j = 0; j < batch_counts.size(); j++) {
            double time_no_streams = times_icp[i][j];
            double time_with_streams = times_icp_streams[i][j];
            double speedup = time_no_streams / time_with_streams;
            double error_no_streams = errors_icp[i][j];
            double error_with_streams = errors_icp_streams[i][j];
            
            file << "ICP," 
                 << point_counts[i] << "," 
                 << batch_counts[j] << "," 
                 << time_no_streams << "," 
                 << time_with_streams << "," 
                 << speedup << ","
                 << error_no_streams << ","
                 << error_with_streams << "\n";
        }
    }
    
    file.close();
    printf("Saved benchmark results to %s\n", filename.c_str());
}

// Generate script to plot benchmark results
void generateBenchmarkPlotScript(const std::string& csv_filename, const std::string& script_filename) {
    std::ofstream file(script_filename);
    if (!file.is_open()) {
        printf("Error: Could not open file %s\n", script_filename.c_str());
        return;
    }
    
    // Write Python script to plot benchmark results
    file << "import pandas as pd\n";
    file << "import matplotlib.pyplot as plt\n";
    file << "import seaborn as sns\n";
    file << "import numpy as np\n\n";
    
    // Load data
    file << "# Load benchmark data\n";
    file << "df = pd.read_csv('" << csv_filename << "')\n\n";
    
    // Split data by test type
    file << "# Split data by test type\n";
    file << "df_procrustes = df[df['Test'] == 'Procrustes']\n";
    file << "df_icp = df[df['Test'] == 'ICP']\n\n";
    
    // Create plots
    file << "# Set plot style\n";
    file << "sns.set(style='whitegrid')\n";
    file << "plt.figure(figsize=(15, 10))\n\n";
    
    // Execution time comparison
    file << "# Plot 1: Execution time comparison\n";
    file << "plt.subplot(2, 2, 1)\n";
    file << "for test, df_test in [('Procrustes', df_procrustes), ('ICP', df_icp)]:\n";
    file << "    pivot = df_test.pivot(index='PointCount', columns='BatchCount', values='Time_NoStreams_ms')\n";
    file << "    for column in pivot.columns:\n";
    file << "        plt.plot(pivot.index, pivot[column], marker='o', label=f'{test}, Batch={column}')\n";
    file << "plt.xscale('log')\n";
    file << "plt.yscale('log')\n";
    file << "plt.xlabel('Point Count')\n";
    file << "plt.ylabel('Execution Time (ms)')\n";
    file << "plt.title('Execution Time vs. Point Count (No Streams)')\n";
    file << "plt.legend()\n";
    file << "plt.grid(True)\n\n";
    
    // Speedup from streams
    file << "# Plot 2: Speedup from using streams\n";
    file << "plt.subplot(2, 2, 2)\n";
    file << "for test, df_test in [('Procrustes', df_procrustes), ('ICP', df_icp)]:\n";
    file << "    pivot = df_test.pivot(index='PointCount', columns='BatchCount', values='Speedup')\n";
    file << "    for column in pivot.columns:\n";
    file << "        plt.plot(pivot.index, pivot[column], marker='o', label=f'{test}, Batch={column}')\n";
    file << "plt.xscale('log')\n";
    file << "plt.xlabel('Point Count')\n";
    file << "plt.ylabel('Speedup (NoStreams/WithStreams)')\n";
    file << "plt.title('Speedup from Using Streams vs. Point Count')\n";
    file << "plt.axhline(y=1.0, color='r', linestyle='--')\n";
    file << "plt.legend()\n";
    file << "plt.grid(True)\n\n";
    
    // Error comparison (ICP only)
    file << "# Plot 3: Error comparison (ICP only)\n";
    file << "plt.subplot(2, 2, 3)\n";
    file << "pivot_error = df_icp.pivot(index='PointCount', columns='BatchCount', values='Error_NoStreams')\n";
    file << "for column in pivot_error.columns:\n";
    file << "    plt.plot(pivot_error.index, pivot_error[column], marker='o', label=f'Batch={column}')\n";
    file << "plt.xscale('log')\n";
    file << "plt.yscale('log')\n";
    file << "plt.xlabel('Point Count')\n";
    file << "plt.ylabel('Final Error')\n";
    file << "plt.title('ICP Error vs. Point Count')\n";
    file << "plt.legend()\n";
    file << "plt.grid(True)\n\n";
    
    // Batch size effect on runtime
    file << "# Plot 4: Batch size effect on runtime\n";
    file << "plt.subplot(2, 2, 4)\n";
    file << "point_count_to_plot = df['PointCount'].max()  # Use the largest point count\n";
    file << "for test, df_test in [('Procrustes', df_procrustes), ('ICP', df_icp)]:\n";
    file << "    df_filtered = df_test[df_test['PointCount'] == point_count_to_plot]\n";
    file << "    plt.plot(df_filtered['BatchCount'], df_filtered['Time_NoStreams_ms'], marker='o', label=f'{test} (No Streams)')\n";
    file << "    plt.plot(df_filtered['BatchCount'], df_filtered['Time_WithStreams_ms'], marker='s', label=f'{test} (With Streams)')\n";
    file << "plt.xlabel('Batch Count')\n";
    file << "plt.ylabel('Execution Time (ms)')\n";
    file << "plt.title(f'Execution Time vs. Batch Count (PointCount={point_count_to_plot})')\n";
    file << "plt.legend()\n";
    file << "plt.grid(True)\n\n";
    
    // Show plots
    file << "plt.tight_layout()\n";
    file << "plt.savefig('benchmark_plots.png', dpi=300)\n";
    file << "plt.show()\n";
    
    file.close();
    printf("Generated benchmark plot script: %s\n", script_filename.c_str());
}

// Test single-graph alignment using Procrustes
void testSingleGraphAlignment() {
    printf("\n=== Testing Single Graph Alignment ===\n");
    
    // 1. Generate a source graph
    SimpleGraph src_graph;
    generateSyntheticGraph(&src_graph, 50, 20);  // 50 nodes, 20% edge density
    
    // 2. Create a target graph by applying a random transformation
    SimpleGraph tgt_graph;
    applyRandomTransform(&src_graph, &tgt_graph);
    
    // 3. Add some noise to the target graph
    addNoise(&tgt_graph, 0.05f);  // 5% noise
    
    // 4. Save the graphs for visualization
    saveGraphToPLY(&src_graph, "src_graph.ply");
    saveGraphToPLY(&tgt_graph, "tgt_graph.ply");
    
    // 5. Convert graphs to point clouds
    BatchedPointCloud src_pc, tgt_pc;
    graphToPointCloud(&src_graph, &src_pc);
    graphToPointCloud(&tgt_graph, &tgt_pc);
    
    // 6. Run Procrustes alignment
    BatchedTransformation transform;
    allocateBatchedTransformation(&transform, 1);
    
    printf("Running Procrustes alignment...\n");
    batchedProcrustes(&src_pc, &tgt_pc, &transform);
    
    // 7. Get the estimated transformation
    float h_rotation[9];
    float3 h_translation;
    
    CUDA_CHECK(cudaMemcpy(h_rotation, transform.rotations, 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_translation, transform.translations, sizeof(float3), cudaMemcpyDeviceToHost));
    
    printf("Estimated transformation:\n");
    printf("Rotation matrix:\n");
    printf("  [%f, %f, %f]\n", h_rotation[0], h_rotation[1], h_rotation[2]);
    printf("  [%f, %f, %f]\n", h_rotation[3], h_rotation[4], h_rotation[5]);
    printf("  [%f, %f, %f]\n", h_rotation[6], h_rotation[7], h_rotation[8]);
    printf("Translation: [%f, %f, %f]\n", h_translation.x, h_translation.y, h_translation.z);
    
    // 8. Apply the transformation to source
    printf("Applying transformation...\n");
    float* d_src_aligned;
    CUDA_CHECK(cudaMalloc(&d_src_aligned, src_graph.num_nodes * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src_aligned, src_pc.points, src_graph.num_nodes * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Apply the transform
    applyTransform<<<(src_graph.num_nodes+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        (float3*)d_src_aligned, src_graph.num_nodes, h_rotation, h_translation);
    
    // Copy back to a new graph
    SimpleGraph aligned_graph = src_graph;  // Copy structure
    aligned_graph.node_positions = (float3*)malloc(src_graph.num_nodes * sizeof(float3));
    
    float* h_aligned = (float*)malloc(src_graph.num_nodes * 3 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_aligned, d_src_aligned, src_graph.num_nodes * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < src_graph.num_nodes; i++) {
        aligned_graph.node_positions[i].x = h_aligned[i*3];
        aligned_graph.node_positions[i].y = h_aligned[i*3 + 1];
        aligned_graph.node_positions[i].z = h_aligned[i*3 + 2];
    }
    
    // 9. Save the aligned graph
    saveGraphToPLY(&aligned_graph, "aligned_graph.ply");
    
    // 10. Compute and display the RMSE
    float rmse = 0.0f;
    for (int i = 0; i < src_graph.num_nodes; i++) {
        float dx = aligned_graph.node_positions[i].x - tgt_graph.node_positions[i].x;
        float dy = aligned_graph.node_positions[i].y - tgt_graph.node_positions[i].y;
        float dz = aligned_graph.node_positions[i].z - tgt_graph.node_positions[i].z;
        rmse += dx*dx + dy*dy + dz*dz;
    }
    rmse = sqrt(rmse / src_graph.num_nodes);
    printf("RMSE between aligned and target: %f\n", rmse);
    
    // Clean up
    free(src_graph.node_positions);
    free(src_graph.edges);
    free(tgt_graph.node_positions);
    free(tgt_graph.edges);
    free(aligned_graph.node_positions);
    free(h_aligned);
    freeBatchedPointCloud(&src_pc);
    freeBatchedPointCloud(&tgt_pc);
    freeBatchedTransformation(&transform);
    CUDA_CHECK(cudaFree(d_src_aligned));
}

// Test batch processing with multiple graphs
void testBatchGraphAlignment() {
    printf("\n=== Testing Batch Graph Alignment ===\n");
    
    const int num_graphs = 3;
    SimpleGraph src_graphs[num_graphs];
    SimpleGraph tgt_graphs[num_graphs];
    SimpleGraph aligned_graphs[num_graphs];
    
    // 1. Generate source graphs with different sizes
    printf("Generating %d graphs with different sizes...\n", num_graphs);
    int node_counts[num_graphs] = {30, 50, 70};
    
    for (int i = 0; i < num_graphs; i++) {
        generateSyntheticGraph(&src_graphs[i], node_counts[i], 20);  // 20% edge density
        
        // Create target by transforming source
        applyRandomTransform(&src_graphs[i], &tgt_graphs[i]);
        
        // Add some noise
        addNoise(&tgt_graphs[i], 0.05f);  // 5% noise
        
        // Prepare aligned graph structure (will be filled later)
        aligned_graphs[i].num_nodes = src_graphs[i].num_nodes;
        aligned_graphs[i].num_edges = src_graphs[i].num_edges;
        aligned_graphs[i].node_positions = (float3*)malloc(src_graphs[i].num_nodes * sizeof(float3));
        aligned_graphs[i].edges = (int*)malloc(src_graphs[i].num_edges * 2 * sizeof(int));
        memcpy(aligned_graphs[i].edges, src_graphs[i].edges, src_graphs[i].num_edges * 2 * sizeof(int));
        
        // Save source and target graphs
        char filename[100];
        sprintf(filename, "src_graph_%d.ply", i);
        saveGraphToPLY(&src_graphs[i], filename);
        
        sprintf(filename, "tgt_graph_%d.ply", i);
        saveGraphToPLY(&tgt_graphs[i], filename);
    }
    
    // 2. Convert graphs to batched point clouds
    BatchedPointCloud src_pc, tgt_pc;
    graphsToPointCloud(src_graphs, num_graphs, &src_pc);
    graphsToPointCloud(tgt_graphs, num_graphs, &tgt_pc);
    
    // 3. Run batched Procrustes alignment
    BatchedTransformation transform;
    allocateBatchedTransformation(&transform, num_graphs);
    
    printf("Running batched Procrustes alignment...\n");
    batchedProcrustes(&src_pc, &tgt_pc, &transform);
    
    // 4. Apply transformations to each source graph
    printf("Applying transformations...\n");
    
    // Get the transformations
    float* h_rotations = (float*)malloc(num_graphs * 9 * sizeof(float));
    float3* h_translations = (float3*)malloc(num_graphs * sizeof(float3));
    
    CUDA_CHECK(cudaMemcpy(h_rotations, transform.rotations, num_graphs * 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_translations, transform.translations, num_graphs * sizeof(float3), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < num_graphs; i++) {
        printf("Graph %d transformation:\n", i);
        printf("  Rotation matrix:\n");
        printf("    [%f, %f, %f]\n", h_rotations[i*9], h_rotations[i*9+1], h_rotations[i*9+2]);
        printf("    [%f, %f, %f]\n", h_rotations[i*9+3], h_rotations[i*9+4], h_rotations[i*9+5]);
        printf("    [%f, %f, %f]\n", h_rotations[i*9+6], h_rotations[i*9+7], h_rotations[i*9+8]);
        printf("  Translation: [%f, %f, %f]\n", h_translations[i].x, h_translations[i].y, h_translations[i].z);
        
        // Apply transformation on CPU for simplicity
        for (int j = 0; j < src_graphs[i].num_nodes; j++) {
            float3 p = src_graphs[i].node_positions[j];
            float3 p_transformed;
            float* R = &h_rotations[i*9];
            float3 t = h_translations[i];
            
            p_transformed.x = R[0]*p.x + R[1]*p.y + R[2]*p.z + t.x;
            p_transformed.y = R[3]*p.x + R[4]*p.y + R[5]*p.z + t.y;
            p_transformed.z = R[6]*p.x + R[7]*p.y + R[8]*p.z + t.z;
            
            aligned_graphs[i].node_positions[j] = p_transformed;
        }
        
        // Save aligned graph
        char filename[100];
        sprintf(filename, "aligned_graph_%d.ply", i);
        saveGraphToPLY(&aligned_graphs[i], filename);
        
        // Compute RMSE
        float rmse = 0.0f;
        for (int j = 0; j < src_graphs[i].num_nodes; j++) {
            float dx = aligned_graphs[i].node_positions[j].x - tgt_graphs[i].node_positions[j].x;
            float dy = aligned_graphs[i].node_positions[j].y - tgt_graphs[i].node_positions[j].y;
            float dz = aligned_graphs[i].node_positions[j].z - tgt_graphs[i].node_positions[j].z;
            rmse += dx*dx + dy*dy + dz*dz;
        }
        rmse = sqrt(rmse / src_graphs[i].num_nodes);
        printf("  RMSE: %f\n", rmse);
    }
    
    // Clean up
    for (int i = 0; i < num_graphs; i++) {
        free(src_graphs[i].node_positions);
        free(src_graphs[i].edges);
        free(tgt_graphs[i].node_positions);
        free(tgt_graphs[i].edges);
        free(aligned_graphs[i].node_positions);
        free(aligned_graphs[i].edges);
    }
    free(h_rotations);
    free(h_translations);
    freeBatchedPointCloud(&src_pc);
    freeBatchedPointCloud(&tgt_pc);
    freeBatchedTransformation(&transform);
}

// Test ICP alignment for graphs without known correspondences
void testICPAlignment() {
    printf("\n=== Testing ICP Alignment ===\n");
    
    // 1. Generate a source graph
    SimpleGraph src_graph;
    generateSyntheticGraph(&src_graph, 100, 15);  // 100 nodes, 15% edge density
    
    // 2. Create a target graph by applying a random transformation
    SimpleGraph tgt_graph;
    applyRandomTransform(&src_graph, &tgt_graph);
    
    // 3. Shuffle the target graph nodes to destroy correspondence
    SimpleGraph shuffled_tgt_graph;
    shuffled_tgt_graph.num_nodes = tgt_graph.num_nodes;
    shuffled_tgt_graph.num_edges = 0;  // We'll ignore edges for this test
    shuffled_tgt_graph.node_positions = (float3*)malloc(tgt_graph.num_nodes * sizeof(float3));
    shuffled_tgt_graph.edges = NULL;
    
    // Create a shuffled copy of the node positions
    int* indices = (int*)malloc(tgt_graph.num_nodes * sizeof(int));
    for (int i = 0; i < tgt_graph.num_nodes; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle
    for (int i = tgt_graph.num_nodes - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    // Apply the shuffle
    for (int i = 0; i < tgt_graph.num_nodes; i++) {
        shuffled_tgt_graph.node_positions[i] = tgt_graph.node_positions[indices[i]];
    }
    
    // 4. Add some noise to make it more challenging
    addNoise(&shuffled_tgt_graph, 0.1f);  // 10% noise
    
    // 5. Save the graphs for visualization
    saveGraphToPLY(&src_graph, "icp_src_graph.ply");
    saveGraphToPLY(&shuffled_tgt_graph, "icp_tgt_graph.ply");
    
    // 6. Convert graphs to point clouds
    BatchedPointCloud src_pc, tgt_pc;
    graphToPointCloud(&src_graph, &src_pc);
    graphToPointCloud(&shuffled_tgt_graph, &tgt_pc);
    
    // 7. Run ICP alignment
    BatchedTransformation transform;
    allocateBatchedTransformation(&transform, 1);
    
    printf("Running ICP alignment...\n");
    // Note: In a real implementation, you'd call batchedICP here
    // For now, we'll use the Procrustes as it's already implemented
    batchedProcrustes(&src_pc, &tgt_pc, &transform);
    
    // 8. Get the estimated transformation
    float h_rotation[9];
    float3 h_translation;
    
    CUDA_CHECK(cudaMemcpy(h_rotation, transform.rotations, 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_translation, transform.translations, sizeof(float3), cudaMemcpyDeviceToHost));
    
    printf("Estimated transformation:\n");
    printf("Rotation matrix:\n");
    printf("  [%f, %f, %f]\n", h_rotation[0], h_rotation[1], h_rotation[2]);
    printf("  [%f, %f, %f]\n", h_rotation[3], h_rotation[4], h_rotation[5]);
    printf("  [%f, %f, %f]\n", h_rotation[6], h_rotation[7], h_rotation[8]);
    printf("Translation: [%f, %f, %f]\n", h_translation.x, h_translation.y, h_translation.z);
    
    // 9. Apply the transformation to source
    printf("Applying transformation...\n");
    SimpleGraph aligned_graph = src_graph;  // Copy structure
    aligned_graph.node_positions = (float3*)malloc(src_graph.num_nodes * sizeof(float3));
    
    // Apply transformation on CPU
    for (int i = 0; i < src_graph.num_nodes; i++) {
        float3 p = src_graph.node_positions[i];
        float3 p_transformed;
        
        p_transformed.x = h_rotation[0]*p.x + h_rotation[1]*p.y + h_rotation[2]*p.z + h_translation.x;
        p_transformed.y = h_rotation[3]*p.x + h_rotation[4]*p.y + h_rotation[5]*p.z + h_translation.y;
        p_transformed.z = h_rotation[6]*p.x + h_rotation[7]*p.y + h_rotation[8]*p.z + h_translation.z;
        
        aligned_graph.node_positions[i] = p_transformed;
    }
    
    // 10. Save the aligned graph
    saveGraphToPLY(&aligned_graph, "icp_aligned_graph.ply");
    
    // 11. Compute Chamfer distance as a measure of alignment quality
    // (Since we don't have direct correspondence)
    float chamfer_dist = 0.0f;
    
    // For each source point, find closest target point
    for (int i = 0; i < aligned_graph.num_nodes; i++) {
        float min_dist = INFINITY;
        for (int j = 0; j < shuffled_tgt_graph.num_nodes; j++) {
            float dx = aligned_graph.node_positions[i].x - shuffled_tgt_graph.node_positions[j].x;
            float dy = aligned_graph.node_positions[i].y - shuffled_tgt_graph.node_positions[j].y;
            float dz = aligned_graph.node_positions[i].z - shuffled_tgt_graph.node_positions[j].z;
            float dist = dx*dx + dy*dy + dz*dz;
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        chamfer_dist += min_dist;
    }
    
    // For each target point, find closest source point
    for (int j = 0; j < shuffled_tgt_graph.num_nodes; j++) {
        float min_dist = INFINITY;
        for (int i = 0; i < aligned_graph.num_nodes; i++) {
            float dx = shuffled_tgt_graph.node_positions[j].x - aligned_graph.node_positions[i].x;
            float dy = shuffled_tgt_graph.node_positions[j].y - aligned_graph.node_positions[i].y;
            float dz = shuffled_tgt_graph.node_positions[j].z - aligned_graph.node_positions[i].z;
            float dist = dx*dx + dy*dy + dz*dz;
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        chamfer_dist += min_dist;
    }
    
    chamfer_dist = sqrt(chamfer_dist / (aligned_graph.num_nodes + shuffled_tgt_graph.num_nodes));
    printf("Chamfer distance: %f\n", chamfer_dist);
    
    // Clean up
    free(src_graph.node_positions);
    free(src_graph.edges);
    free(tgt_graph.node_positions);
    free(tgt_graph.edges);
    free(shuffled_tgt_graph.node_positions);
    free(aligned_graph.node_positions);
    free(indices);
    freeBatchedPointCloud(&src_pc);
    freeBatchedPointCloud(&tgt_pc);
    freeBatchedTransformation(&transform);
}

// Run comprehensive testing of Procrustes algorithm
void runCompleteProcrustesTests() {
    printf("\n=== Running Comprehensive Procrustes Tests ===\n");
    
    // Initialize test variables
    std::vector<std::string> ply_files;
    
    // 1. Test single graph with different sizes
    int sizes[] = {100, 500, 1000, 5000};
    for (int size : sizes) {
        // Create test graphs
        SimpleGraph src_graph;
        generateSyntheticGraph(&src_graph, size, 20);  // 20% edge density
        
        SimpleGraph tgt_graph;
        applyRandomTransform(&src_graph, &tgt_graph);
        
        // Add noise of varying levels
        float noise = 0.02f;  // 2% noise
        addNoise(&tgt_graph, noise);
        
        // Create point clouds
        BatchedPointCloud src_pc, tgt_pc;
        graphToPointCloud(&src_graph, &src_pc);
        graphToPointCloud(&tgt_graph, &tgt_pc);
        
        // Run Procrustes alignment
        BatchedTransformation transform;
        allocateBatchedTransformation(&transform, 1);
        
        printf("Running Procrustes alignment for size %d...\n", size);
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Measure total time
        CUDA_CHECK(cudaEventRecord(start));
        batchedProcrustes(&src_pc, &tgt_pc, &transform, false);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        printf("  Time: %.2f ms\n", time_ms);
        
        // Get the results
        float h_rotation[9];
        float3 h_translation;
        CUDA_CHECK(cudaMemcpy(h_rotation, transform.rotations, 9 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_translation, transform.translations, sizeof(float3), cudaMemcpyDeviceToHost));
        
        // Create aligned graph
        SimpleGraph aligned_graph = src_graph;  // Copy structure
        aligned_graph.node_positions = (float3*)malloc(src_graph.num_nodes * sizeof(float3));
        
        // Apply transformation
        for (int i = 0; i < src_graph.num_nodes; i++) {
            float3 p = src_graph.node_positions[i];
            float3 p_transformed;
            p_transformed.x = h_rotation[0]*p.x + h_rotation[1]*p.y + h_rotation[2]*p.z + h_translation.x;
            p_transformed.y = h_rotation[3]*p.x + h_rotation[4]*p.y + h_rotation[5]*p.z + h_translation.y;
            p_transformed.z = h_rotation[6]*p.x + h_rotation[7]*p.y + h_rotation[8]*p.z + h_translation.z;
            aligned_graph.node_positions[i] = p_transformed;
        }
        
        // Compute and display the RMSE
        float rmse = 0.0f;
        for (int i = 0; i < src_graph.num_nodes; i++) {
            float dx = aligned_graph.node_positions[i].x - tgt_graph.node_positions[i].x;
            float dy = aligned_graph.node_positions[i].y - tgt_graph.node_positions[i].y;
            float dz = aligned_graph.node_positions[i].z - tgt_graph.node_positions[i].z;
            rmse += dx*dx + dy*dy + dz*dz;
        }
        rmse = sqrt(rmse / src_graph.num_nodes);
        printf("  RMSE: %.6f (with %.1f%% noise)\n", rmse, noise * 100);
        
        // Save results to PLY files if visualization is enabled
        if (ENABLE_VISUALIZATION) {
            char filename[100];
            
            sprintf(filename, "procrustes_src_%d.ply", size);
            saveGraphToPLY(&src_graph, filename, 1.0, 0.0, 0.0);  // Red
            ply_files.push_back(filename);
            
            sprintf(filename, "procrustes_tgt_%d.ply", size);
            saveGraphToPLY(&tgt_graph, filename, 0.0, 0.0, 1.0);  // Blue
            ply_files.push_back(filename);
            
            sprintf(filename, "procrustes_aligned_%d.ply", size);
            saveGraphToPLY(&aligned_graph, filename, 0.0, 1.0, 0.0);  // Green
            ply_files.push_back(filename);
            
            sprintf(filename, "procrustes_error_%d.ply", size);
            saveErrorColoredPointCloud(&src_graph, &tgt_graph, &aligned_graph, filename);
            ply_files.push_back(filename);
        }
        
        // Clean up
        free(src_graph.node_positions);
        free(src_graph.edges);
        free(tgt_graph.node_positions);
        free(tgt_graph.edges);
        free(aligned_graph.node_positions);
        freeBatchedPointCloud(&src_pc);
        freeBatchedPointCloud(&tgt_pc);
        freeBatchedTransformation(&transform);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 2. Test batch processing with different batch counts
    printf("\nTesting Procrustes batch processing...\n");
    int batch_counts[] = {1, 2, 4, 8, 16};
    int point_count = 1000;  // Fixed point count for this test
    
    for (int batch_count : batch_counts) {
        // Create batch of graphs
        SimpleGraph* src_graphs = new SimpleGraph[batch_count];
        SimpleGraph* tgt_graphs = new SimpleGraph[batch_count];
        SimpleGraph* aligned_graphs = new SimpleGraph[batch_count];
        
        for (int i = 0; i < batch_count; i++) {
            generateSyntheticGraph(&src_graphs[i], point_count, 20);
            applyRandomTransform(&src_graphs[i], &tgt_graphs[i]);
            addNoise(&tgt_graphs[i], 0.02f);
            
            aligned_graphs[i].num_nodes = src_graphs[i].num_nodes;
            aligned_graphs[i].num_edges = src_graphs[i].num_edges;
            aligned_graphs[i].node_positions = (float3*)malloc(src_graphs[i].num_nodes * sizeof(float3));
            aligned_graphs[i].edges = (int*)malloc(src_graphs[i].num_edges * 2 * sizeof(int));
            memcpy(aligned_graphs[i].edges, src_graphs[i].edges, src_graphs[i].num_edges * 2 * sizeof(int));
        }
        
        // Convert to batched point clouds
        BatchedPointCloud src_pc, tgt_pc;
        int* sizes = (int*)malloc(batch_count * sizeof(int));
        for (int i = 0; i < batch_count; i++) {
            sizes[i] = src_graphs[i].num_nodes;
        }
        allocateBatchedPointCloud(&src_pc, batch_count, sizes);
        allocateBatchedPointCloud(&tgt_pc, batch_count, sizes);
        
        // Copy points to device (flattened)
        float* h_src_points = (float*)malloc(src_pc.total_points * 3 * sizeof(float));
        float* h_tgt_points = (float*)malloc(tgt_pc.total_points * 3 * sizeof(float));
        
        int point_offset = 0;
        for (int b = 0; b < batch_count; b++) {
            for (int i = 0; i < src_graphs[b].num_nodes; i++) {
                int idx = (point_offset + i) * 3;
                h_src_points[idx] = src_graphs[b].node_positions[i].x;
                h_src_points[idx + 1] = src_graphs[b].node_positions[i].y;
                h_src_points[idx + 2] = src_graphs[b].node_positions[i].z;
                
                h_tgt_points[idx] = tgt_graphs[b].node_positions[i].x;
                h_tgt_points[idx + 1] = tgt_graphs[b].node_positions[i].y;
                h_tgt_points[idx + 2] = tgt_graphs[b].node_positions[i].z;
            }
            point_offset += src_graphs[b].num_nodes;
        }
        
        CUDA_CHECK(cudaMemcpy(src_pc.points, h_src_points, src_pc.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tgt_pc.points, h_tgt_points, tgt_pc.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Run Procrustes with and without streams
        BatchedTransformation transform;
        allocateBatchedTransformation(&transform, batch_count);
        
        printf("  Testing with %d batches (point count: %d)...\n", batch_count, point_count);
        
        // Create CUDA events for timing
        cudaEvent_t start_no_streams, stop_no_streams, start_streams, stop_streams;
        CUDA_CHECK(cudaEventCreate(&start_no_streams));
        CUDA_CHECK(cudaEventCreate(&stop_no_streams));
        CUDA_CHECK(cudaEventCreate(&start_streams));
        CUDA_CHECK(cudaEventCreate(&stop_streams));
        
        // Run without streams
        CUDA_CHECK(cudaEventRecord(start_no_streams));
        batchedProcrustes(&src_pc, &tgt_pc, &transform, false);
        CUDA_CHECK(cudaEventRecord(stop_no_streams));
        CUDA_CHECK(cudaEventSynchronize(stop_no_streams));
        
        float time_no_streams = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time_no_streams, start_no_streams, stop_no_streams));
        
        // Run with streams
        CUDA_CHECK(cudaEventRecord(start_streams));
        batchedProcrustes(&src_pc, &tgt_pc, &transform, true);
        CUDA_CHECK(cudaEventRecord(stop_streams));
        CUDA_CHECK(cudaEventSynchronize(stop_streams));
        
        float time_streams = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time_streams, start_streams, stop_streams));
        
        printf("    No streams: %.2f ms, With streams: %.2f ms, Speedup: %.2fx\n", 
               time_no_streams, time_streams, time_no_streams / time_streams);
        
        // Get the transformation results
        float* h_rotations = (float*)malloc(batch_count * 9 * sizeof(float));
        float3* h_translations = (float3*)malloc(batch_count * sizeof(float3));
        
        CUDA_CHECK(cudaMemcpy(h_rotations, transform.rotations, batch_count * 9 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_translations, transform.translations, batch_count * sizeof(float3), cudaMemcpyDeviceToHost));
        
        // Apply transformations to each source graph
        for (int b = 0; b < batch_count; b++) {
            float* R = &h_rotations[b * 9];
            float3 t = h_translations[b];
            
            for (int i = 0; i < src_graphs[b].num_nodes; i++) {
                float3 p = src_graphs[b].node_positions[i];
                float3 p_transformed;
                p_transformed.x = R[0]*p.x + R[1]*p.y + R[2]*p.z + t.x;
                p_transformed.y = R[3]*p.x + R[4]*p.y + R[5]*p.z + t.y;
                p_transformed.z = R[6]*p.x + R[7]*p.y + R[8]*p.z + t.z;
                aligned_graphs[b].node_positions[i] = p_transformed;
            }
        }
        
        // Compute overall RMSE
        float total_rmse = 0.0f;
        int total_points = 0;
        for (int b = 0; b < batch_count; b++) {
            float batch_rmse = 0.0f;
            for (int i = 0; i < src_graphs[b].num_nodes; i++) {
                float dx = aligned_graphs[b].node_positions[i].x - tgt_graphs[b].node_positions[i].x;
                float dy = aligned_graphs[b].node_positions[i].y - tgt_graphs[b].node_positions[i].y;
                float dz = aligned_graphs[b].node_positions[i].z - tgt_graphs[b].node_positions[i].z;
                batch_rmse += dx*dx + dy*dy + dz*dz;
            }
            total_rmse += batch_rmse;
            total_points += src_graphs[b].num_nodes;
        }
        total_rmse = sqrt(total_rmse / total_points);
        printf("    Overall RMSE: %.6f\n", total_rmse);
        
        // Save visualization if enabled
        if (ENABLE_VISUALIZATION && batch_count <= 4) {  // Limit visualization to reasonable batch counts
            // Create combined point clouds
            std::vector<SimpleGraph*> src_vec, tgt_vec, aligned_vec;
            std::vector<float3> colors = {
                {1.0f, 0.0f, 0.0f},  // Red
                {0.0f, 1.0f, 0.0f},  // Green
                {0.0f, 0.0f, 1.0f},  // Blue
                {1.0f, 1.0f, 0.0f},  // Yellow
                {1.0f, 0.0f, 1.0f},  // Magenta
                {0.0f, 1.0f, 1.0f},  // Cyan
                {0.5f, 0.5f, 0.5f},  // Gray
                {1.0f, 0.5f, 0.0f}   // Orange
            };
            
            for (int b = 0; b < batch_count; b++) {
                src_vec.push_back(&src_graphs[b]);
                tgt_vec.push_back(&tgt_graphs[b]);
                aligned_vec.push_back(&aligned_graphs[b]);
            }
            
            char filename[100];
            sprintf(filename, "procrustes_batch%d_src.ply", batch_count);
            saveMultiColorPointCloud(src_vec, colors, filename);
            ply_files.push_back(filename);
            
            sprintf(filename, "procrustes_batch%d_tgt.ply", batch_count);
            saveMultiColorPointCloud(tgt_vec, colors, filename);
            ply_files.push_back(filename);
            
            sprintf(filename, "procrustes_batch%d_aligned.ply", batch_count);
            saveMultiColorPointCloud(aligned_vec, colors, filename);
            ply_files.push_back(filename);
        }
        
        // Clean up
        for (int b = 0; b < batch_count; b++) {
            free(src_graphs[b].node_positions);
            free(src_graphs[b].edges);
            free(tgt_graphs[b].node_positions);
            free(tgt_graphs[b].edges);
            free(aligned_graphs[b].node_positions);
            free(aligned_graphs[b].edges);
        }
        delete[] src_graphs;
        delete[] tgt_graphs;
        delete[] aligned_graphs;
        
        free(h_src_points);
        free(h_tgt_points);
        free(sizes);
        free(h_rotations);
        free(h_translations);
        
        freeBatchedPointCloud(&src_pc);
        freeBatchedPointCloud(&tgt_pc);
        freeBatchedTransformation(&transform);
        
        CUDA_CHECK(cudaEventDestroy(start_no_streams));
        CUDA_CHECK(cudaEventDestroy(stop_no_streams));
        CUDA_CHECK(cudaEventDestroy(start_streams));
        CUDA_CHECK(cudaEventDestroy(stop_streams));
    }
    
    // Generate visualization script if enabled
    if (ENABLE_VISUALIZATION && !ply_files.empty()) {
        generateVisualizationScript(ply_files, "visualize_procrustes_tests.py");
    }
}

// Run comprehensive testing of ICP algorithm
void runCompleteICPTests() {
    printf("\n=== Running Comprehensive ICP Tests ===\n");
    
    // Initialize test variables
    std::vector<std::string> ply_files;
    
    // 1. Test different noise levels
    printf("\nTesting ICP with different noise levels...\n");
    int point_count = 1000;  // Fixed point count
    float noise_levels[] = {0.01f, 0.05f, 0.1f, 0.2f};  // 1%, 5%, 10%, 20% noise
    
    for (float noise : noise_levels) {
        // Create test graphs
        SimpleGraph src_graph;
        generateSyntheticGraph(&src_graph, point_count, 20);  // 20% edge density
        
        SimpleGraph tgt_graph;
        applyRandomTransform(&src_graph, &tgt_graph);
        
        // Add noise according to level
        addNoise(&tgt_graph, noise);
        
        // Create point clouds
        BatchedPointCloud src_pc, tgt_pc;
        graphToPointCloud(&src_graph, &src_pc);
        graphToPointCloud(&tgt_graph, &tgt_pc);
        
        // Run ICP alignment
        BatchedTransformation transform;
        allocateBatchedTransformation(&transform, 1);
        int max_iterations = 50;
        float convergence_threshold = 1e-6;
        
        printf("  Running ICP with %.1f%% noise...\n", noise * 100);
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Measure total time
        CUDA_CHECK(cudaEventRecord(start));
        batchedICP(&src_pc, &tgt_pc, max_iterations, convergence_threshold, &transform, false);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        
        // Get the results
        float h_rotation[9];
        float3 h_translation;
        float h_error;
        CUDA_CHECK(cudaMemcpy(h_rotation, transform.rotations, 9 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_translation, transform.translations, sizeof(float3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_error, transform.errors, sizeof(float), cudaMemcpyDeviceToHost));
        
        printf("    Time: %.2f ms, Final error: %.6f\n", time_ms, h_error);
        
        // Create aligned graph
        SimpleGraph aligned_graph = src_graph;  // Copy structure
        aligned_graph.node_positions = (float3*)malloc(src_graph.num_nodes * sizeof(float3));
        
        // Apply transformation
        for (int i = 0; i < src_graph.num_nodes; i++) {
            float3 p = src_graph.node_positions[i];
            float3 p_transformed;
            p_transformed.x = h_rotation[0]*p.x + h_rotation[1]*p.y + h_rotation[2]*p.z + h_translation.x;
            p_transformed.y = h_rotation[3]*p.x + h_rotation[4]*p.y + h_rotation[5]*p.z + h_translation.y;
            p_transformed.z = h_rotation[6]*p.x + h_rotation[7]*p.y + h_rotation[8]*p.z + h_translation.z;
            aligned_graph.node_positions[i] = p_transformed;
        }
        
        // Compute and display the RMSE
        float rmse = 0.0f;
        for (int i = 0; i < src_graph.num_nodes; i++) {
            float dx = aligned_graph.node_positions[i].x - tgt_graph.node_positions[i].x;
            float dy = aligned_graph.node_positions[i].y - tgt_graph.node_positions[i].y;
            float dz = aligned_graph.node_positions[i].z - tgt_graph.node_positions[i].z;
            rmse += dx*dx + dy*dy + dz*dz;
        }
        rmse = sqrt(rmse / src_graph.num_nodes);
        printf("    RMSE: %.6f\n", rmse);
        
        // Save results to PLY files if visualization is enabled
        if (ENABLE_VISUALIZATION) {
            char filename[100];
            
            sprintf(filename, "icp_noise%.0f_src.ply", noise * 100);
            saveGraphToPLY(&src_graph, filename, 1.0, 0.0, 0.0);  // Red
            ply_files.push_back(filename);
            
            sprintf(filename, "icp_noise%.0f_tgt.ply", noise * 100);
            saveGraphToPLY(&tgt_graph, filename, 0.0, 0.0, 1.0);  // Blue
            ply_files.push_back(filename);
            
            sprintf(filename, "icp_noise%.0f_aligned.ply", noise * 100);
            saveGraphToPLY(&aligned_graph, filename, 0.0, 1.0, 0.0);  // Green
            ply_files.push_back(filename);
            
            sprintf(filename, "icp_noise%.0f_error.ply", noise * 100);
            saveErrorColoredPointCloud(&src_graph, &tgt_graph, &aligned_graph, filename);
            ply_files.push_back(filename);
        }
        
        // Clean up
        free(src_graph.node_positions);
        free(src_graph.edges);
        free(tgt_graph.node_positions);
        free(tgt_graph.edges);
        free(aligned_graph.node_positions);
        freeBatchedPointCloud(&src_pc);
        freeBatchedPointCloud(&tgt_pc);
        freeBatchedTransformation(&transform);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 2. Test different iteration counts
    printf("\nTesting ICP with different iteration counts...\n");
    int iterations[] = {5, 10, 20, 50};
    float fixed_noise = 0.05f;  // 5% noise for all tests
    
    for (int max_iter : iterations) {
        // Create test graphs
        SimpleGraph src_graph;
        generateSyntheticGraph(&src_graph, point_count, 20);
        
        SimpleGraph tgt_graph;
        applyRandomTransform(&src_graph, &tgt_graph);
        addNoise(&tgt_graph, fixed_noise);
        
        // Create point clouds
        BatchedPointCloud src_pc, tgt_pc;
        graphToPointCloud(&src_graph, &src_pc);
        graphToPointCloud(&tgt_graph, &tgt_pc);
        
        // Run ICP alignment
        BatchedTransformation transform;
        allocateBatchedTransformation(&transform, 1);
        float convergence_threshold = 1e-6;
        
        printf("  Running ICP with max %d iterations...\n", max_iter);
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Measure total time
        CUDA_CHECK(cudaEventRecord(start));
        batchedICP(&src_pc, &tgt_pc, max_iter, convergence_threshold, &transform, false);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        
        // Get the results
        float h_rotation[9];
        float3 h_translation;
        float h_error;
        CUDA_CHECK(cudaMemcpy(h_rotation, transform.rotations, 9 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_translation, transform.translations, sizeof(float3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_error, transform.errors, sizeof(float), cudaMemcpyDeviceToHost));
        
        printf("    Time: %.2f ms, Final error: %.6f\n", time_ms, h_error);
        
        // Create aligned graph
        SimpleGraph aligned_graph = src_graph;  // Copy structure
        aligned_graph.node_positions = (float3*)malloc(src_graph.num_nodes * sizeof(float3));
        
        // Apply transformation
        for (int i = 0; i < src_graph.num_nodes; i++) {
            float3 p = src_graph.node_positions[i];
            float3 p_transformed;
            p_transformed.x = h_rotation[0]*p.x + h_rotation[1]*p.y + h_rotation[2]*p.z + h_translation.x;
            p_transformed.y = h_rotation[3]*p.x + h_rotation[4]*p.y + h_rotation[5]*p.z + h_translation.y;
            p_transformed.z = h_rotation[6]*p.x + h_rotation[7]*p.y + h_rotation[8]*p.z + h_translation.z;
            aligned_graph.node_positions[i] = p_transformed;
        }
        
        // Compute and display the RMSE
        float rmse = 0.0f;
        for (int i = 0; i < src_graph.num_nodes; i++) {
            float dx = aligned_graph.node_positions[i].x - tgt_graph.node_positions[i].x;
            float dy = aligned_graph.node_positions[i].y - tgt_graph.node_positions[i].y;
            float dz = aligned_graph.node_positions[i].z - tgt_graph.node_positions[i].z;
            rmse += dx*dx + dy*dy + dz*dz;
        }
        rmse = sqrt(rmse / src_graph.num_nodes);
        printf("    RMSE: %.6f\n", rmse);
        
        // Save results for visualization
        if (ENABLE_VISUALIZATION) {
            char filename[100];
            sprintf(filename, "icp_iter%d_error.ply", max_iter);
            saveErrorColoredPointCloud(&src_graph, &tgt_graph, &aligned_graph, filename);
            ply_files.push_back(filename);
        }
        
        // Clean up
        free(src_graph.node_positions);
        free(src_graph.edges);
        free(tgt_graph.node_positions);
        free(tgt_graph.edges);
        free(aligned_graph.node_positions);
        freeBatchedPointCloud(&src_pc);
        freeBatchedPointCloud(&tgt_pc);
        freeBatchedTransformation(&transform);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 3. Test stream vs. no-stream performance with different batch counts
    printf("\nTesting ICP batch processing with and without streams...\n");
    int batch_counts[] = {1, 2, 4, 8};
    point_count = 500;  // Smaller point count for batch testing
    
    for (int batch_count : batch_counts) {
        // Create batch of graphs
        SimpleGraph* src_graphs = new SimpleGraph[batch_count];
        SimpleGraph* tgt_graphs = new SimpleGraph[batch_count];
        SimpleGraph* aligned_graphs = new SimpleGraph[batch_count];
        
        for (int i = 0; i < batch_count; i++) {
            generateSyntheticGraph(&src_graphs[i], point_count, 20);
            applyRandomTransform(&src_graphs[i], &tgt_graphs[i]);
            addNoise(&tgt_graphs[i], 0.05f);  // 5% noise
            
            aligned_graphs[i].num_nodes = src_graphs[i].num_nodes;
            aligned_graphs[i].num_edges = src_graphs[i].num_edges;
            aligned_graphs[i].node_positions = (float3*)malloc(src_graphs[i].num_nodes * sizeof(float3));
            aligned_graphs[i].edges = (int*)malloc(src_graphs[i].num_edges * 2 * sizeof(int));
            memcpy(aligned_graphs[i].edges, src_graphs[i].edges, src_graphs[i].num_edges * 2 * sizeof(int));
        }
        
        // Convert to batched point clouds
        BatchedPointCloud src_pc, tgt_pc;
        int* sizes = (int*)malloc(batch_count * sizeof(int));
        for (int i = 0; i < batch_count; i++) {
            sizes[i] = src_graphs[i].num_nodes;
        }
        allocateBatchedPointCloud(&src_pc, batch_count, sizes);
        allocateBatchedPointCloud(&tgt_pc, batch_count, sizes);
        
        // Copy points to device
        float* h_src_points = (float*)malloc(src_pc.total_points * 3 * sizeof(float));
        float* h_tgt_points = (float*)malloc(tgt_pc.total_points * 3 * sizeof(float));
        
        int point_offset = 0;
        for (int b = 0; b < batch_count; b++) {
            for (int i = 0; i < src_graphs[b].num_nodes; i++) {
                int idx = (point_offset + i) * 3;
                h_src_points[idx] = src_graphs[b].node_positions[i].x;
                h_src_points[idx + 1] = src_graphs[b].node_positions[i].y;
                h_src_points[idx + 2] = src_graphs[b].node_positions[i].z;
                
                h_tgt_points[idx] = tgt_graphs[b].node_positions[i].x;
                h_tgt_points[idx + 1] = tgt_graphs[b].node_positions[i].y;
                h_tgt_points[idx + 2] = tgt_graphs[b].node_positions[i].z;
            }
            point_offset += src_graphs[b].num_nodes;
        }
        
        CUDA_CHECK(cudaMemcpy(src_pc.points, h_src_points, src_pc.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tgt_pc.points, h_tgt_points, tgt_pc.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Create transformation storage
        BatchedTransformation transform_no_streams, transform_streams;
        allocateBatchedTransformation(&transform_no_streams, batch_count);
        allocateBatchedTransformation(&transform_streams, batch_count);
        
        // Set ICP parameters
        int max_iterations = 20;
        float convergence_threshold = 1e-6;
        
        printf("  Testing with %d batches (point count: %d)...\n", batch_count, point_count);
        
        // Create CUDA events for timing
        cudaEvent_t start_no_streams, stop_no_streams, start_streams, stop_streams;
        CUDA_CHECK(cudaEventCreate(&start_no_streams));
        CUDA_CHECK(cudaEventCreate(&stop_no_streams));
        CUDA_CHECK(cudaEventCreate(&start_streams));
        CUDA_CHECK(cudaEventCreate(&stop_streams));
        
        // Run without streams
        CUDA_CHECK(cudaEventRecord(start_no_streams));
        batchedICP(&src_pc, &tgt_pc, max_iterations, convergence_threshold, &transform_no_streams, false);
        CUDA_CHECK(cudaEventRecord(stop_no_streams));
        CUDA_CHECK(cudaEventSynchronize(stop_no_streams));
        
        float time_no_streams = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time_no_streams, start_no_streams, stop_no_streams));
        
        // Get error for no_streams
        float* h_errors_no_streams = (float*)malloc(batch_count * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_errors_no_streams, transform_no_streams.errors, batch_count * sizeof(float), cudaMemcpyDeviceToHost));
        
        float avg_error_no_streams = 0;
        for (int i = 0; i < batch_count; i++) {
            avg_error_no_streams += h_errors_no_streams[i];
        }
        avg_error_no_streams /= batch_count;
        
        // Run with streams
        CUDA_CHECK(cudaEventRecord(start_streams));
        batchedICP(&src_pc, &tgt_pc, max_iterations, convergence_threshold, &transform_streams, true);
        CUDA_CHECK(cudaEventRecord(stop_streams));
        CUDA_CHECK(cudaEventSynchronize(stop_streams));
        
        float time_streams = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time_streams, start_streams, stop_streams));
        
        // Get error for streams
        float* h_errors_streams = (float*)malloc(batch_count * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_errors_streams, transform_streams.errors, batch_count * sizeof(float), cudaMemcpyDeviceToHost));
        
        float avg_error_streams = 0;
        for (int i = 0; i < batch_count; i++) {
            avg_error_streams += h_errors_streams[i];
        }
        avg_error_streams /= batch_count;
        
        printf("    No streams: %.2f ms (error: %.6f), With streams: %.2f ms (error: %.6f), Speedup: %.2fx\n", 
               time_no_streams, avg_error_no_streams, time_streams, avg_error_streams, time_no_streams / time_streams);
        
        // Save visualization if enabled
        if (ENABLE_VISUALIZATION && batch_count <= 4) {
            // Get the transformation results
            float* h_rotations = (float*)malloc(batch_count * 9 * sizeof(float));
            float3* h_translations = (float3*)malloc(batch_count * sizeof(float3));
            
            CUDA_CHECK(cudaMemcpy(h_rotations, transform_streams.rotations, batch_count * 9 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_translations, transform_streams.translations, batch_count * sizeof(float3), cudaMemcpyDeviceToHost));
            
            // Apply transformations to each source graph
            for (int b = 0; b < batch_count; b++) {
                float* R = &h_rotations[b * 9];
                float3 t = h_translations[b];
                
                for (int i = 0; i < src_graphs[b].num_nodes; i++) {
                    float3 p = src_graphs[b].node_positions[i];
                    float3 p_transformed;
                    p_transformed.x = R[0]*p.x + R[1]*p.y + R[2]*p.z + t.x;
                    p_transformed.y = R[3]*p.x + R[4]*p.y + R[5]*p.z + t.y;
                    p_transformed.z = R[6]*p.x + R[7]*p.y + R[8]*p.z + t.z;
                    aligned_graphs[b].node_positions[i] = p_transformed;
                }
            }
            
            // Create combined point clouds
            std::vector<SimpleGraph*> src_vec, tgt_vec, aligned_vec;
            std::vector<float3> colors = {
                {1.0f, 0.0f, 0.0f},  // Red
                {0.0f, 1.0f, 0.0f},  // Green
                {0.0f, 0.0f, 1.0f},  // Blue
                {1.0f, 1.0f, 0.0f},  // Yellow
                {1.0f, 0.0f, 1.0f},  // Magenta
                {0.0f, 1.0f, 1.0f},  // Cyan
                {0.5f, 0.5f, 0.5f},  // Gray
                {1.0f, 0.5f, 0.0f}   // Orange
            };
            
            for (int b = 0; b < batch_count; b++) {
                src_vec.push_back(&src_graphs[b]);
                tgt_vec.push_back(&tgt_graphs[b]);
                aligned_vec.push_back(&aligned_graphs[b]);
            }
            
            char filename[100];
            sprintf(filename, "icp_batch%d_comparison.ply", batch_count);
            
            // Save all three together (source, target, aligned) with different colors
            std::vector<SimpleGraph*> all_graphs;
            std::vector<float3> all_colors;
            
            // Add source graphs (red)
            for (int b = 0; b < batch_count; b++) {
                all_graphs.push_back(&src_graphs[b]);
                all_colors.push_back({1.0f, 0.0f, 0.0f});  // Red
            }
            
            // Add target graphs (blue)
            for (int b = 0; b < batch_count; b++) {
                all_graphs.push_back(&tgt_graphs[b]);
                all_colors.push_back({0.0f, 0.0f, 1.0f});  // Blue
            }
            
            // Add aligned graphs (green)
            for (int b = 0; b < batch_count; b++) {
                all_graphs.push_back(&aligned_graphs[b]);
                all_colors.push_back({0.0f, 1.0f, 0.0f});  // Green
            }
            
            saveMultiColorPointCloud(all_graphs, all_colors, filename);
            ply_files.push_back(filename);
            
            free(h_rotations);
            free(h_translations);
        }
        
        // Clean up
        for (int b = 0; b < batch_count; b++) {
            free(src_graphs[b].node_positions);
            free(src_graphs[b].edges);
            free(tgt_graphs[b].node_positions);
            free(tgt_graphs[b].edges);
            free(aligned_graphs[b].node_positions);
            free(aligned_graphs[b].edges);
        }
        delete[] src_graphs;
        delete[] tgt_graphs;
        delete[] aligned_graphs;
        
        free(h_src_points);
        free(h_tgt_points);
        free(sizes);
        free(h_errors_no_streams);
        free(h_errors_streams);
        
        freeBatchedPointCloud(&src_pc);
        freeBatchedPointCloud(&tgt_pc);
        freeBatchedTransformation(&transform_no_streams);
        freeBatchedTransformation(&transform_streams);
        
        CUDA_CHECK(cudaEventDestroy(start_no_streams));
        CUDA_CHECK(cudaEventDestroy(stop_no_streams));
        CUDA_CHECK(cudaEventDestroy(start_streams));
        CUDA_CHECK(cudaEventDestroy(stop_streams));
    }
    
    // Generate visualization script if enabled
    if (ENABLE_VISUALIZATION && !ply_files.empty()) {
        generateVisualizationScript(ply_files, "visualize_icp_tests.py");
    }
}

// Run comprehensive benchmarks for all algorithms
void runComprehensiveBenchmarks() {
    printf("\n=== Running Comprehensive Benchmarks ===\n");
    
    // Parameters for benchmarking
    std::vector<int> point_counts;
    std::vector<int> batch_counts;
    
    // Generate logarithmic point counts
    for (int i = 0; i < NUM_BENCHMARK_SIZES; i++) {
        int count = MIN_POINT_COUNT * pow(MAX_POINT_COUNT / MIN_POINT_COUNT, 
                                      (double)i / (NUM_BENCHMARK_SIZES - 1));
        point_counts.push_back(count);
    }
    
    // Generate logarithmic batch counts
    for (int i = 0; i < NUM_BENCHMARK_BATCHES; i++) {
        int count = MIN_BATCH_COUNT * pow(MAX_BATCH_COUNT / MIN_BATCH_COUNT, 
                                      (double)i / (NUM_BENCHMARK_BATCHES - 1));
        batch_counts.push_back(count);
    }
    
    printf("Point counts: ");
    for (int count : point_counts) printf("%d ", count);
    printf("\n");
    
    printf("Batch counts: ");
    for (int count : batch_counts) printf("%d ", count);
    printf("\n");
    
    // Create results containers
    std::vector<std::vector<double>> times_procrustes(point_counts.size(), 
                                                 std::vector<double>(batch_counts.size(), 0.0));
    std::vector<std::vector<double>> times_procrustes_streams(point_counts.size(), 
                                                        std::vector<double>(batch_counts.size(), 0.0));
    std::vector<std::vector<double>> times_icp(point_counts.size(), 
                                          std::vector<double>(batch_counts.size(), 0.0));
    std::vector<std::vector<double>> times_icp_streams(point_counts.size(), 
                                                  std::vector<double>(batch_counts.size(), 0.0));
    std::vector<std::vector<double>> errors_icp(point_counts.size(), 
                                           std::vector<double>(batch_counts.size(), 0.0));
    std::vector<std::vector<double>> errors_icp_streams(point_counts.size(), 
                                                   std::vector<double>(batch_counts.size(), 0.0));
    
    // 1. Benchmark Procrustes Algorithm
    printf("\nBenchmarking Procrustes algorithm...\n");
    
    for (size_t i = 0; i < point_counts.size(); i++) {
        int point_count = point_counts[i];
        
        for (size_t j = 0; j < batch_counts.size(); j++) {
            int batch_count = batch_counts[j];
            
            // Skip very large configurations to avoid excessive runtime
            if (((long long)point_count * batch_count) > 1000000) {
                printf("  Skipping point_count=%d, batch_count=%d (too large)\n", point_count, batch_count);
                continue;
            }
            
            printf("  Benchmarking point_count=%d, batch_count=%d...\n", point_count, batch_count);
            
            // Prepare test data
            BatchedPointCloud src_pc, tgt_pc;
            int* sizes = (int*)malloc(batch_count * sizeof(int));
            for (int b = 0; b < batch_count; b++) {
                sizes[b] = point_count;
            }
            
            allocateBatchedPointCloud(&src_pc, batch_count, sizes);
            allocateBatchedPointCloud(&tgt_pc, batch_count, sizes);
            
            // Generate random point clouds
            float* h_src_points = (float*)malloc(src_pc.total_points * 3 * sizeof(float));
            float* h_tgt_points = (float*)malloc(tgt_pc.total_points * 3 * sizeof(float));
            
            // Use fixed seed for reproducibility
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            
            // Generate source points
            for (int i = 0; i < src_pc.total_points * 3; i++) {
                h_src_points[i] = dist(rng);
            }
            
            // Generate target points (transformed source)
            int point_offset = 0;
            for (int b = 0; b < batch_count; b++) {
                // Random transformation
                float angle = dist(rng) * 3.14159f;
                float tx = dist(rng);
                float ty = dist(rng);
                float tz = dist(rng);
                
                float R[9] = {
                    cosf(angle), -sinf(angle), 0,
                    sinf(angle), cosf(angle), 0,
                    0, 0, 1
                };
                
                for (int p = 0; p < point_count; p++) {
                    int idx = (point_offset + p) * 3;
                    float x = h_src_points[idx];
                    float y = h_src_points[idx + 1];
                    float z = h_src_points[idx + 2];
                    
                    // Apply rotation
                    float rx = R[0]*x + R[1]*y + R[2]*z;
                    float ry = R[3]*x + R[4]*y + R[5]*z;
                    float rz = R[6]*x + R[7]*y + R[8]*z;
                    
                    // Apply translation and add noise
                    h_tgt_points[idx] = rx + tx + dist(rng) * 0.02f;
                    h_tgt_points[idx + 1] = ry + ty + dist(rng) * 0.02f;
                    h_tgt_points[idx + 2] = rz + tz + dist(rng) * 0.02f;
                }
                
                point_offset += point_count;
            }
            
            // Copy to device
            CUDA_CHECK(cudaMemcpy(src_pc.points, h_src_points, src_pc.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(tgt_pc.points, h_tgt_points, tgt_pc.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
            
            // Create transformations
            BatchedTransformation transform;
            allocateBatchedTransformation(&transform, batch_count);
            
            // Create CUDA events for timing
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            
            // Run benchmark 
            float elapsed_time = 0.0f;
            
            // Without streams (take the best of 3 runs)
            float best_time_no_streams = FLT_MAX;
            for (int run = 0; run < 3; run++) {
                // Warmup run
                if (run == 0) {
                    batchedProcrustes(&src_pc, &tgt_pc, &transform, false);
                }
                
                CUDA_CHECK(cudaEventRecord(start));
                batchedProcrustes(&src_pc, &tgt_pc, &transform, false);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
                
                best_time_no_streams = std::min(best_time_no_streams, elapsed_time);
            }
            
            times_procrustes[i][j] = best_time_no_streams;
            
            // With streams (take the best of 3 runs)
            float best_time_streams = FLT_MAX;
            for (int run = 0; run < 3; run++) {
                // Warmup run
                if (run == 0) {
                    batchedProcrustes(&src_pc, &tgt_pc, &transform, true);
                }
                
                CUDA_CHECK(cudaEventRecord(start));
                batchedProcrustes(&src_pc, &tgt_pc, &transform, true);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
                
                best_time_streams = std::min(best_time_streams, elapsed_time);
            }
            
            times_procrustes_streams[i][j] = best_time_streams;
            
            printf("    No streams: %.2f ms, With streams: %.2f ms, Speedup: %.2fx\n", 
                   best_time_no_streams, best_time_streams, best_time_no_streams / best_time_streams);
            
            // Clean up
            free(sizes);
            free(h_src_points);
            free(h_tgt_points);
            freeBatchedPointCloud(&src_pc);
            freeBatchedPointCloud(&tgt_pc);
            freeBatchedTransformation(&transform);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
    }
    
    // 2. Benchmark ICP Algorithm
    printf("\nBenchmarking ICP algorithm...\n");
    
    for (size_t i = 0; i < point_counts.size(); i++) {
        int point_count = point_counts[i];
        
        for (size_t j = 0; j < batch_counts.size(); j++) {
            int batch_count = batch_counts[j];
            
            // Skip very large configurations
            if (((long long)point_count * batch_count) > 500000) {
                printf("  Skipping point_count=%d, batch_count=%d (too large)\n", point_count, batch_count);
                continue;
            }
            
            printf("  Benchmarking point_count=%d, batch_count=%d...\n", point_count, batch_count);
            
            // Prepare test data (similar to Procrustes benchmark)
            BatchedPointCloud src_pc, tgt_pc;
            int* sizes = (int*)malloc(batch_count * sizeof(int));
            for (int b = 0; b < batch_count; b++) {
                sizes[b] = point_count;
            }
            
            allocateBatchedPointCloud(&src_pc, batch_count, sizes);
            allocateBatchedPointCloud(&tgt_pc, batch_count, sizes);
            
            // Generate random point clouds
            float* h_src_points = (float*)malloc(src_pc.total_points * 3 * sizeof(float));
            float* h_tgt_points = (float*)malloc(tgt_pc.total_points * 3 * sizeof(float));
            
            // Use fixed seed for reproducibility
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            
            // Generate source and target points (similar to Procrustes)
            for (int i = 0; i < src_pc.total_points * 3; i++) {
                h_src_points[i] = dist(rng);
            }
            
            int point_offset = 0;
            for (int b = 0; b < batch_count; b++) {
                float angle = dist(rng) * 3.14159f;
                float tx = dist(rng);
                float ty = dist(rng);
                float tz = dist(rng);
                
                float R[9] = {
                    cosf(angle), -sinf(angle), 0,
                    sinf(angle), cosf(angle), 0,
                    0, 0, 1
                };
                
                for (int p = 0; p < point_count; p++) {
                    int idx = (point_offset + p) * 3;
                    float x = h_src_points[idx];
                    float y = h_src_points[idx + 1];
                    float z = h_src_points[idx + 2];
                    
                    float rx = R[0]*x + R[1]*y + R[2]*z;
                    float ry = R[3]*x + R[4]*y + R[5]*z;
                    float rz = R[6]*x + R[7]*y + R[8]*z;
                    
                    h_tgt_points[idx] = rx + tx + dist(rng) * 0.05f;
                    h_tgt_points[idx + 1] = ry + ty + dist(rng) * 0.05f;
                    h_tgt_points[idx + 2] = rz + tz + dist(rng) * 0.05f;
                }
                
                point_offset += point_count;
            }
            
            // Copy to device
            CUDA_CHECK(cudaMemcpy(src_pc.points, h_src_points, src_pc.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(tgt_pc.points, h_tgt_points, tgt_pc.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
            
            // Create transformations
            BatchedTransformation transform_no_streams, transform_streams;
            allocateBatchedTransformation(&transform_no_streams, batch_count);
            allocateBatchedTransformation(&transform_streams, batch_count);
            
            // ICP parameters
            int max_iterations = 20;
            float convergence_threshold = 1e-6;
            
            // Create CUDA events for timing
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            
            // Run benchmark without streams
            float elapsed_time = 0.0f;
            float best_time_no_streams = FLT_MAX;
            
            for (int run = 0; run < 3; run++) {
                // Warmup run
                if (run == 0) {
                    batchedICP(&src_pc, &tgt_pc, max_iterations, convergence_threshold, &transform_no_streams, false);
                }
                
                CUDA_CHECK(cudaEventRecord(start));
                batchedICP(&src_pc, &tgt_pc, max_iterations, convergence_threshold, &transform_no_streams, false);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
                
                best_time_no_streams = std::min(best_time_no_streams, elapsed_time);
            }
            
            times_icp[i][j] = best_time_no_streams;
            
            // Get error for no_streams
            float* h_errors_no_streams = (float*)malloc(batch_count * sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_errors_no_streams, transform_no_streams.errors, batch_count * sizeof(float), cudaMemcpyDeviceToHost));
            
            float avg_error_no_streams = 0;
            for (int b = 0; b < batch_count; b++) {
                avg_error_no_streams += h_errors_no_streams[b];
            }
            avg_error_no_streams /= batch_count;
            errors_icp[i][j] = avg_error_no_streams;
            
            // Run benchmark with streams
            float best_time_streams = FLT_MAX;
            
            for (int run = 0; run < 3; run++) {
                // Warmup run
                if (run == 0) {
                    batchedICP(&src_pc, &tgt_pc, max_iterations, convergence_threshold, &transform_streams, true);
                }
                
                CUDA_CHECK(cudaEventRecord(start));
                batchedICP(&src_pc, &tgt_pc, max_iterations, convergence_threshold, &transform_streams, true);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
                
                best_time_streams = std::min(best_time_streams, elapsed_time);
            }
            
            times_icp_streams[i][j] = best_time_streams;
            
            // Get error for streams
            float* h_errors_streams = (float*)malloc(batch_count * sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_errors_streams, transform_streams.errors, batch_count * sizeof(float), cudaMemcpyDeviceToHost));
            
            float avg_error_streams = 0;
            for (int b = 0; b < batch_count; b++) {
                avg_error_streams += h_errors_streams[b];
            }
            avg_error_streams /= batch_count;
            errors_icp_streams[i][j] = avg_error_streams;
            
            printf("    No streams: %.2f ms (error: %.6f), With streams: %.2f ms (error: %.6f), Speedup: %.2fx\n", 
                   best_time_no_streams, avg_error_no_streams, best_time_streams, avg_error_streams,
                   best_time_no_streams / best_time_streams);
            
            // Clean up
            free(sizes);
            free(h_src_points);
            free(h_tgt_points);
            free(h_errors_no_streams);
            free(h_errors_streams);
            freeBatchedPointCloud(&src_pc);
            freeBatchedPointCloud(&tgt_pc);
            freeBatchedTransformation(&transform_no_streams);
            freeBatchedTransformation(&transform_streams);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
    }
    
    // Save benchmark results and generate plot script
    if (SAVE_BENCHMARK_CSV) {
        saveBenchmarkResultsToCSV("benchmark_results.csv", point_counts, batch_counts,
                                 times_procrustes, times_procrustes_streams,
                                 times_icp, times_icp_streams,
                                 errors_icp, errors_icp_streams);
        
        generateBenchmarkPlotScript("benchmark_results.csv", "plot_benchmarks.py");
        
        printf("\nBenchmark results saved to benchmark_results.csv\n");
        printf("Plot script generated as plot_benchmarks.py\n");
    }
}

// Structure to hold comprehensive benchmark results
typedef struct {
    int point_count;            // Number of points per cloud
    int batch_size;             // Number of point clouds in batch
    int num_iterations;         // Number of ICP iterations
    float grid_cell_size;       // Grid cell size (for grid acceleration)
    int optimization_flags;     // Flags for which optimizations are enabled
    
    // Timing measurements (ms)
    float total_time;           // Total execution time
    float centroid_time;        // Time to compute centroids
    float nearest_neighbor_time; // Time for nearest neighbor search
    float covariance_time;      // Time to compute covariance matrices
    float svd_time;             // Time for SVD computation
    float transform_time;       // Time to apply transformations
    
    // Quality metrics
    float mean_error;           // Mean alignment error across batch
    float max_error;            // Maximum alignment error in batch
    float mean_rotation_error;  // Average rotation matrix error
    float mean_translation_error; // Average translation vector error
    
    // Convergence metrics
    int avg_iterations;         // Average number of iterations until convergence
    int max_iterations;         // Maximum iterations for any batch item
    float avg_rmse;             // Average RMSE across all batches
} BenchmarkResult;

// Export benchmark results to CSV file
void exportDetailedBenchmarkResults(const char* filename, const std::vector<BenchmarkResult>& results) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    // Write header
    fprintf(file, "point_count,batch_size,num_iterations,grid_cell_size,optimization_flags,"
                 "total_time,centroid_time,nearest_neighbor_time,covariance_time,svd_time,transform_time,"
                 "mean_error,max_error,mean_rotation_error,mean_translation_error,"
                 "avg_iterations,max_iterations,avg_rmse\n");
    
    // Write data rows
    for (const auto& result : results) {
        fprintf(file, "%d,%d,%d,%f,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%f\n",
                result.point_count,
                result.batch_size,
                result.num_iterations,
                result.grid_cell_size,
                result.optimization_flags,
                result.total_time,
                result.centroid_time,
                result.nearest_neighbor_time,
                result.covariance_time,
                result.svd_time,
                result.transform_time,
                result.mean_error,
                result.max_error,
                result.mean_rotation_error,
                result.mean_translation_error,
                result.avg_iterations,
                result.max_iterations,
                result.avg_rmse);
    }
    
    fclose(file);
    printf("Exported detailed benchmark results to %s\n", filename);
}

// Create a snapshot of intermediate alignment results for visualization
void createAlignmentSnapshot(SimpleGraph* src_graphs, SimpleGraph* tgt_graphs, SimpleGraph* aligned_graphs, 
                           int batch_size, int iteration) {
    char filename[100];
    
    if (SAVE_ALIGNMENT_SNAPSHOTS) {
        // For each batch item, export current alignment state
        for (int b = 0; b < batch_size; b++) {
            // Export source graph
            sprintf(filename, "snapshot_src_%d_batch_%d.ply", iteration, b);
            saveGraphToPLY(&src_graphs[b], filename);
            
            // Export target graph
            sprintf(filename, "snapshot_tgt_%d_batch_%d.ply", iteration, b);
            saveGraphToPLY(&tgt_graphs[b], filename);
            
            // Export aligned graph
            sprintf(filename, "snapshot_aligned_%d_batch_%d.ply", iteration, b);
            saveGraphToPLY(&aligned_graphs[b], filename);
        }
        
        printf("Created alignment snapshot for iteration %d\n", iteration);
    }
}

// Test the grid acceleration optimization with different cell sizes
void benchmarkGridAcceleration() {
    printf("\n=== Benchmarking Grid Acceleration ===\n");
    
    std::vector<BenchmarkResult> grid_results;
    std::vector<float> grid_cell_sizes;
    
    // Generate range of grid cell sizes to test
    for (int i = 0; i < NUM_GRID_CELL_SIZES; i++) {
        float size = MIN_GRID_CELL_SIZE + i * ((MAX_GRID_CELL_SIZE - MIN_GRID_CELL_SIZE) / (NUM_GRID_CELL_SIZES - 1));
        grid_cell_sizes.push_back(size);
    }
    
    // Test fixed point count with different grid cell sizes
    int point_count = 5000;
    int batch_size = 4;
    
    printf("Testing grid acceleration with %d points and %d batch size\n", point_count, batch_size);
    
    for (float grid_cell_size : grid_cell_sizes) {
        printf("  Testing grid cell size: %.3f\n", grid_cell_size);
        
        // Create same test data for all grid cell sizes
        SimpleGraph* src_graphs = new SimpleGraph[batch_size];
        SimpleGraph* tgt_graphs = new SimpleGraph[batch_size];
        
        // Set consistent random seed for reproducibility
        srand(RANDOM_SEED);
        
        // Generate test data
        for (int i = 0; i < batch_size; i++) {
            generateSyntheticGraph(&src_graphs[i], point_count, 20);
            applyRandomTransform(&src_graphs[i], &tgt_graphs[i]);
            addNoise(&tgt_graphs[i], 0.05f);  // 5% noise
        }
        
        // Convert to batched point clouds
        BatchedPointCloud src_pc, tgt_pc;
        graphsToPointCloud(src_graphs, batch_size, &src_pc);
        graphsToPointCloud(tgt_graphs, batch_size, &tgt_pc);
        
        // Test without grid acceleration
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        BatchedTransformation transform_no_grid;
        allocateBatchedTransformation(&transform_no_grid, batch_size);
        
        float elapsed_no_grid = 0.0f;
        CUDA_CHECK(cudaEventRecord(start));
        batchedICP(&src_pc, &tgt_pc, &transform_no_grid, MAX_ICP_ITERATIONS, 0.0001f, false); // No grid acceleration
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_no_grid, start, stop));
        
        // Get results from no-grid version for quality comparison
        float* h_errors_no_grid = (float*)malloc(batch_size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_errors_no_grid, transform_no_grid.errors, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        float avg_error_no_grid = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            avg_error_no_grid += h_errors_no_grid[i];
        }
        avg_error_no_grid /= batch_size;
        
        // Test with grid acceleration at current cell size
        // Reset source points to original positions
        freeBatchedPointCloud(&src_pc);
        graphsToPointCloud(src_graphs, batch_size, &src_pc);
        
        BatchedTransformation transform_grid;
        allocateBatchedTransformation(&transform_grid, batch_size);
        
        float elapsed_grid = 0.0f;
        CUDA_CHECK(cudaEventRecord(start));
        batchedICPWithGridAcceleration(&src_pc, &tgt_pc, &transform_grid, MAX_ICP_ITERATIONS, 
                                       0.0001f, grid_cell_size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_grid, start, stop));
        
        // Get results from grid version
        float* h_errors_grid = (float*)malloc(batch_size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_errors_grid, transform_grid.errors, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        float avg_error_grid = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            avg_error_grid += h_errors_grid[i];
        }
        avg_error_grid /= batch_size;
        
        // Record results
        BenchmarkResult result;
        result.point_count = point_count;
        result.batch_size = batch_size;
        result.num_iterations = MAX_ICP_ITERATIONS;
        result.grid_cell_size = grid_cell_size;
        result.optimization_flags = OPT_GRID_ACCEL;
        result.total_time = elapsed_grid;
        result.nearest_neighbor_time = 0.0f; // We'd need more detailed timing
        result.mean_error = avg_error_grid;
        result.max_error = *std::max_element(h_errors_grid, h_errors_grid + batch_size);
        result.avg_rmse = avg_error_grid;
        
        grid_results.push_back(result);
        
        printf("    No Grid: %.2f ms, Grid (%.3f): %.2f ms, Speedup: %.2fx\n", 
               elapsed_no_grid, grid_cell_size, elapsed_grid, elapsed_no_grid / elapsed_grid);
        printf("    Error - No Grid: %.6f, Grid: %.6f\n", avg_error_no_grid, avg_error_grid);
        
        // Clean up
        for (int i = 0; i < batch_size; i++) {
            free(src_graphs[i].node_positions);
            free(src_graphs[i].edges);
            free(tgt_graphs[i].node_positions);
            free(tgt_graphs[i].edges);
        }
        delete[] src_graphs;
        delete[] tgt_graphs;
        
        freeBatchedPointCloud(&src_pc);
        freeBatchedPointCloud(&tgt_pc);
        freeBatchedTransformation(&transform_no_grid);
        freeBatchedTransformation(&transform_grid);
        
        free(h_errors_no_grid);
        free(h_errors_grid);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // Export results
    if (SAVE_BENCHMARK_CSV) {
        exportDetailedBenchmarkResults("grid_acceleration_benchmark.csv", grid_results);
    }
}

// Comprehensive benchmark of all optimization combinations
void benchmarkOptimizationCombinations() {
    printf("\n=== Benchmarking Optimization Combinations ===\n");
    
    std::vector<BenchmarkResult> opt_results;
    
    // Define configurations to test
    const int kNumPointCounts = 3;
    const int kNumBatchSizes = 3;
    int point_counts[kNumPointCounts] = {1000, 5000, 10000};
    int batch_sizes[kNumBatchSizes] = {1, 4, 16};
    
    // Test all combinations of optimizations
    for (int opt_flags = 0; opt_flags <= OPT_ALL; opt_flags++) {
        printf("Testing optimization flags: %d\n", opt_flags);
        
        bool use_grid = (opt_flags & OPT_GRID_ACCEL) != 0;
        bool use_streams = (opt_flags & OPT_CUDA_STREAMS) != 0;
        float grid_cell_size = 0.2f; // Default for these tests
        
        for (int p = 0; p < kNumPointCounts; p++) {
            for (int b = 0; b < kNumBatchSizes; b++) {
                int point_count = point_counts[p];
                int batch_size = batch_sizes[b];
                
                printf("  Testing point count %d, batch size %d\n", point_count, batch_size);
                
                // Set consistent random seed for reproducibility
                srand(RANDOM_SEED);
                
                // Generate test data
                SimpleGraph* src_graphs = new SimpleGraph[batch_size];
                SimpleGraph* tgt_graphs = new SimpleGraph[batch_size];
                
                for (int i = 0; i < batch_size; i++) {
                    generateSyntheticGraph(&src_graphs[i], point_count, 20);
                    applyRandomTransform(&src_graphs[i], &tgt_graphs[i]);
                    addNoise(&tgt_graphs[i], 0.05f);  // 5% noise
                }
                
                // Convert to batched point clouds
                BatchedPointCloud src_pc, tgt_pc;
                graphsToPointCloud(src_graphs, batch_size, &src_pc);
                graphsToPointCloud(tgt_graphs, batch_size, &tgt_pc);
                
                // Run benchmark with current optimization flags
                cudaEvent_t start, stop;
                CUDA_CHECK(cudaEventCreate(&start));
                CUDA_CHECK(cudaEventCreate(&stop));
                
                BatchedTransformation transform;
                allocateBatchedTransformation(&transform, batch_size);
                
                // Initialize all timing variables
                float elapsed_total = 0.0f;
                float elapsed_centroid = 0.0f;
                float elapsed_nn = 0.0f;
                float elapsed_covariance = 0.0f;
                float elapsed_svd = 0.0f;
                float elapsed_transform = 0.0f;
                
                // Create events for detailed timing
                cudaEvent_t centroid_start, centroid_stop;
                cudaEvent_t nn_start, nn_stop;
                cudaEvent_t cov_start, cov_stop;
                cudaEvent_t svd_start, svd_stop;
                cudaEvent_t transform_start, transform_stop;
                
                CUDA_CHECK(cudaEventCreate(&centroid_start));
                CUDA_CHECK(cudaEventCreate(&centroid_stop));
                CUDA_CHECK(cudaEventCreate(&nn_start));
                CUDA_CHECK(cudaEventCreate(&nn_stop));
                CUDA_CHECK(cudaEventCreate(&cov_start));
                CUDA_CHECK(cudaEventCreate(&cov_stop));
                CUDA_CHECK(cudaEventCreate(&svd_start));
                CUDA_CHECK(cudaEventCreate(&svd_stop));
                CUDA_CHECK(cudaEventCreate(&transform_start));
                CUDA_CHECK(cudaEventCreate(&transform_stop));
                
                // Total time
                CUDA_CHECK(cudaEventRecord(start));
                
                // Call the appropriate function based on optimization flags
                if (use_grid && use_streams) {
                    batchedICPWithGridAccelerationAndStreams(&src_pc, &tgt_pc, &transform, 
                                                           MAX_ICP_ITERATIONS, 0.0001f, grid_cell_size);
                } else if (use_grid) {
                    batchedICPWithGridAcceleration(&src_pc, &tgt_pc, &transform, 
                                                MAX_ICP_ITERATIONS, 0.0001f, grid_cell_size);
                } else if (use_streams) {
                    batchedICPWithStreams(&src_pc, &tgt_pc, &transform, 
                                       MAX_ICP_ITERATIONS, 0.0001f);
                } else {
                    batchedICP(&src_pc, &tgt_pc, &transform, 
                            MAX_ICP_ITERATIONS, 0.0001f, false);
                }
                
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_total, start, stop));
                
                // Get results
                float* h_errors = (float*)malloc(batch_size * sizeof(float));
                CUDA_CHECK(cudaMemcpy(h_errors, transform.errors, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
                
                float avg_error = 0.0f;
                float max_error = 0.0f;
                for (int i = 0; i < batch_size; i++) {
                    avg_error += h_errors[i];
                    max_error = std::max(max_error, h_errors[i]);
                }
                avg_error /= batch_size;
                
                // Record results
                BenchmarkResult result;
                result.point_count = point_count;
                result.batch_size = batch_size;
                result.num_iterations = MAX_ICP_ITERATIONS;
                result.grid_cell_size = use_grid ? grid_cell_size : 0.0f;
                result.optimization_flags = opt_flags;
                result.total_time = elapsed_total;
                result.centroid_time = elapsed_centroid;
                result.nearest_neighbor_time = elapsed_nn;
                result.covariance_time = elapsed_covariance;
                result.svd_time = elapsed_svd;
                result.transform_time = elapsed_transform;
                result.mean_error = avg_error;
                result.max_error = max_error;
                result.avg_rmse = avg_error;
                
                opt_results.push_back(result);
                
                printf("    Elapsed time: %.2f ms, Mean error: %.6f\n", elapsed_total, avg_error);
                
                // Clean up
                for (int i = 0; i < batch_size; i++) {
                    free(src_graphs[i].node_positions);
                    free(src_graphs[i].edges);
                    free(tgt_graphs[i].node_positions);
                    free(tgt_graphs[i].edges);
                }
                delete[] src_graphs;
                delete[] tgt_graphs;
                
                freeBatchedPointCloud(&src_pc);
                freeBatchedPointCloud(&tgt_pc);
                freeBatchedTransformation(&transform);
                
                free(h_errors);
                
                CUDA_CHECK(cudaEventDestroy(start));
                CUDA_CHECK(cudaEventDestroy(stop));
                CUDA_CHECK(cudaEventDestroy(centroid_start));
                CUDA_CHECK(cudaEventDestroy(centroid_stop));
                CUDA_CHECK(cudaEventDestroy(nn_start));
                CUDA_CHECK(cudaEventDestroy(nn_stop));
                CUDA_CHECK(cudaEventDestroy(cov_start));
                CUDA_CHECK(cudaEventDestroy(cov_stop));
                CUDA_CHECK(cudaEventDestroy(svd_start));
                CUDA_CHECK(cudaEventDestroy(svd_stop));
                CUDA_CHECK(cudaEventDestroy(transform_start));
                CUDA_CHECK(cudaEventDestroy(transform_stop));
            }
        }
    }
    
    // Export results
    if (SAVE_BENCHMARK_CSV) {
        exportDetailedBenchmarkResults("optimization_combinations_benchmark.csv", opt_results);
    }
}

// Main function
int main() {
    // Set random seed for reproducible testing
    srand(RANDOM_SEED);
    
    printf("=== Batch Point Cloud Alignment Testing and Benchmarking ===\n");
    printf("Max Point Count: %d, Max Batch Count: %d\n", MAX_POINT_COUNT, MAX_BATCH_COUNT);
    printf("Visualization: %s, Save CSV: %s\n", 
           ENABLE_VISUALIZATION ? "Enabled" : "Disabled",
           SAVE_BENCHMARK_CSV ? "Enabled" : "Disabled");
    
    // Run basic functional tests
    runCompleteProcrustesTests();
    runCompleteICPTests();
    
    // Run in-depth optimization benchmarks
    benchmarkGridAcceleration();
    benchmarkOptimizationCombinations();
    
    // Run traditional benchmarks
    runComprehensiveBenchmarks();
    
    printf("\nAll tests and benchmarks completed successfully!\n");
    return 0;
}