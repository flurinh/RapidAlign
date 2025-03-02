#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>

// Structure to hold a 3D point
struct Point3D {
    float x, y, z;
};

// Structure to hold a simple graph
struct SimpleGraph {
    std::vector<Point3D> nodes;
    std::vector<std::pair<int, int>> edges;
};

// Function to generate a synthetic graph
void generateSyntheticGraph(SimpleGraph& graph, int num_nodes, int density) {
    graph.nodes.resize(num_nodes);
    
    // Generate random node positions in a unit cube
    for (int i = 0; i < num_nodes; i++) {
        graph.nodes[i].x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
        graph.nodes[i].y = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        graph.nodes[i].z = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Generate edges - simple random connections based on density (0-100%)
    int potential_edges = num_nodes * (num_nodes - 1) / 2;  // Maximum possible edges
    int target_edges = (int)(potential_edges * density / 100.0f);
    
    graph.edges.clear();
    
    // Generate random edges
    for (int i = 0; i < num_nodes && (int)graph.edges.size() < target_edges; i++) {
        for (int j = i+1; j < num_nodes && (int)graph.edges.size() < target_edges; j++) {
            // Add edge with probability based on density
            if ((float)rand() / RAND_MAX < (float)density / 100.0f) {
                graph.edges.push_back({i, j});
            }
        }
    }
    
    printf("Generated graph with %d nodes and %d edges\n", 
           (int)graph.nodes.size(), (int)graph.edges.size());
}

// Apply a transformation to a graph
void applyTransformation(const SimpleGraph& src, SimpleGraph& dst, 
                        float angle_deg, float tx, float ty, float tz) {
    // Copy structure
    dst = src;
    
    // Convert angle to radians
    float angle = angle_deg * M_PI / 180.0f;
    
    // Define rotation matrix (around z-axis)
    float R[9] = {
        cosf(angle), -sinf(angle), 0,
        sinf(angle),  cosf(angle), 0,
        0,           0,           1
    };
    
    // Apply transformation to each node
    for (size_t i = 0; i < src.nodes.size(); i++) {
        float x = src.nodes[i].x;
        float y = src.nodes[i].y;
        float z = src.nodes[i].z;
        
        dst.nodes[i].x = R[0]*x + R[1]*y + R[2]*z + tx;
        dst.nodes[i].y = R[3]*x + R[4]*y + R[5]*z + ty;
        dst.nodes[i].z = R[6]*x + R[7]*y + R[8]*z + tz;
    }
    
    printf("Applied transformation:\n");
    printf("Rotation angle: %.2f degrees (around z-axis)\n", angle_deg);
    printf("Translation: [%.2f, %.2f, %.2f]\n", tx, ty, tz);
}

// Add random noise to graph node positions
void addNoise(SimpleGraph& graph, float noise_level) {
    for (size_t i = 0; i < graph.nodes.size(); i++) {
        graph.nodes[i].x += ((float)rand() / RAND_MAX) * noise_level - noise_level/2;
        graph.nodes[i].y += ((float)rand() / RAND_MAX) * noise_level - noise_level/2;
        graph.nodes[i].z += ((float)rand() / RAND_MAX) * noise_level - noise_level/2;
    }
    printf("Added noise with level %.2f\n", noise_level);
}

// Simple Procrustes alignment
void procrustes(const SimpleGraph& src, const SimpleGraph& tgt, float R[9], float t[3]) {
    int n = src.nodes.size();
    if (n != (int)tgt.nodes.size()) {
        printf("Error: Source and target graphs must have the same number of nodes\n");
        return;
    }
    
    // 1. Compute centroids
    Point3D src_centroid = {0, 0, 0};
    Point3D tgt_centroid = {0, 0, 0};
    
    for (int i = 0; i < n; i++) {
        src_centroid.x += src.nodes[i].x;
        src_centroid.y += src.nodes[i].y;
        src_centroid.z += src.nodes[i].z;
        
        tgt_centroid.x += tgt.nodes[i].x;
        tgt_centroid.y += tgt.nodes[i].y;
        tgt_centroid.z += tgt.nodes[i].z;
    }
    
    src_centroid.x /= n;
    src_centroid.y /= n;
    src_centroid.z /= n;
    
    tgt_centroid.x /= n;
    tgt_centroid.y /= n;
    tgt_centroid.z /= n;
    
    // 2. Create centered versions of the point sets
    std::vector<Point3D> src_centered(n);
    std::vector<Point3D> tgt_centered(n);
    
    for (int i = 0; i < n; i++) {
        src_centered[i].x = src.nodes[i].x - src_centroid.x;
        src_centered[i].y = src.nodes[i].y - src_centroid.y;
        src_centered[i].z = src.nodes[i].z - src_centroid.z;
        
        tgt_centered[i].x = tgt.nodes[i].x - tgt_centroid.x;
        tgt_centered[i].y = tgt.nodes[i].y - tgt_centroid.y;
        tgt_centered[i].z = tgt.nodes[i].z - tgt_centroid.z;
    }
    
    // 3. Compute covariance matrix
    float cov[9] = {0};
    
    for (int i = 0; i < n; i++) {
        cov[0] += src_centered[i].x * tgt_centered[i].x;
        cov[1] += src_centered[i].x * tgt_centered[i].y;
        cov[2] += src_centered[i].x * tgt_centered[i].z;
        cov[3] += src_centered[i].y * tgt_centered[i].x;
        cov[4] += src_centered[i].y * tgt_centered[i].y;
        cov[5] += src_centered[i].y * tgt_centered[i].z;
        cov[6] += src_centered[i].z * tgt_centered[i].x;
        cov[7] += src_centered[i].z * tgt_centered[i].y;
        cov[8] += src_centered[i].z * tgt_centered[i].z;
    }
    
    // 4. For this simple demo, we'll assume rotation is only around z-axis
    // This is a simplification; a real implementation would use SVD
    float angle = atan2(cov[3] - cov[1], cov[0] + cov[4]);
    
    R[0] = cos(angle); R[1] = -sin(angle); R[2] = 0;
    R[3] = sin(angle); R[4] =  cos(angle); R[5] = 0;
    R[6] = 0;          R[7] = 0;           R[8] = 1;
    
    // 5. Compute translation: t = tgt_centroid - R * src_centroid
    float rotated_centroid[3];
    rotated_centroid[0] = R[0]*src_centroid.x + R[1]*src_centroid.y + R[2]*src_centroid.z;
    rotated_centroid[1] = R[3]*src_centroid.x + R[4]*src_centroid.y + R[5]*src_centroid.z;
    rotated_centroid[2] = R[6]*src_centroid.x + R[7]*src_centroid.y + R[8]*src_centroid.z;
    
    t[0] = tgt_centroid.x - rotated_centroid[0];
    t[1] = tgt_centroid.y - rotated_centroid[1];
    t[2] = tgt_centroid.z - rotated_centroid[2];
    
    printf("Estimated transformation:\n");
    printf("Rotation angle: %.2f degrees (around z-axis)\n", angle * 180.0f / M_PI);
    printf("Translation: [%.2f, %.2f, %.2f]\n", t[0], t[1], t[2]);
}

// Apply a transformation to a graph
void applyEstimatedTransform(const SimpleGraph& src, SimpleGraph& dst, const float R[9], const float t[3]) {
    // Copy structure
    dst = src;
    
    // Apply transformation to each node
    for (size_t i = 0; i < src.nodes.size(); i++) {
        float x = src.nodes[i].x;
        float y = src.nodes[i].y;
        float z = src.nodes[i].z;
        
        dst.nodes[i].x = R[0]*x + R[1]*y + R[2]*z + t[0];
        dst.nodes[i].y = R[3]*x + R[4]*y + R[5]*z + t[1];
        dst.nodes[i].z = R[6]*x + R[7]*y + R[8]*z + t[2];
    }
}

// Compute RMSE between two graphs
float computeRMSE(const SimpleGraph& a, const SimpleGraph& b) {
    if (a.nodes.size() != b.nodes.size()) {
        printf("Error: Graphs must have the same number of nodes\n");
        return -1;
    }
    
    float rmse = 0;
    for (size_t i = 0; i < a.nodes.size(); i++) {
        float dx = a.nodes[i].x - b.nodes[i].x;
        float dy = a.nodes[i].y - b.nodes[i].y;
        float dz = a.nodes[i].z - b.nodes[i].z;
        rmse += dx*dx + dy*dy + dz*dz;
    }
    
    return sqrt(rmse / a.nodes.size());
}

// Save graph to PLY file for visualization
void saveGraphToPLY(const SimpleGraph& graph, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        printf("Error: Could not open file %s\n", filename.c_str());
        return;
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << graph.nodes.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "element edge " << graph.edges.size() << "\n";
    file << "property int vertex1\n";
    file << "property int vertex2\n";
    file << "end_header\n";
    
    // Write vertices
    for (size_t i = 0; i < graph.nodes.size(); i++) {
        file << graph.nodes[i].x << " "
             << graph.nodes[i].y << " "
             << graph.nodes[i].z << "\n";
    }
    
    // Write edges
    for (size_t i = 0; i < graph.edges.size(); i++) {
        file << graph.edges[i].first << " " << graph.edges[i].second << "\n";
    }
    
    file.close();
    printf("Saved graph to %s\n", filename.c_str());
}

int main() {
    // Set random seed
    srand(time(NULL));
    
    // Generate a source graph
    SimpleGraph src_graph;
    generateSyntheticGraph(src_graph, 50, 20);  // 50 nodes, 20% edge density
    
    // Create a target graph by applying a known transformation
    SimpleGraph tgt_graph;
    applyTransformation(src_graph, tgt_graph, 30.0f, 0.5f, -0.3f, 0.8f);  // 30° rotation, [0.5, -0.3, 0.8] translation
    
    // Add some noise
    addNoise(tgt_graph, 0.05f);  // 5% noise
    
    // Save the original graphs for visualization
    saveGraphToPLY(src_graph, "cpu_src_graph.ply");
    saveGraphToPLY(tgt_graph, "cpu_tgt_graph.ply");
    
    // Run Procrustes alignment
    float R_est[9];
    float t_est[3];
    procrustes(src_graph, tgt_graph, R_est, t_est);
    
    // Apply the estimated transformation
    SimpleGraph aligned_graph;
    applyEstimatedTransform(src_graph, aligned_graph, R_est, t_est);
    
    // Save the aligned graph
    saveGraphToPLY(aligned_graph, "cpu_aligned_graph.ply");
    
    // Compute and display RMSE
    float rmse = computeRMSE(aligned_graph, tgt_graph);
    printf("RMSE between aligned and target: %.6f\n", rmse);
    
    // Generate and align multiple graphs with varying sizes
    printf("\n=== Batch Alignment Simulation ===\n");
    
    const int num_graphs = 3;
    SimpleGraph src_batch[num_graphs];
    SimpleGraph tgt_batch[num_graphs];
    SimpleGraph aligned_batch[num_graphs];
    
    int sizes[num_graphs] = {30, 50, 70};
    
    for (int i = 0; i < num_graphs; i++) {
        printf("\nProcessing graph %d (%d nodes):\n", i, sizes[i]);
        
        // Generate source graph
        generateSyntheticGraph(src_batch[i], sizes[i], 15);
        
        // Create target by transformation
        float angle = (float)(20 + i*10);  // Different angles for each graph
        float tx = 0.3f + i*0.2f;
        float ty = -0.2f - i*0.1f;
        float tz = 0.5f + i*0.3f;
        
        applyTransformation(src_batch[i], tgt_batch[i], angle, tx, ty, tz);
        
        // Add noise
        addNoise(tgt_batch[i], 0.05f);
        
        // Save graphs
        saveGraphToPLY(src_batch[i], "cpu_src_" + std::to_string(i) + ".ply");
        saveGraphToPLY(tgt_batch[i], "cpu_tgt_" + std::to_string(i) + ".ply");
        
        // Align
        float R_batch[9];
        float t_batch[3];
        procrustes(src_batch[i], tgt_batch[i], R_batch, t_batch);
        
        // Apply transformation
        applyEstimatedTransform(src_batch[i], aligned_batch[i], R_batch, t_batch);
        
        // Save aligned graph
        saveGraphToPLY(aligned_batch[i], "cpu_aligned_" + std::to_string(i) + ".ply");
        
        // Compute RMSE
        float batch_rmse = computeRMSE(aligned_batch[i], tgt_batch[i]);
        printf("RMSE for graph %d: %.6f\n", i, batch_rmse);
    }
    
    printf("\nRun python visualize_graphs.py to visualize the results\n");
    
    return 0;
}