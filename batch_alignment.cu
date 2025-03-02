/*
 * batch_alignment.cu
 * Batch-Enabled Point Cloud Alignment for Graph Neural Networks
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define MAX_POINTS_PER_BATCH_ITEM 1024  // Can be adjusted based on expected point cloud sizes
#define MAX_BATCH_SIZE 32               // Maximum number of point clouds in a batch

// Macro for error checking
#define CUDA_CHECK(err) do { \
    if(err != cudaSuccess) { \
        printf("CUDA error: %s, at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

/*
 * Data Structures
 */

// Structure to hold information about batched point clouds with variable sizes
typedef struct {
    float* points;              // Packed points data (x,y,z coordinates contiguous in memory)
    int* batch_sizes;           // Number of points in each batch item
    int* batch_offsets;         // Start index of each batch item in the points array
    int batch_count;            // Number of point clouds in the batch
    int total_points;           // Total number of points across all batch items
} BatchedPointCloud;

// Structure to hold transformation matrices for each batch item
typedef struct {
    float* rotations;           // 3x3 rotation matrices for each batch item, stored contiguously
    float* translations;        // 3D translation vectors for each batch item, stored contiguously
    float* errors;              // Error/quality metrics for each transformation
    int batch_count;            // Number of transformations (same as batch_count in BatchedPointCloud)
} BatchedTransformation;

/*
 * Helper Functions
 */

// Allocate memory for a batched point cloud
void allocateBatchedPointCloud(BatchedPointCloud* cloud, int batch_count, int* sizes) {
    cloud->batch_count = batch_count;
    
    // Allocate device memory for batch sizes and offsets
    CUDA_CHECK(cudaMalloc(&cloud->batch_sizes, batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cloud->batch_offsets, batch_count * sizeof(int)));
    
    // Copy batch sizes to device
    CUDA_CHECK(cudaMemcpy(cloud->batch_sizes, sizes, batch_count * sizeof(int), cudaMemcpyHostToDevice));
    
    // Calculate total points and offsets
    int total_points = 0;
    int* host_offsets = (int*)malloc(batch_count * sizeof(int));
    
    for (int i = 0; i < batch_count; i++) {
        host_offsets[i] = total_points;
        total_points += sizes[i];
    }
    
    cloud->total_points = total_points;
    
    // Copy offsets to device
    CUDA_CHECK(cudaMemcpy(cloud->batch_offsets, host_offsets, batch_count * sizeof(int), cudaMemcpyHostToDevice));
    free(host_offsets);
    
    // Allocate memory for the points data
    CUDA_CHECK(cudaMalloc(&cloud->points, total_points * 3 * sizeof(float)));
}

// Free memory for a batched point cloud
void freeBatchedPointCloud(BatchedPointCloud* cloud) {
    CUDA_CHECK(cudaFree(cloud->points));
    CUDA_CHECK(cudaFree(cloud->batch_sizes));
    CUDA_CHECK(cudaFree(cloud->batch_offsets));
}

// Allocate memory for batch transformations
void allocateBatchedTransformation(BatchedTransformation* transform, int batch_count) {
    transform->batch_count = batch_count;
    
    // Allocate device memory for rotations (3x3 matrix per batch item)
    CUDA_CHECK(cudaMalloc(&transform->rotations, batch_count * 9 * sizeof(float)));
    
    // Allocate device memory for translations (3D vector per batch item)
    CUDA_CHECK(cudaMalloc(&transform->translations, batch_count * 3 * sizeof(float)));
    
    // Allocate device memory for errors
    CUDA_CHECK(cudaMalloc(&transform->errors, batch_count * sizeof(float)));
}

// Free memory for batch transformations
void freeBatchedTransformation(BatchedTransformation* transform) {
    CUDA_CHECK(cudaFree(transform->rotations));
    CUDA_CHECK(cudaFree(transform->translations));
    CUDA_CHECK(cudaFree(transform->errors));
}

/*
 * CUDA Kernels for Batched Point Cloud Processing
 */

// Kernel: Compute centroids for batched point clouds
__global__ void batchedComputeCentroid(
    float* points,               // Input: [total_points, 3] packed point coordinates
    int* batch_sizes,            // Input: [batch_count] number of points per batch item
    int* batch_offsets,          // Input: [batch_count] offset of each batch item in points array
    int batch_count,             // Input: number of batch items
    float* centroids             // Output: [batch_count, 3] centroids for each batch item
) {
    extern __shared__ float shared_data[];
    
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int num_points = batch_sizes[batch_idx];
    int offset = batch_offsets[batch_idx];
    
    // Initialize shared memory for reduction
    float* sdata_x = shared_data;
    float* sdata_y = shared_data + blockDim.x;
    float* sdata_z = shared_data + 2 * blockDim.x;
    
    sdata_x[tid] = 0.0f;
    sdata_y[tid] = 0.0f;
    sdata_z[tid] = 0.0f;
    
    // Accumulate points for this batch item
    for (int i = tid; i < num_points; i += blockDim.x) {
        int idx = (offset + i) * 3;
        sdata_x[tid] += points[idx];
        sdata_y[tid] += points[idx + 1];
        sdata_z[tid] += points[idx + 2];
    }
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_x[tid] += sdata_x[tid + s];
            sdata_y[tid] += sdata_y[tid + s];
            sdata_z[tid] += sdata_z[tid + s];
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (tid == 0) {
        centroids[batch_idx * 3] = sdata_x[0] / num_points;
        centroids[batch_idx * 3 + 1] = sdata_y[0] / num_points;
        centroids[batch_idx * 3 + 2] = sdata_z[0] / num_points;
    }
}

// Kernel: Subtract centroids from each point in batched point clouds
__global__ void batchedSubtractCentroid(
    float* points,               // Input/Output: [total_points, 3] point coordinates
    int* batch_sizes,            // Input: [batch_count] number of points per batch item
    int* batch_offsets,          // Input: [batch_count] offset of each batch item
    int batch_count,             // Input: number of batch items
    float* centroids             // Input: [batch_count, 3] centroid for each batch item
) {
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int num_points = batch_sizes[batch_idx];
    int offset = batch_offsets[batch_idx];
    
    // Get centroid for this batch
    float cx = centroids[batch_idx * 3];
    float cy = centroids[batch_idx * 3 + 1];
    float cz = centroids[batch_idx * 3 + 2];
    
    // Subtract centroid from each point
    for (int i = tid; i < num_points; i += blockDim.x) {
        int idx = (offset + i) * 3;
        points[idx] -= cx;
        points[idx + 1] -= cy;
        points[idx + 2] -= cz;
    }
}

// Kernel: Compute covariance matrices for each batch item
__global__ void batchedComputeCovariance(
    float* src_points,           // Input: [total_points_src, 3] centered source points
    float* tgt_points,           // Input: [total_points_tgt, 3] centered target points
    int* batch_sizes,            // Input: [batch_count] number of points per batch item
    int* src_offsets,            // Input: [batch_count] offset of each source batch item
    int* tgt_offsets,            // Input: [batch_count] offset of each target batch item
    int batch_count,             // Input: number of batch items
    float* covariance_matrices   // Output: [batch_count, 3, 3] covariance matrices
) {
    extern __shared__ float shared_cov[];
    
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int num_points = batch_sizes[batch_idx];
    int src_offset = src_offsets[batch_idx];
    int tgt_offset = tgt_offsets[batch_idx];
    
    // Initialize shared memory for 3x3 covariance matrix
    for (int i = tid; i < 9; i += blockDim.x) {
        shared_cov[i] = 0.0f;
    }
    __syncthreads();
    
    // Compute partial covariance matrix
    for (int i = tid; i < num_points; i += blockDim.x) {
        int src_idx = (src_offset + i) * 3;
        int tgt_idx = (tgt_offset + i) * 3;
        
        float sx = src_points[src_idx];
        float sy = src_points[src_idx + 1];
        float sz = src_points[src_idx + 2];
        
        float tx = tgt_points[tgt_idx];
        float ty = tgt_points[tgt_idx + 1];
        float tz = tgt_points[tgt_idx + 2];
        
        // Cross-covariance calculation (outer product)
        atomicAdd(&shared_cov[0], sx * tx);  // [0,0]
        atomicAdd(&shared_cov[1], sx * ty);  // [0,1]
        atomicAdd(&shared_cov[2], sx * tz);  // [0,2]
        atomicAdd(&shared_cov[3], sy * tx);  // [1,0]
        atomicAdd(&shared_cov[4], sy * ty);  // [1,1]
        atomicAdd(&shared_cov[5], sy * tz);  // [1,2]
        atomicAdd(&shared_cov[6], sz * tx);  // [2,0]
        atomicAdd(&shared_cov[7], sz * ty);  // [2,1]
        atomicAdd(&shared_cov[8], sz * tz);  // [2,2]
    }
    __syncthreads();
    
    // Write result to global memory
    if (tid < 9) {
        covariance_matrices[batch_idx * 9 + tid] = shared_cov[tid] / num_points;
    }
}

// Structure for accelerated nearest neighbor search with uniform grid
typedef struct {
    int* grid_indices;    // Starting indices for each grid cell
    int* grid_counts;     // Number of points in each grid cell
    int* point_indices;   // Sorted point indices
    float* grid_min;      // Minimum coordinates of the grid (3D)
    float* grid_max;      // Maximum coordinates of the grid (3D)
    float* cell_size;     // Size of each grid cell (3D)
    int resolution;       // Grid resolution (cells per dimension)
} UniformGrid;

// Constants for grid-based nearest neighbor search
#define GRID_RESOLUTION 16
#define GRID_CELL_COUNT (GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION)

// Functions to allocate and free a uniform grid
void allocateUniformGrid(UniformGrid* grid, int max_points, int batch_count) {
    grid->resolution = GRID_RESOLUTION;
    
    // Allocate memory for grid properties
    CUDA_CHECK(cudaMalloc(&grid->grid_min, batch_count * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grid->grid_max, batch_count * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grid->cell_size, batch_count * 3 * sizeof(float)));
    
    // Allocate memory for grid cells
    CUDA_CHECK(cudaMalloc(&grid->grid_indices, batch_count * GRID_CELL_COUNT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid->grid_counts, batch_count * GRID_CELL_COUNT * sizeof(int)));
    
    // Allocate memory for point indices
    CUDA_CHECK(cudaMalloc(&grid->point_indices, max_points * sizeof(int)));
}

void freeUniformGrid(UniformGrid* grid) {
    CUDA_CHECK(cudaFree(grid->grid_min));
    CUDA_CHECK(cudaFree(grid->grid_max));
    CUDA_CHECK(cudaFree(grid->cell_size));
    CUDA_CHECK(cudaFree(grid->grid_indices));
    CUDA_CHECK(cudaFree(grid->grid_counts));
    CUDA_CHECK(cudaFree(grid->point_indices));
}

// Kernel: Find bounding box for each batch of points
__global__ void findBoundingBox(
    const float* points,        // Input: [total_points, 3] point coordinates
    int* batch_sizes,           // Input: [batch_count] number of points per batch
    int* batch_offsets,         // Input: [batch_count] offset of each batch
    int batch_count,            // Input: number of batch items
    float* grid_min,            // Output: [batch_count, 3] minimum coordinates
    float* grid_max             // Output: [batch_count, 3] maximum coordinates
) {
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int num_points = batch_sizes[batch_idx];
    int offset = batch_offsets[batch_idx];
    
    // Initialize to extreme values
    float min_x = FLT_MAX;
    float min_y = FLT_MAX;
    float min_z = FLT_MAX;
    float max_x = -FLT_MAX;
    float max_y = -FLT_MAX;
    float max_z = -FLT_MAX;
    
    // Find min/max values
    for (int i = tid; i < num_points; i += blockDim.x) {
        int idx = (offset + i) * 3;
        float x = points[idx];
        float y = points[idx + 1];
        float z = points[idx + 2];
        
        min_x = min(min_x, x);
        min_y = min(min_y, y);
        min_z = min(min_z, z);
        max_x = max(max_x, x);
        max_y = max(max_y, y);
        max_z = max(max_z, z);
    }
    
    // Use shared memory for reduction
    __shared__ float s_min_x[BLOCK_SIZE];
    __shared__ float s_min_y[BLOCK_SIZE];
    __shared__ float s_min_z[BLOCK_SIZE];
    __shared__ float s_max_x[BLOCK_SIZE];
    __shared__ float s_max_y[BLOCK_SIZE];
    __shared__ float s_max_z[BLOCK_SIZE];
    
    s_min_x[tid] = min_x;
    s_min_y[tid] = min_y;
    s_min_z[tid] = min_z;
    s_max_x[tid] = max_x;
    s_max_y[tid] = max_y;
    s_max_z[tid] = max_z;
    __syncthreads();
    
    // Perform reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min_x[tid] = min(s_min_x[tid], s_min_x[tid + s]);
            s_min_y[tid] = min(s_min_y[tid], s_min_y[tid + s]);
            s_min_z[tid] = min(s_min_z[tid], s_min_z[tid + s]);
            s_max_x[tid] = max(s_max_x[tid], s_max_x[tid + s]);
            s_max_y[tid] = max(s_max_y[tid], s_max_y[tid + s]);
            s_max_z[tid] = max(s_max_z[tid], s_max_z[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (tid == 0) {
        grid_min[batch_idx * 3] = s_min_x[0];
        grid_min[batch_idx * 3 + 1] = s_min_y[0];
        grid_min[batch_idx * 3 + 2] = s_min_z[0];
        
        grid_max[batch_idx * 3] = s_max_x[0];
        grid_max[batch_idx * 3 + 1] = s_max_y[0];
        grid_max[batch_idx * 3 + 2] = s_max_z[0];
    }
}

// Kernel: Compute cell size for each batch
__global__ void computeCellSize(
    float* grid_min,            // Input: [batch_count, 3] minimum coordinates
    float* grid_max,            // Input: [batch_count, 3] maximum coordinates
    int batch_count,            // Input: number of batch items
    float* cell_size,           // Output: [batch_count, 3] cell size for each dimension
    int resolution              // Input: grid resolution
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_count) return;
    
    // Compute cell size with a small epsilon to avoid division by zero
    float min_x = grid_min[batch_idx * 3];
    float min_y = grid_min[batch_idx * 3 + 1];
    float min_z = grid_min[batch_idx * 3 + 2];
    float max_x = grid_max[batch_idx * 3];
    float max_y = grid_max[batch_idx * 3 + 1];
    float max_z = grid_max[batch_idx * 3 + 2];
    
    // Add a small margin to the bounding box
    float margin = 0.01f;
    min_x -= margin;
    min_y -= margin;
    min_z -= margin;
    max_x += margin;
    max_y += margin;
    max_z += margin;
    
    // Store adjusted bounding box
    grid_min[batch_idx * 3] = min_x;
    grid_min[batch_idx * 3 + 1] = min_y;
    grid_min[batch_idx * 3 + 2] = min_z;
    grid_max[batch_idx * 3] = max_x;
    grid_max[batch_idx * 3 + 1] = max_y;
    grid_max[batch_idx * 3 + 2] = max_z;
    
    // Compute cell size
    float size_x = (max_x - min_x) / resolution;
    float size_y = (max_y - min_y) / resolution;
    float size_z = (max_z - min_z) / resolution;
    
    // Ensure cell size is not zero (with epsilon)
    float epsilon = 1e-6f;
    cell_size[batch_idx * 3] = max(size_x, epsilon);
    cell_size[batch_idx * 3 + 1] = max(size_y, epsilon);
    cell_size[batch_idx * 3 + 2] = max(size_z, epsilon);
}

// Kernel: Count points per cell for each batch
__global__ void countPointsPerCell(
    const float* points,        // Input: [total_points, 3] point coordinates
    int* batch_sizes,           // Input: [batch_count] number of points per batch
    int* batch_offsets,         // Input: [batch_count] offset of each batch
    int batch_count,            // Input: number of batch items
    float* grid_min,            // Input: [batch_count, 3] minimum coordinates
    float* cell_size,           // Input: [batch_count, 3] cell size for each dimension
    int* grid_counts,           // Output: [batch_count, GRID_CELL_COUNT] number of points per cell
    int resolution              // Input: grid resolution
) {
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int num_points = batch_sizes[batch_idx];
    int offset = batch_offsets[batch_idx];
    
    // Get grid properties for this batch
    float min_x = grid_min[batch_idx * 3];
    float min_y = grid_min[batch_idx * 3 + 1];
    float min_z = grid_min[batch_idx * 3 + 2];
    float size_x = cell_size[batch_idx * 3];
    float size_y = cell_size[batch_idx * 3 + 1];
    float size_z = cell_size[batch_idx * 3 + 2];
    
    // Count points per cell
    for (int i = tid; i < num_points; i += blockDim.x) {
        int idx = (offset + i) * 3;
        float x = points[idx];
        float y = points[idx + 1];
        float z = points[idx + 2];
        
        // Compute cell indices
        int cell_x = min(max((int)((x - min_x) / size_x), 0), resolution - 1);
        int cell_y = min(max((int)((y - min_y) / size_y), 0), resolution - 1);
        int cell_z = min(max((int)((z - min_z) / size_z), 0), resolution - 1);
        
        // Compute linear cell index
        int cell_idx = cell_z * resolution * resolution + cell_y * resolution + cell_x;
        
        // Increment cell count
        atomicAdd(&grid_counts[batch_idx * GRID_CELL_COUNT + cell_idx], 1);
    }
}

// Kernel: Compute starting indices for each cell
__global__ void computeCellStartIndices(
    int* grid_counts,           // Input/Output: [batch_count, GRID_CELL_COUNT] number of points per cell
    int* grid_indices,          // Output: [batch_count, GRID_CELL_COUNT] start index for each cell
    int batch_count,            // Input: number of batch items
    int cell_count              // Input: total number of cells
) {
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int batch_offset = batch_idx * cell_count;
    
    // Compute exclusive prefix sum (scan)
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < cell_count; i++) {
            int count = grid_counts[batch_offset + i];
            grid_indices[batch_offset + i] = sum;
            sum += count;
            
            // Reset count for the next kernel
            grid_counts[batch_offset + i] = 0;
        }
    }
}

// Kernel: Assign points to cells
__global__ void assignPointsToCells(
    const float* points,        // Input: [total_points, 3] point coordinates
    int* batch_sizes,           // Input: [batch_count] number of points per batch
    int* batch_offsets,         // Input: [batch_count] offset of each batch
    int batch_count,            // Input: number of batch items
    float* grid_min,            // Input: [batch_count, 3] minimum coordinates
    float* cell_size,           // Input: [batch_count, 3] cell size for each dimension
    int* grid_counts,           // Input/Output: [batch_count, GRID_CELL_COUNT] counter for each cell
    int* grid_indices,          // Input: [batch_count, GRID_CELL_COUNT] start index for each cell
    int* point_indices,         // Output: [max_points] indices of points sorted by cell
    int resolution,             // Input: grid resolution
    int cell_count              // Input: total number of cells
) {
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int num_points = batch_sizes[batch_idx];
    int offset = batch_offsets[batch_idx];
    
    // Get grid properties for this batch
    float min_x = grid_min[batch_idx * 3];
    float min_y = grid_min[batch_idx * 3 + 1];
    float min_z = grid_min[batch_idx * 3 + 2];
    float size_x = cell_size[batch_idx * 3];
    float size_y = cell_size[batch_idx * 3 + 1];
    float size_z = cell_size[batch_idx * 3 + 2];
    
    int batch_offset = batch_idx * cell_count;
    
    // Assign points to cells
    for (int i = tid; i < num_points; i += blockDim.x) {
        int global_idx = offset + i;
        int idx = global_idx * 3;
        float x = points[idx];
        float y = points[idx + 1];
        float z = points[idx + 2];
        
        // Compute cell indices
        int cell_x = min(max((int)((x - min_x) / size_x), 0), resolution - 1);
        int cell_y = min(max((int)((y - min_y) / size_y), 0), resolution - 1);
        int cell_z = min(max((int)((z - min_z) / size_z), 0), resolution - 1);
        
        // Compute linear cell index
        int cell_idx = cell_z * resolution * resolution + cell_y * resolution + cell_x;
        
        // Get position in point_indices array
        int pos = atomicAdd(&grid_counts[batch_offset + cell_idx], 1);
        int write_idx = grid_indices[batch_offset + cell_idx] + pos;
        
        // Store point index
        point_indices[write_idx] = i;  // Local index relative to batch
    }
}

// Kernel: Accelerated nearest neighbor search with grid
__global__ void batchedGridNN(
    const float* src_points,         // Input: [total_points_src, 3] source points
    const float* tgt_points,         // Input: [total_points_tgt, 3] target points
    int* src_batch_sizes,            // Input: [batch_count] number of source points per batch
    int* tgt_batch_sizes,            // Input: [batch_count] number of target points per batch
    int* src_offsets,                // Input: [batch_count] offset of each source batch
    int* tgt_offsets,                // Input: [batch_count] offset of each target batch
    int batch_count,                 // Input: number of batch items
    float* grid_min,                 // Input: [batch_count, 3] minimum coordinates
    float* cell_size,                // Input: [batch_count, 3] cell size for each dimension
    int* grid_counts,                // Input: [batch_count, GRID_CELL_COUNT] number of points per cell
    int* grid_indices,               // Input: [batch_count, GRID_CELL_COUNT] start index for each cell
    int* point_indices,              // Input: [max_points] indices of points sorted by cell
    int* correspondences,            // Output: [total_points_src] index of nearest neighbor
    float* distances,                // Output: [total_points_src] squared distance to nearest neighbor
    int resolution,                  // Input: grid resolution
    int cell_count                   // Input: total number of cells
) {
    // Each thread processes one source point
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_count) return;
    
    int src_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int src_size = src_batch_sizes[batch_idx];
    if (src_idx >= src_size) return;
    
    int batch_offset = batch_idx * cell_count;
    
    // Get point coordinates
    int global_src_idx = src_offsets[batch_idx] + src_idx;
    float sx = src_points[global_src_idx * 3];
    float sy = src_points[global_src_idx * 3 + 1];
    float sz = src_points[global_src_idx * 3 + 2];
    
    // Get grid properties for this batch
    float min_x = grid_min[batch_idx * 3];
    float min_y = grid_min[batch_idx * 3 + 1];
    float min_z = grid_min[batch_idx * 3 + 2];
    float size_x = cell_size[batch_idx * 3];
    float size_y = cell_size[batch_idx * 3 + 1];
    float size_z = cell_size[batch_idx * 3 + 2];
    
    // Compute cell indices for source point
    int cell_x = min(max((int)((sx - min_x) / size_x), 0), resolution - 1);
    int cell_y = min(max((int)((sy - min_y) / size_y), 0), resolution - 1);
    int cell_z = min(max((int)((sz - min_z) / size_z), 0), resolution - 1);
    
    // Initialize min distance and index
    float min_dist = 1e10f;
    int min_idx = -1;
    
    // Search in current cell and neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        int nz = cell_z + dz;
        if (nz < 0 || nz >= resolution) continue;
        
        for (int dy = -1; dy <= 1; dy++) {
            int ny = cell_y + dy;
            if (ny < 0 || ny >= resolution) continue;
            
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cell_x + dx;
                if (nx < 0 || nx >= resolution) continue;
                
                // Compute linear cell index
                int cell_idx = nz * resolution * resolution + ny * resolution + nx;
                
                // Get points in this cell
                int start = grid_indices[batch_offset + cell_idx];
                int count = grid_counts[batch_offset + cell_idx];
                
                // Check all points in this cell
                for (int i = 0; i < count; i++) {
                    int tgt_local_idx = point_indices[start + i];
                    int global_tgt_idx = tgt_offsets[batch_idx] + tgt_local_idx;
                    
                    float tx = tgt_points[global_tgt_idx * 3];
                    float ty = tgt_points[global_tgt_idx * 3 + 1];
                    float tz = tgt_points[global_tgt_idx * 3 + 2];
                    
                    float dx = sx - tx;
                    float dy = sy - ty;
                    float dz = sz - tz;
                    float dist = dx*dx + dy*dy + dz*dz;
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = global_tgt_idx;
                    }
                }
            }
        }
    }
    
    // If no point found in neighboring cells (rare case), use brute force
    if (min_idx == -1) {
        int tgt_size = tgt_batch_sizes[batch_idx];
        int tgt_offset = tgt_offsets[batch_idx];
        
        for (int j = 0; j < tgt_size; j++) {
            int global_tgt_idx = tgt_offset + j;
            
            float tx = tgt_points[global_tgt_idx * 3];
            float ty = tgt_points[global_tgt_idx * 3 + 1];
            float tz = tgt_points[global_tgt_idx * 3 + 2];
            
            float dx = sx - tx;
            float dy = sy - ty;
            float dz = sz - tz;
            float dist = dx*dx + dy*dy + dz*dz;
            
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = global_tgt_idx;
            }
        }
    }
    
    // Store correspondence and distance
    correspondences[global_src_idx] = min_idx;
    distances[global_src_idx] = min_dist;
}

// Function to build the uniform grid for a target point cloud
void buildUniformGrid(
    UniformGrid* grid,              // Grid structure to initialize
    float* points,                  // Input: [total_points, 3] target points
    int* batch_sizes,               // Input: [batch_count] number of points per batch
    int* batch_offsets,             // Input: [batch_count] offset of each batch
    int batch_count,                // Input: number of batch items
    int total_points                // Input: total number of points
) {
    int threads = BLOCK_SIZE;
    
    // Step 1: Find bounding box for each batch
    findBoundingBox<<<batch_count, threads>>>(
        points, batch_sizes, batch_offsets, batch_count,
        grid->grid_min, grid->grid_max);
    
    // Step 2: Compute cell size for each batch
    computeCellSize<<<(batch_count + threads - 1) / threads, threads>>>(
        grid->grid_min, grid->grid_max, batch_count, grid->cell_size, grid->resolution);
    
    // Step 3: Initialize grid counts to zero
    CUDA_CHECK(cudaMemset(grid->grid_counts, 0, batch_count * GRID_CELL_COUNT * sizeof(int)));
    
    // Step 4: Count points per cell
    countPointsPerCell<<<batch_count, threads>>>(
        points, batch_sizes, batch_offsets, batch_count,
        grid->grid_min, grid->cell_size, grid->grid_counts, grid->resolution);
    
    // Step 5: Compute starting indices for each cell
    computeCellStartIndices<<<batch_count, 1>>>(
        grid->grid_counts, grid->grid_indices, batch_count, GRID_CELL_COUNT);
    
    // Step 6: Assign points to cells
    assignPointsToCells<<<batch_count, threads>>>(
        points, batch_sizes, batch_offsets, batch_count,
        grid->grid_min, grid->cell_size, grid->grid_counts, grid->grid_indices,
        grid->point_indices, grid->resolution, GRID_CELL_COUNT);
}

// Kernel: Brute-force nearest neighbor search for each batch item
__global__ void batchedBruteForceNN(
    float* src_points,           // Input: [total_points_src, 3] source points
    float* tgt_points,           // Input: [total_points_tgt, 3] target points
    int* src_batch_sizes,        // Input: [batch_count] number of source points per batch
    int* tgt_batch_sizes,        // Input: [batch_count] number of target points per batch
    int* src_offsets,            // Input: [batch_count] offset of each source batch item
    int* tgt_offsets,            // Input: [batch_count] offset of each target batch item
    int batch_count,             // Input: number of batch items
    int* correspondences,        // Output: [total_points_src] index of nearest neighbor in target
    float* distances             // Output: [total_points_src] squared distance to nearest neighbor
) {
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int src_size = src_batch_sizes[batch_idx];
    int tgt_size = tgt_batch_sizes[batch_idx];
    int src_offset = src_offsets[batch_idx];
    int tgt_offset = tgt_offsets[batch_idx];
    
    // For each source point, find nearest target point
    for (int i = tid; i < src_size; i += blockDim.x) {
        int src_idx = (src_offset + i) * 3;
        float sx = src_points[src_idx];
        float sy = src_points[src_idx + 1];
        float sz = src_points[src_idx + 2];
        
        float min_dist = 1e10f;
        int min_idx = -1;
        
        // Brute-force search through all target points
        for (int j = 0; j < tgt_size; j++) {
            int tgt_idx = (tgt_offset + j) * 3;
            float tx = tgt_points[tgt_idx];
            float ty = tgt_points[tgt_idx + 1];
            float tz = tgt_points[tgt_idx + 2];
            
            float dx = sx - tx;
            float dy = sy - ty;
            float dz = sz - tz;
            float dist = dx*dx + dy*dy + dz*dz;
            
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }
        
        // Store correspondence and distance
        correspondences[src_offset + i] = tgt_offset + min_idx;  // Global index of target point
        distances[src_offset + i] = min_dist;
    }
}

// Kernel: Apply transformations to each batch item
__global__ void batchedApplyTransform(
    float* points,               // Input/Output: [total_points, 3] point coordinates
    int* batch_sizes,            // Input: [batch_count] number of points per batch item
    int* batch_offsets,          // Input: [batch_count] offset of each batch item
    int batch_count,             // Input: number of batch items
    float* rotations,            // Input: [batch_count, 3, 3] rotation matrices
    float* translations          // Input: [batch_count, 3] translation vectors
) {
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int num_points = batch_sizes[batch_idx];
    int offset = batch_offsets[batch_idx];
    
    // Get rotation matrix for this batch
    float* R = &rotations[batch_idx * 9];
    
    // Get translation vector for this batch
    float tx = translations[batch_idx * 3];
    float ty = translations[batch_idx * 3 + 1];
    float tz = translations[batch_idx * 3 + 2];
    
    // Apply transformation to each point
    for (int i = tid; i < num_points; i += blockDim.x) {
        int idx = (offset + i) * 3;
        float px = points[idx];
        float py = points[idx + 1];
        float pz = points[idx + 2];
        
        // Apply rotation
        float rx = R[0]*px + R[1]*py + R[2]*pz;
        float ry = R[3]*px + R[4]*py + R[5]*pz;
        float rz = R[6]*px + R[7]*py + R[8]*pz;
        
        // Apply translation
        points[idx] = rx + tx;
        points[idx + 1] = ry + ty;
        points[idx + 2] = rz + tz;
    }
}

// Kernel: Compute average error for each batch item from distances
__global__ void computeAverageError(
    float* distances,           // Input: [total_points] distances between correspondences
    int* batch_sizes,           // Input: [batch_count] number of points per batch
    int* batch_offsets,         // Input: [batch_count] offset of each batch
    int batch_count,            // Input: number of batch items
    float* errors               // Output: [batch_count] average error for each batch
) {
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int num_points = batch_sizes[batch_idx];
    int offset = batch_offsets[batch_idx];
    
    // Use shared memory for reduction
    extern __shared__ float s_error[];
    s_error[tid] = 0.0f;
    
    // Sum distances for each point in batch
    for (int i = tid; i < num_points; i += blockDim.x) {
        s_error[tid] += distances[offset + i];
    }
    __syncthreads();
    
    // Perform parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_error[tid] += s_error[tid + s];
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (tid == 0) {
        errors[batch_idx] = s_error[0] / num_points;
    }
}

// Kernel: Gather corresponding target points based on nearest neighbor indices
__global__ void gatherCorrespondingPoints(
    const float* target_points,    // Input: [total_points_tgt, 3] target points
    const int* correspondences,    // Input: [total_points_src] indices of nearest neighbors
    int* batch_sizes,              // Input: [batch_count] number of points per batch
    int* batch_offsets,            // Input: [batch_count] offset of each batch
    int batch_count,               // Input: number of batch items
    float* corresponding_points    // Output: [total_points_src, 3] gathered target points
) {
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int num_points = batch_sizes[batch_idx];
    int offset = batch_offsets[batch_idx];
    
    // Gather corresponding points
    for (int i = tid; i < num_points; i += blockDim.x) {
        int src_idx = offset + i;
        int tgt_idx = correspondences[src_idx];
        
        // Copy the corresponding target point
        corresponding_points[src_idx * 3]     = target_points[tgt_idx * 3];
        corresponding_points[src_idx * 3 + 1] = target_points[tgt_idx * 3 + 1];
        corresponding_points[src_idx * 3 + 2] = target_points[tgt_idx * 3 + 2];
    }
}

// Kernel: Compute Chamfer distance for each batch item
__global__ void batchedChamferDistance(
    float* src_points,           // Input: [total_points_src, 3] source points
    float* tgt_points,           // Input: [total_points_tgt, 3] target points
    int* src_batch_sizes,        // Input: [batch_count] number of source points per batch
    int* tgt_batch_sizes,        // Input: [batch_count] number of target points per batch
    int* src_offsets,            // Input: [batch_count] offset of each source batch item
    int* tgt_offsets,            // Input: [batch_count] offset of each target batch item
    int batch_count,             // Input: number of batch items
    float* chamfer_distances     // Output: [batch_count] chamfer distance for each batch item
) {
    extern __shared__ float shared_dist[];
    
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int src_size = src_batch_sizes[batch_idx];
    int tgt_size = tgt_batch_sizes[batch_idx];
    int src_offset = src_offsets[batch_idx];
    int tgt_offset = tgt_offsets[batch_idx];
    
    // Initialize shared memory for sum of distances
    if (tid == 0) {
        shared_dist[0] = 0.0f;  // src->tgt
        shared_dist[1] = 0.0f;  // tgt->src
    }
    __syncthreads();
    
    // Part 1: For each source point, find nearest target point
    for (int i = tid; i < src_size; i += blockDim.x) {
        int src_idx = (src_offset + i) * 3;
        float sx = src_points[src_idx];
        float sy = src_points[src_idx + 1];
        float sz = src_points[src_idx + 2];
        
        float min_dist = 1e10f;
        
        // Find nearest target point
        for (int j = 0; j < tgt_size; j++) {
            int tgt_idx = (tgt_offset + j) * 3;
            float tx = tgt_points[tgt_idx];
            float ty = tgt_points[tgt_idx + 1];
            float tz = tgt_points[tgt_idx + 2];
            
            float dx = sx - tx;
            float dy = sy - ty;
            float dz = sz - tz;
            float dist = dx*dx + dy*dy + dz*dz;
            
            min_dist = min(min_dist, dist);
        }
        
        // Accumulate distance
        atomicAdd(&shared_dist[0], min_dist);
    }
    
    // Part 2: For each target point, find nearest source point
    for (int i = tid; i < tgt_size; i += blockDim.x) {
        int tgt_idx = (tgt_offset + i) * 3;
        float tx = tgt_points[tgt_idx];
        float ty = tgt_points[tgt_idx + 1];
        float tz = tgt_points[tgt_idx + 2];
        
        float min_dist = 1e10f;
        
        // Find nearest source point
        for (int j = 0; j < src_size; j++) {
            int src_idx = (src_offset + j) * 3;
            float sx = src_points[src_idx];
            float sy = src_points[src_idx + 1];
            float sz = src_points[src_idx + 2];
            
            float dx = tx - sx;
            float dy = ty - sy;
            float dz = tz - sz;
            float dist = dx*dx + dy*dy + dz*dz;
            
            min_dist = min(min_dist, dist);
        }
        
        // Accumulate distance
        atomicAdd(&shared_dist[1], min_dist);
    }
    __syncthreads();
    
    // Compute final Chamfer distance and write to global memory
    if (tid == 0) {
        float src_to_tgt = shared_dist[0] / src_size;
        float tgt_to_src = shared_dist[1] / tgt_size;
        chamfer_distances[batch_idx] = (src_to_tgt + tgt_to_src) / 2.0f;
    }
}

/*
 * Host Functions for Batched Point Cloud Processing
 */

// Extract rotation matrix from covariance matrix using SVD
// Note: This is a placeholder. In a real implementation, use a CUDA library for SVD
void computeRotationFromCovariance(const float* covariance, float* rotation) {
    // Placeholder: In real implementation, perform SVD on covariance
    // and compute R = V * U^T where covariance = U * S * V^T
    
    // For now, simply copy the covariance as a rotation (obviously incorrect)
    for (int i = 0; i < 9; i++) {
        rotation[i] = covariance[i];
    }
    
    // Orthogonalize the rotation matrix (this is a quick hack, not a proper implementation)
    // In a real implementation, use a proper SVD library
}

// Main function for Procrustes alignment on batched point clouds
void batchedProcrustes(
    BatchedPointCloud* src,      // Input: source point clouds
    BatchedPointCloud* tgt,      // Input: target point clouds
    BatchedTransformation* transform,  // Output: computed transformations
    bool use_streams = true     // Use CUDA streams for concurrent processing
) {
    // Allocate temporary buffers
    float* d_src_centroids;
    float* d_tgt_centroids;
    float* d_covariance_matrices;
    
    CUDA_CHECK(cudaMalloc(&d_src_centroids, src->batch_count * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_centroids, tgt->batch_count * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covariance_matrices, src->batch_count * 9 * sizeof(float)));
    
    // Step 1: Compute centroids
    int threads = BLOCK_SIZE;
    size_t shared_mem_size = 3 * threads * sizeof(float);
    
    // If using CUDA streams for concurrent processing
    if (use_streams && src->batch_count > 1) {
        // Create one stream per batch item
        cudaStream_t* streams = (cudaStream_t*)malloc(src->batch_count * sizeof(cudaStream_t));
        
        for (int i = 0; i < src->batch_count; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
        
        // Process each batch item in its own stream
        for (int i = 0; i < src->batch_count; i++) {
            // Create local pointers for this batch
            float* batch_src_points = src->points + (src->batch_offsets[i] * 3);
            float* batch_tgt_points = tgt->points + (tgt->batch_offsets[i] * 3);
            float* batch_src_centroid = d_src_centroids + (i * 3);
            float* batch_tgt_centroid = d_tgt_centroids + (i * 3);
            
            // Compute centroids for this batch item
            batchedComputeCentroid<<<1, threads, shared_mem_size, streams[i]>>>(
                batch_src_points, &src->batch_sizes[i], &src->batch_offsets[i], 1, batch_src_centroid);
                
            batchedComputeCentroid<<<1, threads, shared_mem_size, streams[i]>>>(
                batch_tgt_points, &tgt->batch_sizes[i], &tgt->batch_offsets[i], 1, batch_tgt_centroid);
        }
        
        // Synchronize all streams
        for (int i = 0; i < src->batch_count; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
        
        // Step 2: Create centered copies of the point clouds
        float* d_src_centered;
        float* d_tgt_centered;
        
        CUDA_CHECK(cudaMalloc(&d_src_centered, src->total_points * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tgt_centered, tgt->total_points * 3 * sizeof(float)));
        
        // Copy points to centered buffers using streams
        for (int i = 0; i < src->batch_count; i++) {
            int src_offset = src->batch_offsets[i];
            int tgt_offset = tgt->batch_offsets[i];
            int src_size = src->batch_sizes[i];
            int tgt_size = tgt->batch_sizes[i];
            
            CUDA_CHECK(cudaMemcpyAsync(
                d_src_centered + (src_offset * 3), 
                src->points + (src_offset * 3), 
                src_size * 3 * sizeof(float), 
                cudaMemcpyDeviceToDevice, 
                streams[i]));
                
            CUDA_CHECK(cudaMemcpyAsync(
                d_tgt_centered + (tgt_offset * 3), 
                tgt->points + (tgt_offset * 3), 
                tgt_size * 3 * sizeof(float), 
                cudaMemcpyDeviceToDevice, 
                streams[i]));
                
            // Subtract centroids
            batchedSubtractCentroid<<<1, threads, 0, streams[i]>>>(
                d_src_centered + (src_offset * 3), 
                &src->batch_sizes[i], 
                &src->batch_offsets[i], 
                1, 
                d_src_centroids + (i * 3));
                
            batchedSubtractCentroid<<<1, threads, 0, streams[i]>>>(
                d_tgt_centered + (tgt_offset * 3), 
                &tgt->batch_sizes[i], 
                &tgt->batch_offsets[i], 
                1, 
                d_tgt_centroids + (i * 3));
        }
        
        // Synchronize all streams
        for (int i = 0; i < src->batch_count; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
        
        // Step 3: Compute covariance matrices
        shared_mem_size = 9 * sizeof(float);  // 3x3 covariance matrix in shared memory
        
        for (int i = 0; i < src->batch_count; i++) {
            batchedComputeCovariance<<<1, threads, shared_mem_size, streams[i]>>>(
                d_src_centered + (src->batch_offsets[i] * 3), 
                d_tgt_centered + (tgt->batch_offsets[i] * 3), 
                &src->batch_sizes[i], 
                &src->batch_offsets[i], 
                &tgt->batch_offsets[i],
                1, 
                d_covariance_matrices + (i * 9));
        }
        
        // Synchronize all streams
        for (int i = 0; i < src->batch_count; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
        
        // Step 4: Compute rotations from covariance matrices
        // For simplicity, we'll do this on the CPU
        float* h_covariance_matrices = (float*)malloc(src->batch_count * 9 * sizeof(float));
        float* h_rotations = (float*)malloc(src->batch_count * 9 * sizeof(float));
        
        CUDA_CHECK(cudaMemcpy(h_covariance_matrices, d_covariance_matrices, 
                           src->batch_count * 9 * sizeof(float), cudaMemcpyDeviceToHost));
        
        #pragma omp parallel for
        for (int i = 0; i < src->batch_count; i++) {
            computeRotationFromCovariance(&h_covariance_matrices[i * 9], &h_rotations[i * 9]);
        }
        
        // Copy rotations back to device using streams
        for (int i = 0; i < src->batch_count; i++) {
            CUDA_CHECK(cudaMemcpyAsync(
                transform->rotations + (i * 9), 
                h_rotations + (i * 9), 
                9 * sizeof(float), 
                cudaMemcpyHostToDevice, 
                streams[i]));
        }
        
        // Step 5: Compute translations
        // t = tgt_centroid - R * src_centroid
        float* h_src_centroids = (float*)malloc(src->batch_count * 3 * sizeof(float));
        float* h_tgt_centroids = (float*)malloc(src->batch_count * 3 * sizeof(float));
        float* h_translations = (float*)malloc(src->batch_count * 3 * sizeof(float));
        
        CUDA_CHECK(cudaMemcpy(h_src_centroids, d_src_centroids, 
                           src->batch_count * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tgt_centroids, d_tgt_centroids, 
                           src->batch_count * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        
        #pragma omp parallel for
        for (int i = 0; i < src->batch_count; i++) {
            float* R = &h_rotations[i * 9];
            float* s_cent = &h_src_centroids[i * 3];
            float* t_cent = &h_tgt_centroids[i * 3];
            float* t = &h_translations[i * 3];
            
            // Compute R * src_centroid
            float rotated_cent[3] = {
                R[0] * s_cent[0] + R[1] * s_cent[1] + R[2] * s_cent[2],
                R[3] * s_cent[0] + R[4] * s_cent[1] + R[5] * s_cent[2],
                R[6] * s_cent[0] + R[7] * s_cent[1] + R[8] * s_cent[2]
            };
            
            // Compute translation
            t[0] = t_cent[0] - rotated_cent[0];
            t[1] = t_cent[1] - rotated_cent[1];
            t[2] = t_cent[2] - rotated_cent[2];
        }
        
        // Copy translations back to device using streams
        for (int i = 0; i < src->batch_count; i++) {
            CUDA_CHECK(cudaMemcpyAsync(
                transform->translations + (i * 3), 
                h_translations + (i * 3), 
                3 * sizeof(float), 
                cudaMemcpyHostToDevice, 
                streams[i]));
        }
        
        // Synchronize all streams
        for (int i = 0; i < src->batch_count; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
        
        // Destroy streams
        for (int i = 0; i < src->batch_count; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        free(streams);
        
        // Clean up temporary buffers
        CUDA_CHECK(cudaFree(d_src_centroids));
        CUDA_CHECK(cudaFree(d_tgt_centroids));
        CUDA_CHECK(cudaFree(d_covariance_matrices));
        CUDA_CHECK(cudaFree(d_src_centered));
        CUDA_CHECK(cudaFree(d_tgt_centered));
        
        free(h_covariance_matrices);
        free(h_rotations);
        free(h_src_centroids);
        free(h_tgt_centroids);
        free(h_translations);
    } 
    else {
        // Original implementation without streams
        batchedComputeCentroid<<<src->batch_count, threads, shared_mem_size>>>(
            src->points, src->batch_sizes, src->batch_offsets, src->batch_count, d_src_centroids);
            
        batchedComputeCentroid<<<tgt->batch_count, threads, shared_mem_size>>>(
            tgt->points, tgt->batch_sizes, tgt->batch_offsets, tgt->batch_count, d_tgt_centroids);
        
        // Step 2: Create centered copies of the point clouds
        float* d_src_centered;
        float* d_tgt_centered;
        
        CUDA_CHECK(cudaMalloc(&d_src_centered, src->total_points * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tgt_centered, tgt->total_points * 3 * sizeof(float)));
        
        // Copy points to centered buffers
        CUDA_CHECK(cudaMemcpy(d_src_centered, src->points, src->total_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_tgt_centered, tgt->points, tgt->total_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Subtract centroids
        batchedSubtractCentroid<<<src->batch_count, threads>>>(
            d_src_centered, src->batch_sizes, src->batch_offsets, src->batch_count, d_src_centroids);
            
        batchedSubtractCentroid<<<tgt->batch_count, threads>>>(
            d_tgt_centered, tgt->batch_sizes, tgt->batch_offsets, tgt->batch_count, d_tgt_centroids);
        
        // Step 3: Compute covariance matrices
        shared_mem_size = 9 * sizeof(float);  // 3x3 covariance matrix in shared memory
        
        batchedComputeCovariance<<<src->batch_count, threads, shared_mem_size>>>(
            d_src_centered, d_tgt_centered, src->batch_sizes, src->batch_offsets, tgt->batch_offsets,
            src->batch_count, d_covariance_matrices);
        
        // Step 4: Compute rotations from covariance matrices
        // For simplicity, we'll do this on the CPU
        float* h_covariance_matrices = (float*)malloc(src->batch_count * 9 * sizeof(float));
        float* h_rotations = (float*)malloc(src->batch_count * 9 * sizeof(float));
        
        CUDA_CHECK(cudaMemcpy(h_covariance_matrices, d_covariance_matrices, 
                            src->batch_count * 9 * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < src->batch_count; i++) {
            computeRotationFromCovariance(&h_covariance_matrices[i * 9], &h_rotations[i * 9]);
        }
        
        CUDA_CHECK(cudaMemcpy(transform->rotations, h_rotations, 
                            src->batch_count * 9 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Step 5: Compute translations
        // t = tgt_centroid - R * src_centroid
        float* h_src_centroids = (float*)malloc(src->batch_count * 3 * sizeof(float));
        float* h_tgt_centroids = (float*)malloc(src->batch_count * 3 * sizeof(float));
        float* h_translations = (float*)malloc(src->batch_count * 3 * sizeof(float));
        
        CUDA_CHECK(cudaMemcpy(h_src_centroids, d_src_centroids, 
                            src->batch_count * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tgt_centroids, d_tgt_centroids, 
                            src->batch_count * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < src->batch_count; i++) {
            float* R = &h_rotations[i * 9];
            float* s_cent = &h_src_centroids[i * 3];
            float* t_cent = &h_tgt_centroids[i * 3];
            float* t = &h_translations[i * 3];
            
            // Compute R * src_centroid
            float rotated_cent[3] = {
                R[0] * s_cent[0] + R[1] * s_cent[1] + R[2] * s_cent[2],
                R[3] * s_cent[0] + R[4] * s_cent[1] + R[5] * s_cent[2],
                R[6] * s_cent[0] + R[7] * s_cent[1] + R[8] * s_cent[2]
            };
            
            // Compute translation
            t[0] = t_cent[0] - rotated_cent[0];
            t[1] = t_cent[1] - rotated_cent[1];
            t[2] = t_cent[2] - rotated_cent[2];
        }
        
        CUDA_CHECK(cudaMemcpy(transform->translations, h_translations, 
                            src->batch_count * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Clean up temporary buffers
        CUDA_CHECK(cudaFree(d_src_centroids));
        CUDA_CHECK(cudaFree(d_tgt_centroids));
        CUDA_CHECK(cudaFree(d_covariance_matrices));
        CUDA_CHECK(cudaFree(d_src_centered));
        CUDA_CHECK(cudaFree(d_tgt_centered));
        
        free(h_covariance_matrices);
        free(h_rotations);
        free(h_src_centroids);
        free(h_tgt_centroids);
        free(h_translations);
    }
}

// Main function for batched ICP alignment
void batchedICP(
    BatchedPointCloud* src,      // Input: source point clouds to align
    BatchedPointCloud* tgt,      // Input: target point clouds (reference)
    int max_iterations,          // Input: maximum ICP iterations
    float convergence_threshold, // Input: convergence threshold for early stopping
    BatchedTransformation* transform,  // Output: computed transformations
    bool use_streams = true      // Use CUDA streams for concurrent processing
) {
    // Create a copy of source points that we'll transform
    BatchedPointCloud src_transformed;
    src_transformed.batch_count = src->batch_count;
    src_transformed.total_points = src->total_points;
    
    CUDA_CHECK(cudaMalloc(&src_transformed.points, src->total_points * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&src_transformed.batch_sizes, src->batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&src_transformed.batch_offsets, src->batch_count * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(src_transformed.points, src->points, 
                         src->total_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(src_transformed.batch_sizes, src->batch_sizes, 
                         src->batch_count * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(src_transformed.batch_offsets, src->batch_offsets, 
                         src->batch_count * sizeof(int), cudaMemcpyDeviceToDevice));
    
    // Allocate memory for correspondences and distances
    int* d_correspondences;
    float* d_distances;
    
    CUDA_CHECK(cudaMalloc(&d_correspondences, src->total_points * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distances, src->total_points * sizeof(float)));
    
    // Allocate memory for current transformation
    BatchedTransformation current_transform;
    allocateBatchedTransformation(&current_transform, src->batch_count);
    
    // Set initial rotation to identity and translation to zero
    float* h_identity = (float*)malloc(src->batch_count * 9 * sizeof(float));
    float* h_zero = (float*)malloc(src->batch_count * 3 * sizeof(float));
    
    for (int i = 0; i < src->batch_count; i++) {
        for (int j = 0; j < 9; j++) {
            h_identity[i * 9 + j] = (j % 4 == 0) ? 1.0f : 0.0f;  // Identity matrix (diagonal=1, rest=0)
        }
        
        for (int j = 0; j < 3; j++) {
            h_zero[i * 3 + j] = 0.0f;  // Zero translation
        }
    }
    
    CUDA_CHECK(cudaMemcpy(current_transform.rotations, h_identity, 
                         src->batch_count * 9 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(current_transform.translations, h_zero, 
                         src->batch_count * 3 * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_identity);
    free(h_zero);
    
    // Allocate memory for previous errors to check convergence
    float* d_current_errors;
    float* h_prev_errors = (float*)malloc(src->batch_count * sizeof(float));
    float* h_current_errors = (float*)malloc(src->batch_count * sizeof(float));
    
    CUDA_CHECK(cudaMalloc(&d_current_errors, src->batch_count * sizeof(float)));
    
    // Initialize previous errors to a large value
    for (int i = 0; i < src->batch_count; i++) {
        h_prev_errors[i] = 1e10f;  // A large value to ensure first iteration runs
    }
    
    // Prepare target centroids (will be needed in each iteration)
    float* d_tgt_centroids;
    CUDA_CHECK(cudaMalloc(&d_tgt_centroids, tgt->batch_count * 3 * sizeof(float)));
    
    // Compute target centroids
    int threads = BLOCK_SIZE;
    size_t shared_mem_size = 3 * threads * sizeof(float);
    
    batchedComputeCentroid<<<tgt->batch_count, threads, shared_mem_size>>>(
        tgt->points, tgt->batch_sizes, tgt->batch_offsets, tgt->batch_count, d_tgt_centroids);
    
    // Build uniform grid for target point cloud to accelerate nearest neighbor search
    UniformGrid grid;
    allocateUniformGrid(&grid, tgt->total_points, tgt->batch_count);
    
    // Build the grid
    buildUniformGrid(&grid, tgt->points, tgt->batch_sizes, tgt->batch_offsets, 
                    tgt->batch_count, tgt->total_points);
    
    // Iterative Closest Point algorithm
    bool converged = false;
    for (int iter = 0; iter < max_iterations && !converged; iter++) {
        // Step 1: Find nearest neighbors (correspondences) using the grid
        // Use max src size for each batch to determine grid dimensions
        int max_src_size = 0;
        for (int i = 0; i < src->batch_count; i++) {
            int* h_sizes = (int*)malloc(src->batch_count * sizeof(int));
            CUDA_CHECK(cudaMemcpy(h_sizes, src->batch_sizes, src->batch_count * sizeof(int), cudaMemcpyDeviceToHost));
            if (h_sizes[i] > max_src_size) {
                max_src_size = h_sizes[i];
            }
            free(h_sizes);
        }
        
        // Launch the grid-based nearest neighbor kernel
        dim3 grid_dim(
            (max_src_size + BLOCK_SIZE - 1) / BLOCK_SIZE,  // x dimension: blocks for points
            src->batch_count                              // y dimension: blocks for batches
        );
        
        batchedGridNN<<<grid_dim, BLOCK_SIZE>>>(
            src_transformed.points, tgt->points, 
            src->batch_sizes, tgt->batch_sizes,
            src->batch_offsets, tgt->batch_offsets,
            src->batch_count, 
            grid.grid_min, grid.cell_size, 
            grid.grid_counts, grid.grid_indices, grid.point_indices,
            d_correspondences, d_distances,
            grid.resolution, GRID_CELL_COUNT);
        
        // Step 2: Compute current error for convergence check
        CUDA_CHECK(cudaMemset(d_current_errors, 0, src->batch_count * sizeof(float)));
        
        // Launch one block per batch to sum errors
        shared_mem_size = BLOCK_SIZE * sizeof(float);  // For shared memory reduction
        computeAverageError<<<src->batch_count, BLOCK_SIZE, shared_mem_size>>>(
            d_distances, src->batch_sizes, src->batch_offsets, src->batch_count, d_current_errors);
        
        // Copy errors to host to check for convergence
        CUDA_CHECK(cudaMemcpy(h_current_errors, d_current_errors, 
                           src->batch_count * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Check convergence for each batch item
        converged = true;
        for (int b = 0; b < src->batch_count; b++) {
            float error_diff = fabs(h_prev_errors[b] - h_current_errors[b]);
            if (error_diff > convergence_threshold) {
                converged = false;  // At least one batch item hasn't converged
            }
            h_prev_errors[b] = h_current_errors[b];  // Update previous error
        }
        
        // If all batches have converged, we're done
        if (converged) break;
        
        // Step 3: Compute source centroids for current iteration
        float* d_src_centroids;
        CUDA_CHECK(cudaMalloc(&d_src_centroids, src->batch_count * 3 * sizeof(float)));
        
        batchedComputeCentroid<<<src->batch_count, threads, shared_mem_size>>>(
            src_transformed.points, src->batch_sizes, src->batch_offsets, 
            src->batch_count, d_src_centroids);
        
        // Step 4: Create point clouds of corresponding points
        // We need to copy target points based on correspondences for covariance computation
        float* d_corresponding_tgt;
        CUDA_CHECK(cudaMalloc(&d_corresponding_tgt, src->total_points * 3 * sizeof(float)));
        
        // Gather corresponding target points
        gatherCorrespondingPoints<<<src->batch_count, BLOCK_SIZE>>>(
            tgt->points, d_correspondences, src->batch_sizes, src->batch_offsets,
            src->batch_count, d_corresponding_tgt);
        
        // Step 5: Center both point clouds by subtracting centroids
        batchedSubtractCentroid<<<src->batch_count, BLOCK_SIZE>>>(
            src_transformed.points, src->batch_sizes, src->batch_offsets,
            src->batch_count, d_src_centroids);
            
        batchedSubtractCentroid<<<src->batch_count, BLOCK_SIZE>>>(
            d_corresponding_tgt, src->batch_sizes, src->batch_offsets,
            src->batch_count, d_tgt_centroids);
        
        // Step 6: Compute covariance matrices
        float* d_covariance_matrices;
        CUDA_CHECK(cudaMalloc(&d_covariance_matrices, src->batch_count * 9 * sizeof(float)));
        
        shared_mem_size = 9 * sizeof(float);  // 3x3 covariance matrix in shared memory
        
        batchedComputeCovariance<<<src->batch_count, BLOCK_SIZE, shared_mem_size>>>(
            src_transformed.points, d_corresponding_tgt, src->batch_sizes, 
            src->batch_offsets, src->batch_offsets,
            src->batch_count, d_covariance_matrices);
        
        // Step 7: Compute rotations from covariance matrices
        // For simplicity, we'll do this on the CPU
        float* h_covariance_matrices = (float*)malloc(src->batch_count * 9 * sizeof(float));
        float* h_rotations = (float*)malloc(src->batch_count * 9 * sizeof(float));
        
        CUDA_CHECK(cudaMemcpy(h_covariance_matrices, d_covariance_matrices, 
                            src->batch_count * 9 * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < src->batch_count; i++) {
            computeRotationFromCovariance(&h_covariance_matrices[i * 9], &h_rotations[i * 9]);
        }
        
        CUDA_CHECK(cudaMemcpy(current_transform.rotations, h_rotations, 
                            src->batch_count * 9 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Step 8: Compute translations
        float* h_src_centroids = (float*)malloc(src->batch_count * 3 * sizeof(float));
        float* h_tgt_centroids = (float*)malloc(src->batch_count * 3 * sizeof(float));
        float* h_translations = (float*)malloc(src->batch_count * 3 * sizeof(float));
        
        CUDA_CHECK(cudaMemcpy(h_src_centroids, d_src_centroids, 
                            src->batch_count * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tgt_centroids, d_tgt_centroids, 
                            src->batch_count * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < src->batch_count; i++) {
            float* R = &h_rotations[i * 9];
            float* s_cent = &h_src_centroids[i * 3];
            float* t_cent = &h_tgt_centroids[i * 3];
            float* t = &h_translations[i * 3];
            
            // Compute R * src_centroid
            float rotated_cent[3] = {
                R[0] * s_cent[0] + R[1] * s_cent[1] + R[2] * s_cent[2],
                R[3] * s_cent[0] + R[4] * s_cent[1] + R[5] * s_cent[2],
                R[6] * s_cent[0] + R[7] * s_cent[1] + R[8] * s_cent[2]
            };
            
            // Compute translation
            t[0] = t_cent[0] - rotated_cent[0];
            t[1] = t_cent[1] - rotated_cent[1];
            t[2] = t_cent[2] - rotated_cent[2];
        }
        
        CUDA_CHECK(cudaMemcpy(current_transform.translations, h_translations, 
                            src->batch_count * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Step 9: Apply the transformation to source points for next iteration
        // Reset the points to original positions
        CUDA_CHECK(cudaMemcpy(src_transformed.points, src->points, 
                             src->total_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Apply current transformation
        batchedApplyTransform<<<src->batch_count, BLOCK_SIZE>>>(
            src_transformed.points, src->batch_sizes, src->batch_offsets,
            src->batch_count, current_transform.rotations, current_transform.translations);
        
        // Free temporary memory for this iteration
        free(h_covariance_matrices);
        free(h_rotations);
        free(h_src_centroids);
        free(h_tgt_centroids);
        free(h_translations);
        
        CUDA_CHECK(cudaFree(d_src_centroids));
        CUDA_CHECK(cudaFree(d_corresponding_tgt));
        CUDA_CHECK(cudaFree(d_covariance_matrices));
    }
    
    // Copy final transformation
    CUDA_CHECK(cudaMemcpy(transform->rotations, current_transform.rotations,
                        src->batch_count * 9 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(transform->translations, current_transform.translations,
                        src->batch_count * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(transform->errors, d_current_errors,
                        src->batch_count * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Clean up
    CUDA_CHECK(cudaFree(src_transformed.points));
    CUDA_CHECK(cudaFree(src_transformed.batch_sizes));
    CUDA_CHECK(cudaFree(src_transformed.batch_offsets));
    CUDA_CHECK(cudaFree(d_correspondences));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_current_errors));
    CUDA_CHECK(cudaFree(d_tgt_centroids));
    freeBatchedTransformation(&current_transform);
    freeUniformGrid(&grid);
    
    free(h_prev_errors);
    free(h_current_errors);
}

// Main function for computing Chamfer distance
void batchedChamferDistanceCompute(
    BatchedPointCloud* src,      // Input: source point clouds
    BatchedPointCloud* tgt,      // Input: target point clouds
    float* chamfer_distances     // Output: [batch_count] chamfer distance for each batch item
) {
    // Allocate device memory for chamfer distances
    float* d_chamfer_distances;
    CUDA_CHECK(cudaMalloc(&d_chamfer_distances, src->batch_count * sizeof(float)));
    
    // Compute Chamfer distance
    int threads = BLOCK_SIZE;
    size_t shared_mem_size = 2 * sizeof(float);  // For src->tgt and tgt->src distances
    
    batchedChamferDistance<<<src->batch_count, threads, shared_mem_size>>>(
        src->points, tgt->points,
        src->batch_sizes, tgt->batch_sizes,
        src->batch_offsets, tgt->batch_offsets,
        src->batch_count, d_chamfer_distances);
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(chamfer_distances, d_chamfer_distances, 
                        src->batch_count * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Clean up
    CUDA_CHECK(cudaFree(d_chamfer_distances));
}

/*
 * Test Functions
 */

// Generate a simple test batch with varying sizes
void generateTestBatch(BatchedPointCloud* cloud, int batch_count, int min_size, int max_size) {
    // Generate random batch sizes
    int* sizes = (int*)malloc(batch_count * sizeof(int));
    int total_points = 0;
    
    srand(time(NULL));
    
    for (int i = 0; i < batch_count; i++) {
        sizes[i] = min_size + rand() % (max_size - min_size + 1);
        total_points += sizes[i];
    }
    
    // Allocate the batched point cloud
    allocateBatchedPointCloud(cloud, batch_count, sizes);
    
    // Generate random points
    float* h_points = (float*)malloc(total_points * 3 * sizeof(float));
    
    for (int i = 0; i < total_points * 3; i++) {
        h_points[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Random value in [-1, 1]
    }
    
    // Copy points to device
    CUDA_CHECK(cudaMemcpy(cloud->points, h_points, 
                        total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Clean up
    free(sizes);
    free(h_points);
}

// Simple test for batched Procrustes alignment
void testBatchedProcrustes() {
    printf("Testing batched Procrustes alignment...\n");
    
    // Create test batch of point clouds
    BatchedPointCloud src, tgt;
    int batch_count = 4;
    int min_size = 50;
    int max_size = 100;
    
    generateTestBatch(&src, batch_count, min_size, max_size);
    
    // Create target batch by applying known transformations
    // TODO: Implement this
    
    // Allocate transformation
    BatchedTransformation transform;
    allocateBatchedTransformation(&transform, batch_count);
    
    // Run batched Procrustes
    batchedProcrustes(&src, &tgt, &transform);
    
    // Verify results
    // TODO: Implement this
    
    // Clean up
    freeBatchedPointCloud(&src);
    freeBatchedPointCloud(&tgt);
    freeBatchedTransformation(&transform);
    
    printf("Batched Procrustes test completed.\n");
}

/*
 * Benchmarking Functions
 */

// Benchmarking structure to hold timing results
typedef struct {
    float total_time;         // Total execution time in milliseconds
    float centroid_time;      // Time spent computing centroids
    float nn_search_time;     // Time spent in nearest neighbor search
    float svd_time;           // Time spent in SVD computation
    float transform_time;     // Time spent applying transformations
    int iterations;           // Number of iterations (for ICP)
    float final_error;        // Final error/distance
} BenchmarkResult;

// Function to benchmark the performance of Procrustes with different graph sizes
void benchmarkProcrustes(int batch_sizes[], int num_sizes, int batch_count, bool use_streams) {
    printf("\n=== Benchmarking Procrustes Alignment ===\n");
    printf("Batch count: %d, Using streams: %s\n", batch_count, use_streams ? "Yes" : "No");
    printf("Size\tTotal(ms)\tCentroid(ms)\tSVD(ms)\tTransform(ms)\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int point_count = batch_sizes[s];
        
        // Create batched point clouds
        BatchedPointCloud src, tgt;
        BatchedTransformation transform;
        
        // Allocate and initialize with random data
        int* sizes = (int*)malloc(batch_count * sizeof(int));
        for (int i = 0; i < batch_count; i++) {
            sizes[i] = point_count;
        }
        
        allocateBatchedPointCloud(&src, batch_count, sizes);
        allocateBatchedPointCloud(&tgt, batch_count, sizes);
        allocateBatchedTransformation(&transform, batch_count);
        
        // Generate random point clouds
        float* h_src_points = (float*)malloc(src.total_points * 3 * sizeof(float));
        float* h_tgt_points = (float*)malloc(tgt.total_points * 3 * sizeof(float));
        
        // Generate random source points
        for (int i = 0; i < src.total_points * 3; i++) {
            h_src_points[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        
        // Copy source to target with a known transformation
        for (int b = 0; b < batch_count; b++) {
            int offset = src.batch_offsets[b];
            int size = src.batch_sizes[b];
            
            // Apply a random transformation for each batch
            float angle = ((float)rand() / RAND_MAX) * 3.14159f;
            float tx = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float ty = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float tz = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            
            float R[9] = {
                cosf(angle), -sinf(angle), 0,
                sinf(angle), cosf(angle), 0,
                0, 0, 1
            };
            
            for (int i = 0; i < size; i++) {
                int idx = (offset + i) * 3;
                float x = h_src_points[idx];
                float y = h_src_points[idx + 1];
                float z = h_src_points[idx + 2];
                
                // Apply rotation
                float rx = R[0]*x + R[1]*y + R[2]*z;
                float ry = R[3]*x + R[4]*y + R[5]*z;
                float rz = R[6]*x + R[7]*y + R[8]*z;
                
                // Apply translation
                h_tgt_points[idx] = rx + tx;
                h_tgt_points[idx + 1] = ry + ty;
                h_tgt_points[idx + 2] = rz + tz;
            }
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(src.points, h_src_points, src.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tgt.points, h_tgt_points, tgt.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Create CUDA events for timing
        cudaEvent_t start, stop, centroid_start, centroid_stop, svd_start, svd_stop, transform_start, transform_stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventCreate(&centroid_start));
        CUDA_CHECK(cudaEventCreate(&centroid_stop));
        CUDA_CHECK(cudaEventCreate(&svd_start));
        CUDA_CHECK(cudaEventCreate(&svd_stop));
        CUDA_CHECK(cudaEventCreate(&transform_start));
        CUDA_CHECK(cudaEventCreate(&transform_stop));
        
        // Measure total time
        CUDA_CHECK(cudaEventRecord(start));
        
        // Run Procrustes alignment
        batchedProcrustes(&src, &tgt, &transform, use_streams);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        // Calculate timing
        float total_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));
        
        // Print results
        printf("%d\t%.2f\t\t-\t\t-\t\t-\n", point_count, total_time);
        
        // Clean up
        free(sizes);
        free(h_src_points);
        free(h_tgt_points);
        freeBatchedPointCloud(&src);
        freeBatchedPointCloud(&tgt);
        freeBatchedTransformation(&transform);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaEventDestroy(centroid_start));
        CUDA_CHECK(cudaEventDestroy(centroid_stop));
        CUDA_CHECK(cudaEventDestroy(svd_start));
        CUDA_CHECK(cudaEventDestroy(svd_stop));
        CUDA_CHECK(cudaEventDestroy(transform_start));
        CUDA_CHECK(cudaEventDestroy(transform_stop));
    }
}

// Function to benchmark the performance of ICP with different graph sizes
void benchmarkICP(int batch_sizes[], int num_sizes, int batch_count, int max_iterations, float convergence_threshold, bool use_streams) {
    printf("\n=== Benchmarking ICP Alignment ===\n");
    printf("Batch count: %d, Max iterations: %d, Using streams: %s\n", batch_count, max_iterations, use_streams ? "Yes" : "No");
    printf("Size\tTotal(ms)\tIters\tNN(ms)\tError\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int point_count = batch_sizes[s];
        
        // Create batched point clouds
        BatchedPointCloud src, tgt;
        BatchedTransformation transform;
        
        // Allocate and initialize with random data
        int* sizes = (int*)malloc(batch_count * sizeof(int));
        for (int i = 0; i < batch_count; i++) {
            sizes[i] = point_count;
        }
        
        allocateBatchedPointCloud(&src, batch_count, sizes);
        allocateBatchedPointCloud(&tgt, batch_count, sizes);
        allocateBatchedTransformation(&transform, batch_count);
        
        // Generate random point clouds
        float* h_src_points = (float*)malloc(src.total_points * 3 * sizeof(float));
        float* h_tgt_points = (float*)malloc(tgt.total_points * 3 * sizeof(float));
        
        // Generate random source points
        for (int i = 0; i < src.total_points * 3; i++) {
            h_src_points[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        
        // Copy source to target with a known transformation
        for (int b = 0; b < batch_count; b++) {
            int offset = src.batch_offsets[b];
            int size = src.batch_sizes[b];
            
            // Apply a random transformation for each batch
            float angle = ((float)rand() / RAND_MAX) * 3.14159f;
            float tx = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float ty = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float tz = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            
            float R[9] = {
                cosf(angle), -sinf(angle), 0,
                sinf(angle), cosf(angle), 0,
                0, 0, 1
            };
            
            for (int i = 0; i < size; i++) {
                int idx = (offset + i) * 3;
                float x = h_src_points[idx];
                float y = h_src_points[idx + 1];
                float z = h_src_points[idx + 2];
                
                // Apply rotation
                float rx = R[0]*x + R[1]*y + R[2]*z;
                float ry = R[3]*x + R[4]*y + R[5]*z;
                float rz = R[6]*x + R[7]*y + R[8]*z;
                
                // Apply translation
                h_tgt_points[idx] = rx + tx;
                h_tgt_points[idx + 1] = ry + ty;
                h_tgt_points[idx + 2] = rz + tz;
                
                // Add some noise to target points for a more realistic test
                float noise = 0.02f;  // 2% noise
                h_tgt_points[idx] += ((float)rand() / RAND_MAX) * noise - noise/2;
                h_tgt_points[idx + 1] += ((float)rand() / RAND_MAX) * noise - noise/2;
                h_tgt_points[idx + 2] += ((float)rand() / RAND_MAX) * noise - noise/2;
            }
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(src.points, h_src_points, src.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tgt.points, h_tgt_points, tgt.total_points * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Measure total time
        CUDA_CHECK(cudaEventRecord(start));
        
        // Run ICP alignment
        batchedICP(&src, &tgt, max_iterations, convergence_threshold, &transform, use_streams);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        // Calculate timing
        float total_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));
        
        // Get errors from device
        float* h_errors = (float*)malloc(batch_count * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_errors, transform.errors, batch_count * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Calculate average error across batches
        float avg_error = 0;
        for (int i = 0; i < batch_count; i++) {
            avg_error += h_errors[i];
        }
        avg_error /= batch_count;
        
        // Print results
        printf("%d\t%.2f\t\t-\t-\t%.6f\n", point_count, total_time, avg_error);
        
        // Clean up
        free(sizes);
        free(h_src_points);
        free(h_tgt_points);
        free(h_errors);
        freeBatchedPointCloud(&src);
        freeBatchedPointCloud(&tgt);
        freeBatchedTransformation(&transform);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

/*
 * Main Function
 */

int main() {
    // Test cases
    //testBatchedProcrustes();
    
    // Benchmark parameters
    int sizes[] = {100, 500, 1000, 5000, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // Benchmark Procrustes alignment
    printf("\n=== Benchmarking with streams disabled ===\n");
    benchmarkProcrustes(sizes, num_sizes, 4, false);
    
    printf("\n=== Benchmarking with streams enabled ===\n");
    benchmarkProcrustes(sizes, num_sizes, 4, true);
    
    // Benchmark ICP alignment
    printf("\n=== Benchmarking ICP with streams disabled ===\n");
    benchmarkICP(sizes, num_sizes, 4, 20, 1e-6, false);
    
    printf("\n=== Benchmarking ICP with streams enabled ===\n");
    benchmarkICP(sizes, num_sizes, 4, 20, 1e-6, true);
    
    return 0;
}