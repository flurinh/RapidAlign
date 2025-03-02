#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

// Define constants
#define BLOCK_SIZE 256
#define MAX_POINTS_PER_BATCH_ITEM 1024  // Can be adjusted as needed
#define MAX_BATCH_SIZE 32               // Maximum number of point clouds in a batch

// Define kernel functions needed for the operations

/***************************
 * Utility CUDA Kernels
 ***************************/

// Convert batched torch tensor to BatchedPointCloud format
__global__ void convert_tensor_to_batched_format(
    const float* points,         // [total_points, 3] tensor data
    const int64_t* batch_idx,    // [total_points] batch indices
    float* packed_points,        // Output: packed points in consecutive memory
    int* offsets,                // Output: offsets for each batch item
    int* sizes,                  // Output: number of points in each batch item
    int batch_count,             // Number of batch items
    int total_points             // Total number of points
) {
    extern __shared__ int s_counts[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory counts
    for (int i = tid; i < batch_count; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();
    
    // Count points per batch
    if (gid < total_points) {
        int b_idx = batch_idx[gid];
        if (b_idx >= 0 && b_idx < batch_count) {
            atomicAdd(&s_counts[b_idx], 1);
        }
    }
    __syncthreads();
    
    // Copy counts to global memory
    for (int i = tid; i < batch_count; i += blockDim.x) {
        sizes[i] = s_counts[i];
    }
    __syncthreads();
    
    // Compute offsets (exclusive scan)
    if (tid == 0) {
        int running_sum = 0;
        for (int i = 0; i < batch_count; i++) {
            offsets[i] = running_sum;
            running_sum += s_counts[i];
        }
    }
    __syncthreads();
    
    // Copy points to packed buffer
    if (gid < total_points) {
        int b_idx = batch_idx[gid];
        if (b_idx >= 0 && b_idx < batch_count) {
            // Atomically increment the write position for this batch
            int write_pos = atomicAdd(&s_counts[b_idx], 0);
            write_pos += offsets[b_idx];
            
            // Copy the point data
            packed_points[write_pos * 3]     = points[gid * 3];
            packed_points[write_pos * 3 + 1] = points[gid * 3 + 1];
            packed_points[write_pos * 3 + 2] = points[gid * 3 + 2];
            
            // Atomically increment our position counter
            atomicAdd(&s_counts[b_idx], 1);
        }
    }
}

// Convert batched format back to torch tensor with batch indices
__global__ void convert_batched_to_tensor_format(
    const float* packed_points,   // Input: packed points
    const int* offsets,           // Input: batch offsets
    const int* sizes,             // Input: batch sizes
    const int64_t* batch_indices, // Input: original batch indices
    float* output_points,         // Output: points in original ordering
    int total_points,             // Total points in all batches
    int batch_count               // Number of batch items
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    if (gid < total_points) {
        int b_idx = -1;
        for (int i = 0; i < batch_count; i++) {
            if (batch_indices[gid] == i) {
                b_idx = i;
                break;
            }
        }
        
        if (b_idx >= 0) {
            // Find position in the packed array
            int local_offset = 0;
            for (int i = 0; i < gid; i++) {
                if (batch_indices[i] == b_idx) {
                    local_offset++;
                }
            }
            
            int packed_idx = offsets[b_idx] + local_offset;
            output_points[gid * 3]     = packed_points[packed_idx * 3];
            output_points[gid * 3 + 1] = packed_points[packed_idx * 3 + 1];
            output_points[gid * 3 + 2] = packed_points[packed_idx * 3 + 2];
        }
    }
}

/***************************
 * SVD Computation Kernel
 ***************************/

// Compute 3x3 singular value decomposition (SVD) using Jacobi iterations
// This is a simplified implementation - in production code, use cuSOLVER
__device__ void compute_svd_3x3(
    const float* A,     // Input: 3x3 matrix in row-major order
    float* U,           // Output: 3x3 left singular vectors
    float* S,           // Output: 3 singular values
    float* V            // Output: 3x3 right singular vectors
) {
    // Initialize U and V to identity matrices
    for (int i = 0; i < 9; i++) {
        U[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        V[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }
    
    // Copy A to a local matrix B
    float B[9];
    for (int i = 0; i < 9; i++) {
        B[i] = A[i];
    }
    
    // Jacobi iterations
    const int max_iters = 30;
    const float threshold = 1e-6f;
    
    for (int iter = 0; iter < max_iters; iter++) {
        // Find the largest off-diagonal element
        float max_val = 0.0f;
        int p = 0, q = 1;
        
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                float val = fabsf(B[i*3 + j]);
                if (val > max_val) {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        
        // Check for convergence
        if (max_val < threshold) {
            break;
        }
        
        // Compute Jacobi rotation
        float alpha = (B[p*3 + p] - B[q*3 + q]) / 2.0f;
        float beta = -B[p*3 + q];
        float gamma = fabsf(alpha) / sqrtf(alpha*alpha + beta*beta);
        float s = sqrtf((1.0f - gamma) / 2.0f);
        float c = sqrtf((1.0f + gamma) / 2.0f);
        if (alpha * beta < 0) {
            s = -s;
        }
        
        // Apply Jacobi rotation to B
        float B_pp = B[p*3 + p];
        float B_pq = B[p*3 + q];
        float B_qp = B[q*3 + p];
        float B_qq = B[q*3 + q];
        
        B[p*3 + p] = c*c*B_pp - 2*c*s*B_pq + s*s*B_qq;
        B[p*3 + q] = 0.0f;
        B[q*3 + p] = 0.0f;
        B[q*3 + q] = s*s*B_pp + 2*c*s*B_pq + c*c*B_qq;
        
        // Update other elements
        for (int i = 0; i < 3; i++) {
            if (i != p && i != q) {
                float B_ip = B[i*3 + p];
                float B_iq = B[i*3 + q];
                B[i*3 + p] = c*B_ip - s*B_iq;
                B[p*3 + i] = B[i*3 + p];
                B[i*3 + q] = s*B_ip + c*B_iq;
                B[q*3 + i] = B[i*3 + q];
            }
        }
        
        // Update U and V
        for (int i = 0; i < 3; i++) {
            float U_ip = U[i*3 + p];
            float U_iq = U[i*3 + q];
            U[i*3 + p] = c*U_ip - s*U_iq;
            U[i*3 + q] = s*U_ip + c*U_iq;
            
            float V_ip = V[i*3 + p];
            float V_iq = V[i*3 + q];
            V[i*3 + p] = c*V_ip - s*V_iq;
            V[i*3 + q] = s*V_ip + c*V_iq;
        }
    }
    
    // Extract singular values
    for (int i = 0; i < 3; i++) {
        S[i] = sqrtf(B[i*3 + i]);
    }
}

// Compute R = V * U^T ensuring that det(R) = 1 (proper rotation)
__device__ void compute_rotation_from_svd(
    const float* U,     // Input: 3x3 left singular vectors
    const float* V,     // Input: 3x3 right singular vectors
    float* R            // Output: 3x3 rotation matrix
) {
    // Compute V * U^T
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += V[i*3 + k] * U[j*3 + k];  // Transposing U
            }
            R[i*3 + j] = sum;
        }
    }
    
    // Check determinant to ensure we have a proper rotation
    float det = R[0]*(R[4]*R[8] - R[5]*R[7]) - 
                R[1]*(R[3]*R[8] - R[5]*R[6]) + 
                R[2]*(R[3]*R[7] - R[4]*R[6]);
    
    // If det < 0, we need to flip the sign of the third column of V
    if (det < 0) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0f;
                for (int k = 0; k < 3; k++) {
                    float v_ik = V[i*3 + k];
                    if (k == 2) v_ik = -v_ik;  // Flip third column
                    sum += v_ik * U[j*3 + k];  // Transposing U
                }
                R[i*3 + j] = sum;
            }
        }
    }
}

/***************************
 * Procrustes Alignment Kernels
 ***************************/

__global__ void batch_procrustes_kernel(
    const float* src_points,       // Input: [total_points_src, 3] source points
    const float* tgt_points,       // Input: [total_points_tgt, 3] target points
    const int* src_offsets,        // Input: [batch_count] offsets of source points
    const int* tgt_offsets,        // Input: [batch_count] offsets of target points
    const int* src_sizes,          // Input: [batch_count] sizes of source point clouds
    const int* tgt_sizes,          // Input: [batch_count] sizes of target point clouds
    float* aligned_points,         // Output: [total_points_src, 3] aligned source points
    float* rotations,              // Output: [batch_count, 3, 3] rotation matrices
    float* translations,           // Output: [batch_count, 3] translation vectors
    int batch_count,               // Number of batch items
    int total_points_src           // Total source points
) {
    extern __shared__ float shared_mem[];
    
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int src_offset = src_offsets[batch_idx];
    int tgt_offset = tgt_offsets[batch_idx];
    int src_size = src_sizes[batch_idx];
    int tgt_size = tgt_sizes[batch_idx];
    
    // Use shared memory for centroids and covariance matrix
    float* s_src_centroid = shared_mem;                     // 3 floats
    float* s_tgt_centroid = shared_mem + 3;                 // 3 floats
    float* s_cov = shared_mem + 6;                          // 9 floats
    float* s_svd_workspace = shared_mem + 15;               // Workspace for SVD
    
    // Step 1: Compute centroids
    if (tid < 3) {
        s_src_centroid[tid] = 0.0f;
        s_tgt_centroid[tid] = 0.0f;
    }
    __syncthreads();
    
    // Sum up source points
    for (int i = tid; i < src_size; i += blockDim.x) {
        int idx = (src_offset + i) * 3;
        atomicAdd(&s_src_centroid[0], src_points[idx]);
        atomicAdd(&s_src_centroid[1], src_points[idx + 1]);
        atomicAdd(&s_src_centroid[2], src_points[idx + 2]);
    }
    
    // Sum up target points
    for (int i = tid; i < tgt_size; i += blockDim.x) {
        int idx = (tgt_offset + i) * 3;
        atomicAdd(&s_tgt_centroid[0], tgt_points[idx]);
        atomicAdd(&s_tgt_centroid[1], tgt_points[idx + 1]);
        atomicAdd(&s_tgt_centroid[2], tgt_points[idx + 2]);
    }
    __syncthreads();
    
    // Compute average
    if (tid == 0) {
        s_src_centroid[0] /= src_size;
        s_src_centroid[1] /= src_size;
        s_src_centroid[2] /= src_size;
        
        s_tgt_centroid[0] /= tgt_size;
        s_tgt_centroid[1] /= tgt_size;
        s_tgt_centroid[2] /= tgt_size;
    }
    __syncthreads();
    
    // Step 2: Compute covariance matrix
    if (tid < 9) {
        s_cov[tid] = 0.0f;
    }
    __syncthreads();
    
    // Compute the elements of the covariance matrix
    for (int i = tid; i < src_size; i += blockDim.x) {
        int src_idx = (src_offset + i) * 3;
        int tgt_idx = (tgt_offset + i) * 3;
        
        // Centered coordinates
        float src_x = src_points[src_idx] - s_src_centroid[0];
        float src_y = src_points[src_idx + 1] - s_src_centroid[1];
        float src_z = src_points[src_idx + 2] - s_src_centroid[2];
        
        float tgt_x = tgt_points[tgt_idx] - s_tgt_centroid[0];
        float tgt_y = tgt_points[tgt_idx + 1] - s_tgt_centroid[1];
        float tgt_z = tgt_points[tgt_idx + 2] - s_tgt_centroid[2];
        
        // Update covariance matrix elements
        atomicAdd(&s_cov[0], src_x * tgt_x);  // cov(0,0)
        atomicAdd(&s_cov[1], src_x * tgt_y);  // cov(0,1)
        atomicAdd(&s_cov[2], src_x * tgt_z);  // cov(0,2)
        atomicAdd(&s_cov[3], src_y * tgt_x);  // cov(1,0)
        atomicAdd(&s_cov[4], src_y * tgt_y);  // cov(1,1)
        atomicAdd(&s_cov[5], src_y * tgt_z);  // cov(1,2)
        atomicAdd(&s_cov[6], src_z * tgt_x);  // cov(2,0)
        atomicAdd(&s_cov[7], src_z * tgt_y);  // cov(2,1)
        atomicAdd(&s_cov[8], src_z * tgt_z);  // cov(2,2)
    }
    __syncthreads();
    
    // Step 3: Compute SVD and rotation matrix
    float U[9], S[3], V[9], R[9];
    if (tid == 0) {
        // Normalize covariance matrix
        for (int i = 0; i < 9; i++) {
            s_cov[i] /= src_size;
        }
        
        // Compute SVD
        compute_svd_3x3(s_cov, U, S, V);
        
        // Compute rotation matrix R = V * U^T
        compute_rotation_from_svd(U, V, R);
        
        // Save rotation matrix to global memory
        for (int i = 0; i < 9; i++) {
            rotations[batch_idx * 9 + i] = R[i];
        }
        
        // Compute translation: t = tgt_centroid - R * src_centroid
        float Rs[3] = {0, 0, 0};  // R * src_centroid
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Rs[i] += R[i*3 + j] * s_src_centroid[j];
            }
            translations[batch_idx * 3 + i] = s_tgt_centroid[i] - Rs[i];
        }
    }
    __syncthreads();
    
    // Load rotation and translation for this batch
    if (tid < 9) {
        R[tid] = rotations[batch_idx * 9 + tid];
    }
    float t[3];
    if (tid < 3) {
        t[tid] = translations[batch_idx * 3 + tid];
    }
    __syncthreads();
    
    // Step 4: Apply transformation to source points
    for (int i = tid; i < src_size; i += blockDim.x) {
        int src_idx = (src_offset + i) * 3;
        float x = src_points[src_idx];
        float y = src_points[src_idx + 1];
        float z = src_points[src_idx + 2];
        
        // Apply rotation
        float new_x = R[0]*x + R[1]*y + R[2]*z;
        float new_y = R[3]*x + R[4]*y + R[5]*z;
        float new_z = R[6]*x + R[7]*y + R[8]*z;
        
        // Apply translation
        aligned_points[src_idx] = new_x + t[0];
        aligned_points[src_idx + 1] = new_y + t[1];
        aligned_points[src_idx + 2] = new_z + t[2];
    }
}

/***************************
 * ICP Alignment Kernels
 ***************************/

// Optimal grid for k-nearest neighbor search
#define GRID_RES 32
#define GRID_SIZE (GRID_RES * GRID_RES * GRID_RES)

// Structure for accelerated nearest neighbor search
typedef struct {
    int* grid_indices;    // Starting indices for each grid cell
    int* grid_counts;     // Number of points in each grid cell
    int* point_indices;   // Sorted point indices
    float grid_min[3];    // Minimum coordinates of the grid
    float grid_max[3];    // Maximum coordinates of the grid
    float cell_size[3];   // Size of each grid cell
} UniformGrid;

// Build a uniform grid for accelerated nearest neighbor search
__global__ void build_uniform_grid(
    const float* points,        // [num_points, 3] point coordinates
    int num_points,             // Number of points
    UniformGrid grid,           // Uniform grid structure
    int offset                  // Offset in the points array
) {
    // Step 1: Find bounding box
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        grid.grid_min[0] = grid.grid_min[1] = grid.grid_min[2] = INFINITY;
        grid.grid_max[0] = grid.grid_max[1] = grid.grid_max[2] = -INFINITY;
    }
    __syncthreads();
    
    // Each thread processes some points
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_points; i += blockDim.x * gridDim.x) {
        int idx = (offset + i) * 3;
        float x = points[idx];
        float y = points[idx + 1];
        float z = points[idx + 2];
        
        atomicMin((int*)&grid.grid_min[0], __float_as_int(x));
        atomicMin((int*)&grid.grid_min[1], __float_as_int(y));
        atomicMin((int*)&grid.grid_min[2], __float_as_int(z));
        
        atomicMax((int*)&grid.grid_max[0], __float_as_int(x));
        atomicMax((int*)&grid.grid_max[1], __float_as_int(y));
        atomicMax((int*)&grid.grid_max[2], __float_as_int(z));
    }
    __syncthreads();
    
    // Step 2: Compute cell size
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        grid.cell_size[0] = (grid.grid_max[0] - grid.grid_min[0]) / GRID_RES;
        grid.cell_size[1] = (grid.grid_max[1] - grid.grid_min[1]) / GRID_RES;
        grid.cell_size[2] = (grid.grid_max[2] - grid.grid_min[2]) / GRID_RES;
        
        // Ensure cell size is not zero
        grid.cell_size[0] = fmaxf(grid.cell_size[0], 1e-5f);
        grid.cell_size[1] = fmaxf(grid.cell_size[1], 1e-5f);
        grid.cell_size[2] = fmaxf(grid.cell_size[2], 1e-5f);
    }
    __syncthreads();
    
    // Step 3: Count points per cell
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_points; i += blockDim.x * gridDim.x) {
        int idx = (offset + i) * 3;
        float x = points[idx];
        float y = points[idx + 1];
        float z = points[idx + 2];
        
        int gx = min(max((int)((x - grid.grid_min[0]) / grid.cell_size[0]), 0), GRID_RES - 1);
        int gy = min(max((int)((y - grid.grid_min[1]) / grid.cell_size[1]), 0), GRID_RES - 1);
        int gz = min(max((int)((z - grid.grid_min[2]) / grid.cell_size[2]), 0), GRID_RES - 1);
        
        int grid_idx = gz * GRID_RES * GRID_RES + gy * GRID_RES + gx;
        atomicAdd(&grid.grid_counts[grid_idx], 1);
    }
    __syncthreads();
    
    // Step 4: Compute starting indices for each cell
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int offset = 0;
        for (int i = 0; i < GRID_SIZE; i++) {
            grid.grid_indices[i] = offset;
            offset += grid.grid_counts[i];
            grid.grid_counts[i] = 0; // Reset counts for next step
        }
    }
    __syncthreads();
    
    // Step 5: Assign points to cells
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_points; i += blockDim.x * gridDim.x) {
        int idx = (offset + i) * 3;
        float x = points[idx];
        float y = points[idx + 1];
        float z = points[idx + 2];
        
        int gx = min(max((int)((x - grid.grid_min[0]) / grid.cell_size[0]), 0), GRID_RES - 1);
        int gy = min(max((int)((y - grid.grid_min[1]) / grid.cell_size[1]), 0), GRID_RES - 1);
        int gz = min(max((int)((z - grid.grid_min[2]) / grid.cell_size[2]), 0), GRID_RES - 1);
        
        int grid_idx = gz * GRID_RES * GRID_RES + gy * GRID_RES + gx;
        int pos = atomicAdd(&grid.grid_counts[grid_idx], 1);
        grid.point_indices[grid.grid_indices[grid_idx] + pos] = i;
    }
}

// Find nearest neighbors using a uniform grid
__global__ void find_nearest_neighbors_grid(
    const float* src_points,        // [total_points_src, 3] source points
    const float* tgt_points,        // [total_points_tgt, 3] target points
    const int* src_offsets,         // [batch_count] offsets of source points
    const int* tgt_offsets,         // [batch_count] offsets of target points
    const int* src_sizes,           // [batch_count] sizes of source point clouds
    const int* tgt_sizes,           // [batch_count] sizes of target point clouds
    int* correspondences,           // [total_points_src] indices of nearest targets
    float* distances,               // [total_points_src] squared distances to nearest
    const UniformGrid grid,         // Uniform grid for accelerated search
    int batch_idx,                  // Batch index to process
    int total_points_src            // Total source points
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int src_offset = src_offsets[batch_idx];
    int tgt_offset = tgt_offsets[batch_idx];
    int src_size = src_sizes[batch_idx];
    int tgt_size = tgt_sizes[batch_idx];
    
    // Check bounds
    if (tid >= src_size) return;
    
    int src_idx = (src_offset + tid) * 3;
    float src_x = src_points[src_idx];
    float src_y = src_points[src_idx + 1];
    float src_z = src_points[src_idx + 2];
    
    // Find grid cell of the source point
    int gx = min(max((int)((src_x - grid.grid_min[0]) / grid.cell_size[0]), 0), GRID_RES - 1);
    int gy = min(max((int)((src_y - grid.grid_min[1]) / grid.cell_size[1]), 0), GRID_RES - 1);
    int gz = min(max((int)((src_z - grid.grid_min[2]) / grid.cell_size[2]), 0), GRID_RES - 1);
    
    // Search in current cell and neighboring cells
    float min_dist = INFINITY;
    int min_idx = -1;
    
    for (int dz = -1; dz <= 1; dz++) {
        int z = gz + dz;
        if (z < 0 || z >= GRID_RES) continue;
        
        for (int dy = -1; dy <= 1; dy++) {
            int y = gy + dy;
            if (y < 0 || y >= GRID_RES) continue;
            
            for (int dx = -1; dx <= 1; dx++) {
                int x = gx + dx;
                if (x < 0 || x >= GRID_RES) continue;
                
                int grid_idx = z * GRID_RES * GRID_RES + y * GRID_RES + x;
                int start = grid.grid_indices[grid_idx];
                int count = grid.grid_counts[grid_idx];
                
                // Search through points in this cell
                for (int j = 0; j < count; j++) {
                    int tgt_local_idx = grid.point_indices[start + j];
                    int tgt_idx = (tgt_offset + tgt_local_idx) * 3;
                    
                    float tgt_x = tgt_points[tgt_idx];
                    float tgt_y = tgt_points[tgt_idx + 1];
                    float tgt_z = tgt_points[tgt_idx + 2];
                    
                    float dx = src_x - tgt_x;
                    float dy = src_y - tgt_y;
                    float dz = src_z - tgt_z;
                    float dist = dx*dx + dy*dy + dz*dz;
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = tgt_local_idx;
                    }
                }
            }
        }
    }
    
    // If we didn't find a point in neighboring cells, fallback to global search
    if (min_idx == -1) {
        for (int j = 0; j < tgt_size; j++) {
            int tgt_idx = (tgt_offset + j) * 3;
            float tgt_x = tgt_points[tgt_idx];
            float tgt_y = tgt_points[tgt_idx + 1];
            float tgt_z = tgt_points[tgt_idx + 2];
            
            float dx = src_x - tgt_x;
            float dy = src_y - tgt_y;
            float dz = src_z - tgt_z;
            float dist = dx*dx + dy*dy + dz*dz;
            
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }
    }
    
    // Store correspondence and distance
    correspondences[src_offset + tid] = tgt_offset + min_idx;
    distances[src_offset + tid] = min_dist;
}

// ICP alignment kernel for one batch item
__global__ void batch_icp_kernel(
    const float* src_points,       // Input: [total_points_src, 3] source points
    const float* tgt_points,       // Input: [total_points_tgt, 3] target points
    const int* src_offsets,        // Input: [batch_count] offsets of source points
    const int* tgt_offsets,        // Input: [batch_count] offsets of target points
    const int* src_sizes,          // Input: [batch_count] sizes of source point clouds
    const int* tgt_sizes,          // Input: [batch_count] sizes of target point clouds
    float* aligned_points,         // Output: [total_points_src, 3] aligned source points
    float* rotations,              // Output: [batch_count, 3, 3] rotation matrices
    float* translations,           // Output: [batch_count, 3] translation vectors
    float* errors,                 // Output: [batch_count] error for each batch item
    int* correspondences,          // Workspace: [total_points_src] indices for NN
    float* distances,              // Workspace: [total_points_src] distances for NN
    UniformGrid grid,              // Workspace: uniform grid for acceleration
    int batch_count,               // Number of batch items
    int total_points_src,          // Total source points
    int max_iterations,            // Maximum ICP iterations
    float convergence_threshold    // ICP convergence threshold
) {
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int src_offset = src_offsets[batch_idx];
    int tgt_offset = tgt_offsets[batch_idx];
    int src_size = src_sizes[batch_idx];
    int tgt_size = tgt_sizes[batch_idx];
    
    // Use shared memory for various temporary data
    extern __shared__ float shared_mem[];
    float* s_errors = shared_mem;                     // 2 floats (current and previous error)
    float* s_src_centroid = shared_mem + 2;           // 3 floats
    float* s_tgt_centroid = shared_mem + 5;           // 3 floats
    float* s_cov = shared_mem + 8;                    // 9 floats
    float* s_R = shared_mem + 17;                     // 9 floats (current rotation)
    float* s_t = shared_mem + 26;                     // 3 floats (current translation)
    float* s_svd_workspace = shared_mem + 29;         // Workspace for SVD
    
    // Initialize the current transformation to identity
    if (tid < 9) {
        s_R[tid] = (tid % 4 == 0) ? 1.0f : 0.0f;  // Identity matrix
    }
    if (tid < 3) {
        s_t[tid] = 0.0f;  // Zero translation
    }
    if (tid == 0) {
        s_errors[0] = INFINITY;  // Previous error
        s_errors[1] = 0.0f;      // Current error
    }
    __syncthreads();
    
    // Copy source points to aligned points (initial guess)
    for (int i = tid; i < src_size; i += blockDim.x) {
        int src_idx = (src_offset + i) * 3;
        aligned_points[src_idx] = src_points[src_idx];
        aligned_points[src_idx + 1] = src_points[src_idx + 1];
        aligned_points[src_idx + 2] = src_points[src_idx + 2];
    }
    __syncthreads();
    
    // Build uniform grid for target points (once before the iterations)
    const int grid_blocks = (tgt_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (tid == 0) {
        build_uniform_grid<<<grid_blocks, BLOCK_SIZE>>>(
            tgt_points, tgt_size, grid, tgt_offset);
        cudaDeviceSynchronize();  // Wait for grid construction
    }
    __syncthreads();
    
    // Main ICP loop
    for (int iter = 0; iter < max_iterations; iter++) {
        // Step 1: Find nearest neighbors using uniform grid
        const int nn_blocks = (src_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (tid == 0) {
            find_nearest_neighbors_grid<<<nn_blocks, BLOCK_SIZE>>>(
                aligned_points, tgt_points, src_offsets, tgt_offsets,
                src_sizes, tgt_sizes, correspondences, distances,
                grid, batch_idx, total_points_src);
            cudaDeviceSynchronize();  // Wait for NN search
        }
        __syncthreads();
        
        // Step 2: Compute error and check for convergence
        if (tid == 0) {
            // Save previous error
            s_errors[0] = s_errors[1];
            
            // Compute current error (average distance)
            s_errors[1] = 0.0f;
            for (int i = 0; i < src_size; i++) {
                s_errors[1] += distances[src_offset + i];
            }
            s_errors[1] /= src_size;
            
            // Check for convergence
            if (iter > 0 && fabs(s_errors[0] - s_errors[1]) < convergence_threshold) {
                // Break the loop (since we can't break in CUDA, we'll set iter to max)
                iter = max_iterations;
            }
        }
        __syncthreads();
        
        // Check if we should break
        if (iter == max_iterations) break;
        
        // Step 3: Compute centroids
        if (tid < 3) {
            s_src_centroid[tid] = 0.0f;
            s_tgt_centroid[tid] = 0.0f;
        }
        __syncthreads();
        
        // Sum up source points
        for (int i = tid; i < src_size; i += blockDim.x) {
            int src_idx = (src_offset + i) * 3;
            atomicAdd(&s_src_centroid[0], aligned_points[src_idx]);
            atomicAdd(&s_src_centroid[1], aligned_points[src_idx + 1]);
            atomicAdd(&s_src_centroid[2], aligned_points[src_idx + 2]);
        }
        
        // Sum up corresponding target points
        for (int i = tid; i < src_size; i += blockDim.x) {
            int tgt_idx = correspondences[src_offset + i] * 3;
            atomicAdd(&s_tgt_centroid[0], tgt_points[tgt_idx]);
            atomicAdd(&s_tgt_centroid[1], tgt_points[tgt_idx + 1]);
            atomicAdd(&s_tgt_centroid[2], tgt_points[tgt_idx + 2]);
        }
        __syncthreads();
        
        // Compute average
        if (tid == 0) {
            s_src_centroid[0] /= src_size;
            s_src_centroid[1] /= src_size;
            s_src_centroid[2] /= src_size;
            
            s_tgt_centroid[0] /= src_size;
            s_tgt_centroid[1] /= src_size;
            s_tgt_centroid[2] /= src_size;
        }
        __syncthreads();
        
        // Step 4: Compute covariance matrix
        if (tid < 9) {
            s_cov[tid] = 0.0f;
        }
        __syncthreads();
        
        // Compute the elements of the covariance matrix
        for (int i = tid; i < src_size; i += blockDim.x) {
            int src_idx = (src_offset + i) * 3;
            int tgt_idx = correspondences[src_offset + i] * 3;
            
            // Centered coordinates
            float src_x = aligned_points[src_idx] - s_src_centroid[0];
            float src_y = aligned_points[src_idx + 1] - s_src_centroid[1];
            float src_z = aligned_points[src_idx + 2] - s_src_centroid[2];
            
            float tgt_x = tgt_points[tgt_idx] - s_tgt_centroid[0];
            float tgt_y = tgt_points[tgt_idx + 1] - s_tgt_centroid[1];
            float tgt_z = tgt_points[tgt_idx + 2] - s_tgt_centroid[2];
            
            // Update covariance matrix elements
            atomicAdd(&s_cov[0], src_x * tgt_x);  // cov(0,0)
            atomicAdd(&s_cov[1], src_x * tgt_y);  // cov(0,1)
            atomicAdd(&s_cov[2], src_x * tgt_z);  // cov(0,2)
            atomicAdd(&s_cov[3], src_y * tgt_x);  // cov(1,0)
            atomicAdd(&s_cov[4], src_y * tgt_y);  // cov(1,1)
            atomicAdd(&s_cov[5], src_y * tgt_z);  // cov(1,2)
            atomicAdd(&s_cov[6], src_z * tgt_x);  // cov(2,0)
            atomicAdd(&s_cov[7], src_z * tgt_y);  // cov(2,1)
            atomicAdd(&s_cov[8], src_z * tgt_z);  // cov(2,2)
        }
        __syncthreads();
        
        // Step 5: Compute SVD and rotation matrix
        float U[9], S[3], V[9], R[9];
        if (tid == 0) {
            // Normalize covariance matrix
            for (int i = 0; i < 9; i++) {
                s_cov[i] /= src_size;
            }
            
            // Compute SVD
            compute_svd_3x3(s_cov, U, S, V);
            
            // Compute rotation matrix R = V * U^T
            compute_rotation_from_svd(U, V, R);
            
            // Copy to shared memory
            for (int i = 0; i < 9; i++) {
                s_R[i] = R[i];
            }
            
            // Compute translation: t = tgt_centroid - R * src_centroid
            float Rs[3] = {0, 0, 0};  // R * src_centroid
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    Rs[i] += R[i*3 + j] * s_src_centroid[j];
                }
                s_t[i] = s_tgt_centroid[i] - Rs[i];
            }
        }
        __syncthreads();
        
        // Step 6: Apply transformation to source points
        for (int i = tid; i < src_size; i += blockDim.x) {
            int src_idx = (src_offset + i) * 3;
            float x = src_points[src_idx];
            float y = src_points[src_idx + 1];
            float z = src_points[src_idx + 2];
            
            // Apply rotation
            float new_x = s_R[0]*x + s_R[1]*y + s_R[2]*z;
            float new_y = s_R[3]*x + s_R[4]*y + s_R[5]*z;
            float new_z = s_R[6]*x + s_R[7]*y + s_R[8]*z;
            
            // Apply translation
            aligned_points[src_idx] = new_x + s_t[0];
            aligned_points[src_idx + 1] = new_y + s_t[1];
            aligned_points[src_idx + 2] = new_z + s_t[2];
        }
        __syncthreads();
    }
    
    // Save final transformation and error
    if (tid < 9) {
        rotations[batch_idx * 9 + tid] = s_R[tid];
    }
    if (tid < 3) {
        translations[batch_idx * 3 + tid] = s_t[tid];
    }
    if (tid == 0) {
        errors[batch_idx] = s_errors[1];
    }
}

/***************************
 * Chamfer Distance Kernel
 ***************************/

// Compute bidirectional Chamfer distance between point clouds
__global__ void batch_chamfer_distance_kernel(
    const float* src_points,       // Input: [total_points_src, 3] source points
    const float* tgt_points,       // Input: [total_points_tgt, 3] target points
    const int* src_offsets,        // Input: [batch_count] offsets of source points
    const int* tgt_offsets,        // Input: [batch_count] offsets of target points
    const int* src_sizes,          // Input: [batch_count] sizes of source point clouds
    const int* tgt_sizes,          // Input: [batch_count] sizes of target point clouds
    float* distances,              // Output: [batch_count] chamfer distances
    int batch_count                // Number of batch items
) {
    extern __shared__ float shared_mem[];
    
    // Each block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_count) return;
    
    int tid = threadIdx.x;
    int src_offset = src_offsets[batch_idx];
    int tgt_offset = tgt_offsets[batch_idx];
    int src_size = src_sizes[batch_idx];
    int tgt_size = tgt_sizes[batch_idx];
    
    // Initialize shared memory for accumulating distances
    float* s_src_to_tgt = shared_mem;
    float* s_tgt_to_src = shared_mem + 1;
    
    if (tid == 0) {
        s_src_to_tgt[0] = 0.0f;
        s_tgt_to_src[0] = 0.0f;
    }
    __syncthreads();
    
    // Step 1: For each source point, find closest target point
    for (int i = tid; i < src_size; i += blockDim.x) {
        int src_idx = (src_offset + i) * 3;
        float src_x = src_points[src_idx];
        float src_y = src_points[src_idx + 1];
        float src_z = src_points[src_idx + 2];
        
        float min_dist = INFINITY;
        
        // Find the closest target point
        for (int j = 0; j < tgt_size; j++) {
            int tgt_idx = (tgt_offset + j) * 3;
            float tgt_x = tgt_points[tgt_idx];
            float tgt_y = tgt_points[tgt_idx + 1];
            float tgt_z = tgt_points[tgt_idx + 2];
            
            float dx = src_x - tgt_x;
            float dy = src_y - tgt_y;
            float dz = src_z - tgt_z;
            float dist = dx*dx + dy*dy + dz*dz;
            
            min_dist = fminf(min_dist, dist);
        }
        
        // Add to sum
        atomicAdd(s_src_to_tgt, min_dist);
    }
    
    // Step 2: For each target point, find closest source point
    for (int i = tid; i < tgt_size; i += blockDim.x) {
        int tgt_idx = (tgt_offset + i) * 3;
        float tgt_x = tgt_points[tgt_idx];
        float tgt_y = tgt_points[tgt_idx + 1];
        float tgt_z = tgt_points[tgt_idx + 2];
        
        float min_dist = INFINITY;
        
        // Find the closest source point
        for (int j = 0; j < src_size; j++) {
            int src_idx = (src_offset + j) * 3;
            float src_x = src_points[src_idx];
            float src_y = src_points[src_idx + 1];
            float src_z = src_points[src_idx + 2];
            
            float dx = tgt_x - src_x;
            float dy = tgt_y - src_y;
            float dz = tgt_z - src_z;
            float dist = dx*dx + dy*dy + dz*dz;
            
            min_dist = fminf(min_dist, dist);
        }
        
        // Add to sum
        atomicAdd(s_tgt_to_src, min_dist);
    }
    __syncthreads();
    
    // Compute final Chamfer distance
    if (tid == 0) {
        float src_to_tgt_avg = s_src_to_tgt[0] / src_size;
        float tgt_to_src_avg = s_tgt_to_src[0] / tgt_size;
        distances[batch_idx] = (src_to_tgt_avg + tgt_to_src_avg) / 2.0f;
    }
}

/***************************
 * PyTorch to CUDA Interface Functions
 ***************************/

void procrustes_align_cuda(
    const torch::Tensor& src_points,
    const torch::Tensor& tgt_points,
    const torch::Tensor& src_batch_idx,
    const torch::Tensor& tgt_batch_idx,
    torch::Tensor& aligned_points,
    torch::Tensor& rotations,
    torch::Tensor& translations
) {
    const int threads = BLOCK_SIZE;
    const int batch_count = rotations.size(0);
    const int total_points_src = src_points.size(0);
    const int total_points_tgt = tgt_points.size(0);
    
    // Compute max shared memory needed (conservative estimate)
    size_t shared_mem_size = 512; // Plenty for our needs
    
    // Allocate and initialize device memory for batched format
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(src_points.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(src_points.device());
    
    // Offsets and sizes for source and target point clouds
    auto src_offsets = torch::zeros({batch_count}, options_int);
    auto tgt_offsets = torch::zeros({batch_count}, options_int);
    auto src_sizes = torch::zeros({batch_count}, options_int);
    auto tgt_sizes = torch::zeros({batch_count}, options_int);
    
    // Convert src_batch_idx to internal format
    auto convert_src_kernel = convert_tensor_to_batched_format;
    convert_src_kernel<<<(total_points_src + threads - 1) / threads, threads, batch_count * sizeof(int)>>>(
        src_points.data_ptr<float>(),
        src_batch_idx.data_ptr<int64_t>(),
        aligned_points.data_ptr<float>(),  // Reuse aligned_points as temporary buffer
        src_offsets.data_ptr<int>(),
        src_sizes.data_ptr<int>(),
        batch_count,
        total_points_src
    );
    
    // Convert tgt_batch_idx to internal format
    auto tgt_packed = torch::zeros({total_points_tgt, 3}, options_float);
    auto convert_tgt_kernel = convert_tensor_to_batched_format;
    convert_tgt_kernel<<<(total_points_tgt + threads - 1) / threads, threads, batch_count * sizeof(int)>>>(
        tgt_points.data_ptr<float>(),
        tgt_batch_idx.data_ptr<int64_t>(),
        tgt_packed.data_ptr<float>(),
        tgt_offsets.data_ptr<int>(),
        tgt_sizes.data_ptr<int>(),
        batch_count,
        total_points_tgt
    );
    
    // Run Procrustes alignment kernel
    auto proc_kernel = batch_procrustes_kernel;
    proc_kernel<<<batch_count, threads, shared_mem_size>>>(
        src_points.data_ptr<float>(),
        tgt_points.data_ptr<float>(),
        src_offsets.data_ptr<int>(),
        tgt_offsets.data_ptr<int>(),
        src_sizes.data_ptr<int>(),
        tgt_sizes.data_ptr<int>(),
        aligned_points.data_ptr<float>(),
        rotations.data_ptr<float>(),
        translations.data_ptr<float>(),
        batch_count,
        total_points_src
    );
}

void icp_align_cuda(
    const torch::Tensor& src_points,
    const torch::Tensor& tgt_points,
    const torch::Tensor& src_batch_idx,
    const torch::Tensor& tgt_batch_idx,
    torch::Tensor& aligned_points,
    torch::Tensor& rotations,
    torch::Tensor& translations,
    int max_iterations,
    float convergence_threshold
) {
    const int threads = BLOCK_SIZE;
    const int batch_count = rotations.size(0);
    const int total_points_src = src_points.size(0);
    const int total_points_tgt = tgt_points.size(0);
    
    // Compute max shared memory needed (conservative estimate)
    size_t shared_mem_size = 1024; // Plenty for our needs
    
    // Allocate and initialize device memory for batched format
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(src_points.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(src_points.device());
    
    // Offsets and sizes for source and target point clouds
    auto src_offsets = torch::zeros({batch_count}, options_int);
    auto tgt_offsets = torch::zeros({batch_count}, options_int);
    auto src_sizes = torch::zeros({batch_count}, options_int);
    auto tgt_sizes = torch::zeros({batch_count}, options_int);
    
    // Temporary buffers for ICP
    auto errors = torch::zeros({batch_count}, options_float);
    auto correspondences = torch::zeros({total_points_src}, options_int);
    auto distances = torch::zeros({total_points_src}, options_float);
    
    // Allocate memory for uniform grid
    auto grid_indices = torch::zeros({GRID_SIZE}, options_int);
    auto grid_counts = torch::zeros({GRID_SIZE}, options_int);
    auto point_indices = torch::zeros({total_points_tgt}, options_int);
    
    // Convert src_batch_idx to internal format
    auto convert_src_kernel = convert_tensor_to_batched_format;
    convert_src_kernel<<<(total_points_src + threads - 1) / threads, threads, batch_count * sizeof(int)>>>(
        src_points.data_ptr<float>(),
        src_batch_idx.data_ptr<int64_t>(),
        aligned_points.data_ptr<float>(),  // Reuse aligned_points as temporary buffer
        src_offsets.data_ptr<int>(),
        src_sizes.data_ptr<int>(),
        batch_count,
        total_points_src
    );
    
    // Convert tgt_batch_idx to internal format
    auto tgt_packed = torch::zeros({total_points_tgt, 3}, options_float);
    auto convert_tgt_kernel = convert_tensor_to_batched_format;
    convert_tgt_kernel<<<(total_points_tgt + threads - 1) / threads, threads, batch_count * sizeof(int)>>>(
        tgt_points.data_ptr<float>(),
        tgt_batch_idx.data_ptr<int64_t>(),
        tgt_packed.data_ptr<float>(),
        tgt_offsets.data_ptr<int>(),
        tgt_sizes.data_ptr<int>(),
        batch_count,
        total_points_tgt
    );
    
    // Create uniform grid struct
    UniformGrid grid;
    grid.grid_indices = grid_indices.data_ptr<int>();
    grid.grid_counts = grid_counts.data_ptr<int>();
    grid.point_indices = point_indices.data_ptr<int>();
    
    // Run ICP alignment kernel for each batch item
    for (int b = 0; b < batch_count; b++) {
        auto icp_kernel = batch_icp_kernel;
        icp_kernel<<<1, threads, shared_mem_size>>>(
            src_points.data_ptr<float>(),
            tgt_points.data_ptr<float>(),
            src_offsets.data_ptr<int>(),
            tgt_offsets.data_ptr<int>(),
            src_sizes.data_ptr<int>(),
            tgt_sizes.data_ptr<int>(),
            aligned_points.data_ptr<float>(),
            rotations.data_ptr<float>(),
            translations.data_ptr<float>(),
            errors.data_ptr<float>(),
            correspondences.data_ptr<int>(),
            distances.data_ptr<float>(),
            grid,
            batch_count,
            total_points_src,
            max_iterations,
            convergence_threshold
        );
    }
}

void chamfer_distance_cuda(
    const torch::Tensor& src_points,
    const torch::Tensor& tgt_points,
    const torch::Tensor& src_batch_idx,
    const torch::Tensor& tgt_batch_idx,
    torch::Tensor& distances
) {
    const int threads = BLOCK_SIZE;
    const int batch_count = distances.size(0);
    const int total_points_src = src_points.size(0);
    const int total_points_tgt = tgt_points.size(0);
    
    // Compute max shared memory needed
    size_t shared_mem_size = 2 * sizeof(float); // Just need two floats for the distance sums
    
    // Allocate and initialize device memory for batched format
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(src_points.device());
    
    // Offsets and sizes for source and target point clouds
    auto src_offsets = torch::zeros({batch_count}, options_int);
    auto tgt_offsets = torch::zeros({batch_count}, options_int);
    auto src_sizes = torch::zeros({batch_count}, options_int);
    auto tgt_sizes = torch::zeros({batch_count}, options_int);
    
    // Convert src_batch_idx to internal format
    auto temp_points = torch::zeros_like(src_points);
    auto convert_src_kernel = convert_tensor_to_batched_format;
    convert_src_kernel<<<(total_points_src + threads - 1) / threads, threads, batch_count * sizeof(int)>>>(
        src_points.data_ptr<float>(),
        src_batch_idx.data_ptr<int64_t>(),
        temp_points.data_ptr<float>(),  // Just a temporary buffer
        src_offsets.data_ptr<int>(),
        src_sizes.data_ptr<int>(),
        batch_count,
        total_points_src
    );
    
    // Convert tgt_batch_idx to internal format
    auto convert_tgt_kernel = convert_tensor_to_batched_format;
    convert_tgt_kernel<<<(total_points_tgt + threads - 1) / threads, threads, batch_count * sizeof(int)>>>(
        tgt_points.data_ptr<float>(),
        tgt_batch_idx.data_ptr<int64_t>(),
        temp_points.data_ptr<float>(),  // Just reuse the buffer
        tgt_offsets.data_ptr<int>(),
        tgt_sizes.data_ptr<int>(),
        batch_count,
        total_points_tgt
    );
    
    // Run Chamfer distance kernel
    auto chamfer_kernel = batch_chamfer_distance_kernel;
    chamfer_kernel<<<batch_count, threads, shared_mem_size>>>(
        src_points.data_ptr<float>(),
        tgt_points.data_ptr<float>(),
        src_offsets.data_ptr<int>(),
        tgt_offsets.data_ptr<int>(),
        src_sizes.data_ptr<int>(),
        tgt_sizes.data_ptr<int>(),
        distances.data_ptr<float>(),
        batch_count
    );
}