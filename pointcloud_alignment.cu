// pointcloud_alignment.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define BLOCK_SIZE 256

// Macro for error checking
#define CUDA_CHECK(err) do { \
    if(err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)


/* ===============================================
   Device Kernels
   =============================================== */

// Kernel: Compute centroid of an array of points using reduction.
// The kernel uses shared memory to sum over points; the block–level
// partial sums are then atomically added to a global accumulator.
__global__ void computeCentroid(const float3* points, int N, float3* centroid) {
    __shared__ float3 sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        sum.x += points[i].x;
        sum.y += points[i].y;
        sum.z += points[i].z;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid].x += sdata[tid+s].x;
            sdata[tid].y += sdata[tid+s].y;
            sdata[tid].z += sdata[tid+s].z;
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&centroid->x, sdata[0].x);
        atomicAdd(&centroid->y, sdata[0].y);
        atomicAdd(&centroid->z, sdata[0].z);
    }
}

// Kernel: Subtract a given centroid from every point.
__global__ void subtractCentroid(float3* points, int N, float3 centroid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        points[idx].x -= centroid.x;
        points[idx].y -= centroid.y;
        points[idx].z -= centroid.z;
    }
}

// Kernel: Compute the 3x3 covariance matrix (in row–major order) between two point sets.
// Each thread computes products for one pair of points and uses atomicAdd to accumulate.
__global__ void computeCovariance(const float3* src, const float3* tgt, int N, float* cov) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float3 p = src[idx];
        float3 q = tgt[idx];
        float vals[9];
        vals[0] = p.x * q.x;
        vals[1] = p.x * q.y;
        vals[2] = p.x * q.z;
        vals[3] = p.y * q.x;
        vals[4] = p.y * q.y;
        vals[5] = p.y * q.z;
        vals[6] = p.z * q.x;
        vals[7] = p.z * q.y;
        vals[8] = p.z * q.z;
        for (int j = 0; j < 9; j++) {
            atomicAdd(&cov[j], vals[j]);
        }
    }
}

// Kernel: Apply a rigid transformation (rotation R and translation t) to each point.
__global__ void applyTransform(float3* points, int N, const float* R, float3 t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float3 p = points[idx];
        float x = R[0]*p.x + R[1]*p.y + R[2]*p.z;
        float y = R[3]*p.x + R[4]*p.y + R[5]*p.z;
        float z = R[6]*p.x + R[7]*p.y + R[8]*p.z;
        points[idx].x = x + t.x;
        points[idx].y = y + t.y;
        points[idx].z = z + t.z;
    }
}

// Kernel: Brute-force nearest neighbor search (each source point finds the closest target).
__global__ void findNearestNeighbor(const float3* src, const float3* tgt, int Nsrc, int Ntgt, int* indices, float* dists) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nsrc) {
        float minDist = 1e10f;
        int minIdx = -1;
        float3 p = src[idx];
        for (int j = 0; j < Ntgt; j++) {
            float3 q = tgt[j];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float dz = p.z - q.z;
            float dist = dx*dx + dy*dy + dz*dz;
            if (dist < minDist) {
                minDist = dist;
                minIdx = j;
            }
        }
        indices[idx] = minIdx;
        dists[idx] = minDist;
    }
}

// Kernel: Given an array of correspondence indices, gather the corresponding target points.
__global__ void gatherCorrespondences(const float3* tgt, const int* indices, int N, float3* corr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int tidx = indices[idx];
        corr[idx] = tgt[tidx];
    }
}

/* ===============================================
   Host Helper Functions
   =============================================== */

// Multiply a 3x3 matrix (row-major) with a float3 vector.
void mat3_mul_vec3(const float R[9], const float3 &v, float3 &out) {
    out.x = R[0]*v.x + R[1]*v.y + R[2]*v.z;
    out.y = R[3]*v.x + R[4]*v.y + R[5]*v.z;
    out.z = R[6]*v.x + R[7]*v.y + R[8]*v.z;
}

// Convert quaternion (q0,q1,q2,q3) to a 3x3 rotation matrix (row-major).
void quat_to_rot(const float q[4], float R[9]) {
    float q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    R[0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
    R[1] = 2.0f*(q1*q2 - q0*q3);
    R[2] = 2.0f*(q1*q3 + q0*q2);
    R[3] = 2.0f*(q1*q2 + q0*q3);
    R[4] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
    R[5] = 2.0f*(q2*q3 - q0*q1);
    R[6] = 2.0f*(q1*q3 - q0*q2);
    R[7] = 2.0f*(q2*q3 + q0*q1);
    R[8] = q0*q0 - q1*q1 - q2*q2 + q3*q3;
}

// Compute rotation matrix from a 3x3 covariance matrix using Horn's method.
// The covariance matrix "cov" is a 9-element array in row–major order.
void computeRotationFromCovariance(const float cov[9], float R[9]) {
    // Construct symmetric 4x4 matrix N (from Horn 1987)
    float N[16];
    float trace = cov[0] + cov[4] + cov[8];
    N[0]  = trace;
    N[1]  = cov[7] - cov[5];
    N[2]  = cov[2] - cov[6];
    N[3]  = cov[3] - cov[1];

    N[4]  = cov[7] - cov[5];
    N[5]  = cov[0] - cov[4] - cov[8];
    N[6]  = cov[1] + cov[3];
    N[7]  = cov[2] + cov[6];

    N[8]  = cov[2] - cov[6];
    N[9]  = cov[1] + cov[3];
    N[10] = -cov[0] + cov[4] - cov[8];
    N[11] = cov[7] + cov[5];

    N[12] = cov[3] - cov[1];
    N[13] = cov[2] + cov[6];
    N[14] = cov[7] + cov[5];
    N[15] = -cov[0] - cov[4] + cov[8];

    // Use power iteration to compute the dominant eigenvector (quaternion).
    float q[4] = {1, 0, 0, 0}; // initial guess
    const int iterations = 50;
    for (int iter = 0; iter < iterations; iter++) {
        float q_new[4] = {0, 0, 0, 0};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                q_new[i] += N[i*4+j] * q[j];
            }
        }
        float norm = sqrt(q_new[0]*q_new[0] + q_new[1]*q_new[1] +
                          q_new[2]*q_new[2] + q_new[3]*q_new[3]);
        if (norm > 1e-6) {
            for (int i = 0; i < 4; i++) {
                q[i] = q_new[i] / norm;
            }
        }
    }
    // Convert quaternion to rotation matrix.
    quat_to_rot(q, R);
}

// Compute centroid on the host (for validation).
void computeHostCentroid(const float3* points, int N, float3 &centroid) {
    centroid = make_float3(0,0,0);
    for (int i = 0; i < N; i++) {
        centroid.x += points[i].x;
        centroid.y += points[i].y;
        centroid.z += points[i].z;
    }
    centroid.x /= N;
    centroid.y /= N;
    centroid.z /= N;
}

// Apply transformation (R, t) to a point.
void transformPoint(const float R[9], const float3 &p, const float3 &t, float3 &out) {
    out.x = R[0]*p.x + R[1]*p.y + R[2]*p.z + t.x;
    out.y = R[3]*p.x + R[4]*p.y + R[5]*p.z + t.y;
    out.z = R[6]*p.x + R[7]*p.y + R[8]*p.z + t.z;
}

// Compute root mean squared error between two point clouds.
float computeRMSE(const float3* a, const float3* b, int N) {
    float err = 0;
    for (int i = 0; i < N; i++) {
        float dx = a[i].x - b[i].x;
        float dy = a[i].y - b[i].y;
        float dz = a[i].z - b[i].z;
        err += dx*dx + dy*dy + dz*dz;
    }
    return sqrt(err / N);
}

// Generate a synthetic point cloud (points uniformly distributed in [-1,1]^3).
void generatePointCloud(float3* points, int N) {
    for (int i = 0; i < N; i++) {
        points[i].x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        points[i].y = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        points[i].z = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// Apply a known transformation on the CPU.
void applyTransformationCPU(const float3* src, float3* dst, int N, const float R[9], const float3 &t) {
    for (int i = 0; i < N; i++) {
        transformPoint(R, src[i], t, dst[i]);
    }
}

/* ===============================================
   Main: Validation and Test Scripts
   =============================================== */

int main() {
    srand(time(NULL));
    const int N = 1024; // Number of points
    size_t size = N * sizeof(float3);

    // Allocate host memory for source and target clouds.
    float3* h_src = (float3*)malloc(size);
    float3* h_tgt = (float3*)malloc(size);
    float3* h_aligned = (float3*)malloc(size);

    // Generate synthetic source point cloud.
    generatePointCloud(h_src, N);

    // Define a known transformation: rotation (30° about Z) and translation.
    float angle = 30.0f * 3.1415926f / 180.0f;
    float R_true[9] = {
        cos(angle), -sin(angle), 0,
        sin(angle),  cos(angle), 0,
        0,           0,          1
    };
    float3 t_true = make_float3(0.5f, -0.3f, 0.8f);

    // Create target point cloud by applying the transformation.
    applyTransformationCPU(h_src, h_tgt, N, R_true, t_true);

    // Allocate device memory and copy source and target.
    float3 *d_src, *d_tgt;
    CUDA_CHECK(cudaMalloc(&d_src, size));
    CUDA_CHECK(cudaMalloc(&d_tgt, size));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, size, cudaMemcpyHostToDevice));

    // --------- Procrustes Alignment (Known Correspondences) ---------
    // Compute centroids on device.
    float3 h_centroid_src = make_float3(0,0,0);
    float3 h_centroid_tgt = make_float3(0,0,0);
    float3 *d_centroid;
    CUDA_CHECK(cudaMalloc(&d_centroid, sizeof(float3)));

    CUDA_CHECK(cudaMemset(d_centroid, 0, sizeof(float3)));
    computeCentroid<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_src, N, d_centroid);
    CUDA_CHECK(cudaMemcpy(&h_centroid_src, d_centroid, sizeof(float3), cudaMemcpyDeviceToHost));
    h_centroid_src.x /= N; h_centroid_src.y /= N; h_centroid_src.z /= N;

    CUDA_CHECK(cudaMemset(d_centroid, 0, sizeof(float3)));
    computeCentroid<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_tgt, N, d_centroid);
    CUDA_CHECK(cudaMemcpy(&h_centroid_tgt, d_centroid, sizeof(float3), cudaMemcpyDeviceToHost));
    h_centroid_tgt.x /= N; h_centroid_tgt.y /= N; h_centroid_tgt.z /= N;

    // Subtract centroids.
    subtractCentroid<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_src, N, h_centroid_src);
    subtractCentroid<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_tgt, N, h_centroid_tgt);

    // Compute covariance matrix.
    float *d_cov;
    CUDA_CHECK(cudaMalloc(&d_cov, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cov, 0, 9 * sizeof(float)));
    computeCovariance<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_src, d_tgt, N, d_cov);

    float h_cov[9];
    CUDA_CHECK(cudaMemcpy(h_cov, d_cov, 9 * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute estimated rotation using Horn's method.
    float R_est[9];
    computeRotationFromCovariance(h_cov, R_est);

    // Compute translation: t_est = centroid_target - R_est * centroid_source.
    float3 R_centroid_src;
    mat3_mul_vec3(R_est, h_centroid_src, R_centroid_src);
    float3 t_est;
    t_est.x = h_centroid_tgt.x - R_centroid_src.x;
    t_est.y = h_centroid_tgt.y - R_centroid_src.y;
    t_est.z = h_centroid_tgt.z - R_centroid_src.z;

    // Apply estimated transformation to source.
    applyTransform<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_src, N, R_est, t_est);

    // Copy aligned source back to host.
    CUDA_CHECK(cudaMemcpy(h_aligned, d_src, size, cudaMemcpyDeviceToHost));
    float rmse = computeRMSE(h_aligned, h_tgt, N);
    printf("Procrustes alignment RMSE: %f\n", rmse);

    // --------- ICP Alignment (Unknown Correspondences) ---------
    // For ICP, we simulate a misalignment by applying a small perturbation
    // to the original source (h_src) and then try to recover the transform.
    float3* d_src_icp;
    CUDA_CHECK(cudaMalloc(&d_src_icp, size));
    // Create a perturbed source on host.
    float3* h_src_icp = (float3*)malloc(size);
    // Perturbation: a slight rotation and translation.
    float anglePerturb = 0.1f; // radians
    float R_perturb[9] = {
        cos(anglePerturb), -sin(anglePerturb), 0,
        sin(anglePerturb),  cos(anglePerturb), 0,
        0,                  0,                 1
    };
    float3 t_perturb = make_float3(0.1f, -0.1f, 0.05f);
    applyTransformationCPU(h_src, h_src_icp, N, R_perturb, t_perturb);
    CUDA_CHECK(cudaMemcpy(d_src_icp, h_src_icp, size, cudaMemcpyHostToDevice));

    // For ICP, we assume d_tgt (target) remains the same.
    // Allocate memory for nearest-neighbor indices and distances.
    int* d_indices;
    float* d_nn_dists;
    CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nn_dists, N * sizeof(float)));
    float3* d_corr;
    CUDA_CHECK(cudaMalloc(&d_corr, size));

    const int max_iters = 10;
    for (int iter = 0; iter < max_iters; iter++) {
        // 1. For each point in d_src_icp, find the nearest neighbor in d_tgt.
        findNearestNeighbor<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_src_icp, d_tgt, N, N, d_indices, d_nn_dists);
        // 2. Gather corresponding target points.
        gatherCorrespondences<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_tgt, d_indices, N, d_corr);

        // 3. Compute centroids for current d_src_icp and gathered correspondences.
        CUDA_CHECK(cudaMemset(d_centroid, 0, sizeof(float3)));
        computeCentroid<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_src_icp, N, d_centroid);
        float3 centroid_src_icp;
        CUDA_CHECK(cudaMemcpy(&centroid_src_icp, d_centroid, sizeof(float3), cudaMemcpyDeviceToHost));
        centroid_src_icp.x /= N; centroid_src_icp.y /= N; centroid_src_icp.z /= N;

        CUDA_CHECK(cudaMemset(d_centroid, 0, sizeof(float3)));
        computeCentroid<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_corr, N, d_centroid);
        float3 centroid_corr;
        CUDA_CHECK(cudaMemcpy(&centroid_corr, d_centroid, sizeof(float3), cudaMemcpyDeviceToHost));
        centroid_corr.x /= N; centroid_corr.y /= N; centroid_corr.z /= N;

        // 4. Subtract centroids.
        subtractCentroid<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_src_icp, N, centroid_src_icp);
        subtractCentroid<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_corr, N, centroid_corr);

        // 5. Compute covariance between d_src_icp and d_corr.
        CUDA_CHECK(cudaMemset(d_cov, 0, 9 * sizeof(float)));
        computeCovariance<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_src_icp, d_corr, N, d_cov);
        CUDA_CHECK(cudaMemcpy(h_cov, d_cov, 9 * sizeof(float), cudaMemcpyDeviceToHost));

        // 6. Compute transformation (rotation and translation).
        float R_icp[9];
        computeRotationFromCovariance(h_cov, R_icp);
        float3 R_centroid;
        mat3_mul_vec3(R_icp, centroid_src_icp, R_centroid);
        float3 t_icp;
        t_icp.x = centroid_corr.x - R_centroid.x;
        t_icp.y = centroid_corr.y - R_centroid.y;
        t_icp.z = centroid_corr.z - R_centroid.z;

        // 7. Apply transformation to d_src_icp.
        applyTransform<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_src_icp, N, R_icp, t_icp);
    }

    // Copy ICP-aligned result back to host.
    float3* h_icp_aligned = (float3*)malloc(size);
    CUDA_CHECK(cudaMemcpy(h_icp_aligned, d_src_icp, size, cudaMemcpyDeviceToHost));
    float rmse_icp = computeRMSE(h_icp_aligned, h_tgt, N);
    printf("ICP alignment RMSE: %f\n", rmse_icp);

    // Clean up.
    free(h_src);
    free(h_tgt);
    free(h_aligned);
    free(h_src_icp);
    free(h_icp_aligned);
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_tgt));
    CUDA_CHECK(cudaFree(d_centroid));
    CUDA_CHECK(cudaFree(d_cov));
    CUDA_CHECK(cudaFree(d_src_icp));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_nn_dists));
    CUDA_CHECK(cudaFree(d_corr));

    return 0;
}
