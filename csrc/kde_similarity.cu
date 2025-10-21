#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace rapidalign {

namespace {

__device__ inline double gaussian_kernel(const float* a, const float* b, int dim, double inv_two_sigma2) {
    double r2 = 0.0;
    for (int d = 0; d < dim; ++d) {
        double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
        r2 += diff * diff;
    }
    return exp(-r2 * inv_two_sigma2);
}

__global__ void self_kernel_accumulate(
    const float* __restrict__ pts,
    const float* __restrict__ weights,
    const uint8_t* __restrict__ mask,
    int B,
    int N,
    int dim,
    double inv_two_sigma2,
    double* __restrict__ out)
{
    int b = blockIdx.z;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= B || i >= N || j >= N) {
        return;
    }

    if (!mask[b * N + i] || !mask[b * N + j]) {
        return;
    }

    const float* pi = pts + ((static_cast<size_t>(b) * N + i) * dim);
    const float* pj = pts + ((static_cast<size_t>(b) * N + j) * dim);
    double k = gaussian_kernel(pi, pj, dim, inv_two_sigma2);
    double contrib = weights[b * N + i] * weights[b * N + j] * k;
    atomicAdd(out + b, contrib);
}

__global__ void cross_kernel_accumulate(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ w_x,
    const float* __restrict__ w_y,
    const uint8_t* __restrict__ mask_x,
    const uint8_t* __restrict__ mask_y,
    int B,
    int Nx,
    int Ny,
    int dim,
    double inv_two_sigma2,
    double* __restrict__ out)
{
    int b = blockIdx.z;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= B || ix >= Nx || iy >= Ny) {
        return;
    }

    if (!mask_x[b * Nx + ix] || !mask_y[b * Ny + iy]) {
        return;
    }

    const float* px = x + ((static_cast<size_t>(b) * Nx + ix) * dim);
    const float* py = y + ((static_cast<size_t>(b) * Ny + iy) * dim);
    double k = gaussian_kernel(px, py, dim, inv_two_sigma2);
    double contrib = w_x[b * Nx + ix] * w_y[b * Ny + iy] * k;
    atomicAdd(out + b, contrib);
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> kde_mmd_forward_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor w_x,
    torch::Tensor w_y,
    torch::Tensor mask_x,
    torch::Tensor mask_y,
    double sigma)
{
    TORCH_CHECK(x.is_cuda(), "kde_mmd_forward_cuda: x must be CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "kde_mmd_forward_cuda: y must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "kde_mmd_forward_cuda: x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "kde_mmd_forward_cuda: y must be float32");
    TORCH_CHECK(x.dim() == 3 && y.dim() == 3 && x.size(2) == y.size(2),
                "x and y must have shape (B, N, D)");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    x = x.contiguous();
    y = y.contiguous();

    int64_t B = x.size(0);
    int64_t Nx = x.size(1);
    int64_t Ny = y.size(1);
    int64_t dim = x.size(2);

    TORCH_CHECK(w_x.dim() == 2 && w_x.size(0) == B && w_x.size(1) == Nx,
                "w_x must have shape (B, Nx)");
    TORCH_CHECK(w_y.dim() == 2 && w_y.size(0) == B && w_y.size(1) == Ny,
                "w_y must have shape (B, Ny)");
    TORCH_CHECK(w_x.scalar_type() == torch::kFloat32,
                "kde_mmd_forward_segmented_cuda: w_x must be float32");
    TORCH_CHECK(w_y.scalar_type() == torch::kFloat32,
                "kde_mmd_forward_segmented_cuda: w_y must be float32");
    TORCH_CHECK(mask_x.dim() == 2 && mask_x.size(0) == B && mask_x.size(1) == Nx,
                "mask_x must have shape (B, Nx)");
    TORCH_CHECK(mask_y.dim() == 2 && mask_y.size(0) == B && mask_y.size(1) == Ny,
                "mask_y must have shape (B, Ny)");

    auto w_x_f = w_x.contiguous();
    auto w_y_f = w_y.contiguous();
    auto mask_x_b = mask_x.to(torch::kUInt8).contiguous();
    auto mask_y_b = mask_y.to(torch::kUInt8).contiguous();

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(x.device());

    torch::Tensor Kxx = torch::zeros({B}, options);
    torch::Tensor Kyy = torch::zeros({B}, options);
    torch::Tensor Kxy = torch::zeros({B}, options);

    double inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);

    dim3 threads(16, 16, 1);
    dim3 blocks_self(
        (Nx + threads.x - 1) / threads.x,
        (Nx + threads.y - 1) / threads.y,
        static_cast<unsigned int>(B));

    dim3 blocks_self_y(
        (Ny + threads.x - 1) / threads.x,
        (Ny + threads.y - 1) / threads.y,
        static_cast<unsigned int>(B));

    dim3 blocks_cross(
        (Nx + threads.x - 1) / threads.x,
        (Ny + threads.y - 1) / threads.y,
        static_cast<unsigned int>(B));

    auto stream = at::cuda::getCurrentCUDAStream();

    self_kernel_accumulate<<<blocks_self, threads, 0, stream>>>(
        x.data_ptr<float>(),
        w_x_f.data_ptr<float>(),
        mask_x_b.data_ptr<uint8_t>(),
        static_cast<int>(B),
        static_cast<int>(Nx),
        static_cast<int>(dim),
        inv_two_sigma2,
        Kxx.data_ptr<double>());

    self_kernel_accumulate<<<blocks_self_y, threads, 0, stream>>>(
        y.data_ptr<float>(),
        w_y_f.data_ptr<float>(),
        mask_y_b.data_ptr<uint8_t>(),
        static_cast<int>(B),
        static_cast<int>(Ny),
        static_cast<int>(dim),
        inv_two_sigma2,
        Kyy.data_ptr<double>());

    cross_kernel_accumulate<<<blocks_cross, threads, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        w_x_f.data_ptr<float>(),
        w_y_f.data_ptr<float>(),
        mask_x_b.data_ptr<uint8_t>(),
        mask_y_b.data_ptr<uint8_t>(),
        static_cast<int>(B),
        static_cast<int>(Nx),
        static_cast<int>(Ny),
        static_cast<int>(dim),
        inv_two_sigma2,
        Kxy.data_ptr<double>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(Kxx, Kyy, Kxy);
}

namespace {

__global__ void self_kernel_segmented(
    const float* __restrict__ pts,
    const float* __restrict__ weights,
    const int64_t* __restrict__ ptr,
    int num_pairs,
    int dim,
    double inv_two_sigma2,
    double* __restrict__ out)
{
    int b = blockIdx.x;
    if (b >= num_pairs) {
        return;
    }

    int64_t start = ptr[b];
    int64_t end = ptr[b + 1];
    int64_t count = end - start;
    if (count <= 0) {
        return;
    }

    double accum = 0.0;
    int64_t total = count * count;
    for (int64_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int64_t i = idx / count;
        int64_t j = idx % count;
        const float* pi = pts + ((start + i) * dim);
        const float* pj = pts + ((start + j) * dim);
        double wi = static_cast<double>(weights[start + i]);
        double wj = static_cast<double>(weights[start + j]);
        double k = gaussian_kernel(pi, pj, dim, inv_two_sigma2);
        accum += wi * wj * k;
    }

    if (accum != 0.0) {
        atomicAdd(out + b, accum);
    }
}

__global__ void cross_kernel_segmented(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ w_x,
    const float* __restrict__ w_y,
    const int64_t* __restrict__ ptr_x,
    const int64_t* __restrict__ ptr_y,
    int num_pairs,
    int dim,
    double inv_two_sigma2,
    double* __restrict__ out)
{
    int b = blockIdx.x;
    if (b >= num_pairs) {
        return;
    }

    int64_t start_x = ptr_x[b];
    int64_t end_x = ptr_x[b + 1];
    int64_t start_y = ptr_y[b];
    int64_t end_y = ptr_y[b + 1];
    int64_t count_x = end_x - start_x;
    int64_t count_y = end_y - start_y;

    if (count_x <= 0 || count_y <= 0) {
        return;
    }

    double accum = 0.0;
    int64_t total = count_x * count_y;
    for (int64_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int64_t i = idx / count_y;
        int64_t j = idx % count_y;
        const float* px = x + ((start_x + i) * dim);
        const float* py = y + ((start_y + j) * dim);
        double wi = static_cast<double>(w_x[start_x + i]);
        double wj = static_cast<double>(w_y[start_y + j]);
        double k = gaussian_kernel(px, py, dim, inv_two_sigma2);
        accum += wi * wj * k;
    }

    if (accum != 0.0) {
        atomicAdd(out + b, accum);
    }
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> kde_mmd_forward_segmented_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor w_x,
    torch::Tensor w_y,
    torch::Tensor ptr_x,
    torch::Tensor ptr_y,
    double sigma)
{
    TORCH_CHECK(x.is_cuda(), "kde_mmd_forward_segmented_cuda: x must be CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "kde_mmd_forward_segmented_cuda: y must be CUDA tensor");
    TORCH_CHECK(ptr_x.is_cuda(), "kde_mmd_forward_segmented_cuda: ptr_x must be CUDA tensor");
    TORCH_CHECK(ptr_y.is_cuda(), "kde_mmd_forward_segmented_cuda: ptr_y must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "kde_mmd_forward_segmented_cuda: x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32,
                "kde_mmd_forward_segmented_cuda: y must be float32");
    TORCH_CHECK(w_x.scalar_type() == torch::kFloat32,
                "kde_mmd_forward_segmented_cuda: w_x must be float32");
    TORCH_CHECK(w_y.scalar_type() == torch::kFloat32,
                "kde_mmd_forward_segmented_cuda: w_y must be float32");
    TORCH_CHECK(ptr_x.scalar_type() == torch::kInt64,
                "kde_mmd_forward_segmented_cuda: ptr_x must be int64");
    TORCH_CHECK(ptr_y.scalar_type() == torch::kInt64,
                "kde_mmd_forward_segmented_cuda: ptr_y must be int64");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == y.size(1),
                "x and y must have shape (N, D) and (M, D)");
    TORCH_CHECK(ptr_x.dim() == 1 && ptr_y.dim() == 1,
                "ptr_x and ptr_y must be 1-D");
    TORCH_CHECK(ptr_x.numel() >= 1 && ptr_y.numel() >= 1,
                "ptr tensors must contain at least one element");
    TORCH_CHECK(ptr_x.numel() == ptr_y.numel(),
                "ptr_x and ptr_y must have the same length");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();
    auto w_x_contig = w_x.contiguous();
    auto w_y_contig = w_y.contiguous();
    auto ptr_x_contig = ptr_x.contiguous();
    auto ptr_y_contig = ptr_y.contiguous();

    int64_t num_pairs = ptr_x_contig.size(0) - 1;
    TORCH_CHECK(num_pairs >= 0, "invalid ptr length");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(x.device());

    torch::Tensor Kxx = torch::zeros({num_pairs}, options);
    torch::Tensor Kyy = torch::zeros({num_pairs}, options);
    torch::Tensor Kxy = torch::zeros({num_pairs}, options);

    if (num_pairs == 0) {
        return std::make_tuple(Kxx, Kyy, Kxy);
    }

    double inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);
    int dim = static_cast<int>(x_contig.size(1));

    auto stream = at::cuda::getCurrentCUDAStream();

    int threads = 256;
    dim3 blocks(static_cast<unsigned int>(num_pairs));

    self_kernel_segmented<<<blocks, threads, 0, stream>>>(
        x_contig.data_ptr<float>(),
        w_x_contig.data_ptr<float>(),
        ptr_x_contig.data_ptr<int64_t>(),
        static_cast<int>(num_pairs),
        dim,
        inv_two_sigma2,
        Kxx.data_ptr<double>());

    self_kernel_segmented<<<blocks, threads, 0, stream>>>(
        y_contig.data_ptr<float>(),
        w_y_contig.data_ptr<float>(),
        ptr_y_contig.data_ptr<int64_t>(),
        static_cast<int>(num_pairs),
        dim,
        inv_two_sigma2,
        Kyy.data_ptr<double>());

    cross_kernel_segmented<<<blocks, threads, 0, stream>>>(
        x_contig.data_ptr<float>(),
        y_contig.data_ptr<float>(),
        w_x_contig.data_ptr<float>(),
        w_y_contig.data_ptr<float>(),
        ptr_x_contig.data_ptr<int64_t>(),
        ptr_y_contig.data_ptr<int64_t>(),
        static_cast<int>(num_pairs),
        dim,
        inv_two_sigma2,
        Kxy.data_ptr<double>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(Kxx, Kyy, Kxy);
}

namespace {

__global__ void kde_backward_sources_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ w_x,
    const float* __restrict__ w_y,
    const int64_t* __restrict__ ptr_x,
    const int64_t* __restrict__ ptr_y,
    int dim,
    double sigma,
    double inv_sigma2,
    double inv_two_sigma2,
    const double* __restrict__ grad_loss,
    float* __restrict__ grad_x,
    double* __restrict__ grad_wx,
    double* __restrict__ sigma_xx,
    double* __restrict__ sigma_xy)
{
    int pair = blockIdx.x;
    int64_t start_x = ptr_x[pair];
    int64_t end_x = ptr_x[pair + 1];
    int64_t start_y = ptr_y[pair];
    int64_t end_y = ptr_y[pair + 1];
    int Nx = static_cast<int>(end_x - start_x);
    int Ny = static_cast<int>(end_y - start_y);

    if (Nx <= 0) {
        return;
    }

    double gl = grad_loss[pair];

    for (int idx = threadIdx.x; idx < Nx; idx += blockDim.x) {
        int64_t offset_i = start_x + idx;
        const float* xi = x + offset_i * dim;
        double wi = static_cast<double>(w_x[offset_i]);

        double grad_self[3] = {0.0, 0.0, 0.0};
        double grad_cross[3] = {0.0, 0.0, 0.0};
        double sum_wxx = 0.0;
        double sum_wxy = 0.0;
        double sigma_xx_local = 0.0;
        double sigma_xy_local = 0.0;

        for (int j = 0; j < Nx; ++j) {
            int64_t offset_j = start_x + j;
            const float* xj = x + offset_j * dim;
            double wj = static_cast<double>(w_x[offset_j]);

            double dist2 = 0.0;
            double diff[3] = {0.0, 0.0, 0.0};
            for (int d = 0; d < dim; ++d) {
                double delta = static_cast<double>(xi[d]) - static_cast<double>(xj[d]);
                diff[d] = delta;
                dist2 += delta * delta;
            }

            double G = exp(-dist2 * inv_two_sigma2);
            double common = wi * wj * G;
            for (int d = 0; d < dim; ++d) {
                grad_self[d] += common * diff[d];
            }
            sigma_xx_local += common * dist2;
            sum_wxx += wj * G;
        }

        for (int j = 0; j < Ny; ++j) {
            int64_t offset_j = start_y + j;
            const float* yj = y + offset_j * dim;
            double vj = static_cast<double>(w_y[offset_j]);

            double dist2 = 0.0;
            double diff[3] = {0.0, 0.0, 0.0};
            for (int d = 0; d < dim; ++d) {
                double delta = static_cast<double>(xi[d]) - static_cast<double>(yj[d]);
                diff[d] = delta;
                dist2 += delta * delta;
            }

            double G = exp(-dist2 * inv_two_sigma2);
            double common = wi * vj * G;
            for (int d = 0; d < dim; ++d) {
                grad_cross[d] += common * diff[d];
            }
            sigma_xy_local += common * dist2;
            sum_wxy += vj * G;
        }

        double scale = 2.0 * inv_sigma2;
        float* out_x = grad_x + offset_i * dim;
        for (int d = 0; d < dim; ++d) {
            double grad_val = (-scale) * grad_self[d] + (scale) * grad_cross[d];
            out_x[d] = static_cast<float>(gl * grad_val);
        }

        grad_wx[offset_i] = gl * (2.0 * sum_wxx - 2.0 * sum_wxy);

        if (sigma_xx_local != 0.0) {
            atomicAdd(sigma_xx + pair, gl * sigma_xx_local);
        }
        if (sigma_xy_local != 0.0) {
            atomicAdd(sigma_xy + pair, gl * sigma_xy_local);
        }
    }
}

__global__ void kde_backward_targets_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ w_x,
    const float* __restrict__ w_y,
    const int64_t* __restrict__ ptr_x,
    const int64_t* __restrict__ ptr_y,
    int dim,
    double sigma,
    double inv_sigma2,
    double inv_two_sigma2,
    const double* __restrict__ grad_loss,
    float* __restrict__ grad_y,
    double* __restrict__ grad_wy,
    double* __restrict__ sigma_yy)
{
    int pair = blockIdx.x;
    int64_t start_x = ptr_x[pair];
    int64_t end_x = ptr_x[pair + 1];
    int64_t start_y = ptr_y[pair];
    int64_t end_y = ptr_y[pair + 1];
    int Nx = static_cast<int>(end_x - start_x);
    int Ny = static_cast<int>(end_y - start_y);

    if (Ny <= 0) {
        return;
    }

    double gl = grad_loss[pair];

    for (int idx = threadIdx.x; idx < Ny; idx += blockDim.x) {
        int64_t offset_j = start_y + idx;
        const float* yj = y + offset_j * dim;
        double vj = static_cast<double>(w_y[offset_j]);

        double grad_self[3] = {0.0, 0.0, 0.0};
        double grad_cross[3] = {0.0, 0.0, 0.0};
        double sum_wyy = 0.0;
        double sum_wxy = 0.0;
        double sigma_yy_local = 0.0;

        for (int k = 0; k < Ny; ++k) {
            int64_t offset_k = start_y + k;
            const float* yk = y + offset_k * dim;
            double vk = static_cast<double>(w_y[offset_k]);

            double dist2 = 0.0;
            double diff[3] = {0.0, 0.0, 0.0};
            for (int d = 0; d < dim; ++d) {
                double delta = static_cast<double>(yj[d]) - static_cast<double>(yk[d]);
                diff[d] = delta;
                dist2 += delta * delta;
            }

            double G = exp(-dist2 * inv_two_sigma2);
            double common = vj * vk * G;
            for (int d = 0; d < dim; ++d) {
                grad_self[d] += common * diff[d];
            }
            sigma_yy_local += common * dist2;
            sum_wyy += vk * G;
        }

        for (int i = 0; i < Nx; ++i) {
            int64_t offset_i = start_x + i;
            const float* xi = x + offset_i * dim;
            double wi = static_cast<double>(w_x[offset_i]);

            double dist2 = 0.0;
            double diff[3] = {0.0, 0.0, 0.0};
            for (int d = 0; d < dim; ++d) {
                double delta = static_cast<double>(yj[d]) - static_cast<double>(xi[d]);
                diff[d] = delta;
                dist2 += delta * delta;
            }

            double G = exp(-dist2 * inv_two_sigma2);
            double common = vj * wi * G;
            for (int d = 0; d < dim; ++d) {
                grad_cross[d] += common * diff[d];
            }
            sum_wxy += wi * G;
        }

        double scale = 2.0 * inv_sigma2;
        float* out_y = grad_y + offset_j * dim;
        for (int d = 0; d < dim; ++d) {
            double grad_val = (-scale) * grad_self[d] + (scale) * grad_cross[d];
            out_y[d] = static_cast<float>(gl * grad_val);
        }

        grad_wy[offset_j] = gl * (2.0 * sum_wyy - 2.0 * sum_wxy);

        if (sigma_yy_local != 0.0) {
            atomicAdd(sigma_yy + pair, gl * sigma_yy_local);
        }
    }
}

__global__ void finalize_sigma_kernel(
    int num_pairs,
    double inv_sigma3,
    const double* __restrict__ sigma_xx,
    const double* __restrict__ sigma_yy,
    const double* __restrict__ sigma_xy,
    double* __restrict__ grad_sigma)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) {
        return;
    }
    double value = (sigma_xx[idx] + sigma_yy[idx] - 2.0 * sigma_xy[idx]) * inv_sigma3;
    grad_sigma[idx] = value;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
kde_mmd_backward_segmented_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor w_x,
    torch::Tensor w_y,
    torch::Tensor ptr_x,
    torch::Tensor ptr_y,
    torch::Tensor grad_loss,
    double sigma)
{
    TORCH_CHECK(x.is_cuda(), "kde_mmd_backward_segmented_cuda: x must be CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "kde_mmd_backward_segmented_cuda: y must be CUDA tensor");
    TORCH_CHECK(w_x.is_cuda() && w_y.is_cuda(), "weights must be CUDA tensors");
    TORCH_CHECK(ptr_x.is_cuda() && ptr_y.is_cuda(), "ptr tensors must be CUDA tensors");
    TORCH_CHECK(grad_loss.is_cuda(), "grad_loss must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(w_x.scalar_type() == torch::kFloat32, "w_x must be float32");
    TORCH_CHECK(w_y.scalar_type() == torch::kFloat32, "w_y must be float32");
    TORCH_CHECK(grad_loss.scalar_type() == torch::kFloat64, "grad_loss must be float64");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    x = x.contiguous();
    y = y.contiguous();
    w_x = w_x.contiguous();
    w_y = w_y.contiguous();
    ptr_x = ptr_x.contiguous();
    ptr_y = ptr_y.contiguous();
    grad_loss = grad_loss.contiguous();

    int num_pairs = static_cast<int>(ptr_x.size(0) - 1);
    int dim = static_cast<int>(x.size(1));

    auto options_x = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto options_w = torch::TensorOptions().dtype(torch::kFloat64).device(x.device());

    torch::Tensor grad_x = torch::zeros_like(x);
    torch::Tensor grad_y = torch::zeros_like(y);
    torch::Tensor grad_wx = torch::zeros({w_x.size(0)}, options_w);
    torch::Tensor grad_wy = torch::zeros({w_y.size(0)}, options_w);
    torch::Tensor grad_sigma_pairs = torch::zeros({num_pairs}, options_w);

    torch::Tensor sigma_xx = torch::zeros({num_pairs}, options_w);
    torch::Tensor sigma_yy = torch::zeros({num_pairs}, options_w);
    torch::Tensor sigma_xy = torch::zeros({num_pairs}, options_w);

    double inv_sigma2 = 1.0 / (sigma * sigma);
    double inv_two_sigma2 = 0.5 * inv_sigma2;
    double inv_sigma3 = 1.0 / (sigma * sigma * sigma);

    auto stream = at::cuda::getCurrentCUDAStream();

    int threads = 256;
    dim3 blocks(static_cast<unsigned int>(num_pairs));

    auto w_x_f = w_x.contiguous();
    auto w_y_f = w_y.contiguous();

    kde_backward_sources_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        w_x_f.data_ptr<float>(),
        w_y_f.data_ptr<float>(),
        ptr_x.data_ptr<int64_t>(),
        ptr_y.data_ptr<int64_t>(),
        dim,
        sigma,
        inv_sigma2,
        inv_two_sigma2,
        grad_loss.data_ptr<double>(),
        grad_x.data_ptr<float>(),
        grad_wx.data_ptr<double>(),
        sigma_xx.data_ptr<double>(),
        sigma_xy.data_ptr<double>());

    kde_backward_targets_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        w_x_f.data_ptr<float>(),
        w_y_f.data_ptr<float>(),
        ptr_x.data_ptr<int64_t>(),
        ptr_y.data_ptr<int64_t>(),
        dim,
        sigma,
        inv_sigma2,
        inv_two_sigma2,
        grad_loss.data_ptr<double>(),
        grad_y.data_ptr<float>(),
        grad_wy.data_ptr<double>(),
        sigma_yy.data_ptr<double>());

    int threads_finalize = 128;
    int blocks_finalize = (num_pairs + threads_finalize - 1) / threads_finalize;
    finalize_sigma_kernel<<<blocks_finalize, threads_finalize, 0, stream>>>(
        num_pairs,
        inv_sigma3,
        sigma_xx.data_ptr<double>(),
        sigma_yy.data_ptr<double>(),
        sigma_xy.data_ptr<double>(),
        grad_sigma_pairs.data_ptr<double>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(grad_x, grad_y, grad_wx, grad_wy, grad_sigma_pairs);
}

} // namespace rapidalign
