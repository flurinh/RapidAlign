#include <torch/extension.h>

// Forward declarations for CUDA functions
void procrustes_align_cuda(
    const torch::Tensor& src_points,
    const torch::Tensor& tgt_points,
    const torch::Tensor& src_batch_idx,
    const torch::Tensor& tgt_batch_idx,
    torch::Tensor& aligned_points,
    torch::Tensor& rotations,
    torch::Tensor& translations
);

void icp_align_cuda(
    const torch::Tensor& src_points,
    const torch::Tensor& tgt_points,
    const torch::Tensor& src_batch_idx,
    const torch::Tensor& tgt_batch_idx,
    torch::Tensor& aligned_points,
    torch::Tensor& rotations,
    torch::Tensor& translations,
    int max_iterations,
    float convergence_threshold,
    bool use_grid_acceleration,
    bool use_cuda_streams,
    float grid_cell_size
);

void chamfer_distance_cuda(
    const torch::Tensor& src_points,
    const torch::Tensor& tgt_points,
    const torch::Tensor& src_batch_idx,
    const torch::Tensor& tgt_batch_idx,
    torch::Tensor& distances,
    bool use_grid_acceleration,
    float grid_cell_size
);

// Wrapper for Procrustes alignment
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> procrustes_align(
    torch::Tensor src_points,
    torch::Tensor tgt_points,
    torch::Tensor src_batch_idx,
    torch::Tensor tgt_batch_idx
) {
    // Check inputs
    TORCH_CHECK(src_points.dim() == 2 && src_points.size(1) == 3,
               "Source points must have shape [N, 3]");
    TORCH_CHECK(tgt_points.dim() == 2 && tgt_points.size(1) == 3,
               "Target points must have shape [M, 3]");
    TORCH_CHECK(src_batch_idx.dim() == 1 && src_batch_idx.size(0) == src_points.size(0),
               "Source batch indices must have shape [N]");
    TORCH_CHECK(tgt_batch_idx.dim() == 1 && tgt_batch_idx.size(0) == tgt_points.size(0),
               "Target batch indices must have shape [M]");
    
    // Determine number of batches
    int batch_size = src_batch_idx.max().item<int>() + 1;
    
    // Allocate outputs
    auto aligned_points = torch::zeros_like(src_points);
    auto rotations = torch::zeros({batch_size, 3, 3}, src_points.options());
    auto translations = torch::zeros({batch_size, 3}, src_points.options());
    
    // Run CUDA implementation
    procrustes_align_cuda(
        src_points, tgt_points, src_batch_idx, tgt_batch_idx,
        aligned_points, rotations, translations
    );
    
    return std::make_tuple(aligned_points, rotations, translations);
}

// Wrapper for ICP alignment
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> icp_align(
    torch::Tensor src_points,
    torch::Tensor tgt_points,
    torch::Tensor src_batch_idx,
    torch::Tensor tgt_batch_idx,
    torch::Tensor aligned_points,
    torch::Tensor rotations,
    torch::Tensor translations,
    int max_iterations,
    float convergence_threshold,
    bool use_grid_acceleration,
    bool use_cuda_streams,
    float grid_cell_size
) {
    // Check inputs
    TORCH_CHECK(src_points.dim() == 2 && src_points.size(1) == 3,
               "Source points must have shape [N, 3]");
    TORCH_CHECK(tgt_points.dim() == 2 && tgt_points.size(1) == 3,
               "Target points must have shape [M, 3]");
    TORCH_CHECK(src_batch_idx.dim() == 1 && src_batch_idx.size(0) == src_points.size(0),
               "Source batch indices must have shape [N]");
    TORCH_CHECK(tgt_batch_idx.dim() == 1 && tgt_batch_idx.size(0) == tgt_points.size(0),
               "Target batch indices must have shape [M]");
    
    // Run CUDA implementation with optimization parameters
    icp_align_cuda(
        src_points, tgt_points, src_batch_idx, tgt_batch_idx,
        aligned_points, rotations, translations,
        max_iterations, convergence_threshold,
        use_grid_acceleration, use_cuda_streams, grid_cell_size
    );
    
    return std::make_tuple(aligned_points, rotations, translations);
}

// Wrapper for Chamfer distance
torch::Tensor chamfer_distance(
    torch::Tensor src_points,
    torch::Tensor tgt_points,
    torch::Tensor src_batch_idx,
    torch::Tensor tgt_batch_idx,
    torch::Tensor distances,
    bool use_grid_acceleration,
    float grid_cell_size
) {
    // Check inputs
    TORCH_CHECK(src_points.dim() == 2 && src_points.size(1) == 3,
               "Source points must have shape [N, 3]");
    TORCH_CHECK(tgt_points.dim() == 2 && tgt_points.size(1) == 3,
               "Target points must have shape [M, 3]");
    TORCH_CHECK(src_batch_idx.dim() == 1 && src_batch_idx.size(0) == src_points.size(0),
               "Source batch indices must have shape [N]");
    TORCH_CHECK(tgt_batch_idx.dim() == 1 && tgt_batch_idx.size(0) == tgt_points.size(0),
               "Target batch indices must have shape [M]");
    
    // Run CUDA implementation with optimization parameters
    chamfer_distance_cuda(
        src_points, tgt_points, src_batch_idx, tgt_batch_idx,
        distances, use_grid_acceleration, grid_cell_size
    );
    
    return distances;
}

// Define Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("procrustes_align", &procrustes_align, "Batched Procrustes alignment (CUDA)",
          py::arg("src_points"), py::arg("tgt_points"), 
          py::arg("src_batch_idx"), py::arg("tgt_batch_idx"));
    
    m.def("icp_align", &icp_align, "Batched ICP alignment with optimization options (CUDA)",
          py::arg("src_points"), py::arg("tgt_points"), 
          py::arg("src_batch_idx"), py::arg("tgt_batch_idx"),
          py::arg("aligned_points"), py::arg("rotations"), py::arg("translations"),
          py::arg("max_iterations") = 20, py::arg("convergence_threshold") = 1e-6,
          py::arg("use_grid_acceleration") = true, py::arg("use_cuda_streams") = true,
          py::arg("grid_cell_size") = 0.2f);
    
    m.def("chamfer_distance", &chamfer_distance, "Batched Chamfer distance with grid acceleration (CUDA)",
          py::arg("src_points"), py::arg("tgt_points"), 
          py::arg("src_batch_idx"), py::arg("tgt_batch_idx"),
          py::arg("distances"), py::arg("use_grid_acceleration") = true,
          py::arg("grid_cell_size") = 0.2f);
}