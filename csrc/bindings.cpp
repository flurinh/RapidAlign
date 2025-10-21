#include "common.h"

namespace rapidalign {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> kde_mmd_forward_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor w_x,
    torch::Tensor w_y,
    torch::Tensor mask_x,
    torch::Tensor mask_y,
    double sigma);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> kde_mmd_forward_segmented_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor w_x,
    torch::Tensor w_y,
    torch::Tensor ptr_x,
    torch::Tensor ptr_y,
    double sigma);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
kde_mmd_backward_segmented_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor w_x,
    torch::Tensor w_y,
    torch::Tensor ptr_x,
    torch::Tensor ptr_y,
    torch::Tensor grad_loss,
    double sigma);
} // namespace rapidalign

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "kde_mmd_forward",
      &rapidalign::kde_mmd_forward_cuda,
      "Batched KDE/MMD forward (CUDA)",
      py::arg("x"),
      py::arg("y"),
      py::arg("w_x"),
      py::arg("w_y"),
      py::arg("mask_x"),
      py::arg("mask_y"),
      py::arg("sigma"));

  m.def(
      "kde_mmd_forward_segmented",
      &rapidalign::kde_mmd_forward_segmented_cuda,
      "Segmented KDE/MMD forward (CUDA)",
      py::arg("x"),
      py::arg("y"),
      py::arg("w_x"),
      py::arg("w_y"),
      py::arg("ptr_x"),
      py::arg("ptr_y"),
      py::arg("sigma"));

  m.def(
      "kde_mmd_backward_segmented",
      &rapidalign::kde_mmd_backward_segmented_cuda,
      "Segmented KDE/MMD backward (CUDA)",
      py::arg("x"),
      py::arg("y"),
      py::arg("w_x"),
      py::arg("w_y"),
      py::arg("ptr_x"),
      py::arg("ptr_y"),
      py::arg("grad_loss"),
      py::arg("sigma"));
}
