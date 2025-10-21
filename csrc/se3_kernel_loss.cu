#include "common.h"

namespace rapidalign {

std::tuple<at::Tensor, at::Tensor, at::Tensor> se3_kernel_loss_cuda(
    const at::Tensor& src,
    const at::Tensor& tgt,
    const c10::optional<at::Tensor>& src_batch,
    const c10::optional<at::Tensor>& tgt_batch,
    double sigma,
    int64_t iterations) {
  TORCH_CHECK(src.is_cuda(), "se3_kernel_loss expects src on CUDA device");
  TORCH_CHECK(tgt.is_cuda(), "se3_kernel_loss expects tgt on CUDA device");
  TORCH_CHECK(false, "CUDA SE3 kernel loss not yet implemented");
  auto loss = at::zeros({1}, src.options());
  auto rotations = at::zeros({1, 3, 3}, src.options());
  auto translations = at::zeros({1, 3}, src.options());
  return std::make_tuple(loss, rotations, translations);
}

} // namespace rapidalign
