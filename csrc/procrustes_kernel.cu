#include "common.h"

namespace rapidalign {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor> procrustes_stub(
    const at::Tensor& src,
    const at::Tensor& tgt,
    const c10::optional<at::Tensor>& src_batch,
    const c10::optional<at::Tensor>& tgt_batch) {
  TORCH_CHECK(src.is_cuda(), "procrustes_stub expects src on CUDA device");
  TORCH_CHECK(tgt.is_cuda(), "procrustes_stub expects tgt on CUDA device");
  // Placeholder implementation: fall back to CPU reference by moving tensors.
  auto src_cpu = src.to(at::kCPU);
  auto tgt_cpu = tgt.to(at::kCPU);
  auto result = torch::zeros_like(src_cpu);
  auto rotations = torch::zeros({1, 3, 3}, src_cpu.options());
  auto translations = torch::zeros({1, 3}, src_cpu.options());
  TORCH_CHECK(false, "CUDA procrustes kernel not yet implemented");
  return std::make_tuple(result.to(src.device()), rotations.to(src.device()), translations.to(src.device()));
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> procrustes_align_cuda(
    const at::Tensor& src,
    const at::Tensor& tgt,
    const c10::optional<at::Tensor>& src_batch,
    const c10::optional<at::Tensor>& tgt_batch) {
  return procrustes_stub(src, tgt, src_batch, tgt_batch);
}

} // namespace rapidalign
