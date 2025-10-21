#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>

namespace rapidalign {

inline void check_device(const at::Tensor& tensor, c10::DeviceType expected) {
    TORCH_CHECK(tensor.device().type() == expected,
                "Expected tensor on ", c10::DeviceTypeName(expected),
                " but got ", tensor.device());
}

inline void check_dim(const at::Tensor& tensor, int64_t dim, const char* name) {
    TORCH_CHECK(tensor.dim() == dim,
                name, " must have dimension ", dim,
                ". Got ", tensor.dim());
}

inline void check_last_dim(const at::Tensor& tensor, int64_t dim, const char* name) {
    TORCH_CHECK(tensor.size(-1) == dim,
                name, " must have last dimension size ", dim,
                ". Got ", tensor.size(-1));
}

} // namespace rapidalign
