#pragma once

#include <vector>
#include "tinytensor/core/tensor.hpp"

namespace tinytensor {
namespace ops {

// Elementwise binary operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// Elementwise unary operations
Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);

// Matrix multiplication
Tensor matmul(const Tensor& a, const Tensor& b);

// Reductions
Tensor reduce_sum(const Tensor& a, const std::vector<int64_t>& axes = {}, bool keepdims = false);
Tensor reduce_mean(const Tensor& a, const std::vector<int64_t>& axes = {}, bool keepdims = false);

// Transpose / Permute helpers
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1);
Tensor permute(const Tensor& a, const std::vector<int64_t>& dims);

} // namespace ops
} // namespace tinytensor
