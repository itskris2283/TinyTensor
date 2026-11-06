#include "tinytensor/core/operators.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {
namespace ops {

namespace {

// Generic binary elementwise operator
template <typename FFloat, typename FDouble>
Tensor elementwise_binary_op(const Tensor& a, const Tensor& b,
                             FFloat op_f, FDouble op_d) {
    if (a.dtype() != b.dtype()) {
        throw TypeError("DType mismatch between operands: " +
                        dtype_to_string(a.dtype()) + " vs " +
                        dtype_to_string(b.dtype()));
    }

    Shape out_shape = Tensor::broadcast_shapes(a.shape(), b.shape());
    Tensor b_a = (a.shape() == out_shape) ? a : a.broadcast_to(out_shape);
    Tensor b_b = (b.shape() == out_shape) ? b : b.broadcast_to(out_shape);

    Tensor out(out_shape, a.dtype());
    int64_t n = out.numel();
    if (n == 0) return out;

    // Fast contiguous path (when both are contiguous and non-broadcasted)
    if (b_a.is_contiguous() && b_b.is_contiguous() &&
        b_a.storage_offset() == 0 && b_b.storage_offset() == 0) {
        if (a.dtype() == DType::Float32) {
            const float* pa = b_a.data<float>();
            const float* pb = b_b.data<float>();
            float* po = out.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                po[i] = op_f(pa[i], pb[i]);
            }
            return out;
        } else {
            const double* pa = b_a.data<double>();
            const double* pb = b_b.data<double>();
            double* po = out.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                po[i] = op_d(pa[i], pb[i]);
            }
            return out;
        }
    }

    // Strided / broadcast coordinate path
    size_t rank = out_shape.ndim();
    std::vector<int64_t> coords(rank, 0);

    const int64_t* strides_a = b_a.strides().strides().data();
    const int64_t* strides_b = b_b.strides().strides().data();
    int64_t off_a_base = b_a.storage_offset();
    int64_t off_b_base = b_b.storage_offset();

    if (a.dtype() == DType::Float32) {
        const float* raw_a = static_cast<const float*>(b_a.storage()->data());
        const float* raw_b = static_cast<const float*>(b_b.storage()->data());
        float* po = out.data<float>();

        for (int64_t i = 0; i < n; ++i) {
            int64_t off_a = off_a_base;
            int64_t off_b = off_b_base;
            for (size_t d = 0; d < rank; ++d) {
                off_a += coords[d] * strides_a[d];
                off_b += coords[d] * strides_b[d];
            }
            po[i] = op_f(raw_a[off_a], raw_b[off_b]);

            for (int64_t d = static_cast<int64_t>(rank) - 1; d >= 0; --d) {
                coords[static_cast<size_t>(d)]++;
                if (coords[static_cast<size_t>(d)] < out_shape[static_cast<size_t>(d)]) {
                    break;
                }
                coords[static_cast<size_t>(d)] = 0;
            }
        }
    } else {
        const double* raw_a = static_cast<const double*>(b_a.storage()->data());
        const double* raw_b = static_cast<const double*>(b_b.storage()->data());
        double* po = out.data<double>();

        for (int64_t i = 0; i < n; ++i) {
            int64_t off_a = off_a_base;
            int64_t off_b = off_b_base;
            for (size_t d = 0; d < rank; ++d) {
                off_a += coords[d] * strides_a[d];
                off_b += coords[d] * strides_b[d];
            }
            po[i] = op_d(raw_a[off_a], raw_b[off_b]);

            for (int64_t d = static_cast<int64_t>(rank) - 1; d >= 0; --d) {
                coords[static_cast<size_t>(d)]++;
                if (coords[static_cast<size_t>(d)] < out_shape[static_cast<size_t>(d)]) {
                    break;
                }
                coords[static_cast<size_t>(d)] = 0;
            }
        }
    }

    return out;
}

// Generic unary elementwise operator
template <typename FFloat, typename FDouble>
Tensor elementwise_unary_op(const Tensor& a, FFloat op_f, FDouble op_d) {
    Tensor out(a.shape(), a.dtype());
    int64_t n = out.numel();
    if (n == 0) return out;

    if (a.is_contiguous() && a.storage_offset() == 0) {
        if (a.dtype() == DType::Float32) {
            const float* pa = a.data<float>();
            float* po = out.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                po[i] = op_f(pa[i]);
            }
        } else {
            const double* pa = a.data<double>();
            double* po = out.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                po[i] = op_d(pa[i]);
            }
        }
        return out;
    }

    size_t rank = a.ndim();
    std::vector<int64_t> coords(rank, 0);
    const int64_t* strides = a.strides().strides().data();
    int64_t off_base = a.storage_offset();

    if (a.dtype() == DType::Float32) {
        const float* raw_a = static_cast<const float*>(a.storage()->data());
        float* po = out.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            int64_t off = off_base;
            for (size_t d = 0; d < rank; ++d) {
                off += coords[d] * strides[d];
            }
            po[i] = op_f(raw_a[off]);
            for (int64_t d = static_cast<int64_t>(rank) - 1; d >= 0; --d) {
                coords[static_cast<size_t>(d)]++;
                if (coords[static_cast<size_t>(d)] < a.shape()[static_cast<size_t>(d)]) {
                    break;
                }
                coords[static_cast<size_t>(d)] = 0;
            }
        }
    } else {
        const double* raw_a = static_cast<const double*>(a.storage()->data());
        double* po = out.data<double>();
        for (int64_t i = 0; i < n; ++i) {
            int64_t off = off_base;
            for (size_t d = 0; d < rank; ++d) {
                off += coords[d] * strides[d];
            }
            po[i] = op_d(raw_a[off]);
            for (int64_t d = static_cast<int64_t>(rank) - 1; d >= 0; --d) {
                coords[static_cast<size_t>(d)]++;
                if (coords[static_cast<size_t>(d)] < a.shape()[static_cast<size_t>(d)]) {
                    break;
                }
                coords[static_cast<size_t>(d)] = 0;
            }
        }
    }

    return out;
}

} // anonymous namespace

Tensor add(const Tensor& a, const Tensor& b) {
    return elementwise_binary_op(a, b,
        [](float x, float y) { return x + y; },
        [](double x, double y) { return x + y; });
}

Tensor sub(const Tensor& a, const Tensor& b) {
    return elementwise_binary_op(a, b,
        [](float x, float y) { return x - y; },
        [](double x, double y) { return x - y; });
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return elementwise_binary_op(a, b,
        [](float x, float y) { return x * y; },
        [](double x, double y) { return x * y; });
}

Tensor div(const Tensor& a, const Tensor& b) {
    return elementwise_binary_op(a, b,
        [](float x, float y) { return x / y; },
        [](double x, double y) { return x / y; });
}

Tensor relu(const Tensor& a) {
    return elementwise_unary_op(a,
        [](float x) { return x > 0.0f ? x : 0.0f; },
        [](double x) { return x > 0.0 ? x : 0.0; });
}

Tensor sigmoid(const Tensor& a) {
    // Numerically stable sigmoid:
    // for x >= 0: 1 / (1 + exp(-x))
    // for x < 0: exp(x) / (1 + exp(x))
    return elementwise_unary_op(a,
        [](float x) {
            if (x >= 0.0f) {
                return 1.0f / (1.0f + std::exp(-x));
            } else {
                float ex = std::exp(x);
                return ex / (1.0f + ex);
            }
        },
        [](double x) {
            if (x >= 0.0) {
                return 1.0 / (1.0 + std::exp(-x));
            } else {
                double ex = std::exp(x);
                return ex / (1.0 + ex);
            }
        });
}

// 2D Cache-blocked matrix multiplication with IKJ loop ordering
Tensor matmul_2d(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw DimensionError("2D matmul requires rank 2 tensors");
    }
    if (a.dtype() != b.dtype()) {
        throw TypeError("DType mismatch in matmul");
    }

    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    int64_t K_b = b.shape()[0];
    int64_t N = b.shape()[1];

    if (K != K_b) {
        throw ShapeError("Inner matrix dimensions must match in matmul: (" +
                         std::to_string(M) + ", " + std::to_string(K) + ") vs (" +
                         std::to_string(K_b) + ", " + std::to_string(N) + ")");
    }

    Tensor out = Tensor::zeros({M, N}, a.dtype());
    if (M == 0 || K == 0 || N == 0) {
        return out;
    }

    // Ensure contiguous inputs for maximum cache locality
    Tensor a_contig = a.contiguous();
    Tensor b_contig = b.contiguous();

    constexpr int64_t BLOCK_SIZE = 32;

    if (a.dtype() == DType::Float32) {
        const float* A = a_contig.data<float>();
        const float* B = b_contig.data<float>();
        float* C = out.data<float>();

        // Blocked IKJ loop order for optimal cache reuse and auto-vectorization
        for (int64_t bi = 0; bi < M; bi += BLOCK_SIZE) {
            int64_t i_max = std::min(bi + BLOCK_SIZE, M);
            for (int64_t bk = 0; bk < K; bk += BLOCK_SIZE) {
                int64_t k_max = std::min(bk + BLOCK_SIZE, K);
                for (int64_t bj = 0; bj < N; bj += BLOCK_SIZE) {
                    int64_t j_max = std::min(bj + BLOCK_SIZE, N);

                    for (int64_t i = bi; i < i_max; ++i) {
                        for (int64_t k = bk; k < k_max; ++k) {
                            float a_ik = A[i * K + k];
                            const float* b_row = &B[k * N];
                            float* c_row = &C[i * N];
                            for (int64_t j = bj; j < j_max; ++j) {
                                c_row[j] += a_ik * b_row[j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        const double* A = a_contig.data<double>();
        const double* B = b_contig.data<double>();
        double* C = out.data<double>();

        for (int64_t bi = 0; bi < M; bi += BLOCK_SIZE) {
            int64_t i_max = std::min(bi + BLOCK_SIZE, M);
            for (int64_t bk = 0; bk < K; bk += BLOCK_SIZE) {
                int64_t k_max = std::min(bk + BLOCK_SIZE, K);
                for (int64_t bj = 0; bj < N; bj += BLOCK_SIZE) {
                    int64_t j_max = std::min(bj + BLOCK_SIZE, N);

                    for (int64_t i = bi; i < i_max; ++i) {
                        for (int64_t k = bk; k < k_max; ++k) {
                            double a_ik = A[i * K + k];
                            const double* b_row = &B[k * N];
                            double* c_row = &C[i * N];
                            for (int64_t j = bj; j < j_max; ++j) {
                                c_row[j] += a_ik * b_row[j];
                            }
                        }
                    }
                }
            }
        }
    }

    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 1 || b.ndim() < 1) {
        throw DimensionError("Both arguments to matmul must be at least 1D");
    }
    if (a.ndim() == 2 && b.ndim() == 2) {
        return matmul_2d(a, b);
    }
    // 1D x 2D or 2D x 1D vector extensions
    if (a.ndim() == 1 && b.ndim() == 2) {
        Tensor a_2d = a.view({1, a.shape()[0]});
        Tensor res = matmul_2d(a_2d, b);
        return res.view({b.shape()[1]});
    }
    if (a.ndim() == 2 && b.ndim() == 1) {
        Tensor b_2d = b.view({b.shape()[0], 1});
        Tensor res = matmul_2d(a, b_2d);
        return res.view({a.shape()[0]});
    }
    // Batched matrix multiplication: broadcast leading batch dims
    if (a.ndim() >= 2 && b.ndim() >= 2) {
        size_t rank_a = a.ndim();
        size_t rank_b = b.ndim();
        std::vector<int64_t> batch_a(a.shape().dims().begin(), a.shape().dims().end() - 2);
        std::vector<int64_t> batch_b(b.shape().dims().begin(), b.shape().dims().end() - 2);
        Shape batch_shape = Tensor::broadcast_shapes(Shape(batch_a), Shape(batch_b));

        int64_t M = a.shape()[rank_a - 2];
        int64_t K_a = a.shape()[rank_a - 1];
        int64_t K_b = b.shape()[rank_b - 2];
        int64_t N = b.shape()[rank_b - 1];

        if (K_a != K_b) {
            throw ShapeError("Inner dimensions mismatch in batched matmul");
        }

        std::vector<int64_t> out_dims = batch_shape.dims();
        out_dims.push_back(M);
        out_dims.push_back(N);
        Shape out_shape(std::move(out_dims));

        Tensor out = Tensor::zeros(out_shape, a.dtype());
        int64_t batch_count = batch_shape.numel();

        // Broadcast a and b to include full batch shape
        std::vector<int64_t> broad_dims_a = batch_shape.dims();
        broad_dims_a.push_back(M);
        broad_dims_a.push_back(K_a);
        Tensor a_broad = a.broadcast_to(Shape(broad_dims_a)).contiguous();

        std::vector<int64_t> broad_dims_b = batch_shape.dims();
        broad_dims_b.push_back(K_b);
        broad_dims_b.push_back(N);
        Tensor b_broad = b.broadcast_to(Shape(broad_dims_b)).contiguous();

        for (int64_t b_idx = 0; b_idx < batch_count; ++b_idx) {
            Tensor a_slice = Tensor::from_blob(
                static_cast<char*>(a_broad.raw_data()) + static_cast<size_t>(b_idx * M * K_a) * a.itemsize(),
                Shape({M, K_a}), a.dtype());
            Tensor b_slice = Tensor::from_blob(
                static_cast<char*>(b_broad.raw_data()) + static_cast<size_t>(b_idx * K_b * N) * b.itemsize(),
                Shape({K_b, N}), b.dtype());
            Tensor out_slice = matmul_2d(a_slice, b_slice);
            std::memcpy(
                static_cast<char*>(out.raw_data()) + static_cast<size_t>(b_idx * M * N) * out.itemsize(),
                out_slice.raw_data(),
                out_slice.nbytes());
        }
        return out;
    }

    throw DimensionError("Unsupported ranks for matmul");
}

Tensor reduce_sum(const Tensor& a, const std::vector<int64_t>& axes, bool keepdims) {
    int64_t rank = static_cast<int64_t>(a.ndim());
    std::set<int64_t> norm_axes;

    if (axes.empty()) {
        // Empty axes reduces all dimensions to scalar
        for (int64_t i = 0; i < rank; ++i) {
            norm_axes.insert(i);
        }
    } else {
        for (int64_t ax : axes) {
            if (ax < 0) ax += rank;
            if (ax < 0 || ax >= rank) {
                throw DimensionError("Reduction axis " + std::to_string(ax) + " out of bounds");
            }
            norm_axes.insert(ax);
        }
    }

    std::vector<int64_t> out_dims;
    for (int64_t i = 0; i < rank; ++i) {
        if (norm_axes.count(i)) {
            if (keepdims) {
                out_dims.push_back(1);
            }
        } else {
            out_dims.push_back(a.shape()[static_cast<size_t>(i)]);
        }
    }

    Shape out_shape(out_dims);
    Tensor out = Tensor::zeros(out_shape, a.dtype());
    if (a.numel() == 0) {
        return out;
    }

    // Iterate through input elements and accumulate into output
    Tensor a_contig = a.contiguous();
    size_t in_rank = a.ndim();
    std::vector<int64_t> in_coords(in_rank, 0);

    if (a.dtype() == DType::Float32) {
        const float* src = a_contig.data<float>();
        // Use double accumulation buffer for precision
        std::vector<double> accum(static_cast<size_t>(out.numel()), 0.0);

        for (int64_t i = 0; i < a.numel(); ++i) {
            // Compute corresponding output coords
            std::vector<int64_t> out_coords;
            for (int64_t d = 0; d < rank; ++d) {
                if (norm_axes.count(d)) {
                    if (keepdims) {
                        out_coords.push_back(0);
                    }
                } else {
                    out_coords.push_back(in_coords[static_cast<size_t>(d)]);
                }
            }

            int64_t out_idx = 0;
            if (!out_coords.empty()) {
                out_idx = out.compute_offset(out_coords);
            }
            accum[static_cast<size_t>(out_idx)] += static_cast<double>(src[i]);

            // Advance input coordinates
            for (int64_t d = rank - 1; d >= 0; --d) {
                in_coords[static_cast<size_t>(d)]++;
                if (in_coords[static_cast<size_t>(d)] < a.shape()[static_cast<size_t>(d)]) {
                    break;
                }
                in_coords[static_cast<size_t>(d)] = 0;
            }
        }

        float* dst = out.data<float>();
        for (size_t i = 0; i < accum.size(); ++i) {
            dst[i] = static_cast<float>(accum[i]);
        }
    } else {
        const double* src = a_contig.data<double>();
        double* dst = out.data<double>();

        for (int64_t i = 0; i < a.numel(); ++i) {
            std::vector<int64_t> out_coords;
            for (int64_t d = 0; d < rank; ++d) {
                if (norm_axes.count(d)) {
                    if (keepdims) {
                        out_coords.push_back(0);
                    }
                } else {
                    out_coords.push_back(in_coords[static_cast<size_t>(d)]);
                }
            }

            int64_t out_idx = 0;
            if (!out_coords.empty()) {
                out_idx = out.compute_offset(out_coords);
            }
            dst[out_idx] += src[i];

            for (int64_t d = rank - 1; d >= 0; --d) {
                in_coords[static_cast<size_t>(d)]++;
                if (in_coords[static_cast<size_t>(d)] < a.shape()[static_cast<size_t>(d)]) {
                    break;
                }
                in_coords[static_cast<size_t>(d)] = 0;
            }
        }
    }

    return out;
}

Tensor reduce_mean(const Tensor& a, const std::vector<int64_t>& axes, bool keepdims) {
    Tensor sum_val = reduce_sum(a, axes, keepdims);
    int64_t reduced_elements = a.numel() / (sum_val.numel() > 0 ? sum_val.numel() : 1);
    if (reduced_elements <= 0) reduced_elements = 1;

    double factor = 1.0 / static_cast<double>(reduced_elements);
    Tensor factor_tensor = Tensor::scalar(factor, a.dtype());
    return mul(sum_val, factor_tensor);
}

Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
    return a.transpose(dim0, dim1);
}

Tensor permute(const Tensor& a, const std::vector<int64_t>& dims) {
    return a.permute(dims);
}

} // namespace ops
} // namespace tinytensor
