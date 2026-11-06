#include "tinytensor/core/tensor.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>

namespace tinytensor {

Tensor Tensor::empty(const Shape& shape, DType dtype) {
    return Tensor(shape, dtype);
}

Tensor Tensor::zeros(const Shape& shape, DType dtype) {
    Tensor t(shape, dtype);
    if (t.numel() > 0 && t.raw_data()) {
        std::memset(t.raw_data(), 0, t.nbytes());
    }
    return t;
}

Tensor Tensor::ones(const Shape& shape, DType dtype) {
    return full(shape, 1.0, dtype);
}

Tensor Tensor::full(const Shape& shape, double val, DType dtype) {
    Tensor t(shape, dtype);
    if (t.numel() == 0 || !t.raw_data()) {
        return t;
    }
    if (dtype == DType::Float32) {
        float* ptr = t.data<float>();
        float fval = static_cast<float>(val);
        std::fill(ptr, ptr + t.numel(), fval);
    } else if (dtype == DType::Float64) {
        double* ptr = t.data<double>();
        std::fill(ptr, ptr + t.numel(), val);
    } else {
        throw TypeError("Unsupported dtype in full()");
    }
    return t;
}

Tensor Tensor::scalar(double val, DType dtype) {
    return full(Shape({}), val, dtype);
}

Tensor Tensor::from_blob(void* ptr, const Shape& shape, DType dtype,
                         const Strides& strides, int64_t offset, bool owning) {
    Strides s = strides;
    if (s.ndim() == 0 && shape.ndim() > 0) {
        s = Strides::compute_contiguous(shape);
    }
    size_t nbytes = math::checked_size_mul(
        static_cast<size_t>(shape.numel()), dtype_size(dtype));
    StoragePtr storage = make_storage_from_blob(ptr, nbytes, owning);
    return Tensor(shape, s, offset, storage, dtype);
}

int64_t Tensor::compute_offset(const std::vector<int64_t>& indices) const {
    if (indices.size() != shape_.ndim()) {
        throw DimensionError("Index rank " + std::to_string(indices.size()) +
                             " does not match tensor rank " + std::to_string(shape_.ndim()));
    }
    int64_t offset = storage_offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        int64_t idx = indices[i];
        int64_t dim = shape_[i];
        if (idx < 0) {
            idx += dim;
        }
        if (idx < 0 || idx >= dim) {
            throw DimensionError("Index " + std::to_string(indices[i]) +
                                 " out of bounds for axis " + std::to_string(i) +
                                 " of size " + std::to_string(dim));
        }
        offset = math::checked_add(offset, math::checked_mul(idx, strides_[i]));
    }
    return offset;
}

Tensor Tensor::view(const Shape& new_shape) const {
    if (new_shape.numel() != numel()) {
        throw ShapeError("Cannot view tensor of size " + std::to_string(numel()) +
                         " as shape " + new_shape.to_string() + " of size " +
                         std::to_string(new_shape.numel()));
    }
    if (!is_contiguous()) {
        throw ShapeError("view() is only supported on contiguous tensors. Use reshape() instead.");
    }
    Strides new_strides = Strides::compute_contiguous(new_shape);
    return Tensor(new_shape, new_strides, storage_offset_, storage_, dtype_);
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (new_shape.numel() != numel()) {
        throw ShapeError("Cannot reshape tensor of size " + std::to_string(numel()) +
                         " as shape " + new_shape.to_string() + " of size " +
                         std::to_string(new_shape.numel()));
    }
    if (is_contiguous()) {
        return view(new_shape);
    }
    Tensor contig = contiguous();
    return contig.view(new_shape);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    int64_t rank = static_cast<int64_t>(ndim());
    if (dim0 < 0) dim0 += rank;
    if (dim1 < 0) dim1 += rank;
    if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
        throw DimensionError("Invalid transpose axes: " + std::to_string(dim0) +
                             ", " + std::to_string(dim1) + " for rank " + std::to_string(rank));
    }
    std::vector<int64_t> new_dims = shape_.dims();
    std::vector<int64_t> new_strides = strides_.strides();
    std::swap(new_dims[static_cast<size_t>(dim0)], new_dims[static_cast<size_t>(dim1)]);
    std::swap(new_strides[static_cast<size_t>(dim0)], new_strides[static_cast<size_t>(dim1)]);
    return Tensor(Shape(std::move(new_dims)), Strides(std::move(new_strides)),
                  storage_offset_, storage_, dtype_);
}

Tensor Tensor::permute(const std::vector<int64_t>& dims) const {
    if (dims.size() != ndim()) {
        throw DimensionError("Permutation dims size " + std::to_string(dims.size()) +
                             " must match tensor rank " + std::to_string(ndim()));
    }
    int64_t rank = static_cast<int64_t>(ndim());
    std::vector<bool> seen(ndim(), false);
    std::vector<int64_t> new_dims(ndim());
    std::vector<int64_t> new_strides(ndim());

    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t d = dims[i];
        if (d < 0) d += rank;
        if (d < 0 || d >= rank) {
            throw DimensionError("Axis " + std::to_string(dims[i]) +
                                 " out of bounds for rank " + std::to_string(rank));
        }
        if (seen[static_cast<size_t>(d)]) {
            throw DimensionError("Duplicate axis " + std::to_string(d) + " in permute");
        }
        seen[static_cast<size_t>(d)] = true;
        new_dims[i] = shape_[static_cast<size_t>(d)];
        new_strides[i] = strides_[static_cast<size_t>(d)];
    }

    return Tensor(Shape(std::move(new_dims)), Strides(std::move(new_strides)),
                  storage_offset_, storage_, dtype_);
}

Tensor Tensor::broadcast_to(const Shape& target_shape) const {
    size_t src_rank = ndim();
    size_t tgt_rank = target_shape.ndim();

    if (src_rank > tgt_rank) {
        throw BroadcastError("Cannot broadcast tensor of rank " + std::to_string(src_rank) +
                             " to lower rank " + std::to_string(tgt_rank));
    }

    std::vector<int64_t> new_strides(tgt_rank, 0);
    size_t rank_diff = tgt_rank - src_rank;

    for (size_t i = 0; i < tgt_rank; ++i) {
        if (i < rank_diff) {
            // Prepended dimension: extent in target must be >= 0; stride is 0
            new_strides[i] = 0;
        } else {
            size_t src_idx = i - rank_diff;
            int64_t src_dim = shape_[src_idx];
            int64_t tgt_dim = target_shape[i];

            if (src_dim == tgt_dim) {
                new_strides[i] = strides_[src_idx];
            } else if (src_dim == 1) {
                // Singleton dimension expanded: stride is 0!
                new_strides[i] = 0;
            } else {
                throw BroadcastError("Cannot broadcast shape " + shape_.to_string() +
                                     " to " + target_shape.to_string() +
                                     ": mismatch at dimension " + std::to_string(i));
            }
        }
    }

    return Tensor(target_shape, Strides(std::move(new_strides)),
                  storage_offset_, storage_, dtype_);
}

Shape Tensor::broadcast_shapes(const Shape& a, const Shape& b) {
    size_t rank_a = a.ndim();
    size_t rank_b = b.ndim();
    size_t max_rank = std::max(rank_a, rank_b);

    std::vector<int64_t> out_dims(max_rank);

    for (size_t i = 0; i < max_rank; ++i) {
        int64_t dim_a = (i < rank_a) ? a[rank_a - 1 - i] : 1;
        int64_t dim_b = (i < rank_b) ? b[rank_b - 1 - i] : 1;

        if (dim_a == dim_b) {
            out_dims[max_rank - 1 - i] = dim_a;
        } else if (dim_a == 1) {
            out_dims[max_rank - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            out_dims[max_rank - 1 - i] = dim_a;
        } else {
            throw BroadcastError("Shapes " + a.to_string() + " and " +
                                 b.to_string() + " are not broadcastable");
        }
    }

    return Shape(std::move(out_dims));
}

Tensor Tensor::contiguous() const {
    if (is_contiguous() && storage_offset_ == 0) {
        return *this;
    }

    Tensor result(shape_, dtype_);
    if (numel() == 0) {
        return result;
    }

    // Copy elements in coordinate order
    size_t rank = ndim();
    std::vector<int64_t> coords(rank, 0);

    if (dtype_ == DType::Float32) {
        float* dst = result.data<float>();
        const float* src_base = static_cast<const float*>(storage_->data());
        for (int64_t i = 0; i < numel(); ++i) {
            int64_t offset = storage_offset_;
            for (size_t d = 0; d < rank; ++d) {
                offset += coords[d] * strides_[d];
            }
            dst[i] = src_base[offset];

            // Advance coordinates
            for (int64_t d = static_cast<int64_t>(rank) - 1; d >= 0; --d) {
                coords[static_cast<size_t>(d)]++;
                if (coords[static_cast<size_t>(d)] < shape_[static_cast<size_t>(d)]) {
                    break;
                }
                coords[static_cast<size_t>(d)] = 0;
            }
        }
    } else if (dtype_ == DType::Float64) {
        double* dst = result.data<double>();
        const double* src_base = static_cast<const double*>(storage_->data());
        for (int64_t i = 0; i < numel(); ++i) {
            int64_t offset = storage_offset_;
            for (size_t d = 0; d < rank; ++d) {
                offset += coords[d] * strides_[d];
            }
            dst[i] = src_base[offset];

            for (int64_t d = static_cast<int64_t>(rank) - 1; d >= 0; --d) {
                coords[static_cast<size_t>(d)]++;
                if (coords[static_cast<size_t>(d)] < shape_[static_cast<size_t>(d)]) {
                    break;
                }
                coords[static_cast<size_t>(d)] = 0;
            }
        }
    }

    return result;
}

Tensor Tensor::clone() const {
    Tensor result(shape_, dtype_);
    if (numel() == 0) {
        return result;
    }
    if (is_contiguous()) {
        std::memcpy(result.raw_data(), raw_data(), nbytes());
        return result;
    }
    return contiguous();
}

} // namespace tinytensor
