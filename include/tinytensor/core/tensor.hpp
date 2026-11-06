#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "tinytensor/core/dtype.hpp"
#include "tinytensor/core/shape.hpp"
#include "tinytensor/core/storage.hpp"
#include "tinytensor/core/strides.hpp"
#include "tinytensor/utils/checked_math.hpp"
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

class Tensor {
public:
    Tensor()
        : dtype_(DType::Float32),
          shape_({}),
          strides_({}),
          storage_offset_(0),
          storage_(nullptr) {}

    Tensor(Shape shape, DType dtype = DType::Float32)
        : dtype_(dtype),
          shape_(std::move(shape)),
          strides_(Strides::compute_contiguous(shape_)),
          storage_offset_(0) {
        size_t nbytes = math::checked_size_mul(
            static_cast<size_t>(shape_.numel()), dtype_size(dtype_));
        storage_ = make_storage(nbytes);
    }

    Tensor(Shape shape, Strides strides, int64_t storage_offset,
           StoragePtr storage, DType dtype)
        : dtype_(dtype),
          shape_(std::move(shape)),
          strides_(std::move(strides)),
          storage_offset_(storage_offset),
          storage_(std::move(storage)) {
        if (storage_offset_ < 0) {
            throw MemoryError("Negative storage offset: " + std::to_string(storage_offset_));
        }
    }

    // Factory methods
    static Tensor empty(const Shape& shape, DType dtype = DType::Float32);
    static Tensor zeros(const Shape& shape, DType dtype = DType::Float32);
    static Tensor ones(const Shape& shape, DType dtype = DType::Float32);
    static Tensor full(const Shape& shape, double val, DType dtype = DType::Float32);
    static Tensor scalar(double val, DType dtype = DType::Float32);
    static Tensor from_blob(void* ptr, const Shape& shape, DType dtype = DType::Float32,
                            const Strides& strides = Strides(), int64_t offset = 0,
                            bool owning = false);

    // Metadata
    const Shape& shape() const noexcept { return shape_; }
    const Strides& strides() const noexcept { return strides_; }
    size_t ndim() const noexcept { return shape_.ndim(); }
    int64_t numel() const noexcept { return shape_.numel(); }
    DType dtype() const noexcept { return dtype_; }
    size_t itemsize() const { return dtype_size(dtype_); }
    size_t nbytes() const { return math::checked_size_mul(static_cast<size_t>(numel()), itemsize()); }
    int64_t storage_offset() const noexcept { return storage_offset_; }
    const StoragePtr& storage() const noexcept { return storage_; }
    bool is_contiguous() const noexcept { return strides_.is_contiguous(shape_); }

    // Pointer access
    void* raw_data() {
        if (!storage_ || !storage_->data()) return nullptr;
        return static_cast<char*>(storage_->data()) + static_cast<size_t>(storage_offset_) * itemsize();
    }

    const void* raw_data() const {
        if (!storage_ || !storage_->data()) return nullptr;
        return static_cast<const char*>(storage_->data()) + static_cast<size_t>(storage_offset_) * itemsize();
    }

    template <typename T>
    T* data() {
        if (DTypeTraits<T>::dtype != dtype_) {
            throw TypeError("Type mismatch in data<T>(): requested " +
                            std::string(DTypeTraits<T>::name) + " but tensor is " +
                            dtype_to_string(dtype_));
        }
        return static_cast<T*>(raw_data());
    }

    template <typename T>
    const T* data() const {
        if (DTypeTraits<T>::dtype != dtype_) {
            throw TypeError("Type mismatch in data<T>(): requested " +
                            std::string(DTypeTraits<T>::name) + " but tensor is " +
                            dtype_to_string(dtype_));
        }
        return static_cast<const T*>(raw_data());
    }

    // Element access with bounds checking
    int64_t compute_offset(const std::vector<int64_t>& indices) const;

    template <typename T>
    T& at(const std::vector<int64_t>& indices) {
        int64_t offset = compute_offset(indices);
        if (DTypeTraits<T>::dtype != dtype_) {
            throw TypeError("Type mismatch in at()");
        }
        return *(static_cast<T*>(static_cast<void*>(static_cast<char*>(storage_->data()) + static_cast<size_t>(offset) * itemsize())));
    }

    template <typename T>
    const T& at(const std::vector<int64_t>& indices) const {
        int64_t offset = compute_offset(indices);
        if (DTypeTraits<T>::dtype != dtype_) {
            throw TypeError("Type mismatch in at()");
        }
        return *(static_cast<const T*>(static_cast<const void*>(static_cast<const char*>(storage_->data()) + static_cast<size_t>(offset) * itemsize())));
    }

    template <typename T>
    T item() const {
        if (numel() != 1) {
            throw ShapeError("item() can only be called on tensors with exactly 1 element, but got " +
                             std::to_string(numel()));
        }
        if (DTypeTraits<T>::dtype != dtype_) {
            throw TypeError("Type mismatch in item()");
        }
        return *(static_cast<const T*>(raw_data()));
    }

    // Transformations
    Tensor view(const Shape& new_shape) const;
    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor permute(const std::vector<int64_t>& dims) const;
    Tensor contiguous() const;
    Tensor clone() const;
    Tensor broadcast_to(const Shape& target_shape) const;

    // Broadcasting helper
    static Shape broadcast_shapes(const Shape& a, const Shape& b);

private:
    DType dtype_;
    Shape shape_;
    Strides strides_;
    int64_t storage_offset_{0};
    StoragePtr storage_;
};

} // namespace tinytensor
