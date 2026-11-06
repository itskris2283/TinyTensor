#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include "tinytensor/utils/checked_math.hpp"
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

constexpr size_t MAX_TENSOR_RANK = 32;
constexpr int64_t MAX_TENSOR_ELEMENTS = 2147483648LL; // 2^31 elements (~8GB for f32)
constexpr size_t MAX_ALLOCATION_BYTES = 16ULL * 1024ULL * 1024ULL * 1024ULL; // 16 GB

class Shape {
public:
    Shape() = default;

    Shape(std::initializer_list<int64_t> dims)
        : dims_(dims) {
        validate_and_compute();
    }

    explicit Shape(const std::vector<int64_t>& dims)
        : dims_(dims) {
        validate_and_compute();
    }

    explicit Shape(std::vector<int64_t>&& dims)
        : dims_(std::move(dims)) {
        validate_and_compute();
    }

    size_t ndim() const noexcept { return dims_.size(); }
    int64_t numel() const noexcept { return numel_; }
    const std::vector<int64_t>& dims() const noexcept { return dims_; }

    int64_t operator[](size_t index) const {
        if (index >= dims_.size()) {
            throw DimensionError("Dimension index out of range: " + std::to_string(index) +
                                 " for rank " + std::to_string(dims_.size()));
        }
        return dims_[index];
    }

    int64_t& operator[](size_t index) {
        if (index >= dims_.size()) {
            throw DimensionError("Dimension index out of range: " + std::to_string(index) +
                                 " for rank " + std::to_string(dims_.size()));
        }
        return dims_[index];
    }

    bool operator==(const Shape& other) const noexcept {
        return dims_ == other.dims_;
    }

    bool operator!=(const Shape& other) const noexcept {
        return dims_ != other.dims_;
    }

    std::string to_string() const {
        std::string s = "(";
        for (size_t i = 0; i < dims_.size(); ++i) {
            s += std::to_string(dims_[i]);
            if (i + 1 < dims_.size() || dims_.size() == 1) {
                s += ",";
                if (i + 1 < dims_.size()) s += " ";
            }
        }
        s += ")";
        return s;
    }

    void recompute_numel() {
        validate_and_compute();
    }

private:
    std::vector<int64_t> dims_;
    int64_t numel_{1};

    void validate_and_compute() {
        if (dims_.size() > MAX_TENSOR_RANK) {
            throw ResourceLimitError("Tensor rank " + std::to_string(dims_.size()) +
                                     " exceeds maximum allowed rank " + std::to_string(MAX_TENSOR_RANK));
        }

        if (dims_.empty()) {
            numel_ = 1; // 0D scalar has 1 element
            return;
        }

        int64_t count = 1;
        for (size_t i = 0; i < dims_.size(); ++i) {
            int64_t d = dims_[i];
            if (d < 0) {
                throw ShapeError("Dimension at axis " + std::to_string(i) + " is negative: " + std::to_string(d));
            }
            if (d == 0) {
                count = 0;
            } else if (count != 0) {
                count = math::checked_mul(count, d);
            }
        }

        if (count > MAX_TENSOR_ELEMENTS) {
            throw ResourceLimitError("Element count " + std::to_string(count) +
                                     " exceeds maximum allowed elements " + std::to_string(MAX_TENSOR_ELEMENTS));
        }

        numel_ = count;
    }
};

} // namespace tinytensor
