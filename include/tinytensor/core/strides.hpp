#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include "tinytensor/core/shape.hpp"
#include "tinytensor/utils/checked_math.hpp"
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

class Strides {
public:
    Strides() = default;

    Strides(std::initializer_list<int64_t> strides)
        : strides_(strides) {}

    explicit Strides(const std::vector<int64_t>& strides)
        : strides_(strides) {}

    explicit Strides(std::vector<int64_t>&& strides)
        : strides_(std::move(strides)) {}

    static Strides compute_contiguous(const Shape& shape) {
        size_t rank = shape.ndim();
        if (rank == 0) {
            return Strides({});
        }

        std::vector<int64_t> strides(rank);
        int64_t current_stride = 1;

        for (int64_t i = static_cast<int64_t>(rank) - 1; i >= 0; --i) {
            strides[static_cast<size_t>(i)] = current_stride;
            int64_t dim = shape[static_cast<size_t>(i)];
            if (dim > 0) {
                current_stride = math::checked_mul(current_stride, dim);
            }
        }

        return Strides(std::move(strides));
    }

    size_t ndim() const noexcept { return strides_.size(); }
    const std::vector<int64_t>& strides() const noexcept { return strides_; }

    int64_t operator[](size_t index) const {
        if (index >= strides_.size()) {
            throw DimensionError("Stride index out of range: " + std::to_string(index));
        }
        return strides_[index];
    }

    int64_t& operator[](size_t index) {
        if (index >= strides_.size()) {
            throw DimensionError("Stride index out of range: " + std::to_string(index));
        }
        return strides_[index];
    }

    bool operator==(const Strides& other) const noexcept {
        return strides_ == other.strides_;
    }

    bool operator!=(const Strides& other) const noexcept {
        return strides_ != other.strides_;
    }

    bool is_contiguous(const Shape& shape) const noexcept {
        if (shape.ndim() != strides_.size()) {
            return false;
        }
        if (shape.ndim() == 0) {
            return true;
        }
        if (shape.numel() == 0) {
            return true;
        }

        int64_t expected_stride = 1;
        for (int64_t i = static_cast<int64_t>(shape.ndim()) - 1; i >= 0; --i) {
            size_t idx = static_cast<size_t>(i);
            int64_t dim = shape[idx];
            if (dim == 1) {
                // Singleton dimensions don't affect contiguous traversal order
                continue;
            }
            if (strides_[idx] != expected_stride) {
                return false;
            }
            expected_stride = math::checked_mul(expected_stride, dim);
        }
        return true;
    }

    std::string to_string() const {
        std::string s = "(";
        for (size_t i = 0; i < strides_.size(); ++i) {
            s += std::to_string(strides_[i]);
            if (i + 1 < strides_.size() || strides_.size() == 1) {
                s += ",";
                if (i + 1 < strides_.size()) s += " ";
            }
        }
        s += ")";
        return s;
    }

private:
    std::vector<int64_t> strides_;
};

} // namespace tinytensor
