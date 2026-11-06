#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

enum class DType : uint8_t {
    Float32 = 0,
    Float64 = 1
};

inline std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return "float32";
        case DType::Float64:
            return "float64";
        default:
            throw TypeError("Unknown DType");
    }
}

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return sizeof(float);
        case DType::Float64:
            return sizeof(double);
        default:
            throw TypeError("Unknown DType size");
    }
}

template <typename T>
struct DTypeTraits;

template <>
struct DTypeTraits<float> {
    static constexpr DType dtype = DType::Float32;
    static constexpr const char* name = "float32";
};

template <>
struct DTypeTraits<double> {
    static constexpr DType dtype = DType::Float64;
    static constexpr const char* name = "float64";
};

} // namespace tinytensor
