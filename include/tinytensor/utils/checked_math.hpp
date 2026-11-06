#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {
namespace math {

inline int64_t checked_add(int64_t a, int64_t b) {
    if ((b > 0 && a > std::numeric_limits<int64_t>::max() - b) ||
        (b < 0 && a < std::numeric_limits<int64_t>::min() - b)) {
        throw MemoryError("Integer overflow detected during addition");
    }
    return a + b;
}

inline int64_t checked_mul(int64_t a, int64_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > 0) {
        if (b > 0) {
            if (a > std::numeric_limits<int64_t>::max() / b) {
                throw MemoryError("Integer overflow detected during multiplication");
            }
        } else {
            if (b < std::numeric_limits<int64_t>::min() / a) {
                throw MemoryError("Integer overflow detected during multiplication");
            }
        }
    } else {
        if (b > 0) {
            if (a < std::numeric_limits<int64_t>::min() / b) {
                throw MemoryError("Integer overflow detected during multiplication");
            }
        } else {
            if (a < std::numeric_limits<int64_t>::max() / b) {
                throw MemoryError("Integer overflow detected during multiplication");
            }
        }
    }
    return a * b;
}

inline size_t checked_size_mul(size_t a, size_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > std::numeric_limits<size_t>::max() / b) {
        throw MemoryError("Size overflow detected during byte calculation");
    }
    return a * b;
}

} // namespace math
} // namespace tinytensor
