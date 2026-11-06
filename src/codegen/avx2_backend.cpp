#include "tinytensor/codegen/avx2_backend.hpp"
#include <algorithm>
#include <cstdint>
#include <immintrin.h>

namespace tinytensor {
namespace simd {

void scalar_add_f32(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void scalar_sub_f32(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

void scalar_mul_f32(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void scalar_relu_f32(const float* a, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

void avx2_add_f32(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    size_t vec_limit = n >= 8 ? n - 7 : 0;

    for (; i < vec_limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }

    // Scalar tail fallback
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void avx2_sub_f32(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    size_t vec_limit = n >= 8 ? n - 7 : 0;

    for (; i < vec_limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

void avx2_mul_f32(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    size_t vec_limit = n >= 8 ? n - 7 : 0;

    for (; i < vec_limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void avx2_relu_f32(const float* a, float* out, size_t n) {
    size_t i = 0;
    size_t vec_limit = n >= 8 ? n - 7 : 0;
    __m256 zero = _mm256_setzero_ps();

    for (; i < vec_limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_max_ps(va, zero);
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

} // namespace simd
} // namespace tinytensor
