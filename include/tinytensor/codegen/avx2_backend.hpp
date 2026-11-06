#pragma once

#include <cstddef>
#include <cstdint>
#include "tinytensor/codegen/cpu_features.hpp"

namespace tinytensor {
namespace simd {

// CPU feature detection
bool has_avx2();

// AVX2 contiguous elementwise kernels (f32)
void avx2_add_f32(const float* a, const float* b, float* out, size_t n);
void avx2_sub_f32(const float* a, const float* b, float* out, size_t n);
void avx2_mul_f32(const float* a, const float* b, float* out, size_t n);
void avx2_relu_f32(const float* a, float* out, size_t n);

// Portable scalar fallback kernels (f32)
void scalar_add_f32(const float* a, const float* b, float* out, size_t n);
void scalar_sub_f32(const float* a, const float* b, float* out, size_t n);
void scalar_mul_f32(const float* a, const float* b, float* out, size_t n);
void scalar_relu_f32(const float* a, float* out, size_t n);

} // namespace simd
} // namespace tinytensor
