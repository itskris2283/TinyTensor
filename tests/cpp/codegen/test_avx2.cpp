#include <gtest/gtest.h>
#include "tinytensor/codegen/avx2_backend.hpp"

using namespace tinytensor::simd;

TEST(AVX2Test, FeatureDetection) {
    // Check hardware feature flag
    bool supported = has_avx2();
    // On x86_64 CPU tested in Phase 0 audit, AVX2 was detected
    EXPECT_TRUE(supported);
}

TEST(AVX2Test, ElementwiseAddVectorTail) {
    // Test with size 19 (two 8-wide AVX2 chunks + 3 scalar tail elements)
    constexpr size_t N = 19;
    alignas(32) float a[N];
    alignas(32) float b[N];
    alignas(32) float out_simd[N];
    alignas(32) float out_scalar[N];

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) * 1.5f;
        b[i] = static_cast<float>(i) * 2.5f;
    }

    avx2_add_f32(a, b, out_simd, N);
    scalar_add_f32(a, b, out_scalar, N);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(out_simd[i], out_scalar[i]);
    }
}

TEST(AVX2Test, ElementwiseReluVectorTail) {
    constexpr size_t N = 21;
    alignas(32) float a[N];
    alignas(32) float out_simd[N];
    alignas(32) float out_scalar[N];

    for (size_t i = 0; i < N; ++i) {
        a[i] = (i % 2 == 0) ? -static_cast<float>(i) : static_cast<float>(i);
    }

    avx2_relu_f32(a, out_simd, N);
    scalar_relu_f32(a, out_scalar, N);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(out_simd[i], out_scalar[i]);
    }
}

TEST(AVX2Test, ElementwiseSubAndMulVectorTail) {
    constexpr size_t N = 25;
    alignas(32) float a[N];
    alignas(32) float b[N];
    alignas(32) float out_simd[N];
    alignas(32) float out_scalar[N];

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) * 2.5f + 1.0f;
        b[i] = static_cast<float>(i) * 1.2f + 0.5f;
    }

    // Sub
    avx2_sub_f32(a, b, out_simd, N);
    scalar_sub_f32(a, b, out_scalar, N);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(out_simd[i], out_scalar[i]);
    }

    // Mul
    avx2_mul_f32(a, b, out_simd, N);
    scalar_mul_f32(a, b, out_scalar, N);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(out_simd[i], out_scalar[i]);
    }
}
