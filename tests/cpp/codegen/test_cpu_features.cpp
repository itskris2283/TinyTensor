#include <gtest/gtest.h>
#include "tinytensor/codegen/cpu_features.hpp"
#include "tinytensor/codegen/cpp_backend.hpp"
#include "tinytensor/graph/graph.hpp"

using namespace tinytensor;
using namespace tinytensor::simd;

namespace {

struct SyntheticCPUState {
    uint32_t max_leaf{7};
    bool hw_osxsave{true};
    bool hw_avx{true};
    bool hw_fma3{true};
    bool hw_avx2{true};
    uint64_t xcr0{0x6ULL}; // XMM (bit 1) and YMM (bit 2) enabled
};

static SyntheticCPUState g_synthetic_state;

static bool synthetic_cpuid(uint32_t leaf, uint32_t subleaf, CPUIDResult& out) {
    out = {0, 0, 0, 0};
    if (leaf > g_synthetic_state.max_leaf) {
        return false;
    }
    if (leaf == 0) {
        out.eax = g_synthetic_state.max_leaf;
        return true;
    }
    if (leaf == 1) {
        if (g_synthetic_state.hw_osxsave) out.ecx |= (1u << 27);
        if (g_synthetic_state.hw_avx) out.ecx |= (1u << 28);
        if (g_synthetic_state.hw_fma3) out.ecx |= (1u << 12);
        return true;
    }
    if (leaf == 7 && subleaf == 0) {
        if (g_synthetic_state.hw_avx2) out.ebx |= (1u << 5);
        return true;
    }
    return true;
}

static uint64_t synthetic_xgetbv(uint32_t xcr) {
    if (xcr == 0) return g_synthetic_state.xcr0;
    return 0;
}

} // anonymous namespace

TEST(CPUFeaturesTest, MaxLeafTooLowReturnsScalar) {
    g_synthetic_state = SyntheticCPUState();
    g_synthetic_state.max_leaf = 0;
    auto feat = detect_features_custom(synthetic_cpuid, synthetic_xgetbv);
    EXPECT_FALSE(feat.has_avx);
    EXPECT_FALSE(feat.has_avx2);
    EXPECT_FALSE(feat.has_fma3);
    EXPECT_EQ(supported_target_isa_custom(synthetic_cpuid, synthetic_xgetbv), TargetISA::Scalar);
}

TEST(CPUFeaturesTest, AVXUnavailableReturnsScalar) {
    g_synthetic_state = SyntheticCPUState();
    g_synthetic_state.hw_avx = false;
    auto feat = detect_features_custom(synthetic_cpuid, synthetic_xgetbv);
    EXPECT_FALSE(feat.has_avx);
    EXPECT_FALSE(feat.has_avx2);
    EXPECT_FALSE(feat.has_fma3);
    EXPECT_EQ(supported_target_isa_custom(synthetic_cpuid, synthetic_xgetbv), TargetISA::Scalar);
}

TEST(CPUFeaturesTest, OSXSAVEUnavailableReturnsScalar) {
    g_synthetic_state = SyntheticCPUState();
    g_synthetic_state.hw_osxsave = false;
    auto feat = detect_features_custom(synthetic_cpuid, synthetic_xgetbv);
    EXPECT_FALSE(feat.has_avx);
    EXPECT_FALSE(feat.has_avx2);
    EXPECT_FALSE(feat.has_fma3);
    EXPECT_EQ(supported_target_isa_custom(synthetic_cpuid, synthetic_xgetbv), TargetISA::Scalar);
}

TEST(CPUFeaturesTest, XCR0XMMBitDisabledReturnsScalar) {
    g_synthetic_state = SyntheticCPUState();
    g_synthetic_state.xcr0 = 0x4ULL; // YMM bit only, XMM missing
    auto feat = detect_features_custom(synthetic_cpuid, synthetic_xgetbv);
    EXPECT_FALSE(feat.os_avx_support);
    EXPECT_FALSE(feat.has_avx2);
    EXPECT_FALSE(feat.has_fma3);
    EXPECT_EQ(supported_target_isa_custom(synthetic_cpuid, synthetic_xgetbv), TargetISA::Scalar);
}

TEST(CPUFeaturesTest, XCR0YMMBitDisabledReturnsScalar) {
    g_synthetic_state = SyntheticCPUState();
    g_synthetic_state.xcr0 = 0x2ULL; // XMM bit only, YMM missing
    auto feat = detect_features_custom(synthetic_cpuid, synthetic_xgetbv);
    EXPECT_FALSE(feat.os_avx_support);
    EXPECT_FALSE(feat.has_avx2);
    EXPECT_FALSE(feat.has_fma3);
    EXPECT_EQ(supported_target_isa_custom(synthetic_cpuid, synthetic_xgetbv), TargetISA::Scalar);
}

TEST(CPUFeaturesTest, AVX2UnavailableReturnsScalar) {
    g_synthetic_state = SyntheticCPUState();
    g_synthetic_state.hw_avx2 = false;
    auto feat = detect_features_custom(synthetic_cpuid, synthetic_xgetbv);
    EXPECT_TRUE(feat.has_avx);
    EXPECT_FALSE(feat.has_avx2);
    EXPECT_TRUE(feat.has_fma3);
    EXPECT_EQ(supported_target_isa_custom(synthetic_cpuid, synthetic_xgetbv), TargetISA::Scalar);
}

TEST(CPUFeaturesTest, AVX2AvailableWithoutFMA3) {
    g_synthetic_state = SyntheticCPUState();
    g_synthetic_state.hw_fma3 = false;
    auto feat = detect_features_custom(synthetic_cpuid, synthetic_xgetbv);
    EXPECT_TRUE(feat.has_avx);
    EXPECT_TRUE(feat.has_avx2);
    EXPECT_FALSE(feat.has_fma3);
    EXPECT_EQ(supported_target_isa_custom(synthetic_cpuid, synthetic_xgetbv), TargetISA::AVX2);
}

TEST(CPUFeaturesTest, BothAVX2AndFMA3Available) {
    g_synthetic_state = SyntheticCPUState();
    auto feat = detect_features_custom(synthetic_cpuid, synthetic_xgetbv);
    EXPECT_TRUE(feat.has_avx);
    EXPECT_TRUE(feat.has_avx2);
    EXPECT_TRUE(feat.has_fma3);
    EXPECT_EQ(supported_target_isa_custom(synthetic_cpuid, synthetic_xgetbv), TargetISA::AVX2_FMA);
}

TEST(CPUFeaturesTest, HostConsistencyAndFeatureString) {
    auto host_feat = detect_host_features();
    EXPECT_EQ(has_avx(), host_feat.has_avx);
    EXPECT_EQ(has_avx2(), host_feat.has_avx2);
    EXPECT_EQ(has_fma3(), host_feat.has_fma3);

    std::string feat_str = supported_kernel_features();
    EXPECT_FALSE(feat_str.empty());
    if (host_feat.has_avx2) {
        EXPECT_NE(feat_str.find("AVX2"), std::string::npos);
    }
}

TEST(CPUFeaturesTest, MatMulCodegenEmitsNonFMAWhenFMADisabled) {
    Graph g;
    Node* a = g.create_input({32, 32}, DType::Float32, "A");
    Node* b = g.create_input({32, 32}, DType::Float32, "B");
    Node* c = g.matmul(a, b);
    g.mark_output(c);

    CompileOptions opts;
    opts.target_isa = TargetISA::AVX2;
    opts.enable_fma = false;
    CppBackend backend(opts);
    std::string code = backend.generate_code(g);

    // Assert that non-FMA separate add/mul intrinsics are emitted
    EXPECT_NE(code.find("_mm256_add_ps(_mm256_mul_ps("), std::string::npos);
    // Assert that FMA intrinsic is not emitted
    EXPECT_EQ(code.find("_mm256_fmadd_ps"), std::string::npos);
}

TEST(CPUFeaturesTest, MatMulCodegenEmitsFMAWhenFMAEnabled) {
    Graph g;
    Node* a = g.create_input({32, 32}, DType::Float32, "A");
    Node* b = g.create_input({32, 32}, DType::Float32, "B");
    Node* c = g.matmul(a, b);
    g.mark_output(c);

    CompileOptions opts;
    opts.target_isa = TargetISA::AVX2_FMA;
    opts.enable_fma = true;
    CppBackend backend(opts);
    std::string code = backend.generate_code(g);

    // Assert that FMA intrinsic is emitted
    EXPECT_NE(code.find("_mm256_fmadd_ps"), std::string::npos);
}

TEST(CPUFeaturesTest, MatMulExecutionAVX2NoFMACorrectness) {
    if (!has_avx2()) {
        GTEST_SKIP() << "Host does not support AVX2";
    }

    Graph g;
    Node* a = g.create_input({8, 8}, DType::Float32, "A");
    Node* b = g.create_input({8, 8}, DType::Float32, "B");
    Node* c = g.matmul(a, b);
    g.mark_output(c);

    CompileOptions opts;
    opts.target_isa = TargetISA::AVX2;
    opts.enable_fma = false; // Strictly test AVX2 non-FMA execution
    CppBackend backend(opts);
    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    Tensor in_a = Tensor::full({8, 8}, 1.5, DType::Float32);
    Tensor in_b = Tensor::full({8, 8}, 2.0, DType::Float32);
    auto outs = exec->run({in_a, in_b});
    ASSERT_EQ(outs.size(), 1u);

    // 8 * (1.5 * 2.0) = 8 * 3.0 = 24.0
    for (int64_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(outs[0].data<float>()[i], 24.0f, 1e-5f);
    }
}
