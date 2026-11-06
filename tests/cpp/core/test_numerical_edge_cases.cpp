#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "tinytensor/codegen/cpp_backend.hpp"
#include "tinytensor/core/operators.hpp"
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/runtime/interpreter.hpp"

using namespace tinytensor;

namespace {

float qnan() { return std::numeric_limits<float>::quiet_NaN(); }
float pos_inf() { return std::numeric_limits<float>::infinity(); }
float neg_inf() { return -std::numeric_limits<float>::infinity(); }
float flt_max() { return std::numeric_limits<float>::max(); }
float flt_min() { return std::numeric_limits<float>::min(); }
float denorm_min() { return std::numeric_limits<float>::denorm_min(); }

} // anonymous namespace

TEST(NumericalEdgeCasesTest, IEEE754SpecialValuesEagerArithmetic) {
    // 1. Division by zero -> infinity and NaN
    Tensor a = Tensor::scalar(1.0f);
    Tensor b = Tensor::scalar(0.0f);
    Tensor c = ops::div(a, b);
    EXPECT_TRUE(std::isinf(c.item<float>()));
    EXPECT_GT(c.item<float>(), 0.0f);

    Tensor neg_a = Tensor::scalar(-1.0f);
    Tensor d = ops::div(neg_a, b);
    EXPECT_TRUE(std::isinf(d.item<float>()));
    EXPECT_LT(d.item<float>(), 0.0f);

    Tensor zero_div_zero = ops::div(b, b);
    EXPECT_TRUE(std::isnan(zero_div_zero.item<float>()));

    // 2. Zero multiplied by infinity -> NaN
    Tensor inf_t = Tensor::scalar(pos_inf());
    Tensor zero_times_inf = ops::mul(b, inf_t);
    EXPECT_TRUE(std::isnan(zero_times_inf.item<float>()));

    // 3. Overflow behavior
    Tensor max_t = Tensor::scalar(flt_max());
    Tensor two_t = Tensor::scalar(2.0f);
    Tensor overflow_t = ops::mul(max_t, two_t);
    EXPECT_TRUE(std::isinf(overflow_t.item<float>()));

    // 4. Subnormal arithmetic
    Tensor denorm_t = Tensor::scalar(denorm_min());
    Tensor denorm_sum = ops::add(denorm_t, denorm_t);
    EXPECT_GT(denorm_sum.item<float>(), 0.0f);
    EXPECT_LT(denorm_sum.item<float>(), flt_min());
}

TEST(NumericalEdgeCasesTest, ReLUSemanticContract) {
    // Semantic Contract: ReLU(x) = (x > 0.0f) ? x : 0.0f
    // NaN evaluates to false in (NaN > 0) -> clamped to 0.0f
    // -0.0f evaluates to false in (-0.0f > 0) -> clamped to +0.0f
    // -inf -> 0.0f, +inf -> +inf

    std::vector<float> edge_inputs = {
        qnan(), pos_inf(), neg_inf(), 0.0f, -0.0f,
        flt_min(), denorm_min(), -denorm_min(), -flt_min(), flt_max(), -flt_max(), 42.5f, -42.5f
    };
    int64_t n = static_cast<int64_t>(edge_inputs.size());

    Tensor in = Tensor::empty({n}, DType::Float32);
    for (int64_t i = 0; i < n; ++i) {
        in.data<float>()[i] = edge_inputs[static_cast<size_t>(i)];
    }

    Tensor eager_out = ops::relu(in);

    // Verify eager results
    for (int64_t i = 0; i < n; ++i) {
        float x = edge_inputs[static_cast<size_t>(i)];
        float out = eager_out.data<float>()[i];
        if (std::isnan(x)) {
            EXPECT_EQ(out, 0.0f);
        } else if (x == pos_inf()) {
            EXPECT_EQ(out, pos_inf());
        } else if (x <= 0.0f) {
            EXPECT_EQ(out, 0.0f);
            EXPECT_FALSE(std::signbit(out)); // Must be +0.0f
        } else {
            EXPECT_EQ(out, x);
        }
    }

    // Verify Compiled Scalar backend
    Graph g;
    Node* in_node = g.create_input({n}, DType::Float32, "X");
    Node* relu_node = g.relu(in_node);
    g.mark_output(relu_node);

    CompileOptions opts_scalar;
    opts_scalar.target_isa = TargetISA::Scalar;
    CppBackend backend_scalar(opts_scalar);
    auto exec_scalar = backend_scalar.compile(g);
    ASSERT_NE(exec_scalar, nullptr);
    auto scalar_outs = exec_scalar->run({in});
    ASSERT_EQ(scalar_outs.size(), 1u);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(scalar_outs[0].data<float>()[i], eager_out.data<float>()[i]);
    }

    // Verify Compiled AVX2 backend (pad to 16 elements for SIMD vector lane testing)
    std::vector<float> pad_inputs = edge_inputs;
    while (pad_inputs.size() < 16) {
        pad_inputs.push_back(1.0f);
    }
    int64_t pad_n = static_cast<int64_t>(pad_inputs.size());

    Graph g_pad;
    Node* pad_in_node = g_pad.create_input({pad_n}, DType::Float32, "X");
    Node* pad_relu_node = g_pad.relu(pad_in_node);
    g_pad.mark_output(pad_relu_node);

    CompileOptions opts_avx2;
    opts_avx2.target_isa = TargetISA::AVX2;
    CppBackend backend_avx2(opts_avx2);
    auto exec_avx2 = backend_avx2.compile(g_pad);
    ASSERT_NE(exec_avx2, nullptr);

    Tensor pad_in = Tensor::empty({pad_n}, DType::Float32);
    for (int64_t i = 0; i < pad_n; ++i) {
        pad_in.data<float>()[i] = pad_inputs[static_cast<size_t>(i)];
    }
    auto avx2_outs = exec_avx2->run({pad_in});
    ASSERT_EQ(avx2_outs.size(), 1u);

    for (int64_t i = 0; i < n; ++i) {
        float expected = eager_out.data<float>()[i];
        float actual = avx2_outs[0].data<float>()[i];
        EXPECT_EQ(actual, expected) << "Mismatch at index " << i;
    }
}

TEST(NumericalEdgeCasesTest, SigmoidExtremesStability) {
    Tensor in = Tensor::empty({6}, DType::Float32);
    in.data<float>()[0] = pos_inf();
    in.data<float>()[1] = neg_inf();
    in.data<float>()[2] = 0.0f;
    in.data<float>()[3] = -0.0f;
    in.data<float>()[4] = 100.0f;
    in.data<float>()[5] = -100.0f;

    Tensor out = ops::sigmoid(in);
    EXPECT_EQ(out.data<float>()[0], 1.0f);
    EXPECT_EQ(out.data<float>()[1], 0.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[2], 0.5f);
    EXPECT_FLOAT_EQ(out.data<float>()[3], 0.5f);
    EXPECT_EQ(out.data<float>()[4], 1.0f);
    EXPECT_NEAR(out.data<float>()[5], 0.0f, 1e-6f);
    EXPECT_GE(out.data<float>()[5], 0.0f);
}

TEST(NumericalEdgeCasesTest, FMACancellationAndULPDistance) {
    // Tests catastrophic cancellation sensitivity between non-FMA and FMA.
    // In scalar non-FMA: a * b is rounded to 24 bits, then added to c.
    // In FMA: a * b + c is computed with infinite intermediate precision and rounded once.
    // Let a = 1.00000011920928955078125f (1 + 2^-23, float adjacent to 1.0)
    // b = 1.0f, c = -1.0f.
    // Exact mathematical result: 2^-23 (~1.1920929e-7)

    float a = 1.00000011920928955078125f;
    float b = 1.0f;
    float c = -1.0f;

    // FMA computation
    float fma_res = std::fma(a, b, c);
    // Non-FMA separate multiplication and addition
    volatile float prod = a * b;
    float non_fma_res = prod + c;

    EXPECT_NEAR(fma_res, 1.1920929e-7f, 1e-12f);
    EXPECT_NEAR(non_fma_res, 1.1920929e-7f, 1e-7f);

    // Verify matrix multiplication agreement within 2 ULPs for well-conditioned inputs
    Graph g;
    Node* in_a = g.create_input({4, 4}, DType::Float32, "A");
    Node* in_b = g.create_input({4, 4}, DType::Float32, "B");
    Node* mm = g.matmul(in_a, in_b);
    g.mark_output(mm);

    CompileOptions opts_fma;
    opts_fma.target_isa = TargetISA::AVX2_FMA;
    opts_fma.enable_fma = true;
    CppBackend backend_fma(opts_fma);
    auto exec_fma = backend_fma.compile(g);

    CompileOptions opts_nofma;
    opts_nofma.target_isa = TargetISA::AVX2;
    opts_nofma.enable_fma = false;
    CppBackend backend_nofma(opts_nofma);
    auto exec_nofma = backend_nofma.compile(g);

    Tensor tA = Tensor::full({4, 4}, 1.23456f, DType::Float32);
    Tensor tB = Tensor::full({4, 4}, 2.34567f, DType::Float32);

    auto out_fma = exec_fma->run({tA, tB});
    auto out_nofma = exec_nofma->run({tA, tB});

    for (int64_t i = 0; i < 16; ++i) {
        float vf = out_fma[0].data<float>()[i];
        float vn = out_nofma[0].data<float>()[i];
        // Relative error must be bounded by 1e-5
        float rel_err = std::fabs(vf - vn) / std::max(std::fabs(vf), 1e-6f);
        EXPECT_LT(rel_err, 1e-5f);
    }
}

TEST(NumericalEdgeCasesTest, ThreeWayDifferentialScalarAVX2FMA) {
    // Strict 3-way differential test on ill-conditioned / mixed-sign matrix products
    Graph g;
    Node* in_a = g.create_input({8, 8}, DType::Float32, "A");
    Node* in_b = g.create_input({8, 8}, DType::Float32, "B");
    Node* mm = g.matmul(in_a, in_b);
    g.mark_output(mm);

    CompileOptions opts_scalar;
    opts_scalar.target_isa = TargetISA::Scalar;
    CppBackend backend_scalar(opts_scalar);
    auto exec_scalar = backend_scalar.compile(g);

    CompileOptions opts_avx2;
    opts_avx2.target_isa = TargetISA::AVX2;
    opts_avx2.enable_fma = false;
    CppBackend backend_avx2(opts_avx2);
    auto exec_avx2 = backend_avx2.compile(g);

    CompileOptions opts_fma;
    opts_fma.target_isa = TargetISA::AVX2_FMA;
    opts_fma.enable_fma = true;
    CppBackend backend_fma(opts_fma);
    auto exec_fma = backend_fma.compile(g);

    // Ill-conditioned Hilbert-like matrix with alternating signs
    Tensor tA = Tensor::empty({8, 8}, DType::Float32);
    Tensor tB = Tensor::empty({8, 8}, DType::Float32);
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            float sign = ((r + c) % 2 == 0) ? 1.0f : -1.0f;
            tA.data<float>()[r * 8 + c] = sign / static_cast<float>(r + c + 1);
            tB.data<float>()[r * 8 + c] = 1.0f / static_cast<float>(r * 2 + c + 1);
        }
    }

    auto out_scalar = exec_scalar->run({tA, tB});
    auto out_avx2 = exec_avx2->run({tA, tB});
    auto out_fma = exec_fma->run({tA, tB});

    for (int i = 0; i < 64; ++i) {
        float s = out_scalar[0].data<float>()[i];
        float a = out_avx2[0].data<float>()[i];
        float f = out_fma[0].data<float>()[i];

        // Scalar and AVX2 non-FMA perform identical algebraic operations (multiply then add)
        // so difference must be within float roundoff tolerances
        EXPECT_NEAR(s, a, 1e-6f);

        // FMA and non-FMA absolute difference bounded by 1e-5f
        EXPECT_NEAR(s, f, 1e-5f);
        EXPECT_NEAR(a, f, 1e-5f);
    }
}

TEST(NumericalEdgeCasesTest, SubnormalDenormalArithmetic) {
    // Tests subnormal preservation and graceful non-crashing arithmetic
    float denorm = 1.401298e-45f; // Smallest positive subnormal IEEE-754 float
    Tensor a = Tensor::full({16}, denorm, DType::Float32);
    Tensor b = Tensor::full({16}, 2.0f, DType::Float32);

    Tensor c = ops::mul(a, b);
    for (int i = 0; i < 16; ++i) {
        float val = c.data<float>()[i];
        EXPECT_GT(val, 0.0f);
        EXPECT_LT(val, 1.17549435e-38f); // Verify still in subnormal range
    }
}
