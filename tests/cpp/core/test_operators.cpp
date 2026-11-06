#include <gtest/gtest.h>
#include <cmath>
#include "tinytensor/core/operators.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;
using namespace tinytensor::ops;

TEST(OperatorsTest, AddSubMulDiv) {
    Tensor a = Tensor::full({2, 3}, 10.0, DType::Float32);
    Tensor b = Tensor::full({2, 3}, 2.0, DType::Float32);

    Tensor c_add = add(a, b);
    EXPECT_FLOAT_EQ((c_add.at<float>({0, 0})), 12.0f);

    Tensor c_sub = sub(a, b);
    EXPECT_FLOAT_EQ((c_sub.at<float>({0, 0})), 8.0f);

    Tensor c_mul = mul(a, b);
    EXPECT_FLOAT_EQ((c_mul.at<float>({0, 0})), 20.0f);

    Tensor c_div = div(a, b);
    EXPECT_FLOAT_EQ((c_div.at<float>({0, 0})), 5.0f);
}

TEST(OperatorsTest, BroadcastBinaryOps) {
    Tensor a = Tensor::zeros({2, 3}, DType::Float32);
    Tensor b = Tensor::full({3}, 5.0, DType::Float32);

    Tensor c = add(a, b);
    EXPECT_EQ(c.shape(), Shape({2, 3}));
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((c.at<float>({i, j})), 5.0f);
        }
    }
}

TEST(OperatorsTest, Relu) {
    Tensor t = Tensor::empty({4}, DType::Float32);
    t.at<float>({0}) = -3.5f;
    t.at<float>({1}) = 0.0f;
    t.at<float>({2}) = 2.5f;
    t.at<float>({3}) = -0.001f;

    Tensor r = relu(t);
    EXPECT_FLOAT_EQ((r.at<float>({0})), 0.0f);
    EXPECT_FLOAT_EQ((r.at<float>({1})), 0.0f);
    EXPECT_FLOAT_EQ((r.at<float>({2})), 2.5f);
    EXPECT_FLOAT_EQ((r.at<float>({3})), 0.0f);
}

TEST(OperatorsTest, SigmoidNumericallyStable) {
    Tensor t = Tensor::empty({4}, DType::Float32);
    t.at<float>({0}) = 0.0f;
    t.at<float>({1}) = 100.0f;  // Large positive -> should be ~1.0, not inf or nan
    t.at<float>({2}) = -100.0f; // Large negative -> should be ~0.0, not overflow
    t.at<float>({3}) = 2.0f;

    Tensor s = sigmoid(t);
    EXPECT_NEAR((s.at<float>({0})), 0.5f, 1e-6);
    EXPECT_NEAR((s.at<float>({1})), 1.0f, 1e-6);
    EXPECT_NEAR((s.at<float>({2})), 0.0f, 1e-6);
    float expected = 1.0f / (1.0f + std::exp(-2.0f));
    EXPECT_NEAR((s.at<float>({3})), expected, 1e-6);
}

TEST(OperatorsTest, Matmul2D) {
    // A: 2x3, B: 3x2
    Tensor a = Tensor::empty({2, 3}, DType::Float32);
    // Row 0: 1, 2, 3
    a.at<float>({0, 0}) = 1.0f; a.at<float>({0, 1}) = 2.0f; a.at<float>({0, 2}) = 3.0f;
    // Row 1: 4, 5, 6
    a.at<float>({1, 0}) = 4.0f; a.at<float>({1, 1}) = 5.0f; a.at<float>({1, 2}) = 6.0f;

    Tensor b = Tensor::empty({3, 2}, DType::Float32);
    b.at<float>({0, 0}) = 7.0f; b.at<float>({0, 1}) = 8.0f;
    b.at<float>({1, 0}) = 9.0f; b.at<float>({1, 1}) = 1.0f;
    b.at<float>({2, 0}) = 2.0f; b.at<float>({2, 1}) = 3.0f;

    Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), Shape({2, 2}));
    // c[0, 0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    EXPECT_FLOAT_EQ((c.at<float>({0, 0})), 31.0f);
    // c[0, 1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    EXPECT_FLOAT_EQ((c.at<float>({0, 1})), 19.0f);
    // c[1, 0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    EXPECT_FLOAT_EQ((c.at<float>({1, 0})), 85.0f);
    // c[1, 1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    EXPECT_FLOAT_EQ((c.at<float>({1, 1})), 55.0f);
}

TEST(OperatorsTest, MatmulDimensionMismatchThrows) {
    Tensor a = Tensor::zeros({2, 3}, DType::Float32);
    Tensor b = Tensor::zeros({4, 2}, DType::Float32);
    EXPECT_THROW(matmul(a, b), ShapeError);
}

TEST(OperatorsTest, ReduceSumAllAndAxes) {
    Tensor t = Tensor::empty({2, 3}, DType::Float32);
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            t.at<float>({i, j}) = static_cast<float>(i * 3 + j + 1); // 1, 2, 3, 4, 5, 6 -> sum = 21
        }
    }

    Tensor s_all = reduce_sum(t);
    EXPECT_EQ(s_all.ndim(), 0u);
    EXPECT_FLOAT_EQ(s_all.item<float>(), 21.0f);

    Tensor s_ax0 = reduce_sum(t, {0}, false);
    EXPECT_EQ(s_ax0.shape(), Shape({3}));
    EXPECT_FLOAT_EQ((s_ax0.at<float>({0})), 1.0f + 4.0f);
    EXPECT_FLOAT_EQ((s_ax0.at<float>({1})), 2.0f + 5.0f);
    EXPECT_FLOAT_EQ((s_ax0.at<float>({2})), 3.0f + 6.0f);

    Tensor s_keep = reduce_sum(t, {1}, true);
    EXPECT_EQ(s_keep.shape(), Shape({2, 1}));
    EXPECT_FLOAT_EQ((s_keep.at<float>({0, 0})), 6.0f);
    EXPECT_FLOAT_EQ((s_keep.at<float>({1, 0})), 15.0f);
}

TEST(OperatorsTest, ReduceMean) {
    Tensor t = Tensor::empty({2, 2}, DType::Float32);
    t.at<float>({0, 0}) = 2.0f; t.at<float>({0, 1}) = 4.0f;
    t.at<float>({1, 0}) = 6.0f; t.at<float>({1, 1}) = 8.0f;

    Tensor m = reduce_mean(t);
    EXPECT_FLOAT_EQ(m.item<float>(), 5.0f);
}

TEST(OperatorsTest, Float64BasicOperations) {
    Tensor a = Tensor::full({2, 2}, 1.5, DType::Float64);
    Tensor b = Tensor::full({2, 2}, 2.5, DType::Float64);

    Tensor c = add(a, b);
    EXPECT_EQ(c.dtype(), DType::Float64);
    EXPECT_DOUBLE_EQ((c.at<double>({0, 0})), 4.0);

    Tensor d = mul(a, b);
    EXPECT_DOUBLE_EQ((d.at<double>({0, 0})), 3.75);
}

TEST(OperatorsTest, BatchedMatMul3D) {
    // 2 batches of 2x2 matrices
    Tensor a = Tensor::zeros({2, 2, 2}, DType::Float32);
    Tensor b = Tensor::zeros({2, 2, 2}, DType::Float32);

    // Batch 0: identity
    a.at<float>({0, 0, 0}) = 1.0f; a.at<float>({0, 1, 1}) = 1.0f;
    b.at<float>({0, 0, 0}) = 5.0f; b.at<float>({0, 0, 1}) = 6.0f;
    b.at<float>({0, 1, 0}) = 7.0f; b.at<float>({0, 1, 1}) = 8.0f;

    // Batch 1: scale by 2
    a.at<float>({1, 0, 0}) = 2.0f; a.at<float>({1, 1, 1}) = 2.0f;
    b.at<float>({1, 0, 0}) = 1.0f; b.at<float>({1, 1, 1}) = 1.0f;

    Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), Shape({2, 2, 2}));
    EXPECT_FLOAT_EQ((c.at<float>({0, 0, 0})), 5.0f);
    EXPECT_FLOAT_EQ((c.at<float>({0, 1, 1})), 8.0f);
    EXPECT_FLOAT_EQ((c.at<float>({1, 0, 0})), 2.0f);
    EXPECT_FLOAT_EQ((c.at<float>({1, 1, 1})), 2.0f);
}

TEST(OperatorsTest, DivisionByZeroIEEE754) {
    Tensor a = Tensor::full({1}, 1.0, DType::Float32);
    Tensor b = Tensor::full({1}, 0.0, DType::Float32);
    Tensor c = div(a, b);
    EXPECT_TRUE(std::isinf(c.item<float>()));
}

TEST(OperatorsTest, ReduceSumMultiAxes) {
    Tensor t = Tensor::ones({2, 3, 4}, DType::Float32);
    Tensor s = reduce_sum(t, {0, 2}, false);
    EXPECT_EQ(s.shape(), Shape({3}));
    // Each element is sum over 2 * 4 = 8 ones
    EXPECT_FLOAT_EQ((s.at<float>({0})), 8.0f);
    EXPECT_FLOAT_EQ((s.at<float>({1})), 8.0f);
    EXPECT_FLOAT_EQ((s.at<float>({2})), 8.0f);
}

TEST(OperatorsTest, ReduceMeanKeepdims) {
    Tensor t = Tensor::full({4, 6}, 10.0, DType::Float32);
    Tensor m = reduce_mean(t, {1}, true);
    EXPECT_EQ(m.shape(), Shape({4, 1}));
    EXPECT_FLOAT_EQ((m.at<float>({0, 0})), 10.0f);
    EXPECT_FLOAT_EQ((m.at<float>({3, 0})), 10.0f);
}

TEST(OperatorsTest, NonContiguousElementwise) {
    Tensor a = Tensor::zeros({3, 3}, DType::Float32);
    a.at<float>({0, 1}) = 10.0f;
    Tensor tr_a = a.transpose(0, 1); // non-contiguous view

    Tensor b = Tensor::full({3, 3}, 2.0f, DType::Float32);
    Tensor c = add(tr_a, b);
    EXPECT_FLOAT_EQ((c.at<float>({1, 0})), 12.0f);
}

TEST(OperatorsTest, Float64AdvancedOperations) {
    Tensor a = Tensor::full({2, 2}, 10.0, DType::Float64);
    Tensor b = Tensor::full({2, 2}, 2.5, DType::Float64);

    Tensor c_add = add(a, b);
    Tensor c_sub = sub(a, b);
    Tensor c_mul = mul(a, b);
    Tensor c_div = div(a, b);

    EXPECT_DOUBLE_EQ((c_add.at<double>({0, 0})), 12.5);
    EXPECT_DOUBLE_EQ((c_sub.at<double>({0, 0})), 7.5);
    EXPECT_DOUBLE_EQ((c_mul.at<double>({0, 0})), 25.0);
    EXPECT_DOUBLE_EQ((c_div.at<double>({0, 0})), 4.0);

    Tensor t64 = Tensor::empty({3}, DType::Float64);
    t64.at<double>({0}) = -5.0;
    t64.at<double>({1}) = 0.0;
    t64.at<double>({2}) = 5.0;

    Tensor r64 = relu(t64);
    EXPECT_DOUBLE_EQ((r64.at<double>({0})), 0.0);
    EXPECT_DOUBLE_EQ((r64.at<double>({2})), 5.0);

    Tensor s64 = sigmoid(t64);
    EXPECT_NEAR((s64.at<double>({1})), 0.5, 1e-7);

    // Float64 MatMul
    Tensor m1 = Tensor::full({2, 3}, 1.0, DType::Float64);
    Tensor m2 = Tensor::full({3, 2}, 2.0, DType::Float64);
    Tensor mm = matmul(m1, m2);
    EXPECT_DOUBLE_EQ((mm.at<double>({0, 0})), 6.0);
}

TEST(OperatorsTest, NonContiguousAndFloat64Coverage) {
    // 1. Non-contiguous Float64 binary op (broadcast and strides)
    Tensor a64 = Tensor::zeros({3, 3}, DType::Float64);
    a64.at<double>({0, 1}) = 20.0;
    Tensor tr_a64 = a64.transpose(0, 1);
    Tensor b64 = Tensor::full({3}, 5.0, DType::Float64); // Broadcasted + non-contiguous
    Tensor c64 = add(tr_a64, b64);
    EXPECT_DOUBLE_EQ((c64.at<double>({1, 0})), 25.0);

    // 2. Non-contiguous unary op on Float32
    Tensor a32 = Tensor::empty({2, 2}, DType::Float32);
    a32.at<float>({0, 0}) = -1.0f; a32.at<float>({0, 1}) = 2.0f;
    a32.at<float>({1, 0}) = -3.0f; a32.at<float>({1, 1}) = 4.0f;
    Tensor tr_a32 = a32.transpose(0, 1);
    Tensor r32 = relu(tr_a32);
    EXPECT_FLOAT_EQ((r32.at<float>({1, 0})), 2.0f);
    EXPECT_FLOAT_EQ((r32.at<float>({0, 1})), 0.0f);

    // 3. Non-contiguous unary op on Float64
    Tensor s_tr64 = sigmoid(tr_a64);
    EXPECT_GT((s_tr64.at<double>({1, 0})), 0.5);

    // 4. Non-contiguous reductions on Float64
    Tensor sum64 = reduce_sum(tr_a64, {0}, false);
    EXPECT_DOUBLE_EQ((sum64.at<double>({0})), 20.0);
    Tensor mean64 = reduce_mean(tr_a64, {0}, false);
    EXPECT_NEAR((mean64.at<double>({0})), 20.0 / 3.0, 1e-7);
}
