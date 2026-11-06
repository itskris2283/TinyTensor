#include <gtest/gtest.h>
#include <cmath>
#include "tinytensor/core/operators.hpp"
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

TEST(CoreBranchExpansionTest, TensorNegativeIndexingAndBounds) {
    Tensor t = Tensor::empty({3, 4}, DType::Float32);
    for (int64_t i = 0; i < 12; ++i) {
        t.data<float>()[i] = static_cast<float>(i);
    }

    // Negative indexing
    EXPECT_FLOAT_EQ(t.at<float>({-1, -1}), 11.0f);
    EXPECT_FLOAT_EQ(t.at<float>({-3, -4}), 0.0f);
    EXPECT_FLOAT_EQ(t.at<float>({-2, 1}), 5.0f);

    // Negative index out of bounds
    EXPECT_THROW(t.at<float>({-4, 0}), DimensionError);
    EXPECT_THROW(t.at<float>({0, -5}), DimensionError);
    EXPECT_THROW(t.at<float>({3, 0}), DimensionError);
    EXPECT_THROW(t.at<float>({0, 4}), DimensionError);
}

TEST(CoreBranchExpansionTest, ZeroSizedTensorSemantics) {
    Tensor zero_t = Tensor::empty({0, 4}, DType::Float32);
    EXPECT_EQ(zero_t.numel(), 0);
    EXPECT_THROW(zero_t.item<float>(), ShapeError);
    EXPECT_THROW(zero_t.at<float>({0, 0}), DimensionError);

    Tensor clone_zero = zero_t.clone();
    EXPECT_EQ(clone_zero.numel(), 0);
}

TEST(CoreBranchExpansionTest, ReshapeInferenceAndErrors) {
    Tensor t = Tensor::empty({2, 6}, DType::Float32);

    // Valid reshape
    Tensor r1 = t.reshape(Shape({3, 4}));
    EXPECT_EQ(r1.shape(), Shape({3, 4}));

    Tensor r2 = t.reshape(Shape({1, 12}));
    EXPECT_EQ(r2.shape(), Shape({1, 12}));

    // Negative dimension in Shape throws ShapeError
    EXPECT_THROW(Shape({-1, -1}), ShapeError);
    EXPECT_THROW(Shape({3, -1}), ShapeError);

    // Total elements mismatch throws ShapeError
    EXPECT_THROW(t.reshape(Shape({5, 5})), ShapeError);
}

TEST(CoreBranchExpansionTest, PermuteValidationAndErrors) {
    Tensor t = Tensor::empty({2, 3, 4}, DType::Float32);

    // Rank mismatch throws DimensionError
    EXPECT_THROW(t.permute({0, 1}), DimensionError);
    EXPECT_THROW(t.permute({0, 1, 2, 3}), DimensionError);

    // Duplicate axes throw DimensionError
    EXPECT_THROW(t.permute({0, 1, 1}), DimensionError);

    // Out of bounds axes throw DimensionError
    EXPECT_THROW(t.permute({0, 1, 5}), DimensionError);
    EXPECT_THROW(t.permute({-5, 1, 2}), DimensionError);

    // Valid permutation
    Tensor p = t.permute({2, 0, 1});
    EXPECT_EQ(p.shape(), Shape({4, 2, 3}));
}

TEST(CoreBranchExpansionTest, Float64OperatorsExhaustive) {
    Tensor a = Tensor::full({2, 2}, 3.0, DType::Float64);
    Tensor b = Tensor::full({2, 2}, 1.5, DType::Float64);

    // Float64 Add, Sub, Mul, Div
    Tensor add_res = ops::add(a, b);
    EXPECT_DOUBLE_EQ(add_res.data<double>()[0], 4.5);

    Tensor sub_res = ops::sub(a, b);
    EXPECT_DOUBLE_EQ(sub_res.data<double>()[0], 1.5);

    Tensor mul_res = ops::mul(a, b);
    EXPECT_DOUBLE_EQ(mul_res.data<double>()[0], 4.5);

    Tensor div_res = ops::div(a, b);
    EXPECT_DOUBLE_EQ(div_res.data<double>()[0], 2.0);

    // Float64 ReLU and Sigmoid
    Tensor relu_res = ops::relu(sub_res);
    EXPECT_DOUBLE_EQ(relu_res.data<double>()[0], 1.5);

    Tensor sig_res = ops::sigmoid(Tensor::zeros({1}, DType::Float64));
    EXPECT_NEAR(sig_res.item<double>(), 0.5, 1e-7);

    // Float64 MatMul
    Tensor mm_res = ops::matmul(a, b);
    EXPECT_DOUBLE_EQ(mm_res.data<double>()[0], 9.0);

    // Float64 ReduceSum and ReduceMean
    Tensor sum_res = ops::reduce_sum(a);
    EXPECT_DOUBLE_EQ(sum_res.item<double>(), 12.0);

    Tensor mean_res = ops::reduce_mean(a);
    EXPECT_DOUBLE_EQ(mean_res.item<double>(), 3.0);
}

TEST(CoreBranchExpansionTest, NonContiguousOperandsBinaryOps) {
    Tensor t1 = Tensor::empty({3, 3}, DType::Float32);
    Tensor t2 = Tensor::empty({3, 3}, DType::Float32);
    for (int64_t i = 0; i < 9; ++i) {
        t1.data<float>()[i] = static_cast<float>(i);
        t2.data<float>()[i] = static_cast<float>(10 + i);
    }

    Tensor t1_t = t1.transpose(0, 1);
    Tensor t2_t = t2.transpose(0, 1);

    EXPECT_FALSE(t1_t.is_contiguous());
    EXPECT_FALSE(t2_t.is_contiguous());

    Tensor sum_t = ops::add(t1_t, t2_t);
    EXPECT_TRUE(sum_t.is_contiguous());

    for (int64_t r = 0; r < 3; ++r) {
        for (int64_t c = 0; c < 3; ++c) {
            float expected = t1.at<float>({c, r}) + t2.at<float>({c, r});
            EXPECT_FLOAT_EQ(sum_t.at<float>({r, c}), expected);
        }
    }
}

TEST(CoreBranchExpansionTest, OperatorsErrorBranches) {
    // Incompatible shapes for binary op throws BroadcastError
    Tensor a = Tensor::ones({2, 3}, DType::Float32);
    Tensor b = Tensor::ones({4, 5}, DType::Float32);
    EXPECT_THROW(ops::add(a, b), BroadcastError);
    EXPECT_THROW(ops::sub(a, b), BroadcastError);
    EXPECT_THROW(ops::mul(a, b), BroadcastError);
    EXPECT_THROW(ops::div(a, b), BroadcastError);

    // Matmul scalar (rank < 1) throws DimensionError
    Tensor s0 = Tensor::scalar(1.0f);
    Tensor m1 = Tensor::ones({3, 3}, DType::Float32);
    EXPECT_THROW(ops::matmul(s0, m1), DimensionError);
    EXPECT_THROW(ops::matmul(m1, s0), DimensionError);

    // Matmul inner dimension mismatch throws ShapeError
    Tensor m2 = Tensor::ones({2, 4}, DType::Float32);
    Tensor m3 = Tensor::ones({5, 3}, DType::Float32);
    EXPECT_THROW(ops::matmul(m2, m3), ShapeError);

    // Reduction axis out of bounds throws DimensionError
    EXPECT_THROW(ops::reduce_sum(a, {5}), DimensionError);
    EXPECT_THROW(ops::reduce_mean(a, {-5}), DimensionError);
}
