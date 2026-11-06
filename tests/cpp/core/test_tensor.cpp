#include <gtest/gtest.h>
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

TEST(TensorTest, CreationAndBasics) {
    Tensor t = Tensor::zeros({2, 3}, DType::Float32);
    EXPECT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.dtype(), DType::Float32);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.nbytes(), 6 * sizeof(float));

    float* data = t.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST(TensorTest, OnesAndFull) {
    Tensor t1 = Tensor::ones({3, 2}, DType::Float32);
    for (int64_t i = 0; i < 3; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ((t1.at<float>({i, j})), 1.0f);
        }
    }

    Tensor t2 = Tensor::full({2, 2}, 3.14159, DType::Float64);
    EXPECT_EQ(t2.dtype(), DType::Float64);
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ((t2.at<double>({i, j})), 3.14159);
        }
    }
}

TEST(TensorTest, ScalarItem) {
    Tensor s = Tensor::scalar(7.5, DType::Float64);
    EXPECT_EQ(s.ndim(), 0u);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_DOUBLE_EQ(s.item<double>(), 7.5);

    Tensor m = Tensor::zeros({2, 2}, DType::Float32);
    EXPECT_THROW(m.item<float>(), ShapeError);
}

TEST(TensorTest, NegativeIndexing) {
    Tensor t = Tensor::zeros({3, 4}, DType::Float32);
    t.at<float>({2, 3}) = 99.0f;
    EXPECT_FLOAT_EQ((t.at<float>({-1, -1})), 99.0f);
    EXPECT_THROW((t.at<float>({-4, 0})), DimensionError);
}

TEST(TensorTest, TransposeSemantics) {
    Tensor t = Tensor::zeros({2, 3}, DType::Float32);
    t.at<float>({0, 1}) = 5.0f;
    t.at<float>({1, 2}) = 10.0f;

    Tensor tr = t.transpose(0, 1);
    EXPECT_EQ(tr.shape(), Shape({3, 2}));
    EXPECT_EQ(tr.strides()[0], 1);
    EXPECT_EQ(tr.strides()[1], 3);
    EXPECT_FALSE(tr.is_contiguous());

    EXPECT_FLOAT_EQ((tr.at<float>({1, 0})), 5.0f);
    EXPECT_FLOAT_EQ((tr.at<float>({2, 1})), 10.0f);

    // Modifying through view affects underlying shared storage
    tr.at<float>({1, 0}) = 42.0f;
    EXPECT_FLOAT_EQ((t.at<float>({0, 1})), 42.0f);
}

TEST(TensorTest, PermuteSemantics) {
    Tensor t = Tensor::zeros({2, 3, 4}, DType::Float32);
    t.at<float>({1, 2, 3}) = 77.0f;

    Tensor p = t.permute({2, 0, 1});
    EXPECT_EQ(p.shape(), Shape({4, 2, 3}));
    EXPECT_FLOAT_EQ((p.at<float>({3, 1, 2})), 77.0f);
}

TEST(TensorTest, ViewAndReshape) {
    Tensor t = Tensor::zeros({2, 6}, DType::Float32);
    Tensor v = t.view({3, 4});
    EXPECT_EQ(v.shape(), Shape({3, 4}));
    EXPECT_TRUE(v.is_contiguous());

    // View on non-contiguous tensor throws
    Tensor tr = t.transpose(0, 1);
    EXPECT_THROW(tr.view({4, 3}), ShapeError);

    // Reshape on non-contiguous makes contiguous copy
    Tensor r = tr.reshape({4, 3});
    EXPECT_EQ(r.shape(), Shape({4, 3}));
    EXPECT_TRUE(r.is_contiguous());
}

TEST(TensorTest, CloneIndependence) {
    Tensor t1 = Tensor::full({2, 2}, 5.0, DType::Float32);
    Tensor t2 = t1.clone();
    EXPECT_FLOAT_EQ((t2.at<float>({0, 0})), 5.0f);

    t2.at<float>({0, 0}) = 999.0f;
    EXPECT_FLOAT_EQ((t1.at<float>({0, 0})), 5.0f);
    EXPECT_FLOAT_EQ((t2.at<float>({0, 0})), 999.0f);
}

TEST(TensorTest, ContiguousCopy) {
    Tensor t = Tensor::zeros({2, 3}, DType::Float32);
    t.at<float>({0, 1}) = 1.0f;
    t.at<float>({1, 2}) = 2.0f;

    Tensor tr = t.transpose(0, 1);
    EXPECT_FALSE(tr.is_contiguous());

    Tensor c = tr.contiguous();
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_FLOAT_EQ((c.at<float>({1, 0})), 1.0f);
    EXPECT_FLOAT_EQ((c.at<float>({2, 1})), 2.0f);
}

TEST(TensorTest, TypeMismatchThrows) {
    Tensor t = Tensor::zeros({2, 2}, DType::Float32);
    EXPECT_THROW(t.data<double>(), TypeError);
    EXPECT_THROW(t.at<double>({0, 0}), TypeError);
}

TEST(TensorTest, HigherDimensionalTensors) {
    Tensor t = Tensor::zeros({2, 3, 4, 5}, DType::Float32);
    EXPECT_EQ(t.ndim(), 4u);
    EXPECT_EQ(t.numel(), 120);
    t.at<float>({1, 2, 3, 4}) = 88.0f;
    EXPECT_FLOAT_EQ((t.at<float>({1, 2, 3, 4})), 88.0f);
}

TEST(TensorTest, ZeroSizedTensor) {
    Tensor t = Tensor::zeros({0, 5}, DType::Float32);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.nbytes(), 0u);
    Tensor c = t.clone();
    EXPECT_EQ(c.numel(), 0);
}

TEST(TensorTest, DoublePrecisionArithmeticAndStorage) {
    Tensor d = Tensor::full({3, 3}, 2.718281828459045, DType::Float64);
    EXPECT_EQ(d.dtype(), DType::Float64);
    EXPECT_EQ(d.itemsize(), sizeof(double));
    EXPECT_DOUBLE_EQ((d.at<double>({2, 2})), 2.718281828459045);
}

TEST(TensorTest, OffsetAccess) {
    Tensor t = Tensor::zeros({4}, DType::Float32);
    t.at<float>({0}) = 10.0f;
    t.at<float>({1}) = 20.0f;
    t.at<float>({2}) = 30.0f;
    t.at<float>({3}) = 40.0f;

    // View with storage offset = 2
    Tensor view_offset(Shape({2}), Strides({1}), 2, t.storage(), DType::Float32);
    EXPECT_EQ(view_offset.numel(), 2);
    EXPECT_FLOAT_EQ((view_offset.at<float>({0})), 30.0f);
    EXPECT_FLOAT_EQ((view_offset.at<float>({1})), 40.0f);
}

TEST(TensorTest, MultipleTransposes) {
    Tensor t = Tensor::zeros({2, 3, 4}, DType::Float32);
    t.at<float>({1, 2, 3}) = 42.0f;
    Tensor tr1 = t.transpose(0, 2);
    Tensor tr2 = tr1.transpose(0, 2);
    EXPECT_EQ(tr2.shape(), t.shape());
    EXPECT_FLOAT_EQ((tr2.at<float>({1, 2, 3})), 42.0f);
}

TEST(TensorTest, ContiguousOnContiguousIsZeroCopy) {
    Tensor t = Tensor::zeros({3, 3}, DType::Float32);
    Tensor c = t.contiguous();
    EXPECT_EQ(c.raw_data(), t.raw_data());
}

TEST(TensorTest, EmptyFactory) {
    Tensor t = Tensor::empty({5, 5}, DType::Float32);
    EXPECT_EQ(t.numel(), 25);
    EXPECT_EQ(t.shape(), Shape({5, 5}));
}
