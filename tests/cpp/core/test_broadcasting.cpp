#include <gtest/gtest.h>
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

TEST(BroadcastingTest, BroadcastShapesRules) {
    Shape a({3, 1, 4});
    Shape b({1, 5, 4});
    Shape c = Tensor::broadcast_shapes(a, b);
    EXPECT_EQ(c, Shape({3, 5, 4}));

    Shape v({4});
    Shape m({2, 4});
    EXPECT_EQ(Tensor::broadcast_shapes(v, m), Shape({2, 4}));

    Shape scalar({});
    Shape target({3, 7});
    EXPECT_EQ(Tensor::broadcast_shapes(scalar, target), Shape({3, 7}));
}

TEST(BroadcastingTest, IncompatibleShapesThrows) {
    Shape a({3, 4});
    Shape b({2, 4});
    EXPECT_THROW(Tensor::broadcast_shapes(a, b), BroadcastError);
}

TEST(BroadcastingTest, BroadcastToZeroStride) {
    Tensor t = Tensor::empty({1, 4}, DType::Float32);
    t.at<float>({0, 0}) = 1.0f;
    t.at<float>({0, 1}) = 2.0f;
    t.at<float>({0, 2}) = 3.0f;
    t.at<float>({0, 3}) = 4.0f;

    Tensor b = t.broadcast_to({3, 4});
    EXPECT_EQ(b.shape(), Shape({3, 4}));
    EXPECT_EQ(b.strides()[0], 0); // Stride 0 for broadcast dimension!
    EXPECT_EQ(b.strides()[1], 1);

    // Verify all rows read identical values without copying
    for (int64_t r = 0; r < 3; ++r) {
        EXPECT_FLOAT_EQ((b.at<float>({r, 0})), 1.0f);
        EXPECT_FLOAT_EQ((b.at<float>({r, 1})), 2.0f);
        EXPECT_FLOAT_EQ((b.at<float>({r, 2})), 3.0f);
        EXPECT_FLOAT_EQ((b.at<float>({r, 3})), 4.0f);
    }
}

TEST(BroadcastingTest, BroadcastScalarToND) {
    Tensor s = Tensor::scalar(42.0f, DType::Float32);
    Tensor b = s.broadcast_to({2, 3});
    EXPECT_EQ(b.shape(), Shape({2, 3}));
    EXPECT_EQ(b.strides()[0], 0);
    EXPECT_EQ(b.strides()[1], 0);
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((b.at<float>({i, j})), 42.0f);
        }
    }
}

TEST(BroadcastingTest, IncompatibleBroadcastToThrows) {
    Tensor t = Tensor::empty({2, 3}, DType::Float32);
    EXPECT_THROW(t.broadcast_to({4, 3}), BroadcastError);
}
