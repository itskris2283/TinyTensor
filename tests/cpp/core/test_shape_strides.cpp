#include <gtest/gtest.h>
#include "tinytensor/core/shape.hpp"
#include "tinytensor/core/strides.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

TEST(ShapeTest, ScalarShape) {
    Shape s({});
    EXPECT_EQ(s.ndim(), 0u);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_EQ(s.to_string(), "()");
}

TEST(ShapeTest, VectorShape) {
    Shape s({5});
    EXPECT_EQ(s.ndim(), 1u);
    EXPECT_EQ(s.numel(), 5);
    EXPECT_EQ(s[0], 5);
    EXPECT_EQ(s.to_string(), "(5,)");
}

TEST(ShapeTest, MultiDimShape) {
    Shape s({2, 3, 4});
    EXPECT_EQ(s.ndim(), 3u);
    EXPECT_EQ(s.numel(), 24);
    EXPECT_EQ(s[0], 2);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(s[2], 4);
    EXPECT_EQ(s.to_string(), "(2, 3, 4)");
}

TEST(ShapeTest, ZeroDimShape) {
    Shape s({2, 0, 5});
    EXPECT_EQ(s.ndim(), 3u);
    EXPECT_EQ(s.numel(), 0);
}

TEST(ShapeTest, NegativeDimensionThrows) {
    EXPECT_THROW(Shape({2, -1, 4}), ShapeError);
}

TEST(ShapeTest, MaxRankExceededThrows) {
    std::vector<int64_t> big_dims(MAX_TENSOR_RANK + 1, 1);
    EXPECT_THROW((void)Shape{big_dims}, ResourceLimitError);
}

TEST(ShapeTest, OutOfBoundsIndexThrows) {
    Shape s({2, 3});
    EXPECT_THROW(s[2], DimensionError);
}

TEST(StridesTest, ContiguousComputation) {
    Shape s({2, 3, 4});
    Strides st = Strides::compute_contiguous(s);
    EXPECT_EQ(st.ndim(), 3u);
    EXPECT_EQ(st[0], 12);
    EXPECT_EQ(st[1], 4);
    EXPECT_EQ(st[2], 1);
    EXPECT_TRUE(st.is_contiguous(s));
}

TEST(StridesTest, ScalarStrides) {
    Shape s({});
    Strides st = Strides::compute_contiguous(s);
    EXPECT_EQ(st.ndim(), 0u);
    EXPECT_TRUE(st.is_contiguous(s));
}

TEST(StridesTest, NonContiguousDetection) {
    Shape s({2, 3});
    Strides non_contig({1, 2}); // Fortran order / transposed
    EXPECT_FALSE(non_contig.is_contiguous(s));
}
