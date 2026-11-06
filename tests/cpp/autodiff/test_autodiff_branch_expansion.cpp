#include <gtest/gtest.h>
#include <cmath>
#include "tinytensor/autodiff/gradient.hpp"
#include "tinytensor/core/operators.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

TEST(AutodiffBranchExpansionTest, UnbroadcastEdgeCases) {
    // 1. Identity unbroadcast
    Tensor g1 = Tensor::ones({2, 3, 4}, DType::Float32);
    Tensor u1 = Autodiff::unbroadcast(g1, Shape({2, 3, 4}));
    EXPECT_EQ(u1.shape(), Shape({2, 3, 4}));

    // 2. Unbroadcast to scalar (target_rank == 0)
    Tensor g2 = Tensor::full({2, 3}, 2.0f, DType::Float32);
    Tensor u2 = Autodiff::unbroadcast(g2, Shape({}));
    EXPECT_EQ(u2.shape(), Shape({}));
    EXPECT_FLOAT_EQ(u2.item<float>(), 12.0f); // 2.0 * 6 = 12.0

    // 3. Unbroadcast reducing leading dimensions (grad_rank > target_rank)
    Tensor g3 = Tensor::full({2, 3, 4}, 1.0f, DType::Float32);
    Tensor u3 = Autodiff::unbroadcast(g3, Shape({4}));
    EXPECT_EQ(u3.shape(), Shape({4}));
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(u3.data<float>()[i], 6.0f); // reduced dims 0 (size 2) and 1 (size 3)
    }

    // 4. Unbroadcast reducing singleton dimensions
    Tensor g4 = Tensor::full({2, 3, 4}, 1.0f, DType::Float32);
    Tensor u4 = Autodiff::unbroadcast(g4, Shape({2, 1, 4}));
    EXPECT_EQ(u4.shape(), Shape({2, 1, 4}));
    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(u4.data<float>()[i], 3.0f); // reduced dim 1 of size 3
    }
}

TEST(AutodiffBranchExpansionTest, NullLossNodeThrows) {
    Graph g;
    Node* in = g.create_input({2, 2}, DType::Float32);
    Tensor t = Tensor::ones({2, 2}, DType::Float32);
    std::unordered_map<const Node*, Tensor> bindings = {{in, t}};

    EXPECT_THROW(Autodiff::compute_gradients(g, nullptr, bindings), TinyTensorException);
}

TEST(AutodiffBranchExpansionTest, SubtractionGradientBroadcasting) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32, "a");
    Node* b = g.create_input({3}, DType::Float32, "b");
    Node* diff = g.sub(a, b);
    Node* loss = g.reduce_sum(diff);
    g.mark_output(loss);

    Tensor ta = Tensor::full({2, 3}, 5.0f, DType::Float32);
    Tensor tb = Tensor::full({3}, 2.0f, DType::Float32);

    auto grads = Autodiff::compute_gradients(g, loss, {{a, ta}, {b, tb}});

    // d(loss)/da = 1, shape {2, 3}
    ASSERT_NE(grads.find(a), grads.end());
    EXPECT_EQ(grads[a].shape(), Shape({2, 3}));
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(grads[a].data<float>()[i], 1.0f);
    }

    // d(loss)/db = -2 (summed along batch dim 0 of size 2)
    ASSERT_NE(grads.find(b), grads.end());
    EXPECT_EQ(grads[b].shape(), Shape({3}));
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(grads[b].data<float>()[i], -2.0f);
    }
}

TEST(AutodiffBranchExpansionTest, DivisionGradientBroadcasting) {
    Graph g;
    Node* a = g.create_input({2, 2}, DType::Float32, "a");
    Node* b = g.create_input({2}, DType::Float32, "b");
    Node* div = g.div(a, b);
    Node* loss = g.reduce_sum(div);
    g.mark_output(loss);

    // a = [[4, 6], [8, 10]], b = [2, 3]
    Tensor ta = Tensor::empty({2, 2}, DType::Float32);
    ta.data<float>()[0] = 4.0f;
    ta.data<float>()[1] = 6.0f;
    ta.data<float>()[2] = 8.0f;
    ta.data<float>()[3] = 10.0f;

    Tensor tb = Tensor::empty({2}, DType::Float32);
    tb.data<float>()[0] = 2.0f;
    tb.data<float>()[1] = 3.0f;

    auto grads = Autodiff::compute_gradients(g, loss, {{a, ta}, {b, tb}});

    // d(loss)/da = 1 / b -> [[1/2, 1/3], [1/2, 1/3]]
    ASSERT_NE(grads.find(a), grads.end());
    EXPECT_EQ(grads[a].shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(grads[a].data<float>()[0], 0.5f);
    EXPECT_FLOAT_EQ(grads[a].data<float>()[1], 1.0f / 3.0f);
    EXPECT_FLOAT_EQ(grads[a].data<float>()[2], 0.5f);
    EXPECT_FLOAT_EQ(grads[a].data<float>()[3], 1.0f / 3.0f);

    // d(loss)/db = sum over rows of (-a / b^2)
    // col 0: -4/(4) + -8/(4) = -1 + -2 = -3.0
    // col 1: -6/(9) + -10/(9) = -16/9
    ASSERT_NE(grads.find(b), grads.end());
    EXPECT_EQ(grads[b].shape(), Shape({2}));
    EXPECT_NEAR(grads[b].data<float>()[0], -3.0f, 1e-5f);
    EXPECT_NEAR(grads[b].data<float>()[1], -16.0f / 9.0f, 1e-5f);
}

TEST(AutodiffBranchExpansionTest, Transpose3DGradient) {
    Graph g;
    Node* a = g.create_input({2, 3, 4}, DType::Float32, "a");
    Node* tr = g.transpose(a, 0, 2); // Shape becomes {4, 3, 2}
    Node* loss = g.reduce_sum(tr);
    g.mark_output(loss);

    Tensor ta = Tensor::ones({2, 3, 4}, DType::Float32);
    auto grads = Autodiff::compute_gradients(g, loss, {{a, ta}});

    ASSERT_NE(grads.find(a), grads.end());
    EXPECT_EQ(grads[a].shape(), Shape({2, 3, 4}));
    for (int64_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(grads[a].data<float>()[i], 1.0f);
    }
}

TEST(AutodiffBranchExpansionTest, UnusedGraphNodeSkippedInBackward) {
    Graph g;
    Node* a = g.create_input({2, 2}, DType::Float32, "a");
    Node* b = g.create_input({2, 2}, DType::Float32, "b"); // unused operand
    Node* loss = g.relu(a);
    g.mark_output(loss);

    Tensor ta = Tensor::full({2, 2}, 3.0f, DType::Float32);
    Tensor tb = Tensor::full({2, 2}, 1.0f, DType::Float32);

    auto grads = Autodiff::compute_gradients(g, loss, {{a, ta}, {b, tb}});

    ASSERT_NE(grads.find(a), grads.end());
    // b did not participate in loss, so it has no gradient
    EXPECT_EQ(grads.find(b), grads.end());
}
