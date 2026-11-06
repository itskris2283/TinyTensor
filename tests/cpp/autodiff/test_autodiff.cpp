#include <gtest/gtest.h>
#include <cmath>
#include "tinytensor/autodiff/gradient.hpp"
#include "tinytensor/core/operators.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/runtime/interpreter.hpp"

using namespace tinytensor;

TEST(AutodiffTest, AddGrad) {
    Graph g;
    Node* a = g.create_input({2}, DType::Float32, "A");
    Node* b = g.create_input({2}, DType::Float32, "B");
    Node* c = g.add(a, b);
    Node* loss = g.reduce_sum(c);
    g.mark_output(loss);

    Tensor t_a = Tensor::full({2}, 3.0f, DType::Float32);
    Tensor t_b = Tensor::full({2}, 5.0f, DType::Float32);

    auto grads = Autodiff::compute_gradients(g, loss, {{a, t_a}, {b, t_b}});
    EXPECT_FLOAT_EQ((grads[a].at<float>({0})), 1.0f);
    EXPECT_FLOAT_EQ((grads[a].at<float>({1})), 1.0f);
    EXPECT_FLOAT_EQ((grads[b].at<float>({0})), 1.0f);
    EXPECT_FLOAT_EQ((grads[b].at<float>({1})), 1.0f);
}

TEST(AutodiffTest, SubGrad) {
    Graph g;
    Node* a = g.create_input({2}, DType::Float32, "A");
    Node* b = g.create_input({2}, DType::Float32, "B");
    Node* c = g.sub(a, b);
    Node* loss = g.reduce_sum(c);
    g.mark_output(loss);

    Tensor t_a = Tensor::full({2}, 3.0f, DType::Float32);
    Tensor t_b = Tensor::full({2}, 5.0f, DType::Float32);

    auto grads = Autodiff::compute_gradients(g, loss, {{a, t_a}, {b, t_b}});
    EXPECT_FLOAT_EQ((grads[a].at<float>({0})), 1.0f);
    EXPECT_FLOAT_EQ((grads[b].at<float>({0})), -1.0f);
}

TEST(AutodiffTest, MulGrad) {
    Graph g;
    Node* a = g.create_input({2}, DType::Float32, "A");
    Node* b = g.create_input({2}, DType::Float32, "B");
    Node* c = g.mul(a, b);
    Node* loss = g.reduce_sum(c);
    g.mark_output(loss);

    Tensor t_a = Tensor::empty({2}, DType::Float32);
    t_a.at<float>({0}) = 2.0f; t_a.at<float>({1}) = 4.0f;
    Tensor t_b = Tensor::empty({2}, DType::Float32);
    t_b.at<float>({0}) = 3.0f; t_b.at<float>({1}) = 5.0f;

    auto grads = Autodiff::compute_gradients(g, loss, {{a, t_a}, {b, t_b}});
    EXPECT_FLOAT_EQ((grads[a].at<float>({0})), 3.0f);
    EXPECT_FLOAT_EQ((grads[a].at<float>({1})), 5.0f);
    EXPECT_FLOAT_EQ((grads[b].at<float>({0})), 2.0f);
    EXPECT_FLOAT_EQ((grads[b].at<float>({1})), 4.0f);
}

TEST(AutodiffTest, MultiPathAccumulation) {
    // Y = X + X -> dY/dX = 2
    Graph g;
    Node* x = g.create_input({2}, DType::Float32, "X");
    Node* y = g.add(x, x);
    Node* loss = g.reduce_sum(y);
    g.mark_output(loss);

    Tensor t_x = Tensor::full({2}, 10.0f, DType::Float32);
    auto grads = Autodiff::compute_gradients(g, loss, {{x, t_x}});
    EXPECT_FLOAT_EQ((grads[x].at<float>({0})), 2.0f);
    EXPECT_FLOAT_EQ((grads[x].at<float>({1})), 2.0f);
}

TEST(AutodiffTest, ReluGrad) {
    Graph g;
    Node* x = g.create_input({3}, DType::Float32, "X");
    Node* r = g.relu(x);
    Node* loss = g.reduce_sum(r);
    g.mark_output(loss);

    Tensor t_x = Tensor::empty({3}, DType::Float32);
    t_x.at<float>({0}) = -2.0f;
    t_x.at<float>({1}) = 0.0f;
    t_x.at<float>({2}) = 3.0f;

    auto grads = Autodiff::compute_gradients(g, loss, {{x, t_x}});
    EXPECT_FLOAT_EQ((grads[x].at<float>({0})), 0.0f);
    EXPECT_FLOAT_EQ((grads[x].at<float>({1})), 0.0f);
    EXPECT_FLOAT_EQ((grads[x].at<float>({2})), 1.0f);
}

TEST(AutodiffTest, MatMulGrad) {
    Graph g;
    Node* a = g.create_input({2, 2}, DType::Float32, "A");
    Node* b = g.create_input({2, 2}, DType::Float32, "B");
    Node* c = g.matmul(a, b);
    Node* loss = g.reduce_sum(c);
    g.mark_output(loss);

    Tensor t_a = Tensor::ones({2, 2}, DType::Float32);
    Tensor t_b = Tensor::full({2, 2}, 3.0f, DType::Float32);

    auto grads = Autodiff::compute_gradients(g, loss, {{a, t_a}, {b, t_b}});
    // For loss = sum(A @ B) where all elements of dL/dC = 1:
    // dL/dA = 1 @ B.T = [[3+3, 3+3], [3+3, 3+3]] = [[6, 6], [6, 6]]
    EXPECT_FLOAT_EQ((grads[a].at<float>({0, 0})), 6.0f);
    EXPECT_FLOAT_EQ((grads[a].at<float>({1, 1})), 6.0f);
}

TEST(AutodiffTest, BroadcastUnbroadcasting) {
    // A: [2, 3], B: [3] -> C: [2, 3] = A + B
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32, "A");
    Node* b = g.create_input({3}, DType::Float32, "B");
    Node* c = g.add(a, b);
    Node* loss = g.reduce_sum(c);
    g.mark_output(loss);

    Tensor t_a = Tensor::zeros({2, 3}, DType::Float32);
    Tensor t_b = Tensor::zeros({3}, DType::Float32);

    auto grads = Autodiff::compute_gradients(g, loss, {{a, t_a}, {b, t_b}});
    EXPECT_EQ(grads[a].shape(), Shape({2, 3}));
    EXPECT_EQ(grads[b].shape(), Shape({3}));
    // B was broadcast across 2 rows, so its gradient is 1 + 1 = 2 for each element!
    EXPECT_FLOAT_EQ((grads[b].at<float>({0})), 2.0f);
    EXPECT_FLOAT_EQ((grads[b].at<float>({1})), 2.0f);
    EXPECT_FLOAT_EQ((grads[b].at<float>({2})), 2.0f);
}

TEST(AutodiffTest, FiniteDifferencesVerification) {
    // Numerical gradient check using central finite differences:
    // f'(x) ~ (f(x + eps) - f(x - eps)) / (2 * eps)
    Graph g;
    Node* x = g.create_input({2}, DType::Float64, "X");
    Node* sig = g.sigmoid(x);
    Node* loss = g.reduce_sum(sig);
    g.mark_output(loss);

    Tensor t_x = Tensor::empty({2}, DType::Float64);
    t_x.at<double>({0}) = 0.5;
    t_x.at<double>({1}) = -1.2;

    auto grads = Autodiff::compute_gradients(g, loss, {{x, t_x}});

    double eps = 1e-5;
    for (int64_t i = 0; i < 2; ++i) {
        Tensor x_plus = t_x.clone();
        x_plus.at<double>({i}) += eps;
        Tensor x_minus = t_x.clone();
        x_minus.at<double>({i}) -= eps;

        Tensor y_plus = ops::reduce_sum(ops::sigmoid(x_plus));
        Tensor y_minus = ops::reduce_sum(ops::sigmoid(x_minus));
        double num_grad = (y_plus.item<double>() - y_minus.item<double>()) / (2.0 * eps);

        double analytical_grad = grads[x].at<double>({i});
        EXPECT_NEAR(analytical_grad, num_grad, 1e-4);
    }
}

TEST(AutodiffTest, DivGrad) {
    Graph g;
    Node* a = g.create_input({2}, DType::Float32, "A");
    Node* b = g.create_input({2}, DType::Float32, "B");
    Node* c = g.div(a, b);
    Node* loss = g.reduce_sum(c);
    g.mark_output(loss);

    Tensor t_a = Tensor::empty({2}, DType::Float32);
    t_a.at<float>({0}) = 6.0f; t_a.at<float>({1}) = 8.0f;
    Tensor t_b = Tensor::empty({2}, DType::Float32);
    t_b.at<float>({0}) = 2.0f; t_b.at<float>({1}) = 4.0f;

    auto grads = Autodiff::compute_gradients(g, loss, {{a, t_a}, {b, t_b}});
    // d(A/B)/dA = 1/B -> 1/2 = 0.5, 1/4 = 0.25
    EXPECT_FLOAT_EQ((grads[a].at<float>({0})), 0.5f);
    EXPECT_FLOAT_EQ((grads[a].at<float>({1})), 0.25f);
    // d(A/B)/dB = -A / B^2 -> -6 / 4 = -1.5, -8 / 16 = -0.5
    EXPECT_FLOAT_EQ((grads[b].at<float>({0})), -1.5f);
    EXPECT_FLOAT_EQ((grads[b].at<float>({1})), -0.5f);
}

TEST(AutodiffTest, TransposeGrad) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32, "A");
    Node* t = g.transpose(a, 0, 1);
    Node* loss = g.reduce_sum(t);
    g.mark_output(loss);

    Tensor t_a = Tensor::zeros({2, 3}, DType::Float32);
    auto grads = Autodiff::compute_gradients(g, loss, {{a, t_a}});
    EXPECT_EQ(grads[a].shape(), Shape({2, 3}));
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((grads[a].at<float>({i, j})), 1.0f);
        }
    }
}

TEST(AutodiffTest, ReduceMeanGrad) {
    Graph g;
    Node* a = g.create_input({4}, DType::Float32, "A");
    Node* loss = g.reduce_mean(a);
    g.mark_output(loss);

    Tensor t_a = Tensor::zeros({4}, DType::Float32);
    auto grads = Autodiff::compute_gradients(g, loss, {{a, t_a}});
    // d(mean)/dA = 1 / 4 = 0.25
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ((grads[a].at<float>({i})), 0.25f);
    }
}

TEST(AutodiffTest, TwoLayerMLPBackward) {
    // Forward: H = relu(X @ W1 + b1), Y = H @ W2 + b2, Loss = mean(Y)
    Graph g;
    Node* x = g.create_input({2, 4}, DType::Float32, "X");
    Node* w1 = g.create_input({4, 3}, DType::Float32, "W1");
    Node* b1 = g.create_input({3}, DType::Float32, "b1");
    Node* h = g.relu(g.add(g.matmul(x, w1), b1));

    Node* w2 = g.create_input({3, 1}, DType::Float32, "W2");
    Node* b2 = g.create_input({1}, DType::Float32, "b2");
    Node* y = g.add(g.matmul(h, w2), b2);
    Node* loss = g.reduce_mean(y);
    g.mark_output(loss);

    Tensor t_x = Tensor::full({2, 4}, 0.5f, DType::Float32);
    Tensor t_w1 = Tensor::full({4, 3}, 0.2f, DType::Float32);
    Tensor t_b1 = Tensor::full({3}, 0.1f, DType::Float32);
    Tensor t_w2 = Tensor::full({3, 1}, 0.3f, DType::Float32);
    Tensor t_b2 = Tensor::full({1}, 0.05f, DType::Float32);

    auto grads = Autodiff::compute_gradients(g, loss, {
        {x, t_x}, {w1, t_w1}, {b1, t_b1}, {w2, t_w2}, {b2, t_b2}
    });

    EXPECT_EQ(grads[w1].shape(), Shape({4, 3}));
    EXPECT_EQ(grads[b1].shape(), Shape({3}));
    EXPECT_EQ(grads[w2].shape(), Shape({3, 1}));
    EXPECT_EQ(grads[b2].shape(), Shape({1}));
    EXPECT_GT((grads[w1].at<float>({0, 0})), 0.0f);
}

TEST(AutodiffTest, ReusedOperandSquare) {
    // y = x * x => dy/dx = 2 * x
    Graph g;
    Node* x = g.create_input({3}, DType::Float32, "X");
    Node* y = g.mul(x, x);
    Node* loss = g.reduce_sum(y);
    g.mark_output(loss);

    Tensor t_x = Tensor::empty({3}, DType::Float32);
    t_x.at<float>({0}) = 2.0f;
    t_x.at<float>({1}) = -3.0f;
    t_x.at<float>({2}) = 5.0f;

    auto grads = Autodiff::compute_gradients(g, loss, {{x, t_x}});
    EXPECT_FLOAT_EQ((grads[x].at<float>({0})), 4.0f);
    EXPECT_FLOAT_EQ((grads[x].at<float>({1})), -6.0f);
    EXPECT_FLOAT_EQ((grads[x].at<float>({2})), 10.0f);
}

TEST(AutodiffTest, BranchingReconvergence) {
    // y = (x + x) * (x - 1.0)
    Graph g;
    Node* x = g.create_input({2}, DType::Float32, "X");
    Node* one = g.create_constant(Tensor::full({2}, 1.0f, DType::Float32));
    Node* branch1 = g.add(x, x);      // 2x
    Node* branch2 = g.sub(x, one);    // x - 1
    Node* prod = g.mul(branch1, branch2); // 2x(x - 1) = 2x^2 - 2x
    Node* loss = g.reduce_sum(prod);
    // d/dx [ 2x^2 - 2x ] = 4x - 2
    g.mark_output(loss);

    Tensor t_x = Tensor::empty({2}, DType::Float32);
    t_x.at<float>({0}) = 3.0f; // 4*3 - 2 = 10
    t_x.at<float>({1}) = -1.0f; // 4*(-1) - 2 = -6

    auto grads = Autodiff::compute_gradients(g, loss, {{x, t_x}});
    EXPECT_FLOAT_EQ((grads[x].at<float>({0})), 10.0f);
    EXPECT_FLOAT_EQ((grads[x].at<float>({1})), -6.0f);
}

TEST(AutodiffTest, HighRankUnbroadcasting) {
    // A: [2, 1, 4], B: [1, 3, 4] -> C: [2, 3, 4]
    Graph g;
    Node* a = g.create_input({2, 1, 4}, DType::Float32, "A");
    Node* b = g.create_input({1, 3, 4}, DType::Float32, "B");
    Node* c = g.add(a, b);
    Node* loss = g.reduce_sum(c);
    g.mark_output(loss);

    Tensor t_a = Tensor::full({2, 1, 4}, 1.0f, DType::Float32);
    Tensor t_b = Tensor::full({1, 3, 4}, 2.0f, DType::Float32);

    auto grads = Autodiff::compute_gradients(g, loss, {{a, t_a}, {b, t_b}});
    EXPECT_EQ(grads[a].shape(), Shape({2, 1, 4}));
    EXPECT_EQ(grads[b].shape(), Shape({1, 3, 4}));
    // dL/da: summed across axis 1 (size 3) => 3.0f
    EXPECT_FLOAT_EQ((grads[a].at<float>({0, 0, 0})), 3.0f);
    // dL/db: summed across axis 0 (size 2) => 2.0f
    EXPECT_FLOAT_EQ((grads[b].at<float>({0, 0, 0})), 2.0f);
}

TEST(AutodiffTest, FiniteDifferenceNumericalCheck) {
    // Loss = sum(sigmoid(X @ W))
    Graph g;
    Node* x = g.create_input({2, 3}, DType::Float32, "X");
    Node* w = g.create_input({3, 2}, DType::Float32, "W");
    Node* xw = g.matmul(x, w);
    Node* s = g.sigmoid(xw);
    Node* loss = g.reduce_sum(s);
    g.mark_output(loss);

    Tensor t_x = Tensor::full({2, 3}, 0.5f, DType::Float32);
    Tensor t_w = Tensor::full({3, 2}, 0.25f, DType::Float32);

    auto grads = Autodiff::compute_gradients(g, loss, {{x, t_x}, {w, t_w}});

    // Numerical gradient checking on W[0, 0]
    float eps = 1e-4f;
    Interpreter interp(&g);

    Tensor t_w_pos = t_w.clone();
    t_w_pos.at<float>({0, 0}) += eps;
    float l_pos = interp.run_single({{x, t_x}, {w, t_w_pos}}).item<float>();

    Tensor t_w_neg = t_w.clone();
    t_w_neg.at<float>({0, 0}) -= eps;
    float l_neg = interp.run_single({{x, t_x}, {w, t_w_neg}}).item<float>();

    float num_grad = (l_pos - l_neg) / (2.0f * eps);
    float ana_grad = grads[w].at<float>({0, 0});

    EXPECT_NEAR(ana_grad, num_grad, 1e-3f);
}
