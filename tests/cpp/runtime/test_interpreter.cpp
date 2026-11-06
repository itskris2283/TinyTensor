#include <gtest/gtest.h>
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/optimizer/fusion.hpp"
#include "tinytensor/runtime/interpreter.hpp"

using namespace tinytensor;

TEST(InterpreterTest, BasicArithmetic) {
    Graph g;
    Node* a = g.create_input({2, 2}, DType::Float32, "A");
    Node* b = g.create_input({2, 2}, DType::Float32, "B");
    Node* sum = g.add(a, b);
    Node* prod = g.mul(sum, a);
    g.mark_output(prod);

    Tensor t_a = Tensor::full({2, 2}, 3.0f, DType::Float32);
    Tensor t_b = Tensor::full({2, 2}, 2.0f, DType::Float32);

    Interpreter interp(&g);
    Tensor res = interp.run_single({{a, t_a}, {b, t_b}});

    // (3 + 2) * 3 = 15
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ((res.at<float>({i, j})), 15.0f);
        }
    }
}

TEST(InterpreterTest, MatMulAndRelu) {
    Graph g;
    Node* a = g.create_input({2, 2}, DType::Float32, "A");
    Node* b = g.create_input({2, 2}, DType::Float32, "B");
    Node* c = g.matmul(a, b);
    Node* r = g.relu(c);
    g.mark_output(r);

    Tensor t_a = Tensor::full({2, 2}, -1.0f, DType::Float32);
    Tensor t_b = Tensor::full({2, 2}, 2.0f, DType::Float32);

    Interpreter interp(&g);
    Tensor res = interp.run_single({{a, t_a}, {b, t_b}});
    // (-1*2 + -1*2) = -4 -> relu is 0
    EXPECT_FLOAT_EQ((res.at<float>({0, 0})), 0.0f);
}

TEST(InterpreterTest, ProfilerRecording) {
    Graph g;
    Node* a = g.create_input({100}, DType::Float32, "A");
    Node* r = g.relu(a);
    g.mark_output(r);

    Tensor t_a = Tensor::full({100}, 5.0f, DType::Float32);
    Interpreter interp(&g);
    Profiler prof;
    interp.run({{a, t_a}}, &prof);

    EXPECT_GT(prof.total_time_us(), 0.0);
    EXPECT_EQ(prof.node_profiles().size(), 2u); // input and relu
}

TEST(InterpreterTest, SubDivSigmoidTransposeReductions) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32, "A");
    Node* b = g.create_input({2, 3}, DType::Float32, "B");
    Node* c = g.create_constant(Tensor::full({2, 3}, 1.0f, DType::Float32), "C");

    Node* diff = g.sub(a, b);
    Node* ratio = g.div(diff, c);
    Node* sig = g.sigmoid(ratio);
    Node* tr = g.transpose(sig, 0, 1); // shape [3, 2]
    Node* rsum = g.reduce_sum(tr, {0, 1});
    Node* rmean = g.reduce_mean(tr, {0, 1});

    g.mark_output(rsum);
    g.mark_output(rmean);

    Tensor t_a = Tensor::full({2, 3}, 5.0f, DType::Float32);
    Tensor t_b = Tensor::full({2, 3}, 5.0f, DType::Float32);

    Interpreter interp(&g);
    auto res = interp.run({{a, t_a}, {b, t_b}});

    // diff = 0, ratio = 0, sigmoid(0) = 0.5.
    // 6 elements * 0.5 = 3.0 sum, 0.5 mean.
    EXPECT_NEAR(res[0].item<float>(), 3.0f, 1e-4f);
    EXPECT_NEAR(res[1].item<float>(), 0.5f, 1e-4f);
}

TEST(InterpreterTest, ErrorPaths) {
    // 1. Null graph throws
    EXPECT_THROW(Interpreter(nullptr), RuntimeExecutionError);

    Graph g;
    Node* a = g.create_input({2, 2}, DType::Float32, "A");
    Node* r = g.relu(a);
    g.mark_output(r);

    Interpreter interp(&g);

    // 2. Missing input throws
    EXPECT_THROW(interp.run({}), RuntimeExecutionError);

    // 3. Shape mismatch throws
    Tensor wrong_shape = Tensor::empty({4}, DType::Float32);
    EXPECT_THROW(interp.run({{a, wrong_shape}}), ShapeError);

    // 4. DType mismatch throws
    Tensor wrong_dtype = Tensor::empty({2, 2}, DType::Float64);
    EXPECT_THROW(interp.run({{a, wrong_dtype}}), TypeError);
}

TEST(InterpreterTest, FusedElementwiseExecution) {
    Graph g;
    Node* a = g.create_input({4}, DType::Float32, "A");
    Node* b = g.create_input({4}, DType::Float32, "B");
    Node* add = g.add(a, b);
    Node* relu = g.relu(add);
    g.mark_output(relu);

    ElementwiseFusionPass fusion;
    auto fused_g = fusion.run(g);
    ASSERT_NE(fused_g, nullptr);

    Interpreter interp(fused_g.get());
    Tensor ta = Tensor::full({4}, -2.0f);
    Tensor tb = Tensor::full({4}, 5.0f);

    Tensor res = interp.run_single({{fused_g->inputs()[0], ta}, {fused_g->inputs()[1], tb}});
    EXPECT_FLOAT_EQ((res.at<float>({0})), 3.0f);
}
