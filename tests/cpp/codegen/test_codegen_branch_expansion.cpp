#include <gtest/gtest.h>
#include <cmath>
#include "tinytensor/codegen/cpp_backend.hpp"
#include "tinytensor/core/operators.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/optimizer/optimizer.hpp"
#include "tinytensor/runtime/executor.hpp"

using namespace tinytensor;

TEST(CodegenBranchExpansionTest, Float64ScalarCodegenAndExecution) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float64, "A");
    Node* b = g.create_input({2, 3}, DType::Float64, "B");
    Node* c = g.add(a, b);
    Node* d = g.sub(c, a); // = b
    Node* e = g.mul(d, b); // = b^2
    Node* r = g.relu(e);
    Node* s = g.sigmoid(r);
    Node* loss = g.reduce_mean(s);
    g.mark_output(loss);

    CompileOptions opts;
    opts.target_isa = TargetISA::Scalar;
    opts.opt_level = "-O2";
    CppBackend backend(opts);

    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    Tensor ta = Tensor::full({2, 3}, 2.0, DType::Float64);
    Tensor tb = Tensor::full({2, 3}, 1.0, DType::Float64);

    auto outputs = exec->run({ta, tb});
    ASSERT_EQ(outputs.size(), 1u);

    // Compute expected via eager ops
    Tensor ea = ops::add(ta, tb);
    Tensor ed = ops::sub(ea, ta);
    Tensor ee = ops::mul(ed, tb);
    Tensor er = ops::relu(ee);
    Tensor es = ops::sigmoid(er);
    Tensor eloss = ops::reduce_mean(es);

    EXPECT_NEAR(outputs[0].item<double>(), eloss.item<double>(), 1e-6);
}

TEST(CodegenBranchExpansionTest, Float64MatMulCodegenAndExecution) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float64, "A");
    Node* b = g.create_input({3, 2}, DType::Float64, "B");
    Node* mm = g.matmul(a, b);
    g.mark_output(mm);

    CompileOptions opts;
    opts.target_isa = TargetISA::Scalar;
    CppBackend backend(opts);

    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    Tensor ta = Tensor::full({2, 3}, 2.5, DType::Float64);
    Tensor tb = Tensor::full({3, 2}, 4.0, DType::Float64);

    auto outputs = exec->run({ta, tb});
    ASSERT_EQ(outputs.size(), 1u);

    Tensor expected = ops::matmul(ta, tb);
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(outputs[0].data<double>()[i], expected.data<double>()[i]);
    }
}

TEST(CodegenBranchExpansionTest, MultiDimReductionsWithAndWithoutKeepdims) {
    Graph g;
    Node* a = g.create_input({2, 3, 4}, DType::Float32, "A");
    Node* r_sum = g.reduce_sum(a, {1}, true);  // keepdims=true, shape {2, 1, 4}
    Node* r_mean = g.reduce_mean(a, {0, 2}, false); // keepdims=false, shape {3}
    g.mark_output(r_sum);
    g.mark_output(r_mean);

    CompileOptions opts;
    opts.target_isa = TargetISA::Scalar;
    CppBackend backend(opts);

    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    Tensor ta = Tensor::empty({2, 3, 4}, DType::Float32);
    for (int64_t i = 0; i < 24; ++i) {
        ta.data<float>()[i] = static_cast<float>(i + 1);
    }

    auto outputs = exec->run({ta});
    ASSERT_EQ(outputs.size(), 2u);

    EXPECT_EQ(outputs[0].shape(), Shape({2, 1, 4}));
    EXPECT_EQ(outputs[1].shape(), Shape({3}));

    Tensor exp_sum = ops::reduce_sum(ta, {1}, true);
    Tensor exp_mean = ops::reduce_mean(ta, {0, 2}, false);

    for (int64_t i = 0; i < outputs[0].numel(); ++i) {
        EXPECT_FLOAT_EQ(outputs[0].data<float>()[i], exp_sum.data<float>()[i]);
    }
    for (int64_t i = 0; i < outputs[1].numel(); ++i) {
        EXPECT_FLOAT_EQ(outputs[1].data<float>()[i], exp_mean.data<float>()[i]);
    }
}

TEST(CodegenBranchExpansionTest, MultiDimTransposeCodegenAndExecution) {
    Graph g;
    Node* a = g.create_input({2, 3, 4}, DType::Float32, "A");
    Node* tr = g.transpose(a, 0, 2); // shape {4, 3, 2}
    g.mark_output(tr);

    CompileOptions opts;
    opts.target_isa = TargetISA::Scalar;
    CppBackend backend(opts);

    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    Tensor ta = Tensor::empty({2, 3, 4}, DType::Float32);
    for (int64_t i = 0; i < 24; ++i) {
        ta.data<float>()[i] = static_cast<float>(i);
    }

    auto outputs = exec->run({ta});
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].shape(), Shape({4, 3, 2}));

    Tensor exp_tr = ops::transpose(ta, 0, 2).contiguous();
    for (int64_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(outputs[0].data<float>()[i], exp_tr.data<float>()[i]);
    }
}

TEST(CodegenBranchExpansionTest, FusedElementwiseKernelExecution) {
    Graph g;
    Node* a = g.create_input({4, 8}, DType::Float32, "A");
    Node* b = g.create_input({4, 8}, DType::Float32, "B");

    Node* sum = g.add(a, b);
    Node* act = g.relu(sum);
    g.mark_output(act);

    // Execute via Executor with full optimization enabled to test fused elementwise execution
    Executor exec(g, ExecutionMode::CompiledScalar, true);

    Tensor ta = Tensor::full({4, 8}, 2.0f, DType::Float32);
    Tensor tb = Tensor::full({4, 8}, -5.0f, DType::Float32);

    auto outputs = exec.run({{a, ta}, {b, tb}});
    ASSERT_EQ(outputs.size(), 1u);

    // 2.0 + (-5.0) = -3.0 -> ReLU(-3.0) = 0.0
    for (int64_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(outputs[0].data<float>()[i], 0.0f);
    }

    Tensor tb_pos = Tensor::full({4, 8}, 3.0f, DType::Float32);
    auto outputs_pos = exec.run({{a, ta}, {b, tb_pos}});
    ASSERT_EQ(outputs_pos.size(), 1u);
    // 2.0 + 3.0 = 5.0 -> ReLU(5.0) = 5.0
    for (int64_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(outputs_pos[0].data<float>()[i], 5.0f);
    }
}
