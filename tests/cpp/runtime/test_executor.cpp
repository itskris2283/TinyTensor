#include <gtest/gtest.h>
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/runtime/executor.hpp"

using namespace tinytensor;

TEST(ExecutorTest, InterpretedMode) {
    Graph g;
    Node* a = g.create_input({2}, DType::Float32, "A");
    Node* b = g.create_input({2}, DType::Float32, "B");
    Node* c = g.add(a, b);
    g.mark_output(c);

    Tensor t_a = Tensor::full({2}, 10.0f, DType::Float32);
    Tensor t_b = Tensor::full({2}, 20.0f, DType::Float32);

    Executor exec(g, ExecutionMode::Interpreted, false);
    Tensor res = exec.run_single({{a, t_a}, {b, t_b}});
    EXPECT_FLOAT_EQ((res.at<float>({0})), 30.0f);
}

TEST(ExecutorTest, CompiledScalarMode) {
    Graph g;
    Node* a = g.create_input({4}, DType::Float32, "A");
    Node* b = g.create_input({4}, DType::Float32, "B");
    Node* c = g.mul(a, b);
    g.mark_output(c);

    Tensor t_a = Tensor::full({4}, 3.0f, DType::Float32);
    Tensor t_b = Tensor::full({4}, 7.0f, DType::Float32);

    Executor exec(g, ExecutionMode::CompiledScalar, false);
    Tensor res = exec.run_single({{a, t_a}, {b, t_b}});
    EXPECT_FLOAT_EQ((res.at<float>({0})), 21.0f);
    EXPECT_FLOAT_EQ((res.at<float>({3})), 21.0f);
}

TEST(ExecutorTest, CompiledAVX2Mode) {
    Graph g;
    Node* a = g.create_input({16}, DType::Float32, "A");
    Node* b = g.create_input({16}, DType::Float32, "B");
    Node* c = g.add(a, b);
    g.mark_output(c);

    Tensor t_a = Tensor::full({16}, 4.0f, DType::Float32);
    Tensor t_b = Tensor::full({16}, 6.0f, DType::Float32);

    Executor exec(g, ExecutionMode::CompiledAVX2, false);
    Tensor res = exec.run_single({{a, t_a}, {b, t_b}});
    for (int64_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ((res.at<float>({i})), 10.0f);
    }
}

TEST(ExecutorTest, DiagnosticApis) {
    Graph g;
    Node* a = g.create_input({4, 4}, DType::Float32, "A");
    Node* b = g.create_input({4, 4}, DType::Float32, "B");
    Node* c = g.add(a, b);
    Node* d = g.relu(c);
    g.mark_output(d);

    Executor exec(g, ExecutionMode::CompiledAVX2, true);

    // Verify diagnostic APIs return non-empty informative strings
    std::string orig_ir = exec.original_ir();
    EXPECT_FALSE(orig_ir.empty());
    EXPECT_NE(orig_ir.find("add"), std::string::npos);

    std::string opt_ir = exec.optimized_ir();
    EXPECT_FALSE(opt_ir.empty());

    std::string source = exec.generated_source();
    EXPECT_FALSE(source.empty());
    EXPECT_NE(source.find("tinytensor_kernel_entry"), std::string::npos);

    std::string plan = exec.execution_plan();
    EXPECT_FALSE(plan.empty());
    EXPECT_NE(plan.find("Execution Mode: Compiled AVX2"), std::string::npos);
    EXPECT_NE(plan.find("Memory Planning:"), std::string::npos);
}

TEST(ExecutorTest, MissingInputThrows) {
    Graph g;
    Node* a = g.create_input({2}, DType::Float32, "A");
    Node* b = g.create_input({2}, DType::Float32, "B");
    Node* c = g.add(a, b);
    g.mark_output(c);

    Executor exec(g, ExecutionMode::CompiledScalar, false);
    Tensor t_a = Tensor::ones({2}, DType::Float32);
    // Missing input B
    EXPECT_THROW(exec.run_single({{a, t_a}}), RuntimeExecutionError);
}

TEST(ExecutorTest, CompiledAVX2FMAAndNoFMAModes) {
    Graph g;
    Node* a = g.create_input({8}, DType::Float32, "A");
    Node* b = g.create_input({8}, DType::Float32, "B");
    Node* c = g.add(a, b);
    g.mark_output(c);

    Tensor ta = Tensor::full({8}, 2.0f);
    Tensor tb = Tensor::full({8}, 3.0f);

    // Test CompiledAVX2_FMA
    Executor exec_fma(g, ExecutionMode::CompiledAVX2_FMA, false);
    Tensor res_fma = exec_fma.run_single({{a, ta}, {b, tb}});
    EXPECT_FLOAT_EQ((res_fma.at<float>({0})), 5.0f);
    EXPECT_NE(exec_fma.execution_plan().find("Compiled AVX2+FMA"), std::string::npos);

    // Test CompiledAVX2_NoFMA
    Executor exec_nofma(g, ExecutionMode::CompiledAVX2_NoFMA, false);
    Tensor res_nofma = exec_nofma.run_single({{a, ta}, {b, tb}});
    EXPECT_FLOAT_EQ((res_nofma.at<float>({0})), 5.0f);
    EXPECT_NE(exec_nofma.execution_plan().find("Compiled AVX2 (No FMA)"), std::string::npos);

    // Test Interpreted execution plan
    Executor exec_interp(g, ExecutionMode::Interpreted, false);
    EXPECT_NE(exec_interp.execution_plan().find("Interpreted"), std::string::npos);
}

TEST(ExecutorTest, ExecutableDefensiveValidation) {
    // Test null kernel execution error
    Executable empty_exec(nullptr, nullptr, {}, {}, {}, {}, {});
    EXPECT_THROW(empty_exec.run({}), RuntimeExecutionError);

    // Compile a valid executable and test input validation branches
    Graph g;
    Node* a = g.create_input({4}, DType::Float32, "A");
    Node* b = g.relu(a);
    g.mark_output(b);

    CppBackend backend;
    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    // 1. Wrong number of inputs
    EXPECT_THROW(exec->run({}), RuntimeExecutionError);

    // 2. Shape mismatch
    Tensor wrong_shape = Tensor::empty({8}, DType::Float32);
    EXPECT_THROW(exec->run({wrong_shape}), ShapeError);

    // 3. DType mismatch
    Tensor wrong_dtype = Tensor::empty({4}, DType::Float64);
    EXPECT_THROW(exec->run({wrong_dtype}), TypeError);

    // 4. Run with profiler
    Profiler prof;
    Tensor valid_in = Tensor::full({4}, 1.0f, DType::Float32);
    auto outs = exec->run({valid_in}, &prof);
    ASSERT_EQ(outs.size(), 1u);
    EXPECT_FLOAT_EQ((outs[0].at<float>({0})), 1.0f);
    EXPECT_GE(prof.total_time_us(), 0.0);
}
