#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include "tinytensor/codegen/cpp_backend.hpp"
#include "tinytensor/graph/graph.hpp"

using namespace tinytensor;

TEST(CodegenTest, GenerateSourceString) {
    Graph g;
    Node* a = g.create_input({4, 4}, DType::Float32, "A");
    Node* b = g.create_input({4, 4}, DType::Float32, "B");
    Node* c = g.matmul(a, b);
    g.mark_output(c);

    CppBackend backend;
    std::string code = backend.generate_code(g);
    EXPECT_NE(code.find("tinytensor_kernel_entry"), std::string::npos);
    EXPECT_NE(code.find("node_"), std::string::npos);
}

TEST(CodegenTest, CompileAndExecuteDirectly) {
    Graph g;
    Node* x = g.create_input({8}, DType::Float32, "X");
    Node* y = g.relu(x);
    g.mark_output(y);

    CppBackend backend;
    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    Tensor in = Tensor::empty({8}, DType::Float32);
    for (int64_t i = 0; i < 8; ++i) {
        in.at<float>({i}) = (i % 2 == 0) ? -5.0f : 5.0f;
    }

    std::vector<Tensor> outputs = exec->run({in});
    ASSERT_EQ(outputs.size(), 1u);
    for (int64_t i = 0; i < 8; ++i) {
        float expected = (i % 2 == 0) ? 0.0f : 5.0f;
        EXPECT_FLOAT_EQ((outputs[0].at<float>({i})), expected);
    }
}

TEST(CodegenTest, ScalarFallbackAgreementWithAVX2) {
    Graph g;
    Node* a = g.create_input({37}, DType::Float32, "A"); // Prime length to test unaligned & vector tail
    Node* b = g.create_input({37}, DType::Float32, "B");
    Node* sum = g.add(a, b);
    Node* out = g.relu(sum);
    g.mark_output(out);

    CompileOptions opt_scalar;
    opt_scalar.target_isa = TargetISA::Scalar;
    CppBackend backend_scalar(opt_scalar);
    auto exec_scalar = backend_scalar.compile(g);

    CompileOptions opt_avx2;
    opt_avx2.target_isa = TargetISA::AVX2;
    CppBackend backend_avx2(opt_avx2);
    auto exec_avx2 = backend_avx2.compile(g);

    Tensor t_a = Tensor::empty({37}, DType::Float32);
    Tensor t_b = Tensor::empty({37}, DType::Float32);
    for (int64_t i = 0; i < 37; ++i) {
        t_a.at<float>({i}) = static_cast<float>(i) - 15.0f;
        t_b.at<float>({i}) = 5.0f;
    }

    auto out_scalar = exec_scalar->run({t_a, t_b});
    auto out_avx2 = exec_avx2->run({t_a, t_b});

    ASSERT_EQ(out_scalar.size(), 1u);
    ASSERT_EQ(out_avx2.size(), 1u);
    for (int64_t i = 0; i < 37; ++i) {
        EXPECT_FLOAT_EQ((out_scalar[0].at<float>({i})), (out_avx2[0].at<float>({i})));
    }
}

TEST(CodegenTest, MatMulKernelCompilation) {
    Graph g;
    Node* a = g.create_input({4, 8}, DType::Float32, "A");
    Node* b = g.create_input({8, 6}, DType::Float32, "B");
    Node* c = g.matmul(a, b);
    g.mark_output(c);

    CppBackend backend;
    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    Tensor t_a = Tensor::ones({4, 8}, DType::Float32);
    Tensor t_b = Tensor::full({8, 6}, 2.0f, DType::Float32);
    auto outputs = exec->run({t_a, t_b});

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].shape(), Shape({4, 6}));
    // 8 * (1.0 * 2.0) = 16.0
    EXPECT_FLOAT_EQ((outputs[0].at<float>({0, 0})), 16.0f);
    EXPECT_FLOAT_EQ((outputs[0].at<float>({3, 5})), 16.0f);
}

TEST(CodegenTest, SigmoidCompilation) {
    Graph g;
    Node* a = g.create_input({4}, DType::Float32, "A");
    Node* s = g.sigmoid(a);
    g.mark_output(s);

    CppBackend backend;
    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    Tensor t_a = Tensor::zeros({4}, DType::Float32);
    auto outputs = exec->run({t_a});
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR((outputs[0].at<float>({0})), 0.5f, 1e-6);
}

TEST(CodegenTest, DiagnosticSourceInspection) {
    Graph g;
    Node* a = g.create_input({8}, DType::Float32, "A");
    Node* b = g.create_input({8}, DType::Float32, "B");
    Node* c = g.add(a, b);
    g.mark_output(c);

    CppBackend backend;
    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    std::string source = exec->source_code();
    EXPECT_FALSE(source.empty());
    EXPECT_NE(source.find("tinytensor_kernel_entry"), std::string::npos);
    EXPECT_NE(source.find("node_"), std::string::npos);
}

TEST(CodegenTest, BatchedMatMulCompiled) {
    Graph g;
    // Batch=2, M=3, K=4, N=2
    Node* a = g.create_input({2, 3, 4}, DType::Float32, "A");
    Node* b = g.create_input({2, 4, 2}, DType::Float32, "B");
    Node* c = g.matmul(a, b);
    g.mark_output(c);

    CppBackend backend;
    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    Tensor t_a = Tensor::ones({2, 3, 4}, DType::Float32);
    Tensor t_b = Tensor::full({2, 4, 2}, 3.0f, DType::Float32);
    auto outputs = exec->run({t_a, t_b});

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].shape(), Shape({2, 3, 2}));
    // 4 * (1.0 * 3.0) = 12.0
    EXPECT_FLOAT_EQ((outputs[0].at<float>({0, 0, 0})), 12.0f);
    EXPECT_FLOAT_EQ((outputs[0].at<float>({1, 2, 1})), 12.0f);
}

TEST(CodegenTest, ExecutableErrorPaths) {
    Graph g;
    Node* a = g.create_input({4}, DType::Float32, "A");
    Node* r = g.relu(a);
    g.mark_output(r);

    CppBackend backend;
    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    // Wrong input count
    EXPECT_THROW(exec->run({}), RuntimeExecutionError);

    // Wrong shape
    Tensor wrong_shape = Tensor::ones({8}, DType::Float32);
    EXPECT_THROW(exec->run({wrong_shape}), ShapeError);

    // Wrong dtype
    Tensor wrong_dtype = Tensor::ones({4}, DType::Float64);
    EXPECT_THROW(exec->run({wrong_dtype}), TypeError);
}

TEST(CodegenTest, ComprehensiveOpsCompilation) {
    Graph g;
    Node* a = g.create_input({2, 4}, DType::Float32, "A");
    Node* b = g.create_input({2, 4}, DType::Float32, "B");
    Node* c = g.create_input({4}, DType::Float32, "C"); // Broadcasted

    Node* diff = g.sub(a, b);
    Node* bcast_add = g.add(diff, c);
    Node* quot = g.div(bcast_add, c);
    Node* tr = g.transpose(quot, 0, 1); // [4, 2]
    Node* rsum = g.reduce_sum(tr, {0}, false); // [2]
    Node* rmean = g.reduce_mean(tr, {0, 1}, false); // []

    g.mark_output(tr);
    g.mark_output(rsum);
    g.mark_output(rmean);

    CppBackend backend;
    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    Tensor t_a = Tensor::full({2, 4}, 10.0f, DType::Float32);
    Tensor t_b = Tensor::full({2, 4}, 2.0f, DType::Float32);
    Tensor t_c = Tensor::full({4}, 2.0f, DType::Float32);

    auto outputs = exec->run({t_a, t_b, t_c});
    ASSERT_EQ(outputs.size(), 3u);
    EXPECT_EQ(outputs[0].shape(), Shape({4, 2}));
    EXPECT_EQ(outputs[1].shape(), Shape({2}));
    EXPECT_EQ(outputs[2].shape(), Shape({}));

    // (10 - 2 + 2) / 2 = 5.0f
    EXPECT_FLOAT_EQ((outputs[0].at<float>({0, 0})), 5.0f);
    // Sum over axis 0 (dim 4) => 4 * 5 = 20.0f
    EXPECT_FLOAT_EQ((outputs[1].at<float>({0})), 20.0f);
    // Mean over all 8 elements => 5.0f
    EXPECT_FLOAT_EQ((outputs[2].item<float>()), 5.0f);
}

TEST(CodegenTest, DisassemblyAVX2Verification) {
    Graph g;
    Node* a = g.create_input({64}, DType::Float32, "A");
    Node* b = g.create_input({64}, DType::Float32, "B");
    Node* sum = g.add(a, b);
    Node* out = g.relu(sum);
    g.mark_output(out);

    CompileOptions opts;
    opts.target_isa = TargetISA::AVX2;
    CppBackend backend(opts);
    auto exec = backend.compile(g);
    ASSERT_NE(exec, nullptr);

    std::string so_path = exec->so_path();
    ASSERT_FALSE(so_path.empty());

    // Disassemble using objdump -d to inspect instructions
    std::string cmd = "objdump -d " + so_path + " > " + so_path + ".asm 2>&1";
    int ret = std::system(cmd.c_str());
    EXPECT_EQ(ret, 0);

    std::ifstream asm_file(so_path + ".asm");
    ASSERT_TRUE(asm_file.is_open());
    std::stringstream buffer;
    buffer << asm_file.rdbuf();
    std::string asm_text = buffer.str();

    // Verify AVX / AVX2 instructions are generated
    bool has_avx_instruction = (asm_text.find("vaddps") != std::string::npos ||
                                asm_text.find("vmaxps") != std::string::npos ||
                                asm_text.find("vmovups") != std::string::npos ||
                                asm_text.find("ymm") != std::string::npos);
    EXPECT_TRUE(has_avx_instruction);
}
