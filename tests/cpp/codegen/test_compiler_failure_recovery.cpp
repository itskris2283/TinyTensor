#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>
#include "tinytensor/codegen/cpp_backend.hpp"
#include "tinytensor/codegen/kernel_cache.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/ir/printer.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

namespace {

Graph make_simple_graph() {
    Graph g;
    Node* in = g.create_input({4}, DType::Float32, "X");
    Node* out = g.relu(in);
    g.mark_output(out);
    return g;
}

} // anonymous namespace

TEST(CompilerFailureRecoveryTest, MissingCompilerBinaryThrowsCompileError) {
    CompileOptions opts;
    opts.compiler_path = "/nonexistent/usr/bin/gcc_missing_binary";
    CppBackend backend(opts);

    Graph g = make_simple_graph();
    EXPECT_THROW({
        try {
            backend.compile(g);
        } catch (const CompileError& e) {
            EXPECT_NE(std::string(e.what()).find("not found"), std::string::npos);
            throw;
        }
    }, CompileError);
}

TEST(CompilerFailureRecoveryTest, CompilerPathIsDirectoryThrowsCompileError) {
    CompileOptions opts;
    opts.compiler_path = "/tmp";
    CppBackend backend(opts);

    Graph g = make_simple_graph();
    EXPECT_THROW({
        try {
            backend.compile(g);
        } catch (const CompileError& e) {
            EXPECT_NE(std::string(e.what()).find("directory"), std::string::npos);
            throw;
        }
    }, CompileError);
}

TEST(CompilerFailureRecoveryTest, CompilationAbnormalTerminationReturnsStderr) {
    CompileOptions opts;
    opts.opt_level = "-Werror -Winvalid-flag-never-seen-before";
    CppBackend backend(opts);

    Graph g = make_simple_graph();
    EXPECT_THROW({
        try {
            backend.compile(g);
        } catch (const CompileError& e) {
            std::string msg = e.what();
            EXPECT_NE(msg.find("failed with code"), std::string::npos);
            throw;
        }
    }, CompileError);
}

TEST(CompilerFailureRecoveryTest, CompilationTimeoutEnforced) {
    // Create a mock hung compiler script that sleeps
    std::string mock_script = "/tmp/tinytensor_hung_compiler.sh";
    {
        std::ofstream out(mock_script);
        out << "#!/bin/sh\nsleep 10\n";
    }
    std::filesystem::permissions(mock_script,
                                 std::filesystem::perms::owner_all |
                                 std::filesystem::perms::group_all |
                                 std::filesystem::perms::others_all);

    CompileOptions opts;
    opts.compiler_path = mock_script;
    opts.timeout_seconds = 1;
    CppBackend backend(opts);

    Graph g = make_simple_graph();
    auto start = std::chrono::steady_clock::now();

    EXPECT_THROW({
        try {
            backend.compile(g);
        } catch (const CompileError& e) {
            std::string msg = e.what();
            EXPECT_NE(msg.find("timed out"), std::string::npos);
            throw;
        }
    }, CompileError);

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start).count();
    // Must terminate within ~2 seconds, not hung for 10 seconds
    EXPECT_LE(elapsed, 2);

    std::filesystem::remove(mock_script);
}

TEST(CompilerFailureRecoveryTest, CorruptedCacheDLOpenFailureRecovery) {
    std::string test_cache_dir = ".test_corrupt_cache_dlopen";
    KernelCache cache(test_cache_dir);
    cache.clear();

    Graph g = make_simple_graph();
    std::string canonical_ir = IRPrinter::to_string(g);
    CompileOptions opts;
    opts.target_isa = TargetISA::Scalar;
    std::string compiler_info = opts.compiler_path + " " + opts.opt_level + " scalar gcc-13.3.0";
    std::string key = KernelCache::compute_key(canonical_ir, "tinytensor_v0.1.1", compiler_info);

    // Create a fake .so with valid 4-byte ELF magic header (0x7f, 'E', 'L', 'F')
    // followed by garbage that causes dlopen() to fail
    std::filesystem::path bad_so = std::filesystem::path(test_cache_dir) / (key + ".so");
    {
        std::ofstream out(bad_so, std::ios::binary);
        const char elf_header[64] = {
            '\x7f', 'E', 'L', 'F', 2, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            3, 0, 0x3e, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        };
        out.write(elf_header, sizeof(elf_header));
    }

    // Backend must detect dlopen failure, remove the corrupted cache file, recompile and succeed
    CppBackend backend(opts);
    auto exec = backend.compile(g, &cache);
    ASSERT_NE(exec, nullptr);

    Tensor in = Tensor::full({4}, 2.0f, DType::Float32);
    auto outs = exec->run({in});
    EXPECT_FLOAT_EQ(outs[0].data<float>()[0], 2.0f);

    cache.clear();
    std::filesystem::remove_all(test_cache_dir);
}

TEST(CompilerFailureRecoveryTest, MissingEntrySymbolDLOpenRecovery) {
    std::string test_cache_dir = ".test_missing_symbol_cache";
    KernelCache cache(test_cache_dir);
    cache.clear();

    Graph g = make_simple_graph();
    std::string canonical_ir = IRPrinter::to_string(g);
    CompileOptions opts;
    opts.target_isa = TargetISA::Scalar;
    std::string compiler_info = opts.compiler_path + " " + opts.opt_level + " scalar gcc-13.3.0";
    std::string key = KernelCache::compute_key(canonical_ir, "tinytensor_v0.1.1", compiler_info);

    // Compile a minimal shared library that lacks tinytensor_kernel_entry
    std::string dummy_cpp = test_cache_dir + "/dummy.cpp";
    std::string dummy_so = test_cache_dir + "/" + key + ".so";
    {
        std::ofstream out(dummy_cpp);
        out << "extern \"C\" void irrelevant_function() {}\n";
    }
    std::string cmd = "g++ -shared -fPIC " + dummy_cpp + " -o " + dummy_so;
    int res = system(cmd.c_str());
    ASSERT_EQ(res, 0);

    // Backend must detect missing symbol, remove bad cache entry, recompile and succeed
    CppBackend backend(opts);
    auto exec = backend.compile(g, &cache);
    ASSERT_NE(exec, nullptr);

    Tensor in = Tensor::full({4}, 3.0f, DType::Float32);
    auto outs = exec->run({in});
    EXPECT_FLOAT_EQ(outs[0].data<float>()[0], 3.0f);

    cache.clear();
    std::filesystem::remove_all(test_cache_dir);
}

TEST(CompilerFailureRecoveryTest, ConcurrentCacheWrites) {
    std::string test_cache_dir = ".test_concurrent_cache";
    KernelCache cache(test_cache_dir);
    cache.clear();

    Graph g = make_simple_graph();

    // 4 threads compiling the same graph with the same cache concurrently
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<Executable>> execs(4);
    std::vector<std::string> errors(4);

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&, i]() {
            try {
                CppBackend backend;
                execs[static_cast<size_t>(i)] = backend.compile(g, &cache);
            } catch (const std::exception& e) {
                errors[static_cast<size_t>(i)] = e.what();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(errors[static_cast<size_t>(i)].empty()) << "Thread " << i << " failed: " << errors[static_cast<size_t>(i)];
        ASSERT_NE(execs[static_cast<size_t>(i)], nullptr);
        Tensor in = Tensor::full({4}, 5.0f, DType::Float32);
        auto outs = execs[static_cast<size_t>(i)]->run({in});
        EXPECT_FLOAT_EQ(outs[0].data<float>()[0], 5.0f);
    }

    cache.clear();
    std::filesystem::remove_all(test_cache_dir);
}
