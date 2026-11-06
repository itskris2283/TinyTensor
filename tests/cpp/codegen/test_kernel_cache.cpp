#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>
#include "tinytensor/codegen/cpp_backend.hpp"
#include "tinytensor/codegen/kernel_cache.hpp"
#include "tinytensor/graph/graph.hpp"

using namespace tinytensor;

TEST(KernelCacheTest, KeyDeterminism) {
    std::string key1 = KernelCache::compute_key("%0 = input\nreturn %0", "meta", "gcc");
    std::string key2 = KernelCache::compute_key("%0 = input\nreturn %0", "meta", "gcc");
    EXPECT_EQ(key1, key2);
    EXPECT_EQ(key1.size(), 64u); // SHA-256 hex string length
}

TEST(KernelCacheTest, CompilationHitAndMiss) {
    KernelCache cache(".test_tinytensor_cache");
    cache.clear();

    Graph g;
    Node* in = g.create_input({4}, DType::Float32, "X");
    Node* out = g.relu(in);
    g.mark_output(out);

    CppBackend backend;

    // First compilation: miss
    auto exec1 = backend.compile(g, &cache);
    ASSERT_NE(exec1, nullptr);
    EXPECT_EQ(cache.misses(), 1u);
    EXPECT_EQ(cache.hits(), 0u);

    // Second compilation with same graph: hit!
    auto exec2 = backend.compile(g, &cache);
    ASSERT_NE(exec2, nullptr);
    EXPECT_EQ(cache.hits(), 1u);
    EXPECT_EQ(cache.misses(), 1u);

    cache.clear();
}

TEST(KernelCacheTest, CorruptionDetectionAndRecovery) {
    KernelCache cache(".test_tinytensor_cache");
    cache.clear();

    std::string key = "corrupted_test_key";
    std::filesystem::path bad_path = std::filesystem::path(".test_tinytensor_cache") / (key + ".so");

    // Write a corrupt truncated file (less than 64 bytes)
    std::ofstream out(bad_path);
    out << "NOT_AN_ELF";
    out.close();

    std::string out_so;
    bool found = cache.get(key, out_so);
    EXPECT_FALSE(found);
    EXPECT_EQ(cache.misses(), 1u);
    // Verifies corrupted file was safely removed
    EXPECT_FALSE(std::filesystem::exists(bad_path));

    cache.clear();
}

TEST(KernelCacheTest, SourceRetentionAndRetrieval) {
    KernelCache cache(".test_tinytensor_cache");
    cache.clear();

    Graph g;
    Node* in = g.create_input({8}, DType::Float32, "X");
    Node* out = g.relu(in);
    g.mark_output(out);

    CppBackend backend;
    auto exec = backend.compile(g, &cache);
    ASSERT_NE(exec, nullptr);

    std::string source = exec->source_code();
    EXPECT_FALSE(source.empty());

    // Verify second run from cache still retains source code
    auto exec_cached = backend.compile(g, &cache);
    ASSERT_NE(exec_cached, nullptr);
    EXPECT_EQ(exec_cached->source_code(), source);

    cache.clear();
}

TEST(KernelCacheTest, MultiProcessConcurrentCompilationStress) {
    const std::vector<int> process_counts = {2, 4, 8};
    for (int num_procs : process_counts) {
        std::string cache_path = ".test_mp_cache_" + std::to_string(num_procs);
        {
            KernelCache cache(cache_path);
            cache.clear();
        }

        std::vector<pid_t> pids;
        for (int i = 0; i < num_procs; ++i) {
            pid_t pid = fork();
            if (pid == 0) {
                // Child worker process
                KernelCache cache(cache_path);
                Graph g;
                Node* a = g.create_input({4, 4}, DType::Float32, "A");
                Node* b = g.create_input({4, 4}, DType::Float32, "B");
                Node* c = g.add(a, b);
                Node* out = g.relu(c);
                g.mark_output(out);

                CppBackend backend;
                auto exec = backend.compile(g, &cache);
                if (!exec) _exit(1);

                Tensor ta = Tensor::full({4, 4}, 1.5f);
                Tensor tb = Tensor::full({4, 4}, 2.5f);
                auto res = exec->run({ta, tb});
                if (res.empty()) _exit(2);
                float val = res[0].data<float>()[0];
                if (std::abs(val - 4.0f) > 1e-4f) _exit(3);

                _exit(0);
            }
            pids.push_back(pid);
        }

        // Parent collects exit statuses
        int failed_workers = 0;
        for (pid_t pid : pids) {
            int status = 0;
            waitpid(pid, &status, 0);
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                failed_workers++;
            }
        }
        EXPECT_EQ(failed_workers, 0);

        // Verify cache directory has a valid .so
        std::error_code ec;
        bool found_so = false;
        for (const auto& entry : std::filesystem::directory_iterator(cache_path, ec)) {
            if (entry.path().extension() == ".so") {
                found_so = true;
                EXPECT_GE(std::filesystem::file_size(entry.path(), ec), 64u);
            }
        }
        EXPECT_TRUE(found_so);

        std::filesystem::remove_all(cache_path, ec);
    }
}
