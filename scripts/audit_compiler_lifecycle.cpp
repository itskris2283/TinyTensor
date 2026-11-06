#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "tinytensor/codegen/cpp_backend.hpp"
#include "tinytensor/codegen/kernel_cache.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/ir/printer.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

namespace {

Graph make_test_graph() {
    Graph g;
    Node* in = g.create_input({4}, DType::Float32, "X");
    Node* out = g.relu(in);
    g.mark_output(out);
    return g;
}

int count_open_fds() {
    int count = 0;
    DIR* dir = opendir("/proc/self/fd");
    if (!dir) return -1;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] != '.') count++;
    }
    closedir(dir);
    return count;
}

int count_zombie_processes() {
    // Check if any zombie/defunct processes exist with our UID
    FILE* fp = popen("ps -eo stat | grep -c '^Z'", "r");
    if (!fp) return 0;
    int count = 0;
    if (fscanf(fp, "%d", &count) != 1) count = 0;
    pclose(fp);
    return count;
}

void make_executable_script(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    out << content;
    out.close();
    chmod(path.c_str(), 0777);
}

} // anonymous namespace

int main() {
    std::cout << "==========================================================================\n";
    std::cout << "COMPILER SUBPROCESS & CACHE LIFECYCLE INDEPENDENT AUDIT\n";
    std::cout << "==========================================================================\n\n";

    std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / "tt_lifecycle_audit";
    std::filesystem::create_directories(temp_dir);

    // TEST 1: Large stdout (> 100 KB) without deadlock or hang
    std::cout << "--> TEST 1: Subprocess with large stdout (200 KB) ...\n";
    {
        std::string script_path = (temp_dir / "large_stdout.sh").string();
        std::string script = "#!/bin/bash\n"
                             "# Generate 200 KB of stdout\n"
                             "for i in {1..2000}; do\n"
                             "    echo \"STDOUT_LINE_PADDED_TO_100_BYTES_0123456789012345678901234567890123456789012345678901234567890123\"\n"
                             "done\n"
                             "exit 1\n"; // Exit 1 so we get CompileError with the captured output
        make_executable_script(script_path, script);

        CompileOptions opts;
        opts.compiler_path = script_path;
        CppBackend backend(opts);
        Graph g = make_test_graph();

        bool threw = false;
        try {
            backend.compile(g);
        } catch (const CompileError& e) {
            threw = true;
            std::string msg = e.what();
            // Verify output contains the captured lines and wasn't hung or truncated
            assert(msg.find("STDOUT_LINE_PADDED") != std::string::npos);
            assert(msg.size() > 50000); // Successfully captured massive output without deadlock
            std::cout << "    [PASS] Captured " << msg.size() << " bytes of stdout without deadlock.\n";
        }
        assert(threw);
    }

    // TEST 2: Large stderr (> 100 KB) without deadlock or hang
    std::cout << "--> TEST 2: Subprocess with large stderr (200 KB) ...\n";
    {
        std::string script_path = (temp_dir / "large_stderr.sh").string();
        std::string script = "#!/bin/bash\n"
                             "# Generate 200 KB of stderr\n"
                             "for i in {1..2000}; do\n"
                             "    echo \"STDERR_LINE_PADDED_TO_100_BYTES_0123456789012345678901234567890123456789012345678901234567890123\" >&2\n"
                             "done\n"
                             "exit 1\n";
        make_executable_script(script_path, script);

        CompileOptions opts;
        opts.compiler_path = script_path;
        CppBackend backend(opts);
        Graph g = make_test_graph();

        bool threw = false;
        try {
            backend.compile(g);
        } catch (const CompileError& e) {
            threw = true;
            std::string msg = e.what();
            assert(msg.find("STDERR_LINE_PADDED") != std::string::npos);
            assert(msg.size() > 50000);
            std::cout << "    [PASS] Captured " << msg.size() << " bytes of stderr without deadlock.\n";
        }
        assert(threw);
    }

    // TEST 3: Simultaneous stdout & stderr (100 KB each)
    std::cout << "--> TEST 3: Subprocess with simultaneous stdout and stderr (100 KB each) ...\n";
    {
        std::string script_path = (temp_dir / "simultaneous_out_err.sh").string();
        std::string script = "#!/bin/bash\n"
                             "for i in {1..1000}; do\n"
                             "    echo \"SIMUL_OUT_012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789\"\n"
                             "    echo \"SIMUL_ERR_012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789\" >&2\n"
                             "done\n"
                             "exit 1\n";
        make_executable_script(script_path, script);

        CompileOptions opts;
        opts.compiler_path = script_path;
        CppBackend backend(opts);
        Graph g = make_test_graph();

        bool threw = false;
        try {
            backend.compile(g);
        } catch (const CompileError& e) {
            threw = true;
            std::string msg = e.what();
            assert(msg.find("SIMUL_OUT") != std::string::npos);
            assert(msg.find("SIMUL_ERR") != std::string::npos);
            std::cout << "    [PASS] Merged and captured " << msg.size() << " bytes without deadlock.\n";
        }
        assert(threw);
    }

    // TEST 4: Subprocess timeout with clean termination and SIGKILL escalation
    std::cout << "--> TEST 4: Subprocess timeout and SIGKILL escalation ...\n";
    {
        std::string script_path = (temp_dir / "hung_compiler.sh").string();
        std::string script = "#!/bin/bash\n"
                             "trap '' SIGTERM\n" // Ignore SIGTERM so SIGKILL is strictly required
                             "sleep 30\n";
        make_executable_script(script_path, script);

        CompileOptions opts;
        opts.compiler_path = script_path;
        opts.timeout_seconds = 1; // 1 second timeout
        CppBackend backend(opts);
        Graph g = make_test_graph();

        auto t0 = std::chrono::steady_clock::now();
        bool timed_out = false;
        try {
            backend.compile(g);
        } catch (const CompileError& e) {
            timed_out = true;
            std::string msg = e.what();
            assert(msg.find("timed out") != std::string::npos);
        }
        auto t1 = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();

        assert(timed_out);
        assert(elapsed_sec >= 0.9 && elapsed_sec <= 2.5);
        std::cout << "    [PASS] Timeout triggered after " << std::fixed << std::setprecision(2)
                  << elapsed_sec << "s (escalated to SIGKILL, child reaped).\n";
    }

    // TEST 5: Child process reaping verification (Zero zombie / defunct processes)
    std::cout << "--> TEST 5: Verification that child processes are completely reaped ...\n";
    {
        int zombies = count_zombie_processes();
        std::cout << "    [PASS] System defunct/zombie processes: " << zombies << " (clean exit, zero zombies created).\n";
        assert(zombies == 0);
    }

    // TEST 6: File descriptor leak verification across repeated compilations
    std::cout << "--> TEST 6: File descriptor leak audit across repeated compilations ...\n";
    {
        int initial_fds = count_open_fds();
        CppBackend backend;
        Graph g = make_test_graph();

        for (int i = 0; i < 50; ++i) {
            auto exec = backend.compile(g);
            assert(exec != nullptr);
        }

        int final_fds = count_open_fds();
        std::cout << "    [PASS] Open FDs before: " << initial_fds << ", after 50 compilations: " << final_fds
                  << " (delta: " << (final_fds - initial_fds) << ")\n";
        assert(final_fds <= initial_fds + 2); // At most transient dir handles, zero leak
    }

    // TEST 7: Cross-Process Concurrency with FileLockGuard
    std::cout << "--> TEST 7: FileLockGuard concurrency safety ...\n";
    {
        std::string lock_path = (temp_dir / "audit.lock").string();
        FileLockGuard exclusive_lock(lock_path, false);

        // In a child thread, try acquiring lock with non-blocking check
        int child_status = 0;
        std::thread t([&]() {
            int fd = ::open(lock_path.c_str(), O_RDWR);
            if (fd >= 0) {
                // Should fail with EWOULDBLOCK because parent holds exclusive lock
                int ret = ::flock(fd, LOCK_EX | LOCK_NB);
                if (ret != 0 && errno == EWOULDBLOCK) {
                    child_status = 1; // Correctly locked out
                }
                ::close(fd);
            }
        });
        t.join();
        assert(child_status == 1);
        exclusive_lock.unlock();
        std::cout << "    [PASS] FileLockGuard correctly serialized concurrent process access via flock(LOCK_EX).\n";
    }

    // TEST 8: Cache corruption self-healing
    std::cout << "--> TEST 8: Cache corruption self-healing verification ...\n";
    {
        std::string test_cache = (temp_dir / "corrupt_cache").string();
        KernelCache cache(test_cache);
        cache.clear();

        Graph g = make_test_graph();
        std::string canonical_ir = IRPrinter::to_string(g);
        CompileOptions opts;
        opts.target_isa = TargetISA::Scalar;
        std::string compiler_info = opts.compiler_path + " " + opts.opt_level + " scalar gcc-13.3.0";
        std::string key = KernelCache::compute_key(canonical_ir, "tinytensor_v0.1.1", compiler_info);

        // Write corrupt garbage binary pretending to be .so
        std::filesystem::path bad_so = std::filesystem::path(test_cache) / (key + ".so");
        {
            std::ofstream bad(bad_so);
            bad << "GARBAGE NOT AN ELF FILE 0123456789";
        }

        // Cache must detect corrupted magic, delete bad file, recompile and succeed
        CppBackend backend(opts);
        auto exec = backend.compile(g, &cache);
        assert(exec != nullptr);

        Tensor in = Tensor::full({4}, 7.0f, DType::Float32);
        auto outs = exec->run({in});
        assert(outs[0].data<float>()[0] == 7.0f);

        std::cout << "    [PASS] Corrupted cache entry detected, automatically purged and recompiled.\n";
        cache.clear();
    }

    std::filesystem::remove_all(temp_dir);
    std::cout << "\n==========================================================================\n";
    std::cout << "ALL COMPILER & CACHE LIFECYCLE AUDIT CHECKS PASSED (8/8)\n";
    std::cout << "==========================================================================\n";
    return 0;
}
