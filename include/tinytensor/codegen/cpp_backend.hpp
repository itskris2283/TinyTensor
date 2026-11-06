#pragma once

#include <memory>
#include <string>
#include "tinytensor/codegen/cpu_features.hpp"
#include "tinytensor/codegen/kernel_cache.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/runtime/executable.hpp"

namespace tinytensor {

struct CompileOptions {
    TargetISA target_isa{TargetISA::Scalar};
    bool enable_fma{true};
    std::string compiler_path{"/usr/bin/g++"};
    std::string opt_level{"-O3"};
    bool enable_fast_math{false};
    int timeout_seconds{30};
};

class CppBackend {
public:
    explicit CppBackend(CompileOptions options = CompileOptions());

    // Generates C++ code string for the given graph
    std::string generate_code(const Graph& graph) const;

    // Compiles graph to an executable native dynamic library
    std::unique_ptr<Executable> compile(const Graph& graph, KernelCache* cache = nullptr);

private:
    CompileOptions options_;
};

} // namespace tinytensor
