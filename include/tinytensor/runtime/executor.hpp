#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "tinytensor/codegen/codegen.hpp"
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/optimizer/optimizer.hpp"
#include "tinytensor/runtime/executable.hpp"
#include "tinytensor/runtime/interpreter.hpp"
#include "tinytensor/runtime/profiler.hpp"

namespace tinytensor {

enum class ExecutionMode {
    Interpreted,
    CompiledScalar,
    CompiledAVX2,
    CompiledAVX2_FMA,
    CompiledAVX2_NoFMA
};

class Executor {
public:
    Executor(const Graph& graph, ExecutionMode mode = ExecutionMode::CompiledAVX2,
             bool optimize_graph = true, KernelCache* cache = nullptr);

    std::vector<Tensor> run(const std::unordered_map<const Node*, Tensor>& inputs,
                            Profiler* profiler = nullptr);

    Tensor run_single(const std::unordered_map<const Node*, Tensor>& inputs,
                      Profiler* profiler = nullptr);

    const Graph& graph() const noexcept { return *graph_; }
    const OptimizationReport& optimization_report() const noexcept { return opt_report_; }
    ExecutionMode mode() const noexcept { return mode_; }

    const std::string& original_ir() const noexcept { return original_ir_; }
    const std::string& optimized_ir() const noexcept { return optimized_ir_; }
    std::string generated_source() const noexcept;
    std::string execution_plan() const;
    const Executable* executable() const noexcept { return executable_.get(); }

private:
    std::unique_ptr<Graph> owned_graph_;
    const Graph* graph_;
    ExecutionMode mode_;
    OptimizationReport opt_report_;
    std::unique_ptr<Interpreter> interpreter_;
    std::unique_ptr<Executable> executable_;
    std::vector<const Node*> original_inputs_;
    std::vector<const Node*> input_nodes_;
    std::vector<const Node*> output_nodes_;
    std::string original_ir_;
    std::string optimized_ir_;
};

} // namespace tinytensor
