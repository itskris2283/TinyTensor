#pragma once

#include <memory>
#include <string>
#include "tinytensor/graph/graph.hpp"

namespace tinytensor {

struct OptimizationReport {
    size_t original_node_count{0};
    size_t optimized_node_count{0};
    size_t constant_folding_count{0};
    size_t dce_eliminated_count{0};
    size_t cse_eliminated_count{0};
    size_t fused_groups_count{0};
    size_t total_intermediate_allocations{0};
    size_t peak_planned_memory_bytes{0};
    size_t reused_buffers_count{0};
    size_t memory_savings_bytes{0};

    std::string to_string() const;
};

class Optimizer {
public:
    Optimizer() = default;

    // Run all optimization passes in order:
    // 1. Constant folding
    // 2. CSE
    // 3. Fusion
    // 4. Dead Code Elimination
    std::unique_ptr<Graph> optimize(const Graph& graph, OptimizationReport* report = nullptr);
};

} // namespace tinytensor
