#include "tinytensor/optimizer/optimizer.hpp"
#include <sstream>
#include "tinytensor/ir/verifier.hpp"
#include "tinytensor/optimizer/constant_folding.hpp"
#include "tinytensor/optimizer/cse.hpp"
#include "tinytensor/optimizer/dead_code_elimination.hpp"
#include "tinytensor/optimizer/fusion.hpp"
#include "tinytensor/optimizer/memory_planner.hpp"

namespace tinytensor {

std::string OptimizationReport::to_string() const {
    std::ostringstream ss;
    ss << "=== TinyTensor Optimization Report ===\n"
       << "  Original Nodes:       " << original_node_count << "\n"
       << "  Optimized Nodes:      " << optimized_node_count << "\n"
       << "  Constant Foldings:    " << constant_folding_count << "\n"
       << "  DCE Eliminated Nodes: " << dce_eliminated_count << "\n"
       << "  CSE Replacements:     " << cse_eliminated_count << "\n"
       << "  Fused Op Groups:      " << fused_groups_count << "\n"
       << "  Intermediate Buffers: " << total_intermediate_allocations << "\n"
       << "  Peak Planned Memory:  " << peak_planned_memory_bytes << " bytes\n"
       << "  Reused Buffers:       " << reused_buffers_count << "\n"
       << "  Memory Savings:       " << memory_savings_bytes << " bytes\n"
       << "======================================";
    return ss.str();
}

std::unique_ptr<Graph> Optimizer::optimize(const Graph& graph, OptimizationReport* report) {
    if (report) {
        report->original_node_count = graph.num_nodes();
    }

    // Pre-optimization verification
    IRVerifier::verify(graph);

    // 1. Constant folding
    auto g1 = ConstantFoldingPass::run(graph, report);
    IRVerifier::verify(*g1);

    // 2. Common Subexpression Elimination
    auto g2 = CommonSubexpressionEliminationPass::run(*g1, report);
    IRVerifier::verify(*g2);

    // 3. Elementwise Fusion
    auto g3 = ElementwiseFusionPass::run(*g2, report);
    IRVerifier::verify(*g3);

    // 4. Dead Code Elimination
    auto g_opt = DeadCodeEliminationPass::run(*g3, report);
    IRVerifier::verify(*g_opt);

    // 5. Memory Planning
    MemoryPlanner::plan(*g_opt, report);

    if (report) {
        report->optimized_node_count = g_opt->num_nodes();
    }

    return g_opt;
}

} // namespace tinytensor
