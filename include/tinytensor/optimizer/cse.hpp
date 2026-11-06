#pragma once

#include <memory>
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/optimizer/optimizer.hpp"

namespace tinytensor {

class CommonSubexpressionEliminationPass {
public:
    static std::unique_ptr<Graph> run(const Graph& graph, OptimizationReport* report = nullptr);
};

} // namespace tinytensor
