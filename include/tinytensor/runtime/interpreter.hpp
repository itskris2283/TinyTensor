#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "tinytensor/core/operators.hpp"
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/runtime/profiler.hpp"

namespace tinytensor {

class Interpreter {
public:
    explicit Interpreter(const Graph* graph);

    std::vector<Tensor> run(const std::unordered_map<const Node*, Tensor>& inputs,
                            Profiler* profiler = nullptr);

    Tensor run_single(const std::unordered_map<const Node*, Tensor>& inputs,
                      Profiler* profiler = nullptr);

private:
    const Graph* graph_;
    std::vector<Node*> execution_order_;
};

} // namespace tinytensor
