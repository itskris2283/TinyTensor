#pragma once

#include <unordered_map>
#include <vector>
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/graph/graph.hpp"

namespace tinytensor {

class Autodiff {
public:
    // Computes numerical gradients for specified input nodes given the input bindings
    static std::unordered_map<const Node*, Tensor> compute_gradients(
        const Graph& graph,
        const Node* loss_node,
        const std::unordered_map<const Node*, Tensor>& input_bindings);

    // Unbroadcasting helper: reduces a gradient tensor to target_shape
    static Tensor unbroadcast(const Tensor& grad, const Shape& target_shape);
};

} // namespace tinytensor
