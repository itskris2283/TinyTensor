#pragma once

#include <memory>
#include <string>
#include <vector>
#include "tinytensor/core/dtype.hpp"
#include "tinytensor/core/shape.hpp"
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/graph/node.hpp"
#include "tinytensor/ir/operation.hpp"

namespace tinytensor {

class Graph {
public:
    Graph() = default;
    ~Graph() = default;

    // Non-copyable, movable
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;

    // Node factories
    Node* create_input(Shape shape, DType dtype = DType::Float32, std::string name = "");
    Node* create_constant(Tensor value, std::string name = "");
    Node* create_op(OpType op_type, std::vector<Node*> inputs, Shape shape, DType dtype);

    // Operator builders
    Node* add(Node* a, Node* b);
    Node* sub(Node* a, Node* b);
    Node* mul(Node* a, Node* b);
    Node* div(Node* a, Node* b);
    Node* relu(Node* a);
    Node* sigmoid(Node* a);
    Node* matmul(Node* a, Node* b);
    Node* transpose(Node* a, int64_t dim0, int64_t dim1);
    Node* reduce_sum(Node* a, const std::vector<int64_t>& axes = {}, bool keepdims = false);
    Node* reduce_mean(Node* a, const std::vector<int64_t>& axes = {}, bool keepdims = false);

    // Output marking
    void mark_output(Node* node);
    const std::vector<Node*>& outputs() const noexcept { return outputs_; }
    const std::vector<Node*>& inputs() const noexcept { return inputs_; }
    const std::vector<std::unique_ptr<Node>>& nodes() const noexcept { return nodes_; }
    size_t num_nodes() const noexcept { return nodes_.size(); }

    // Topological traversal with cycle detection
    std::vector<Node*> topological_sort() const;

    // Deep clone graph
    std::unique_ptr<Graph> clone() const;

private:
    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<Node*> inputs_;
    std::vector<Node*> outputs_;
    size_t next_node_id_{0};

    Node* add_node(std::unique_ptr<Node> node);
};

} // namespace tinytensor
