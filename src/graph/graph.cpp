#include "tinytensor/graph/graph.hpp"
#include <algorithm>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

Node* Graph::add_node(std::unique_ptr<Node> node) {
    Node* ptr = node.get();
    for (Node* in : ptr->inputs()) {
        if (in) {
            in->add_user(ptr);
        }
    }
    nodes_.push_back(std::move(node));
    return ptr;
}

Node* Graph::create_input(Shape shape, DType dtype, std::string name) {
    if (name.empty()) {
        name = "input_" + std::to_string(inputs_.size());
    }
    auto node = std::make_unique<Node>(next_node_id_++, OpType::Input, std::move(shape), dtype,
                                       std::vector<Node*>{}, std::move(name));
    Node* ptr = add_node(std::move(node));
    inputs_.push_back(ptr);
    return ptr;
}

Node* Graph::create_constant(Tensor value, std::string name) {
    if (name.empty()) {
        name = "const_" + std::to_string(next_node_id_);
    }
    Shape s = value.shape();
    DType d = value.dtype();
    auto node = std::make_unique<Node>(next_node_id_++, OpType::Constant, std::move(s), d,
                                       std::vector<Node*>{}, std::move(name));
    node->set_constant_value(std::move(value));
    return add_node(std::move(node));
}

Node* Graph::create_op(OpType op_type, std::vector<Node*> inputs, Shape shape, DType dtype) {
    auto node = std::make_unique<Node>(next_node_id_++, op_type, std::move(shape), dtype,
                                       std::move(inputs));
    return add_node(std::move(node));
}

Node* Graph::add(Node* a, Node* b) {
    if (!a || !b) throw IRError("Null operand in add");
    if (a->dtype() != b->dtype()) throw TypeError("DType mismatch in add");
    Shape s = Tensor::broadcast_shapes(a->shape(), b->shape());
    return create_op(OpType::Add, {a, b}, std::move(s), a->dtype());
}

Node* Graph::sub(Node* a, Node* b) {
    if (!a || !b) throw IRError("Null operand in sub");
    if (a->dtype() != b->dtype()) throw TypeError("DType mismatch in sub");
    Shape s = Tensor::broadcast_shapes(a->shape(), b->shape());
    return create_op(OpType::Sub, {a, b}, std::move(s), a->dtype());
}

Node* Graph::mul(Node* a, Node* b) {
    if (!a || !b) throw IRError("Null operand in mul");
    if (a->dtype() != b->dtype()) throw TypeError("DType mismatch in mul");
    Shape s = Tensor::broadcast_shapes(a->shape(), b->shape());
    return create_op(OpType::Mul, {a, b}, std::move(s), a->dtype());
}

Node* Graph::div(Node* a, Node* b) {
    if (!a || !b) throw IRError("Null operand in div");
    if (a->dtype() != b->dtype()) throw TypeError("DType mismatch in div");
    Shape s = Tensor::broadcast_shapes(a->shape(), b->shape());
    return create_op(OpType::Div, {a, b}, std::move(s), a->dtype());
}

Node* Graph::relu(Node* a) {
    if (!a) throw IRError("Null operand in relu");
    return create_op(OpType::ReLU, {a}, a->shape(), a->dtype());
}

Node* Graph::sigmoid(Node* a) {
    if (!a) throw IRError("Null operand in sigmoid");
    return create_op(OpType::Sigmoid, {a}, a->shape(), a->dtype());
}

Node* Graph::matmul(Node* a, Node* b) {
    if (!a || !b) throw IRError("Null operand in matmul");
    if (a->dtype() != b->dtype()) throw TypeError("DType mismatch in matmul");

    if (a->shape().ndim() == 2 && b->shape().ndim() == 2) {
        int64_t M = a->shape()[0];
        int64_t K_a = a->shape()[1];
        int64_t K_b = b->shape()[0];
        int64_t N = b->shape()[1];
        if (K_a != K_b) {
            throw ShapeError("Matmul inner dimensions mismatch: " +
                             std::to_string(K_a) + " vs " + std::to_string(K_b));
        }
        return create_op(OpType::MatMul, {a, b}, Shape({M, N}), a->dtype());
    }

    // Batched matmul shape deduction
    if (a->shape().ndim() >= 2 && b->shape().ndim() >= 2) {
        size_t rank_a = a->shape().ndim();
        size_t rank_b = b->shape().ndim();
        std::vector<int64_t> batch_a(a->shape().dims().begin(), a->shape().dims().end() - 2);
        std::vector<int64_t> batch_b(b->shape().dims().begin(), b->shape().dims().end() - 2);
        Shape batch_shape = Tensor::broadcast_shapes(Shape(batch_a), Shape(batch_b));

        int64_t M = a->shape()[rank_a - 2];
        int64_t K_a = a->shape()[rank_a - 1];
        int64_t K_b = b->shape()[rank_b - 2];
        int64_t N = b->shape()[rank_b - 1];
        if (K_a != K_b) {
            throw ShapeError("Batched matmul inner dimensions mismatch");
        }
        std::vector<int64_t> out_dims = batch_shape.dims();
        out_dims.push_back(M);
        out_dims.push_back(N);
        return create_op(OpType::MatMul, {a, b}, Shape(std::move(out_dims)), a->dtype());
    }

    throw DimensionError("Unsupported ranks for matmul");
}

Node* Graph::transpose(Node* a, int64_t dim0, int64_t dim1) {
    if (!a) throw IRError("Null operand in transpose");
    int64_t rank = static_cast<int64_t>(a->shape().ndim());
    if (dim0 < 0) dim0 += rank;
    if (dim1 < 0) dim1 += rank;
    if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
        throw DimensionError("Invalid transpose axes for rank " + std::to_string(rank));
    }
    std::vector<int64_t> new_dims = a->shape().dims();
    std::swap(new_dims[static_cast<size_t>(dim0)], new_dims[static_cast<size_t>(dim1)]);
    Node* n = create_op(OpType::Transpose, {a}, Shape(std::move(new_dims)), a->dtype());
    n->set_transpose_dims(dim0, dim1);
    return n;
}

Node* Graph::reduce_sum(Node* a, const std::vector<int64_t>& axes, bool keepdims) {
    if (!a) throw IRError("Null operand in reduce_sum");
    int64_t rank = static_cast<int64_t>(a->shape().ndim());
    std::set<int64_t> norm_axes;
    if (axes.empty()) {
        for (int64_t i = 0; i < rank; ++i) norm_axes.insert(i);
    } else {
        for (int64_t ax : axes) {
            if (ax < 0) ax += rank;
            if (ax < 0 || ax >= rank) throw DimensionError("Axis out of range in reduce_sum");
            norm_axes.insert(ax);
        }
    }

    std::vector<int64_t> out_dims;
    for (int64_t i = 0; i < rank; ++i) {
        if (norm_axes.count(i)) {
            if (keepdims) out_dims.push_back(1);
        } else {
            out_dims.push_back(a->shape()[static_cast<size_t>(i)]);
        }
    }

    Node* n = create_op(OpType::ReduceSum, {a}, Shape(std::move(out_dims)), a->dtype());
    n->set_axes(axes);
    n->set_keepdims(keepdims);
    return n;
}

Node* Graph::reduce_mean(Node* a, const std::vector<int64_t>& axes, bool keepdims) {
    if (!a) throw IRError("Null operand in reduce_mean");
    int64_t rank = static_cast<int64_t>(a->shape().ndim());
    std::set<int64_t> norm_axes;
    if (axes.empty()) {
        for (int64_t i = 0; i < rank; ++i) norm_axes.insert(i);
    } else {
        for (int64_t ax : axes) {
            if (ax < 0) ax += rank;
            if (ax < 0 || ax >= rank) throw DimensionError("Axis out of range in reduce_mean");
            norm_axes.insert(ax);
        }
    }

    std::vector<int64_t> out_dims;
    for (int64_t i = 0; i < rank; ++i) {
        if (norm_axes.count(i)) {
            if (keepdims) out_dims.push_back(1);
        } else {
            out_dims.push_back(a->shape()[static_cast<size_t>(i)]);
        }
    }

    Node* n = create_op(OpType::ReduceMean, {a}, Shape(std::move(out_dims)), a->dtype());
    n->set_axes(axes);
    n->set_keepdims(keepdims);
    return n;
}

void Graph::mark_output(Node* node) {
    if (!node) throw IRError("Cannot mark null node as output");
    if (std::find(outputs_.begin(), outputs_.end(), node) == outputs_.end()) {
        outputs_.push_back(node);
    }
}

std::vector<Node*> Graph::topological_sort() const {
    enum class State { Unvisited, Visiting, Visited };
    std::unordered_map<const Node*, State> states;
    std::vector<Node*> order;

    // DFS with cycle detection
    auto dfs = [&](auto& self, Node* node) -> void {
        State st = states[node];
        if (st == State::Visiting) {
            throw IRError("Cycle detected in computation graph at node %" +
                          std::to_string(node->id()));
        }
        if (st == State::Visited) {
            return;
        }

        states[node] = State::Visiting;
        for (Node* in : node->inputs()) {
            if (in) {
                self(self, in);
            }
        }
        states[node] = State::Visited;
        order.push_back(node);
    };

    // Traverse all nodes in the graph in topological order
    for (const auto& n : nodes_) {
        dfs(dfs, n.get());
    }

    return order;
}

std::unique_ptr<Graph> Graph::clone() const {
    auto new_graph = std::make_unique<Graph>();
    std::unordered_map<const Node*, Node*> old_to_new;

    for (const auto& old_node : nodes_) {
        std::vector<Node*> new_inputs;
        for (Node* in : old_node->inputs()) {
            new_inputs.push_back(old_to_new.at(in));
        }

        Node* new_node = nullptr;
        if (old_node->is_input()) {
            new_node = new_graph->create_input(old_node->shape(), old_node->dtype(), old_node->name());
        } else if (old_node->is_constant()) {
            new_node = new_graph->create_constant(old_node->constant_value().clone(), old_node->name());
        } else {
            new_node = new_graph->create_op(old_node->op_type(), new_inputs, old_node->shape(), old_node->dtype());
            new_node->set_axes(old_node->axes());
            new_node->set_keepdims(old_node->keepdims());
            new_node->set_transpose_dims(old_node->dim0(), old_node->dim1());
            new_node->set_fusion_expression(old_node->fusion_expression());
        }
        old_to_new[old_node.get()] = new_node;
    }

    for (Node* out : outputs_) {
        new_graph->mark_output(old_to_new.at(out));
    }

    return new_graph;
}

} // namespace tinytensor
