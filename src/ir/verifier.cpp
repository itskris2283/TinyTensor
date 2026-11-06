#include "tinytensor/ir/verifier.hpp"
#include <unordered_set>
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

void IRVerifier::verify(const Graph& graph) {
    std::vector<Node*> topo_order = graph.topological_sort();
    std::unordered_set<const Node*> visited_nodes;

    for (Node* node : topo_order) {
        if (!node) {
            throw IRError("Encountered null node in topological order");
        }

        // 1. Operand topological correctness
        for (Node* in : node->inputs()) {
            if (!in) {
                throw IRError("Node %" + std::to_string(node->id()) + " has a null input operand");
            }
            if (visited_nodes.find(in) == visited_nodes.end()) {
                throw IRError("Topological ordering violation: operand %" +
                              std::to_string(in->id()) + " used by %" +
                              std::to_string(node->id()) + " before definition");
            }
        }

        // 2. Arity and operation-specific semantic checks
        switch (node->op_type()) {
            case OpType::Input: {
                if (!node->inputs().empty()) {
                    throw IRError("Input node %" + std::to_string(node->id()) + " must have 0 inputs");
                }
                for (int64_t dim : node->shape().dims()) {
                    if (dim < 0) {
                        throw IRError("Input node %" + std::to_string(node->id()) +
                                      " has negative dimension: " + std::to_string(dim));
                    }
                }
                break;
            }
            case OpType::Constant: {
                if (!node->inputs().empty()) {
                    throw IRError("Constant node %" + std::to_string(node->id()) + " must have 0 inputs");
                }
                if (node->constant_value().shape() != node->shape()) {
                    throw IRError("Constant node %" + std::to_string(node->id()) +
                                  " shape mismatch with underlying tensor");
                }
                if (node->constant_value().dtype() != node->dtype()) {
                    throw IRError("Constant node %" + std::to_string(node->id()) +
                                  " dtype mismatch with underlying tensor");
                }
                break;
            }
            case OpType::Add:
            case OpType::Sub:
            case OpType::Mul:
            case OpType::Div: {
                if (node->inputs().size() != 2) {
                    throw IRError("Binary op %" + std::to_string(node->id()) + " requires exactly 2 inputs");
                }
                Node* a = node->inputs()[0];
                Node* b = node->inputs()[1];
                if (a->dtype() != b->dtype() || a->dtype() != node->dtype()) {
                    throw IRError("Type mismatch in binary op %" + std::to_string(node->id()));
                }
                Shape expected_shape = Tensor::broadcast_shapes(a->shape(), b->shape());
                if (node->shape() != expected_shape) {
                    throw IRError("Shape mismatch in binary op %" + std::to_string(node->id()) +
                                  ": expected " + expected_shape.to_string() + " but got " +
                                  node->shape().to_string());
                }
                break;
            }
            case OpType::ReLU:
            case OpType::Sigmoid: {
                if (node->inputs().size() != 1) {
                    throw IRError("Unary op %" + std::to_string(node->id()) + " requires exactly 1 input");
                }
                Node* a = node->inputs()[0];
                if (a->dtype() != node->dtype()) {
                    throw IRError("Type mismatch in unary op %" + std::to_string(node->id()));
                }
                if (node->shape() != a->shape()) {
                    throw IRError("Shape mismatch in unary op %" + std::to_string(node->id()));
                }
                break;
            }
            case OpType::MatMul: {
                if (node->inputs().size() != 2) {
                    throw IRError("MatMul node %" + std::to_string(node->id()) + " requires 2 inputs");
                }
                Node* a = node->inputs()[0];
                Node* b = node->inputs()[1];
                if (a->dtype() != b->dtype() || a->dtype() != node->dtype()) {
                    throw IRError("Type mismatch in matmul %" + std::to_string(node->id()));
                }
                if (a->shape().ndim() < 2 || b->shape().ndim() < 2) {
                    throw IRError("MatMul operands must have rank >= 2");
                }
                size_t rank_a = a->shape().ndim();
                size_t rank_b = b->shape().ndim();
                int64_t k_a = a->shape()[rank_a - 1];
                int64_t k_b = b->shape()[rank_b - 2];
                if (k_a != k_b) {
                    throw IRError("Inner dimension mismatch in matmul %" + std::to_string(node->id()));
                }
                std::vector<int64_t> batch_a(a->shape().dims().begin(), a->shape().dims().end() - 2);
                std::vector<int64_t> batch_b(b->shape().dims().begin(), b->shape().dims().end() - 2);
                Shape batch_shape = Tensor::broadcast_shapes(Shape(batch_a), Shape(batch_b));
                std::vector<int64_t> expected_dims = batch_shape.dims();
                expected_dims.push_back(a->shape()[rank_a - 2]);
                expected_dims.push_back(b->shape()[rank_b - 1]);
                if (node->shape() != Shape(expected_dims)) {
                    throw IRError("Shape mismatch in matmul %" + std::to_string(node->id()));
                }
                break;
            }
            case OpType::Transpose: {
                if (node->inputs().size() != 1) {
                    throw IRError("Transpose node requires 1 input");
                }
                Node* a = node->inputs()[0];
                if (a->dtype() != node->dtype()) {
                    throw IRError("Type mismatch in transpose %" + std::to_string(node->id()));
                }
                int64_t rank = static_cast<int64_t>(a->shape().ndim());
                int64_t d0 = node->dim0();
                int64_t d1 = node->dim1();
                if (d0 < 0 || d0 >= rank || d1 < 0 || d1 >= rank) {
                    throw IRError("Transpose axes out of bounds for rank " + std::to_string(rank));
                }
                std::vector<int64_t> expected_dims = a->shape().dims();
                std::swap(expected_dims[static_cast<size_t>(d0)], expected_dims[static_cast<size_t>(d1)]);
                if (node->shape() != Shape(expected_dims)) {
                    throw IRError("Shape mismatch in transpose %" + std::to_string(node->id()));
                }
                break;
            }
            case OpType::ReduceSum:
            case OpType::ReduceMean: {
                if (node->inputs().size() != 1) {
                    throw IRError("Reduction node requires 1 input");
                }
                Node* a = node->inputs()[0];
                if (a->dtype() != node->dtype()) {
                    throw IRError("Type mismatch in reduction %" + std::to_string(node->id()));
                }
                int64_t rank = static_cast<int64_t>(a->shape().ndim());
                std::unordered_set<int64_t> norm_axes;
                const auto& axes = node->axes();
                for (int64_t ax : axes) {
                    int64_t orig_ax = ax;
                    if (ax < 0) ax += rank;
                    if (ax < 0 || ax >= rank) {
                        throw IRError("Reduction axis " + std::to_string(orig_ax) +
                                      " out of bounds for rank " + std::to_string(rank));
                    }
                    norm_axes.insert(ax);
                }
                std::vector<int64_t> expected_dims;
                for (int64_t i = 0; i < rank; ++i) {
                    if (axes.empty() || norm_axes.count(i)) {
                        if (node->keepdims()) expected_dims.push_back(1);
                    } else {
                        expected_dims.push_back(a->shape()[static_cast<size_t>(i)]);
                    }
                }
                if (node->shape() != Shape(expected_dims)) {
                    throw IRError("Shape mismatch in reduction %" + std::to_string(node->id()));
                }
                break;
            }
            case OpType::FusedElementwise: {
                if (node->inputs().empty()) {
                    throw IRError("Fused elementwise op requires at least 1 input");
                }
                for (Node* in : node->inputs()) {
                    if (in->dtype() != node->dtype()) {
                        throw IRError("Type mismatch in fused elementwise op %" + std::to_string(node->id()));
                    }
                }
                break;
            }
            default:
                break;
        }

        visited_nodes.insert(node);
    }

    // 3. Verify outputs
    for (Node* out : graph.outputs()) {
        if (!out) {
            throw IRError("Graph output cannot be null");
        }
        if (visited_nodes.find(out) == visited_nodes.end()) {
            throw IRError("Graph output %" + std::to_string(out->id()) + " does not exist in graph");
        }
    }
}

} // namespace tinytensor
