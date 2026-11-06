#include "tinytensor/optimizer/constant_folding.hpp"
#include <unordered_map>
#include "tinytensor/core/operators.hpp"
#include "tinytensor/optimizer/dead_code_elimination.hpp"

namespace tinytensor {

std::unique_ptr<Graph> ConstantFoldingPass::run(const Graph& graph, OptimizationReport* report) {
    auto new_graph = std::make_unique<Graph>();
    std::unordered_map<const Node*, Node*> old_to_new;
    size_t fold_count = 0;

    std::vector<Node*> topo = graph.topological_sort();

    for (Node* old_node : topo) {
        if (old_node->is_input()) {
            Node* n = new_graph->create_input(old_node->shape(), old_node->dtype(), old_node->name());
            old_to_new[old_node] = n;
            continue;
        }

        if (old_node->is_constant()) {
            Node* n = new_graph->create_constant(old_node->constant_value().clone(), old_node->name());
            old_to_new[old_node] = n;
            continue;
        }

        // Check if all inputs are compile-time constants
        bool all_const = true;
        for (Node* in : old_node->inputs()) {
            Node* mapped_in = old_to_new.at(in);
            if (!mapped_in->is_constant()) {
                all_const = false;
                break;
            }
        }

        if (all_const && !old_node->inputs().empty()) {
            // Fold constant operation
            Tensor val;
            switch (old_node->op_type()) {
                case OpType::Add:
                    val = ops::add(old_to_new.at(old_node->inputs()[0])->constant_value(),
                                   old_to_new.at(old_node->inputs()[1])->constant_value());
                    break;
                case OpType::Sub:
                    val = ops::sub(old_to_new.at(old_node->inputs()[0])->constant_value(),
                                   old_to_new.at(old_node->inputs()[1])->constant_value());
                    break;
                case OpType::Mul:
                    val = ops::mul(old_to_new.at(old_node->inputs()[0])->constant_value(),
                                   old_to_new.at(old_node->inputs()[1])->constant_value());
                    break;
                case OpType::Div:
                    val = ops::div(old_to_new.at(old_node->inputs()[0])->constant_value(),
                                   old_to_new.at(old_node->inputs()[1])->constant_value());
                    break;
                case OpType::ReLU:
                    val = ops::relu(old_to_new.at(old_node->inputs()[0])->constant_value());
                    break;
                case OpType::Sigmoid:
                    val = ops::sigmoid(old_to_new.at(old_node->inputs()[0])->constant_value());
                    break;
                case OpType::MatMul:
                    val = ops::matmul(old_to_new.at(old_node->inputs()[0])->constant_value(),
                                     old_to_new.at(old_node->inputs()[1])->constant_value());
                    break;
                case OpType::Transpose:
                    val = ops::transpose(old_to_new.at(old_node->inputs()[0])->constant_value(),
                                        old_node->dim0(), old_node->dim1());
                    break;
                case OpType::ReduceSum:
                    val = ops::reduce_sum(old_to_new.at(old_node->inputs()[0])->constant_value(),
                                         old_node->axes(), old_node->keepdims());
                    break;
                case OpType::ReduceMean:
                    val = ops::reduce_mean(old_to_new.at(old_node->inputs()[0])->constant_value(),
                                          old_node->axes(), old_node->keepdims());
                    break;
                default:
                    all_const = false;
                    break;
            }

            if (all_const) {
                Node* n = new_graph->create_constant(std::move(val), "folded_" + std::to_string(old_node->id()));
                old_to_new[old_node] = n;
                fold_count++;
                continue;
            }
        }

        // Otherwise copy node with mapped inputs
        std::vector<Node*> new_inputs;
        for (Node* in : old_node->inputs()) {
            new_inputs.push_back(old_to_new.at(in));
        }
        Node* n = new_graph->create_op(old_node->op_type(), new_inputs, old_node->shape(), old_node->dtype());
        n->set_axes(old_node->axes());
        n->set_keepdims(old_node->keepdims());
        n->set_transpose_dims(old_node->dim0(), old_node->dim1());
        n->set_fusion_expression(old_node->fusion_expression());
        old_to_new[old_node] = n;
    }

    for (Node* out : graph.outputs()) {
        new_graph->mark_output(old_to_new.at(out));
    }

    if (report) {
        report->constant_folding_count += fold_count;
    }

    return DeadCodeEliminationPass::run(*new_graph, report);
}

} // namespace tinytensor
