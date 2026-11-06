#include "tinytensor/optimizer/fusion.hpp"
#include <unordered_map>
#include <unordered_set>

namespace tinytensor {

std::unique_ptr<Graph> ElementwiseFusionPass::run(const Graph& graph, OptimizationReport* report) {
    auto new_graph = std::make_unique<Graph>();
    std::unordered_map<const Node*, Node*> old_to_new;
    size_t fused_count = 0;

    std::vector<Node*> topo = graph.topological_sort();
    std::unordered_set<const Node*> output_set(graph.outputs().begin(), graph.outputs().end());

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

        // Check if old_node is an elementwise op suitable for fusion
        // Candidate: ReLU(Add(A, B)) or ReLU(Mul(A, B)) where inner op has 1 user and is not an output
        // Both operands must have matching shapes (no broadcasting) to allow a single flat loop
        if (old_node->op_type() == OpType::ReLU && old_node->inputs().size() == 1) {
            Node* inner_old = old_node->inputs()[0];
            bool same_shape = true;
            for (Node* in : inner_old->inputs()) {
                if (in->shape() != inner_old->shape()) {
                    same_shape = false;
                    break;
                }
            }
            if (inner_old->is_elementwise() &&
                same_shape &&
                inner_old->users().size() == 1 &&
                output_set.find(inner_old) == output_set.end() &&
                old_to_new.count(inner_old)) {

                Node* inner_new = old_to_new.at(inner_old);
                // Fuse inner_new and old_node
                std::string inner_op_str = op_type_to_string(inner_old->op_type());
                std::string expr = "relu(" + inner_op_str + "(@0, @1))";

                std::vector<Node*> fused_inputs = inner_new->inputs();
                Node* fused = new_graph->create_op(OpType::FusedElementwise, fused_inputs,
                                                   old_node->shape(), old_node->dtype());
                fused->set_fusion_expression(expr);
                old_to_new[old_node] = fused;
                fused_count++;
                continue;
            }
        }

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
        report->fused_groups_count += fused_count;
    }

    return new_graph;
}

} // namespace tinytensor
