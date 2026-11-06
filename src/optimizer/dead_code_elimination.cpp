#include "tinytensor/optimizer/dead_code_elimination.hpp"
#include <unordered_map>
#include <unordered_set>

namespace tinytensor {

std::unique_ptr<Graph> DeadCodeEliminationPass::run(const Graph& graph, OptimizationReport* report) {
    // 1. Mark reachable nodes from outputs
    std::unordered_set<const Node*> reachable;
    auto dfs = [&](auto& self, const Node* n) -> void {
        if (!n || reachable.count(n)) return;
        reachable.insert(n);
        for (Node* in : n->inputs()) {
            self(self, in);
        }
    };

    for (Node* out : graph.outputs()) {
        dfs(dfs, out);
    }

    size_t eliminated_count = 0;
    auto new_graph = std::make_unique<Graph>();
    std::unordered_map<const Node*, Node*> old_to_new;

    std::vector<Node*> topo = graph.topological_sort();

    for (Node* old_node : topo) {
        if (reachable.find(old_node) == reachable.end()) {
            eliminated_count++;
            continue;
        }

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
        if (old_to_new.count(out)) {
            new_graph->mark_output(old_to_new.at(out));
        }
    }

    if (report) {
        report->dce_eliminated_count += eliminated_count;
    }

    return new_graph;
}

} // namespace tinytensor
