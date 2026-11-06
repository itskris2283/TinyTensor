#include "tinytensor/optimizer/cse.hpp"
#include <tuple>
#include <unordered_map>

namespace tinytensor {

namespace {

struct CSENodeKey {
    OpType op_type;
    std::vector<size_t> input_ids;
    Shape shape;
    DType dtype;
    std::vector<int64_t> axes;
    bool keepdims;
    int64_t dim0;
    int64_t dim1;

    bool operator==(const CSENodeKey& other) const {
        return op_type == other.op_type &&
               input_ids == other.input_ids &&
               shape == other.shape &&
               dtype == other.dtype &&
               axes == other.axes &&
               keepdims == other.keepdims &&
               dim0 == other.dim0 &&
               dim1 == other.dim1;
    }
};

struct CSENodeKeyHash {
    size_t operator()(const CSENodeKey& k) const {
        size_t h = static_cast<size_t>(k.op_type);
        for (size_t id : k.input_ids) {
            h ^= (id + 0x9e3779b9 + (h << 6) + (h >> 2));
        }
        h ^= (static_cast<size_t>(k.dtype) << 3);
        return h;
    }
};

} // anonymous namespace

std::unique_ptr<Graph> CommonSubexpressionEliminationPass::run(const Graph& graph,
                                                              OptimizationReport* report) {
    auto new_graph = std::make_unique<Graph>();
    std::unordered_map<const Node*, Node*> old_to_new;
    std::unordered_map<CSENodeKey, Node*, CSENodeKeyHash> seen_ops;
    size_t cse_count = 0;

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

        std::vector<Node*> new_inputs;
        std::vector<size_t> input_ids;
        for (Node* in : old_node->inputs()) {
            Node* mapped_in = old_to_new.at(in);
            new_inputs.push_back(mapped_in);
            input_ids.push_back(mapped_in->id());
        }

        CSENodeKey key{
            old_node->op_type(),
            input_ids,
            old_node->shape(),
            old_node->dtype(),
            old_node->axes(),
            old_node->keepdims(),
            old_node->dim0(),
            old_node->dim1()
        };

        auto it = seen_ops.find(key);
        if (it != seen_ops.end()) {
            // Found duplicate computation! Reuse existing node
            old_to_new[old_node] = it->second;
            cse_count++;
        } else {
            Node* n = new_graph->create_op(old_node->op_type(), new_inputs, old_node->shape(), old_node->dtype());
            n->set_axes(old_node->axes());
            n->set_keepdims(old_node->keepdims());
            n->set_transpose_dims(old_node->dim0(), old_node->dim1());
            n->set_fusion_expression(old_node->fusion_expression());
            seen_ops[key] = n;
            old_to_new[old_node] = n;
        }
    }

    for (Node* out : graph.outputs()) {
        new_graph->mark_output(old_to_new.at(out));
    }

    if (report) {
        report->cse_eliminated_count += cse_count;
    }

    return new_graph;
}

} // namespace tinytensor
