#include "tinytensor/ir/printer.hpp"
#include <sstream>

namespace tinytensor {

std::string IRPrinter::to_string(const Graph& graph) {
    std::ostringstream ss;
    std::vector<Node*> topo = graph.topological_sort();

    for (Node* node : topo) {
        ss << "%" << node->id() << " = " << op_type_to_string(node->op_type());

        if (node->is_input()) {
            ss << "<" << dtype_to_string(node->dtype()) << ">[";
            for (size_t i = 0; i < node->shape().ndim(); ++i) {
                ss << node->shape()[i];
                if (i + 1 < node->shape().ndim()) ss << ", ";
            }
            ss << "]";
            if (!node->name().empty()) {
                ss << " ; name=\"" << node->name() << "\"";
            }
        } else if (node->is_constant()) {
            ss << "<" << dtype_to_string(node->dtype()) << ">[";
            for (size_t i = 0; i < node->shape().ndim(); ++i) {
                ss << node->shape()[i];
                if (i + 1 < node->shape().ndim()) ss << ", ";
            }
            ss << "]";
        } else {
            ss << " ";
            for (size_t i = 0; i < node->inputs().size(); ++i) {
                ss << "%" << node->inputs()[i]->id();
                if (i + 1 < node->inputs().size()) ss << ", ";
            }
            if (node->op_type() == OpType::Transpose) {
                ss << " dims=[" << node->dim0() << ", " << node->dim1() << "]";
            } else if (node->op_type() == OpType::ReduceSum || node->op_type() == OpType::ReduceMean) {
                ss << " axes=[";
                for (size_t i = 0; i < node->axes().size(); ++i) {
                    ss << node->axes()[i];
                    if (i + 1 < node->axes().size()) ss << ", ";
                }
                ss << "] keepdims=" << (node->keepdims() ? "true" : "false");
            } else if (node->op_type() == OpType::FusedElementwise) {
                ss << " expr=\"" << node->fusion_expression() << "\"";
            }
        }
        ss << "\n";
    }

    if (!graph.outputs().empty()) {
        ss << "return ";
        for (size_t i = 0; i < graph.outputs().size(); ++i) {
            ss << "%" << graph.outputs()[i]->id();
            if (i + 1 < graph.outputs().size()) ss << ", ";
        }
        ss << "\n";
    }

    return ss.str();
}

std::string IRPrinter::to_dot(const Graph& graph) {
    std::ostringstream ss;
    ss << "digraph TinyTensorGraph {\n";
    ss << "  rankdir=TB;\n";
    ss << "  node [shape=box, fontname=\"Helvetica\", style=filled, fillcolor=\"#f0f4f8\"];\n";

    std::vector<Node*> topo = graph.topological_sort();
    for (Node* node : topo) {
        ss << "  node_" << node->id() << " [label=\"%" << node->id() << "\\n"
           << op_type_to_string(node->op_type()) << "\\n"
           << node->shape().to_string() << "\\n"
           << dtype_to_string(node->dtype()) << "\"";
        if (node->is_input()) {
            ss << ", fillcolor=\"#d1e7dd\"";
        } else if (node->is_constant()) {
            ss << ", fillcolor=\"#fff3cd\"";
        }
        ss << "];\n";

        for (Node* in : node->inputs()) {
            ss << "  node_" << in->id() << " -> node_" << node->id() << ";\n";
        }
    }

    for (Node* out : graph.outputs()) {
        ss << "  node_" << out->id() << " [penwidth=2.5, color=\"#0d6efd\"];\n";
    }

    ss << "}\n";
    return ss.str();
}

} // namespace tinytensor
