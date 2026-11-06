#include "tinytensor/runtime/executor.hpp"
#include <sstream>
#include "tinytensor/ir/printer.hpp"
#include "tinytensor/ir/verifier.hpp"
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

Executor::Executor(const Graph& graph, ExecutionMode mode,
                   bool optimize_graph, KernelCache* cache)
    : mode_(mode) {
    original_ir_ = IRPrinter::to_string(graph);

    for (Node* in : graph.inputs()) {
        original_inputs_.push_back(in);
    }

    if (optimize_graph) {
        Optimizer opt;
        owned_graph_ = opt.optimize(graph, &opt_report_);
        graph_ = owned_graph_.get();
    } else {
        graph_ = &graph;
    }

    optimized_ir_ = IRPrinter::to_string(*graph_);
    IRVerifier::verify(*graph_);

    for (Node* in : graph_->inputs()) {
        input_nodes_.push_back(in);
    }
    for (Node* out : graph_->outputs()) {
        output_nodes_.push_back(out);
    }

    if (mode_ == ExecutionMode::Interpreted) {
        interpreter_ = std::make_unique<Interpreter>(graph_);
    } else {
        CompileOptions opts;
        if (mode_ == ExecutionMode::CompiledAVX2) {
            opts.target_isa = simd::supported_target_isa();
            opts.enable_fma = simd::has_fma3();
        } else if (mode_ == ExecutionMode::CompiledAVX2_FMA) {
            opts.target_isa = TargetISA::AVX2_FMA;
            opts.enable_fma = true;
        } else if (mode_ == ExecutionMode::CompiledAVX2_NoFMA) {
            opts.target_isa = TargetISA::AVX2;
            opts.enable_fma = false;
        } else {
            opts.target_isa = TargetISA::Scalar;
            opts.enable_fma = false;
        }
        CppBackend backend(opts);
        executable_ = backend.compile(*graph_, cache);
    }
}

std::string Executor::generated_source() const noexcept {
    return executable_ ? executable_->source_code() : "";
}

std::string Executor::execution_plan() const {
    std::ostringstream ss;
    ss << "=== TinyTensor Execution Plan ===\n";
    ss << "Execution Mode: ";
    switch (mode_) {
        case ExecutionMode::Interpreted: ss << "Interpreted\n"; break;
        case ExecutionMode::CompiledScalar: ss << "Compiled Scalar\n"; break;
        case ExecutionMode::CompiledAVX2: ss << "Compiled AVX2\n"; break;
        case ExecutionMode::CompiledAVX2_FMA: ss << "Compiled AVX2+FMA\n"; break;
        case ExecutionMode::CompiledAVX2_NoFMA: ss << "Compiled AVX2 (No FMA)\n"; break;
    }
    ss << "Input Nodes (" << input_nodes_.size() << "):\n";
    for (const Node* in : input_nodes_) {
        ss << "  %" << in->id() << " [" << in->shape().to_string() << ", "
           << dtype_to_string(in->dtype()) << "]";
        if (!in->name().empty()) ss << " name=\"" << in->name() << "\"";
        ss << "\n";
    }
    ss << "Output Nodes (" << output_nodes_.size() << "):\n";
    for (const Node* out : output_nodes_) {
        ss << "  %" << out->id() << " [" << out->shape().to_string() << ", "
           << dtype_to_string(out->dtype()) << "]\n";
    }
    ss << "Execution Sequence (Topological, " << graph_->num_nodes() << " nodes):\n";
    for (Node* n : graph_->topological_sort()) {
        ss << "  %" << n->id() << " = " << op_type_to_string(n->op_type()) << " "
           << n->shape().to_string() << " (" << dtype_to_string(n->dtype()) << ")\n";
    }
    if (executable_) {
        ss << "Binary Object: " << executable_->so_path() << "\n";
    }
    ss << "Memory Planning: Peak " << opt_report_.peak_planned_memory_bytes
       << " bytes (saved " << opt_report_.memory_savings_bytes << " bytes via "
       << opt_report_.reused_buffers_count << " reused buffers)\n";
    ss << "=================================";
    return ss.str();
}

std::vector<Tensor> Executor::run(const std::unordered_map<const Node*, Tensor>& inputs,
                                  Profiler* profiler) {
    std::unordered_map<const Node*, Tensor> translated_inputs;
    std::vector<Tensor> in_vec;
    in_vec.reserve(input_nodes_.size());

    for (size_t i = 0; i < input_nodes_.size(); ++i) {
        const Node* in_node = input_nodes_[i];
        const Node* orig_node = (i < original_inputs_.size()) ? original_inputs_[i] : in_node;

        auto it = inputs.find(in_node);
        if (it == inputs.end()) {
            it = inputs.find(orig_node);
        }
        if (it == inputs.end()) {
            throw RuntimeExecutionError("Missing input binding for node %" +
                                        std::to_string(orig_node->id()));
        }
        translated_inputs[in_node] = it->second;
        in_vec.push_back(it->second);
    }

    if (mode_ == ExecutionMode::Interpreted) {
        return interpreter_->run(translated_inputs, profiler);
    }

    return executable_->run(in_vec, profiler);
}

Tensor Executor::run_single(const std::unordered_map<const Node*, Tensor>& inputs,
                            Profiler* profiler) {
    std::vector<Tensor> outputs = run(inputs, profiler);
    if (outputs.empty()) {
        throw RuntimeExecutionError("Executor graph produced no outputs");
    }
    return outputs[0];
}

} // namespace tinytensor
