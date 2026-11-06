#include "tinytensor/runtime/interpreter.hpp"
#include <chrono>
#include "tinytensor/ir/verifier.hpp"
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

Interpreter::Interpreter(const Graph* graph)
    : graph_(graph) {
    if (!graph_) {
        throw RuntimeExecutionError("Null graph passed to Interpreter");
    }
    IRVerifier::verify(*graph_);
    execution_order_ = graph_->topological_sort();
}

std::vector<Tensor> Interpreter::run(const std::unordered_map<const Node*, Tensor>& inputs,
                                     Profiler* profiler) {
    if (profiler) {
        profiler->clear();
        profiler->start_timer();
    }

    // 1. Validate inputs
    for (const Node* input_node : graph_->inputs()) {
        auto it = inputs.find(input_node);
        if (it == inputs.end()) {
            throw RuntimeExecutionError("Missing input binding for node %" +
                                        std::to_string(input_node->id()) +
                                        " (\"" + input_node->name() + "\")");
        }
        const Tensor& t = it->second;
        if (t.shape() != input_node->shape()) {
            throw ShapeError("Provided input tensor shape " + t.shape().to_string() +
                             " does not match graph input node shape " +
                             input_node->shape().to_string());
        }
        if (t.dtype() != input_node->dtype()) {
            throw TypeError("Provided input tensor dtype does not match graph input node dtype");
        }
    }

    std::unordered_map<const Node*, Tensor> env;

    // 2. Execute nodes in topological order
    for (Node* node : execution_order_) {
        auto start_node_time = std::chrono::high_resolution_clock::now();
        Tensor out;

        switch (node->op_type()) {
            case OpType::Input: {
                out = inputs.at(node);
                break;
            }
            case OpType::Constant: {
                out = node->constant_value();
                break;
            }
            case OpType::Add: {
                out = ops::add(env.at(node->inputs()[0]), env.at(node->inputs()[1]));
                break;
            }
            case OpType::Sub: {
                out = ops::sub(env.at(node->inputs()[0]), env.at(node->inputs()[1]));
                break;
            }
            case OpType::Mul: {
                out = ops::mul(env.at(node->inputs()[0]), env.at(node->inputs()[1]));
                break;
            }
            case OpType::Div: {
                out = ops::div(env.at(node->inputs()[0]), env.at(node->inputs()[1]));
                break;
            }
            case OpType::ReLU: {
                out = ops::relu(env.at(node->inputs()[0]));
                break;
            }
            case OpType::Sigmoid: {
                out = ops::sigmoid(env.at(node->inputs()[0]));
                break;
            }
            case OpType::MatMul: {
                out = ops::matmul(env.at(node->inputs()[0]), env.at(node->inputs()[1]));
                break;
            }
            case OpType::Transpose: {
                out = ops::transpose(env.at(node->inputs()[0]), node->dim0(), node->dim1());
                break;
            }
            case OpType::ReduceSum: {
                out = ops::reduce_sum(env.at(node->inputs()[0]), node->axes(), node->keepdims());
                break;
            }
            case OpType::ReduceMean: {
                out = ops::reduce_mean(env.at(node->inputs()[0]), node->axes(), node->keepdims());
                break;
            }
            case OpType::FusedElementwise: {
                const auto& expr = node->fusion_expression();
                const auto& in0 = env.at(node->inputs()[0]);
                const auto& in1 = env.at(node->inputs()[1]);
                Tensor inner;
                if (expr.find("add") != std::string::npos) {
                    inner = ops::add(in0, in1);
                } else if (expr.find("sub") != std::string::npos) {
                    inner = ops::sub(in0, in1);
                } else if (expr.find("mul") != std::string::npos) {
                    inner = ops::mul(in0, in1);
                } else {
                    inner = ops::add(in0, in1);
                }
                if (expr.find("relu") != std::string::npos) {
                    out = ops::relu(inner);
                } else if (expr.find("sigmoid") != std::string::npos) {
                    out = ops::sigmoid(inner);
                } else {
                    out = inner;
                }
                break;
            }
            default:
                throw RuntimeExecutionError("Unsupported op in Interpreter: " +
                                            op_type_to_string(node->op_type()));
        }

        if (profiler) {
            auto end_node_time = std::chrono::high_resolution_clock::now();
            double duration_us = std::chrono::duration<double, std::micro>(
                end_node_time - start_node_time).count();
            profiler->record_node(node->id(), op_type_to_string(node->op_type()),
                                  duration_us, out.nbytes());
        }

        env[node] = std::move(out);
    }

    if (profiler) {
        profiler->stop_timer();
    }

    // 3. Collect outputs
    std::vector<Tensor> results;
    results.reserve(graph_->outputs().size());
    for (Node* out_node : graph_->outputs()) {
        results.push_back(env.at(out_node));
    }

    return results;
}

Tensor Interpreter::run_single(const std::unordered_map<const Node*, Tensor>& inputs,
                               Profiler* profiler) {
    std::vector<Tensor> outputs = run(inputs, profiler);
    if (outputs.empty()) {
        throw RuntimeExecutionError("Graph has no outputs");
    }
    return outputs[0];
}

} // namespace tinytensor
