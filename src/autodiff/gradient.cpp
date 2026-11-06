#include "tinytensor/autodiff/gradient.hpp"
#include <algorithm>
#include <set>
#include "tinytensor/core/operators.hpp"
#include "tinytensor/runtime/interpreter.hpp"
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

Tensor Autodiff::unbroadcast(const Tensor& grad, const Shape& target_shape) {
    if (grad.shape() == target_shape) {
        return grad;
    }
    size_t grad_rank = grad.ndim();
    size_t target_rank = target_shape.ndim();

    if (target_rank == 0) {
        return ops::reduce_sum(grad);
    }

    Tensor current = grad;

    // 1. Reduce leading dimensions if grad has higher rank
    if (grad_rank > target_rank) {
        size_t rank_diff = grad_rank - target_rank;
        std::vector<int64_t> lead_axes;
        for (size_t i = 0; i < rank_diff; ++i) {
            lead_axes.push_back(static_cast<int64_t>(i));
        }
        current = ops::reduce_sum(current, lead_axes, false);
    }

    // 2. Reduce singleton dimensions where target is 1 and current > 1
    std::vector<int64_t> single_axes;
    for (size_t i = 0; i < target_rank; ++i) {
        if (target_shape[i] == 1 && current.shape()[i] > 1) {
            single_axes.push_back(static_cast<int64_t>(i));
        }
    }

    if (!single_axes.empty()) {
        current = ops::reduce_sum(current, single_axes, true);
    }

    if (current.shape() != target_shape) {
        current = current.reshape(target_shape);
    }

    return current;
}

std::unordered_map<const Node*, Tensor> Autodiff::compute_gradients(
    const Graph& graph,
    const Node* loss_node,
    const std::unordered_map<const Node*, Tensor>& input_bindings) {

    if (!loss_node) {
        throw TinyTensorException("Loss node cannot be null in compute_gradients");
    }

    // 1. Forward pass: compute activations for all nodes
    std::vector<Node*> topo = graph.topological_sort();
    std::unordered_map<const Node*, Tensor> activations;

    for (Node* node : topo) {
        switch (node->op_type()) {
            case OpType::Input: {
                activations[node] = input_bindings.at(node);
                break;
            }
            case OpType::Constant: {
                activations[node] = node->constant_value();
                break;
            }
            case OpType::Add: {
                activations[node] = ops::add(activations.at(node->inputs()[0]),
                                             activations.at(node->inputs()[1]));
                break;
            }
            case OpType::Sub: {
                activations[node] = ops::sub(activations.at(node->inputs()[0]),
                                             activations.at(node->inputs()[1]));
                break;
            }
            case OpType::Mul: {
                activations[node] = ops::mul(activations.at(node->inputs()[0]),
                                             activations.at(node->inputs()[1]));
                break;
            }
            case OpType::Div: {
                activations[node] = ops::div(activations.at(node->inputs()[0]),
                                             activations.at(node->inputs()[1]));
                break;
            }
            case OpType::ReLU: {
                activations[node] = ops::relu(activations.at(node->inputs()[0]));
                break;
            }
            case OpType::Sigmoid: {
                activations[node] = ops::sigmoid(activations.at(node->inputs()[0]));
                break;
            }
            case OpType::MatMul: {
                activations[node] = ops::matmul(activations.at(node->inputs()[0]),
                                                activations.at(node->inputs()[1]));
                break;
            }
            case OpType::Transpose: {
                activations[node] = ops::transpose(activations.at(node->inputs()[0]),
                                                   node->dim0(), node->dim1());
                break;
            }
            case OpType::ReduceSum: {
                activations[node] = ops::reduce_sum(activations.at(node->inputs()[0]),
                                                    node->axes(), node->keepdims());
                break;
            }
            case OpType::ReduceMean: {
                activations[node] = ops::reduce_mean(activations.at(node->inputs()[0]),
                                                     node->axes(), node->keepdims());
                break;
            }
            default:
                throw TinyTensorException("Unsupported op type in forward autodiff pass");
        }
    }

    // 2. Backward pass: reverse topological traversal
    std::unordered_map<const Node*, Tensor> grads;
    // Seed gradient dL/dL = 1
    grads[loss_node] = Tensor::ones(loss_node->shape(), loss_node->dtype());

    auto accumulate_grad = [&](const Node* target, const Tensor& grad_contribution) {
        if (grads.find(target) == grads.end()) {
            grads[target] = grad_contribution;
        } else {
            grads[target] = ops::add(grads[target], grad_contribution);
        }
    };

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const Node* node = *it;
        auto git = grads.find(node);
        if (git == grads.end()) {
            continue;
        }
        const Tensor& g = git->second;

        switch (node->op_type()) {
            case OpType::Input:
            case OpType::Constant:
                // Leaf nodes
                break;

            case OpType::Add: {
                Node* a = node->inputs()[0];
                Node* b = node->inputs()[1];
                accumulate_grad(a, unbroadcast(g, a->shape()));
                accumulate_grad(b, unbroadcast(g, b->shape()));
                break;
            }

            case OpType::Sub: {
                Node* a = node->inputs()[0];
                Node* b = node->inputs()[1];
                accumulate_grad(a, unbroadcast(g, a->shape()));
                Tensor neg_g = ops::mul(g, Tensor::scalar(-1.0, g.dtype()));
                accumulate_grad(b, unbroadcast(neg_g, b->shape()));
                break;
            }

            case OpType::Mul: {
                Node* a = node->inputs()[0];
                Node* b = node->inputs()[1];
                Tensor ga = ops::mul(g, activations.at(b));
                Tensor gb = ops::mul(g, activations.at(a));
                accumulate_grad(a, unbroadcast(ga, a->shape()));
                accumulate_grad(b, unbroadcast(gb, b->shape()));
                break;
            }

            case OpType::Div: {
                Node* a = node->inputs()[0];
                Node* b = node->inputs()[1];
                const Tensor& act_a = activations.at(a);
                const Tensor& act_b = activations.at(b);
                // d/da = g / b
                Tensor ga = ops::div(g, act_b);
                accumulate_grad(a, unbroadcast(ga, a->shape()));

                // d/db = -g * a / (b^2)
                Tensor b_sq = ops::mul(act_b, act_b);
                Tensor neg_g = ops::mul(g, Tensor::scalar(-1.0, g.dtype()));
                Tensor neg_g_a = ops::mul(neg_g, act_a);
                Tensor gb = ops::div(neg_g_a, b_sq);
                accumulate_grad(b, unbroadcast(gb, b->shape()));
                break;
            }

            case OpType::ReLU: {
                Node* a = node->inputs()[0];
                const Tensor& act_a = activations.at(a);
                Tensor mask = Tensor::zeros(act_a.shape(), act_a.dtype());
                if (act_a.dtype() == DType::Float32) {
                    Tensor contig_a = act_a.contiguous();
                    const float* pa = contig_a.data<float>();
                    float* pm = mask.data<float>();
                    for (int64_t i = 0; i < act_a.numel(); ++i) {
                        pm[i] = (pa[i] > 0.0f) ? 1.0f : 0.0f;
                    }
                } else {
                    Tensor contig_a = act_a.contiguous();
                    const double* pa = contig_a.data<double>();
                    double* pm = mask.data<double>();
                    for (int64_t i = 0; i < act_a.numel(); ++i) {
                        pm[i] = (pa[i] > 0.0) ? 1.0 : 0.0;
                    }
                }
                Tensor ga = ops::mul(g, mask);
                accumulate_grad(a, ga);
                break;
            }

            case OpType::Sigmoid: {
                Node* a = node->inputs()[0];
                const Tensor& s = activations.at(node); // sigmoid(a)
                Tensor one = Tensor::ones(s.shape(), s.dtype());
                Tensor one_minus_s = ops::sub(one, s);
                Tensor ds = ops::mul(s, one_minus_s); // s * (1 - s)
                Tensor ga = ops::mul(g, ds);
                accumulate_grad(a, ga);
                break;
            }

            case OpType::MatMul: {
                Node* a = node->inputs()[0];
                Node* b = node->inputs()[1];
                const Tensor& act_a = activations.at(a);
                const Tensor& act_b = activations.at(b);

                // C = A @ B
                // dL/dA = dL/dC @ B.T
                // dL/dB = A.T @ dL/dC
                Tensor b_t = ops::transpose(act_b, -2, -1);
                Tensor a_t = ops::transpose(act_a, -2, -1);

                Tensor ga = ops::matmul(g, b_t);
                Tensor gb = ops::matmul(a_t, g);

                accumulate_grad(a, unbroadcast(ga, a->shape()));
                accumulate_grad(b, unbroadcast(gb, b->shape()));
                break;
            }

            case OpType::Transpose: {
                Node* a = node->inputs()[0];
                Tensor ga = ops::transpose(g, node->dim0(), node->dim1());
                accumulate_grad(a, ga);
                break;
            }

            case OpType::ReduceSum: {
                Node* a = node->inputs()[0];
                Tensor g_shaped = g;
                if (!node->keepdims()) {
                    // Reshape g to insert 1s at reduced dimensions so it broadcasts cleanly
                    int64_t rank = static_cast<int64_t>(a->shape().ndim());
                    std::set<int64_t> norm_axes;
                    if (node->axes().empty()) {
                        for (int64_t i = 0; i < rank; ++i) norm_axes.insert(i);
                    } else {
                        for (int64_t ax : node->axes()) {
                            if (ax < 0) ax += rank;
                            norm_axes.insert(ax);
                        }
                    }
                    std::vector<int64_t> target_shape_dims;
                    for (int64_t i = 0; i < rank; ++i) {
                        if (norm_axes.count(i)) {
                            target_shape_dims.push_back(1);
                        } else {
                            target_shape_dims.push_back(a->shape()[static_cast<size_t>(i)]);
                        }
                    }
                    g_shaped = g.reshape(Shape(target_shape_dims));
                }
                Tensor ga = g_shaped.broadcast_to(a->shape()).clone();
                accumulate_grad(a, ga);
                break;
            }

            case OpType::ReduceMean: {
                Node* a = node->inputs()[0];
                int64_t num_reduced = a->shape().numel() / (g.numel() > 0 ? g.numel() : 1);
                if (num_reduced <= 0) num_reduced = 1;
                double factor = 1.0 / static_cast<double>(num_reduced);
                Tensor scaled_g = ops::mul(g, Tensor::scalar(factor, g.dtype()));

                Tensor g_shaped = scaled_g;
                if (!node->keepdims()) {
                    int64_t rank = static_cast<int64_t>(a->shape().ndim());
                    std::set<int64_t> norm_axes;
                    if (node->axes().empty()) {
                        for (int64_t i = 0; i < rank; ++i) norm_axes.insert(i);
                    } else {
                        for (int64_t ax : node->axes()) {
                            if (ax < 0) ax += rank;
                            norm_axes.insert(ax);
                        }
                    }
                    std::vector<int64_t> target_shape_dims;
                    for (int64_t i = 0; i < rank; ++i) {
                        if (norm_axes.count(i)) {
                            target_shape_dims.push_back(1);
                        } else {
                            target_shape_dims.push_back(a->shape()[static_cast<size_t>(i)]);
                        }
                    }
                    g_shaped = scaled_g.reshape(Shape(target_shape_dims));
                }
                Tensor ga = g_shaped.broadcast_to(a->shape()).clone();
                accumulate_grad(a, ga);
                break;
            }

            default:
                break;
        }
    }

    return grads;
}

} // namespace tinytensor
