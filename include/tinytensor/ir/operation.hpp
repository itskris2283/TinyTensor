#pragma once

#include <string>

namespace tinytensor {

enum class OpType {
    Input,
    Constant,
    Add,
    Sub,
    Mul,
    Div,
    ReLU,
    Sigmoid,
    MatMul,
    Transpose,
    ReduceSum,
    ReduceMean,
    Output,
    FusedElementwise
};

inline std::string op_type_to_string(OpType op) {
    switch (op) {
        case OpType::Input: return "input";
        case OpType::Constant: return "constant";
        case OpType::Add: return "add";
        case OpType::Sub: return "sub";
        case OpType::Mul: return "mul";
        case OpType::Div: return "div";
        case OpType::ReLU: return "relu";
        case OpType::Sigmoid: return "sigmoid";
        case OpType::MatMul: return "matmul";
        case OpType::Transpose: return "transpose";
        case OpType::ReduceSum: return "reduce_sum";
        case OpType::ReduceMean: return "reduce_mean";
        case OpType::Output: return "output";
        case OpType::FusedElementwise: return "fused_elementwise";
        default: return "unknown";
    }
}

} // namespace tinytensor
