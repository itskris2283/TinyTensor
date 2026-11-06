#pragma once

#include <memory>
#include <string>
#include <vector>
#include "tinytensor/core/dtype.hpp"
#include "tinytensor/core/shape.hpp"
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/ir/operation.hpp"

namespace tinytensor {

class Node {
public:
    Node(size_t id, OpType op_type, Shape shape, DType dtype,
         std::vector<Node*> inputs = {}, std::string name = "")
        : id_(id),
          op_type_(op_type),
          shape_(std::move(shape)),
          dtype_(dtype),
          inputs_(std::move(inputs)),
          name_(std::move(name)) {}

    size_t id() const noexcept { return id_; }
    OpType op_type() const noexcept { return op_type_; }
    void set_op_type(OpType op) noexcept { op_type_ = op; }
    const Shape& shape() const noexcept { return shape_; }
    void set_shape(Shape s) { shape_ = std::move(s); }
    DType dtype() const noexcept { return dtype_; }
    void set_dtype(DType d) { dtype_ = d; }
    const std::string& name() const noexcept { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

    const std::vector<Node*>& inputs() const noexcept { return inputs_; }
    void set_inputs(std::vector<Node*> inputs) { inputs_ = std::move(inputs); }
    void replace_input(size_t idx, Node* new_input) {
        if (idx < inputs_.size()) {
            inputs_[idx] = new_input;
        }
    }

    const std::vector<Node*>& users() const noexcept { return users_; }
    void add_user(Node* user) { users_.push_back(user); }
    void remove_user(Node* user) {
        users_.erase(std::remove(users_.begin(), users_.end(), user), users_.end());
    }

    // Attributes
    const Tensor& constant_value() const noexcept { return constant_val_; }
    void set_constant_value(Tensor t) { constant_val_ = std::move(t); }

    const std::vector<int64_t>& axes() const noexcept { return axes_; }
    void set_axes(std::vector<int64_t> axes) { axes_ = std::move(axes); }

    bool keepdims() const noexcept { return keepdims_; }
    void set_keepdims(bool kd) noexcept { keepdims_ = kd; }

    int64_t dim0() const noexcept { return dim0_; }
    int64_t dim1() const noexcept { return dim1_; }
    void set_transpose_dims(int64_t d0, int64_t d1) noexcept {
        dim0_ = d0;
        dim1_ = d1;
    }

    // Fusion attributes
    const std::string& fusion_expression() const noexcept { return fusion_expr_; }
    void set_fusion_expression(std::string expr) { fusion_expr_ = std::move(expr); }

    bool is_input() const noexcept { return op_type_ == OpType::Input; }
    bool is_constant() const noexcept { return op_type_ == OpType::Constant; }
    bool is_elementwise() const noexcept {
        return op_type_ == OpType::Add || op_type_ == OpType::Sub ||
               op_type_ == OpType::Mul || op_type_ == OpType::Div ||
               op_type_ == OpType::ReLU || op_type_ == OpType::Sigmoid;
    }

private:
    size_t id_;
    OpType op_type_;
    Shape shape_;
    DType dtype_;
    std::vector<Node*> inputs_;
    std::vector<Node*> users_;
    std::string name_;

    // Specific attributes
    Tensor constant_val_;
    std::vector<int64_t> axes_;
    bool keepdims_{false};
    int64_t dim0_{0};
    int64_t dim1_{0};
    std::string fusion_expr_;
};

} // namespace tinytensor
