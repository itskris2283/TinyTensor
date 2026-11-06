#include <gtest/gtest.h>
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/ir/verifier.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

TEST(IRVerifierTest, ValidGraphPasses) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* b = g.create_input({2, 3}, DType::Float32);
    Node* c = g.add(a, b);
    g.mark_output(c);

    EXPECT_NO_THROW(IRVerifier::verify(g));
}

TEST(IRVerifierTest, ArityViolationThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    // Artificially change arity of add to 1 input
    Node* c = g.add(a, a);
    c->set_inputs({a}); // Violation: Add needs 2 inputs

    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, TypeMismatchThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* b = g.create_input({2, 3}, DType::Float64);
    // Trying to add float32 and float64
    EXPECT_THROW(g.add(a, b), TypeError);
}

TEST(IRVerifierTest, ShapeMismatchInMatmulThrows) {
    Graph g;
    Node* a = g.create_input({2, 4}, DType::Float32);
    Node* b = g.create_input({5, 3}, DType::Float32);
    EXPECT_THROW(g.matmul(a, b), ShapeError);
}

TEST(IRVerifierTest, ValidTransposeVerifier) {
    Graph g;
    Node* a = g.create_input({3, 5}, DType::Float32);
    Node* tr = g.transpose(a, 0, 1);
    g.mark_output(tr);

    EXPECT_NO_THROW(IRVerifier::verify(g));
    EXPECT_EQ(tr->shape(), Shape({5, 3}));
}

TEST(IRVerifierTest, ValidReductionVerifier) {
    Graph g;
    Node* a = g.create_input({2, 3, 4}, DType::Float32);
    Node* sum = g.reduce_sum(a, {1}, true);
    g.mark_output(sum);

    EXPECT_NO_THROW(IRVerifier::verify(g));
    EXPECT_EQ(sum->shape(), Shape({2, 1, 4}));
}

TEST(IRVerifierTest, NullOutputThrows) {
    Graph g;
    EXPECT_THROW(g.mark_output(nullptr), IRError);
}

TEST(IRVerifierTest, UnaryArityViolationThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* r = g.relu(a);
    r->set_inputs({a, a}); // ReLU needs 1 input
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, TransposeArityThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* tr = g.transpose(a, 0, 1);
    tr->set_inputs({}); // Transpose needs 1 input
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, MatMulInvalidInputsThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* b = g.create_input({3, 4}, DType::Float32);
    Node* mm = g.matmul(a, b);
    mm->set_inputs({a}); // MatMul needs 2 inputs
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, ReductionArityThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* r = g.reduce_sum(a, {0});
    r->set_inputs({}); // Reduction needs 1 input
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, CycleDetectionThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* b = g.add(a, a);
    b->set_inputs({b, a}); // Cycle: b depends on b
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, NegativeDimensionInputThrows) {
    Graph g;
    EXPECT_THROW(g.create_input({-2, 3}, DType::Float32), ShapeError);
}

TEST(IRVerifierTest, TransposeOutOfBoundsAxesThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* tr = g.transpose(a, 0, 1);
    tr->set_transpose_dims(0, 5); // Axis 5 out of bounds for rank 2
    g.mark_output(tr);
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, TransposeShapeMismatchThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* tr = g.transpose(a, 0, 1);
    tr->set_shape(Shape({2, 3})); // Should be {3, 2}
    g.mark_output(tr);
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, ReductionAxisOutOfBoundsThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* r = g.reduce_sum(a, {0});
    r->set_axes({9}); // Axis 9 out of bounds for rank 2
    g.mark_output(r);
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, ReductionShapeMismatchThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* r = g.reduce_sum(a, {0}, false);
    r->set_shape(Shape({99, 99})); // Should be {3}
    g.mark_output(r);
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, BinaryOpShapeMismatchThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* b = g.create_input({2, 3}, DType::Float32);
    Node* c = g.add(a, b);
    c->set_shape(Shape({5, 5})); // Should be {2, 3}
    g.mark_output(c);
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, UnaryOpShapeMismatchThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* r = g.relu(a);
    r->set_shape(Shape({5, 5})); // Should be {2, 3}
    g.mark_output(r);
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, MatMulBatchShapeMismatchThrows) {
    Graph g;
    Node* a = g.create_input({2, 3}, DType::Float32);
    Node* b = g.create_input({3, 4}, DType::Float32);
    Node* mm = g.matmul(a, b);
    mm->set_shape(Shape({99, 99})); // Should be {2, 4}
    g.mark_output(mm);
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, ConstantShapeMismatchThrows) {
    Graph g;
    Tensor t = Tensor::ones({2, 2}, DType::Float32);
    Node* c = g.create_constant(t);
    c->set_shape(Shape({3, 3})); // Mismatched shape with underlying tensor
    g.mark_output(c);
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, FusedElementwiseEmptyInputsThrows) {
    Graph g;
    Node* a = g.create_input({2, 2}, DType::Float32);
    Node* f = g.add(a, a);
    f->set_op_type(OpType::FusedElementwise);
    f->set_inputs({}); // Empty inputs for fused elementwise
    g.mark_output(f);
    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierTest, DeepTypeMismatchesAndInvariants) {
    // 1. Input node with inputs
    {
        Graph g;
        Node* a = g.create_input({2}, DType::Float32);
        a->set_inputs({a});
        g.mark_output(a);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // 2. Constant node with inputs
    {
        Graph g;
        Node* a = g.create_input({2}, DType::Float32);
        Node* c = g.create_constant(Tensor::full({2}, 1.0f, DType::Float32));
        c->set_inputs({a});
        g.mark_output(c);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // 3. Constant dtype mismatch
    {
        Graph g;
        Node* c = g.create_constant(Tensor::full({2}, 1.0f, DType::Float32));
        c->set_dtype(DType::Float64);
        g.mark_output(c);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // 4. Binary op type mismatch
    {
        Graph g;
        Node* a = g.create_input({2}, DType::Float32);
        Node* b = g.create_input({2}, DType::Float32);
        Node* add = g.add(a, b);
        add->set_dtype(DType::Float64);
        g.mark_output(add);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // 5. Unary op type mismatch
    {
        Graph g;
        Node* a = g.create_input({2}, DType::Float32);
        Node* r = g.relu(a);
        r->set_dtype(DType::Float64);
        g.mark_output(r);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // 6. MatMul dtype mismatch and rank < 2
    {
        Graph g;
        Node* a = g.create_input({2, 2}, DType::Float32);
        Node* b = g.create_input({2, 2}, DType::Float32);
        Node* mm = g.matmul(a, b);
        mm->set_dtype(DType::Float64);
        g.mark_output(mm);
        EXPECT_THROW(IRVerifier::verify(g), IRError);

        Node* v1 = g.create_input({2}, DType::Float32);
        Node* v2 = g.create_input({2}, DType::Float32);
        mm->set_dtype(DType::Float32);
        mm->set_inputs({v1, v2}); // Rank 1 operands
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // 7. Transpose & Reduction type mismatches
    {
        Graph g;
        Node* a = g.create_input({2, 3}, DType::Float32);
        Node* tr = g.transpose(a, 0, 1);
        tr->set_dtype(DType::Float64);
        g.mark_output(tr);
        EXPECT_THROW(IRVerifier::verify(g), IRError);

        tr->set_dtype(DType::Float32);
        Node* rsum = g.reduce_sum(a, {0});
        rsum->set_dtype(DType::Float64);
        g.mark_output(rsum);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // 8. Fused elementwise type mismatch
    {
        Graph g;
        Node* a = g.create_input({2}, DType::Float32);
        Node* f = g.add(a, a);
        f->set_op_type(OpType::FusedElementwise);
        f->set_dtype(DType::Float64);
        g.mark_output(f);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }
}
