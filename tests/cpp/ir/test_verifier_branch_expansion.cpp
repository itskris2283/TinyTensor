#include <gtest/gtest.h>
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/ir/verifier.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

TEST(IRVerifierBranchExpansionTest, NullInputOperandThrows) {
    Graph g;
    Node* a = g.create_input({2, 2}, DType::Float32, "a");
    Node* add = g.add(a, a);
    add->set_inputs({nullptr, a});
    g.mark_output(add);

    EXPECT_THROW(IRVerifier::verify(g), IRError);
}

TEST(IRVerifierBranchExpansionTest, BinaryOpArityAndTypeMismatches) {
    // Binary op with 3 inputs
    {
        Graph g;
        Node* a = g.create_input({2, 2}, DType::Float32, "a");
        Node* b = g.create_input({2, 2}, DType::Float32, "b");
        Node* add = g.add(a, b);
        add->set_inputs({a, b, a});
        g.mark_output(add);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // Binary op with mismatched operand dtypes
    {
        Graph g;
        Node* a = g.create_input({2, 2}, DType::Float32, "a");
        Node* b = g.create_input({2, 2}, DType::Float64, "b");
        Node* add = g.create_input({2, 2}, DType::Float32, "dummy");
        add->set_op_type(OpType::Add);
        add->set_inputs({a, b});
        g.mark_output(add);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // Binary op with node dtype mismatch from operands
    {
        Graph g;
        Node* a = g.create_input({2, 2}, DType::Float32, "a");
        Node* b = g.create_input({2, 2}, DType::Float32, "b");
        Node* add = g.add(a, b);
        add->set_dtype(DType::Float64);
        g.mark_output(add);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // Binary op with incompatible broadcast shapes throws BroadcastError
    {
        Graph g;
        Node* a = g.create_input({2, 3}, DType::Float32, "a");
        Node* b = g.create_input({4, 5}, DType::Float32, "b");
        Node* add = g.create_input({2, 3}, DType::Float32, "dummy");
        add->set_op_type(OpType::Mul);
        add->set_inputs({a, b});
        g.mark_output(add);
        EXPECT_THROW(IRVerifier::verify(g), BroadcastError);
    }
}

TEST(IRVerifierBranchExpansionTest, UnaryOpArityAndTypeMismatches) {
    // Unary op with 0 inputs
    {
        Graph g;
        Node* a = g.create_input({2, 2}, DType::Float32, "a");
        Node* r = g.relu(a);
        r->set_inputs({});
        g.mark_output(r);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // Unary op with dtype mismatch
    {
        Graph g;
        Node* a = g.create_input({2, 2}, DType::Float32, "a");
        Node* s = g.sigmoid(a);
        s->set_dtype(DType::Float64);
        g.mark_output(s);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }
}

TEST(IRVerifierBranchExpansionTest, MatMulArityInnerDimAndRankMismatches) {
    // MatMul with 1 input
    {
        Graph g;
        Node* a = g.create_input({2, 2}, DType::Float32, "a");
        Node* mm = g.matmul(a, a);
        mm->set_inputs({a});
        g.mark_output(mm);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // MatMul operand rank < 2
    {
        Graph g;
        Node* a = g.create_input({4}, DType::Float32, "a");
        Node* b = g.create_input({4, 2}, DType::Float32, "b");
        Node* mm = g.create_input({2}, DType::Float32, "dummy");
        mm->set_op_type(OpType::MatMul);
        mm->set_inputs({a, b});
        g.mark_output(mm);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // MatMul inner dimension mismatch: k_a != k_b
    {
        Graph g;
        Node* a = g.create_input({2, 3}, DType::Float32, "a");
        Node* b = g.create_input({4, 5}, DType::Float32, "b");
        Node* mm = g.create_input({2, 5}, DType::Float32, "dummy");
        mm->set_op_type(OpType::MatMul);
        mm->set_inputs({a, b});
        g.mark_output(mm);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // MatMul batch shape mismatch throws BroadcastError
    {
        Graph g;
        Node* a = g.create_input({2, 3, 4}, DType::Float32, "a");
        Node* b = g.create_input({5, 4, 6}, DType::Float32, "b");
        Node* mm = g.create_input({2, 3, 6}, DType::Float32, "dummy");
        mm->set_op_type(OpType::MatMul);
        mm->set_inputs({a, b});
        g.mark_output(mm);
        EXPECT_THROW(IRVerifier::verify(g), BroadcastError);
    }
}

TEST(IRVerifierBranchExpansionTest, TransposeAxesBoundariesAndMismatches) {
    // Transpose d0 < 0
    {
        Graph g;
        Node* a = g.create_input({2, 3}, DType::Float32, "a");
        Node* tr = g.transpose(a, 0, 1);
        tr->set_transpose_dims(-1, 1);
        g.mark_output(tr);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // Transpose d1 < 0
    {
        Graph g;
        Node* a = g.create_input({2, 3}, DType::Float32, "a");
        Node* tr = g.transpose(a, 0, 1);
        tr->set_transpose_dims(0, -1);
        g.mark_output(tr);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // Transpose d0 >= rank
    {
        Graph g;
        Node* a = g.create_input({2, 3}, DType::Float32, "a");
        Node* tr = g.transpose(a, 0, 1);
        tr->set_transpose_dims(2, 1);
        g.mark_output(tr);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // Transpose d1 >= rank
    {
        Graph g;
        Node* a = g.create_input({2, 3}, DType::Float32, "a");
        Node* tr = g.transpose(a, 0, 1);
        tr->set_transpose_dims(0, 2);
        g.mark_output(tr);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // Transpose 3D rank valid verification
    {
        Graph g;
        Node* a = g.create_input({2, 3, 4}, DType::Float32, "a");
        Node* tr = g.transpose(a, 0, 2);
        g.mark_output(tr);
        EXPECT_NO_THROW(IRVerifier::verify(g));
        EXPECT_EQ(tr->shape(), Shape({4, 3, 2}));
    }
}

TEST(IRVerifierBranchExpansionTest, ReductionAxesNegativeAndOutOfBounds) {
    // Negative axis normalized within bounds
    {
        Graph g;
        Node* a = g.create_input({2, 3, 4}, DType::Float32, "a");
        Node* r = g.reduce_sum(a, {-1}, false);
        g.mark_output(r);
        EXPECT_NO_THROW(IRVerifier::verify(g));
        EXPECT_EQ(r->shape(), Shape({2, 3}));
    }

    // Negative axis too negative: ax + rank < 0 (set after creation to test verifier)
    {
        Graph g;
        Node* a = g.create_input({2, 3, 4}, DType::Float32, "a");
        Node* r = g.reduce_sum(a, {0}, false);
        r->set_axes({-4});
        g.mark_output(r);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // Positive axis >= rank (set after creation to test verifier)
    {
        Graph g;
        Node* a = g.create_input({2, 3, 4}, DType::Float32, "a");
        Node* r = g.reduce_sum(a, {0}, false);
        r->set_axes({3});
        g.mark_output(r);
        EXPECT_THROW(IRVerifier::verify(g), IRError);
    }

    // Empty axes (reduces all dimensions) with keepdims=true
    {
        Graph g;
        Node* a = g.create_input({2, 3}, DType::Float32, "a");
        Node* r = g.reduce_mean(a, {}, true);
        g.mark_output(r);
        EXPECT_NO_THROW(IRVerifier::verify(g));
        EXPECT_EQ(r->shape(), Shape({1, 1}));
    }

    // Empty axes with keepdims=false (becomes scalar)
    {
        Graph g;
        Node* a = g.create_input({2, 3}, DType::Float32, "a");
        Node* r = g.reduce_mean(a, {}, false);
        g.mark_output(r);
        EXPECT_NO_THROW(IRVerifier::verify(g));
        EXPECT_EQ(r->shape(), Shape({}));
    }
}

TEST(IRVerifierBranchExpansionTest, GraphOutputValidation) {
    // Output not in graph
    Graph g1;
    Node* a = g1.create_input({2, 2}, DType::Float32, "a");

    Graph g2;
    g2.mark_output(a); // a belongs to g1, not g2!
    EXPECT_THROW(IRVerifier::verify(g2), IRError);
}
