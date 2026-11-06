#include <gtest/gtest.h>
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

TEST(GraphTest, NodeCreationAndTypes) {
    Graph g;
    Node* in1 = g.create_input({2, 3}, DType::Float32, "X");
    Node* in2 = g.create_input({2, 3}, DType::Float32, "Y");
    EXPECT_EQ(in1->name(), "X");
    EXPECT_EQ(in1->shape(), Shape({2, 3}));
    EXPECT_EQ(in1->dtype(), DType::Float32);
    EXPECT_TRUE(in1->is_input());

    Node* sum = g.add(in1, in2);
    EXPECT_EQ(sum->op_type(), OpType::Add);
    EXPECT_EQ(sum->inputs().size(), 2u);
    EXPECT_EQ(sum->inputs()[0], in1);
    EXPECT_EQ(sum->inputs()[1], in2);
    EXPECT_EQ(in1->users().size(), 1u);
    EXPECT_EQ(in1->users()[0], sum);
}

TEST(GraphTest, ConstantNode) {
    Graph g;
    Tensor val = Tensor::full({2, 2}, 3.5f, DType::Float32);
    Node* c = g.create_constant(val, "C");
    EXPECT_TRUE(c->is_constant());
    EXPECT_FLOAT_EQ((c->constant_value().at<float>({0, 0})), 3.5f);
}

TEST(GraphTest, TopologicalSort) {
    Graph g;
    Node* a = g.create_input({4}, DType::Float32, "A");
    Node* b = g.create_input({4}, DType::Float32, "B");
    Node* c = g.mul(a, b);
    Node* d = g.relu(c);
    g.mark_output(d);

    std::vector<Node*> topo = g.topological_sort();
    EXPECT_EQ(topo.size(), 4u);
    EXPECT_EQ(topo[3], d);
}

TEST(GraphTest, CycleDetection) {
    Graph g;
    Node* a = g.create_input({4}, DType::Float32, "A");
    Node* b = g.relu(a);
    // Artificially introduce cycle: make a depend on b
    a->set_inputs({b});
    g.mark_output(b);

    EXPECT_THROW(g.topological_sort(), IRError);
}

TEST(GraphTest, GraphClone) {
    Graph g;
    Node* x = g.create_input({2, 2}, DType::Float32, "X");
    Node* y = g.create_input({2, 2}, DType::Float32, "Y");
    Node* z = g.add(x, y);
    g.mark_output(z);

    auto g2 = g.clone();
    EXPECT_EQ(g2->num_nodes(), 3u);
    EXPECT_EQ(g2->inputs().size(), 2u);
    EXPECT_EQ(g2->outputs().size(), 1u);
}

TEST(GraphTest, MultipleOutputs) {
    Graph g;
    Node* a = g.create_input({2, 2}, DType::Float32, "A");
    Node* b = g.create_input({2, 2}, DType::Float32, "B");
    Node* sum = g.add(a, b);
    Node* prod = g.mul(a, b);
    g.mark_output(sum);
    g.mark_output(prod);

    EXPECT_EQ(g.outputs().size(), 2u);
    auto topo = g.topological_sort();
    EXPECT_GE(topo.size(), 4u);
}

TEST(GraphTest, UnaryOperationChains) {
    Graph g;
    Node* x = g.create_input({4}, DType::Float32, "X");
    Node* s = g.sigmoid(x);
    Node* r1 = g.relu(s);
    Node* r2 = g.relu(r1);
    g.mark_output(r2);

    auto topo = g.topological_sort();
    EXPECT_EQ(topo.size(), 4u);
    EXPECT_EQ(topo.back(), r2);
}

TEST(GraphTest, InputNameUniqueness) {
    Graph g;
    Node* in1 = g.create_input({2}, DType::Float32);
    Node* in2 = g.create_input({2}, DType::Float32);
    EXPECT_NE(in1->name(), in2->name());
}

TEST(GraphTest, NodeUserTracking) {
    Graph g;
    Node* x = g.create_input({2}, DType::Float32, "X");
    Node* y1 = g.relu(x);
    Node* y2 = g.sigmoid(x);
    EXPECT_EQ(x->users().size(), 2u);
    EXPECT_EQ(x->users()[0], y1);
    EXPECT_EQ(x->users()[1], y2);
}
