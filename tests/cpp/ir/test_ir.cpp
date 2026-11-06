#include <gtest/gtest.h>
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/ir/printer.hpp"

using namespace tinytensor;

TEST(IRTest, SerializationOutput) {
    Graph g;
    Node* in0 = g.create_input({64, 128}, DType::Float32, "input_0");
    Node* in1 = g.create_input({128, 32}, DType::Float32, "input_1");
    Node* mm = g.matmul(in0, in1);
    Node* r = g.relu(mm);
    Node* loss = g.reduce_mean(r);
    g.mark_output(loss);

    std::string ir = IRPrinter::to_string(g);
    EXPECT_NE(ir.find("%0 = input<float32>[64, 128]"), std::string::npos);
    EXPECT_NE(ir.find("%1 = input<float32>[128, 32]"), std::string::npos);
    EXPECT_NE(ir.find("matmul %0, %1"), std::string::npos);
    EXPECT_NE(ir.find("relu %2"), std::string::npos);
    EXPECT_NE(ir.find("reduce_mean %3"), std::string::npos);
    EXPECT_NE(ir.find("return %4"), std::string::npos);
}

TEST(IRTest, DotGraphExport) {
    Graph g;
    Node* a = g.create_input({2, 2}, DType::Float32, "A");
    Node* b = g.create_input({2, 2}, DType::Float32, "B");
    Node* c = g.add(a, b);
    g.mark_output(c);

    std::string dot = IRPrinter::to_dot(g);
    EXPECT_NE(dot.find("digraph TinyTensorGraph"), std::string::npos);
    EXPECT_NE(dot.find("node_"), std::string::npos);
    EXPECT_NE(dot.find("->"), std::string::npos);
}
