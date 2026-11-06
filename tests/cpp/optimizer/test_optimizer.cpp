#include <gtest/gtest.h>
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/optimizer/constant_folding.hpp"
#include "tinytensor/optimizer/cse.hpp"
#include "tinytensor/optimizer/dead_code_elimination.hpp"
#include "tinytensor/optimizer/fusion.hpp"
#include "tinytensor/optimizer/memory_planner.hpp"
#include "tinytensor/optimizer/optimizer.hpp"

using namespace tinytensor;

TEST(OptimizerTest, ConstantFolding) {
    Graph g;
    Node* c1 = g.create_constant(Tensor::full({2}, 4.0f, DType::Float32));
    Node* c2 = g.create_constant(Tensor::full({2}, 5.0f, DType::Float32));
    Node* sum = g.add(c1, c2);
    g.mark_output(sum);

    OptimizationReport report;
    auto opt_g = ConstantFoldingPass::run(g, &report);
    EXPECT_EQ(report.constant_folding_count, 1u);
    EXPECT_EQ(opt_g->num_nodes(), 1u);
    EXPECT_TRUE(opt_g->outputs()[0]->is_constant());
    EXPECT_FLOAT_EQ((opt_g->outputs()[0]->constant_value().at<float>({0})), 9.0f);
}

TEST(OptimizerTest, ConstantFoldingExtendedOps) {
    Graph g;
    // 1. Sub & Div
    Node* c1 = g.create_constant(Tensor::full({2, 2}, 10.0f, DType::Float32));
    Node* c2 = g.create_constant(Tensor::full({2, 2}, 2.0f, DType::Float32));
    Node* diff = g.sub(c1, c2); // 8.0f
    Node* quot = g.div(diff, c2); // 4.0f

    // 2. ReLU & Sigmoid
    Node* r = g.relu(quot); // 4.0f
    Node* s = g.sigmoid(r); // sigmoid(4.0f)

    // 3. MatMul
    Node* mm = g.matmul(r, c2); // [2, 2] @ [2, 2] -> 4 * 2 + 4 * 2 = 16.0f

    // 4. Transpose
    Node* tr = g.transpose(mm, 0, 1);

    // 5. ReduceSum & ReduceMean
    Node* rsum = g.reduce_sum(tr, {0}, true);
    Node* rmean = g.reduce_mean(tr, {0}, false);

    g.mark_output(s);
    g.mark_output(rsum);
    g.mark_output(rmean);

    OptimizationReport report;
    auto opt_g = ConstantFoldingPass::run(g, &report);
    EXPECT_EQ(report.constant_folding_count, 8u);
    for (Node* out : opt_g->outputs()) {
        EXPECT_TRUE(out->is_constant());
    }
}

TEST(OptimizerTest, DeadCodeElimination) {
    Graph g;
    Node* in = g.create_input({2}, DType::Float32, "X");
    Node* used = g.relu(in);
    Node* dead = g.mul(in, in); // Unused node
    (void)dead;
    g.mark_output(used);

    OptimizationReport report;
    auto opt_g = DeadCodeEliminationPass::run(g, &report);
    EXPECT_EQ(report.dce_eliminated_count, 1u);
    EXPECT_EQ(opt_g->num_nodes(), 2u); // X and used
}

TEST(OptimizerTest, CommonSubexpressionElimination) {
    Graph g;
    Node* a = g.create_input({2}, DType::Float32, "A");
    Node* b = g.create_input({2}, DType::Float32, "B");
    Node* sum1 = g.add(a, b);
    Node* sum2 = g.add(a, b); // Duplicate computation
    Node* out = g.mul(sum1, sum2);
    g.mark_output(out);

    OptimizationReport report;
    auto opt_g = CommonSubexpressionEliminationPass::run(g, &report);
    EXPECT_EQ(report.cse_eliminated_count, 1u);
    // After CSE, out should use the single sum node for both inputs
    EXPECT_EQ(opt_g->outputs()[0]->inputs()[0], opt_g->outputs()[0]->inputs()[1]);
}

TEST(OptimizerTest, ElementwiseFusion) {
    Graph g;
    Node* a = g.create_input({4}, DType::Float32, "A");
    Node* b = g.create_input({4}, DType::Float32, "B");
    Node* sum = g.add(a, b);
    Node* r = g.relu(sum);
    g.mark_output(r);

    OptimizationReport report;
    auto opt_g = ElementwiseFusionPass::run(g, &report);
    EXPECT_EQ(report.fused_groups_count, 1u);
    EXPECT_EQ(opt_g->outputs()[0]->op_type(), OpType::FusedElementwise);
}

TEST(OptimizerTest, MemoryPlannerDiagnostics) {
    Graph g;
    Node* a = g.create_input({1000}, DType::Float32, "A");
    Node* b = g.create_input({1000}, DType::Float32, "B");
    Node* t1 = g.add(a, b);
    Node* t2 = g.relu(t1);
    Node* t3 = g.mul(t2, a);
    g.mark_output(t3);

    OptimizationReport report;
    MemoryPlanner::plan(g, &report);
    EXPECT_GT(report.total_intermediate_allocations, 0u);
    EXPECT_GT(report.peak_planned_memory_bytes, 0u);
}

TEST(OptimizerTest, FullPipelineOptimization) {
    Graph g;
    Node* c1 = g.create_constant(Tensor::full({4}, 2.0f, DType::Float32));
    Node* c2 = g.create_constant(Tensor::full({4}, 3.0f, DType::Float32));
    Node* c_sum = g.add(c1, c2); // 5.0f folded

    Node* in = g.create_input({4}, DType::Float32, "X");
    Node* dead = g.mul(in, in); // DCE eliminated
    (void)dead;

    Node* prod1 = g.mul(in, c_sum);
    Node* prod2 = g.mul(in, c_sum); // CSE eliminated

    Node* sum = g.add(prod1, prod2);
    Node* out = g.relu(sum); // Fused
    g.mark_output(out);

    Optimizer opt;
    OptimizationReport report;
    auto opt_g = opt.optimize(g, &report);

    EXPECT_GT(report.constant_folding_count, 0u);
    EXPECT_GT(report.dce_eliminated_count, 0u);
    EXPECT_GT(report.cse_eliminated_count, 0u);
    EXPECT_GT(report.fused_groups_count, 0u);
    std::string report_str = report.to_string();
    EXPECT_NE(report_str.find("TinyTensor Optimization Report"), std::string::npos);
}
