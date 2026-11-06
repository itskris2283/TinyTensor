#include <gtest/gtest.h>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/optimizer/memory_planner.hpp"
#include "tinytensor/utils/memory_tracker.hpp"

using namespace tinytensor;

namespace {

// Helper to check the fundamental memory safety invariant:
// If two nodes have overlapping lifetime intervals, they MUST NOT share the same buffer_id.
void verify_lifetime_overlap_invariant(const MemoryPlanResult& plan) {
    for (size_t i = 0; i < plan.lifetimes.size(); ++i) {
        for (size_t j = i + 1; j < plan.lifetimes.size(); ++j) {
            const auto& a = plan.lifetimes[i];
            const auto& b = plan.lifetimes[j];

            // Check if intervals [first_step, last_step] overlap
            bool overlap = (std::max(a.first_step, b.first_step) <= std::min(a.last_step, b.last_step));
            if (overlap) {
                EXPECT_NE(a.buffer_id, b.buffer_id)
                    << "Memory safety violation! Node " << a.node->id() << " [" << a.first_step << ", " << a.last_step
                    << "] and Node " << b.node->id() << " [" << b.first_step << ", " << b.last_step
                    << "] overlap but both were assigned buffer_id " << a.buffer_id;
            }
        }
    }
}

} // anonymous namespace

TEST(AdversarialMemoryPlannerTest, DeepMLPAlternatingReuse) {
    // 12-layer deep linear/activation chain
    Graph g;
    Node* cur = g.create_input({128, 128}, DType::Float32, "Input");

    std::vector<Node*> layers;
    for (int i = 0; i < 12; ++i) {
        cur = g.relu(cur);
        layers.push_back(cur);
    }
    g.mark_output(cur);

    MemoryPlanResult plan = MemoryPlanner::compute_plan(g);

    verify_lifetime_overlap_invariant(plan);

    // In a strictly linear chain, at most 2 buffers need to be active simultaneously:
    // the input buffer to the layer and the output buffer.
    EXPECT_LE(plan.num_unique_buffers, 2u);
    EXPECT_GE(plan.reused_buffers_count, 9u);

    // Theoretical peak live bytes must be <= planned peak memory bytes
    EXPECT_LE(plan.theoretical_peak_live_bytes, plan.peak_planned_memory_bytes);

    // Total naive bytes is 11 intermediate buffers * size
    // Peak planned memory is 2 buffers * size -> ~81.8% savings
    EXPECT_LT(plan.peak_planned_memory_bytes, plan.total_alloc_bytes / 2);
}

TEST(AdversarialMemoryPlannerTest, BranchingAndReconvergentNoAliasing) {
    // Diamond DAG: Input -> 3 concurrent branches -> recombined into sum
    Graph g;
    Node* in = g.create_input({64, 64}, DType::Float32, "X");

    Node* b1 = g.relu(in);
    Node* b2 = g.sigmoid(in);
    Node* b3 = g.mul(in, in);

    Node* sum12 = g.add(b1, b2);
    Node* sum123 = g.add(sum12, b3);
    g.mark_output(sum123);

    MemoryPlanResult plan = MemoryPlanner::compute_plan(g);

    verify_lifetime_overlap_invariant(plan);

    // b1, b2, b3 are concurrently live before sum123 consumes them
    size_t id_b1 = plan.node_to_buffer_id.at(b1);
    size_t id_b2 = plan.node_to_buffer_id.at(b2);
    size_t id_b3 = plan.node_to_buffer_id.at(b3);

    EXPECT_NE(id_b1, id_b2);
    EXPECT_NE(id_b2, id_b3);
    EXPECT_NE(id_b1, id_b3);
}

TEST(AdversarialMemoryPlannerTest, MultipleOutputsRetention) {
    // Graph with 3 outputs produced at early, mid, and late steps
    Graph g;
    Node* in = g.create_input({32, 32}, DType::Float32, "X");

    Node* out_early = g.relu(in);
    g.mark_output(out_early); // Output 1

    Node* mid = g.sigmoid(out_early);
    Node* out_mid = g.mul(mid, in);
    g.mark_output(out_mid); // Output 2

    Node* late = g.add(out_mid, out_early);
    Node* out_late = g.relu(late);
    g.mark_output(out_late); // Output 3

    MemoryPlanResult plan = MemoryPlanner::compute_plan(g);

    verify_lifetime_overlap_invariant(plan);

    // Verify all 3 outputs have last_step == topo.size()
    for (const auto& lt : plan.lifetimes) {
        if (lt.node == out_early || lt.node == out_mid || lt.node == out_late) {
            EXPECT_EQ(lt.last_step, g.num_nodes());
        }
    }
}

TEST(AdversarialMemoryPlannerTest, SharedIntermediateStaggeredUsers) {
    // Node A is consumed immediately by B, but also needed by Z at the very end
    Graph g;
    Node* in = g.create_input({16, 16}, DType::Float32, "X");

    Node* shared_a = g.relu(in);
    Node* cur = g.add(shared_a, in); // Consumes shared_a immediately

    // Intervening intermediate operations
    for (int i = 0; i < 5; ++i) {
        cur = g.sigmoid(cur);
    }

    // Node Z uses shared_a at the very end
    Node* z = g.add(cur, shared_a);
    g.mark_output(z);

    MemoryPlanResult plan = MemoryPlanner::compute_plan(g);

    verify_lifetime_overlap_invariant(plan);

    size_t shared_buf_id = plan.node_to_buffer_id.at(shared_a);

    // Ensure none of the intervening nodes reused shared_a's buffer
    for (const auto& lt : plan.lifetimes) {
        if (lt.node != shared_a && lt.first_step < plan.lifetimes.back().first_step) {
            // Overlapping intervals with shared_a must not have shared_buf_id
            bool overlaps_with_a = (lt.first_step <= plan.lifetimes[0].last_step);
            if (overlaps_with_a) {
                EXPECT_NE(lt.buffer_id, shared_buf_id);
            }
        }
    }
}

TEST(AdversarialMemoryPlannerTest, DisjointSubgraphsReuse) {
    // Two independent subgraphs executed sequentially and combined
    Graph g;
    Node* in1 = g.create_input({32, 32}, DType::Float32, "X1");
    Node* in2 = g.create_input({32, 32}, DType::Float32, "X2");

    // Subgraph 1
    Node* a1 = g.relu(in1);
    Node* a2 = g.sigmoid(a1);

    // Subgraph 2
    Node* b1 = g.relu(in2);
    Node* b2 = g.sigmoid(b1);

    Node* final_out = g.add(a2, b2);
    g.mark_output(final_out);

    MemoryPlanResult plan = MemoryPlanner::compute_plan(g);

    verify_lifetime_overlap_invariant(plan);

    // Buffer reuse should occur across the subgraphs
    EXPECT_GT(plan.reused_buffers_count, 0u);
    EXPECT_LT(plan.peak_planned_memory_bytes, plan.total_alloc_bytes);
}

TEST(AdversarialMemoryPlannerTest, ProcessMemoryAndRSSInstrumentation) {
    ProcessMemoryStats stats_before = get_process_memory_stats();

    // Allocate a sizable tensor (16 MB)
    std::vector<Tensor> tensors;
    for (int i = 0; i < 4; ++i) {
        tensors.push_back(Tensor::full({1024, 1024}, 1.0f, DType::Float32));
    }

    ProcessMemoryStats stats_after = get_process_memory_stats();

#if defined(__linux__)
    EXPECT_GT(stats_after.current_rss_bytes, 0u);
    EXPECT_GT(stats_after.peak_rss_bytes, 0u);
    EXPECT_GE(stats_after.peak_rss_bytes, stats_after.current_rss_bytes);
    EXPECT_GE(stats_after.peak_rss_bytes, stats_before.peak_rss_bytes);
#endif
}
