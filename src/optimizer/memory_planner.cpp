#include "tinytensor/optimizer/memory_planner.hpp"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace tinytensor {

MemoryPlanResult MemoryPlanner::compute_plan(const Graph& graph, OptimizationReport* report) {
    MemoryPlanResult res;
    std::vector<Node*> topo = graph.topological_sort();
    if (topo.empty()) return res;

    std::unordered_map<const Node*, size_t> node_step;
    for (size_t step = 0; step < topo.size(); ++step) {
        node_step[topo[step]] = step;
    }

    std::unordered_set<const Node*> output_set(graph.outputs().begin(), graph.outputs().end());
    size_t total_alloc_bytes = 0;

    for (size_t step = 0; step < topo.size(); ++step) {
        const Node* node = topo[step];
        if (node->is_input() || node->is_constant()) {
            continue;
        }

        size_t first_step = step;
        size_t last_step = step;

        if (output_set.count(node)) {
            last_step = topo.size(); // Lives until end of program
        }

        for (const Node* user : node->users()) {
            if (node_step.count(user)) {
                last_step = std::max(last_step, node_step[user]);
            }
        }

        size_t bytes = static_cast<size_t>(node->shape().numel()) * dtype_size(node->dtype());
        total_alloc_bytes += bytes;
        res.lifetimes.push_back({node, first_step, last_step, bytes, 0});
    }

    // Compute exact theoretical peak live tensor bytes across all execution steps
    size_t theoretical_peak = 0;
    for (size_t s = 0; s <= topo.size(); ++s) {
        size_t live_at_s = 0;
        for (const auto& lt : res.lifetimes) {
            if (lt.first_step <= s && s <= lt.last_step) {
                live_at_s += lt.size_bytes;
            }
        }
        theoretical_peak = std::max(theoretical_peak, live_at_s);
    }
    res.theoretical_peak_live_bytes = theoretical_peak;
    res.total_alloc_bytes = total_alloc_bytes;

    // Sort by first step
    std::sort(res.lifetimes.begin(), res.lifetimes.end(), [](const auto& a, const auto& b) {
        return a.first_step < b.first_step;
    });

    // Greedy buffer allocation
    std::vector<BufferPoolSlotInfo>& buffer_pool = res.buffer_pool;
    size_t reused_count = 0;

    for (auto& lt : res.lifetimes) {
        bool found = false;
        for (auto& slot : buffer_pool) {
            if (slot.last_free_step < lt.first_step) {
                // Reuse this slot
                slot.last_free_step = lt.last_step;
                slot.capacity_bytes = std::max(slot.capacity_bytes, lt.size_bytes);
                lt.buffer_id = slot.id;
                res.node_to_buffer_id[lt.node] = slot.id;
                found = true;
                reused_count++;
                break;
            }
        }

        if (!found) {
            size_t new_id = buffer_pool.size();
            buffer_pool.push_back({new_id, lt.last_step, lt.size_bytes});
            lt.buffer_id = new_id;
            res.node_to_buffer_id[lt.node] = new_id;
        }
    }

    size_t peak_memory_bytes = 0;
    for (const auto& slot : buffer_pool) {
        peak_memory_bytes += slot.capacity_bytes;
    }

    res.peak_planned_memory_bytes = peak_memory_bytes;
    res.reused_buffers_count = reused_count;
    res.num_unique_buffers = buffer_pool.size();

    if (report) {
        report->total_intermediate_allocations = res.lifetimes.size();
        report->peak_planned_memory_bytes = peak_memory_bytes;
        report->reused_buffers_count = reused_count;
        if (total_alloc_bytes > peak_memory_bytes) {
            report->memory_savings_bytes = total_alloc_bytes - peak_memory_bytes;
        } else {
            report->memory_savings_bytes = 0;
        }
    }

    return res;
}

void MemoryPlanner::plan(const Graph& graph, OptimizationReport* report) {
    compute_plan(graph, report);
}

} // namespace tinytensor
