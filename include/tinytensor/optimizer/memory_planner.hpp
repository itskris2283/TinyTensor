#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/optimizer/optimizer.hpp"

namespace tinytensor {

struct BufferLifetime {
    const Node* node;
    size_t first_step;
    size_t last_step;
    size_t size_bytes;
    size_t buffer_id{0};
};

struct BufferPoolSlotInfo {
    size_t id;
    size_t last_free_step;
    size_t capacity_bytes;
};

struct MemoryPlanResult {
    std::vector<BufferLifetime> lifetimes;
    std::unordered_map<const Node*, size_t> node_to_buffer_id;
    std::vector<BufferPoolSlotInfo> buffer_pool;
    size_t total_alloc_bytes{0};
    size_t peak_planned_memory_bytes{0};
    size_t theoretical_peak_live_bytes{0};
    size_t reused_buffers_count{0};
    size_t num_unique_buffers{0};
};

class MemoryPlanner {
public:
    static MemoryPlanResult compute_plan(const Graph& graph, OptimizationReport* report = nullptr);
    static void plan(const Graph& graph, OptimizationReport* report = nullptr);
};

} // namespace tinytensor
