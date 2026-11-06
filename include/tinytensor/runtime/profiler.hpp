#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinytensor {

struct NodeProfile {
    size_t node_id;
    std::string op_name;
    double duration_us{0.0};
    size_t output_bytes{0};
};

class Profiler {
public:
    Profiler() = default;

    void start_timer() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void stop_timer() {
        auto end = std::chrono::high_resolution_clock::now();
        total_time_us_ = std::chrono::duration<double, std::micro>(end - start_time_).count();
    }

    void record_node(size_t node_id, const std::string& op_name,
                     double duration_us, size_t output_bytes) {
        node_profiles_.push_back({node_id, op_name, duration_us, output_bytes});
    }

    double total_time_us() const noexcept { return total_time_us_; }
    const std::vector<NodeProfile>& node_profiles() const noexcept { return node_profiles_; }

    void clear() {
        node_profiles_.clear();
        total_time_us_ = 0.0;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    double total_time_us_{0.0};
    std::vector<NodeProfile> node_profiles_;
};

} // namespace tinytensor
