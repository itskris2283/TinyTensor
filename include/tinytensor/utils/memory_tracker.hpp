#pragma once

#include <cstddef>

namespace tinytensor {

struct ProcessMemoryStats {
    size_t current_rss_bytes{0};
    size_t peak_rss_bytes{0};
};

ProcessMemoryStats get_process_memory_stats();

} // namespace tinytensor
