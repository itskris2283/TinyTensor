#include "tinytensor/utils/memory_tracker.hpp"
#include <fstream>
#include <string>

#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace tinytensor {

ProcessMemoryStats get_process_memory_stats() {
    ProcessMemoryStats stats;

#if defined(__linux__)
    // 1. Current RSS from /proc/self/statm
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        unsigned long size_pages = 0;
        unsigned long resident_pages = 0;
        statm >> size_pages >> resident_pages;
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            stats.current_rss_bytes = resident_pages * static_cast<size_t>(page_size);
        }
    }

    // 2. Peak RSS from getrusage
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss is in kilobytes on Linux
        stats.peak_rss_bytes = static_cast<size_t>(usage.ru_maxrss) * 1024;
    }
    // Kernel hiwater_rss can occasionally lag live mm counters in WSL2; enforce invariant
    stats.peak_rss_bytes = std::max(stats.peak_rss_bytes, stats.current_rss_bytes);
#endif

    return stats;
}

} // namespace tinytensor
