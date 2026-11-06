#pragma once

#include <atomic>
#include <mutex>
#include <string>

namespace tinytensor {

class FileLockGuard {
public:
    explicit FileLockGuard(const std::string& lock_file_path, bool shared = false);
    ~FileLockGuard();

    FileLockGuard(const FileLockGuard&) = delete;
    FileLockGuard& operator=(const FileLockGuard&) = delete;

    FileLockGuard(FileLockGuard&& other) noexcept;
    FileLockGuard& operator=(FileLockGuard&& other) noexcept;

    bool is_locked() const noexcept { return fd_ >= 0; }
    void unlock();

private:
    int fd_{-1};
    std::string lock_file_path_;
};

class KernelCache {
public:
    explicit KernelCache(std::string cache_dir = ".tinytensor_cache");

    // Computes SHA-256 hash of IR, shapes, types, compiler flags, and TinyTensor version
    static std::string compute_key(const std::string& canonical_ir,
                                   const std::string& metadata,
                                   const std::string& compiler_info);

    bool get(const std::string& key, std::string& out_so_path, bool count_miss = true);
    bool get_source(const std::string& key, std::string& out_source) const;
    void put(const std::string& key, const std::string& temp_so_path, const std::string& source_code = "");

    FileLockGuard acquire_compile_lock(const std::string& key) const;

    size_t hits() const noexcept { return hits_.load(); }
    size_t misses() const noexcept { return misses_.load(); }
    double total_compile_time_ms() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return compile_time_ms_;
    }
    void add_compile_time(double ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        compile_time_ms_ += ms;
    }

    const std::string& cache_dir() const noexcept { return cache_dir_; }
    void clear();

private:
    std::string cache_dir_;
    mutable std::mutex mutex_;
    std::atomic<size_t> hits_{0};
    std::atomic<size_t> misses_{0};
    double compile_time_ms_{0.0};
};

} // namespace tinytensor
