#include "tinytensor/codegen/kernel_cache.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/file.h>
#include <unistd.h>
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

namespace {

// Self-contained SHA-256 implementation (FIPS 180-4)
class SHA256 {
public:
    SHA256() { reset(); }

    void update(const uint8_t* data, size_t length) {
        for (size_t i = 0; i < length; ++i) {
            buffer_[buffer_len_++] = data[i];
            if (buffer_len_ == 64) {
                transform();
                bitlen_ += 512;
                buffer_len_ = 0;
            }
        }
    }

    void update(const std::string& str) {
        update(reinterpret_cast<const uint8_t*>(str.data()), str.size());
    }

    std::string finalize() {
        uint64_t total_bits = bitlen_ + buffer_len_ * 8;
        buffer_[buffer_len_++] = 0x80;
        if (buffer_len_ > 56) {
            while (buffer_len_ < 64) buffer_[buffer_len_++] = 0x00;
            transform();
            buffer_len_ = 0;
        }
        while (buffer_len_ < 56) buffer_[buffer_len_++] = 0x00;
        for (int i = 7; i >= 0; --i) {
            buffer_[buffer_len_++] = static_cast<uint8_t>((total_bits >> (i * 8)) & 0xff);
        }
        transform();

        std::ostringstream ss;
        for (uint32_t val : state_) {
            ss << std::hex << std::setw(8) << std::setfill('0') << val;
        }
        return ss.str();
    }

private:
    uint32_t state_[8];
    uint8_t buffer_[64];
    size_t buffer_len_{0};
    uint64_t bitlen_{0};

    static inline uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    void reset() {
        state_[0] = 0x6a09e667;
        state_[1] = 0xbb67ae85;
        state_[2] = 0x3c6ef372;
        state_[3] = 0xa54ff53a;
        state_[4] = 0x510e527f;
        state_[5] = 0x9b05688c;
        state_[6] = 0x1f83d9ab;
        state_[7] = 0x5be0cd19;
        buffer_len_ = 0;
        bitlen_ = 0;
    }

    void transform() {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };

        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(buffer_[i * 4]) << 24) |
                   (static_cast<uint32_t>(buffer_[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(buffer_[i * 4 + 2]) << 8) |
                   (static_cast<uint32_t>(buffer_[i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + s1 + ch + k[i] + w[i];
            uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }
};

} // anonymous namespace

FileLockGuard::FileLockGuard(const std::string& lock_file_path, bool shared)
    : lock_file_path_(lock_file_path) {
    fd_ = ::open(lock_file_path_.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ >= 0) {
        int op = shared ? LOCK_SH : LOCK_EX;
        if (::flock(fd_, op) != 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }
}

FileLockGuard::~FileLockGuard() {
    unlock();
}

FileLockGuard::FileLockGuard(FileLockGuard&& other) noexcept
    : fd_(other.fd_), lock_file_path_(std::move(other.lock_file_path_)) {
    other.fd_ = -1;
}

FileLockGuard& FileLockGuard::operator=(FileLockGuard&& other) noexcept {
    if (this != &other) {
        unlock();
        fd_ = other.fd_;
        lock_file_path_ = std::move(other.lock_file_path_);
        other.fd_ = -1;
    }
    return *this;
}

void FileLockGuard::unlock() {
    if (fd_ >= 0) {
        ::flock(fd_, LOCK_UN);
        ::close(fd_);
        fd_ = -1;
    }
}

KernelCache::KernelCache(std::string cache_dir)
    : cache_dir_(std::move(cache_dir)) {
    std::filesystem::create_directories(cache_dir_);
}

FileLockGuard KernelCache::acquire_compile_lock(const std::string& key) const {
    std::filesystem::path lock_path = std::filesystem::path(cache_dir_) / (key + ".compile.lock");
    return FileLockGuard(lock_path.string(), false);
}

std::string KernelCache::compute_key(const std::string& canonical_ir,
                                     const std::string& metadata,
                                     const std::string& compiler_info) {
    SHA256 sha;
    sha.update("tinytensor_v0.1.1\n");
    sha.update("abi=KernelEntryFn_v1;arch=x86_64;flags=-fPIC,-O3,-shared,-std=c++20\n");
    sha.update(compiler_info);
    sha.update("\n---IR---\n");
    sha.update(canonical_ir);
    sha.update("\n---META---\n");
    sha.update(metadata);
    return sha.finalize();
}

bool KernelCache::get(const std::string& key, std::string& out_so_path, bool count_miss) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::filesystem::path p = std::filesystem::path(cache_dir_) / (key + ".so");
    if (!std::filesystem::exists(p)) {
        if (count_miss) misses_++;
        return false;
    }

    std::error_code ec;
    auto file_size = std::filesystem::file_size(p, ec);
    if (ec || file_size < 64) {
        // Truncated or unreadable file: invalidate entry
        std::filesystem::remove(p, ec);
        if (count_miss) misses_++;
        return false;
    }

    // Verify ELF binary magic header: 0x7f, 'E', 'L', 'F'
    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) {
        if (count_miss) misses_++;
        return false;
    }
    char magic[4];
    f.read(magic, 4);
    if (f.gcount() != 4 ||
        static_cast<uint8_t>(magic[0]) != 0x7f ||
        magic[1] != 'E' || magic[2] != 'L' || magic[3] != 'F') {
        f.close();
        std::filesystem::remove(p, ec);
        if (count_miss) misses_++;
        return false;
    }
    f.close();

    out_so_path = p.string();
    hits_++;
    return true;
}

bool KernelCache::get_source(const std::string& key, std::string& out_source) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::filesystem::path p = std::filesystem::path(cache_dir_) / (key + ".cpp");
    if (!std::filesystem::exists(p)) {
        return false;
    }
    std::ifstream f(p);
    if (!f.is_open()) {
        return false;
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    out_source = buffer.str();
    return true;
}

void KernelCache::put(const std::string& key, const std::string& temp_so_path, const std::string& source_code) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Cross-process file locking for publication race safety
    std::filesystem::path lock_path = std::filesystem::path(cache_dir_) / (key + ".publish.lock");
    FileLockGuard process_lock(lock_path.string(), false);

    std::filesystem::path final_path = std::filesystem::path(cache_dir_) / (key + ".so");
    std::error_code ec;

    // Concurrency guard: if another thread/process already published a valid file, do not overwrite in-place
    if (std::filesystem::exists(final_path, ec) && std::filesystem::file_size(final_path, ec) >= 64) {
        std::filesystem::remove(temp_so_path, ec);
        return;
    }

    // Direct atomic rename (if on the same filesystem)
    std::filesystem::rename(temp_so_path, final_path, ec);
    if (ec) {
        // Cross-device link (e.g. /tmp to /mnt/d/...): Stage to unique file in target cache directory first
        std::string unique_suffix = std::to_string(reinterpret_cast<uintptr_t>(&ec)) + "_" +
                                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        std::filesystem::path staging_path = std::filesystem::path(cache_dir_) / (key + ".tmp." + unique_suffix + ".so");

        std::filesystem::copy_file(temp_so_path, staging_path,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(temp_so_path, ec);

        if (!ec) {
            // Atomic publication via same-directory rename
            if (!std::filesystem::exists(final_path, ec) || std::filesystem::file_size(final_path, ec) < 64) {
                std::filesystem::rename(staging_path, final_path, ec);
            } else {
                std::filesystem::remove(staging_path, ec);
            }
        }
    }

    // Persist C++ source code if supplied
    if (!source_code.empty()) {
        std::filesystem::path cpp_path = std::filesystem::path(cache_dir_) / (key + ".cpp");
        if (!std::filesystem::exists(cpp_path, ec)) {
            std::string unique_suffix = std::to_string(reinterpret_cast<uintptr_t>(&ec)) + "_" +
                                        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
            std::filesystem::path temp_cpp_path = std::filesystem::path(cache_dir_) / (key + ".tmp." + unique_suffix + ".cpp");
            std::ofstream cpp_out(temp_cpp_path);
            if (cpp_out.is_open()) {
                cpp_out << source_code;
                cpp_out.close();
                std::filesystem::rename(temp_cpp_path, cpp_path, ec);
            }
        }
    }
}

void KernelCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::error_code ec;
    std::filesystem::remove_all(cache_dir_, ec);
    std::filesystem::create_directories(cache_dir_, ec);
}

} // namespace tinytensor
