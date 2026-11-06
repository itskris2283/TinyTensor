#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include "tinytensor/core/shape.hpp"
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

constexpr size_t MEMORY_ALIGNMENT = 64; // 64-byte alignment for AVX2 / cache lines

class Storage {
public:
    explicit Storage(size_t nbytes)
        : nbytes_(nbytes), is_owning_(true) {
        if (nbytes > MAX_ALLOCATION_BYTES) {
            throw ResourceLimitError("Requested allocation size " + std::to_string(nbytes) +
                                     " bytes exceeds maximum allowed " +
                                     std::to_string(MAX_ALLOCATION_BYTES));
        }

        if (nbytes > 0) {
            void* ptr = nullptr;
            int ret = posix_memalign(&ptr, MEMORY_ALIGNMENT, nbytes);
            if (ret != 0 || !ptr) {
                throw MemoryError("Failed to allocate " + std::to_string(nbytes) +
                                  " bytes of aligned memory");
            }
            data_ = ptr;
            // Zero memory on allocation for deterministic state
            std::memset(data_, 0, nbytes);
        } else {
            data_ = nullptr;
        }
    }

    Storage(void* external_data, size_t nbytes, bool owning = false)
        : data_(external_data), nbytes_(nbytes), is_owning_(owning) {
        if (nbytes > MAX_ALLOCATION_BYTES) {
            throw ResourceLimitError("Storage size exceeds maximum limit");
        }
    }

    ~Storage() {
        if (is_owning_ && data_) {
            std::free(data_);
            data_ = nullptr;
        }
    }

    // Storage is non-copyable to prevent accidental double-free
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    Storage(Storage&& other) noexcept
        : data_(other.data_), nbytes_(other.nbytes_), is_owning_(other.is_owning_) {
        other.data_ = nullptr;
        other.nbytes_ = 0;
        other.is_owning_ = false;
    }

    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            if (is_owning_ && data_) {
                std::free(data_);
            }
            data_ = other.data_;
            nbytes_ = other.nbytes_;
            is_owning_ = other.is_owning_;
            other.data_ = nullptr;
            other.nbytes_ = 0;
            other.is_owning_ = false;
        }
        return *this;
    }

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    size_t nbytes() const noexcept { return nbytes_; }
    bool is_owning() const noexcept { return is_owning_; }

private:
    void* data_{nullptr};
    size_t nbytes_{0};
    bool is_owning_{true};
};

using StoragePtr = std::shared_ptr<Storage>;

inline StoragePtr make_storage(size_t nbytes) {
    return std::make_shared<Storage>(nbytes);
}

inline StoragePtr make_storage_from_blob(void* ptr, size_t nbytes, bool owning = false) {
    return std::make_shared<Storage>(ptr, nbytes, owning);
}

} // namespace tinytensor
