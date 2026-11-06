#pragma once

#include <dlfcn.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "tinytensor/core/tensor.hpp"
#include "tinytensor/graph/graph.hpp"
#include "tinytensor/runtime/profiler.hpp"

namespace tinytensor {

using KernelEntryFn = int (*)(void** input_ptrs, void** output_ptrs, void** temp_ptrs);

class Executable {
public:
    Executable(void* dl_handle, KernelEntryFn fn,
               std::vector<Shape> input_shapes, std::vector<DType> input_dtypes,
               std::vector<Shape> output_shapes, std::vector<DType> output_dtypes,
               std::vector<size_t> temp_buffer_sizes,
               std::string so_path = "",
               std::string source_code = "")
        : dl_handle_(dl_handle),
          kernel_fn_(fn),
          input_shapes_(std::move(input_shapes)),
          input_dtypes_(std::move(input_dtypes)),
          output_shapes_(std::move(output_shapes)),
          output_dtypes_(std::move(output_dtypes)),
          temp_buffer_sizes_(std::move(temp_buffer_sizes)),
          so_path_(std::move(so_path)),
          source_code_(std::move(source_code)) {}

    ~Executable() {
        if (dl_handle_) {
            dlclose(dl_handle_);
            dl_handle_ = nullptr;
        }
    }

    Executable(const Executable&) = delete;
    Executable& operator=(const Executable&) = delete;

    Executable(Executable&& other) noexcept
        : dl_handle_(other.dl_handle_),
          kernel_fn_(other.kernel_fn_),
          input_shapes_(std::move(other.input_shapes_)),
          input_dtypes_(std::move(other.input_dtypes_)),
          output_shapes_(std::move(other.output_shapes_)),
          output_dtypes_(std::move(other.output_dtypes_)),
          temp_buffer_sizes_(std::move(other.temp_buffer_sizes_)),
          so_path_(std::move(other.so_path_)),
          source_code_(std::move(other.source_code_)) {
        other.dl_handle_ = nullptr;
        other.kernel_fn_ = nullptr;
    }

    Executable& operator=(Executable&& other) noexcept {
        if (this != &other) {
            if (dl_handle_) {
                dlclose(dl_handle_);
            }
            dl_handle_ = other.dl_handle_;
            kernel_fn_ = other.kernel_fn_;
            input_shapes_ = std::move(other.input_shapes_);
            input_dtypes_ = std::move(other.input_dtypes_);
            output_shapes_ = std::move(other.output_shapes_);
            output_dtypes_ = std::move(other.output_dtypes_);
            temp_buffer_sizes_ = std::move(other.temp_buffer_sizes_);
            so_path_ = std::move(other.so_path_);
            source_code_ = std::move(other.source_code_);
            other.dl_handle_ = nullptr;
            other.kernel_fn_ = nullptr;
        }
        return *this;
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs, Profiler* profiler = nullptr);
    const std::string& so_path() const noexcept { return so_path_; }
    const std::string& source_code() const noexcept { return source_code_; }

private:
    void* dl_handle_{nullptr};
    KernelEntryFn kernel_fn_{nullptr};
    std::vector<Shape> input_shapes_;
    std::vector<DType> input_dtypes_;
    std::vector<Shape> output_shapes_;
    std::vector<DType> output_dtypes_;
    std::vector<size_t> temp_buffer_sizes_;
    std::string so_path_;
    std::string source_code_;
};

} // namespace tinytensor
