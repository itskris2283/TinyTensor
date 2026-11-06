#include "tinytensor/runtime/executable.hpp"
#include <chrono>
#include "tinytensor/utils/errors.hpp"

namespace tinytensor {

std::vector<Tensor> Executable::run(const std::vector<Tensor>& inputs, Profiler* profiler) {
    if (!kernel_fn_) {
        throw RuntimeExecutionError("Attempted to execute an uninitialized or closed Executable");
    }

    if (inputs.size() != input_shapes_.size()) {
        throw RuntimeExecutionError("Expected " + std::to_string(input_shapes_.size()) +
                                    " inputs, but got " + std::to_string(inputs.size()));
    }

    std::vector<void*> in_ptrs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].shape() != input_shapes_[i]) {
            throw ShapeError("Input " + std::to_string(i) + " shape mismatch: expected " +
                             input_shapes_[i].to_string() + " but got " +
                             inputs[i].shape().to_string());
        }
        if (inputs[i].dtype() != input_dtypes_[i]) {
            throw TypeError("Input " + std::to_string(i) + " dtype mismatch");
        }
        in_ptrs[i] = const_cast<void*>(inputs[i].raw_data());
    }

    // Allocate outputs
    std::vector<Tensor> outputs;
    outputs.reserve(output_shapes_.size());
    std::vector<void*> out_ptrs(output_shapes_.size());

    for (size_t i = 0; i < output_shapes_.size(); ++i) {
        outputs.push_back(Tensor::empty(output_shapes_[i], output_dtypes_[i]));
        out_ptrs[i] = outputs[i].raw_data();
    }

    // Allocate temporary buffers
    std::vector<std::vector<char>> temp_storage(temp_buffer_sizes_.size());
    std::vector<void*> temp_ptrs(temp_buffer_sizes_.size());
    for (size_t i = 0; i < temp_buffer_sizes_.size(); ++i) {
        temp_storage[i].resize(temp_buffer_sizes_[i]);
        temp_ptrs[i] = temp_storage[i].data();
    }

    if (profiler) {
        profiler->clear();
        profiler->start_timer();
    }

    int ret = kernel_fn_(in_ptrs.data(), out_ptrs.data(), temp_ptrs.data());
    if (ret != 0) {
        throw RuntimeExecutionError("Compiled kernel returned error code: " + std::to_string(ret));
    }

    if (profiler) {
        profiler->stop_timer();
    }

    return outputs;
}

} // namespace tinytensor
