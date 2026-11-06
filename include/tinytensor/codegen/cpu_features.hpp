#pragma once

#include <cstdint>
#include <string>

namespace tinytensor {

enum class TargetISA {
    Scalar = 0,
    AVX2 = 1,
    AVX2_FMA = 2
};

namespace simd {

struct CPUFeatures {
    bool has_avx{false};
    bool has_avx2{false};
    bool has_fma3{false};
    bool osxsave{false};
    bool os_avx_support{false}; // XCR0 XMM & YMM enabled
};

struct CPUIDResult {
    uint32_t eax{0};
    uint32_t ebx{0};
    uint32_t ecx{0};
    uint32_t edx{0};
};

using CPUIDFn = bool (*)(uint32_t leaf, uint32_t subleaf, CPUIDResult& out);
using XGetBVFn = uint64_t (*)(uint32_t xcr);

// Hardware detection on host
bool has_avx();
bool has_avx2();
bool has_fma3();
CPUFeatures detect_host_features();
TargetISA supported_target_isa();
std::string supported_kernel_features();

// Injectable feature detection for synthetic/mock testing
CPUFeatures detect_features_custom(CPUIDFn cpuid_fn, XGetBVFn xgetbv_fn);
TargetISA supported_target_isa_custom(CPUIDFn cpuid_fn, XGetBVFn xgetbv_fn);

} // namespace simd
} // namespace tinytensor
