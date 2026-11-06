#include "tinytensor/codegen/cpu_features.hpp"
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>

static bool host_cpuid_impl(uint32_t leaf, uint32_t subleaf, tinytensor::simd::CPUIDResult& out) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
    out.eax = eax;
    out.ebx = ebx;
    out.ecx = ecx;
    out.edx = edx;
    return true;
}

static uint64_t host_xgetbv_impl(uint32_t xcr) {
    uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
    return (static_cast<uint64_t>(edx) << 32) | eax;
}
#else
static bool host_cpuid_impl(uint32_t, uint32_t, tinytensor::simd::CPUIDResult&) {
    return false;
}

static uint64_t host_xgetbv_impl(uint32_t) {
    return 0;
}
#endif

namespace tinytensor {
namespace simd {

CPUFeatures detect_features_custom(CPUIDFn cpuid_fn, XGetBVFn xgetbv_fn) {
    CPUFeatures feat;
    if (!cpuid_fn || !xgetbv_fn) {
        return feat;
    }

    // 1. Check maximum supported basic leaf
    CPUIDResult r0;
    if (!cpuid_fn(0, 0, r0) || r0.eax < 1) {
        return feat;
    }

    // 2. Query leaf 1 for AVX (ECX bit 28), OSXSAVE (ECX bit 27), FMA3 (ECX bit 12)
    CPUIDResult r1;
    if (!cpuid_fn(1, 0, r1)) {
        return feat;
    }

    feat.osxsave = (r1.ecx & (1u << 27)) != 0;
    bool hw_avx = (r1.ecx & (1u << 28)) != 0;
    bool hw_fma = (r1.ecx & (1u << 12)) != 0;

    // Check OS state enablement: both XMM (bit 1) and YMM (bit 2) must be active
    if (feat.osxsave) {
        uint64_t xcr0 = xgetbv_fn(0);
        feat.os_avx_support = ((xcr0 & 0x6ULL) == 0x6ULL);
    }

    if (hw_avx && feat.os_avx_support) {
        feat.has_avx = true;
    }

    // FMA requires both hardware FMA3, hardware AVX, and OS AVX context enablement
    if (feat.has_avx && hw_fma) {
        feat.has_fma3 = true;
    }

    // AVX2 requires hardware AVX and leaf 7 AVX2 support
    if (r0.eax >= 7 && feat.has_avx) {
        CPUIDResult r7;
        if (cpuid_fn(7, 0, r7)) {
            bool hw_avx2 = (r7.ebx & (1u << 5)) != 0;
            if (hw_avx2) {
                feat.has_avx2 = true;
            }
        }
    }

    return feat;
}

TargetISA supported_target_isa_custom(CPUIDFn cpuid_fn, XGetBVFn xgetbv_fn) {
    CPUFeatures feat = detect_features_custom(cpuid_fn, xgetbv_fn);
    if (feat.has_avx2 && feat.has_fma3) {
        return TargetISA::AVX2_FMA;
    }
    if (feat.has_avx2) {
        return TargetISA::AVX2;
    }
    return TargetISA::Scalar;
}

CPUFeatures detect_host_features() {
    return detect_features_custom(host_cpuid_impl, host_xgetbv_impl);
}

bool has_avx() {
    return detect_host_features().has_avx;
}

bool has_avx2() {
    return detect_host_features().has_avx2;
}

bool has_fma3() {
    return detect_host_features().has_fma3;
}

TargetISA supported_target_isa() {
    return supported_target_isa_custom(host_cpuid_impl, host_xgetbv_impl);
}

std::string supported_kernel_features() {
    auto feat = detect_host_features();
    std::string s;
    if (feat.has_avx) s += "AVX ";
    if (feat.has_avx2) s += "AVX2 ";
    if (feat.has_fma3) s += "FMA3 ";
    if (s.empty()) {
        s = "Scalar";
    } else if (s.back() == ' ') {
        s.pop_back();
    }
    return s;
}

} // namespace simd
} // namespace tinytensor
