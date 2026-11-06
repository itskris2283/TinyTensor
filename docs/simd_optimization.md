# TinyTensor SIMD Optimization and Hardware Acceleration

## 1. AVX2 Architectural Specialization

TinyTensor provides native AVX2 SIMD acceleration for modern x86-64 processors. AVX2 introduces 256-bit wide `ymm` vector registers, allowing simultaneous processing of 8 single-precision (`float32`) elements per cycle.

### 1.1 Vectorized Intrinsics Used
- `_mm256_loadu_ps(ptr)`: Unaligned 256-bit memory load (8 floats).
- `_mm256_storeu_ps(ptr, val)`: Unaligned 256-bit memory store.
- `_mm256_add_ps(a, b)`: 8-wide parallel addition.
- `_mm256_sub_ps(a, b)`: 8-wide parallel subtraction.
- `_mm256_mul_ps(a, b)`: 8-wide parallel multiplication.
- `_mm256_div_ps(a, b)`: 8-wide parallel division.
- `_mm256_max_ps(a, zero)`: 8-wide parallel ReLU activation.
- `_mm256_fmadd_ps(a, b, c)`: Fused Multiply-Add (FMA3) computing $a \cdot b + c$ with single rounding.

## 2. Scalar Tail Handling

When tensor element counts are not multiples of 8, processing the final $N \pmod 8$ elements via SIMD would read or write past allocated memory bounds. TinyTensor divides vectorized loops into two distinct phases:

```cpp
// 1. Vectorized bulk chunk (8 elements per step)
size_t simd_limit = n & ~7ULL; // n - (n % 8)
for (size_t i = 0; i < simd_limit; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vout = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(out + i, vout);
}

// 2. Scalar tail loop for remaining 0 to 7 elements
for (size_t i = simd_limit; i < n; ++i) {
    out[i] = a[i] + b[i];
}
```

## 3. Portable Scalar Fallback & CPUID Detection

### 3.1 Non-AVX2 Compilation Isolation
To maintain strict binary portability across heterogeneous execution targets without crashing with `SIGILL` (Illegal Instruction):
- No global `-mavx2` compiler flag is set in the main codebase.
- AVX2 kernels are isolated in dedicated translation units (`src/codegen/avx2_backend.cpp`).
- At runtime, `tinytensor::simd::has_avx2()` executes the `__cpuid` instruction (EAX=7, ECX=0, testing EBX bit 5).
- If AVX2 is absent or if the user requests `mode="scalar"`, the runtime seamlessly executes the portable pure C++ scalar fallback.
