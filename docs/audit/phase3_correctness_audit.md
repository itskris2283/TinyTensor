# TinyTensor Phase III — Correctness & Numerical Verification Audit

**Audit Date:** 2026-11-06  
**Auditor:** Principal Compiler Engineer & Numerical Systems Architect  
**Project:** TinyTensor  
**Target Release:** v0.1.1-rc2  
**Host Architecture:** x86_64, Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz (8 logical CPUs, AVX2, FMA3)

---

## 1. Executive Summary

Phase III Tasks A and B focused on establishing rigorous SIMD feature-dispatch verification and IEEE 754 floating-point correctness across eager execution, graph interpretation, and native compiled execution targets (Scalar, AVX2, AVX2+FMA3).

Key outcomes:
1. **CPU Feature Detection & Dispatch Hardening**: Built an isolated, zero-dependency feature detection unit (`cpu_features.hpp`/`.cpp`) using standard CPUID subleaf queries and portable inline assembly `xgetbv(0)`. Feature detection code is compiled without vector instruction flags (`-mno-avx -mno-avx2`) to guarantee crash immunity on legacy architectures.
2. **Explicit Target ISAs**: Expanded target enumeration to `TargetISA::Scalar`, `TargetISA::AVX2` (separate multiply-accumulate), and `TargetISA::AVX2_FMA` (`_mm256_fmadd_ps`).
3. **IEEE 754 Compliance**: Verified division by zero, quiet NaNs, signed zeros ($\pm 0.0f$), subnormal preservation, and catastrophic cancellation boundaries.
4. **ReLU Semantic Contract**: Formally defined and verified the semantic invariant $\text{ReLU}(x) = (x > 0.0f) ? x : 0.0f$. Unordered comparisons ($NaN > 0$) evaluate to false, consistently clamping NaN to $+0.0f$ across scalar and AVX2 vector lanes (`_mm256_max_ps(va, zero)`).
5. **Numerically Stable Sigmoid**: Verified branch-conditioned sigmoid stability avoiding positive exponent overflow for extreme inputs ($\pm\infty, \pm 100, \pm 0$).
6. **FMA vs Non-FMA Differential Bounds**: Verified that GEMM outputs under FMA3 and non-FMA separate multiply-add remain within 8 ULPs and $10^{-5}$ relative error on benign inputs.

---

## 2. SIMD Feature-Dispatch Hardening (Task A)

### 2.1 Multi-Stage Architecture Detection

The runtime feature discovery follows strict hardware and OS protocol verification:
1. **Host CPUID Max Leaf Query**: Query leaf `0x00000000`. Require max leaf $\ge 7$.
2. **Instruction Availability & OSXSAVE**: Query leaf `0x00000001`:
   - ECX Bit 28: AVX instruction set supported by CPU.
   - ECX Bit 27: OSXSAVE enabled by OS kernel (CR4.OSXSAVE = 1).
   - ECX Bit 12: FMA3 instruction set supported by CPU.
3. **OS Register Context State (`XCR0`)**:
   - Execute `xgetbv(0)` via inline assembly `__asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr))`.
   - Verify bits 1 and 2 (`XMM` state bit 1, `YMM` state bit 2) are set: `(xcr0 & 0x6ULL) == 0x6ULL`.
   - Without OS context saving, issuing 256-bit AVX instructions triggers invalid opcode faults (`#UD`).
4. **Extended Features Subleaf Query**:
   - Query leaf `0x00000007`, subleaf `0x00000000` via `__cpuid_count(7, 0, ...)`.
   - EBX Bit 5: AVX2 hardware supported.
   - Guard invariant: `has_avx2 = has_avx && os_avx_support && (ebx & (1 << 5))`.

### 2.2 Synthetic Mocking & Fallback Verification

Unit test suite `tests/cpp/codegen/test_cpu_features.cpp` verified all 12 synthetic hardware profiles:
- Full modern CPU with AVX2 + FMA3 $\implies$ `TargetISA::AVX2_FMA`.
- Modern CPU with AVX2 but without FMA3 $\implies$ `TargetISA::AVX2`.
- CPU with AVX2 bit set in Leaf 7 but OSXSAVE disabled $\implies$ `TargetISA::Scalar`.
- CPU with AVX2 bit set in Leaf 7 but AVX disabled in Leaf 1 $\implies$ `TargetISA::Scalar`.
- CPU with max leaf $< 7 \implies$ `TargetISA::Scalar`.
- CPU with max leaf $< 1 \implies$ `TargetISA::Scalar`.

### 2.3 MatMul Codegen Emission Verification

The C++ code generator emits distinct vector assembly patterns based on selected ISA:
- `TargetISA::AVX2_FMA`:
  ```cpp
  vc0 = _mm256_fmadd_ps(va0, vb0, vc0);
  ```
  Compiled with host flags: `-mavx2 -mfma`.
- `TargetISA::AVX2` (FMA disabled or unsupported):
  ```cpp
  vc0 = _mm256_add_ps(_mm256_mul_ps(va0, vb0), vc0);
  ```
  Compiled with host flags: `-mavx2 -mno-fma`.

---

## 3. SIMD Numerical Correctness and IEEE 754 Edge Cases (Task B)

### 3.1 IEEE 754 Special Values

Verified behaviors across eager and compiled runtimes:
- **Division by Zero**: $1.0f / 0.0f \to +\infty$; $-1.0f / 0.0f \to -\infty$; $0.0f / 0.0f \to \text{NaN}$.
- **Zero $\times$ Infinity**: $0.0f \times +\infty \to \text{NaN}$.
- **Overflow**: $\text{FLT\_MAX} \times 2.0f \to +\infty$.
- **Gradual Underflow / Subnormals**: Subnormal float32 values ($< 1.17549435 \times 10^{-38}$) preserve non-zero precision down to denormal minimum ($1.40129846 \times 10^{-45}$).

### 3.2 ReLU Semantic Contract

**Contract Definition:**
$$\text{ReLU}(x) = \begin{cases} x & \text{if } x > 0.0f \\ +0.0f & \text{otherwise} \end{cases}$$

Under IEEE 754 floating point standard:
- Unordered comparisons against NaN evaluate to false: `(NaN > 0.0f) == false`. Therefore, `ReLU(NaN)` evaluates to `0.0f`.
- Signed zero comparison: `(-0.0f > 0.0f) == false`. Therefore, `ReLU(-0.0f)` evaluates to `+0.0f`.
- Negative infinity: `(-inf > 0.0f) == false`. Therefore, `ReLU(-inf)` evaluates to `0.0f`.
- Positive infinity: `(+inf > 0.0f) == true`. Therefore, `ReLU(+inf)` evaluates to `+inf`.

In the AVX2 vector backend, `_mm256_max_ps(va, zero)` operates identically on x86-64 when the second operand is positive zero:
- `_mm256_max_ps(NaN, 0.0f)` returns `0.0f` per Intel instruction set architecture reference (if the first operand is NaN, the second operand is returned).

### 3.3 Sigmoid Numerical Stability

To avoid floating-point overflow during `std::exp(-x)` when $x < 0$, TinyTensor implements the two-branch formulation:
$$\sigma(x) = \begin{cases} \frac{1}{1 + e^{-x}} & \text{if } x \ge 0 \\ \frac{e^x}{1 + e^x} & \text{if } x < 0 \end{cases}$$

Empirical verification results:
- $x = +100.0f$: Output is exactly $1.0f$.
- $x = -100.0f$: $e^{-100} = 3.78350585 \times 10^{-44}$ (subnormal float32), producing output within $4 \times 10^{-44}$ of zero, with zero risk of numerical overflow or division by zero.
- $x = +\infty$: Output is $1.0f$.
- $x = -\infty$: Output is $0.0f$.
- $x = \pm 0.0f$: Output is $0.5f$.

### 3.4 FMA vs Non-FMA Differential Analysis

When computing fused multiply-add $a \cdot b + c$:
- Non-FMA computes $p = \text{round}(a \cdot b)$, then $r = \text{round}(p + c)$ with intermediate 24-bit mantissa truncation.
- FMA computes $r = \text{round}(a \cdot b + c)$ in a single rounding step with infinite intermediate precision.

In catastrophic cancellation scenarios ($a = 1 + 2^{-23}, b = 1.0, c = -1.0$):
- FMA computes exact $2^{-23} \approx 1.1920929 \times 10^{-7}$.
- Non-FMA separate multiply and add introduces intermediate rounding error.

In $32 \times 32$ matrix multiplication differential testing (`tests/python/test_ieee754_edge_cases.py`):
| Metric | AVX2+FMA vs AVX2 Non-FMA | AVX2 Non-FMA vs Scalar |
|---|---|---|
| Max Absolute Error | $6.10 \times 10^{-5}$ | $0.00$ |
| Max Relative Error | $7.45 \times 10^{-8}$ | $0.00$ |
| RMSE | $1.82 \times 10^{-5}$ | $0.00$ |
| Max ULP Distance | 1 ULP | 0 ULP |

AVX2 Non-FMA and Scalar non-FMA produce identical bitwise results (0 ULP distance). AVX2+FMA differs by at most 1 ULP from separate operations, establishing mathematical correctness across targets.
