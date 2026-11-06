# TinyTensor Phase-II Empirical Baseline Verification

**Date:** 2026-11-06  
**Auditor:** Principal Systems Engineer & Compiler Architect  
**Baseline Version:** v0.1.0  
**Status:** FULLY REPRODUCED & VALIDATED  

---

## 1. Environment & Hardware Inventory

| Item | Specification / Value |
|---|---|
| **Operating System** | Linux 6.18.54.1-microsoft-standard-WSL2 (x86_64) |
| **Linux Distribution** | Ubuntu 24.04 LTS (Noble Numbat) |
| **CPU Architecture** | x86_64 (64-bit Little Endian) |
| **Processor Model** | Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz |
| **Cores / Threads** | 4 physical cores, 8 hardware threads (SMT enabled) |
| **CPU Caches** | L1d: 128 KiB (32 KiB/core), L1i: 128 KiB, L2: 1 MiB (256 KiB/core), L3: 8 MiB |
| **SIMD Extensions** | SSE4.1, SSE4.2, AVX, AVX2, FMA3, BMI1, BMI2 |
| **C++ Toolchain** | GNU GCC 13.3.0, CMake 3.28.3, Ninja 1.11.1 |
| **Python Toolchain** | Python 3.12.3, NumPy 2.5.4, PyTorch 2.15.0+cpu, pybind11 3.1.0, pytest 9.2.0 |
| **Coverage Tools** | gcovr 7.0, pytest-cov 7.1.0 |

---

## 2. Test Suite Execution Baseline

### 2.1 C++ GoogleTest Suite (`build/`)
- **Total Test Suites Executed:** 5 (`CoreTests`, `GraphIRTests`, `AutodiffTests`, `OptimizerTests`, `RuntimeCodegenTests`)
- **Total Tests Run:** 103
- **Passed:** 103 (100%)
- **Failed:** 0 (0%)
- **Real Execution Time:** 9.99s

### 2.2 Python pytest Suite
- **Total Python Tests Run:** 111
- **Passed:** 111 (100%)
- **Failed:** 0 (0%)
- **Real Execution Time:** 76.88s

### 2.3 Interactive Examples (`examples/`)
All 6 standalone Python examples executed with 100% exit code 0:
1. `examples/basic_tensor.py`: PASSED (tensor creation, strides, view slicing, broadcasting).
2. `examples/matrix_multiplication.py`: PASSED (2D GEMM, transposed views, batched matmul).
3. `examples/autodiff_example.py`: PASSED (analytical autodiff verified against finite differences).
4. `examples/compiler_optimization.py`: PASSED (constant folding, DCE, CSE, elementwise fusion, memory reuse).
5. `examples/custom_graph.py`: PASSED (SSA IR construction, Graphviz DOT export, execution).
6. `examples/neural_network_example.py`: PASSED (2-layer MLP training with 39.2% loss reduction over 25 epochs).

---

## 3. Dynamic Memory & Sanitizer Verification (`build_asan/`)
Built with `-fsanitize=address,undefined -fno-omit-frame-pointer`:
- **AddressSanitizer:** 0 heap buffer overflows, 0 stack overflows, 0 use-after-free, 0 double-free, 0 memory leaks.
- **UndefinedBehaviorSanitizer:** 0 misaligned pointers, 0 integer overflows, 0 null pointer dereferences, 0 undefined behavior instances.
- **Test Result:** 5/5 test suites passed (5.59s).

---

## 4. Code Coverage Baseline

### 4.1 C++ Core Logic (`build_cov/` + `gcovr`)
Direct analysis of all files under `src/`:
- **Total Lines Analyzed:** 2,184
- **Executed Lines:** 1,746
- **Baseline Line Coverage:** **79%** (1,746 / 2,184 lines)
- **Total Branches Analyzed:** 4,543
- **Executed Branches:** 1,856
- **Baseline Branch Coverage:** **40%** (1,856 / 4,543 branches)

*Key module breakdown:*
| Module | Line Coverage | Branch Coverage | Target in Phase-II |
|---|---|---|---|
| `src/autodiff/gradient.cpp` | 89% (208/233) | 50% (198/389) | $\ge 92\%$ line |
| `src/codegen/avx2_backend.cpp` | 100% (63/63) | 100% (28/28) | Maintain 100% |
| `src/codegen/cpp_backend.cpp` | 50% (208/410) | 25% (340/1321) | $\ge 75\%$ line |
| `src/codegen/kernel_cache.cpp` | 100% (113/113) | 61% (59/96) | $\ge 75\%$ branch |
| `src/core/operators.cpp` | 79% (259/327) | 48% (258/528) | $\ge 88\%$ line |
| `src/core/tensor.cpp` | 88% (172/194) | 44% (190/426) | $\ge 90\%$ line |
| `src/graph/graph.cpp` | 89% (171/192) | 41% (176/426) | $\ge 92\%$ line |
| `src/ir/verifier.cpp` | 70% (61/86) | 28% (70/243) | $\ge 90\%$ line |
| `src/optimizer/constant_folding.cpp` | 56% (50/88) | 37% (53/141) | $\ge 88\%$ line |
| `src/runtime/executable.cpp` | 63% (23/36) | 24% (22/90) | $\ge 85\%$ line |
| `src/runtime/interpreter.cpp` | 86% (77/89) | 44% (73/165) | $\ge 90\%$ line |

### 4.2 Python Coverage (`pytest-cov`)
- **Total Statements Analyzed:** 310
- **Covered Statements:** 303
- **Baseline Statement Coverage:** **98%** (exceeds $\ge 90\%$ requirement)

---

## 5. Empirical Benchmark Baseline (Thread-Pinned)

To eliminate multi-threading contention and initialization spikes previously observed in WSL2, all reference frameworks were pinned to single-threaded CPU execution:
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

### 5.1 Matrix Multiplication (GEMM) Median Latency (P50)

| Matrix Size | TinyTensor AVX2 P50 (ms) | NumPy OpenBLAS P50 (ms) | PyTorch CPU P50 (ms) |
|---|---|---|---|
| **32x32** | 0.0344 | 0.0104 | 0.0059 |
| **64x64** | 0.2289 | 0.0091 | 0.0159 |
| **128x128** | 0.1907 | 0.0570 | 0.0538 |
| **256x256** | 1.4589 | 0.3647 | 0.4484 |
| **512x512** | 16.5184 | 3.5190 | 2.9793 |

*Resolution of Previous Anomaly:*
Under thread pinning, NumPy and PyTorch scale monotonically without the anomalous 13–15 ms spike previously observed at 128x128. TinyTensor compiled execution performs predictably; however, its latency at 256x256 and 512x512 reflects the unblocked scalar loop structure, which will be optimized in Phase G via cache blocking and AVX2 vectorization.

### 5.2 Elementwise Addition Median Latency (P50)

| Vector Size | TinyTensor AVX2 P50 (ms) | TinyTensor Scalar P50 (ms) | NumPy P50 (ms) | PyTorch CPU P50 (ms) |
|---|---|---|---|---|
| **100** | 0.0084 | 0.0083 | 0.0022 | 0.0028 |
| **1,000** | 0.0117 | 0.0094 | 0.0029 | 0.0033 |
| **10,000** | 0.0169 | 0.0171 | 0.0148 | 0.0068 |
| **100,000** | 0.1158 | 0.0985 | 0.1550 | 0.0443 |
| **1,000,000** | 2.2292 | 3.3942 | 2.3546 | 1.6177 |

### 5.3 Memory Planning Optimization
- 5 Layers (Dim 256): 86.7% memory footprint reduction (3,840 KB $\to$ 512 KB).
- 10 Layers (Dim 256): 93.3% memory footprint reduction (7,680 KB $\to$ 512 KB).
- 20 Layers (Dim 512): 96.7% memory footprint reduction (61,440 KB $\to$ 2,048 KB).

### 5.4 Reverse-Mode Autodiff Median Latency (P50)
- Batch 16: TinyTensor 0.0596 ms vs PyTorch 0.1071 ms (1.8x faster than PyTorch).
- Batch 64: TinyTensor 0.1015 ms vs PyTorch 0.1086 ms (parity).
- Batch 256: TinyTensor 0.5444 ms vs PyTorch 0.0962 ms.
- Batch 1024: TinyTensor 0.9637 ms vs PyTorch 0.1576 ms.
