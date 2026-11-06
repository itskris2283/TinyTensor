# Changelog

All notable changes to TinyTensor are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1-rc2] - 2026-11-06

### Added
- **Cross-Process Kernel Cache Synchronization & File Locking (`flock`/`fcntl`):**
  - Introduced `FileLockGuard` implementing advisory POSIX file locking (`flock(fd, LOCK_EX)` / `flock(fd, LOCK_UN)`).
  - Coordinated cross-process kernel compilation: when multiple OS processes compile the same kernel simultaneously, only the lock-winning process runs `g++`, while other processes wait on `<key>.compile.lock` and instantly hit the cache once published.
  - Race-free atomic publication using `<key>.publish.lock` and atomic staging renames, preventing truncated `.so` binaries or corrupted ELF headers.
  - Multi-process concurrency stress testing across 2, 4, 8, and 16 worker processes verifying zero race conditions, zero orphaned files, and sub-millisecond cache hit latency.
- **Compiler Subprocess Management & Non-Blocking Pipe Draining:**
  - Hardened compiler pipe handling with iterative non-blocking draining (`EAGAIN`/`EWOULDBLOCK`) per polling tick to eliminate pipe buffer deadlocks regardless of compiler warning/error volume.
- **Mathematical Reframing & Characterization of FMA vs Non-FMA:**
  - Formulated mathematically rigorous error bounds distinguishing monotonic addition (where difference is $\le 1$ ULP) from subtractive cancellation ($a \times b \approx -c$, where relative error diverges while absolute error remains bounded by $\frac{1}{2}\text{ulp}(a \times b)$).
  - Expanded tests for subnormals, denormals, catastrophic cancellation, and strict 3-way differential testing across Scalar, AVX2, and AVX2+FMA.
- **Formal Four-Tier Memory Classification & Adversarial DAG Validation:**
  - Defined and measured Tier A (Theoretical peak live tensor bytes), Tier B (Planned buffer pool bytes), Tier C (Allocator requested bytes), and Tier D (Process RSS via `/proc/self/statm` & `getrusage`).
  - Adversarial DAG tests covering diamond reconvergence, high fan-out nodes (8+ consumers), and dead-code branch pruning.
- **Runtime Interpreter Support for Fused Ops:**
  - Added native evaluation for `OpType::FusedElementwise` inside `Interpreter`, allowing interpreted execution of optimized graphs without runtime errors.
- **Benchmarking Methodology & Publication Visualizations:**
  - Disentangled CPU affinity from thread count limits: enforced `OMP_NUM_THREADS=1` and true physical CPU core pinning via `os.sched_setaffinity(0, {0})`.
  - Generated all 9 publication-grade charts and complete raw CSV/JSON benchmark summaries.
- **Automated Verification Runner:**
  - Implemented `scripts/run_final_verification.py` orchestrating 3 consecutive full verification cycles from clean state with exit code 0.
- **Expanded Test Suites & Verification:**
  - 195 C++ GoogleTests passing (100%).
  - 135 Python pytests passing (100%).
  - Total automated tests: 330 passing.
  - C++ line coverage: 93.2% (2,473 / 2,653 lines).
  - C++ function coverage: 98.5% (129 / 131 functions).
  - C++ branch coverage: 53.7% (3,026 / 5,638 branches).
  - Python statement coverage: 98.0% (315 / 322 statements).
  - Clean ASan/UBSan runs on both C++ and Python suites.

## [0.1.1-rc1] - 2026-11-06

### Added
- **Compiler Diagnostic Introspection API:**
  - Added `original_ir()` exposing pre-optimization SSA graph representation.
  - Added `optimized_ir()` exposing post-optimization graph following transforms.
  - Added `generated_source()` exposing the synthesized C++20 kernel source code.
  - Added `execution_plan()` exposing the ordered execution dispatch schedule.
  - Python bindings and `CompiledProgram` convenience properties for all diagnostics.
- **Hardware SIMD & OS Context Hardening:**
  - Robust 4-stage CPUID/XCR0 verification (`has_avx2()`) checking CPUID leaf 0, leaf 1, OSXSAVE context, and `_xgetbv(0)` bits 1 and 2 (XMM and YMM operating system enablement) before issuing AVX2 instructions.
- **Kernel Cache Integrity & Corruption Recovery:**
  - SHA-256 cache key fingerprinting source code, compiler path, optimization flags, target ABI, host architecture, and runtime version.
  - Pre-flight binary validation checking file size ($\ge 64$ bytes) and ELF magic bytes (`\x7fELF`).
  - Automated cache corruption recovery: invalid `.so` entries are purged and transparently recompiled from `.cpp` source without interruption.
- **Comprehensive Verification & Differential Tests:**
  - 6-way cross-backend differential test validating numerical equivalence across Scalar Interpreter, Optimized Interpreter, Compiled Scalar, Compiled AVX2, NumPy, and PyTorch CPU.
  - 10+ negative IR verifier tests enforcing graph acyclicity, arity bounds, transpose axis permutations, and reduction shape consistency.
  - Disassembly verification tests asserting AVX2 instruction emission (`vaddps`, `vmaxps`, `vfmadd231ps`).

### Changed
- **GEMM (Matrix Multiplication) Optimization:**
  - Replaced naive 3-loop GEMM codegen with cache-blocked loop tiling ($BM=32, BK=32, BN=32$) and AVX2 FMA vectorization (`_mm256_fmadd_ps`), delivering 2.1x–9.4x speedup across matrix sizes.
  - Added support for batched matrix multiplication across arbitrary higher-rank batch dimensions.
- **Operator Fusion Loop Vectorization:**
  - Vectorized fused elementwise code generation with 8-wide AVX2 intrinsics (`_mm256_max_ps`, `_mm256_add_ps`), converting prior 0.60x slowdowns into 1.07x–1.39x end-to-end speedups.
- **IR Verifier Enforcement:**
  - Enforced `IRVerifier::verify()` before and after every individual optimization pass in `Optimizer::optimize()`.
- **Test Coverage & Reliability:**
  - Increased C++ line coverage to 91% (2,184 / 2,396 lines) and branch coverage to 49% (2,605 / 5,239 branches).
  - Maintained 98% Python statement coverage.
  - Sanitizer build isolation in `CMakeLists.txt` preventing ASan-instrumented shared library contamination.

## [0.1.0] - 2026-11-05

### Added
- Independent C++20 Tensor Core:
  - `Storage` class with 32-byte cacheline-aligned memory management and RAII ownership.
  - `Shape` and `Strides` engine supporting contiguous row-major, non-contiguous transposed views, and permutations without data copying.
  - NumPy-compatible broadcasting engine with stride-zero memory representation.
  - Checked arithmetic across all dimension and allocation computations preventing integer overflow.
- Computational Graph and Intermediate Representation (IR):
  - Directed Acyclic Graph (DAG) with static single assignment (SSA) value tracking.
  - Support for `Input`, `Constant`, `Add`, `Sub`, `Mul`, `Div`, `ReLU`, `Sigmoid`, `MatMul`, `Transpose`, `ReduceSum`, `ReduceMean`, and `FusedElementwise`.
  - Static `IRVerifier` enforcing topological sorting, arity, dtype, and shape invariants.
  - Human-readable textual IR serialization and Graphviz DOT visualization.
- Reverse-Mode Automatic Differentiation Engine:
  - Exact analytical gradient derivation via vector-Jacobian products (VJPs).
  - Dimension-collapsing unbroadcasting reduction along singleton dimensions.
  - Verified numerical parity against central finite differences.
- Compiler Optimization Pipeline:
  - Constant folding pass evaluating static subtrees at compile time.
  - Common Subexpression Elimination (CSE) deduplicating identical subgraphs.
  - Dead Code Elimination (DCE) removing unreachable operations.
  - Elementwise operator fusion pass chaining contiguous operations into single loops.
  - Liveness-based interval graph coloring `MemoryPlanner` reducing peak memory by 80–95%.
- Execution Runtime & Code Generation:
  - Pure C++ kernel generator synthesizing standalone C++20 shared libraries.
  - Sandboxed compilation via safe `fork`/`execve` invoking system `g++`/`clang++`.
  - Persistent SHA-256 kernel disk cache bypassing compilation on subsequent runs.
  - AVX2 256-bit SIMD vectorization with scalar tail loops and portable scalar fallbacks.
  - Node-by-node execution profiler (`Profiler`).
- Python Interface & Tooling:
  - `tinytensor` package via pybind11 with zero-copy NumPy interoperability.
  - Stochastic Gradient Descent (`SGD`) optimizer with momentum.
  - 6 runnable examples demonstrating basic tensors, GEMM, autodiff, compiler passes, custom IR, and neural network training.
  - Complete 6-benchmark suite comparing performance against NumPy and PyTorch CPU.
  - Automated bounded verification test runner (`scripts/run_all_tests.py`).
