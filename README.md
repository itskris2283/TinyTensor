# TinyTensor

[![CI](https://github.com/tinytensor/tinytensor/actions/workflows/ci.yml/badge.svg)](https://github.com/tinytensor/tinytensor/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Coverage](https://img.shields.io/badge/C%2B%2B%20Coverage-93%25-brightgreen.svg)](docs/audit/phase3_release_readiness.md)
[![Python Coverage](https://img.shields.io/badge/Python%20Coverage-98%25-brightgreen.svg)](docs/audit/phase3_release_readiness.md)
[![Sanitizers](https://img.shields.io/badge/Sanitizers-ASan%20%2B%20UBSan%20Clean-success.svg)](artifacts/phase3/sanitizers/asan_report.txt)

TinyTensor is an independent, research-grade tensor computation framework, numerical runtime, optimizing compiler, and reverse-mode automatic differentiation engine written from scratch in C++20 with a Python 3.12 interface via pybind11.

TinyTensor does **not** wrap or call NumPy, PyTorch, Eigen, or TensorFlow for its internal representations or computations. The entire tensor storage layer, strided view engine, computation graph, optimization passes, autodiff logic, and JIT C++ / AVX2 kernel compiler are built completely independently.

---

## Key Features

- **Tensor Core:** RAII memory management, 32-byte cacheline alignment, non-contiguous strided views, $O(1)$ zero-copy transpositions, and NumPy-compatible stride-zero broadcasting.
- **Computational Graph (DAG IR):** Static Single Assignment (SSA) intermediate representation with strict static verification, text serialization, and Graphviz DOT visualization.
- **Compiler Optimizations:** Constant Folding, Common Subexpression Elimination (CSE), Dead Code Elimination (DCE), Elementwise Operator Fusion, and Interval-Coloring Memory Planning.
- **Reverse-Mode Autodiff:** Exact analytical gradient derivation via vector-Jacobian products (VJPs) with automatic unbroadcasting reductions, verified against finite differences.
- **C++ Codegen & Execution Runtime:** Synthesizes standalone C++20 kernels compiled out-of-process via safe `fork`/`execve` with persistent two-tier SHA-256 caching and ELF magic validation.
- **Compiler Diagnostic Introspection:** Full transparency exposing `original_ir()`, `optimized_ir()`, `generated_source()`, and `execution_plan()`.
- **SIMD Acceleration & GEMM Optimization:** Cache-blocked loop tiling ($32 \times 32 \times 32$) and AVX2 FMA vectorization (2.1x–9.4x GEMM speedup), vectorized fused elementwise kernels (1.07x–1.39x speedup), CPUID/XCR0-detected portable scalar fallback.
- **Rigorous Verification:** 195 C++ GoogleTests, 135 Python pytests, 330 total automated tests, 93.2% C++ line coverage, 98.5% C++ function coverage, 53.7% C++ branch coverage (with exhaustive unwinding branch audit), 98.0% Python statement coverage, zero compiler warnings under `-Wall -Wextra -Wpedantic -Werror`, clean Ruff linting, and 100% clean under AddressSanitizer and UndefinedBehaviorSanitizer across 3 consecutive clean test cycles.

---

## Architecture Overview

```
                    +--------------------------------+
                    |       Python User Code         |
                    | (tinytensor.Tensor, tt.compile)|
                    +---------------+----------------+
                                    | pybind11
                                    v
+-----------------------------------------------------------------------------+
|                          TinyTensor C++20 Core                              |
|                                                                             |
|  +--------------------+     +---------------------+     +-----------------+ |
|  |    Tensor Core     |     | Computational Graph |     | Reverse Autodiff| |
|  | - Storage & Memory |     | - Node & Graph      |     | - Gradient Rules| |
|  | - Shape & Strides  |     | - IR Verifier       |     | - Unbroadcasting| |
|  | - View Transforms  |     | - Textual / DOT IR  |     | - Adjoint Eval  | |
|  +---------+----------+     +----------+----------+     +--------+--------+ |
|            |                           |                         |          |
|            +---------------------------+-------------------------+          |
|                                        v                                    |
|                      +----------------------------------+                   |
|                      |        Compiler Optimizer        |                   |
|                      |  - Constant Folding Pass         |                   |
|                      |  - Common Subexpr Elimination    |                   |
|                      |  - Dead Code Elimination         |                   |
|                      |  - Elementwise Operator Fusion   |                   |
|                      |  - Interval Memory Planner       |                   |
|                      +-----------------+----------------+                   |
|                                        |                                    |
|                                        v                                    |
|                      +----------------------------------+                   |
|                      |      Code Generator & Runtime    |                   |
|                      |  - C++ Backend Kernel Generator  |                   |
|                      |  - AVX2 8-wide SIMD Kernels      |                   |
|                      |  - Portable Scalar Fallback      |                   |
|                      |  - Subprocess Compiler Fork/Exec |                   |
|                      |  - Persistent SHA-256 Cache      |                   |
|                      +-----------------+----------------+                   |
|                                        |                                    |
+----------------------------------------|------------------------------------+
                                         v
                         +-------------------------------+
                         |   Dynamic Shared Object (.so) |
                         |   dlopen() -> Function Pointer|
                         +-------------------------------+
```

---

## Quick Start

### 1. Eager Tensor Operations & Strides

```python
import tinytensor as tt

# Create tensors
a = tt.tensor([[10.0], [20.0], [30.0]], dtype=tt.float32)  # Shape (3, 1)
b = tt.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=tt.float32)  # Shape (1, 4)

# Broadcasting addition: (3, 1) + (1, 4) -> (3, 4)
c = a + b
print(c.shape)  # [3, 4]
print(c.numpy())

# Non-copying transposed view
tr = c.T
print(tr.shape)  # [4, 3]
print(tr.is_contiguous)  # False (O(1) zero-copy view with swapped strides)
```

### 2. Graph Compilation & Operator Fusion

```python
import tinytensor as tt

g = tt.Graph()
x = g.input([1000], dtype=tt.float32, name="x")
w = g.constant(2.5, name="w")
b = g.constant(1.0, name="b")

# Build computation graph: relu(x * w + b)
xw = g.mul(x, w)
lin = g.add(xw, b)
out = g.relu(lin)
g.mark_output(out)

# Compile with AVX2 SIMD specialization & operator fusion
prog = tt.compile(g, outputs=[out], mode="avx2", optimize=True)

# Run compiled program
result = prog.run({x: [1.0, -2.0, 3.0, ...]})

# Inspect compiler diagnostics & generated C++ source code
print(prog.original_ir)  # Unoptimized SSA graph
print(prog.optimized_ir)  # Post-fusion SSA graph
print(prog.generated_source)  # Synthesized C++20 kernel source
print(prog.execution_plan)  # Buffer allocation & dispatch schedule
```

### 3. Reverse-Mode Automatic Differentiation

```python
import tinytensor as tt

g = tt.Graph()
w = g.input([3, 2], dtype=tt.float32, name="W")
x = g.input([4, 3], dtype=tt.float32, name="X")
y = g.input([4, 2], dtype=tt.float32, name="Y")

# Forward pass
y_pred = g.matmul(x, w)
diff = g.sub(y_pred, y)
loss = g.reduce_sum(g.mul(diff, diff))
g.mark_output(loss)

# Analytical gradients via reverse-mode autodiff
grads = tt.gradients(g, loss, {w: w_val, x: x_val, y: y_val})
print(grads[w].shape)  # [3, 2] exact analytical dL/dW
```

---

## Performance & Benchmarks

TinyTensor was benchmarked across 50 timed iterations with warmup under single-thread pinning (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) against NumPy (OpenBLAS) and PyTorch CPU (ATen):

- **Operator Fusion:** **$1.07\times$ to $1.39\times$ speedup** over unfused sequential execution by keeping intermediate values in CPU registers, eliminating intermediate DRAM roundtrips.
- **Large-Tensor Throughput:** At $1,000,000$ elements, TinyTensor AVX2 elementwise addition executes in **$1.762\text{ ms}$ (p50)**, outperforming NumPy's **$1.848\text{ ms}$**.
- **GEMM Acceleration:** Cache-blocked tiling ($32 \times 32 \times 32$) with AVX2 FMA vectorization achieves **$2.1\times$ to $9.4\times$ speedup** over unoptimized baseline code across matrix dimensions.
- **Memory Planning:** Up to **$96.7\%$ reduction** in peak resident heap buffer allocation across multi-layer deep neural networks via interval liveness graph coloring.

Artifacts and publication-grade summary tables are available in `benchmarks/results/` and `docs/audit/`.

---

## Repository Structure

```text
tinytensor/
├── CMakeLists.txt            # Root CMake build configuration
├── pyproject.toml            # Python packaging and tool configuration
├── include/tinytensor/       # Public C++20 API headers
│   ├── core/                 # Storage, Shape, Strides, Tensor, Operators
│   ├── graph/                # Node, Graph
│   ├── ir/                   # IRVerifier, IRPrinter
│   ├── autodiff/             # Reverse-mode Autodiff & Gradient VJPs
│   ├── optimizer/            # Constant Folding, CSE, DCE, Fusion, MemoryPlanner
│   ├── runtime/              # Executor, Interpreter, Executable, Profiler
│   ├── codegen/              # CppBackend, AVX2Backend, KernelCache
│   └── utils/                # CheckedMath, Errors
├── src/                      # C++ implementation files
├── python/tinytensor/        # Python module interface (pybind11)
├── bindings/                 # pybind11 binding source
├── tests/
│   ├── cpp/                  # 118 C++ unit tests (GoogleTest)
│   └── python/               # 114 Python unit tests (pytest)
├── examples/                 # 6 runnable end-to-end Python examples
├── benchmarks/               # 6 comprehensive benchmarks with plotting
├── scripts/                  # Bounded test runners and verification scripts
└── docs/                     # Full architecture, semantics, audits & guides
```

---

## License

TinyTensor is open-source software licensed under the [Apache License 2.0](LICENSE).
