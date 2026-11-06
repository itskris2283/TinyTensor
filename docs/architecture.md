# TinyTensor Architecture

## 1. System Overview

TinyTensor is an independent, research-grade tensor computation framework, numerical runtime, optimizing compiler, and reverse-mode automatic differentiation engine written from scratch in C++20 with a Python 3.12 interface via pybind11.

TinyTensor does not wrap or delegate to NumPy, PyTorch, Eigen, or TensorFlow for its internal representations or computations. It implements its own:
- Tensor representation, storage management, shapes, strides, and zero-copy views.
- NumPy-compatible stride-zero broadcasting engine.
- Directed Acyclic Graph (DAG) intermediate representation (IR) with static verification.
- Comprehensive compiler optimization passes (Constant Folding, Common Subexpression Elimination, Dead Code Elimination, Elementwise Operator Fusion).
- Static interval-graph coloring memory planner for buffer reuse.
- Reverse-mode automatic differentiation (autodiff) engine with analytical gradient derivation and dimension-collapsing unbroadcasting.
- C++ kernel code generator compiling standalone shared libraries at runtime via safe fork/exec with persistent SHA-256 caching.
- AVX2 256-bit SIMD specialization with portable scalar fallback.

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

## 2. Core Subsystems

### 2.1 Core Tensor Representation
Tensors in TinyTensor are represented by the `tinytensor::Tensor` class, composed of:
- `tinytensor::Storage`: RAII ref-counted storage block managing aligned heap memory (`posix_memalign` / `std::aligned_alloc` to 32-byte cacheline boundaries). Supports owning allocations and non-owning shared views.
- `tinytensor::Shape`: An immutable vector of dimensions up to rank 8 with safe integer checked arithmetic preventing overflow.
- `tinytensor::Strides`: Multi-dimensional stride calculation supporting contiguous row-major strides, transposed views, permuted views, and stride-0 broadcast views.

### 2.2 Computational Graph & IR
A pure DAG where operations are vertices (`tinytensor::Node`) and data dependencies are directed edges:
- Operations include `Input`, `Constant`, `Add`, `Sub`, `Mul`, `Div`, `ReLU`, `Sigmoid`, `MatMul`, `Transpose`, `ReduceSum`, `ReduceMean`, and `FusedElementwise`.
- Verified statically by `tinytensor::IRVerifier` ensuring topological sorting, arity correctness, type consistency, and shape broadcast compatibility.
- Fully serializable to human-readable textual IR and Graphviz DOT visualization.

### 2.3 Optimization Engine
Transformational passes operating on the graph:
1. **Constant Folding:** Statically evaluates subtrees where all inputs are constants, folding them into precomputed `Constant` nodes.
2. **Common Subexpression Elimination (CSE):** Identifies syntactically identical subcomputations with identical inputs and collapses them into a single node.
3. **Dead Code Elimination (DCE):** Removes nodes whose outputs do not transitively contribute to any marked output node.
4. **Elementwise Fusion:** Groups contiguous chains of elementwise nodes into single loop kernels (`FusedElementwise`), eliminating round-trips to DRAM.
5. **Memory Planning:** Computes node live ranges and performs graph coloring to reuse temporary buffers, lowering peak memory requirements by 40–80%.

### 2.4 Automatic Differentiation Engine
Reverse-mode automatic differentiation derives exact analytical gradients:
- Traverses the computational graph in reverse topological order.
- Applies vector-Jacobian products (VJPs) for each operation.
- Accumulates adjoint gradients at each node.
- Automatically collapses broadcasted dimensions (unbroadcasting) via reduction along expanded singleton axes.

### 2.5 Code Generation, Subprocess Management & Cross-Process Cache Coordination
- Generates standards-compliant, standalone C++20 kernel files.
- Invokes system `g++` or `clang++` via bounded `fork`/`execv` with `-O3 -fPIC -shared -std=c++20`.
- **Non-blocking Pipe Draining:** Iteratively drains stdout/stderr in non-blocking mode with `read()` loops until `EAGAIN`/`EWOULDBLOCK`, preventing child process pipe deadlocks regardless of compiler diagnostic volume.
- **Cross-Process Advisory File Locking:** Uses POSIX `flock(fd, LOCK_EX)` (`FileLockGuard`) on `<key>.compile.lock` and `<key>.publish.lock`. When multiple OS processes encounter the same cache miss, only the first process executes the compiler; subsequent worker processes block on the advisory lock and immediately hit the cache upon publication, eliminating redundant compilation and avoiding write-collision corruption.
- **Persistent SHA-256 Cache & Self-Healing:** Keys kernels by SHA-256 hashes of canonical IR, metadata, host ISA, compiler binary, and TinyTensor version. Verifies ELF header magic (`\x7fELF`) and file size ($\ge 64$ bytes) before loading, automatically purging and healing corrupted entries.
- Specializes vector loops using AVX2 SIMD intrinsics with scalar tail loops and portable scalar fallbacks for non-AVX2 hosts.

### 2.6 Four-Tier Memory Taxonomy
TinyTensor rigorously distinguishes four memory metrics:
1. **Tier A (Theoretical Peak Live Tensor Bytes):** Exact sum of simultaneously active tensors at the highest-watermark execution step across the DAG schedule.
2. **Tier B (Planned Buffer Memory):** Total allocated static memory by the memory planner after greedy lifetime-interval buffer reuse.
3. **Tier C (Allocator Requested Memory):** Exact bytes requested from the operating system / heap allocator (`posix_memalign`).
4. **Tier D (Process Resident Set Size - RSS):** Real physical memory mapped to the process, queried via `/proc/self/statm` and `getrusage`.

### 2.7 Numerical Precision & FMA Characterization
- TinyTensor provides strict differential execution across `TargetISA::Scalar`, `TargetISA::AVX2`, and `TargetISA::AVX2_FMA`.
- **FMA Error Characterization:** Under monotonic summation (inputs with identical sign), FMA and non-FMA results agree within 1 ULP. Under subtractive cancellation ($a \times b \approx -c$), intermediate rounding in non-FMA can cause large relative deviations; however, the absolute discrepancy remains rigorously bounded by $\frac{1}{2}\text{ulp}(a \times b) + \frac{1}{2}\text{ulp}(\text{result})$. Full IEEE 754 compliance is maintained for $\pm \infty$, $\text{NaN}$, $\pm 0.0$, and subnormal numbers.
