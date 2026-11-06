# TinyTensor Compiler Pipeline

## 1. High-Level Compilation Pipeline

TinyTensor converts high-level mathematical expressions into optimized native machine code through a deterministic multi-stage pipeline:

```
[ User Python / C++ Code ]
           |
           v
[ Computation Graph (DAG IR) ]
           |
           v
[ IR Verifier ] --------------------> Throws IRError if ill-formed
           |
           v
[ Optimization Engine ]
   ├── Constant Folding
   ├── Common Subexpression Elimination (CSE)
   ├── Dead Code Elimination (DCE)
   └── Elementwise Operator Fusion
           |
           v
[ Memory Planner ] -----------------> Liveness Analysis & Buffer Reuse
           |
           v
[ C++ Code Generator ]
   ├── Loop Nest Generation
   ├── AVX2 SIMD Specialization
   └── Stride-Zero Address Computations
           |
           v
[ Kernel Cache Check ] -------------> Cache Hit? Return dlopen handle
           | Cache Miss
           v
[ Safe Subprocess Compiler ] -------> fork/exec g++ -O3 -shared -fPIC
           |
           v
[ Dynamic Shared Object (.so) ] ----> Persistent Cache (~/.cache/tinytensor/kernels)
           |
           v
[ CompiledProgram Executable ] -----> dlopen() & dlsym() function pointer
```

## 2. Compilation Stages

### Stage 1: IR Verification
`tinytensor::IRVerifier::verify(graph)` performs static semantic checking:
- Validates topological DAG ordering (no cycles).
- Enforces arity correctness for all nodes.
- Checks data type uniformity.
- Verifies broadcast compatibility and matrix multiplication inner dimension alignment ($K_A == K_B$).

### Stage 2: Target-Independent Optimization Passes
The optimizer executes passes until convergence:
1. **Constant Folding:** Replaces static constant subgraphs with precomputed `Constant` nodes.
2. **Common Subexpression Elimination:** Deduplicates duplicate computations.
3. **Dead Code Elimination:** Prunes unreachable nodes.
4. **Elementwise Fusion:** Groups chains of affine operations into single multi-operation loop bodies.

### Stage 3: Memory Planning
Analyzes the live ranges of all intermediate tensors. Identifies tensors with non-overlapping lifetimes and assigns them shared buffer slots, minimizing heap allocations and footprint.

### Stage 4: C++ Kernel Code Synthesis
`CppBackend` translates the optimized IR into clean, standalone C++20 source code:
- Generates aligned function signatures taking raw data pointers (`const float*`, `float*`).
- Generates SIMD vectorized inner loops using AVX2 intrinsics (`_mm256_loadu_ps`, `_mm256_fmadd_ps`, `_mm256_storeu_ps`).
- Emits portable scalar tail loops for remaining elements.
- Implements general multi-index loops for reductions and transpositions.

### Stage 5: Out-of-Process Compilation
The C++ source is compiled using the system compiler (`g++` or `clang++`):
- Invoked via safe `fork`/`execve` in a distinct process group.
- Compiled with `-O3 -fPIC -shared -std=c++20 -march=native`.
- Bounded execution timeout prevents hangs.

### Stage 6: Persistent Two-Tier Kernel Cache
Compiled kernels are cached in memory and on disk (`~/.cache/tinytensor/kernels/`):
- Key: SHA-256 hash of the generated C++ source code, compiler flags, and architecture identifier.
- Cache hit: Bypasses compilation entirely, loading the `.so` via `dlopen()` in under 1 millisecond.
