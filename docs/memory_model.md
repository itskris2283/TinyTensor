# TinyTensor Memory Model and Allocation Architecture

## 1. RAII Resource Ownership

All memory allocations in TinyTensor adhere strictly to Resource Acquisition Is Initialization (RAII):
- `tinytensor::Storage` manages heap memory using aligned allocation:
  - Linux/POSIX: `posix_memalign` aligned to 32 bytes (256-bit AVX2 vector register alignment).
  - Deallocation uses `std::free` in the storage destructor.
- Zero risk of memory leaks, double-free, or uninitialized pointer access.
- Non-owning tensors (e.g. external data views) wrap foreign memory with an explicit non-owning flag, ensuring the foreign buffer is never freed on tensor destruction.

## 2. Resource Limits & Checked Arithmetic

To defend against heap exploitation, integer overflow, and system out-of-memory crashes:
- Maximum tensor rank: 8 dimensions.
- Maximum allocation per tensor: 4 GB.
- Maximum element count: $2^{31} - 1$.
- Dimension bounds: $0 \le d_i \le 2^{31} - 1$.
All byte-size and stride calculations are checked via `checked_multiply`:
```cpp
int64_t checked_multiply(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > 0 && b > 0 && a > (std::numeric_limits<int64_t>::max() / b)) {
        throw ResourceLimitError("Integer overflow during dimension calculation");
    }
    // ...
    return a * b;
}
```

## 3. Compiler Memory Planning & Buffer Reuse

In unoptimized graph execution, every intermediate operation allocates a dedicated tensor output buffer. For deep computational graphs (such as multi-layer neural networks), this leads to high peak memory usage and memory bandwidth bottlenecks.

TinyTensor's `MemoryPlanner` implements a greedy interval graph coloring algorithm:
1. **Liveness Analysis:** Traverses nodes in topological order. A node's live range $[start, end]$ begins at its evaluation step and ends at the maximum evaluation step among all of its consumer nodes.
2. **Disjoint Buffer Reuse:** When an operation finishes execution and its output is no longer needed by any subsequent downstream node, its physical buffer is returned to the pool.
3. **Linear Scan Allocation:** Assigns existing free buffers of compatible size and dtype, reusing memory addresses across disjoint liveness intervals.
4. **Peak Allocation Diagnostics:** Computes and reports `peak_bytes_unoptimized` vs `peak_bytes_optimized`, achieving 40% to 80% reduction in peak resident buffer memory.
