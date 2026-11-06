# TinyTensor Benchmarking Methodology and Performance Evaluation

## 1. Benchmarking Rigor and Methodology

To ensure scientifically valid and reproducible results, the benchmarking suite follows strict experimental controls:
- **Warmup Iterations:** 2 warmup cycles execute before measurement to warm the OS disk cache, populate processor L1/L2/L3 instruction and data caches, and initialize dynamic libraries.
- **Trial Repetitions:** 5 independent trials are measured for every tensor size and operator.
- **Statistical Metrics:** For every configuration, TinyTensor records:
  - Median ($P_{50}$): Primary robust measure of central tendency.
  - $P_{90}, P_{95}, P_{99}$: Tail latency metrics reflecting worst-case execution jitter.
  - Mean ($\mu$) and Standard Deviation ($\sigma$): Gauging runtime stability.
- **CPU Affinity & Thread Limits Disentanglement:**
  - Standard mathematical benchmarks can be contaminated if external libraries silently spawn background worker threads while the target compiler runs single-threaded.
  - To prevent this, TinyTensor explicitly sets `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`.
  - Furthermore, on Linux systems, the process is bound strictly to physical CPU Core 0 via `os.sched_setaffinity(0, {0})`, eliminating OS thread migration and cache-thrashing noise across cores.
- **Fair Baseline Comparisons:**
  - NumPy: OpenBLAS single-threaded routines.
  - PyTorch CPU: ATen library with single-threaded vectorized execution.
- **Artifact Generation:** Raw samples exported to CSV (`benchmarks/results/raw_samples.csv`), summary statistics exported to JSON (`benchmarks/results/benchmark_summary.json`), and 9 publication-grade plots saved to `benchmarks/plots/` and `artifacts/phase3/benchmarks/`.

## 2. Benchmark Suite Components

1. **Elementwise Arithmetic (`benchmark_elementwise.py`):** Evaluates `relu(a + b)` across input sizes $10^2$ to $10^6$. Measures AVX2 SIMD, scalar fallback, interpreter, NumPy, and PyTorch CPU.
2. **Matrix Multiplication (`benchmark_matmul.py`):** Evaluates GEMM across square matrices from $32 \times 32$ to $512 \times 512$ with cache-blocked tiling ($32 \times 32 \times 32$).
3. **Reductions (`benchmark_reductions.py`):** Measures full sum and axis-specific sum reductions across 2D tensors up to $2000 \times 2000$.
4. **Operator Fusion (`benchmark_fusion.py`):** Directly compares fused single-loop kernels against unfused multi-pass kernels for $Y = \text{relu}(A + B)$, demonstrating up to $1.22\times$ speedup through elimination of intermediate memory traffic.
5. **Memory Planning (`benchmark_memory.py`):** Quantifies peak heap memory across deep MLP graphs (5 to 20 layers), demonstrating 86.7% to 96.7% reduction in peak buffer pool allocation.
6. **Automatic Differentiation (`benchmark_autodiff.py`):** Benchmarks reverse-mode autodiff gradient calculation against PyTorch autograd across batch sizes $16$ to $1024$.
7. **Compilation vs Runtime Breakdown:** Measures cold-start compilation overhead (g++) versus steady-state warm execution latency across workloads.
8. **FMA vs Non-FMA Error Distribution:** Quantifies relative error and ULP differences across monotonic summation vs subtractive cancellation.
9. **Multi-Process Cache Scaling:** Measures total wall-clock time across 1, 2, 4, 8, 16 concurrent worker processes using the POSIX file-locked persistent cache.

## 3. Publication Visualizations (9 Generated Artifacts)
- `elementwise_latency_vs_size.png`: Scaling across element sizes $10^2$ to $10^6$ on log-log coordinates.
- `matmul_comparison.png`: GEMM latency across dimensions $32 \times 32$ to $512 \times 512$.
- `operator_fusion_speedup.png`: Fused vs unfused kernel execution comparison.
- `memory_savings_by_layer.png`: MemoryPlanner buffer reduction across 5, 10, 20 layer MLPs.
- `reductions_comparison.png`: Reductions performance across matrix dimensions.
- `autodiff_throughput.png`: Reverse-mode autodiff throughput (samples/sec) across batch sizes.
- `compilation_vs_runtime_breakdown.png`: Cold start compilation latency vs warm runtime execution.
- `fma_vs_nofma_relative_error_distribution.png`: Empirical error distributions under monotonic vs cancelling workloads.
- `multi_process_cache_scaling.png`: Scaling of concurrent worker processes compiling into shared cache.
