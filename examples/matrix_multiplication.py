#!/usr/bin/env python3
"""Example 2: Matrix Multiplication and Batched Matmul in TinyTensor.

Demonstrates:
- 2D matrix multiplication via operator '@' and tt.matmul
- Transposed matrix multiplication with zero-copy views
- Batched 3D matrix multiplication
- Direct numerical verification against NumPy
- Execution latency check
"""

import time

import numpy as np
import tinytensor as tt


def main():
    print("============================================================")
    print("TinyTensor Example 2: Matrix Multiplication & Batched GEMM")
    print("============================================================\n")

    # 1. 2D Standard Matrix Multiplication
    print("--- 1. 2D Matrix Multiplication (M=4, K=3, N=5) ---")
    np.random.seed(42)
    a_np = np.random.randn(4, 3).astype(np.float32)
    b_np = np.random.randn(3, 5).astype(np.float32)

    a = tt.tensor(a_np)
    b = tt.tensor(b_np)

    c = a @ b
    c_np = a_np @ b_np

    max_diff = np.max(np.abs(c.numpy() - c_np))
    print(f"Matrix A shape: {a.shape}")
    print(f"Matrix B shape: {b.shape}")
    print(f"Result C shape: {c.shape}")
    print(f"Max absolute discrepancy vs NumPy: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Discrepancy too large: {max_diff}"
    print("Verification: PASSED\n")

    # 2. Transposed Matmul (A @ B.T)
    print("--- 2. Transposed Views Matmul (A @ B.T) ---")
    b2_np = np.random.randn(5, 3).astype(np.float32)
    b2 = tt.tensor(b2_np)

    # b2.T is a non-contiguous view with swapped strides
    c_transposed = a @ b2.T
    expected_transposed = a_np @ b2_np.T

    max_diff_t = np.max(np.abs(c_transposed.numpy() - expected_transposed))
    print(f"Matrix A shape:    {a.shape}")
    print(f"Matrix B2.T shape: {b2.T.shape} (strides: {b2.T.strides})")
    print(f"Result shape:      {c_transposed.shape}")
    print(f"Max discrepancy vs NumPy: {max_diff_t:.2e}")
    assert max_diff_t < 1e-5, f"Discrepancy too large: {max_diff_t}"
    print("Verification: PASSED\n")

    # 3. Batched 3D Matrix Multiplication
    print("--- 3. Batched Matrix Multiplication (Batch=4, M=16, K=8, N=12) ---")
    batch_a_np = np.random.randn(4, 16, 8).astype(np.float32)
    batch_b_np = np.random.randn(4, 8, 12).astype(np.float32)

    batch_a = tt.tensor(batch_a_np)
    batch_b = tt.tensor(batch_b_np)

    batch_c = batch_a @ batch_b
    expected_batch_c = batch_a_np @ batch_b_np

    max_diff_batch = np.max(np.abs(batch_c.numpy() - expected_batch_c))
    print(f"Batch A shape: {batch_a.shape}")
    print(f"Batch B shape: {batch_b.shape}")
    print(f"Batch C shape: {batch_c.shape}")
    print(f"Max discrepancy vs NumPy: {max_diff_batch:.2e}")
    assert max_diff_batch < 1e-5, f"Discrepancy too large: {max_diff_batch}"
    print("Verification: PASSED\n")

    # 4. Larger Benchmark Size Latency Test
    print("--- 4. Execution Timing (M=256, K=256, N=256) ---")
    large_a_np = np.random.randn(256, 256).astype(np.float32)
    large_b_np = np.random.randn(256, 256).astype(np.float32)

    large_a = tt.tensor(large_a_np)
    large_b = tt.tensor(large_b_np)

    # Warmup
    _ = large_a @ large_b

    # Time TinyTensor
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = large_a @ large_b
    tt_time = (time.perf_counter() - t0) / iters

    # Time NumPy
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = large_a_np @ large_b_np
    np_time = (time.perf_counter() - t0) / iters

    print(f"TinyTensor average latency: {tt_time * 1e3:.3f} ms")
    print(f"NumPy BLAS average latency: {np_time * 1e3:.3f} ms\n")

    print("SUCCESS: Matrix multiplication verified!")


if __name__ == "__main__":
    main()
