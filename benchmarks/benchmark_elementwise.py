#!/usr/bin/env python3
"""Benchmark 1: Elementwise Arithmetic & Non-linearities.

Compares TinyTensor (Compiled AVX2, Compiled Scalar, Interpreted) against
NumPy and PyTorch CPU across sizes 10^2, 10^3, 10^4, 10^5, 10^6.
"""

import time

import numpy as np
import tinytensor as tt
import torch


def benchmark_size(size: int, num_trials: int = 5, warmup: int = 2):
    np.random.seed(42)
    a_np = np.random.randn(size).astype(np.float32)
    b_np = np.random.randn(size).astype(np.float32)

    # PyTorch setup
    a_pt = torch.from_numpy(a_np)
    b_pt = torch.from_numpy(b_np)

    # TinyTensor Graph setup
    g = tt.Graph()
    na = g.input([size], dtype=tt.float32, name="A")
    nb = g.input([size], dtype=tt.float32, name="B")
    # Expression: relu(a + b)
    sum_node = g.add(na, nb)
    out_node = g.relu(sum_node)
    g.mark_output(out_node)

    # Compile programs
    prog_avx2 = tt.compile(g, outputs=[out_node], mode="avx2", optimize=True)
    prog_scalar = tt.compile(g, outputs=[out_node], mode="scalar", optimize=True)
    prog_interp = tt.compile(g, outputs=[out_node], mode="interpreted", optimize=False)

    feed = {na: a_np, nb: b_np}

    # 1. Warmups
    for _ in range(warmup):
        _ = np.maximum(0.0, a_np + b_np)
        _ = torch.relu(a_pt + b_pt)
        _ = prog_avx2.run(feed)
        _ = prog_scalar.run(feed)
        if size <= 100000:
            _ = prog_interp.run(feed)

    results = {}

    # 2. Benchmark NumPy
    times_numpy = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = np.maximum(0.0, a_np + b_np)
        times_numpy.append(time.perf_counter() - t0)
    results["numpy"] = times_numpy

    # 3. Benchmark PyTorch CPU
    times_torch = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = torch.relu(a_pt + b_pt)
        times_torch.append(time.perf_counter() - t0)
    results["torch"] = times_torch

    # 4. Benchmark TinyTensor AVX2
    times_avx2 = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = prog_avx2.run(feed)
        times_avx2.append(time.perf_counter() - t0)
    results["tt_avx2"] = times_avx2

    # 5. Benchmark TinyTensor Scalar
    times_scalar = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = prog_scalar.run(feed)
        times_scalar.append(time.perf_counter() - t0)
    results["tt_scalar"] = times_scalar

    # 6. Benchmark TinyTensor Interpreted
    if size <= 100000:
        times_interp = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            _ = prog_interp.run(feed)
            times_interp.append(time.perf_counter() - t0)
        results["tt_interp"] = times_interp

    return results


def run_all(sizes=None, trials=5):
    if sizes is None:
        sizes = [100, 1000, 10000, 100000, 1000000]

    all_data = {}
    print(
        f"{'Size':<10} | {'AVX2 P50 (ms)':<14} | {'Scalar P50 (ms)':<16} | {'NumPy P50 (ms)':<15} | {'PyTorch P50 (ms)':<16}"
    )
    print("-" * 80)
    for s in sizes:
        res = benchmark_size(s, num_trials=trials)
        all_data[s] = res
        avx2_p50 = np.median(res["tt_avx2"]) * 1e3
        scalar_p50 = np.median(res["tt_scalar"]) * 1e3
        np_p50 = np.median(res["numpy"]) * 1e3
        pt_p50 = np.median(res["torch"]) * 1e3
        print(
            f"{s:<10} | {avx2_p50:<14.4f} | {scalar_p50:<16.4f} | {np_p50:<15.4f} | {pt_p50:<16.4f}"
        )
    return all_data


if __name__ == "__main__":
    print("=== TinyTensor Elementwise Benchmark ===")
    run_all()
