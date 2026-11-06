#!/usr/bin/env python3
"""Benchmark 2: Matrix Multiplication (GEMM).

Compares TinyTensor Compiled vs NumPy (BLAS) vs PyTorch CPU
across matrix sizes 32x32, 64x64, 128x128, 256x256, 512x512.
"""

import time

import numpy as np
import tinytensor as tt
import torch


def benchmark_matmul_size(dim: int, num_trials: int = 5, warmup: int = 2):
    np.random.seed(42)
    a_np = np.random.randn(dim, dim).astype(np.float32)
    b_np = np.random.randn(dim, dim).astype(np.float32)

    a_pt = torch.from_numpy(a_np)
    b_pt = torch.from_numpy(b_np)

    # TinyTensor Graph
    g = tt.Graph()
    na = g.input([dim, dim], dtype=tt.float32, name="A")
    nb = g.input([dim, dim], dtype=tt.float32, name="B")
    out = g.matmul(na, nb)
    g.mark_output(out)

    prog = tt.compile(g, outputs=[out], mode="avx2", optimize=True)
    feed = {na: a_np, nb: b_np}

    # Warmup
    for _ in range(warmup):
        _ = a_np @ b_np
        _ = a_pt @ b_pt
        _ = prog.run(feed)

    results = {}

    # NumPy
    times_numpy = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = a_np @ b_np
        times_numpy.append(time.perf_counter() - t0)
    results["numpy"] = times_numpy

    # PyTorch CPU
    times_torch = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = a_pt @ b_pt
        times_torch.append(time.perf_counter() - t0)
    results["torch"] = times_torch

    # TinyTensor Compiled
    times_tt = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = prog.run(feed)
        times_tt.append(time.perf_counter() - t0)
    results["tt_compiled"] = times_tt

    return results


def run_all(dims=None, trials=5):
    if dims is None:
        dims = [32, 64, 128, 256, 512]

    all_data = {}
    print(
        f"{'Matrix Size':<12} | {'TinyTensor P50 (ms)':<20} | {'NumPy P50 (ms)':<16} | {'PyTorch P50 (ms)':<16}"
    )
    print("-" * 72)
    for d in dims:
        res = benchmark_matmul_size(d, num_trials=trials)
        all_data[d] = res
        tt_p50 = np.median(res["tt_compiled"]) * 1e3
        np_p50 = np.median(res["numpy"]) * 1e3
        pt_p50 = np.median(res["torch"]) * 1e3
        print(f"{f'{d}x{d}':<12} | {tt_p50:<20.4f} | {np_p50:<16.4f} | {pt_p50:<16.4f}")
    return all_data


if __name__ == "__main__":
    print("=== TinyTensor MatMul Benchmark ===")
    run_all()
