#!/usr/bin/env python3
"""Benchmark 4: Operator Fusion Performance.

Compares Fused vs Unfused operator chains:
Expression: Y = relu((A + B) * C - D)
Demonstrating cache locality improvements and elimination of intermediate memory traffic.
"""

import time

import numpy as np
import tinytensor as tt


def benchmark_fusion_size(size: int, num_trials: int = 5):
    np.random.seed(42)
    a_np = np.random.randn(size).astype(np.float32)
    b_np = np.random.randn(size).astype(np.float32)
    c_np = np.random.randn(size).astype(np.float32)
    d_np = np.random.randn(size).astype(np.float32)

    # 1. Unfused Graph
    g_unfused = tt.Graph()
    na1 = g_unfused.input([size], dtype=tt.float32, name="A")
    nb1 = g_unfused.input([size], dtype=tt.float32, name="B")
    nc1 = g_unfused.input([size], dtype=tt.float32, name="C")
    nd1 = g_unfused.input([size], dtype=tt.float32, name="D")

    add1 = g_unfused.add(na1, nb1)
    mul1 = g_unfused.mul(add1, nc1)
    sub1 = g_unfused.sub(mul1, nd1)
    out1 = g_unfused.relu(sub1)
    g_unfused.mark_output(out1)

    prog_unfused = tt.compile(g_unfused, outputs=[out1], mode="avx2", optimize=False)

    # 2. Fused Graph (Optimized)
    g_fused = tt.Graph()
    na2 = g_fused.input([size], dtype=tt.float32, name="A")
    nb2 = g_fused.input([size], dtype=tt.float32, name="B")
    nc2 = g_fused.input([size], dtype=tt.float32, name="C")
    nd2 = g_fused.input([size], dtype=tt.float32, name="D")

    add2 = g_fused.add(na2, nb2)
    mul2 = g_fused.mul(add2, nc2)
    sub2 = g_fused.sub(mul2, nd2)
    out2 = g_fused.relu(sub2)
    g_fused.mark_output(out2)

    prog_fused = tt.compile(g_fused, outputs=[out2], mode="avx2", optimize=True)

    feed_unfused = {na1: a_np, nb1: b_np, nc1: c_np, nd1: d_np}
    feed_fused = {na2: a_np, nb2: b_np, nc2: c_np, nd2: d_np}

    # Warmup
    _ = prog_unfused.run(feed_unfused)
    _ = prog_fused.run(feed_fused)

    # NumPy benchmark
    times_numpy = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = np.maximum(0.0, (a_np + b_np) * c_np - d_np)
        times_numpy.append(time.perf_counter() - t0)

    # Unfused benchmark
    times_unfused = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = prog_unfused.run(feed_unfused)
        times_unfused.append(time.perf_counter() - t0)

    # Fused benchmark
    times_fused = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = prog_fused.run(feed_fused)
        times_fused.append(time.perf_counter() - t0)

    return {
        "numpy": times_numpy,
        "unfused": times_unfused,
        "fused": times_fused,
        "report": prog_fused.optimization_report,
    }


def run_all(sizes=None, trials=5):
    if sizes is None:
        sizes = [1000, 10000, 100000, 1000000]

    all_data = {}
    print(
        f"{'Size':<10} | {'Fused P50 (ms)':<16} | {'Unfused P50 (ms)':<18} | {'NumPy P50 (ms)':<16} | {'Speedup':<10}"
    )
    print("-" * 78)
    for s in sizes:
        res = benchmark_fusion_size(s, num_trials=trials)
        all_data[s] = res
        fused_p50 = np.median(res["fused"]) * 1e3
        unfused_p50 = np.median(res["unfused"]) * 1e3
        np_p50 = np.median(res["numpy"]) * 1e3
        speedup = unfused_p50 / fused_p50 if fused_p50 > 0 else 1.0
        print(
            f"{s:<10} | {fused_p50:<16.4f} | {unfused_p50:<18.4f} | {np_p50:<16.4f} | {speedup:<10.2f}x"
        )
    return all_data


if __name__ == "__main__":
    print("=== TinyTensor Operator Fusion Benchmark ===")
    run_all()
