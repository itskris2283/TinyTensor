#!/usr/bin/env python3
"""Benchmark 3: Reduction Operations (Sum and Mean).

Compares TinyTensor Compiled vs NumPy vs PyTorch CPU for full and axis-specific reductions.
"""

import time

import numpy as np
import tinytensor as tt
import torch


def benchmark_reductions_size(rows: int, cols: int, num_trials: int = 5):
    np.random.seed(42)
    a_np = np.random.randn(rows, cols).astype(np.float32)
    a_pt = torch.from_numpy(a_np)

    # TinyTensor Graph for full sum and axis reduction
    g = tt.Graph()
    na = g.input([rows, cols], dtype=tt.float32, name="A")
    out_full = g.reduce_sum(na, axes=[0, 1])
    out_axis0 = g.reduce_sum(na, axes=[0])
    out_axis1 = g.reduce_sum(na, axes=[1])
    g.mark_output(out_full)
    g.mark_output(out_axis0)
    g.mark_output(out_axis1)

    prog = tt.compile(g, outputs=[out_full, out_axis0, out_axis1], mode="avx2", optimize=True)
    feed = {na: a_np}

    # Warmup
    _ = a_np.sum()
    _ = a_pt.sum()
    _ = prog.run(feed)

    # Benchmark full sum
    times_np = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = a_np.sum()
        times_np.append(time.perf_counter() - t0)

    times_pt = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = a_pt.sum()
        times_pt.append(time.perf_counter() - t0)

    times_tt = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = prog.run(feed)
        times_tt.append(time.perf_counter() - t0)

    return {"numpy": times_np, "torch": times_pt, "tt_compiled": times_tt}


def run_all(shapes=None, trials=5):
    if shapes is None:
        shapes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]

    all_data = {}
    print(
        f"{'Shape':<14} | {'TinyTensor P50 (ms)':<20} | {'NumPy P50 (ms)':<16} | {'PyTorch P50 (ms)':<16}"
    )
    print("-" * 74)
    for r, c in shapes:
        res = benchmark_reductions_size(r, c, num_trials=trials)
        all_data[f"{r}x{c}"] = res
        tt_p50 = np.median(res["tt_compiled"]) * 1e3
        np_p50 = np.median(res["numpy"]) * 1e3
        pt_p50 = np.median(res["torch"]) * 1e3
        print(f"{f'{r}x{c}':<14} | {tt_p50:<20.4f} | {np_p50:<16.4f} | {pt_p50:<16.4f}")
    return all_data


if __name__ == "__main__":
    print("=== TinyTensor Reductions Benchmark ===")
    run_all()
