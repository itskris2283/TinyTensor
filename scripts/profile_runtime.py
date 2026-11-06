#!/usr/bin/env python3
"""Profile TinyTensor execution runtime across multiple execution modes.

Evaluates:
- Compilation overhead
- Runtime kernel execution latency
- Node-by-node runtime profiling via tt.Profiler
"""

import os
import sys

import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(root_dir, "python"))

import tinytensor as tt


def main():
    print("============================================================")
    print("TinyTensor Runtime Profiling Tool")
    print("============================================================\n")

    # Construct representative workload
    g = tt.Graph()
    x = g.input([128, 128], dtype=tt.float32, name="X")
    w = g.input([128, 128], dtype=tt.float32, name="W")
    b = g.input([1, 128], dtype=tt.float32, name="b")

    mm = g.matmul(x, w)
    lin = g.add(mm, b)
    act = g.relu(lin)
    loss = g.reduce_mean(act)
    g.mark_output(loss)

    x_np = np.random.randn(128, 128).astype(np.float32)
    w_np = np.random.randn(128, 128).astype(np.float32)
    b_np = np.random.randn(1, 128).astype(np.float32)

    feed = {x: x_np, w: w_np, b: b_np}

    modes = ["avx2", "scalar", "interpreted"]
    for mode in modes:
        print(f"--- Profiling Mode: {mode.upper()} ---")
        prog = tt.compile(g, outputs=[loss], mode=mode, optimize=True)

        prof = tt.Profiler()
        # Execute with profiler
        _ = prog.run(feed, profiler=prof)

        print(f"Total recorded runtime: {prof.total_time_us:.2f} microseconds")
        print(f"Summary: {prof.summary()}\n")

    print("SUCCESS: Runtime profiling complete!")


if __name__ == "__main__":
    main()
