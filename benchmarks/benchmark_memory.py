#!/usr/bin/env python3
"""Benchmark 5: Memory Allocation & Buffer Reuse (MemoryPlanner).

Quantifies peak memory consumption with and without the MemoryPlanner pass.
Measures the reduction in total allocated bytes across a multi-layer deep network graph.
"""

import tinytensor as tt


def benchmark_memory_graph(num_layers: int = 10, layer_dim: int = 512):
    # Construct a chain of operations: x -> (op1) -> t1 -> (op2) -> t2 ... -> out
    g = tt.Graph()
    current = g.input([layer_dim, layer_dim], dtype=tt.float32, name="input")

    # Add layers with weights
    for i in range(num_layers):
        w = g.input([layer_dim, layer_dim], dtype=tt.float32, name=f"w_{i}")
        # mm: [layer_dim, layer_dim]
        mm = g.matmul(current, w)
        # bias add
        b = g.input([1, layer_dim], dtype=tt.float32, name=f"b_{i}")
        biased = g.add(mm, b)
        # relu activation
        current = g.relu(biased)

    g.mark_output(current)

    # 1. Compile with MemoryPlanner enabled
    prog_opt = tt.compile(g, outputs=[current], mode="avx2", optimize=True)
    rep = prog_opt.optimization_report

    peak_opt = rep.peak_memory_bytes
    savings_bytes = rep.memory_savings_bytes
    unopt_estimate = peak_opt + savings_bytes
    savings_pct = (savings_bytes / unopt_estimate * 100.0) if unopt_estimate > 0 else 0.0

    return {
        "num_layers": num_layers,
        "layer_dim": layer_dim,
        "peak_unoptimized_bytes": unopt_estimate,
        "peak_optimized_bytes": peak_opt,
        "savings_pct": savings_pct,
        "reused_buffers": rep.reused_buffers,
        "fused_groups": rep.fused_groups,
        "dce_eliminated": rep.dce_eliminated,
    }


def run_all():
    print("=== TinyTensor MemoryPlanner Benchmark ===")
    print(
        f"{'Layers':<8} | {'Dim':<8} | {'Unoptimized (KB)':<18} | {'Optimized (KB)':<16} | {'Reduction':<12}"
    )
    print("-" * 72)
    configs = [(5, 256), (10, 256), (10, 512), (20, 512)]
    all_data = []
    for layers, dim in configs:
        res = benchmark_memory_graph(layers, dim)
        all_data.append(res)
        unopt_kb = res["peak_unoptimized_bytes"] / 1024.0
        opt_kb = res["peak_optimized_bytes"] / 1024.0
        print(
            f"{layers:<8} | {dim:<8} | {unopt_kb:<18.1f} | {opt_kb:<16.1f} | {res['savings_pct']:<11.1f}%"
        )
    return all_data


if __name__ == "__main__":
    run_all()
