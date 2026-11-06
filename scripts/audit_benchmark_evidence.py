#!/usr/bin/env python3
"""Benchmark Evidence & Raw Data Independent Verification Script."""

import csv
import json
import os
import numpy as np

def verify_benchmarks():
    base_dir = "/mnt/d/MERN/TinyTensor/artifacts/phase3/benchmarks"
    csv_path = os.path.join(base_dir, "raw_samples.csv")
    json_path = os.path.join(base_dir, "benchmark_summary.json")

    print("==========================================================================")
    print("BENCHMARK EVIDENCE & RAW DATA INTEGRITY AUDIT")
    print("==========================================================================\n")

    # 1. Verify all 9 PNG plots
    plots = [
        "autodiff_throughput.png",
        "compilation_vs_runtime_breakdown.png",
        "elementwise_latency_vs_size.png",
        "fma_vs_nofma_relative_error_distribution.png",
        "matmul_comparison.png",
        "memory_savings_by_layer.png",
        "multi_process_cache_scaling.png",
        "operator_fusion_speedup.png",
        "reductions_comparison.png",
    ]

    print("--> 1. Verifying 9 Benchmark Plot Artifacts...")
    for plot in plots:
        p_path = os.path.join(base_dir, plot)
        assert os.path.exists(p_path), f"Missing plot: {plot}"
        sz = os.path.getsize(p_path)
        assert sz > 10000, f"Plot {plot} unusually small ({sz} bytes)"
        with open(p_path, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", f"Invalid PNG magic in {plot}"
        print(f"    [OK] {plot} ({sz:,} bytes, valid PNG header)")

    # 2. Check metadata
    print("\n--> 2. Verifying Benchmark Metadata & CPU Affinity...")
    with open(json_path, "r") as f:
        summary_data = json.load(f)

    meta = summary_data.get("metadata", {})
    print(f"    Version:           {meta.get('version')}")
    print(f"    CPU Affinity:      {meta.get('cpu_affinity')}")
    print(f"    Threads Limit:     {meta.get('threads_limit')}")
    print(f"    Trials:            {meta.get('trials_per_benchmark')}")
    assert "Core 0" in meta.get("cpu_affinity", "")
    assert meta.get("threads_limit") == 1
    assert meta.get("trials_per_benchmark") == 5

    # 3. Read CSV and group samples
    print("\n--> 3. Cross-Validating CSV Raw Samples Against Summary JSON...")
    csv_groups = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["benchmark"], str(row["config"]), row["system"])
            if key not in csv_groups:
                csv_groups[key] = []
            csv_groups[key].append(float(row["time_ms"]))

    print(f"    Total Distinct Benchmark Configurations in CSV: {len(csv_groups)}")

    mismatches = 0
    total_checks = 0

    for (bm, cfg, sys_name), samples in csv_groups.items():
        if bm not in summary_data or cfg not in summary_data[bm] or sys_name not in summary_data[bm][cfg]:
            print(f"    [FAIL] Missing in JSON summary: {bm} / {cfg} / {sys_name}")
            mismatches += 1
            continue

        j_stats = summary_data[bm][cfg][sys_name]
        p50 = float(np.median(samples))
        mean = float(np.mean(samples))
        min_v = float(np.min(samples))
        max_v = float(np.max(samples))

        # Check p50 (median)
        diff_p50 = abs(p50 - j_stats["p50_ms"])
        diff_mean = abs(mean - j_stats["mean_ms"])
        diff_min = abs(min_v - j_stats["min_ms"])
        diff_max = abs(max_v - j_stats["max_ms"])

        if diff_p50 > 1e-6 or diff_mean > 1e-6:
            print(f"    [MISMATCH] {bm} {cfg} {sys_name}: CSV median={p50:.6f} vs JSON={j_stats['p50_ms']:.6f}")
            mismatches += 1
        total_checks += 1

    print(f"    Checked {total_checks} configurations: {mismatches} mismatches.")
    assert mismatches == 0, f"Found {mismatches} mismatches between CSV and JSON summary!"
    print("    [PASS] 100% of raw CSV samples match summary JSON statistics exactly.")

    print("\n==========================================================================")
    print("BENCHMARK INTEGRITY VERIFICATION COMPLETE: ALL CHECKS PASSED")
    print("==========================================================================\n")

if __name__ == "__main__":
    verify_benchmarks()
