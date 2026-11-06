#!/usr/bin/env python3
"""Benchmark 6: Reverse-Mode Automatic Differentiation Performance.

Compares TinyTensor Reverse-Mode Autodiff against PyTorch CPU Autograd.
Measures backward pass latency and gradient derivation overhead for deep composite expressions.
"""

import time

import numpy as np
import tinytensor as tt
import torch


def benchmark_autodiff_batch(batch_size: int, d_in: int = 16, d_out: int = 8, num_trials: int = 5):
    np.random.seed(42)
    x_np = np.random.randn(batch_size, d_in).astype(np.float32)
    w_np = np.random.randn(d_in, d_out).astype(np.float32)
    b_np = np.random.randn(1, d_out).astype(np.float32)
    y_np = np.random.randn(batch_size, d_out).astype(np.float32)

    # 1. PyTorch Autograd Setup
    x_pt = torch.tensor(x_np, requires_grad=False)
    w_pt = torch.tensor(w_np, requires_grad=True)
    b_pt = torch.tensor(b_np, requires_grad=True)
    y_pt = torch.tensor(y_np, requires_grad=False)

    # Warmup PyTorch
    pred_pt = torch.relu(x_pt @ w_pt + b_pt)
    loss_pt = torch.sum((pred_pt - y_pt) ** 2)
    loss_pt.backward()

    times_torch = []
    for _ in range(num_trials):
        w_pt.grad = None
        b_pt.grad = None
        pred_pt = torch.relu(x_pt @ w_pt + b_pt)
        loss_pt = torch.sum((pred_pt - y_pt) ** 2)
        t0 = time.perf_counter()
        loss_pt.backward()
        times_torch.append(time.perf_counter() - t0)

    # 2. TinyTensor Autodiff Setup
    g = tt.Graph()
    x_node = g.input([batch_size, d_in], dtype=tt.float32, name="X")
    w_node = g.input([d_in, d_out], dtype=tt.float32, name="W")
    b_node = g.input([1, d_out], dtype=tt.float32, name="B")
    y_node = g.input([batch_size, d_out], dtype=tt.float32, name="Y")

    mm = g.matmul(x_node, w_node)
    lin = g.add(mm, b_node)
    act = g.relu(lin)
    diff = g.sub(act, y_node)
    sq = g.mul(diff, diff)
    loss_node = g.reduce_sum(sq)
    g.mark_output(loss_node)

    feed = {x_node: x_np, w_node: w_np, b_node: b_np, y_node: y_np}

    # Warmup TinyTensor
    _ = tt.gradients(g, loss_node, feed)

    times_tt = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = tt.gradients(g, loss_node, feed)
        times_tt.append(time.perf_counter() - t0)

    return {"torch": times_torch, "tinytensor": times_tt}


def run_all():
    print("=== TinyTensor Reverse-Mode Autodiff Benchmark ===")
    print(
        f"{'Batch Size':<12} | {'TinyTensor P50 (ms)':<22} | {'PyTorch P50 (ms)':<18} | {'Relative Ratio':<15}"
    )
    print("-" * 75)
    batches = [16, 64, 256, 1024]
    all_data = {}
    for b in batches:
        res = benchmark_autodiff_batch(b)
        all_data[b] = res
        tt_p50 = np.median(res["tinytensor"]) * 1e3
        pt_p50 = np.median(res["torch"]) * 1e3
        ratio = tt_p50 / pt_p50 if pt_p50 > 0 else 1.0
        print(f"{b:<12} | {tt_p50:<22.4f} | {pt_p50:<18.4f} | {ratio:<14.2f}x")
    return all_data


if __name__ == "__main__":
    run_all()
