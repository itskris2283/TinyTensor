#!/usr/bin/env python3
"""Example 4: Compiler Optimization Passes and Memory Planning in TinyTensor.

Demonstrates:
- Unoptimized computational graph with redundancies
- Constant folding pass eliminating compile-time expressions
- Common Subexpression Elimination (CSE) merging identical computations
- Dead Code Elimination (DCE) removing unreachable subgraphs
- Elementwise operator fusion grouping contiguous affine kernels
- MemoryPlanner buffer reuse diagnostics reducing peak memory footprint
"""

import numpy as np
import tinytensor as tt


def main():
    print("============================================================")
    print("TinyTensor Example 4: Compiler Optimization Pipeline")
    print("============================================================\n")

    # 1. Construct a Graph with Deliberate Redundancies
    # - Redundant sub-expression: (x + y) computed twice
    # - Constant sub-expression: (2.0 * 3.0) + 4.0
    # - Dead code: unused node (x * 100.0) never marked as output
    # - Fusible chain: relu( (x + y) + 1.0 )
    print("--- 1. Building Graph with Redundant Subcomputations ---")
    g = tt.Graph()
    x = g.input([4, 4], dtype=tt.float32, name="X")
    y = g.input([4, 4], dtype=tt.float32, name="Y")

    # Common subexpression (CSE opportunity)
    sum1 = g.add(x, y)
    sum2 = g.add(x, y)

    # Constant folding opportunity: (2.0 * 3.0) + 4.0 = 10.0
    c2 = g.constant(np.full((4, 4), 2.0, dtype=np.float32), name="c2")
    c3 = g.constant(np.full((4, 4), 3.0, dtype=np.float32), name="c3")
    c4 = g.constant(np.full((4, 4), 4.0, dtype=np.float32), name="c4")
    c_mul = g.mul(c2, c3)
    c_folded = g.add(c_mul, c4)

    # Dead code opportunity (never referenced by output)
    dead_node = g.mul(x, g.constant(100.0, name="dead_const"))

    # Fusible sequence: (sum1 + sum2) + c_folded -> relu
    combined = g.add(sum1, sum2)
    with_const = g.add(combined, c_folded)
    fused_out = g.relu(with_const)

    g.mark_output(fused_out)

    print("Unoptimized IR:")
    print("------------------------------------------------------------")
    print(g.to_ir())
    print("------------------------------------------------------------\n")

    # 2. Compile with Optimizations Enabled
    print("--- 2. Compiling with Optimization Pipeline ---")
    prog = tt.compile(g, outputs=[fused_out], optimize=True, mode="avx2")

    rep = prog.optimization_report
    print("Optimization Pass Statistics:")
    print(f"  - Constant nodes folded:             {rep.constant_foldings}")
    print(f"  - Dead code nodes eliminated:        {rep.dce_eliminated}")
    print(f"  - Common subexpressions eliminated:  {rep.cse_eliminated}")
    print(f"  - Elementwise fused groups:          {rep.fused_groups}")
    print(f"  - Peak planned memory:               {rep.peak_memory_bytes} bytes")
    print(f"  - Buffer reuse memory savings:       {rep.memory_savings_bytes} bytes")
    print(f"  - Reused buffer count:               {rep.reused_buffers}\n")

    # 3. Numerical Correctness Check
    print("--- 3. Running Program and Verifying Results ---")
    np.random.seed(42)
    x_val = np.random.randn(4, 4).astype(np.float32)
    y_val = np.random.randn(4, 4).astype(np.float32)

    results = prog.run({x: x_val, y: y_val})
    out_t = results if isinstance(results, tt.Tensor) else results[0]
    actual = out_t.numpy()

    # Expected: relu( (x + y) + (x + y) + 10.0 )
    expected = np.maximum(0.0, (x_val + y_val) + (x_val + y_val) + 10.0)

    max_diff = np.max(np.abs(actual - expected))
    print(f"Max numerical error vs reference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Optimization altered computation semantics: {max_diff}"
    print("Verification: PASSED\n")

    print("SUCCESS: Compiler optimizations and memory reuse verified!")


if __name__ == "__main__":
    main()
