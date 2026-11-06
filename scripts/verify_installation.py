#!/usr/bin/env python3
"""Quick Installation & Smoke Verification Script for TinyTensor.

Verifies:
- Python module importation
- C++ shared extension functionality
- Basic tensor creation and memory allocation
- Compilation and execution of a fused graph
- Autodiff gradient correctness
"""

import sys

import numpy as np

try:
    import tinytensor as tt
except ImportError as e:
    print(f"FAILED to import tinytensor: {e}")
    sys.exit(1)


def main():
    print("============================================================")
    print("TinyTensor Smoke Verification")
    print("============================================================\n")

    print(f"TinyTensor version: {tt.__version__}")
    print(f"AVX2 hardware acceleration detected: {tt.has_avx2()}")

    # 1. Eager Tensor Smoke Test
    print("--> 1. Testing Eager Tensor Creation & Strides...")
    t = tt.zeros([2, 4], dtype=tt.float32)
    assert t.shape == [2, 4], f"Bad shape: {t.shape}"
    assert t.numel == 8, f"Bad numel: {t.numel}"
    assert t.is_contiguous, "Should be contiguous"

    tr = t.transpose(0, 1)
    assert tr.shape == [4, 2], f"Bad transposed shape: {tr.shape}"
    print("  [OK] Tensor creation and transposition verified.")

    # 2. Graph & Compiler Smoke Test
    print("--> 2. Testing Graph Construction, Compilation & Execution...")
    g = tt.Graph()
    x = g.input([4], dtype=tt.float32, name="x")
    w = g.constant([2.0, 3.0, 4.0, 5.0], name="w")
    b = g.constant(10.0, name="b")

    # y = relu(x * w + b)
    xw = g.mul(x, w)
    lin = g.add(xw, b)
    y = g.relu(lin)
    g.mark_output(y)

    prog = tt.compile(g, outputs=[y], mode="avx2", optimize=True)

    test_input = np.array([-10.0, 0.0, 2.0, 4.0], dtype=np.float32)
    res = prog.run({x: test_input})
    out = (res if isinstance(res, tt.Tensor) else res[0]).numpy()

    # Expected: relu([-10*2+10, 0*3+10, 2*4+10, 4*5+10]) = relu([-10, 10, 18, 30]) = [0, 10, 18, 30]
    expected = np.array([0.0, 10.0, 18.0, 30.0], dtype=np.float32)
    np.testing.assert_allclose(out, expected)
    print("  [OK] Graph execution & AVX2 kernel verified.")

    # 3. Autodiff Smoke Test
    print("--> 3. Testing Reverse-Mode Autodiff Engine...")
    g2 = tt.Graph()
    v = g2.input([2], dtype=tt.float32, name="v")
    sq = g2.mul(v, v)
    l = g2.reduce_sum(sq)
    g2.mark_output(l)

    grads = tt.gradients(g2, l, {v: [3.0, 5.0]})
    grad_v = grads[v].numpy()
    # d/dv (v^2) = 2v -> [6.0, 10.0]
    np.testing.assert_allclose(grad_v, [6.0, 10.0])
    print("  [OK] Autodiff gradients verified.")

    print("\n------------------------------------------------------------")
    print("ALL SMOKE TESTS PASSED - TINYTENSOR IS FULLY FUNCTIONAL!")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()
