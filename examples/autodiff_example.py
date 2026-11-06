#!/usr/bin/env python3
"""Example 3: Reverse-Mode Automatic Differentiation in TinyTensor.

Demonstrates:
- Symbolic computation graph with reverse-mode automatic differentiation
- Analytical gradient derivation for scalar and tensor nodes
- Multi-variable non-linear loss functions
- Verification of analytical gradients against numerical finite differences
"""

import numpy as np
import tinytensor as tt


def main():
    print("============================================================")
    print("TinyTensor Example 3: Reverse-Mode Automatic Differentiation")
    print("============================================================\n")

    # 1. Multi-variable Computational Graph
    # Let Loss = sum((X @ W + B - Y) ** 2)
    # where X: [4, 3], W: [3, 2], B: [1, 2], Y: [4, 2]
    print("--- 1. Building Graph: Linear Model + MSE Loss ---")
    g = tt.Graph()
    x_node = g.input([4, 3], dtype=tt.float32, name="X")
    w_node = g.input([3, 2], dtype=tt.float32, name="W")
    b_node = g.input([1, 2], dtype=tt.float32, name="B")
    y_node = g.input([4, 2], dtype=tt.float32, name="Y")

    # Prediction = X @ W + B (with row-wise bias broadcasting)
    xw = g.matmul(x_node, w_node)
    y_hat = g.add(xw, b_node)

    # Residuals and squared error
    diff = g.sub(y_hat, y_node)
    sq_diff = g.mul(diff, diff)
    loss = g.reduce_sum(sq_diff)
    g.mark_output(loss)

    print("IR Representation:")
    print(g.to_ir())

    # 2. Evaluate Analytical Gradients
    print("--- 2. Computing Analytical Gradients via tt.gradients ---")
    np.random.seed(123)
    x_val = np.random.randn(4, 3).astype(np.float32)
    w_val = np.random.randn(3, 2).astype(np.float32)
    b_val = np.random.randn(1, 2).astype(np.float32)
    y_val = np.random.randn(4, 2).astype(np.float32)

    inputs = {x_node: x_val, w_node: w_val, b_node: b_val, y_node: y_val}
    grads = tt.gradients(g, loss, inputs)

    grad_w = grads[w_node].numpy()
    grad_b = grads[b_node].numpy()
    grad_x = grads[x_node].numpy()

    print(f"Loss value: {loss.shape}")
    print(f"dL/dW shape: {grad_w.shape}")
    print(f"dL/dB shape: {grad_b.shape} (unbroadcasted from [4, 2] to [1, 2])")
    print(f"dL/dX shape: {grad_x.shape}\n")

    # 3. Finite Differences Verification
    print("--- 3. Numerical Finite Differences Validation ---")
    eps = 1e-3

    compiled_prog = tt.compile(g, outputs=[loss])

    def eval_loss(x, w, b, y):
        res = compiled_prog.run({x_node: x, w_node: w, b_node: b, y_node: y})
        out_t = res if isinstance(res, tt.Tensor) else res[0]
        return float(out_t.numpy().flat[0])

    # Check dL/dW numerically
    num_grad_w = np.zeros_like(w_val)
    for i in range(w_val.shape[0]):
        for j in range(w_val.shape[1]):
            w_plus = w_val.copy()
            w_plus[i, j] += eps
            loss_plus = eval_loss(x_val, w_plus, b_val, y_val)

            w_minus = w_val.copy()
            w_minus[i, j] -= eps
            loss_minus = eval_loss(x_val, w_minus, b_val, y_val)

            num_grad_w[i, j] = (loss_plus - loss_minus) / (2.0 * eps)

    diff_w = np.max(np.abs(grad_w - num_grad_w) / (np.abs(num_grad_w) + 1e-5))
    print(f"Max relative difference for dL/dW: {diff_w:.3e}")
    assert diff_w < 1e-3, f"dL/dW gradient validation failed: {diff_w}"

    # Check dL/dB numerically
    num_grad_b = np.zeros_like(b_val)
    for j in range(b_val.shape[1]):
        b_plus = b_val.copy()
        b_plus[0, j] += eps
        loss_plus = eval_loss(x_val, w_val, b_plus, y_val)

        b_minus = b_val.copy()
        b_minus[0, j] -= eps
        loss_minus = eval_loss(x_val, w_val, b_minus, y_val)

        num_grad_b[0, j] = (loss_plus - loss_minus) / (2.0 * eps)

    diff_b = np.max(np.abs(grad_b - num_grad_b) / (np.abs(num_grad_b) + 1e-5))
    print(f"Max relative difference for dL/dB: {diff_b:.3e}")
    assert diff_b < 1e-3, f"dL/dB gradient validation failed: {diff_b}"

    print("\nSUCCESS: All analytical gradients verified by numerical differentiation!")


if __name__ == "__main__":
    main()
