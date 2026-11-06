#!/usr/bin/env python3
"""Example 6: Multi-Layer Perceptron (MLP) Neural Network Training with TinyTensor.

Demonstrates:
- 2-layer Neural Network (Input: 4 -> Hidden: 8 -> Output: 1)
- Forward pass with Matrix Multiplication, Bias addition, and ReLU activation
- Mean Squared Error (MSE) loss computation
- Reverse-mode automatic differentiation computing parameter gradients
- SGD optimizer with momentum updating parameters in-place
- Multi-epoch training loop demonstrating loss convergence
"""

import numpy as np
import tinytensor as tt


def main():
    print("============================================================")
    print("TinyTensor Example 6: 2-Layer Neural Network Training")
    print("============================================================\n")

    # 1. Generate Synthetic Dataset
    # y = 2.0*x0 - 1.5*x1 + 0.5*x2 + 3.0
    np.random.seed(42)
    n_samples = 64
    d_in = 4
    d_hidden = 8
    d_out = 1

    x_data = np.random.randn(n_samples, d_in).astype(np.float32)
    true_w = np.array([[2.0], [-1.5], [0.5], [0.0]], dtype=np.float32)
    y_data = (x_data @ true_w + 3.0).astype(np.float32)

    # 2. Initialize Model Parameters as TinyTensor Tensors
    w1_val = np.random.randn(d_in, d_hidden).astype(np.float32) * 0.1
    b1_val = np.zeros((1, d_hidden), dtype=np.float32)
    w2_val = np.random.randn(d_hidden, d_out).astype(np.float32) * 0.1
    b2_val = np.zeros((1, d_out), dtype=np.float32)

    t_w1 = tt.tensor(w1_val)
    t_b1 = tt.tensor(b1_val)
    t_w2 = tt.tensor(w2_val)
    t_b2 = tt.tensor(b2_val)

    params = [t_w1, t_b1, t_w2, t_b2]
    optimizer = tt.SGD(params, lr=0.005, momentum=0.5)

    # 3. Build Static Computation Graph
    g = tt.Graph()
    x_node = g.input([n_samples, d_in], dtype=tt.float32, name="X")
    y_node = g.input([n_samples, d_out], dtype=tt.float32, name="Y")

    w1_node = g.input([d_in, d_hidden], dtype=tt.float32, name="W1")
    b1_node = g.input([1, d_hidden], dtype=tt.float32, name="b1")
    w2_node = g.input([d_hidden, d_out], dtype=tt.float32, name="W2")
    b2_node = g.input([1, d_out], dtype=tt.float32, name="b2")

    # Layer 1: Linear + ReLU
    z1 = g.matmul(x_node, w1_node)
    z1_b = g.add(z1, b1_node)
    a1 = g.relu(z1_b)

    # Layer 2: Linear
    z2 = g.matmul(a1, w2_node)
    y_pred = g.add(z2, b2_node)

    # MSE Loss: mean((y_pred - y)**2)
    err = g.sub(y_pred, y_node)
    sq_err = g.mul(err, err)
    loss_node = g.reduce_mean(sq_err)
    g.mark_output(loss_node)

    # Compile the graph for high-speed forward execution
    prog = tt.compile(g, outputs=[loss_node], mode="avx2", optimize=True)

    print(f"Model: Linear({d_in} -> {d_hidden}) -> ReLU -> Linear({d_hidden} -> {d_out})")
    print(f"Dataset: {n_samples} samples, {d_in} features")
    print("Optimizer: SGD (learning_rate=0.01, momentum=0.9)\n")

    # 4. Training Loop
    epochs = 25
    initial_loss = 0.0
    final_loss = 0.0

    print("--- Starting Training Loop ---")
    for epoch in range(epochs):
        feed_dict = {
            x_node: x_data,
            y_node: y_data,
            w1_node: t_w1,
            b1_node: t_b1,
            w2_node: t_w2,
            b2_node: t_b2,
        }

        # Forward pass
        res = prog.run(feed_dict)
        out_t = res if isinstance(res, tt.Tensor) else res[0]
        loss_val = float(out_t.numpy().flat[0])
        if epoch == 0:
            initial_loss = loss_val

        # Backward pass (Analytical Autodiff)
        grads_dict = tt.gradients(g, loss_node, feed_dict)
        grads_list = [
            grads_dict[w1_node],
            grads_dict[b1_node],
            grads_dict[w2_node],
            grads_dict[b2_node],
        ]

        # Optimizer update step
        optimizer.step(grads_list)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:2d}/{epochs:2d} | MSE Loss: {loss_val:.6f}")
        final_loss = loss_val

    print("\n--- Training Results ---")
    print(f"Initial Loss: {initial_loss:.6f}")
    print(f"Final Loss:   {final_loss:.6f}")
    assert final_loss < initial_loss, "Training did not decrease loss!"
    print(f"Loss Reduction: {(initial_loss - final_loss) / initial_loss * 100.0:.2f}%")

    print("\nSUCCESS: Neural network training verified!")


if __name__ == "__main__":
    main()
