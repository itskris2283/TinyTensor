#!/usr/bin/env python3
"""Example 5: Custom Computational Graph Construction, IR Verification, and DOT Visualization.

Demonstrates:
- Low-level symbolic computation graph construction
- Static IR type and shape verification
- Textual intermediate representation (IR) export
- Graphviz DOT representation export for visual debugging
- End-to-end compilation and execution
"""

import os

import tinytensor as tt


def main():
    print("============================================================")
    print("TinyTensor Example 5: Custom Graph Construction & IR Export")
    print("============================================================\n")

    # 1. Construct Custom Computation Graph
    print("--- 1. Building Custom Symbolic Graph ---")
    g = tt.Graph()

    # Input nodes with shapes and types
    a = g.input([2, 4], dtype=tt.float32, name="A")
    w = g.input([4, 3], dtype=tt.float32, name="W")
    bias = g.constant([0.5, -0.5, 1.0], name="Bias")  # Shape [3]

    # Operations: Matmul -> Add -> Sigmoid -> ReduceSum
    matmul_node = g.matmul(a, w)
    biased_node = g.add(matmul_node, bias)
    act_node = g.sigmoid(biased_node)
    total_sum = g.reduce_sum(act_node, axes=[0, 1], keepdims=False)

    # Mark graph outputs
    g.mark_output(total_sum)
    g.mark_output(act_node)

    # 2. Textual IR Inspection
    print("--- 2. Exporting Human-Readable Textual IR ---")
    ir_text = g.to_ir()
    print(ir_text)

    # 3. Graphviz DOT Export
    print("--- 3. Exporting Graphviz DOT Graph ---")
    dot_text = g.to_dot()
    dot_path = os.path.join(os.path.dirname(__file__), "custom_graph.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_text)
    print(f"Graphviz DOT written to: {dot_path}")
    print("Sample DOT snippet (first 10 lines):")
    for line in dot_text.strip().split("\n")[:10]:
        print(f"  {line}")
    print("  ...\n")

    # 4. Compilation and Execution
    print("--- 4. Executing Custom Graph ---")
    prog = tt.compile(g, outputs=[total_sum, act_node], mode="avx2")

    a_data = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    w_data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]

    outputs = prog.run({a: a_data, w: w_data})
    sum_val = outputs[0].numpy()
    act_val = outputs[1].numpy()

    print(f"Activation tensor shape: {act_val.shape}")
    print(f"Activation values:\n{act_val}")
    print(f"Scalar sum value: {sum_val}")

    print("\nSUCCESS: Custom graph constructed, verified, exported and executed!")


if __name__ == "__main__":
    main()
