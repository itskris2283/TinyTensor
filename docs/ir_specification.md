# TinyTensor Intermediate Representation (IR) Specification

## 1. Overview and Design Principles

TinyTensor uses a pure Directed Acyclic Graph (DAG) intermediate representation based on Static Single Assignment (SSA) principles. Each operation in the computation graph is an immutable value producing a single output tensor with a statically inferred shape and data type.

Key properties:
- Explicit dataflow edges: Each node maintains references to its input operands (`inputs()`) and reverse edges to its consumer nodes (`users()`).
- Static shape and dtype inference: Shapes and types are checked and deduced during graph construction.
- Distinct node identities: Every node is assigned a monotonically increasing unique 64-bit ID.
- Static verification: Strict type and shape invariants verified prior to compilation.

## 2. Operation Types

| OpType | Description | Input Arity | Inferred Shape Rule | Inferred DType |
|---|---|---|---|---|
| `Input` | External input tensor provided at runtime | 0 | Explicitly specified $(d_0, \dots, d_{k-1})$ | Specified $\tau$ |
| `Constant` | Compile-time embedded tensor constant | 0 | Shape of underlying tensor | $\tau$ of tensor |
| `Add` | Elementwise addition | 2 | $\text{Broadcast}(A, B)$ | $\tau_A == \tau_B$ |
| `Sub` | Elementwise subtraction | 2 | $\text{Broadcast}(A, B)$ | $\tau_A == \tau_B$ |
| `Mul` | Elementwise multiplication | 2 | $\text{Broadcast}(A, B)$ | $\tau_A == \tau_B$ |
| `Div` | Elementwise true division (IEEE 754) | 2 | $\text{Broadcast}(A, B)$ | $\tau_A == \tau_B$ |
| `ReLU` | Elementwise rectified linear unit ($\max(0, x)$) | 1 | Same as input | $\tau_{in}$ |
| `Sigmoid` | Elementwise logistic function ($1 / (1 + e^{-x})$) | 1 | Same as input | $\tau_{in}$ |
| `MatMul` | 2D / Batched matrix multiplication | 2 | $(\dots, M, K) \times (\dots, K, N) \to (\dots, M, N)$ | $\tau_A == \tau_B$ |
| `Transpose` | Swaps two dimension axes | 1 | Permuted shape with $d_i \leftrightarrow d_j$ | $\tau_{in}$ |
| `ReduceSum` | Sum reduction over specified axes | 1 | Shape with reduced axes collapsed | $\tau_{in}$ |
| `ReduceMean` | Mean reduction over specified axes | 1 | Shape with reduced axes collapsed | $\tau_{in}$ |
| `FusedElementwise` | Fused composite elementwise kernel | $\ge 1$ | Common broadcasted shape | $\tau_{inputs}$ |

## 3. Textual IR Syntax

The textual IR is human-readable and deterministic, designed for compiler inspection and debugging:
```
graph TinyTensorGraph {
  %0 = Input [4, 4] float32 "X"
  %1 = Input [4, 4] float32 "Y"
  %2 = Add %0, %1 -> [4, 4] float32
  %3 = Constant [1] float32 = [2.0]
  %4 = Mul %2, %3 -> [4, 4] float32
  %5 = ReLU %4 -> [4, 4] float32
  output %5
}
```

## 4. Graphviz DOT Export

The IR can be serialized to Graphviz DOT format via `graph.to_dot()` or `IRPrinter::to_dot(graph)`. Nodes are rendered with distinct shapes and color styles based on their operator class:
- Inputs: Light blue rectangles (`fillcolor="#e1f5fe"`).
- Constants: Light grey ovals (`fillcolor="#f5f5f5"`).
- Operations: Pale green rounded boxes (`fillcolor="#e8f5e9"`).
- Outputs: Highlighted border (`penwidth=2.5`).
