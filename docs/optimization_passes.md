# TinyTensor Optimization Passes

## 1. Overview

TinyTensor employs a multi-pass optimization engine designed to transform high-level computational graphs into streamlined, memory-efficient, and cache-friendly execution plans. The optimization pipeline consists of four target-independent passes and one memory layout pass:

1. **Constant Folding Pass** (`ConstantFoldingPass`)
2. **Common Subexpression Elimination Pass** (`CommonSubexpressionEliminationPass`)
3. **Dead Code Elimination Pass** (`DeadCodeEliminationPass`)
4. **Elementwise Fusion Pass** (`FusionPass`)
5. **Memory Planner Pass** (`MemoryPlanner`)

```
   [ Raw Computational Graph ]
                 │
                 ▼
    ┌─────────────────────────┐
    │  Constant Folding Pass  │ ◄── Evaluates compile-time constant subtrees
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │       CSE Pass          │ ◄── Merges duplicate syntactic computations
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │        DCE Pass         │ ◄── Prunes unused dead branches
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Elementwise Fusion     │ ◄── Chains affine elementwise ops into single loops
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Memory Planner (Reuse) │ ◄── Disjoint liveness buffer reuse
    └────────────┬────────────┘
                 │
                 ▼
  [ Optimized Graph & Buffer Plan ]
```

## 2. Pass Details

### 2.1 Constant Folding Pass
- **Objective:** Statically evaluate subtrees whose inputs are known at compile time.
- **Algorithm:**
  1. Identifies nodes whose inputs are all `Constant` nodes or compile-time constants.
  2. Uses TinyTensor's eager operator engine (`tinytensor::ops`) to evaluate the operation.
  3. Replaces the operation node with a single `Constant` node holding the precomputed result.
  4. Cascades until no further constant expressions remain.

### 2.2 Common Subexpression Elimination (CSE)
- **Objective:** Eliminate redundant computations that produce identical outputs from identical inputs.
- **Algorithm:**
  1. Computes an equivalence signature for each node: `(OpType, InputNodeIDs, Shape, DType)`.
  2. Traverses nodes in topological order.
  3. If a node matches an existing signature in the lookup map, all downstream consumers of the duplicate node are redirected to the existing node.
  4. The duplicate node becomes orphan and is subsequently cleaned up.

### 2.3 Dead Code Elimination (DCE)
- **Objective:** Remove nodes whose values do not contribute to any marked output of the graph.
- **Algorithm:**
  1. Performs reverse reachability analysis starting from the marked output nodes (`graph.outputs()`).
  2. Marks all transitively reachable nodes as `alive`.
  3. Eliminates all unvisited nodes from the graph, freeing their descriptor allocations and pruning graph size.

### 2.4 Elementwise Operator Fusion
- **Objective:** Mitigate the memory bandwidth bottleneck by fusing chains of elementwise operations (`Add`, `Sub`, `Mul`, `Div`, `ReLU`, `Sigmoid`).
- **Algorithm:**
  1. Identifies candidate producer-consumer pairs where:
     - Both nodes are elementwise operations.
     - The producer has exactly one consumer (or all consumers can share the fused loop).
     - Both operations operate on identical broadcast-compatible shapes.
  2. Fuses them into a single `FusedElementwise` operation containing a synthetic C++ expression (e.g. `relu((a + b) * c)`).
  3. Emits a single fused loop in codegen, keeping intermediate values in CPU registers rather than writing and reloading from DRAM.

### 2.5 Memory Planning Pass
- **Objective:** Minimize peak RAM consumption and eliminate runtime memory allocation overhead.
- **Algorithm:**
  1. Performs interval liveness analysis: each tensor buffer is alive from its computation step to the step of its last consumer.
  2. Builds an interference graph where edges connect simultaneously live buffers.
  3. Employs greedy graph coloring to assign memory offsets within a single pooled storage arena.
  4. Achieves 40% to 80% reduction in peak resident buffer memory.
