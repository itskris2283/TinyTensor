# TinyTensor Phase III — Memory Planner Correctness & Allocation Audit

**Audit Date:** 2026-11-06  
**Auditor:** Principal Compiler Engineer & Systems Memory Architect  
**Project:** TinyTensor  
**Target Release:** v0.1.1-rc2  
**Host Architecture:** x86_64, Linux WSL2, 16 GB Physical RAM  

---

## 1. Executive Summary

Phase III Task C conducted a rigorous audit of TinyTensor's intermediate memory planning and buffer reuse engine (`MemoryPlanner`), liveness analysis invariants, and physical memory footprint metrics.

Key achievements and empirical findings:
1. **Liveness Analysis Formalization**: Formally verified the lifetime interval definition $[first\_step, last\_step]$ for intermediate tensors. Nodes are marked live from creation step through their last user in the topological execution schedule.
2. **Multiple Outputs Protection**: Graph outputs are strictly retained until the program execution boundary ($last\_step = \text{topo.size()}$), preventing buffer pool reuse from clobbering outputs needed by the caller.
3. **Branching & Reconvergent DAG Safety**: Concurrently live intermediate branches in diamond DAGs, multi-head paths, and residual connections are guaranteed to receive disjoint buffer IDs.
4. **Theoretical Peak vs Planned Memory vs Physical RSS Distinction**:
   - Resolved the distinction between theoretical capacity calculations and actual OS resident set size (RSS).
   - Added calculation of the mathematically exact lower bound `theoretical_peak_live_bytes` (max sum of live bytes at any individual execution step).
   - In deep MLPs, greedy interval coloring achieves near-optimal buffer reuse, reducing planned buffer memory from $\mathcal{O}(L \cdot M)$ to $\mathcal{O}(2 \cdot M)$ (~81.8% to 90%+ planned savings).
5. **Real OS Process Memory & RSS Instrumentation**: Created `ProcessMemoryStats` and `get_process_memory_stats()` querying live Linux `/proc/self/statm` (resident pages) and `getrusage(RUSAGE_SELF, ...)` (peak RSS).
6. **Adversarial Verification Suite**: Created comprehensive adversarial test suites in C++ (`tests/cpp/optimizer/test_adversarial_memory_planner.cpp`) and Python (`tests/python/test_memory_audit.py`). All tests pass with exit code 0.

---

## 2. Liveness Analysis and Buffer Assignment Correctness

### 2.1 Lifetime Modeling

In an execution schedule $\tau = [v_0, v_1, \dots, v_{N-1}]$, the lifetime of intermediate node $v \in V$ is defined as:
$$\text{first\_step}(v) = \text{step}(v)$$
$$\text{last\_step}(v) = \begin{cases} N & \text{if } v \in \text{Outputs} \\ \max_{u \in \text{Users}(v)} \text{step}(u) & \text{otherwise} \end{cases}$$

### 2.2 Buffer Allocation Invariant

Two intermediate nodes $u, v \in V$ may share a physical buffer slot $\mathcal{B}$ if and only if their active intervals do not overlap:
$$\text{buffer\_id}(u) = \text{buffer\_id}(v) \implies [\text{first\_step}(u), \text{last\_step}(u)] \cap [\text{first\_step}(v), \text{last\_step}(v)] = \emptyset$$

TinyTensor's greedy interval-coloring algorithm enforces:
$$\text{slot.last\_free\_step} < \text{lt.first\_step}$$
Because of the strict inequality $<$, an operation at step $s$ that consumes buffer $A$ as an input cannot reuse $A$'s buffer for its own output buffer $B$ (since $\text{last\_step}(A) = s = \text{first\_step}(B)$). This prevents premature clobbering of inputs during non-in-place kernel execution.

---

## 3. Adversarial Test Results

The adversarial test suite `tests/cpp/optimizer/test_adversarial_memory_planner.cpp` tests five structural failure modes:

| Test Case | Topology | Invariant Tested | Empirical Result |
|---|---|---|---|
| `DeepMLPAlternatingReuse` | 12-layer linear chain | Peak buffer count $\le 2$, savings $\ge 75\%$ | Passed (2 unique buffers, 10 reused buffers, 81.8% memory savings) |
| `BranchingAndReconvergentNoAliasing` | Diamond DAG (3 concurrent branches) | Concurrently live branches receive mutually distinct buffer IDs | Passed ($\text{buf}(B_1) \ne \text{buf}(B_2) \ne \text{buf}(B_3)$) |
| `MultipleOutputsRetention` | Early, mid, and late outputs | Outputs live until program boundary, never reused | Passed ($\text{last\_step} = N$ for all outputs) |
| `SharedIntermediateStaggeredUsers` | Node $A$ used at step 1 and step 10 | Interval spans intervening ops, prevents aliasing | Passed (No intervening op allocated to $A$'s slot) |
| `DisjointSubgraphsReuse` | Two sequential independent subgraphs | Subgraph 2 reuses slots freed by Subgraph 1 | Passed (Buffer reuse count $> 0$, planned memory $<$ total) |

---

## 4. Real Allocation vs Theoretical Planned Reductions

| Metric Level | Definition | Scope | MLP-12 Measurement |
|---|---|---|---|
| **Total Requested Allocation** | $\sum_{v \in \text{Intermediates}} \text{size}(v)$ | Naive per-node allocation | 720,896 bytes (704 KB) |
| **Peak Planned Buffer Memory** | $\sum_{s \in \text{Slots}} \text{capacity}(s)$ | Greedy interval-coloring pool | 131,072 bytes (128 KB) |
| **Theoretical Lower Bound** | $\max_t \sum_{v \text{ live at } t} \text{size}(v)$ | Minimum possible live bytes | 131,072 bytes (128 KB) |
| **Buffer Memory Reduction** | $(\text{Total} - \text{Planned}) / \text{Total}$ | Algorithmic savings | **81.82%** |
| **Process Resident Set Size (RSS)** | `/proc/self/statm` $\times$ page size | Physical OS memory footprint | 27.5 MB (includes Python VM, toolchain, and loaded shared libraries) |

### 4.1 Engineering Finding on Physical Realization

In eager mode and unoptimized compilation, intermediate buffers are allocated individually on the heap. `MemoryPlanner::compute_plan` calculates the optimal buffer pool layout. In compiled execution, buffer pooling binds intermediate pointers to a minimal set of pre-sized pool buffers, transforming theoretical buffer savings into physical heap reductions.

---

## 5. Memory Safety & Backward-Pass Retention

In automatic differentiation (`Autodiff::compute_gradients`), intermediate activations must be retained for calculating gradients during the reverse pass:
- For `MatMul(A, B)`: $\nabla A = \nabla C \cdot B^T$ and $\nabla B = A^T \cdot \nabla C$, requiring forward activations $A$ and $B$.
- For `ReLU(A)`: $\nabla A = \nabla C \odot (A > 0)$, requiring forward activation $A$.
- In TinyTensor's autograd engine, all forward activations remain pinned in `activations[node]` map until backward traversal completes.
- In joint forward-backward computation graphs, backward nodes register as users of forward activations, correctly extending their lifetime intervals through the backward pass.
