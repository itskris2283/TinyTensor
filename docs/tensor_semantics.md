# TinyTensor Tensor Semantics

## 1. Mathematical and Structural Definition

A Tensor $\mathcal{T} = (\mathcal{S}, \mathbf{s}, \mathbf{\sigma}, \delta, \tau)$ in TinyTensor is defined by:
- $\mathcal{S}$: Underlying flat memory storage buffer of raw bytes.
- $\mathbf{s} = (d_0, d_1, \dots, d_{k-1}) \in \mathbb{N}^k$: Shape tuple of rank $k$ ($0 \le k \le 8$).
- $\mathbf{\sigma} = (\sigma_0, \sigma_1, \dots, \sigma_{k-1}) \in \mathbb{Z}^k$: Strides tuple measured in elements.
- $\delta \in \mathbb{N}$: Storage offset (first element index relative to buffer start).
- $\tau \in \{\text{float32}, \text{float64}\}$: Data type specification.

### Memory Offset Equation
For any multi-index $(i_0, i_1, \dots, i_{k-1})$ where $0 \le i_j < d_j$, the physical element location is given by:
$$\text{Offset}(i_0, \dots, i_{k-1}) = \delta + \sum_{j=0}^{k-1} i_j \cdot \sigma_j$$

## 2. Contiguity Semantics

A tensor is defined as **C-contiguous** (row-major) if and only if:
$$\sigma_{k-1} = 1 \quad \text{and} \quad \sigma_j = \sigma_{j+1} \cdot d_{j+1} \quad \forall j \in [0, k-2]$$
For a 0D scalar tensor ($k = 0$), the stride is empty and it is trivially contiguous.

## 3. Metadata Views vs. Data Copies

TinyTensor strictly avoids copying tensor data whenever an operation can be expressed as a linear re-indexing of existing storage:

### 3.1 Transpose & Permutations
Transposing two dimensions $i$ and $j$ on tensor $T$ constructs a new tensor $T'$ with:
$$d'_i = d_j, \quad d'_j = d_i, \quad \sigma'_i = \sigma_j, \quad \sigma'_j = \sigma_i$$
No bytes are allocated or copied. Both $T$ and $T'$ share the same `std::shared_ptr<Storage>` reference-counted memory buffer.

### 3.2 Slicing and Views
A view slices along dimension $j$ starting at offset $s_j$:
$$\delta' = \delta + s_j \cdot \sigma_j$$
The underlying storage pointer remains identical, ensuring zero allocation overhead.

### 3.3 Contiguous Coalescing (`contiguous()`)
When an operation requires linear memory layout (such as direct SIMD vectorized streaming or cache-aligned matrix multiplication), calling `.contiguous()` on a non-contiguous view evaluates whether $\mathcal{T}$ is already contiguous. If true, it returns `*this`; if false, it allocates a new contiguous buffer and packs the strided elements sequentially.

## 4. Broadcasting Mechanics

TinyTensor implements full NumPy broadcasting semantics based on the following invariant:

Two shapes $\mathbf{A} = (a_0, \dots, a_{m-1})$ and $\mathbf{B} = (b_0, \dots, b_{n-1})$ are broadcast-compatible if, starting from trailing dimensions:
1. $a_i == b_i$, or
2. Either $a_i == 1$ or $b_i == 1$, or
3. One tensor has fewer dimensions, in which case singleton dimensions of size 1 are prepended.

### Stride-Zero Virtual Expansion
Rather than physically replicating data along expanded axes, TinyTensor sets the stride of any broadcasted dimension $j$ where $d_j = 1 \to d'_j > 1$ to:
$$\sigma'_j = 0$$
Because $i_j \cdot 0 = 0$ for all index values $i_j \in [0, d'_j - 1]$, any read along dimension $j$ continuously reads the same element without allocating additional memory.
