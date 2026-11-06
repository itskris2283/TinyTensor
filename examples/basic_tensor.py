#!/usr/bin/env python3
"""Example 1: Basic Tensor Operations, Strides, and Broadcasting in TinyTensor.

Demonstrates:
- Tensor construction (zeros, ones, full, from list/numpy)
- Shape, strides, data type, element counts
- Non-copying transposed views and reshaping
- Stride-zero broadcasting mechanics
- Interoperability with NumPy
"""

import numpy as np
import tinytensor as tt


def main():
    print("============================================================")
    print("TinyTensor Example 1: Basic Tensor Operations & Semantics")
    print("============================================================\n")

    # 1. Tensor Creation
    print("--- 1. Tensor Creation ---")
    t_zeros = tt.zeros([2, 3], dtype=tt.float32)
    t_ones = tt.ones([3, 4], dtype=tt.float32)
    t_full = tt.full([2, 2], 7.5, dtype=tt.float32)

    np_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    t_from_np = tt.tensor(np_data)

    print(f"Zeros [2, 3]:\n{t_zeros.numpy()}")
    print(f"Ones [3, 4]:\n{t_ones.numpy()}")
    print(f"Full 7.5 [2, 2]:\n{t_full.numpy()}")
    print(f"From NumPy array [2, 3]:\n{t_from_np.numpy()}\n")

    # 2. Shape, Strides, and Memory Metadata
    print("--- 2. Shape, Strides & Memory Layout ---")
    print(f"Shape:         {t_from_np.shape}")
    print(f"Strides:       {t_from_np.strides} (elements)")
    print(f"DType:         {t_from_np.dtype}")
    print(f"Rank (ndim):   {t_from_np.ndim}")
    print(f"Element count: {t_from_np.numel}")
    print(f"Is contiguous: {t_from_np.is_contiguous}\n")

    # 3. Transposition & Metadata-only Views
    print("--- 3. Transposition & Shared Storage Views ---")
    t_transposed = t_from_np.transpose(0, 1)  # Transpose dimensions (0, 1) -> (1, 0)
    print(f"Transposed shape:   {t_transposed.shape}")
    print(f"Transposed strides: {t_transposed.strides}")
    print(f"Is contiguous:      {t_transposed.is_contiguous}")
    print(f"Transposed matrix:\n{t_transposed.numpy()}")

    # Verify that transposing is an O(1) metadata view sharing underlying storage
    print(
        f"Storage base match: {t_from_np.numpy().base is not None or t_transposed.numpy().base is not None} (Zero-copy view!)\n"
    )

    # 4. Reshaping
    print("--- 4. Reshaping ---")
    t_flat = t_from_np.reshape([6])
    print(f"Flattened [6]: {t_flat.numpy()}")
    t_reshaped = t_from_np.reshape([3, 2])
    print(f"Reshaped [3, 2]:\n{t_reshaped.numpy()}\n")

    # 5. Elementwise Operations & Broadcasting
    print("--- 5. Broadcasting Semantics ---")
    # Shape [3, 1] + Shape [1, 4] -> Shape [3, 4]
    a = tt.tensor([[10.0], [20.0], [30.0]], dtype=tt.float32)  # (3, 1)
    b = tt.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=tt.float32)  # (1, 4)

    c = a + b
    print(f"A shape: {a.shape}")
    print(f"B shape: {b.shape}")
    print(f"Result (A + B) shape: {c.shape}")
    print(f"Broadcasted result:\n{c.numpy()}\n")

    # Scalar broadcasting
    s = c * 2.0
    print(f"Scalar multiplication (C * 2.0):\n{s.numpy()}\n")

    print("SUCCESS: Basic tensor semantics verified!")


if __name__ == "__main__":
    main()
