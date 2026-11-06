from collections.abc import Sequence

import numpy as np

from tinytensor._tinytensor import (
    BroadcastError,
    DimensionError,
    DType,
    MemoryError,
    ResourceLimitError,
    ShapeError,
    TypeError,
    float32,
    float64,
)
from tinytensor._tinytensor import (
    Tensor as _NativeTensor,
)

__all__ = [
    "DType",
    "float32",
    "float64",
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "full",
    "empty",
    "scalar",
    "ShapeError",
    "DimensionError",
    "BroadcastError",
    "TypeError",
    "MemoryError",
    "ResourceLimitError",
]

Tensor = _NativeTensor


def tensor(data, dtype: DType | None = None) -> Tensor:
    """Create a TinyTensor from a Python scalar, list, or NumPy array."""
    if isinstance(data, _NativeTensor):
        if dtype is not None and data.dtype != dtype:
            raise TypeError("DType mismatch in tensor copy")
        return data.clone()

    if isinstance(data, (int, float)):
        target_dtype = dtype if dtype is not None else float32
        return _NativeTensor.scalar(float(data), target_dtype)

    if not isinstance(data, np.ndarray):
        if dtype == float64:
            target_np_dtype = np.float64
        else:
            target_np_dtype = np.float32
        data = np.array(data, dtype=target_np_dtype)
    else:
        if dtype == float32 and data.dtype != np.float32:
            data = data.astype(np.float32)
        elif dtype == float64 and data.dtype != np.float64:
            data = data.astype(np.float64)
        elif data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)

    return _NativeTensor.from_numpy(data)


def zeros(shape: Sequence[int], dtype: DType = float32) -> Tensor:
    return _NativeTensor.zeros(list(shape), dtype)


def ones(shape: Sequence[int], dtype: DType = float32) -> Tensor:
    return _NativeTensor.ones(list(shape), dtype)


def full(shape: Sequence[int], val: float, dtype: DType = float32) -> Tensor:
    return _NativeTensor.full(list(shape), float(val), dtype)


def empty(shape: Sequence[int], dtype: DType = float32) -> Tensor:
    return _NativeTensor.empty(list(shape), dtype)


def scalar(val: float, dtype: DType = float32) -> Tensor:
    return _NativeTensor.scalar(float(val), dtype)
