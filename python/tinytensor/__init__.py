"""TinyTensor: A Research-Grade Tensor Compiler, Numerical Runtime and
Automatic Differentiation Framework.
"""

__version__ = "0.1.1rc2"

from tinytensor._tinytensor import (
    BroadcastError,
    CompileError,
    DimensionError,
    DType,
    ExecutionMode,
    IRError,
    MemoryError,
    ProcessMemoryStats,
    ResourceLimitError,
    RuntimeExecutionError,
    Shape,
    ShapeError,
    TinyTensorError,
    TypeError,
    float32,
    float64,
    get_process_memory_stats,
    has_avx,
    has_avx2,
    has_fma3,
    supported_kernel_features,
    supported_target_isa,
)
from tinytensor.compiler import CompiledProgram, compile, gradients
from tinytensor.graph import (
    Graph,
    Node,
    add,
    div,
    matmul,
    mean,
    mul,
    relu,
    sigmoid,
    sub,
    sum,
    transpose,
)
from tinytensor.optim import SGD
from tinytensor.profiling import Profiler
from tinytensor.tensor import (
    Tensor,
    empty,
    full,
    ones,
    scalar,
    tensor,
    zeros,
)

__all__ = [
    "__version__",
    # Types
    "DType",
    "float32",
    "float64",
    "Shape",
    "Tensor",
    "Node",
    "Graph",
    "ExecutionMode",
    "CompiledProgram",
    "Profiler",
    "SGD",
    # Exceptions
    "ShapeError",
    "DimensionError",
    "BroadcastError",
    "TypeError",
    "MemoryError",
    "ResourceLimitError",
    "RuntimeExecutionError",
    "TinyTensorError",
    "IRError",
    "CompileError",
    # Factories
    "tensor",
    "zeros",
    "ones",
    "full",
    "empty",
    "scalar",
    # Operations
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    "sigmoid",
    "matmul",
    "sum",
    "mean",
    "transpose",
    # Compiler & Autodiff
    "compile",
    "gradients",
    "has_avx",
    "has_avx2",
    "has_fma3",
    "supported_target_isa",
    "supported_kernel_features",
    "get_process_memory_stats",
    "ProcessMemoryStats",
    "__version__",
]
