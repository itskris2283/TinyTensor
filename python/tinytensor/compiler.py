from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tinytensor._tinytensor import (
    ExecutionMode as _ExecutionMode,
)
from tinytensor._tinytensor import (
    Executor as _NativeExecutor,
)
from tinytensor._tinytensor import (
    compute_gradients as _native_compute_gradients,
)
from tinytensor._tinytensor import (
    float32,
)
from tinytensor.graph import Graph, Node
from tinytensor.tensor import Tensor, tensor

__all__ = ["compile", "CompiledProgram", "gradients"]


class CompiledProgram:
    """Executable compiled TinyTensor program."""

    def __init__(self, executor: _NativeExecutor, graph: Graph):
        self._executor = executor
        self._graph = graph

    def run(self, inputs: Mapping[Node | Any, Any], profiler=None):
        native_inputs = {}
        for k, v in inputs.items():
            native_node = k._native if isinstance(k, Node) else k
            expected_dtype = getattr(native_node, "dtype", float32)
            if isinstance(v, Tensor):
                native_val = v
            elif isinstance(v, (np.ndarray, int, float, list)):
                native_val = tensor(v, dtype=expected_dtype)
            else:
                raise TypeError(f"Unsupported input value type for node {k}: {type(v)}")
            native_inputs[native_node] = native_val

        native_prof = getattr(profiler, "_native", profiler)
        return self._executor.run(native_inputs, native_prof)

    @property
    def optimization_report(self):
        return self._executor.optimization_report

    def original_ir(self) -> str:
        """Return the unoptimized initial SSA IR of the computational graph."""
        return self._executor.original_ir

    def optimized_ir(self) -> str:
        """Return the SSA IR after running all enabled optimization passes."""
        return self._executor.optimized_ir

    def generated_source(self) -> str:
        """Return the generated C++ source code for compiled kernels."""
        return self._executor.generated_source

    def execution_plan(self) -> str:
        """Return a structured summary of the execution plan and memory allocation."""
        return self._executor.execution_plan

    @property
    def ir(self) -> str:
        return self._graph.to_ir()

    @property
    def dot(self) -> str:
        return self._graph.to_dot()


def compile(
    graph: Graph,
    outputs: Sequence[Node] | None = None,
    mode: str = "avx2",
    optimize: bool = True,
    cache_dir: str | None = None,
) -> CompiledProgram:
    """Compile a TinyTensor computational graph into an executable program."""
    if outputs is not None:
        for out in outputs:
            graph.mark_output(out)

    mode_str = mode.lower()
    if mode_str in ("avx2", "compiled_avx2", "simd"):
        exec_mode = _ExecutionMode.CompiledAVX2
    elif mode_str in ("avx2_fma", "fma", "compiled_avx2_fma"):
        exec_mode = _ExecutionMode.CompiledAVX2_FMA
    elif mode_str in ("avx2_nofma", "nofma", "compiled_avx2_nofma"):
        exec_mode = _ExecutionMode.CompiledAVX2_NoFMA
    elif mode_str in ("scalar", "compiled_scalar"):
        exec_mode = _ExecutionMode.CompiledScalar
    elif mode_str in ("interpreted", "interp"):
        exec_mode = _ExecutionMode.Interpreted
    else:
        raise ValueError(
            f"Unknown compilation mode: {mode}. Expected 'avx2', 'avx2_fma', "
            "'avx2_nofma', 'scalar', or 'interpreted'."
        )

    native_executor = _NativeExecutor(
        graph._native, exec_mode, optimize, cache_dir if cache_dir is not None else ""
    )
    return CompiledProgram(native_executor, graph)


def gradients(
    graph: Graph,
    loss: Node,
    inputs: Mapping[Node | Any, Any],
) -> dict[Node, Tensor]:
    """Compute analytical gradients for input nodes using reverse-mode automatic differentiation."""
    native_inputs = {}
    node_map = {}
    for k, v in inputs.items():
        native_node = k._native if isinstance(k, Node) else k
        expected_dtype = getattr(native_node, "dtype", float32)
        if isinstance(v, Tensor):
            native_val = v
        else:
            native_val = tensor(v, dtype=expected_dtype)
        native_inputs[native_node] = native_val
        node_map[native_node] = k

    loss_native = loss._native if isinstance(loss, Node) else loss
    native_grads = _native_compute_gradients(graph._native, loss_native, native_inputs)

    result = {}
    for nat_node, grad_t in native_grads.items():
        if nat_node in node_map:
            result[node_map[nat_node]] = grad_t

    return result
