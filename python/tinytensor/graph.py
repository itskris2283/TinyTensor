from collections.abc import Sequence

from tinytensor._tinytensor import (
    CompileError,
    DType,
    IRError,
    RuntimeExecutionError,
    float32,
)
from tinytensor._tinytensor import (
    Graph as _NativeGraph,
)
from tinytensor._tinytensor import (
    Node as _NativeNode,
)
from tinytensor.tensor import Tensor, tensor

__all__ = [
    "Graph",
    "Node",
    "IRError",
    "CompileError",
    "RuntimeExecutionError",
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
]


class Node:
    """Python wrapper around native Graph Node."""

    def __init__(self, native_node: _NativeNode, graph: "Graph"):
        self._native = native_node
        self._graph = graph

    @property
    def id(self) -> int:
        return self._native.id

    @property
    def name(self) -> str:
        return self._native.name

    @property
    def shape(self) -> list[int]:
        return self._native.shape

    @property
    def dtype(self) -> DType:
        return self._native.dtype

    def __add__(self, other):
        return self._graph.add(self, other)

    def __radd__(self, other):
        return self._graph.add(other, self)

    def __sub__(self, other):
        return self._graph.sub(self, other)

    def __rsub__(self, other):
        return self._graph.sub(other, self)

    def __mul__(self, other):
        return self._graph.mul(self, other)

    def __rmul__(self, other):
        return self._graph.mul(other, self)

    def __truediv__(self, other):
        return self._graph.div(self, other)

    def __rtruediv__(self, other):
        return self._graph.div(other, self)

    def __matmul__(self, other):
        return self._graph.matmul(self, other)

    def __repr__(self) -> str:
        return f"Node(id={self.id}, name='{self.name}', shape={self.shape}, dtype={self.dtype})"


class Graph:
    """Computation Graph representing mathematical operations."""

    def __init__(self):
        self._native = _NativeGraph()

    def input(self, shape: Sequence[int], dtype: DType = float32, name: str = "") -> Node:
        native_n = self._native.input(list(shape), dtype, name)
        return Node(native_n, self)

    def constant(self, value, name: str = "") -> Node:
        if not isinstance(value, Tensor):
            value = tensor(value)
        native_n = self._native.constant(value, name)
        return Node(native_n, self)

    def _ensure_node(self, x) -> _NativeNode:
        if isinstance(x, Node):
            return x._native
        if isinstance(x, _NativeNode):
            return x
        # Convert constants into constant nodes
        c = self.constant(x)
        return c._native

    def add(self, a, b) -> Node:
        na = self._ensure_node(a)
        nb = self._ensure_node(b)
        return Node(self._native.add(na, nb), self)

    def sub(self, a, b) -> Node:
        na = self._ensure_node(a)
        nb = self._ensure_node(b)
        return Node(self._native.sub(na, nb), self)

    def mul(self, a, b) -> Node:
        na = self._ensure_node(a)
        nb = self._ensure_node(b)
        return Node(self._native.mul(na, nb), self)

    def div(self, a, b) -> Node:
        na = self._ensure_node(a)
        nb = self._ensure_node(b)
        return Node(self._native.div(na, nb), self)

    def relu(self, a) -> Node:
        na = self._ensure_node(a)
        return Node(self._native.relu(na), self)

    def sigmoid(self, a) -> Node:
        na = self._ensure_node(a)
        return Node(self._native.sigmoid(na), self)

    def matmul(self, a, b) -> Node:
        na = self._ensure_node(a)
        nb = self._ensure_node(b)
        return Node(self._native.matmul(na, nb), self)

    def transpose(self, a, dim0: int, dim1: int) -> Node:
        na = self._ensure_node(a)
        return Node(self._native.transpose(na, dim0, dim1), self)

    def reduce_sum(self, a, axes: Sequence[int] | None = None, keepdims: bool = False) -> Node:
        na = self._ensure_node(a)
        ax = list(axes) if axes is not None else []
        return Node(self._native.reduce_sum(na, ax, keepdims), self)

    def reduce_mean(self, a, axes: Sequence[int] | None = None, keepdims: bool = False) -> Node:
        na = self._ensure_node(a)
        ax = list(axes) if axes is not None else []
        return Node(self._native.reduce_mean(na, ax, keepdims), self)

    def mark_output(self, node: Node) -> None:
        self._native.mark_output(node._native)

    def to_ir(self) -> str:
        return self._native.to_ir()

    def to_dot(self) -> str:
        return self._native.to_dot()


# Global graph builder functions
def add(a, b):
    if isinstance(a, Node):
        return a._graph.add(a, b)
    if isinstance(b, Node):
        return b._graph.add(a, b)
    raise TypeError("At least one operand must be a Node")


def sub(a, b):
    if isinstance(a, Node):
        return a._graph.sub(a, b)
    if isinstance(b, Node):
        return b._graph.sub(a, b)
    raise TypeError("At least one operand must be a Node")


def mul(a, b):
    if isinstance(a, Node):
        return a._graph.mul(a, b)
    if isinstance(b, Node):
        return b._graph.mul(a, b)
    raise TypeError("At least one operand must be a Node")


def div(a, b):
    if isinstance(a, Node):
        return a._graph.div(a, b)
    if isinstance(b, Node):
        return b._graph.div(a, b)
    raise TypeError("At least one operand must be a Node")


def relu(a: Node) -> Node:
    if not isinstance(a, Node):
        raise TypeError("Operand must be a Node")
    return a._graph.relu(a)


def sigmoid(a: Node) -> Node:
    if not isinstance(a, Node):
        raise TypeError("Operand must be a Node")
    return a._graph.sigmoid(a)


def matmul(a, b):
    if isinstance(a, Node):
        return a._graph.matmul(a, b)
    if isinstance(b, Node):
        return b._graph.matmul(a, b)
    raise TypeError("At least one operand must be a Node")


def sum(a: Node, axes: Sequence[int] | None = None, keepdims: bool = False) -> Node:
    if not isinstance(a, Node):
        raise TypeError("Operand must be a Node")
    return a._graph.reduce_sum(a, axes, keepdims)


def mean(a: Node, axes: Sequence[int] | None = None, keepdims: bool = False) -> Node:
    if not isinstance(a, Node):
        raise TypeError("Operand must be a Node")
    return a._graph.reduce_mean(a, axes, keepdims)


def transpose(a: Node, dim0: int, dim1: int) -> Node:
    if not isinstance(a, Node):
        raise TypeError("Operand must be a Node")
    return a._graph.transpose(a, dim0, dim1)
