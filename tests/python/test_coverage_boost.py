import numpy as np
import pytest
import tinytensor as tt
from tinytensor.profiling import benchmark_function


def test_sgd_optimizer_step():
    p1 = tt.tensor([1.0, 2.0, 3.0])
    p2 = tt.tensor([4.0, 5.0, 6.0])
    opt = tt.SGD([p1, p2], lr=0.1, momentum=0.9)

    g1 = tt.tensor([0.1, 0.2, 0.3])
    g2 = tt.tensor([0.4, 0.5, 0.6])

    # First step
    opt.step([g1, g2])
    np.testing.assert_allclose(p1.numpy(), [1.0 - 0.01, 2.0 - 0.02, 3.0 - 0.03], atol=1e-5)

    # Second step with momentum accumulation
    opt.step([g1, g2])
    assert p1.numpy()[0] < 0.99

    # Zero momentum
    opt_no_mom = tt.SGD([p1], lr=0.01, momentum=0.0)
    opt_no_mom.step([g1])


def test_benchmark_function_utility():
    called = [0]

    def dummy():
        called[0] += 1

    times = benchmark_function(dummy, warmup=2, iterations=5)
    assert len(times) == 5
    assert called[0] == 7
    assert all(t >= 0.0 for t in times)


def test_graph_functional_operators_coverage():
    g = tt.Graph()
    a = g.input([2, 2], name="A")
    b = g.input([2, 2], name="B")

    # Direct Graph method calls
    n_add = g.add(a, b)
    n_sub = g.sub(a, b)
    n_mul = g.mul(a, b)
    n_div = g.div(a, b)
    n_mat = g.matmul(a, b)
    n_rel = g.relu(a)
    n_sig = g.sigmoid(a)
    n_trn = g.transpose(a, 0, 1)
    n_sum = g.reduce_sum(a, axes=[0], keepdims=True)
    n_men = g.reduce_mean(a, axes=[1], keepdims=False)

    assert n_add.shape == [2, 2]
    assert n_sub.shape == [2, 2]
    assert n_mul.shape == [2, 2]
    assert n_div.shape == [2, 2]
    assert n_mat.shape == [2, 2]
    assert n_rel.shape == [2, 2]
    assert n_sig.shape == [2, 2]
    assert n_trn.shape == [2, 2]
    assert n_sum.shape == [1, 2]
    assert n_men.shape == [2]

    # Module-level functional calls
    m_add = tt.add(a, b)
    m_sub = tt.sub(a, b)
    m_mul = tt.mul(a, b)
    m_div = tt.div(a, b)
    m_mat = tt.matmul(a, b)
    m_rel = tt.relu(a)
    m_sig = tt.sigmoid(a)
    m_trn = tt.transpose(a, 0, 1)
    m_sum = tt.sum(a)
    m_men = tt.mean(a)

    assert m_add.shape == [2, 2]
    assert m_sub.shape == [2, 2]
    assert m_mul.shape == [2, 2]
    assert m_div.shape == [2, 2]
    assert m_mat.shape == [2, 2]
    assert m_rel.shape == [2, 2]
    assert m_sig.shape == [2, 2]
    assert m_trn.shape == [2, 2]
    assert m_sum.shape == []
    assert m_men.shape == []


def test_tensor_conversion_branches():
    t = tt.tensor([1.0, 2.0])
    # Pass existing tensor with matching dtype
    t_copy = tt.tensor(t, dtype=tt.float32)
    assert t_copy.shape == [2]

    # Pass existing tensor with different dtype -> raises TypeError
    with pytest.raises(tt.TypeError):
        tt.tensor(t, dtype=tt.float64)

    # Cast numpy arrays of different dtypes
    arr64 = np.array([1.0, 2.0], dtype=np.float64)
    t32 = tt.tensor(arr64, dtype=tt.float32)
    assert t32.dtype == tt.float32

    arr32 = np.array([1.0, 2.0], dtype=np.float32)
    t64 = tt.tensor(arr32, dtype=tt.float64)
    assert t64.dtype == tt.float64

    # Integer numpy array auto-converted
    arr_int = np.array([1, 2], dtype=np.int32)
    t_int_to_float = tt.tensor(arr_int)
    assert t_int_to_float.dtype == tt.float32


def test_profiler_repr_and_properties():
    p = tt.Profiler()
    assert "TinyTensor Execution Profile" in repr(p)
    assert p.total_time_us == 0.0


def test_node_dunder_operators():
    g = tt.Graph()
    a = g.input([2], name="test_in")
    assert a.id >= 0
    assert a.name == "test_in"
    assert a.dtype == tt.float32
    assert "Node(" in repr(a)

    # Pass native node directly
    n_nat = g.add(a._native, a._native)
    assert n_nat.shape == [2]

    # TypeErrors for invalid inputs to global ops
    with pytest.raises(TypeError):
        tt.add(1, 2)
    with pytest.raises(TypeError):
        tt.sub(1, 2)
    with pytest.raises(TypeError):
        tt.mul(1, 2)
    with pytest.raises(TypeError):
        tt.div(1, 2)
    with pytest.raises(TypeError):
        tt.matmul(1, 2)
    with pytest.raises(TypeError):
        tt.relu(1)
    with pytest.raises(TypeError):
        tt.sigmoid(1)
    with pytest.raises(TypeError):
        tt.sum(1)
    with pytest.raises(TypeError):
        tt.mean(1)
    with pytest.raises(TypeError):
        tt.transpose(1, 0, 1)

    # radd, rsub, rmul, rdiv
    r1 = 5.0 + a
    r2 = 5.0 - a
    r3 = 5.0 * a
    r4 = 5.0 / a
    assert r1.shape == [2]
    assert r2.shape == [2]
    assert r3.shape == [2]
    assert r4.shape == [2]


def test_compiler_coverage_branches():
    g = tt.Graph()
    x = g.input([2])
    y = x + 1.0
    prog = tt.compile(g, outputs=[y], mode="interpreted")
    assert prog.optimization_report is not None

    with pytest.raises(ValueError):
        tt.compile(g, mode="unsupported_quantum_mode")
