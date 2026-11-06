import numpy as np
import pytest
import tinytensor as tt


def test_elementwise_add_sub_mul_div():
    a = tt.tensor([10.0, 20.0, 30.0])
    b = tt.tensor([2.0, 4.0, 5.0])

    g = tt.Graph()
    na = g.input([3])
    nb = g.input([3])
    out_add = na + nb
    out_sub = na - nb
    out_mul = na * nb
    out_div = na / nb

    prog = tt.compile(g, outputs=[out_add, out_sub, out_mul, out_div], mode="interpreted")
    res_add, res_sub, res_mul, res_div = prog.run({na: a, nb: b})

    np.testing.assert_allclose(res_add.numpy(), [12.0, 24.0, 35.0])
    np.testing.assert_allclose(res_sub.numpy(), [8.0, 16.0, 25.0])
    np.testing.assert_allclose(res_mul.numpy(), [20.0, 80.0, 150.0])
    np.testing.assert_allclose(res_div.numpy(), [5.0, 5.0, 6.0])


def test_relu():
    x = tt.tensor([-5.0, 0.0, 3.0, -0.01])
    g = tt.Graph()
    nx = g.input([4])
    ny = tt.relu(nx)
    prog = tt.compile(g, outputs=[ny])
    res = prog.run({nx: x})
    np.testing.assert_allclose(res.numpy(), [0.0, 0.0, 3.0, 0.0])


def test_sigmoid():
    x = tt.tensor([-100.0, 0.0, 100.0])
    g = tt.Graph()
    nx = g.input([3])
    ny = tt.sigmoid(nx)
    prog = tt.compile(g, outputs=[ny])
    res = prog.run({nx: x})
    np.testing.assert_allclose(res.numpy(), [0.0, 0.5, 1.0], atol=1e-5)


def test_matmul_2d():
    a_np = np.random.randn(4, 5).astype(np.float32)
    b_np = np.random.randn(5, 3).astype(np.float32)

    g = tt.Graph()
    na = g.input([4, 5])
    nb = g.input([5, 3])
    nc = na @ nb
    prog = tt.compile(g, outputs=[nc])
    res = prog.run({na: a_np, nb: b_np})

    np.testing.assert_allclose(res.numpy(), a_np @ b_np, atol=1e-4)


def test_reduce_sum_and_mean():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    g = tt.Graph()
    nx = g.input([2, 3])
    s = tt.sum(nx)
    m = tt.mean(nx)
    prog = tt.compile(g, outputs=[s, m])
    res_s, res_m = prog.run({nx: arr})

    assert pytest.approx(res_s.item()) == 21.0
    assert pytest.approx(res_m.item()) == 3.5


def test_reduce_axes():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    g = tt.Graph()
    nx = g.input([2, 2])
    s0 = tt.sum(nx, axes=[0])
    s1 = tt.sum(nx, axes=[1], keepdims=True)
    prog = tt.compile(g, outputs=[s0, s1])
    res0, res1 = prog.run({nx: arr})

    np.testing.assert_allclose(res0.numpy(), [4.0, 6.0])
    np.testing.assert_allclose(res1.numpy(), [[3.0], [7.0]])


def test_sub_negative_results():
    g = tt.Graph()
    a = g.input([3])
    b = g.input([3])
    out = a - b
    prog = tt.compile(g, outputs=[out])
    res = prog.run({a: [1.0, 2.0, 3.0], b: [5.0, 2.0, 0.0]})
    np.testing.assert_allclose(res.numpy(), [-4.0, 0.0, 3.0])


def test_div_by_constants():
    g = tt.Graph()
    a = g.input([4])
    c = g.constant([2.0, 4.0, 8.0, 10.0])
    out = a / c
    prog = tt.compile(g, outputs=[out])
    res = prog.run({a: [10.0, 20.0, 40.0, 100.0]})
    np.testing.assert_allclose(res.numpy(), [5.0, 5.0, 5.0, 10.0])


def test_chained_arithmetic():
    g = tt.Graph()
    x = g.input([3])
    y = g.input([3])
    z = (x + y) * (x - y)  # x^2 - y^2
    prog = tt.compile(g, outputs=[z])
    res = prog.run({x: [3.0, 4.0, 5.0], y: [1.0, 2.0, 3.0]})
    np.testing.assert_allclose(res.numpy(), [8.0, 12.0, 16.0])


def test_reduce_negative_axes():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    g = tt.Graph()
    nx = g.input([3, 4])
    s = tt.sum(nx, axes=[-1])  # reduce last dimension
    prog = tt.compile(g, outputs=[s])
    res = prog.run({nx: arr})
    np.testing.assert_allclose(res.numpy(), np.sum(arr, axis=-1))


def test_reduce_mean_axes():
    arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    g = tt.Graph()
    nx = g.input([2, 2])
    m0 = tt.mean(nx, axes=[0])
    m1 = tt.mean(nx, axes=[1], keepdims=True)
    prog = tt.compile(g, outputs=[m0, m1])
    res0, res1 = prog.run({nx: arr})
    np.testing.assert_allclose(res0.numpy(), [20.0, 30.0])
    np.testing.assert_allclose(res1.numpy(), [[15.0], [35.0]])


def test_elementwise_float64():
    g = tt.Graph()
    a = g.input([2], dtype=tt.float64)
    b = g.input([2], dtype=tt.float64)
    out = a * b + a
    prog = tt.compile(g, outputs=[out])
    res = prog.run({a: [1.5, 2.5], b: [2.0, 4.0]})
    assert res.dtype == tt.float64
    np.testing.assert_allclose(res.numpy(), [4.5, 12.5])


def test_matmul_identity():
    ident = np.eye(4, dtype=np.float32)
    x = np.random.randn(4, 4).astype(np.float32)
    g = tt.Graph()
    na = g.input([4, 4])
    nb = g.input([4, 4])
    out = na @ nb
    prog = tt.compile(g, outputs=[out])
    res = prog.run({na: x, nb: ident})
    np.testing.assert_allclose(res.numpy(), x, atol=1e-5)
