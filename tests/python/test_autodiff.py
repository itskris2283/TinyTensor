import numpy as np
import pytest
import tinytensor as tt


def test_autodiff_add():
    g = tt.Graph()
    a = g.input([3])
    b = g.input([3])
    loss = tt.sum(a + b)

    grads = tt.gradients(g, loss, {a: [1.0, 2.0, 3.0], b: [4.0, 5.0, 6.0]})
    np.testing.assert_allclose(grads[a].numpy(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(grads[b].numpy(), [1.0, 1.0, 1.0])


def test_autodiff_mul():
    g = tt.Graph()
    a = g.input([2])
    b = g.input([2])
    loss = tt.sum(a * b)

    grads = tt.gradients(g, loss, {a: [3.0, 4.0], b: [5.0, 6.0]})
    np.testing.assert_allclose(grads[a].numpy(), [5.0, 6.0])
    np.testing.assert_allclose(grads[b].numpy(), [3.0, 4.0])


def test_autodiff_relu():
    g = tt.Graph()
    x = g.input([3])
    loss = tt.sum(tt.relu(x))

    grads = tt.gradients(g, loss, {x: [-2.0, 0.0, 3.0]})
    np.testing.assert_allclose(grads[x].numpy(), [0.0, 0.0, 1.0])


def test_autodiff_sigmoid():
    g = tt.Graph()
    x = g.input([1], dtype=tt.float64)
    loss = tt.sum(tt.sigmoid(x))

    val = 0.5
    grads = tt.gradients(g, loss, {x: [val]})
    sig = 1.0 / (1.0 + np.exp(-val))
    expected = sig * (1.0 - sig)
    assert pytest.approx(grads[x].item(), abs=1e-5) == expected


def test_autodiff_matmul():
    g = tt.Graph()
    a = g.input([2, 3])
    b = g.input([3, 2])
    loss = tt.sum(a @ b)

    a_np = np.ones((2, 3), dtype=np.float32)
    b_np = np.full((3, 2), 2.0, dtype=np.float32)

    grads = tt.gradients(g, loss, {a: a_np, b: b_np})
    # dL/dA = 1 @ B.T
    ones_22 = np.ones((2, 2), dtype=np.float32)
    expected_da = ones_22 @ b_np.T
    expected_db = a_np.T @ ones_22

    np.testing.assert_allclose(grads[a].numpy(), expected_da)
    np.testing.assert_allclose(grads[b].numpy(), expected_db)


def test_autodiff_unbroadcasting():
    g = tt.Graph()
    a = g.input([3, 4])
    b = g.input([4])
    loss = tt.sum(a + b)

    grads = tt.gradients(g, loss, {a: np.zeros((3, 4)), b: np.zeros(4)})
    assert grads[b].shape == [4]
    np.testing.assert_allclose(grads[b].numpy(), [3.0, 3.0, 3.0, 3.0])


def test_autodiff_multipath():
    g = tt.Graph()
    x = g.input([2])
    loss = tt.sum(x + x + x)

    grads = tt.gradients(g, loss, {x: [1.0, 1.0]})
    np.testing.assert_allclose(grads[x].numpy(), [3.0, 3.0])


def test_autodiff_finite_difference_composite():
    g = tt.Graph()
    x = g.input([2], dtype=tt.float64)
    w = g.input([2], dtype=tt.float64)
    loss = tt.sum(tt.sigmoid(x * w))

    x_np = np.array([0.7, -1.1], dtype=np.float64)
    w_np = np.array([1.5, 0.4], dtype=np.float64)

    grads = tt.gradients(g, loss, {x: x_np, w: w_np})

    eps = 1e-6
    for i in range(2):
        x_p = x_np.copy()
        x_p[i] += eps
        x_m = x_np.copy()
        x_m[i] -= eps

        f_p = np.sum(1.0 / (1.0 + np.exp(-(x_p * w_np))))
        f_m = np.sum(1.0 / (1.0 + np.exp(-(x_m * w_np))))
        num_grad = (f_p - f_m) / (2.0 * eps)

        assert pytest.approx(grads[x].numpy()[i], abs=1e-4) == num_grad


def test_autodiff_subtraction():
    g = tt.Graph()
    a = g.input([3])
    b = g.input([3])
    loss = tt.sum(a - b)
    grads = tt.gradients(g, loss, {a: [1.0, 2.0, 3.0], b: [4.0, 5.0, 6.0]})
    np.testing.assert_allclose(grads[a].numpy(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(grads[b].numpy(), [-1.0, -1.0, -1.0])


def test_autodiff_division():
    g = tt.Graph()
    a = g.input([2])
    b = g.input([2])
    loss = tt.sum(a / b)
    # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    grads = tt.gradients(g, loss, {a: [6.0, 8.0], b: [2.0, 4.0]})
    np.testing.assert_allclose(grads[a].numpy(), [0.5, 0.25])
    np.testing.assert_allclose(grads[b].numpy(), [-1.5, -0.5])


def test_autodiff_transpose():
    g = tt.Graph()
    x = g.input([2, 3])
    w = g.input([2, 3])
    xt = tt.transpose(x, 0, 1)  # shape [3, 2]
    wt = tt.transpose(w, 0, 1)  # shape [3, 2]
    loss = tt.sum(xt * wt)
    grads = tt.gradients(g, loss, {x: np.ones((2, 3)), w: np.full((2, 3), 4.0)})
    np.testing.assert_allclose(grads[x].numpy(), np.full((2, 3), 4.0))


def test_autodiff_mean_loss():
    g = tt.Graph()
    x = g.input([4])
    loss = tt.mean(x)  # d(mean)/dx = 1 / N = 0.25
    grads = tt.gradients(g, loss, {x: [1.0, 2.0, 3.0, 4.0]})
    np.testing.assert_allclose(grads[x].numpy(), [0.25, 0.25, 0.25, 0.25])


def test_autodiff_higher_rank_unbroadcasting():
    g = tt.Graph()
    a = g.input([2, 3, 4])
    b = g.input([4])  # broadcasted across dims 0 and 1 (2 * 3 = 6 times)
    loss = tt.sum(a * b)
    grads = tt.gradients(g, loss, {a: np.ones((2, 3, 4)), b: np.ones(4)})
    np.testing.assert_allclose(grads[b].numpy(), [6.0, 6.0, 6.0, 6.0])


def test_autodiff_branching_reconvergence():
    g = tt.Graph()
    x = g.input([2])
    y1 = x * 2.0
    y2 = x * 3.0
    loss = tt.sum(y1 + y2)  # dloss/dx = 2 + 3 = 5
    grads = tt.gradients(g, loss, {x: [1.0, 2.0]})
    np.testing.assert_allclose(grads[x].numpy(), [5.0, 5.0])
