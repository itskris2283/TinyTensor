import numpy as np
import pytest
import tinytensor as tt
import torch


@pytest.mark.parametrize("shape", [(10,), (4, 5), (2, 3, 4)])
def test_diff_elementwise_add(shape):
    a_np = np.random.randn(*shape).astype(np.float32)
    b_np = np.random.randn(*shape).astype(np.float32)

    g = tt.Graph()
    na = g.input(shape)
    nb = g.input(shape)
    out = na + nb
    prog = tt.compile(g, outputs=[out])
    tt_res = prog.run({na: a_np, nb: b_np}).numpy()

    torch_res = (torch.from_numpy(a_np) + torch.from_numpy(b_np)).numpy()
    np.testing.assert_allclose(tt_res, torch_res, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [(10,), (4, 5), (2, 3, 4)])
def test_diff_elementwise_mul(shape):
    a_np = np.random.randn(*shape).astype(np.float32)
    b_np = np.random.randn(*shape).astype(np.float32)

    g = tt.Graph()
    na = g.input(shape)
    nb = g.input(shape)
    out = na * nb
    prog = tt.compile(g, outputs=[out])
    tt_res = prog.run({na: a_np, nb: b_np}).numpy()

    torch_res = (torch.from_numpy(a_np) * torch.from_numpy(b_np)).numpy()
    np.testing.assert_allclose(tt_res, torch_res, atol=1e-5, rtol=1e-5)


def test_diff_relu():
    x_np = np.random.randn(5, 5).astype(np.float32)
    g = tt.Graph()
    nx = g.input([5, 5])
    out = tt.relu(nx)
    prog = tt.compile(g, outputs=[out])
    tt_res = prog.run({nx: x_np}).numpy()

    torch_res = torch.relu(torch.from_numpy(x_np)).numpy()
    np.testing.assert_allclose(tt_res, torch_res, atol=1e-5)


def test_diff_sigmoid():
    x_np = np.array([-50.0, -2.0, 0.0, 2.0, 50.0], dtype=np.float32)
    g = tt.Graph()
    nx = g.input([5])
    out = tt.sigmoid(nx)
    prog = tt.compile(g, outputs=[out])
    tt_res = prog.run({nx: x_np}).numpy()

    torch_res = torch.sigmoid(torch.from_numpy(x_np)).numpy()
    np.testing.assert_allclose(tt_res, torch_res, atol=1e-5)


@pytest.mark.parametrize("m,k,n", [(8, 16, 12), (32, 32, 32), (3, 7, 5)])
def test_diff_matmul(m, k, n):
    a_np = np.random.randn(m, k).astype(np.float32)
    b_np = np.random.randn(k, n).astype(np.float32)

    g = tt.Graph()
    na = g.input([m, k])
    nb = g.input([k, n])
    out = na @ nb
    prog = tt.compile(g, outputs=[out])
    tt_res = prog.run({na: a_np, nb: b_np}).numpy()

    torch_res = (torch.from_numpy(a_np) @ torch.from_numpy(b_np)).numpy()
    np.testing.assert_allclose(tt_res, torch_res, atol=1e-4, rtol=1e-4)


def test_diff_reduce_sum_and_mean():
    x_np = np.random.randn(4, 6).astype(np.float32)
    g = tt.Graph()
    nx = g.input([4, 6])
    s = tt.sum(nx, axes=[1], keepdims=True)
    m = tt.mean(nx, axes=[0], keepdims=False)
    prog = tt.compile(g, outputs=[s, m])
    tt_s, tt_m = prog.run({nx: x_np})

    tx = torch.from_numpy(x_np)
    torch_s = torch.sum(tx, dim=1, keepdim=True).numpy()
    torch_m = torch.mean(tx, dim=0, keepdim=False).numpy()

    np.testing.assert_allclose(tt_s.numpy(), torch_s, atol=1e-5)
    np.testing.assert_allclose(tt_m.numpy(), torch_m, atol=1e-5)


def test_diff_autodiff_matmul():
    a_np = np.random.randn(3, 4).astype(np.float32)
    b_np = np.random.randn(4, 2).astype(np.float32)

    # PyTorch Autograd
    t_a = torch.tensor(a_np, requires_grad=True)
    t_b = torch.tensor(b_np, requires_grad=True)
    t_loss = torch.sum(t_a @ t_b)
    t_loss.backward()

    # TinyTensor Autodiff
    g = tt.Graph()
    na = g.input([3, 4])
    nb = g.input([4, 2])
    loss = tt.sum(na @ nb)
    grads = tt.gradients(g, loss, {na: a_np, nb: b_np})

    np.testing.assert_allclose(grads[na].numpy(), t_a.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(grads[nb].numpy(), t_b.grad.numpy(), atol=1e-5)


def test_diff_autodiff_mlp():
    x_np = np.random.randn(2, 4).astype(np.float32)
    w1_np = np.random.randn(4, 3).astype(np.float32)
    b1_np = np.random.randn(3).astype(np.float32)
    w2_np = np.random.randn(3, 1).astype(np.float32)
    b2_np = np.random.randn(1).astype(np.float32)

    # PyTorch
    tx = torch.tensor(x_np, requires_grad=False)
    tw1 = torch.tensor(w1_np, requires_grad=True)
    tb1 = torch.tensor(b1_np, requires_grad=True)
    tw2 = torch.tensor(w2_np, requires_grad=True)
    tb2 = torch.tensor(b2_np, requires_grad=True)

    th = torch.relu(tx @ tw1 + tb1)
    ty = th @ tw2 + tb2
    tloss = torch.mean(ty)
    tloss.backward()

    # TinyTensor
    g = tt.Graph()
    nx = g.input([2, 4])
    nw1 = g.input([4, 3])
    nb1 = g.input([3])
    nh = tt.relu((nx @ nw1) + nb1)
    nw2 = g.input([3, 1])
    nb2 = g.input([1])
    ny = (nh @ nw2) + nb2
    loss = tt.mean(ny)

    grads = tt.gradients(g, loss, {nx: x_np, nw1: w1_np, nb1: b1_np, nw2: w2_np, nb2: b2_np})

    np.testing.assert_allclose(grads[nw1].numpy(), tw1.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(grads[nb1].numpy(), tb1.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(grads[nw2].numpy(), tw2.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(grads[nb2].numpy(), tb2.grad.numpy(), atol=1e-4)


@pytest.mark.parametrize("shape", [(8,), (3, 4), (2, 2, 2)])
def test_diff_elementwise_sub(shape):
    a_np = np.random.randn(*shape).astype(np.float32)
    b_np = np.random.randn(*shape).astype(np.float32)

    g = tt.Graph()
    na = g.input(shape)
    nb = g.input(shape)
    out = na - nb
    prog = tt.compile(g, outputs=[out])
    tt_res = prog.run({na: a_np, nb: b_np}).numpy()

    torch_res = (torch.from_numpy(a_np) - torch.from_numpy(b_np)).numpy()
    np.testing.assert_allclose(tt_res, torch_res, atol=1e-5)


@pytest.mark.parametrize("shape", [(8,), (3, 4), (2, 2, 2)])
def test_diff_elementwise_div(shape):
    a_np = np.random.randn(*shape).astype(np.float32)
    b_np = (np.random.rand(*shape) + 0.5).astype(np.float32)

    g = tt.Graph()
    na = g.input(shape)
    nb = g.input(shape)
    out = na / nb
    prog = tt.compile(g, outputs=[out])
    tt_res = prog.run({na: a_np, nb: b_np}).numpy()

    torch_res = (torch.from_numpy(a_np) / torch.from_numpy(b_np)).numpy()
    np.testing.assert_allclose(tt_res, torch_res, atol=1e-5)


def test_diff_broadcasting_add():
    a_np = np.random.randn(3, 1, 4).astype(np.float32)
    b_np = np.random.randn(1, 5, 4).astype(np.float32)

    g = tt.Graph()
    na = g.input([3, 1, 4])
    nb = g.input([1, 5, 4])
    out = na + nb
    prog = tt.compile(g, outputs=[out])
    tt_res = prog.run({na: a_np, nb: b_np}).numpy()

    torch_res = (torch.from_numpy(a_np) + torch.from_numpy(b_np)).numpy()
    np.testing.assert_allclose(tt_res, torch_res, atol=1e-5)


def test_diff_transpose_matmul():
    a_np = np.random.randn(4, 6).astype(np.float32)
    b_np = np.random.randn(4, 5).astype(np.float32)

    g = tt.Graph()
    na = g.input([4, 6])
    nb = g.input([4, 5])
    nat = tt.transpose(na, 0, 1)  # shape [6, 4]
    out = nat @ nb  # shape [6, 5]
    prog = tt.compile(g, outputs=[out])
    tt_res = prog.run({na: a_np, nb: b_np}).numpy()

    torch_res = (torch.from_numpy(a_np).T @ torch.from_numpy(b_np)).numpy()
    np.testing.assert_allclose(tt_res, torch_res, atol=1e-4)
