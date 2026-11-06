import numpy as np
import pytest
import tinytensor as tt


def test_tensor_creation_from_list():
    t = tt.tensor([1.0, 2.0, 3.0])
    assert t.shape == [3]
    assert t.dtype == tt.float32
    assert t.numel == 3
    assert t.is_contiguous


def test_tensor_creation_from_scalar():
    t = tt.tensor(42.0)
    assert t.ndim == 0
    assert t.numel == 1
    assert t.item() == 42.0


def test_tensor_creation_from_numpy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = tt.tensor(arr)
    assert t.shape == [2, 2]
    np.testing.assert_allclose(t.numpy(), arr)


def test_tensor_zeros():
    t = tt.zeros([3, 4], dtype=tt.float32)
    assert t.shape == [3, 4]
    assert np.all(t.numpy() == 0.0)


def test_tensor_ones():
    t = tt.ones([2, 5], dtype=tt.float64)
    assert t.shape == [2, 5]
    assert t.dtype == tt.float64
    assert np.all(t.numpy() == 1.0)


def test_tensor_full():
    t = tt.full([2, 3], 7.5, dtype=tt.float32)
    assert np.all(t.numpy() == 7.5)


def test_tensor_scalar():
    s = tt.scalar(3.14159, dtype=tt.float64)
    assert s.ndim == 0
    assert pytest.approx(s.item()) == 3.14159


def test_tensor_view_and_reshape():
    t = tt.zeros([2, 6])
    v = t.view([3, 4])
    assert v.shape == [3, 4]
    r = t.reshape([4, 3])
    assert r.shape == [4, 3]


def test_tensor_transpose():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = tt.tensor(arr)
    tr = t.transpose(0, 1)
    assert tr.shape == [3, 2]
    assert not tr.is_contiguous
    np.testing.assert_allclose(tr.numpy(), arr.T)


def test_tensor_permute():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = tt.tensor(arr)
    p = t.permute([2, 0, 1])
    assert p.shape == [4, 2, 3]
    np.testing.assert_allclose(p.numpy(), np.transpose(arr, (2, 0, 1)))


def test_tensor_clone_independence():
    t1 = tt.full([2, 2], 5.0)
    t2 = t1.clone()
    np_t2 = t2.numpy()
    np_t2[0, 0] = 999.0
    assert t1.numpy()[0, 0] == 5.0


def test_tensor_contiguous():
    t = tt.zeros([2, 3])
    tr = t.transpose(0, 1)
    assert not tr.is_contiguous
    c = tr.contiguous()
    assert c.is_contiguous
    assert c.shape == [3, 2]


def test_invalid_view_throws():
    t = tt.zeros([2, 3])
    tr = t.transpose(0, 1)
    with pytest.raises(tt.ShapeError):
        tr.view([6])


def test_invalid_item_call():
    t = tt.zeros([2, 2])
    with pytest.raises(tt.ShapeError):
        t.item()


def test_tensor_repr():
    t = tt.zeros([2, 3])
    r = repr(t)
    assert "Tensor(shape=" in r
    assert "float32" in r


def test_tensor_nbytes():
    t32 = tt.zeros([10, 10], dtype=tt.float32)
    assert t32.nbytes == 100 * 4
    t64 = tt.zeros([10, 10], dtype=tt.float64)
    assert t64.nbytes == 100 * 8


def test_tensor_scalar_item():
    s = tt.scalar(42.5)
    assert s.item() == 42.5
    s64 = tt.scalar(100.25, dtype=tt.float64)
    assert s64.item() == 100.25


def test_tensor_permute_3d():
    t = tt.zeros([2, 3, 4])
    p = t.permute([2, 0, 1])
    assert p.shape == [4, 2, 3]


def test_tensor_empty():
    e = tt.empty([5, 5])
    assert e.shape == [5, 5]
    assert e.numel == 25
