import numpy as np
import pytest
import tinytensor as tt


def test_broadcast_vector_to_matrix():
    v = tt.tensor([1.0, 2.0, 3.0])
    b = v.broadcast_to([4, 3])
    assert b.shape == [4, 3]
    expected = np.broadcast_to(v.numpy(), (4, 3))
    np.testing.assert_allclose(b.numpy(), expected)


def test_broadcast_singleton_expansion():
    t = tt.tensor([[10.0], [20.0]])
    b = t.broadcast_to([2, 5])
    assert b.shape == [2, 5]
    expected = np.broadcast_to(t.numpy(), (2, 5))
    np.testing.assert_allclose(b.numpy(), expected)


def test_broadcast_scalar():
    s = tt.scalar(99.0)
    b = s.broadcast_to([2, 3, 4])
    assert b.shape == [2, 3, 4]
    assert np.all(b.numpy() == 99.0)


def test_broadcast_incompatible_throws():
    t = tt.zeros([2, 3])
    with pytest.raises(tt.BroadcastError):
        t.broadcast_to([2, 4])


def test_broadcast_lower_rank_target_throws():
    t = tt.zeros([2, 3])
    with pytest.raises(tt.BroadcastError):
        t.broadcast_to([3])
