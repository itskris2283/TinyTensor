import numpy as np
import pytest
import tinytensor as tt


def test_numpy_array_lifetime():
    # Verify numpy array returned by .numpy() remains valid after local tensor is deleted
    def get_array():
        t = tt.full([5, 5], 42.0)
        return t.numpy()

    arr = get_array()
    assert np.all(arr == 42.0)
    arr[0, 0] = 99.0
    assert arr[0, 0] == 99.0


def test_repeated_tensor_allocation_and_destruction():
    # Stress test allocation and deallocation without leaks
    for _ in range(100):
        t = tt.zeros([1000, 1000])
        assert t.numel == 1_000_000


def test_view_shared_storage():
    t = tt.tensor([[1.0, 2.0], [3.0, 4.0]])
    v = t.view([4])
    v_np = v.numpy()
    v_np[0] = 777.0
    assert t.numpy()[0, 0] == 777.0


def test_clone_deep_copy():
    t = tt.tensor([[1.0, 2.0], [3.0, 4.0]])
    c = t.clone()
    c_np = c.numpy()
    c_np[0, 0] = 555.0
    assert t.numpy()[0, 0] == 1.0


def test_allocation_limit_throws():
    with pytest.raises(tt.ResourceLimitError):
        # Requesting more than 16 GB elements
        tt.Shape([1000000, 1000000])
