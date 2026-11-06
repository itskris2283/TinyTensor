import pytest
import tinytensor as tt


def test_shape_scalar():
    s = tt.Shape([])
    assert s.ndim() == 0
    assert s.numel() == 1


def test_shape_vector():
    s = tt.Shape([10])
    assert s.ndim() == 1
    assert s.numel() == 10
    assert s.dims == [10]


def test_shape_matrix():
    s = tt.Shape([4, 5])
    assert s.ndim() == 2
    assert s.numel() == 20
    assert s.dims == [4, 5]


def test_shape_higher_rank():
    s = tt.Shape([2, 3, 4, 5])
    assert s.ndim() == 4
    assert s.numel() == 120


def test_shape_zero_dimension():
    s = tt.Shape([3, 0, 4])
    assert s.ndim() == 3
    assert s.numel() == 0


def test_shape_negative_dimension_throws():
    with pytest.raises(tt.ShapeError):
        tt.Shape([2, -5, 3])


def test_shape_equality():
    s1 = tt.Shape([2, 3])
    s2 = tt.Shape([2, 3])
    s3 = tt.Shape([3, 2])
    assert s1 == s2
    assert s1 != s3


def test_shape_max_rank_exceeded():
    with pytest.raises(tt.ResourceLimitError):
        tt.Shape([1] * 33)


def test_shape_repr():
    s = tt.Shape([3, 4, 5])
    assert "3" in repr(s)
    assert "4" in repr(s)
    assert "5" in repr(s)


def test_shape_inequality():
    s1 = tt.Shape([2, 3])
    s2 = tt.Shape([3, 2])
    assert s1 != s2


def test_shape_numel_multiplication():
    s = tt.Shape([2, 3, 4, 5])
    assert s.numel() == 120
    assert s.ndim() == 4
    assert s.dims == [2, 3, 4, 5]


def test_shape_single_element_vector():
    s = tt.Shape([1])
    assert s.numel() == 1
    assert s.ndim() == 1
