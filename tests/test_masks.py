from cdsd.masks import SupportMask, intersect_masks, EmptySupportError
import pytest


def test_intersection():
    a = SupportMask.from_iter(["a", "b"])
    b = SupportMask.from_iter(["b", "c"])
    assert intersect_masks(a, b).allowed == frozenset(["b"])


def test_empty_support_raises():
    m = SupportMask.from_iter([])
    with pytest.raises(EmptySupportError):
        m.assert_nonempty()
