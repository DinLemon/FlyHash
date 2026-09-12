import numpy as np
import pytest
from scipy import sparse

from flyhash.encode import encode, make_compressor, normalize, winner_take_all


def test_normalize_gives_each_row_unit_mean():
    X = np.array([[1.0, 3.0], [2.0, 2.0]], dtype=np.float32)
    out = normalize(X)
    np.testing.assert_allclose(out.mean(axis=1), np.ones(2), rtol=1e-6)


def test_normalize_leaves_all_zero_rows_alone():
    X = np.array([[0.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    out = normalize(X)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[0], np.zeros(2, dtype=np.float32))


def test_compressor_is_deterministic_for_a_seed():
    a = make_compressor(8, 4, seed=0)
    b = make_compressor(8, 4, seed=0)
    c = make_compressor(8, 4, seed=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (8, 4)


def test_wta_keeps_exactly_k_per_row():
    A = np.array([[5.0, 1.0, 3.0, 2.0], [0.0, 9.0, 8.0, 7.0]], dtype=np.float32)
    out = winner_take_all(A, k=2)
    assert isinstance(out, sparse.csr_array)
    np.testing.assert_array_equal(out.sum(axis=1), np.array([2.0, 2.0]))


def test_wta_keeps_the_largest_entries():
    A = np.array([[5.0, 1.0, 3.0, 2.0]], dtype=np.float32)
    out = winner_take_all(A, k=2).toarray()
    np.testing.assert_array_equal(out, np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32))


def test_wta_rejects_k_larger_than_width():
    A = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="k must be"):
        winner_take_all(A, k=4)


def test_encode_produces_k_active_units_per_row():
    rng = np.random.default_rng(0)
    X = rng.random((6, 10), dtype=np.float32)
    comp = make_compressor(10, 5, seed=0)
    proj = sparse.csr_array(rng.integers(0, 2, (12, 5)).astype(np.float32))
    codes = encode(X, comp, proj, k=3)
    assert codes.shape == (6, 12)
    np.testing.assert_array_equal(codes.sum(axis=1), np.full(6, 3.0))
