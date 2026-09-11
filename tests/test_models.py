import numpy as np
import pytest
from scipy import sparse

from flyhash.circuit import Circuit
from flyhash.models import (
    fly_projection,
    hash_length,
    lsh_projection,
    shuffled_projection,
    uniform_projection,
)


def make_toy_circuit():
    m = sparse.csr_array(
        np.array([[5.0, 0.0], [7.0, 3.0], [0.0, 9.0]], dtype=np.float32)
    )
    return Circuit(
        pn_to_kc=m,
        kc_ids=np.array([10, 11, 12], dtype=np.int64),
        kc_types=np.array(["KCg-m", "KCab-s", "KCg-m"], dtype=object),
        kc_sides=np.array(["L", "L", "R"], dtype=object),
        pn_ids=np.array([20, 21], dtype=np.int64),
        pn_types=np.array(["DA1_lPN", "VA1v_adPN"], dtype=object),
        pn_sides=np.array(["L", "R"], dtype=object),
        apl_to_kc=np.array([40.0, 41.0, 42.0], dtype=np.float32),
    )


def test_fly_projection_is_binary_and_keeps_structure():
    p = fly_projection(make_toy_circuit())
    np.testing.assert_array_equal(
        p.toarray(), np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32)
    )


def test_uniform_projection_gives_every_row_exactly_n_claws():
    p = uniform_projection(n_kc=50, n_pn=20, n_claws=6, seed=0)
    assert p.shape == (50, 20)
    np.testing.assert_array_equal(p.sum(axis=1), np.full(50, 6.0))


def test_uniform_projection_is_deterministic_for_a_seed():
    a = uniform_projection(30, 10, 4, seed=7).toarray()
    b = uniform_projection(30, 10, 4, seed=7).toarray()
    c = uniform_projection(30, 10, 4, seed=8).toarray()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_uniform_projection_rejects_too_many_claws():
    with pytest.raises(ValueError, match="n_claws"):
        uniform_projection(10, 3, n_claws=5, seed=0)


def test_lsh_projection_is_dense_and_shaped_right():
    p = lsh_projection(n_kc=40, n_pn=12, seed=0)
    assert p.shape == (40, 12)
    assert p.nnz == 40 * 12


def test_hash_length_is_five_percent_rounded():
    assert hash_length(1865) == 93
    assert hash_length(1875) == 94
    assert hash_length(10) == 1  # never returns zero


def test_hash_length_rounds_half_up_consistently():
    # Verify half-up rounding at .5 boundaries (exact multiples)
    # At 5%: 1850 * 0.05 = 92.5, 1750 * 0.05 = 87.5
    assert hash_length(1850) == 93  # 92.5 rounds up
    assert hash_length(1750) == 88  # 87.5 rounds up
    # Both .5 cases round in the SAME direction (up), unlike banker's rounding


def _fly_like_matrix(seed=0, n_kc=200, n_pn=30):
    """A matrix with a deliberately uneven in-degree distribution."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(n_kc):
        claws = rng.integers(2, 9)
        for c in rng.choice(n_pn, size=claws, replace=False):
            rows.append(i)
            cols.append(c)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_array((data, (rows, cols)), shape=(n_kc, n_pn))


def test_shuffle_preserves_both_degree_sequences():
    m = _fly_like_matrix()
    s = shuffled_projection(m, seed=1000)
    np.testing.assert_array_equal(s.sum(axis=1), m.sum(axis=1))
    np.testing.assert_array_equal(s.sum(axis=0), m.sum(axis=0))


def test_shuffle_preserves_edge_count_and_stays_binary():
    m = _fly_like_matrix()
    s = shuffled_projection(m, seed=1000)
    assert s.nnz == m.nnz
    assert set(np.unique(s.data).tolist()) == {1.0}


def test_shuffle_never_creates_a_duplicate_edge():
    m = _fly_like_matrix()
    s = shuffled_projection(m, seed=1000).tocoo()
    pairs = set(zip(s.row.tolist(), s.col.tolist()))
    assert len(pairs) == s.nnz


def test_shuffle_actually_rewires_something():
    m = _fly_like_matrix()
    s = shuffled_projection(m, seed=1000)
    assert not np.array_equal(s.toarray(), m.toarray())


def test_shuffle_is_deterministic_for_a_seed():
    m = _fly_like_matrix()
    a = shuffled_projection(m, seed=1000).toarray()
    b = shuffled_projection(m, seed=1000).toarray()
    c = shuffled_projection(m, seed=1001).toarray()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
