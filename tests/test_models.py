import numpy as np
import pytest
from scipy import sparse

from flyhash.circuit import Circuit
from flyhash.models import (
    fly_projection,
    hash_length,
    lsh_projection,
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
