import numpy as np
import pytest
from scipy import sparse

from flyhash.circuit import Circuit
from flyhash.hash import FlyHasher, SimilaritySearch, find_duplicates


def test_random_hasher_has_the_documented_default_shape():
    h = FlyHasher.random(n_inputs=50)
    assert h.n_inputs == 50
    assert h.n_cells == 1000  # 20x expansion
    assert h.hash_length == 50  # 5% of 1000


def test_every_cell_samples_exactly_n_claws_distinct_inputs():
    h = FlyHasher.random(n_inputs=40, n_cells=300, n_claws=6, seed=0)
    per_cell = np.diff(h.projection.indptr)
    assert set(per_cell.tolist()) == {6}
    dense = h.projection.toarray()
    assert set(np.unique(dense).tolist()) == {0.0, 1.0}


def test_same_seed_gives_the_same_wiring_and_a_different_seed_does_not():
    a = FlyHasher.random(30, 200, seed=3).projection.toarray()
    b = FlyHasher.random(30, 200, seed=3).projection.toarray()
    c = FlyHasher.random(30, 200, seed=4).projection.toarray()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_encode_gives_exactly_hash_length_active_cells():
    h = FlyHasher.random(20, 400, seed=0)
    codes = h.encode(np.random.default_rng(0).random((7, 20)))
    assert codes.shape == (7, 400)
    np.testing.assert_array_equal(codes.sum(axis=1), np.full(7, h.hash_length))


def test_encode_is_deterministic():
    h = FlyHasher.random(20, 400, seed=0)
    X = np.random.default_rng(1).random((5, 20))
    np.testing.assert_array_equal(h.encode(X).toarray(), h.encode(X).toarray())


def test_encode_ignores_overall_scale():
    """Doubling a vector's magnitude must not change its code — the point of
    dividing by the row mean."""
    h = FlyHasher.random(16, 320, seed=0)
    X = np.random.default_rng(2).random((4, 16)) + 0.1
    np.testing.assert_array_equal(
        h.encode(X).toarray(), h.encode(X * 7.5).toarray()
    )


def test_encode_survives_an_all_zero_row():
    h = FlyHasher.random(12, 240, seed=0)
    X = np.zeros((2, 12), dtype=np.float32)
    X[1] = 1.0
    codes = h.encode(X)
    assert np.isfinite(codes.toarray()).all()
    np.testing.assert_array_equal(codes.sum(axis=1), np.full(2, h.hash_length))


def test_encode_rejects_the_wrong_width():
    h = FlyHasher.random(10, 200, seed=0)
    with pytest.raises(ValueError, match="expects 10"):
        h.encode(np.zeros((3, 9), dtype=np.float32))


def test_encode_rejects_a_one_dimensional_input():
    h = FlyHasher.random(10, 200, seed=0)
    with pytest.raises(ValueError, match="2-D"):
        h.encode(np.zeros(10, dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_inputs": 0}, "n_inputs"),
        ({"n_inputs": 10, "n_claws": 11}, "n_claws"),
        ({"n_inputs": 10, "n_cells": 10, "sparsity": 5.0}, "hash_length"),
    ],
)
def test_random_rejects_impossible_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FlyHasher.random(**kwargs)


def test_from_circuit_uses_the_measured_wiring():
    matrix = sparse.csr_array(
        np.array([[5.0, 0.0, 2.0], [0.0, 9.0, 0.0]], dtype=np.float32)
    )
    circuit = Circuit(
        pn_to_kc=matrix,
        kc_ids=np.array([1, 2], dtype=np.int64),
        kc_types=np.array(["KCg-m", "KCg-m"], dtype=object),
        kc_sides=np.array(["L", "L"], dtype=object),
        pn_ids=np.array([10, 11, 12], dtype=np.int64),
        pn_types=np.array(["DA1_lPN", "VA1v_adPN", "DM2_lPN"], dtype=object),
        pn_sides=np.array(["L", "L", "L"], dtype=object),
        apl_to_kc=np.array([40.0, 41.0], dtype=np.float32),
    )
    h = FlyHasher.from_circuit(circuit)
    assert h.n_cells == 2 and h.n_inputs == 3
    assert set(np.unique(h.projection.toarray()).tolist()) == {0.0, 1.0}


def test_search_finds_a_vector_as_its_own_closest_match():
    rng = np.random.default_rng(0)
    data = rng.random((300, 24)).astype(np.float32)
    index = SimilaritySearch.build(data, seed=0)
    assert len(index) == 300
    neighbours, overlap = index.query(data[:20], k=5)
    assert neighbours.shape == (20, 5)
    np.testing.assert_array_equal(neighbours[:, 0], np.arange(20))
    np.testing.assert_array_equal(overlap[:, 0], np.full(20, index.hasher.hash_length))


def test_search_returns_overlap_in_descending_order():
    rng = np.random.default_rng(1)
    data = rng.random((200, 16)).astype(np.float32)
    index = SimilaritySearch.build(data, seed=0)
    _, overlap = index.query(data[:10], k=8)
    assert np.all(np.diff(overlap, axis=1) <= 0)


def test_search_is_reproducible_across_calls():
    rng = np.random.default_rng(2)
    data = rng.random((150, 12)).astype(np.float32)
    index = SimilaritySearch.build(data, seed=0)
    first, _ = index.query(data[:5], k=6)
    for _ in range(3):
        again, _ = index.query(data[:5], k=6)
        np.testing.assert_array_equal(first, again)


def test_search_rejects_an_out_of_range_k():
    data = np.random.default_rng(3).random((20, 8)).astype(np.float32)
    index = SimilaritySearch.build(data, seed=0)
    with pytest.raises(ValueError, match="k must be"):
        index.query(data[:2], k=21)


def test_find_duplicates_recovers_a_planted_copy():
    rng = np.random.default_rng(0)
    data = rng.random((60, 20)).astype(np.float32) + 0.5
    data[41] = data[7]  # an exact copy
    pairs = find_duplicates(data, threshold=0.95, seed=0)
    assert (7, 41, 1.0) in [(i, j, round(s, 6)) for i, j, s in pairs]


def test_find_duplicates_reports_strongest_first_and_i_before_j():
    rng = np.random.default_rng(1)
    data = rng.random((40, 16)).astype(np.float32) + 0.5
    data[10] = data[3]
    pairs = find_duplicates(data, threshold=0.2, seed=0)
    assert all(i < j for i, j, _ in pairs)
    assert all(a >= b for a, b in zip([s for *_, s in pairs], [s for *_, s in pairs][1:]))


def test_find_duplicates_reports_nothing_when_nothing_matches_perfectly():
    """Random rows share some cells by chance, but never all of them."""
    data = np.random.default_rng(2).random((30, 10)).astype(np.float32) + 0.5
    assert find_duplicates(data, threshold=1.0, seed=0) == []


def test_find_duplicates_rejects_a_threshold_outside_zero_to_one():
    data = np.random.default_rng(2).random((30, 10)).astype(np.float32) + 0.5
    with pytest.raises(ValueError, match="threshold"):
        find_duplicates(data, threshold=1.5, seed=0)
