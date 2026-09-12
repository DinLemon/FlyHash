import numpy as np
from scipy import sparse

from flyhash.bench import (
    average_precision,
    mean_average_precision,
    rank_by_code_overlap,
    true_neighbours,
)


def test_true_neighbours_finds_the_closest_points():
    db = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    queries = np.array([[0.1, 0.0]], dtype=np.float32)
    out = true_neighbours(db, queries, k=2)
    assert out.shape == (1, 2)
    assert out[0].tolist() == [0, 1]


def test_rank_by_code_overlap_prefers_the_most_shared_bits():
    db = sparse.csr_array(
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    )
    q = sparse.csr_array(np.array([[1, 1, 0]], dtype=np.float32))
    out = rank_by_code_overlap(db, q, k=3)
    assert out[0][:2].tolist() == [0, 1]


def test_average_precision_is_one_when_everything_relevant_comes_first():
    retrieved = np.array([3, 1, 7, 9])
    assert average_precision(retrieved, {3, 1}) == 1.0


def test_average_precision_is_zero_when_nothing_is_relevant():
    retrieved = np.array([3, 1, 7, 9])
    assert average_precision(retrieved, {42}) == 0.0


def test_average_precision_handles_an_empty_relevant_set():
    assert average_precision(np.array([1, 2, 3]), set()) == 0.0


def test_average_precision_handles_a_partial_hit():
    # relevant item at rank 2 only: precision 1/2, divided by |relevant| = 1
    retrieved = np.array([5, 3])
    assert average_precision(retrieved, {3}) == 0.5


def test_mean_average_precision_averages_over_queries():
    retrieved = np.array([[1, 2], [3, 4]])
    truth = np.array([[1, 2], [9, 8]])
    assert mean_average_precision(retrieved, truth) == 0.5
