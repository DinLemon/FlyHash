"""Retrieval quality: brute-force ground truth and mean average precision."""

from __future__ import annotations

import numpy as np
from scipy import sparse

GROUND_TRUTH_K = 100


def true_neighbours(
    database: np.ndarray, queries: np.ndarray, k: int = GROUND_TRUTH_K
) -> np.ndarray:
    """Indices of the k nearest database rows to each query, by Euclidean
    distance, computed exactly."""
    db_sq = np.einsum("ij,ij->i", database, database, dtype=np.float64)
    q_sq = np.einsum("ij,ij->i", queries, queries, dtype=np.float64)
    # Squared distances; only their relative ordering matters (nothing here
    # is ever rooted back into a real distance). Accumulated in float64
    # because the expansion q_sq + db_sq - 2*q.db subtracts two large,
    # nearly-equal terms (db_sq alone is ~5e6 for raw 0-255 pixel data) and
    # float32 loses enough precision there to perturb the ranking.
    d = (
        q_sq[:, None]
        + db_sq[None, :]
        - 2.0 * (queries.astype(np.float64) @ database.astype(np.float64).T)
    )
    idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(d, idx, axis=1).argsort(axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)


def rank_by_code_overlap(
    db_codes: sparse.csr_array, query_codes: sparse.csr_array, k: int
) -> np.ndarray:
    """Rank database items by how many active units they share with each query."""
    overlap = np.asarray((query_codes @ db_codes.T).todense(), dtype=np.float32)
    # Overlap scores are small integers over ~10000 database items, so ties
    # at the rank-k boundary are dense. np.argpartition (introselect) is not
    # guaranteed to select the same candidate set on ties across numpy/BLAS
    # versions, so a stable sort applied only to its output is not enough to
    # guarantee bit-for-bit reproducibility. To make the whole ranking
    # deterministic we sort the full row instead of pre-selecting candidates
    # with argpartition: correctness beats speed here, and these arrays are
    # only 1000x10000. np.lexsort is stable and lets us make the tie-break
    # explicit as a secondary key: sort by (descending overlap, ascending
    # database index), i.e. lexsort on (index, -overlap) since lexsort's
    # last key is primary.
    n_db = overlap.shape[1]
    db_index = np.broadcast_to(np.arange(n_db, dtype=np.int64), overlap.shape)
    order = np.lexsort((db_index, -overlap), axis=1)
    return order[:, :k].astype(np.int64)


def average_precision(retrieved: np.ndarray, relevant: set[int]) -> float:
    """Average precision of one ranked list against a relevant set."""
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if int(item) in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def mean_average_precision(retrieved: np.ndarray, truth: np.ndarray) -> float:
    """mAP over all queries. Row i of `truth` is the relevant set for query i."""
    scores = [
        average_precision(retrieved[i], set(int(x) for x in truth[i]))
        for i in range(retrieved.shape[0])
    ]
    return float(np.mean(scores))
