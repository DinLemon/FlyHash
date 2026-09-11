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
    db_sq = np.einsum("ij,ij->i", database, database)
    q_sq = np.einsum("ij,ij->i", queries, queries)
    # Squared distances; the constant q_sq term does not affect the ordering
    # but is kept so the values stay interpretable.
    d = q_sq[:, None] + db_sq[None, :] - 2.0 * (queries @ database.T)
    idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(d, idx, axis=1).argsort(axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)


def rank_by_code_overlap(
    db_codes: sparse.csr_array, query_codes: sparse.csr_array, k: int
) -> np.ndarray:
    """Rank database items by how many active units they share with each query."""
    overlap = np.asarray((query_codes @ db_codes.T).todense(), dtype=np.float32)
    idx = np.argpartition(-overlap, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(-overlap, idx, axis=1).argsort(axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)


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
