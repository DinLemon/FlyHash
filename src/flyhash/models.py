"""The projection models under comparison.

Every model returns a CSR matrix of shape (n_kc, n_pn) so that they are
interchangeable in encode(). FLY, UNIFORM and SHUFFLED are binary;
GAUSSIAN is a dense Gaussian random projection and is stored as CSR only
for interface uniformity.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from flyhash.circuit import Circuit

HASH_FRACTION = 0.05


def hash_length(n_kc: int, fraction: float = HASH_FRACTION) -> int:
    """Number of KCs left active after winner-take-all."""
    return max(1, int(np.floor(fraction * n_kc + 0.5)))


def fly_projection(circuit: Circuit) -> sparse.csr_array:
    """The measured wiring, with synapse counts discarded.

    Binary because the models it is compared against are binary; using
    weights here would confound 'which partners' with 'how strongly'.
    """
    return circuit.binary()


def uniform_projection(
    n_kc: int, n_pn: int, n_claws: int, seed: int
) -> sparse.csr_array:
    """The idealised model from Dasgupta et al. 2017: every KC draws exactly
    n_claws distinct inputs uniformly at random."""
    if not 1 <= n_claws <= n_pn:
        raise ValueError(f"n_claws must be between 1 and {n_pn}, got {n_claws}")
    rng = np.random.default_rng(seed)
    cols = np.empty(n_kc * n_claws, dtype=np.int64)
    for i in range(n_kc):
        cols[i * n_claws : (i + 1) * n_claws] = rng.choice(
            n_pn, size=n_claws, replace=False
        )
    rows = np.repeat(np.arange(n_kc, dtype=np.int64), n_claws)
    data = np.ones(rows.size, dtype=np.float32)
    return sparse.csr_array((data, (rows, cols)), shape=(n_kc, n_pn))


def dense_gaussian_projection(n_kc: int, n_pn: int, seed: int) -> sparse.csr_array:
    """A dense Gaussian random projection, put through the same top-k
    winner-take-all sparsification as every other model here. Serves as a
    dense-wiring baseline.

    This is NOT classical locality-sensitive hashing (which uses sign bits
    of a random projection and Hamming distance) -- it must not be
    described or compared to published LSH results as if it were.
    """
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n_kc, n_pn)).astype(np.float32)
    return sparse.csr_array(dense)


def shuffled_projection(
    matrix: sparse.csr_array, seed: int, swaps_per_edge: int = 10
) -> sparse.csr_array:
    """Rewire the graph at random while preserving both degree sequences.

    Double edge swap on the bipartite graph: pick edges (a, b) and (c, d),
    propose (a, d) and (c, b), and accept only if neither already exists.
    Every KC therefore keeps its exact number of inputs and every PN keeps
    its exact number of targets — only *which* partner goes with which
    changes. That is what isolates partner choice from degree structure.

    An occupancy bitmap gives O(1) duplicate checks; for the real circuit
    it is 1865 x 149 booleans, so the memory cost is negligible.
    """
    coo = matrix.tocoo()
    rows = coo.row.astype(np.int64).copy()
    cols = coo.col.astype(np.int64).copy()
    n_edges = rows.size

    occupied = np.zeros(matrix.shape, dtype=bool)
    occupied[rows, cols] = True

    rng = np.random.default_rng(seed)
    n_swaps = swaps_per_edge * n_edges
    first = rng.integers(0, n_edges, n_swaps)
    second = rng.integers(0, n_edges, n_swaps)

    for i, j in zip(first, second):
        r1, c1 = rows[i], cols[i]
        r2, c2 = rows[j], cols[j]
        if r1 == r2 or c1 == c2:
            continue
        if occupied[r1, c2] or occupied[r2, c1]:
            continue
        occupied[r1, c1] = False
        occupied[r2, c2] = False
        occupied[r1, c2] = True
        occupied[r2, c1] = True
        cols[i] = c2
        cols[j] = c1

    data = np.ones(n_edges, dtype=np.float32)
    return sparse.csr_array((data, (rows, cols)), shape=matrix.shape)
