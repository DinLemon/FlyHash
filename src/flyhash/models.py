"""The projection models under comparison.

Every model returns a CSR matrix of shape (n_kc, n_pn) so that they are
interchangeable in encode(). FLY, UNIFORM and SHUFFLED are binary; LSH is
dense Gaussian and is stored as CSR only for interface uniformity.
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


def lsh_projection(n_kc: int, n_pn: int, seed: int) -> sparse.csr_array:
    """Classical dense Gaussian random projection, the baseline."""
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n_kc, n_pn)).astype(np.float32)
    return sparse.csr_array(dense)
