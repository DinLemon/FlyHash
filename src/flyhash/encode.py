"""Turn data vectors into sparse binary codes through a projection."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def normalize(X: np.ndarray) -> np.ndarray:
    """Divide each row by its own mean.

    This mirrors the gain control the fly applies in its receptor neurons,
    and matches the preprocessing in Dasgupta et al. 2017. Rows that sum to
    zero are left untouched rather than turned into NaNs.
    """
    X = np.asarray(X, dtype=np.float32)
    means = X.mean(axis=1, keepdims=True)
    safe = np.where(means == 0.0, 1.0, means)
    return (X / safe).astype(np.float32)


def make_compressor(d_in: int, d_out: int, seed: int) -> np.ndarray:
    """A Gaussian random projection d_in -> d_out.

    Random rather than PCA on purpose: PCA is data-dependent and could
    interact differently with different projection models, which would
    confound the comparison this study exists to make.
    """
    rng = np.random.default_rng(seed)
    scale = np.float32(1.0 / np.sqrt(d_out))
    return (rng.standard_normal((d_in, d_out)) * scale).astype(np.float32)


def winner_take_all(A: np.ndarray, k: int) -> sparse.csr_array:
    """Keep the k largest entries of each row, as a binary sparse matrix."""
    n_rows, n_cols = A.shape
    if not 1 <= k <= n_cols:
        raise ValueError(f"k must be between 1 and {n_cols}, got {k}")
    top = np.argpartition(-A, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), k)
    cols = top.reshape(-1).astype(np.int64)
    data = np.ones(rows.size, dtype=np.float32)
    return sparse.csr_array((data, (rows, cols)), shape=(n_rows, n_cols))


def encode(
    X: np.ndarray,
    compressor: np.ndarray,
    projection: sparse.csr_array,
    k: int,
) -> sparse.csr_array:
    """normalize -> compress -> project -> winner-take-all."""
    compressed = normalize(X) @ compressor
    activity = compressed @ projection.T.toarray()
    return winner_take_all(np.asarray(activity, dtype=np.float32), k)


def apl_winner_take_all(
    A: np.ndarray, k: int, apl_weights: np.ndarray, gain: float
) -> sparse.csr_array:
    """Winner-take-all with the fly's measured graded inhibition.

    APL is a single giant inhibitory neuron that contacts every Kenyon cell,
    but with synapse counts spanning roughly 6 to 160. It is driven by
    Kenyon-cell spiking, which cannot be negative, so its drive is the row's
    summed POSITIVE activity (Kenyon-cell activity here is signed, so summing
    the raw, signed row would let the drive go negative and the term turn
    excitatory instead of inhibitory). Each cell is suppressed in proportion
    to both its own APL weight and that positive drive:

        x_i - gain * w_i * sum(max(x, 0))

    `gain * sum(max(x, 0))` is constant within a row, so what reorders the
    ranking is the spread of w_i. At gain=0 this reduces exactly to plain
    winner-take-all, which is the correctness check for the whole mechanism.
    """
    if apl_weights.shape[0] != A.shape[1]:
        raise ValueError(
            f"apl_weights has length {apl_weights.shape[0]}, expected {A.shape[1]}"
        )
    if gain == 0.0:
        return winner_take_all(A, k)
    drive = np.maximum(A, 0.0).sum(axis=1, keepdims=True)
    inhibited = A - gain * drive * apl_weights[None, :]
    return winner_take_all(np.asarray(inhibited, dtype=np.float32), k)


def encode_with_apl(
    X: np.ndarray,
    compressor: np.ndarray,
    projection: sparse.csr_array,
    k: int,
    apl_weights: np.ndarray,
    gain: float,
) -> sparse.csr_array:
    """normalize -> compress -> project -> APL-gated winner-take-all."""
    compressed = normalize(X) @ compressor
    activity = compressed @ projection.T.toarray()
    return apl_winner_take_all(
        np.asarray(activity, dtype=np.float32), k, apl_weights, gain
    )
