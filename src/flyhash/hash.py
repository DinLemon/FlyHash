"""FlyHash as a usable tool: encode vectors, find similar ones, find duplicates.

The fly's olfactory circuit computes a locality-sensitive hash — expand the
input into many cells, let each sample a handful of inputs, then keep only the
strongest few percent. Similar inputs land on overlapping sets of cells.

**This ships random wiring by default, and that is a finding, not a shortcut.**
The rest of this repository compares the measured connectome against a
degree-preserving shuffle of itself across 86 conditions, two animals and four
datasets including the fly's own odour repertoire. The measured wiring never
wins reproducibly. So there is no reason to carry a connectome around, and
good reason not to: random wiring frees you to pick any dimensions you like.

`FlyHasher.from_circuit` exists anyway, for reproducing that comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

DEFAULT_CLAWS = 6
DEFAULT_EXPANSION = 20
DEFAULT_SPARSITY = 0.05


def _validate(n_inputs: int, n_cells: int, n_claws: int, hash_length: int) -> None:
    if n_inputs < 1:
        raise ValueError(f"n_inputs must be at least 1, got {n_inputs}")
    if n_cells < 1:
        raise ValueError(f"n_cells must be at least 1, got {n_cells}")
    if not 1 <= n_claws <= n_inputs:
        raise ValueError(
            f"n_claws must be between 1 and n_inputs ({n_inputs}), got {n_claws}"
        )
    if not 1 <= hash_length <= n_cells:
        raise ValueError(
            f"hash_length must be between 1 and n_cells ({n_cells}), "
            f"got {hash_length}"
        )


@dataclass(frozen=True)
class FlyHasher:
    """Turns vectors into sparse binary codes.

    Build one with :meth:`random` rather than calling this directly.

    Attributes:
        projection: (n_cells, n_inputs) sparse binary matrix — which inputs
            each cell samples.
        hash_length: how many cells stay active in every code.
    """

    projection: sparse.csr_array
    hash_length: int

    @property
    def n_inputs(self) -> int:
        return self.projection.shape[1]

    @property
    def n_cells(self) -> int:
        return self.projection.shape[0]

    @classmethod
    def random(
        cls,
        n_inputs: int,
        n_cells: int | None = None,
        n_claws: int = DEFAULT_CLAWS,
        sparsity: float = DEFAULT_SPARSITY,
        seed: int = 0,
    ) -> FlyHasher:
        """A hasher with random sparse wiring.

        Args:
            n_inputs: width of the vectors you will encode.
            n_cells: how many cells to expand into. Defaults to 20x n_inputs,
                the expansion used in the original FlyHash paper.
            n_claws: how many inputs each cell samples. The fly uses about 6.
            sparsity: fraction of cells left active in each code.
            seed: fixes the wiring, so the same seed gives the same hasher.
        """
        if n_cells is None:
            n_cells = DEFAULT_EXPANSION * n_inputs
        hash_length = max(1, int(np.floor(sparsity * n_cells + 0.5)))
        _validate(n_inputs, n_cells, n_claws, hash_length)

        rng = np.random.default_rng(seed)
        columns = np.empty(n_cells * n_claws, dtype=np.int64)
        for cell in range(n_cells):
            columns[cell * n_claws : (cell + 1) * n_claws] = rng.choice(
                n_inputs, size=n_claws, replace=False
            )
        rows = np.repeat(np.arange(n_cells, dtype=np.int64), n_claws)
        projection = sparse.csr_array(
            (np.ones(rows.size, dtype=np.float32), (rows, columns)),
            shape=(n_cells, n_inputs),
        )
        return cls(projection=projection, hash_length=hash_length)

    @classmethod
    def from_circuit(cls, circuit, sparsity: float = DEFAULT_SPARSITY) -> FlyHasher:
        """A hasher wired from a measured connectome.

        Provided for reproducing this repository's comparison. It is not the
        recommended way to build a hasher: across every condition tested, the
        measured wiring did no better than :meth:`random`, and it pins you to
        the fly's dimensions.

        Unlike :meth:`random`, where every cell samples exactly ``n_claws``
        inputs, here the number of inputs per cell follows the circuit's real
        wiring and varies from cell to cell (2 to 13 in the measured data).
        Swapping one hasher for the other is not a drop-in change: activity
        statistics will differ. ``n_cells`` and ``n_inputs`` are also fixed by
        the circuit's shape, not chosen by the caller.
        """
        binary = circuit.binary()
        hash_length = max(1, int(np.floor(sparsity * binary.shape[0] + 0.5)))
        return cls(projection=binary, hash_length=hash_length)

    def encode(self, X: np.ndarray) -> sparse.csr_array:
        """Encode a batch of vectors into sparse binary codes.

        Args:
            X: (n_samples, n_inputs) array.

        Returns:
            (n_samples, n_cells) sparse matrix with exactly ``hash_length``
            ones per row.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != self.n_inputs:
            raise ValueError(
                f"X has width {X.shape[1]}, this hasher expects {self.n_inputs}"
            )
        # Divide each row by its mean, as the fly's receptors do: it makes the
        # code depend on the pattern across inputs rather than overall intensity.
        means = X.mean(axis=1, keepdims=True)
        normalised = X / np.where(means == 0.0, 1.0, means)
        # `normalised @ self.projection.T.toarray()` reads more directly, but
        # `.toarray()` would densify the (n_cells, n_inputs) projection on
        # every call. With the documented default n_cells = 20 * n_inputs,
        # that is a huge, repeated allocation for no benefit — scipy sparse
        # matrices support `sparse @ dense`, so computing it the other way
        # round keeps the projection sparse throughout and returns the same
        # (n_samples, n_cells) result.
        activity = np.asarray(
            (self.projection @ normalised.T).T, dtype=np.float32
        )
        k = self.hash_length
        top = np.argpartition(-activity, kth=k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(X.shape[0], dtype=np.int64), k)
        return sparse.csr_array(
            (np.ones(rows.size, dtype=np.float32), (rows, top.reshape(-1))),
            shape=(X.shape[0], self.n_cells),
        )

    def save(self, path: str | Path) -> None:
        """Save this hasher's wiring so it can be reloaded with :meth:`load`."""
        m = self.projection.tocoo()
        np.savez_compressed(
            path,
            rows=m.row.astype(np.int64),
            cols=m.col.astype(np.int64),
            vals=m.data.astype(np.float32),
            shape=np.array(m.shape, dtype=np.int64),
            hash_length=np.array(self.hash_length, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> FlyHasher:
        """Load a hasher saved with :meth:`save`."""
        z = np.load(path, allow_pickle=False)
        shape = tuple(int(x) for x in z["shape"])
        projection = sparse.csr_array(
            (z["vals"].astype(np.float32), (z["rows"], z["cols"])), shape=shape
        )
        return cls(projection=projection, hash_length=int(z["hash_length"]))


class SimilaritySearch:
    """Nearest-neighbour search over FlyHash codes.

    >>> import numpy as np
    >>> data = np.random.default_rng(0).random((500, 32))
    >>> index = SimilaritySearch.build(data)
    >>> neighbours, overlap = index.query(data[:3], k=5)
    >>> neighbours.shape
    (3, 5)
    """

    def __init__(self, hasher: FlyHasher, codes: sparse.csr_array) -> None:
        self.hasher = hasher
        self.codes = codes

    @classmethod
    def build(cls, X: np.ndarray, hasher: FlyHasher | None = None, **kwargs):
        """Index a dataset. Builds a random hasher sized to X unless given one."""
        X = np.asarray(X, dtype=np.float32)
        if hasher is None:
            hasher = FlyHasher.random(X.shape[1], **kwargs)
        return cls(hasher, hasher.encode(X))

    def __len__(self) -> int:
        return self.codes.shape[0]

    def query(self, X: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Find the k most similar indexed items for each query vector.

        Returns:
            ``(indices, overlap)``, each (n_queries, k). ``overlap`` counts the
            active cells shared with the query — higher is more similar. Ties
            are broken by ascending index, so results are reproducible.
        """
        if not 1 <= k <= len(self):
            raise ValueError(f"k must be between 1 and {len(self)}, got {k}")
        query_codes = self.hasher.encode(X)
        overlap = np.asarray(
            (query_codes @ self.codes.T).todense(), dtype=np.float64
        )
        # overlap is integer-valued, so this key is injective and the ordering
        # is fully determined: most overlap first, lowest index breaking ties.
        n = overlap.shape[1]
        key = overlap * n - np.arange(n, dtype=np.float64)[None, :]
        idx = np.argpartition(-key, kth=k - 1, axis=1)[:, :k]
        order = np.take_along_axis(-key, idx, axis=1).argsort(axis=1)
        indices = np.take_along_axis(idx, order, axis=1).astype(np.int64)
        return indices, np.take_along_axis(overlap, indices, axis=1)


def find_duplicates(
    X: np.ndarray,
    threshold: float = 0.5,
    hasher: FlyHasher | None = None,
    **kwargs,
) -> list[tuple[int, int, float]]:
    """Find pairs of near-identical rows.

    Args:
        X: (n_samples, n_inputs) array.
        threshold: minimum fraction of active cells two rows must share,
            between 0 and 1.

    Returns:
        ``(i, j, similarity)`` triples with ``i < j``, strongest first.

    This builds a dense (n_samples, n_samples) float64 similarity matrix, so
    memory cost is quadratic in ``n_samples``: about 800 MB at 10,000 rows and
    roughly 8 GB at 30,000. Past a few tens of thousands of rows this becomes
    impractical; consider batching or an approximate approach instead.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    X = np.asarray(X, dtype=np.float32)
    if hasher is None:
        hasher = FlyHasher.random(X.shape[1], **kwargs)
    codes = hasher.encode(X)
    overlap = np.asarray((codes @ codes.T).todense(), dtype=np.float64)
    similarity = overlap / hasher.hash_length
    rows, cols = np.triu_indices(X.shape[0], k=1)
    scores = similarity[rows, cols]
    hits = scores >= threshold
    pairs = sorted(
        zip(rows[hits].tolist(), cols[hits].tolist(), scores[hits].tolist()),
        key=lambda t: (-t[2], t[0], t[1]),
    )
    return pairs
