"""The Circuit container: the only thing downstream code sees of the raw data."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy import sparse

SIDES = ("L", "R")


@dataclass(frozen=True)
class Circuit:
    """A PN->KC subcircuit with its metadata.

    pn_to_kc holds synapse counts, so rows are KCs and columns are PNs.
    apl_to_kc holds the APL inhibitory synapse count onto each KC.
    """

    pn_to_kc: sparse.csr_array
    kc_ids: np.ndarray
    kc_types: np.ndarray
    kc_sides: np.ndarray
    pn_ids: np.ndarray
    pn_types: np.ndarray
    pn_sides: np.ndarray
    apl_to_kc: np.ndarray

    @property
    def n_kc(self) -> int:
        return self.pn_to_kc.shape[0]

    @property
    def n_pn(self) -> int:
        return self.pn_to_kc.shape[1]

    def binary(self) -> sparse.csr_array:
        """Same structure, every stored value set to 1.0."""
        out = self.pn_to_kc.copy()
        out.data = np.ones_like(out.data, dtype=np.float32)
        return out

    def hemisphere(self, side: str) -> Circuit:
        """Keep only KCs whose soma is on `side`, and only the PN columns that
        actually reach one of those KCs. Each hemisphere is analysed
        independently with its own compressor, so a shared input axis layout
        with the other hemisphere is not needed."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        mask = self.kc_sides == side
        idx = np.flatnonzero(mask)
        sub = sparse.csr_array(self.pn_to_kc[idx, :])
        col_degrees = np.asarray((sub != 0).sum(axis=0)).ravel()
        col_idx = np.flatnonzero(col_degrees)
        return Circuit(
            pn_to_kc=sparse.csr_array(sub[:, col_idx]),
            kc_ids=self.kc_ids[idx],
            kc_types=self.kc_types[idx],
            kc_sides=self.kc_sides[idx],
            pn_ids=self.pn_ids[col_idx],
            pn_types=self.pn_types[col_idx],
            pn_sides=self.pn_sides[col_idx],
            apl_to_kc=self.apl_to_kc[idx],
        )

    def save(self, path: str | Path) -> None:
        m = self.pn_to_kc.tocoo()
        np.savez_compressed(
            path,
            rows=m.row.astype(np.int64),
            cols=m.col.astype(np.int64),
            vals=m.data.astype(np.float32),
            shape=np.array(m.shape, dtype=np.int64),
            kc_ids=self.kc_ids,
            kc_types=self.kc_types.astype(str),
            kc_sides=self.kc_sides.astype(str),
            pn_ids=self.pn_ids,
            pn_types=self.pn_types.astype(str),
            pn_sides=self.pn_sides.astype(str),
            apl_to_kc=self.apl_to_kc,
        )

    @classmethod
    def load(cls, path: str | Path) -> Circuit:
        z = np.load(path, allow_pickle=False)
        shape = tuple(int(x) for x in z["shape"])
        m = sparse.csr_array(
            (z["vals"].astype(np.float32), (z["rows"], z["cols"])), shape=shape
        )
        return cls(
            pn_to_kc=m,
            kc_ids=z["kc_ids"],
            kc_types=z["kc_types"].astype(object),
            kc_sides=z["kc_sides"].astype(object),
            pn_ids=z["pn_ids"],
            pn_types=z["pn_types"].astype(object),
            pn_sides=z["pn_sides"].astype(object),
            apl_to_kc=z["apl_to_kc"].astype(np.float32),
        )
