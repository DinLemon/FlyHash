"""Aggregate projection neurons into their glomeruli.

Several projection neurons of one glomerulus carry the same olfactory
channel, so the glomerulus — not the individual neuron — is the fly's true
input dimensionality. The glomerulus name is the prefix of the PN type
(`DA1_lPN` -> `DA1`); multiglomerular PNs are named `M_...` and have no
single glomerulus, so they are dropped. On the real circuit they carry
about 1% of the PN->KC edges.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from flyhash.circuit import Circuit

MULTIGLOMERULAR_PREFIX = "M_"


def glomerulus_of(pn_type: str | None) -> str | None:
    """Glomerulus name for a PN type, or None if it has no single one."""
    if not pn_type or not isinstance(pn_type, str):
        return None
    if pn_type.startswith(MULTIGLOMERULAR_PREFIX):
        return None
    name = pn_type.split("_")[0]
    return name or None


def aggregate_by_glomerulus(circuit: Circuit) -> tuple[sparse.csr_array, np.ndarray]:
    """Sum the circuit's PN columns within each glomerulus.

    Returns the (n_kc, n_glomeruli) matrix and the sorted glomerulus names.
    """
    gloms = [glomerulus_of(t) for t in circuit.pn_types]
    keep = [i for i, g in enumerate(gloms) if g is not None]
    if not keep:
        raise ValueError("no uniglomerular PNs in this circuit")

    names = sorted({gloms[i] for i in keep})
    position = {name: j for j, name in enumerate(names)}
    rows = np.array(keep, dtype=np.int64)
    cols = np.array([position[gloms[i]] for i in keep], dtype=np.int64)
    grouping = sparse.csr_array(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)),
        shape=(circuit.n_pn, len(names)),
    )
    aggregated = sparse.csr_array(circuit.pn_to_kc @ grouping)
    return aggregated, np.array(names, dtype=object)
