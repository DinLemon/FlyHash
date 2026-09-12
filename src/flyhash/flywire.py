"""Build the study's Circuit from the FlyWire female-brain connectome.

FlyWire publishes three gzipped CSVs rather than the male dataset's two
Feather files, names its columns differently, splits each connection across
neuropils, and keeps cell types in a separate table. This module absorbs all
of that so every downstream stage sees the same Circuit it already knows.

The class names (Kenyon_Cell, ALPN) and the type names (DA1_lPN, M_lPNm11D)
are shared between the two datasets, so glomeruli.py needs no changes.

No synapse-weight threshold is applied: FlyWire's published table already
imposes one at 5 synapses per pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from flyhash.circuit import Circuit

KC_CLASS = "Kenyon_Cell"
PN_CLASS = "ALPN"
APL_TYPE = "APL"
SIDE_MAP = {"left": "L", "right": "R"}


def _annotations(classification_path, cell_types_path) -> pd.DataFrame:
    classification = pd.read_csv(
        classification_path, usecols=["root_id", "class", "side"]
    )
    cell_types = pd.read_csv(
        cell_types_path, usecols=["root_id", "primary_type"]
    )
    merged = classification.merge(cell_types, on="root_id", how="left")
    merged["side"] = merged["side"].map(SIDE_MAP)
    return merged


def extract_flywire_circuit(
    classification_path: str | Path,
    cell_types_path: str | Path,
    connections_path: str | Path,
    min_indegree: int = 2,
) -> Circuit:
    ann = _annotations(classification_path, cell_types_path)
    kc_rows = ann[ann["class"] == KC_CLASS]
    pn_rows = ann[ann["class"] == PN_CLASS]
    apl_ids = ann.loc[ann["primary_type"] == APL_TYPE, "root_id"].to_numpy(np.int64)

    kc_all = set(kc_rows["root_id"])
    pn_all = set(pn_rows["root_id"])

    edges = pd.read_csv(
        connections_path, usecols=["pre_root_id", "post_root_id", "syn_count"]
    )
    edges = edges[edges["post_root_id"].isin(kc_all)]

    apl_edges = edges[edges["pre_root_id"].isin(set(apl_ids.tolist()))]
    apl_edges = apl_edges.groupby("post_root_id", as_index=False)["syn_count"].sum()

    pn_edges = edges[edges["pre_root_id"].isin(pn_all)]
    # One row per (pair, neuropil) upstream; the study wants one per pair.
    pn_edges = pn_edges.groupby(
        ["pre_root_id", "post_root_id"], as_index=False
    )["syn_count"].sum()
    if pn_edges.empty:
        raise ValueError("no PN->KC edges found; check the input files")

    counts = pn_edges.groupby("post_root_id").size()
    kc_keep = np.sort(counts[counts >= min_indegree].index.to_numpy())
    pn_edges = pn_edges[pn_edges["post_root_id"].isin(set(kc_keep.tolist()))]
    pn_keep = np.sort(pn_edges["pre_root_id"].unique())

    kc_index = {int(b): i for i, b in enumerate(kc_keep)}
    pn_index = {int(b): i for i, b in enumerate(pn_keep)}
    rows = pn_edges["post_root_id"].map(kc_index).to_numpy(np.int64)
    cols = pn_edges["pre_root_id"].map(pn_index).to_numpy(np.int64)
    matrix = sparse.csr_array(
        (pn_edges["syn_count"].to_numpy(np.float32), (rows, cols)),
        shape=(len(kc_keep), len(pn_keep)),
    )

    apl_vec = np.zeros(len(kc_keep), dtype=np.float32)
    for body, weight in zip(apl_edges["post_root_id"], apl_edges["syn_count"]):
        position = kc_index.get(int(body))
        if position is not None:
            apl_vec[position] = weight

    kc_meta = kc_rows.set_index("root_id").loc[kc_keep]
    pn_meta = pn_rows.set_index("root_id").loc[pn_keep]

    # Verify all surviving neurons have a valid side (L or R, not NaN from unmapped values)
    kc_invalid = pd.isna(kc_meta["side"]).sum()
    pn_invalid = pd.isna(pn_meta["side"]).sum()
    if kc_invalid > 0 or pn_invalid > 0:
        raise ValueError(
            f"circuit contains {kc_invalid} Kenyon cells and {pn_invalid} "
            f"projection neurons with unmapped side values"
        )

    return Circuit(
        pn_to_kc=matrix,
        kc_ids=kc_keep.astype(np.int64),
        kc_types=kc_meta["primary_type"].to_numpy(dtype=object),
        kc_sides=kc_meta["side"].to_numpy(dtype=object),
        pn_ids=pn_keep.astype(np.int64),
        pn_types=pn_meta["primary_type"].to_numpy(dtype=object),
        pn_sides=pn_meta["side"].to_numpy(dtype=object),
        apl_to_kc=apl_vec,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build circuit-flywire.npz from FlyWire CSVs")
    p.add_argument("--classification", default="data/raw/flywire/classification.csv.gz")
    p.add_argument("--cell-types", default="data/raw/flywire/consolidated_cell_types.csv.gz")
    p.add_argument("--connections", default="data/raw/flywire/connections.csv.gz")
    p.add_argument("--out", default="data/circuit-flywire.npz")
    p.add_argument("--min-indegree", type=int, default=2)
    args = p.parse_args(argv)

    circuit = extract_flywire_circuit(
        args.classification, args.cell_types, args.connections, args.min_indegree
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    circuit.save(args.out)
    for side in ("L", "R"):
        h = circuit.hemisphere(side)
        print(
            f"{side}: n_kc={h.n_kc} n_pn={h.n_pn} edges={h.pn_to_kc.nnz} "
            f"mean_indeg={h.pn_to_kc.nnz / h.n_kc:.2f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
