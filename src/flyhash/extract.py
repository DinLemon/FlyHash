"""Distil the 1 GB connectome dump into a few-megabyte Circuit.

The weights file has ~152 million rows, so it is streamed in Arrow record
batches rather than loaded whole.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy import sparse

from flyhash.circuit import Circuit

KC_CLASS = "Kenyon_Cell"
PN_CLASS = "ALPN"
APL_TYPE = "APL"


def _stream_edges(weights_path, min_weight, pre_keep, post_keep):
    """Yield (pre, post, weight) arrays for edges whose endpoints are wanted."""
    with pa.memory_map(str(weights_path)) as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            w = batch.column("weight").to_numpy()
            keep = w >= min_weight
            if not keep.any():
                continue
            pre = batch.column("body_pre").to_numpy()[keep]
            post = batch.column("body_post").to_numpy()[keep]
            w = w[keep]
            sel = np.isin(pre, pre_keep) & np.isin(post, post_keep)
            if sel.any():
                yield pre[sel], post[sel], w[sel]


def extract_circuit(
    annotations_path: str | Path,
    weights_path: str | Path,
    min_weight: int = 3,
    min_indegree: int = 2,
) -> Circuit:
    ann = pd.read_feather(
        annotations_path, columns=["bodyId", "type", "class", "somaSide"]
    )
    kc_rows = ann[ann["class"] == KC_CLASS]
    pn_rows = ann[ann["class"] == PN_CLASS]
    apl_ids = ann.loc[ann["type"] == APL_TYPE, "bodyId"].to_numpy(dtype=np.int64)

    kc_all = kc_rows["bodyId"].to_numpy(dtype=np.int64)
    pn_all = pn_rows["bodyId"].to_numpy(dtype=np.int64)
    sources = np.concatenate([pn_all, apl_ids])

    pres, posts, weights = [], [], []
    for pre, post, w in _stream_edges(weights_path, min_weight, sources, kc_all):
        pres.append(pre)
        posts.append(post)
        weights.append(w)
    if not pres:
        raise ValueError("no PN->KC edges found; check the input files")
    pre = np.concatenate(pres)
    post = np.concatenate(posts)
    weight = np.concatenate(weights).astype(np.float32)

    is_apl = np.isin(pre, apl_ids)
    apl_pre, apl_post, apl_w = pre[is_apl], post[is_apl], weight[is_apl]
    pre, post, weight = pre[~is_apl], post[~is_apl], weight[~is_apl]

    # Drop KCs with too few distinct PN inputs, then drop PNs left with no targets.
    # Count distinct (pre, post) pairs to measure in-degree correctly, not total rows.
    unique_pairs = np.unique(np.column_stack([pre, post]), axis=0)
    ids, counts = np.unique(unique_pairs[:, 1], return_counts=True)
    kc_keep = ids[counts >= min_indegree]
    sel = np.isin(post, kc_keep)
    pre, post, weight = pre[sel], post[sel], weight[sel]
    pn_keep = np.unique(pre)

    kc_index = {int(b): i for i, b in enumerate(kc_keep)}
    pn_index = {int(b): i for i, b in enumerate(pn_keep)}
    rows = np.array([kc_index[int(b)] for b in post], dtype=np.int64)
    cols = np.array([pn_index[int(b)] for b in pre], dtype=np.int64)
    matrix = sparse.csr_array(
        (weight, (rows, cols)), shape=(len(kc_keep), len(pn_keep))
    )
    matrix.sum_duplicates()

    apl_vec = np.zeros(len(kc_keep), dtype=np.float32)
    for b, w in zip(apl_post, apl_w):
        i = kc_index.get(int(b))
        if i is not None:
            apl_vec[i] += w

    kc_meta = kc_rows.set_index("bodyId").loc[kc_keep]
    pn_meta = pn_rows.set_index("bodyId").loc[pn_keep]
    return Circuit(
        pn_to_kc=matrix,
        kc_ids=kc_keep,
        kc_types=kc_meta["type"].to_numpy(dtype=object),
        kc_sides=kc_meta["somaSide"].to_numpy(dtype=object),
        pn_ids=pn_keep,
        pn_types=pn_meta["type"].to_numpy(dtype=object),
        pn_sides=pn_meta["somaSide"].to_numpy(dtype=object),
        apl_to_kc=apl_vec,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build circuit.npz from raw Feather files")
    p.add_argument("--annotations", default="data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
    p.add_argument("--weights", default="data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    p.add_argument("--out", default="data/circuit.npz")
    p.add_argument("--min-weight", type=int, default=3)
    p.add_argument("--min-indegree", type=int, default=2)
    args = p.parse_args(argv)

    circuit = extract_circuit(
        args.annotations, args.weights, args.min_weight, args.min_indegree
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    circuit.save(args.out)
    for side in ("L", "R"):
        h = circuit.hemisphere(side)
        used = np.unique(h.pn_to_kc.tocoo().col).size
        print(
            f"{side}: n_kc={h.n_kc} n_pn_used={used} "
            f"edges={h.pn_to_kc.nnz} mean_indeg={h.pn_to_kc.nnz / h.n_kc:.2f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
