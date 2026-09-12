# FlyHash Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer, on MNIST, whether the measured PN→KC wiring of the *Drosophila* mushroom body beats degree-matched random wiring at nearest-neighbour retrieval.

**Architecture:** A four-stage pipeline — `extract` distils two Feather files (1 GB) into a few-megabyte `circuit.npz`; `models` builds five same-shaped projection matrices from it; `encode` turns data vectors into sparse binary codes through any projection; `bench` scores codes against brute-force ground truth. Each stage is a separate module with a narrow interface so it can be tested alone.

**Tech Stack:** Python 3.11+, numpy, scipy.sparse, pyarrow, pandas, pytest. No deep-learning frameworks — every operation is a sparse matmul or a sort.

## Global Constraints

Copied verbatim from the spec (`docs/superpowers/specs/2026-09-11-flyhash-connectome-design.md` §6). These are **pre-registered**: they must not be changed after the first benchmark run.

- Synapse weight threshold: **≥ 3**
- KC exclusion: PN in-degree **< 2** at threshold ≥3 is dropped
- Hash length: **5%** of surviving KCs (93 left, 94 right)
- Database size: **10 000** items
- Query count: **1000**, seed **1**
- Ground truth: top-**100** true neighbours by **Euclidean** distance (MNIST)
- Metric: **mAP@100**
- Shuffles: **100**, seeds **1000–1099**
- Significance: FLY beats SHUFFLED only if its mAP exceeds the **97.5th percentile** of the 100 shuffles
- Compressor: **Gaussian random projection, seed 0**, one per hemisphere, **shared by all five models**
- Unit of analysis: **hemisphere**. L and R are analysed separately end to end.

Verified circuit facts (measured, not assumed) that tests assert against:

| Hemisphere | KC | PN | Edges | Mean in-degree |
|---|---|---|---|---|
| L | 1865 | 149 | 10 395 | 5.57 |
| R | 1875 | 143 | 10 341 | 5.52 |

---

### Task 1: Project scaffolding and the `Circuit` container

The `Circuit` object is the boundary between the 1 GB raw data and everything else. Every later task consumes it and nothing else, so it comes first.

**Files:**
- Create: `pyproject.toml`
- Create: `src/flyhash/__init__.py`
- Create: `src/flyhash/circuit.py`
- Test: `tests/test_circuit.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Circuit` dataclass with fields `pn_to_kc: scipy.sparse.csr_array` (shape `(n_kc, n_pn)`, dtype float32, values = synapse counts), `kc_ids: np.ndarray[int64]`, `kc_types: np.ndarray[object]`, `kc_sides: np.ndarray[object]`, `pn_ids: np.ndarray[int64]`, `pn_types: np.ndarray[object]`, `pn_sides: np.ndarray[object]`, `apl_to_kc: np.ndarray[float32]` (shape `(n_kc,)`). Methods `Circuit.save(path) -> None`, `Circuit.load(path) -> Circuit` (classmethod), `Circuit.hemisphere(side: str) -> Circuit`, properties `n_kc: int`, `n_pn: int`, `binary() -> scipy.sparse.csr_array` (same shape, all stored values 1.0).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "flyhash"
version = "0.1.0"
description = "Does the measured mushroom-body wiring hash better than random?"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pyarrow>=15",
    "pandas>=2.1",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the empty package marker**

```bash
mkdir -p "G:/C#/flyhash/src/flyhash" "G:/C#/flyhash/tests"
printf '' > "G:/C#/flyhash/src/flyhash/__init__.py"
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_circuit.py`:

```python
import numpy as np
import pytest
from scipy import sparse

from flyhash.circuit import Circuit


def make_toy_circuit():
    """Three KCs (L, L, R) and two PNs (L, R). KC0<-PN0, KC1<-PN0+PN1, KC2<-PN1."""
    m = sparse.csr_array(
        np.array([[5.0, 0.0], [7.0, 3.0], [0.0, 9.0]], dtype=np.float32)
    )
    return Circuit(
        pn_to_kc=m,
        kc_ids=np.array([10, 11, 12], dtype=np.int64),
        kc_types=np.array(["KCg-m", "KCab-s", "KCg-m"], dtype=object),
        kc_sides=np.array(["L", "L", "R"], dtype=object),
        pn_ids=np.array([20, 21], dtype=np.int64),
        pn_types=np.array(["DA1_lPN", "VA1v_adPN"], dtype=object),
        pn_sides=np.array(["L", "R"], dtype=object),
        apl_to_kc=np.array([40.0, 41.0, 42.0], dtype=np.float32),
    )


def test_shape_properties():
    c = make_toy_circuit()
    assert c.n_kc == 3
    assert c.n_pn == 2


def test_binary_discards_weights_but_keeps_structure():
    c = make_toy_circuit()
    b = c.binary()
    assert b.shape == (3, 2)
    assert set(np.unique(b.data).tolist()) == {1.0}
    assert b.nnz == c.pn_to_kc.nnz


def test_hemisphere_selects_kcs_and_keeps_all_pn_columns():
    c = make_toy_circuit()
    left = c.hemisphere("L")
    assert left.n_kc == 2
    assert left.n_pn == 2
    assert left.kc_ids.tolist() == [10, 11]
    assert left.apl_to_kc.tolist() == [40.0, 41.0]
    np.testing.assert_array_equal(
        left.pn_to_kc.toarray(), np.array([[5.0, 0.0], [7.0, 3.0]], dtype=np.float32)
    )


def test_hemisphere_rejects_unknown_side():
    c = make_toy_circuit()
    with pytest.raises(ValueError, match="side must be"):
        c.hemisphere("X")


def test_save_load_roundtrip(tmp_path):
    c = make_toy_circuit()
    p = tmp_path / "circuit.npz"
    c.save(p)
    back = Circuit.load(p)
    np.testing.assert_array_equal(back.pn_to_kc.toarray(), c.pn_to_kc.toarray())
    np.testing.assert_array_equal(back.kc_ids, c.kc_ids)
    np.testing.assert_array_equal(back.kc_types, c.kc_types)
    np.testing.assert_array_equal(back.kc_sides, c.kc_sides)
    np.testing.assert_array_equal(back.pn_ids, c.pn_ids)
    np.testing.assert_array_equal(back.pn_types, c.pn_types)
    np.testing.assert_array_equal(back.pn_sides, c.pn_sides)
    np.testing.assert_array_equal(back.apl_to_kc, c.apl_to_kc)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_circuit.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.circuit'`

- [ ] **Step 5: Write `src/flyhash/circuit.py`**

```python
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
        """Keep only KCs whose soma is on `side`. All PN columns are kept so
        that the left and right circuits share an input axis layout."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        mask = self.kc_sides == side
        idx = np.flatnonzero(mask)
        return Circuit(
            pn_to_kc=sparse.csr_array(self.pn_to_kc[idx, :]),
            kc_ids=self.kc_ids[idx],
            kc_types=self.kc_types[idx],
            kc_sides=self.kc_sides[idx],
            pn_ids=self.pn_ids,
            pn_types=self.pn_types,
            pn_sides=self.pn_sides,
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
```

- [ ] **Step 6: Install the package and run the tests**

Run:
```bash
cd "G:/C#/flyhash" && pip install -e ".[dev]" && python -m pytest tests/test_circuit.py -v
```
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
cd "G:/C#/flyhash" && git add pyproject.toml src tests && git commit -m "feat: add Circuit container with save/load and hemisphere split"
```

---

### Task 2: Extract the circuit from the raw Feather files

The weights file is 1 GB and 151 856 684 rows. It must be streamed in Arrow record batches (2318 of them, 65 536 rows each) — loading it whole costs roughly 3.5 GB of RAM and is unnecessary.

**Files:**
- Create: `src/flyhash/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: `Circuit` from Task 1
- Produces: `extract_circuit(annotations_path: str | Path, weights_path: str | Path, min_weight: int = 3, min_indegree: int = 2) -> Circuit`, and `main(argv: list[str] | None = None) -> int` for CLI use.

- [ ] **Step 1: Write the failing test**

Create `tests/test_extract.py`. It builds tiny Feather files in the same schema as the real ones, so it runs in milliseconds and needs no 1 GB download.

```python
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flyhash.extract import extract_circuit


@pytest.fixture
def toy_files(tmp_path):
    """KC 100,101 on L; KC 102 on R; KC 103 is under-connected and must be dropped.
    PN 200,201 reach KCs; PN 202 reaches nothing and must be dropped."""
    ann = pd.DataFrame(
        {
            "bodyId": [100, 101, 102, 103, 200, 201, 202, 900],
            "type": [
                "KCg-m", "KCab-s", "KCg-m", "KCab-p",
                "DA1_lPN", "VA1v_adPN", "DL3_lPN", "APL",
            ],
            "class": [
                "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell",
                "ALPN", "ALPN", "ALPN", None,
            ],
            "somaSide": ["L", "L", "R", "L", "L", "R", "L", "L"],
        }
    )
    ann_path = tmp_path / "ann.feather"
    ann.to_feather(ann_path)

    # weight 2 on (201 -> 101) is below threshold and must be ignored.
    edges = pd.DataFrame(
        {
            "body_pre": [200, 201, 200, 201, 201, 202, 900, 900, 900, 100],
            "body_post": [100, 100, 101, 101, 102, 999, 100, 101, 102, 101],
            "weight": [5, 4, 6, 2, 7, 8, 40, 41, 42, 99],
        }
    ).astype({"body_pre": "int64", "body_post": "int64", "weight": "int64"})
    w_path = tmp_path / "w.feather"
    feather.write_feather(pa.Table.from_pandas(edges, preserve_index=False), w_path)
    return ann_path, w_path


def test_applies_weight_threshold_and_indegree_filter(toy_files):
    ann_path, w_path = toy_files
    c = extract_circuit(ann_path, w_path, min_weight=3, min_indegree=2)
    # KC100 keeps 2 inputs; KC101 keeps only PN200 (the 201 edge is weight 2)
    # so it has in-degree 1 and is dropped; KC102 has in-degree 1, dropped too.
    assert c.kc_ids.tolist() == [100]
    assert c.n_kc == 1


def test_keeps_only_pns_that_reach_kcs(toy_files):
    ann_path, w_path = toy_files
    c = extract_circuit(ann_path, w_path, min_weight=3, min_indegree=1)
    assert sorted(c.pn_ids.tolist()) == [200, 201]


def test_weights_are_synapse_counts(toy_files):
    ann_path, w_path = toy_files
    c = extract_circuit(ann_path, w_path, min_weight=3, min_indegree=1)
    kc = {int(b): i for i, b in enumerate(c.kc_ids)}
    pn = {int(b): i for i, b in enumerate(c.pn_ids)}
    dense = c.pn_to_kc.toarray()
    assert dense[kc[100], pn[200]] == 5.0
    assert dense[kc[100], pn[201]] == 4.0
    assert dense[kc[101], pn[201]] == 0.0  # below threshold


def test_apl_vector_is_collected(toy_files):
    ann_path, w_path = toy_files
    c = extract_circuit(ann_path, w_path, min_weight=3, min_indegree=1)
    kc = {int(b): i for i, b in enumerate(c.kc_ids)}
    assert c.apl_to_kc[kc[100]] == 40.0
    assert c.apl_to_kc[kc[102]] == 42.0


def test_kc_to_kc_edges_are_ignored(toy_files):
    """Edge 100 -> 101 has weight 99 but KC->KC is not a PN input."""
    ann_path, w_path = toy_files
    c = extract_circuit(ann_path, w_path, min_weight=3, min_indegree=1)
    assert 100 not in c.pn_ids.tolist()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_extract.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.extract'`

- [ ] **Step 3: Write `src/flyhash/extract.py`**

```python
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

    # Drop KCs with too few PN inputs, then drop PNs left with no targets.
    ids, counts = np.unique(post, return_counts=True)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_extract.py -v`
Expected: 5 passed

- [ ] **Step 5: Build the real circuit and check it against the measured facts**

Run: `cd "G:/C#/flyhash" && python -m flyhash.extract`
Expected output (these numbers are the verified ground truth from the spec — if they differ, stop and investigate rather than adjusting the plan):
```
L: n_kc=1865 n_pn_used=149 edges=10395 mean_indeg=5.57
R: n_kc=1875 n_pn_used=143 edges=10341 mean_indeg=5.52
wrote data/circuit.npz
```

- [ ] **Step 6: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/extract.py tests/test_extract.py data/circuit.npz && git commit -m "feat: extract PN->KC circuit from streamed connectome dump"
```

Note: `data/circuit.npz` is a few megabytes and **is** committed — it makes the whole study reproducible without the 1 GB download. Confirm `.gitignore` excludes only `data/raw/`, not `data/`.

---

### Task 3: Encoding — normalise, compress, winner-take-all

**Files:**
- Create: `src/flyhash/encode.py`
- Test: `tests/test_encode.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `normalize(X: np.ndarray) -> np.ndarray`, `make_compressor(d_in: int, d_out: int, seed: int) -> np.ndarray` returning shape `(d_in, d_out)` float32, `winner_take_all(A: np.ndarray, k: int) -> scipy.sparse.csr_array` returning a boolean-valued float32 CSR of shape `A.shape`, and `encode(X: np.ndarray, compressor: np.ndarray, projection: scipy.sparse.csr_array, k: int) -> scipy.sparse.csr_array`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_encode.py`:

```python
import numpy as np
import pytest
from scipy import sparse

from flyhash.encode import encode, make_compressor, normalize, winner_take_all


def test_normalize_gives_each_row_unit_mean():
    X = np.array([[1.0, 3.0], [2.0, 2.0]], dtype=np.float32)
    out = normalize(X)
    np.testing.assert_allclose(out.mean(axis=1), np.ones(2), rtol=1e-6)


def test_normalize_leaves_all_zero_rows_alone():
    X = np.array([[0.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    out = normalize(X)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[0], np.zeros(2, dtype=np.float32))


def test_compressor_is_deterministic_for_a_seed():
    a = make_compressor(8, 4, seed=0)
    b = make_compressor(8, 4, seed=0)
    c = make_compressor(8, 4, seed=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (8, 4)


def test_wta_keeps_exactly_k_per_row():
    A = np.array([[5.0, 1.0, 3.0, 2.0], [0.0, 9.0, 8.0, 7.0]], dtype=np.float32)
    out = winner_take_all(A, k=2)
    assert isinstance(out, sparse.csr_array)
    np.testing.assert_array_equal(out.sum(axis=1), np.array([2.0, 2.0]))


def test_wta_keeps_the_largest_entries():
    A = np.array([[5.0, 1.0, 3.0, 2.0]], dtype=np.float32)
    out = winner_take_all(A, k=2).toarray()
    np.testing.assert_array_equal(out, np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32))


def test_wta_rejects_k_larger_than_width():
    A = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="k must be"):
        winner_take_all(A, k=4)


def test_encode_produces_k_active_units_per_row():
    rng = np.random.default_rng(0)
    X = rng.random((6, 10), dtype=np.float32)
    comp = make_compressor(10, 5, seed=0)
    proj = sparse.csr_array(rng.integers(0, 2, (12, 5)).astype(np.float32))
    codes = encode(X, comp, proj, k=3)
    assert codes.shape == (6, 12)
    np.testing.assert_array_equal(codes.sum(axis=1), np.full(6, 3.0))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_encode.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.encode'`

- [ ] **Step 3: Write `src/flyhash/encode.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_encode.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/encode.py tests/test_encode.py && git commit -m "feat: add normalise, random compressor and winner-take-all encoding"
```

---

### Task 4: The projection models — fly, uniform, LSH

The degree-preserving shuffle is deliberately left to Task 5: it is the control the whole study rests on and deserves its own review gate.

**Files:**
- Create: `src/flyhash/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `Circuit` from Task 1
- Produces: `fly_projection(circuit: Circuit) -> scipy.sparse.csr_array` (shape `(n_kc, n_pn)`, binary), `uniform_projection(n_kc: int, n_pn: int, n_claws: int, seed: int) -> scipy.sparse.csr_array`, `lsh_projection(n_kc: int, n_pn: int, seed: int) -> scipy.sparse.csr_array` (dense Gaussian stored as CSR so every model shares one type), `hash_length(n_kc: int, fraction: float = 0.05) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
import numpy as np
import pytest
from scipy import sparse

from flyhash.circuit import Circuit
from flyhash.models import (
    fly_projection,
    hash_length,
    lsh_projection,
    uniform_projection,
)


def make_toy_circuit():
    m = sparse.csr_array(
        np.array([[5.0, 0.0], [7.0, 3.0], [0.0, 9.0]], dtype=np.float32)
    )
    return Circuit(
        pn_to_kc=m,
        kc_ids=np.array([10, 11, 12], dtype=np.int64),
        kc_types=np.array(["KCg-m", "KCab-s", "KCg-m"], dtype=object),
        kc_sides=np.array(["L", "L", "R"], dtype=object),
        pn_ids=np.array([20, 21], dtype=np.int64),
        pn_types=np.array(["DA1_lPN", "VA1v_adPN"], dtype=object),
        pn_sides=np.array(["L", "R"], dtype=object),
        apl_to_kc=np.array([40.0, 41.0, 42.0], dtype=np.float32),
    )


def test_fly_projection_is_binary_and_keeps_structure():
    p = fly_projection(make_toy_circuit())
    np.testing.assert_array_equal(
        p.toarray(), np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32)
    )


def test_uniform_projection_gives_every_row_exactly_n_claws():
    p = uniform_projection(n_kc=50, n_pn=20, n_claws=6, seed=0)
    assert p.shape == (50, 20)
    np.testing.assert_array_equal(p.sum(axis=1), np.full(50, 6.0))


def test_uniform_projection_is_deterministic_for_a_seed():
    a = uniform_projection(30, 10, 4, seed=7).toarray()
    b = uniform_projection(30, 10, 4, seed=7).toarray()
    c = uniform_projection(30, 10, 4, seed=8).toarray()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_uniform_projection_rejects_too_many_claws():
    with pytest.raises(ValueError, match="n_claws"):
        uniform_projection(10, 3, n_claws=5, seed=0)


def test_lsh_projection_is_dense_and_shaped_right():
    p = lsh_projection(n_kc=40, n_pn=12, seed=0)
    assert p.shape == (40, 12)
    assert p.nnz == 40 * 12


def test_hash_length_is_five_percent_rounded():
    assert hash_length(1865) == 93
    assert hash_length(1875) == 94
    assert hash_length(10) == 1  # never returns zero
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_models.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.models'`

- [ ] **Step 3: Write `src/flyhash/models.py`**

```python
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
    return max(1, int(round(fraction * n_kc)))


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_models.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/models.py tests/test_models.py && git commit -m "feat: add fly, uniform and LSH projection models"
```

---

### Task 5: The degree-preserving shuffle

This is the control that separates "the fly has an uneven degree distribution" from "the fly picked specific partners". If it is wrong, the study answers nothing. It must preserve **both** degree sequences: how many inputs each KC has, and how many KCs each PN feeds.

**Files:**
- Modify: `src/flyhash/models.py` (append)
- Modify: `tests/test_models.py` (append)

**Interfaces:**
- Consumes: `fly_projection` output from Task 4
- Produces: `shuffled_projection(matrix: scipy.sparse.csr_array, seed: int, swaps_per_edge: int = 10) -> scipy.sparse.csr_array`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
from flyhash.models import shuffled_projection


def _fly_like_matrix(seed=0, n_kc=200, n_pn=30):
    """A matrix with a deliberately uneven in-degree distribution."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(n_kc):
        claws = rng.integers(2, 9)
        for c in rng.choice(n_pn, size=claws, replace=False):
            rows.append(i)
            cols.append(c)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_array((data, (rows, cols)), shape=(n_kc, n_pn))


def test_shuffle_preserves_both_degree_sequences():
    m = _fly_like_matrix()
    s = shuffled_projection(m, seed=1000)
    np.testing.assert_array_equal(s.sum(axis=1), m.sum(axis=1))
    np.testing.assert_array_equal(s.sum(axis=0), m.sum(axis=0))


def test_shuffle_preserves_edge_count_and_stays_binary():
    m = _fly_like_matrix()
    s = shuffled_projection(m, seed=1000)
    assert s.nnz == m.nnz
    assert set(np.unique(s.data).tolist()) == {1.0}


def test_shuffle_never_creates_a_duplicate_edge():
    m = _fly_like_matrix()
    s = shuffled_projection(m, seed=1000).tocoo()
    pairs = set(zip(s.row.tolist(), s.col.tolist()))
    assert len(pairs) == s.nnz


def test_shuffle_actually_rewires_something():
    m = _fly_like_matrix()
    s = shuffled_projection(m, seed=1000)
    assert not np.array_equal(s.toarray(), m.toarray())


def test_shuffle_is_deterministic_for_a_seed():
    m = _fly_like_matrix()
    a = shuffled_projection(m, seed=1000).toarray()
    b = shuffled_projection(m, seed=1000).toarray()
    c = shuffled_projection(m, seed=1001).toarray()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_models.py -k shuffle -v`
Expected: collection error, `ImportError: cannot import name 'shuffled_projection'`

- [ ] **Step 3: Append the implementation to `src/flyhash/models.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_models.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/models.py tests/test_models.py && git commit -m "feat: add degree-preserving shuffle control"
```

---

### Task 6: MNIST loader

**Files:**
- Create: `src/flyhash/datasets.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: nothing
- Produces: `load_mnist(cache_dir: str | Path = "data/datasets") -> tuple[np.ndarray, np.ndarray]` returning `(database, queries)` of shapes `(10000, 784)` and `(1000, 784)`, dtype float32.

The database comes from the MNIST *training* split and the queries from the *test* split, so no query is its own nearest neighbour.

- [ ] **Step 1: Write the failing test**

Create `tests/test_datasets.py`:

```python
import gzip
import struct

import numpy as np
import pytest

from flyhash.datasets import load_mnist, read_idx_images


def test_read_idx_images_parses_the_header_and_pixels(tmp_path):
    images = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    path = tmp_path / "toy-idx3-ubyte.gz"
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, 2, 4, 3))
        f.write(images.tobytes())
    out = read_idx_images(path)
    assert out.shape == (2, 12)
    np.testing.assert_array_equal(out, images.reshape(2, 12).astype(np.float32))


def test_read_idx_images_rejects_a_bad_magic_number(tmp_path):
    path = tmp_path / "bad-idx3-ubyte.gz"
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 1234, 1, 1, 1))
        f.write(b"\x00")
    with pytest.raises(ValueError, match="magic"):
        read_idx_images(path)


@pytest.mark.network
def test_load_mnist_returns_the_preregistered_shapes(tmp_path):
    db, queries = load_mnist(cache_dir=tmp_path)
    assert db.shape == (10000, 784)
    assert queries.shape == (1000, 784)
    assert db.dtype == np.float32
    assert db.max() <= 255.0
```

- [ ] **Step 2: Register the network marker**

Append to `pyproject.toml`, inside the existing `[tool.pytest.ini_options]` table:

```toml
markers = ["network: needs to download data"]
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_datasets.py -v -m "not network"`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.datasets'`

- [ ] **Step 4: Write `src/flyhash/datasets.py`**

```python
"""Benchmark datasets. Phase 1 needs MNIST only."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"

DATABASE_SIZE = 10_000
QUERY_COUNT = 1_000
IDX_IMAGE_MAGIC = 2051


def read_idx_images(path: str | Path) -> np.ndarray:
    """Read a gzipped IDX3 image file into a (n, pixels) float32 array."""
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != IDX_IMAGE_MAGIC:
            raise ValueError(f"bad magic number {magic}, expected {IDX_IMAGE_MAGIC}")
        buf = f.read(count * rows * cols)
    flat = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    return flat.reshape(count, rows * cols)


def _fetch(name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name
    if not target.exists():
        urllib.request.urlretrieve(MNIST_BASE + name, target)
    return target


def load_mnist(cache_dir: str | Path = "data/datasets") -> tuple[np.ndarray, np.ndarray]:
    """Return (database, queries).

    The database is the first 10 000 training images and the queries are the
    first 1000 test images, so a query is never its own nearest neighbour.
    """
    cache = Path(cache_dir)
    db = read_idx_images(_fetch(TRAIN_IMAGES, cache))[:DATABASE_SIZE]
    queries = read_idx_images(_fetch(TEST_IMAGES, cache))[:QUERY_COUNT]
    return db, queries
```

- [ ] **Step 5: Run the offline tests, then the network test**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_datasets.py -v -m "not network"`
Expected: 2 passed, 1 deselected

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_datasets.py -v`
Expected: 3 passed (downloads ~11 MB on first run)

- [ ] **Step 6: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/datasets.py tests/test_datasets.py pyproject.toml && git commit -m "feat: add MNIST loader with IDX parsing"
```

---

### Task 7: Benchmark — ground truth and mAP@100

**Files:**
- Create: `src/flyhash/bench.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure numpy)
- Produces: `true_neighbours(database: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray` of shape `(n_queries, k)` int64, `rank_by_code_overlap(db_codes, query_codes, k: int) -> np.ndarray` of shape `(n_queries, k)` int64, `average_precision(retrieved: np.ndarray, relevant: set[int]) -> float`, `mean_average_precision(retrieved: np.ndarray, truth: np.ndarray) -> float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bench.py`:

```python
import numpy as np
from scipy import sparse

from flyhash.bench import (
    average_precision,
    mean_average_precision,
    rank_by_code_overlap,
    true_neighbours,
)


def test_true_neighbours_finds_the_closest_points():
    db = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    queries = np.array([[0.1, 0.0]], dtype=np.float32)
    out = true_neighbours(db, queries, k=2)
    assert out.shape == (1, 2)
    assert out[0].tolist() == [0, 1]


def test_rank_by_code_overlap_prefers_the_most_shared_bits():
    db = sparse.csr_array(
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    )
    q = sparse.csr_array(np.array([[1, 1, 0]], dtype=np.float32))
    out = rank_by_code_overlap(db, q, k=3)
    assert out[0][:2].tolist() == [0, 1]


def test_average_precision_is_one_when_everything_relevant_comes_first():
    retrieved = np.array([3, 1, 7, 9])
    assert average_precision(retrieved, {3, 1}) == 1.0


def test_average_precision_is_zero_when_nothing_is_relevant():
    retrieved = np.array([3, 1, 7, 9])
    assert average_precision(retrieved, {42}) == 0.0


def test_average_precision_handles_a_partial_hit():
    # relevant item at rank 2 only: precision 1/2, divided by |relevant| = 1
    retrieved = np.array([5, 3])
    assert average_precision(retrieved, {3}) == 0.5


def test_mean_average_precision_averages_over_queries():
    retrieved = np.array([[1, 2], [3, 4]])
    truth = np.array([[1, 2], [9, 8]])
    assert mean_average_precision(retrieved, truth) == 0.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_bench.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.bench'`

- [ ] **Step 3: Write `src/flyhash/bench.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_bench.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/bench.py tests/test_bench.py && git commit -m "feat: add brute-force ground truth and mAP scoring"
```

---

### Task 8: The Phase 1 run — wire it together and get the answer

**Files:**
- Create: `src/flyhash/phase1.py`
- Test: `tests/test_phase1.py`

**Interfaces:**
- Consumes: everything above
- Produces: `run_hemisphere(circuit: Circuit, side: str, database: np.ndarray, queries: np.ndarray, n_shuffles: int = 100) -> dict`, `verdict(fly_map: float, shuffle_maps: list[float]) -> str`, `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase1.py`:

```python
import numpy as np
import pytest
from scipy import sparse

from flyhash.circuit import Circuit
from flyhash.phase1 import run_hemisphere, verdict


def make_small_circuit(n_kc=60, n_pn=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(n_kc):
        for c in rng.choice(n_pn, size=rng.integers(2, 6), replace=False):
            rows.append(i)
            cols.append(c)
    m = sparse.csr_array(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_kc, n_pn)
    )
    return Circuit(
        pn_to_kc=m,
        kc_ids=np.arange(n_kc, dtype=np.int64),
        kc_types=np.array(["KCg-m"] * n_kc, dtype=object),
        kc_sides=np.array(["L"] * n_kc, dtype=object),
        pn_ids=np.arange(n_pn, dtype=np.int64),
        pn_types=np.array(["DA1_lPN"] * n_pn, dtype=object),
        pn_sides=np.array(["L"] * n_pn, dtype=object),
        apl_to_kc=np.ones(n_kc, dtype=np.float32),
    )


def test_verdict_reports_fly_better_above_the_threshold():
    shuffles = list(np.linspace(0.10, 0.20, 100))
    assert verdict(0.99, shuffles) == "fly_better"


def test_verdict_reports_fly_worse_below_the_threshold():
    shuffles = list(np.linspace(0.10, 0.20, 100))
    assert verdict(0.01, shuffles) == "fly_worse"


def test_verdict_reports_no_difference_inside_the_band():
    shuffles = list(np.linspace(0.10, 0.20, 100))
    assert verdict(0.15, shuffles) == "no_difference"


def test_run_hemisphere_reports_every_model():
    rng = np.random.default_rng(0)
    db = rng.random((80, 20), dtype=np.float32)
    queries = rng.random((10, 20), dtype=np.float32)
    out = run_hemisphere(
        make_small_circuit(), "L", db, queries, n_shuffles=3, ground_truth_k=5
    )
    assert set(out) >= {"side", "n_kc", "n_pn", "k", "fly", "uniform", "lsh",
                        "shuffled", "verdict"}
    assert len(out["shuffled"]) == 3
    assert 0.0 <= out["fly"] <= 1.0


def test_run_hemisphere_uses_the_same_compressor_for_every_model():
    """A different compressor per model would confound the comparison, so the
    seed must not vary with the model. Two runs must reproduce exactly."""
    rng = np.random.default_rng(0)
    db = rng.random((80, 20), dtype=np.float32)
    queries = rng.random((10, 20), dtype=np.float32)
    a = run_hemisphere(make_small_circuit(), "L", db, queries, n_shuffles=2,
                       ground_truth_k=5)
    b = run_hemisphere(make_small_circuit(), "L", db, queries, n_shuffles=2,
                       ground_truth_k=5)
    assert a["fly"] == b["fly"]
    assert a["shuffled"] == b["shuffled"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_phase1.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.phase1'`

- [ ] **Step 3: Write `src/flyhash/phase1.py`**

```python
"""Phase 1: the pre-registered MNIST comparison, one hemisphere at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flyhash.bench import (
    GROUND_TRUTH_K,
    mean_average_precision,
    rank_by_code_overlap,
    true_neighbours,
)
from flyhash.circuit import Circuit
from flyhash.datasets import load_mnist
from flyhash.encode import encode, make_compressor
from flyhash.models import (
    fly_projection,
    hash_length,
    lsh_projection,
    shuffled_projection,
    uniform_projection,
)

COMPRESSOR_SEED = 0
UNIFORM_SEED = 500
LSH_SEED = 600
SHUFFLE_SEED_BASE = 1000
SIGNIFICANCE_PERCENTILE = 97.5


def verdict(fly_map: float, shuffle_maps: list[float]) -> str:
    """Pre-registered two-sided test at alpha = 0.05."""
    upper = float(np.percentile(shuffle_maps, SIGNIFICANCE_PERCENTILE))
    lower = float(np.percentile(shuffle_maps, 100.0 - SIGNIFICANCE_PERCENTILE))
    if fly_map > upper:
        return "fly_better"
    if fly_map < lower:
        return "fly_worse"
    return "no_difference"


def run_hemisphere(
    circuit: Circuit,
    side: str,
    database: np.ndarray,
    queries: np.ndarray,
    n_shuffles: int = 100,
    ground_truth_k: int = GROUND_TRUTH_K,
) -> dict:
    fly = fly_projection(circuit)
    n_kc, n_pn = fly.shape
    k = hash_length(n_kc)

    # One compressor, shared by every model. This is what keeps the
    # comparison about the wiring rather than about the preprocessing.
    compressor = make_compressor(database.shape[1], n_pn, seed=COMPRESSOR_SEED)
    truth = true_neighbours(database, queries, k=ground_truth_k)

    def score(projection) -> float:
        db_codes = encode(database, compressor, projection, k)
        q_codes = encode(queries, compressor, projection, k)
        retrieved = rank_by_code_overlap(db_codes, q_codes, k=ground_truth_k)
        return mean_average_precision(retrieved, truth)

    mean_claws = int(round(fly.nnz / n_kc))
    shuffles = [
        score(shuffled_projection(fly, seed=SHUFFLE_SEED_BASE + i))
        for i in range(n_shuffles)
    ]
    fly_map = score(fly)
    return {
        "side": side,
        "n_kc": int(n_kc),
        "n_pn": int(n_pn),
        "k": int(k),
        "mean_claws": mean_claws,
        "fly": fly_map,
        "uniform": score(uniform_projection(n_kc, n_pn, mean_claws, UNIFORM_SEED)),
        "lsh": score(lsh_projection(n_kc, n_pn, LSH_SEED)),
        "shuffled": shuffles,
        "shuffled_mean": float(np.mean(shuffles)),
        "verdict": verdict(fly_map, shuffles),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the pre-registered Phase 1 comparison")
    p.add_argument("--circuit", default="data/circuit.npz")
    p.add_argument("--out", default="results/phase1-mnist.json")
    p.add_argument("--shuffles", type=int, default=100)
    args = p.parse_args(argv)

    circuit = Circuit.load(args.circuit)
    database, queries = load_mnist()
    results = [
        run_hemisphere(
            circuit.hemisphere(side), side, database, queries, args.shuffles
        )
        for side in ("L", "R")
    ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    for r in results:
        print(
            f"{r['side']}: n_kc={r['n_kc']} n_pn={r['n_pn']} k={r['k']}\n"
            f"   FLY      {r['fly']:.4f}\n"
            f"   SHUFFLED {r['shuffled_mean']:.4f} (mean of {len(r['shuffled'])})\n"
            f"   UNIFORM  {r['uniform']:.4f}\n"
            f"   LSH      {r['lsh']:.4f}\n"
            f"   verdict: {r['verdict']}"
        )
    gap = abs(results[0]["fly"] - results[1]["fly"])
    print(f"\nhemisphere noise floor (|FLY-L - FLY-R|): {gap:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_phase1.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the whole suite**

Run: `cd "G:/C#/flyhash" && python -m pytest -v`
Expected: 42 passed (5 circuit + 5 extract + 7 encode + 11 models + 3 datasets + 6 bench + 5 phase1)

- [ ] **Step 6: Run the real experiment**

Run: `cd "G:/C#/flyhash" && python -m flyhash.phase1`

Expected: `n_kc=1865 n_pn=149 k=93` for L and `n_kc=1875 n_pn=143 k=94` for R, four mAP numbers per hemisphere, and a verdict.

**Do not tune anything based on what comes out.** The parameters are pre-registered in the spec; the number is the answer whatever it is. If the verdict is `no_difference`, that is the finding.

- [ ] **Step 7: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/phase1.py tests/test_phase1.py results/phase1-mnist.json && git commit -m "feat: run pre-registered Phase 1 MNIST comparison"
```

Note: `results/` is already absent from `.gitignore`, so the JSON commits normally. The result *is* the deliverable — do not ignore it.

---

## What Phase 1 does not cover

Deliberately out of scope, deferred to Phase 2 per spec §7: GloVe and SIFT, the 58-glomerulus input level, graded APL inhibition, and the robustness sweeps over synapse threshold and hash length.
