# FlyHash Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether Phase 1's null result — measured PN→KC wiring gives no retrieval advantage over a degree-preserving shuffle — survives changes of dataset, input granularity, hash length, and the winner-take-all rule.

**Architecture:** Phase 1's pipeline is reused unchanged. Phase 2 adds three orthogonal dimensions to vary — a glomerulus-level input aggregation, two more benchmark datasets, and an APL-weighted winner-take-all — plus a sweep runner that crosses them and reports whether any combination flips the verdict.

**Tech Stack:** Python 3.11+, numpy, scipy.sparse, pytest. Same as Phase 1; no new dependencies.

## Global Constraints

Phase 2 does **not** alter the pre-registration in spec §6. The primary comparison stays FLY vs SHUFFLED by mAP@100; everything here is a robustness check around that comparison and cannot retroactively change Phase 1's result.

- Compressor seed **0**, one per (hemisphere, input level), **shared by every model**.
- Uniform seed **500**, Gaussian seed **600**, shuffle seeds **1000..1099**.
- Ground truth top-**100**; metric **mAP@100**; database **10000**, queries **1000**.
- Significance: above the **97.5th** percentile of the shuffles means better, below the **2.5th** means worse. `verdict()` is unchanged and must not be touched.
- Sweeps pre-registered as exploratory in spec §6: hash fraction **2%, 5%, 10%, 20%**; synapse weight threshold **1, 3, 5, 10**.
- **APL gain 0 must reproduce Phase 1 bit-for-bit.** This is the built-in correctness check for Task 2.

Verified circuit facts (measured 2026-09-12, assert against these):

| | KC | PN | Glomeruli | Edges |
|---|---|---|---|---|
| L | 1865 | 149 | 62 | 10395 |
| R | 1875 | 143 | 61 | 10341 |

Uniglomerular PNs carry 98.9% (L) and 99.2% (R) of PN→KC edges. APL reaches every KC; weights run 6..160, mean 48, σ 16.5.

**Deferred with reason:** the ORN→PN relay layer from spec §7 is not implemented. It is a fixed relay applied identically to all five models, so it cannot change the FLY-vs-SHUFFLED comparison; it would cost a `circuit.npz` schema change, a re-stream of the 1 GB dump, and 8% of the edges. Recorded as debt in FINDINGS.

---

### Task 0: Make ranking linear again without losing determinism

Phase 1 replaced `argpartition` with a full-row `np.lexsort` to make tie-breaking deterministic. That was correct but costs a full sort of a 1000x10000 array per scoring — about 3.3 s. Phase 2 needs thousands of scorings, which at that rate is roughly 18 hours. This task restores linear selection while producing **byte-identical** output.

The trick: overlap counts are integers, so the composite key `overlap * n_db - index` is injective — for any two entries with different overlap the gap is at least `n_db - (n_db - 1) = 1`, and for equal overlap the index alone decides. Ordering by that single key is exactly "overlap descending, index ascending", and because no two keys collide, `argpartition`'s instability cannot change which items are selected.

**Files:**
- Modify: `src/flyhash/bench.py`
- Modify: `tests/test_bench.py` (append)

**Interfaces:** `rank_by_code_overlap`'s signature and output are unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bench.py`:

```python
def test_ranking_matches_a_full_lexsort_reference():
    """The fast path must agree exactly with an explicit full sort."""
    rng = np.random.default_rng(0)
    db = sparse.csr_array((rng.random((200, 40)) < 0.2).astype(np.float32))
    q = sparse.csr_array((rng.random((20, 40)) < 0.2).astype(np.float32))
    overlap = np.asarray((q @ db.T).todense(), dtype=np.float64)
    index = np.broadcast_to(np.arange(db.shape[0]), overlap.shape)
    reference = np.lexsort((index, -overlap), axis=1)[:, :10].astype(np.int64)
    np.testing.assert_array_equal(rank_by_code_overlap(db, q, k=10), reference)


def test_ranking_is_stable_across_repeated_calls():
    rng = np.random.default_rng(1)
    db = sparse.csr_array((rng.random((200, 40)) < 0.2).astype(np.float32))
    q = sparse.csr_array((rng.random((20, 40)) < 0.2).astype(np.float32))
    first = rank_by_code_overlap(db, q, k=10)
    for _ in range(3):
        np.testing.assert_array_equal(rank_by_code_overlap(db, q, k=10), first)
```

- [ ] **Step 2: Run the test to verify it currently passes (it should — the reference is the current behaviour)**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_bench.py -v`
Expected: 9 passed. These two tests pin the current output so the rewrite cannot change it.

- [ ] **Step 3: Rewrite the body of `rank_by_code_overlap` in `src/flyhash/bench.py`**

```python
def rank_by_code_overlap(
    db_codes: sparse.csr_array, query_codes: sparse.csr_array, k: int
) -> np.ndarray:
    """Rank database items by how many active units they share with each query.

    Ties are broken by ascending database index, deterministically. Overlap
    counts are integers, so the composite key `overlap * n_db - index` is
    injective: entries with different overlap differ by at least 1, and equal
    overlap is settled by the index. Because no two keys collide, the
    unstable selection below cannot change which items are chosen, and the
    result is identical to a full lexsort at a fraction of the cost.
    """
    overlap = np.asarray((query_codes @ db_codes.T).todense(), dtype=np.float64)
    n_db = overlap.shape[1]
    index = np.arange(n_db, dtype=np.float64)
    key = overlap * n_db - index[None, :]
    idx = np.argpartition(-key, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(-key, idx, axis=1).argsort(axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)
```

- [ ] **Step 4: Run the tests to verify they still pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_bench.py -v`
Expected: 9 passed. The two new tests are the proof that the rewrite changed nothing.

- [ ] **Step 5: Prove the real Phase 1 result is unchanged and measure the speedup**

Run:
```bash
cd "G:/C#/flyhash" && python -c "
import json, time, numpy as np
from flyhash.circuit import Circuit
from flyhash.datasets import load_mnist
from flyhash.phase1 import run_hemisphere
old = {r['side']: r for r in json.load(open('results/phase1-mnist.json'))}
c = Circuit.load('data/circuit.npz'); db, q = load_mnist()
for s in ('L','R'):
    t = time.time()
    r = run_hemisphere(c.hemisphere(s), s, db, q, n_shuffles=5)
    print(s, 'fly=', round(r['fly'], 12), 'phase1 fly=', round(old[s]['fly'], 12),
          'match=', r['fly'] == old[s]['fly'], f'({(time.time()-t)/6:.2f}s per scoring)')
"
```
Expected: `match= True` for both hemispheres, and a per-scoring time well under the previous 3.3 s. If `match` is False, stop — the rewrite is not equivalent and must not be kept.

- [ ] **Step 6: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/bench.py tests/test_bench.py && git commit -m "perf: restore linear ranking via an injective tie-break key"
```

---

### Task 1: Glomerulus-level aggregation

The fly's true olfactory input dimensionality is the glomerulus, not the individual projection neuron: several PNs of one glomerulus carry the same channel. Phase 1 fed 149/143 PN channels; this task adds the 62/61 glomerulus channels. No new data is needed — the glomerulus is the prefix of the PN type name already stored in `Circuit.pn_types` (`DA1_lPN` → `DA1`).

**Files:**
- Create: `src/flyhash/glomeruli.py`
- Test: `tests/test_glomeruli.py`

**Interfaces:**
- Consumes: `Circuit` from Phase 1 (fields `pn_to_kc`, `pn_types`, `pn_ids`)
- Produces: `glomerulus_of(pn_type: str) -> str | None` (returns `None` for multiglomerular types, which start with `M_`, and for missing types), `aggregate_by_glomerulus(circuit: Circuit) -> tuple[scipy.sparse.csr_array, np.ndarray]` returning a `(n_kc, n_glomeruli)` matrix of synapse counts summed within each glomerulus, plus the sorted array of glomerulus names.

- [ ] **Step 1: Write the failing test**

Create `tests/test_glomeruli.py`:

```python
import numpy as np
import pytest
from scipy import sparse

from flyhash.circuit import Circuit
from flyhash.glomeruli import aggregate_by_glomerulus, glomerulus_of


def make_toy_circuit():
    """Four PNs: two of glomerulus DA1, one of VA1v, one multiglomerular."""
    m = sparse.csr_array(
        np.array([[5.0, 7.0, 0.0, 2.0], [0.0, 0.0, 9.0, 0.0]], dtype=np.float32)
    )
    return Circuit(
        pn_to_kc=m,
        kc_ids=np.array([10, 11], dtype=np.int64),
        kc_types=np.array(["KCg-m", "KCab-s"], dtype=object),
        kc_sides=np.array(["L", "L"], dtype=object),
        pn_ids=np.array([20, 21, 22, 23], dtype=np.int64),
        pn_types=np.array(
            ["DA1_lPN", "DA1_adPN", "VA1v_adPN", "M_lPNm11D"], dtype=object
        ),
        pn_sides=np.array(["L", "L", "L", "L"], dtype=object),
        apl_to_kc=np.array([40.0, 41.0], dtype=np.float32),
    )


def test_glomerulus_of_takes_the_prefix():
    assert glomerulus_of("DA1_lPN") == "DA1"
    assert glomerulus_of("VM5d_adPN") == "VM5d"


def test_glomerulus_of_rejects_multiglomerular_and_missing():
    assert glomerulus_of("M_lPNm11D") is None
    assert glomerulus_of("") is None
    assert glomerulus_of(None) is None


def test_aggregate_sums_pns_of_the_same_glomerulus():
    matrix, names = aggregate_by_glomerulus(make_toy_circuit())
    assert names.tolist() == ["DA1", "VA1v"]
    # KC0 gets 5+7 from DA1 and 0 from VA1v; KC1 gets 0 and 9.
    np.testing.assert_array_equal(
        matrix.toarray(), np.array([[12.0, 0.0], [0.0, 9.0]], dtype=np.float32)
    )


def test_aggregate_drops_multiglomerular_columns():
    matrix, names = aggregate_by_glomerulus(make_toy_circuit())
    assert "M_lPNm11D" not in names.tolist()
    assert matrix.shape == (2, 2)


def test_aggregate_returns_sorted_glomerulus_names():
    matrix, names = aggregate_by_glomerulus(make_toy_circuit())
    assert names.tolist() == sorted(names.tolist())


def test_aggregate_raises_when_nothing_survives():
    c = make_toy_circuit()
    stripped = Circuit(
        pn_to_kc=sparse.csr_array(np.array([[1.0], [0.0]], dtype=np.float32)),
        kc_ids=c.kc_ids,
        kc_types=c.kc_types,
        kc_sides=c.kc_sides,
        pn_ids=np.array([23], dtype=np.int64),
        pn_types=np.array(["M_lPNm11D"], dtype=object),
        pn_sides=np.array(["L"], dtype=object),
        apl_to_kc=c.apl_to_kc,
    )
    with pytest.raises(ValueError, match="no uniglomerular"):
        aggregate_by_glomerulus(stripped)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_glomeruli.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.glomeruli'`

- [ ] **Step 3: Write `src/flyhash/glomeruli.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_glomeruli.py -v`
Expected: 6 passed

- [ ] **Step 5: Check against the real circuit**

Run:
```bash
cd "G:/C#/flyhash" && python -c "
from flyhash.circuit import Circuit
from flyhash.glomeruli import aggregate_by_glomerulus
c = Circuit.load('data/circuit.npz')
for s in ('L','R'):
    h = c.hemisphere(s)
    m, names = aggregate_by_glomerulus(h)
    kept = m.sum() / h.pn_to_kc.sum()
    print(s, 'glomeruli=', len(names), 'shape=', m.shape, 'synapses kept=', round(100*kept,1), '%')
"
```
Expected: `L glomeruli= 62 shape= (1865, 62) synapses kept= 99.5 %` and `R glomeruli= 61 shape= (1875, 61) synapses kept= 99.6 %`. If the glomerulus counts differ, stop and report rather than adjusting the plan.

Do not confuse three distinct quantities here. **Synapses kept** (the weight sum, 99.5%/99.6%) is what this check measures. The **edge** fraction whose source PN is uniglomerular is 98.9%/99.2%. The **nonzero-cell** count after aggregation is lower still, 96.2%/95.9%, because aggregation merges two sister PNs of one glomerulus landing on the same Kenyon cell into a single entry — about 4% of KC-glomerulus pairs receive such duplicated input.

- [ ] **Step 6: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/glomeruli.py tests/test_glomeruli.py && git commit -m "feat: aggregate projection neurons into glomerulus channels"
```

---

### Task 2: APL-weighted winner-take-all

Phase 1 used an idealised "keep the top k" rule. The real fly has one giant inhibitory neuron, APL, that touches every Kenyon cell but with weights spanning 6 to 160. This task replaces the abstraction with the measured inhibition while keeping the hash length fixed, so the result stays comparable to every other model.

The rule is `x_i - gain * w_i * sum(x)`, then the same top-k. The `gain * sum(x)` factor is constant within a row, so what changes the ranking is the spread of `w_i`. Because the biological gain is unknown, it is swept rather than chosen; `gain=0` must reproduce Phase 1 exactly.

**Files:**
- Modify: `src/flyhash/encode.py` (append)
- Modify: `tests/test_encode.py` (append)

**Interfaces:**
- Consumes: `winner_take_all` and `encode` from Phase 1
- Produces: `apl_winner_take_all(A: np.ndarray, k: int, apl_weights: np.ndarray, gain: float) -> scipy.sparse.csr_array`, and `encode_with_apl(X, compressor, projection, k, apl_weights, gain) -> scipy.sparse.csr_array`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_encode.py`:

```python
from flyhash.encode import apl_winner_take_all, encode_with_apl


def test_apl_wta_with_zero_gain_matches_plain_wta():
    rng = np.random.default_rng(0)
    A = rng.random((5, 20), dtype=np.float32)
    w = rng.random(20, dtype=np.float32) * 100.0
    plain = winner_take_all(A, k=4).toarray()
    gated = apl_winner_take_all(A, k=4, apl_weights=w, gain=0.0).toarray()
    np.testing.assert_array_equal(gated, plain)


def test_apl_wta_keeps_exactly_k_per_row():
    rng = np.random.default_rng(1)
    A = rng.random((5, 20), dtype=np.float32)
    w = rng.random(20, dtype=np.float32) * 100.0
    out = apl_winner_take_all(A, k=4, apl_weights=w, gain=0.01)
    np.testing.assert_array_equal(out.sum(axis=1), np.full(5, 4.0))


def test_apl_wta_penalises_strongly_inhibited_units():
    # Two units with equal drive; the one with the larger APL weight loses.
    A = np.array([[1.0, 1.0, 0.5]], dtype=np.float32)
    w = np.array([1.0, 100.0, 1.0], dtype=np.float32)
    out = apl_winner_take_all(A, k=2, apl_weights=w, gain=0.001).toarray()
    assert out[0, 0] == 1.0
    assert out[0, 1] == 0.0
    assert out[0, 2] == 1.0


def test_apl_wta_rejects_mismatched_weight_length():
    A = np.zeros((2, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="apl_weights"):
        apl_winner_take_all(A, k=2, apl_weights=np.ones(3, dtype=np.float32), gain=0.1)


def test_encode_with_apl_at_zero_gain_matches_plain_encode():
    rng = np.random.default_rng(2)
    X = rng.random((6, 10), dtype=np.float32)
    comp = make_compressor(10, 5, seed=0)
    proj = sparse.csr_array(rng.integers(0, 2, (12, 5)).astype(np.float32))
    w = rng.random(12, dtype=np.float32) * 100.0
    plain = encode(X, comp, proj, k=3).toarray()
    gated = encode_with_apl(X, comp, proj, k=3, apl_weights=w, gain=0.0).toarray()
    np.testing.assert_array_equal(gated, plain)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_encode.py -k apl -v`
Expected: collection error, `ImportError: cannot import name 'apl_winner_take_all'`

- [ ] **Step 3: Append the implementation to `src/flyhash/encode.py`**

```python
def apl_winner_take_all(
    A: np.ndarray, k: int, apl_weights: np.ndarray, gain: float
) -> sparse.csr_array:
    """Winner-take-all with the fly's measured graded inhibition.

    APL is a single giant inhibitory neuron that contacts every Kenyon cell,
    but with synapse counts spanning roughly 6 to 160. It is driven by total
    Kenyon-cell activity, so each cell is suppressed in proportion to both
    its own APL weight and the row's total drive:

        x_i - gain * w_i * sum(x)

    `gain * sum(x)` is constant within a row, so what reorders the ranking is
    the spread of w_i. At gain=0 this reduces exactly to plain winner-take-all,
    which is the correctness check for the whole mechanism.
    """
    if apl_weights.shape[0] != A.shape[1]:
        raise ValueError(
            f"apl_weights has length {apl_weights.shape[0]}, expected {A.shape[1]}"
        )
    if gain == 0.0:
        return winner_take_all(A, k)
    totals = A.sum(axis=1, keepdims=True)
    inhibited = A - gain * totals * apl_weights[None, :]
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_encode.py -v`
Expected: 12 passed (7 existing + 5 new)

- [ ] **Step 5: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/encode.py tests/test_encode.py && git commit -m "feat: add APL-weighted winner-take-all"
```

---

### Task 3: GloVe and SIFT loaders

Phase 1 answered on MNIST alone. These are the other two datasets from the 2017 paper, so the answer can be stated across data domains rather than one.

**Files:**
- Modify: `src/flyhash/datasets.py` (append)
- Modify: `tests/test_datasets.py` (append)

**Interfaces:**
- Consumes: `read_idx_images`, `_fetch`, `DATABASE_SIZE`, `QUERY_COUNT` from Phase 1
- Produces: `load_glove(cache_dir=...) -> tuple[np.ndarray, np.ndarray]` and `load_sift(cache_dir=...) -> tuple[np.ndarray, np.ndarray]`, both returning `(database, queries)` float32 with 10000 and 1000 rows, and `DATASETS: dict[str, Callable]` mapping `"mnist"`, `"glove"`, `"sift"` to their loaders.

GloVe comes from `https://nlp.stanford.edu/data/glove.6B.zip` (822 MB) — too large. Use the 50-dimensional subset distributed as `glove.6B.50d.txt` inside it. To avoid the 822 MB download, fetch instead from `https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.50d.txt` (171 MB, plain text, one word per line: the token then 50 floats).

SIFT-10K is the `siftsmall` set from `ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz`; an HTTPS mirror is `https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/siftsmall.tar.gz`. It contains `siftsmall_base.fvecs` (10000 vectors, 128-d) and `siftsmall_query.fvecs` (100 queries). **Note the query count is 100, not 1000** — SIFT-small simply does not have 1000 queries. Use all 100 and record the deviation; do not pad or resample.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_datasets.py`:

```python
import struct as _struct

from flyhash.datasets import DATASETS, read_fvecs


def test_read_fvecs_parses_dimension_prefixed_records(tmp_path):
    path = tmp_path / "toy.fvecs"
    with open(path, "wb") as f:
        for row in ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]):
            f.write(_struct.pack("<i", 3))
            f.write(_struct.pack("<3f", *row))
    out = read_fvecs(path)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))


def test_read_fvecs_rejects_inconsistent_dimensions(tmp_path):
    path = tmp_path / "bad.fvecs"
    with open(path, "wb") as f:
        f.write(_struct.pack("<i", 3))
        f.write(_struct.pack("<3f", 1.0, 2.0, 3.0))
        f.write(_struct.pack("<i", 2))
        f.write(_struct.pack("<2f", 4.0, 5.0))
    with pytest.raises(ValueError, match="dimension"):
        read_fvecs(path)


def test_dataset_registry_lists_all_three():
    assert sorted(DATASETS) == ["glove", "mnist", "sift"]


@pytest.mark.network
def test_load_glove_shapes(tmp_path):
    db, q = DATASETS["glove"](cache_dir=tmp_path)
    assert db.shape == (10000, 50)
    assert q.shape == (1000, 50)


@pytest.mark.network
def test_load_sift_shapes(tmp_path):
    db, q = DATASETS["sift"](cache_dir=tmp_path)
    assert db.shape == (10000, 128)
    assert q.shape == (100, 128)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_datasets.py -v -m "not network"`
Expected: collection error, `ImportError: cannot import name 'DATASETS'`

- [ ] **Step 3: Append the implementation to `src/flyhash/datasets.py`**

```python
GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.50d.txt"
GLOVE_NAME = "glove.6B.50d.txt"
SIFT_URL = "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/siftsmall.tar.gz"
SIFT_NAME = "siftsmall.tar.gz"
SIFT_QUERY_COUNT = 100  # siftsmall ships 100 queries, not 1000


def read_fvecs(path: str | Path) -> np.ndarray:
    """Read a .fvecs file: each record is an int32 dimension then that many floats."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1) != 0:
        raise ValueError(f"{path} has an inconsistent record dimension")
    records = raw.reshape(-1, dim + 1)
    if not np.all(records[:, 0] == dim):
        raise ValueError(f"{path} has an inconsistent record dimension")
    return records[:, 1:].copy().view(np.float32)


def load_glove(cache_dir: str | Path = "data/datasets") -> tuple[np.ndarray, np.ndarray]:
    """First 11000 GloVe word vectors: 10000 database, 1000 queries."""
    path = _fetch_url(GLOVE_URL, GLOVE_NAME, Path(cache_dir))
    needed = DATABASE_SIZE + QUERY_COUNT
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            rows.append([float(x) for x in parts[1:]])
            if len(rows) == needed:
                break
    if len(rows) < needed:
        raise ValueError(f"{path} holds {len(rows)} vectors, need {needed}")
    arr = np.asarray(rows, dtype=np.float32)
    return arr[:DATABASE_SIZE], arr[DATABASE_SIZE:needed]


def load_sift(cache_dir: str | Path = "data/datasets") -> tuple[np.ndarray, np.ndarray]:
    """SIFT-small: 10000 base vectors and its 100 shipped queries."""
    cache = Path(cache_dir)
    archive = _fetch_url(SIFT_URL, SIFT_NAME, cache)
    base = cache / "siftsmall" / "siftsmall_base.fvecs"
    if not base.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(cache)
    db = read_fvecs(cache / "siftsmall" / "siftsmall_base.fvecs")[:DATABASE_SIZE]
    queries = read_fvecs(cache / "siftsmall" / "siftsmall_query.fvecs")[:SIFT_QUERY_COUNT]
    if db.shape[0] != DATABASE_SIZE:
        raise ValueError(f"SIFT base holds {db.shape[0]} vectors, need {DATABASE_SIZE}")
    return db, queries


DATASETS = {"mnist": load_mnist, "glove": load_glove, "sift": load_sift}
```

Also add `import tarfile` to the module's imports, and refactor the existing `_fetch(name, cache_dir)` into `_fetch_url(url, name, cache_dir)` keeping its atomic temp-file-and-rename behaviour exactly as it is; update `load_mnist`'s two call sites to pass `MNIST_BASE + name`.

- [ ] **Step 4: Run the offline tests, then the network ones**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_datasets.py -v -m "not network"`
Expected: 6 passed, 3 deselected

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_datasets.py -v`
Expected: 9 passed (downloads about 190 MB on first run)

- [ ] **Step 5: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/datasets.py tests/test_datasets.py && git commit -m "feat: add GloVe and SIFT loaders"
```

---

### Task 4: The sweep runner

Crosses the three new dimensions against the Phase 1 comparison and reports whether any combination flips the verdict.

**Files:**
- Create: `src/flyhash/phase2.py`
- Test: `tests/test_phase2.py`

**Interfaces:**
- Consumes: everything above, plus `run_hemisphere`'s supporting pieces from `phase1`
- Produces: `run_condition(circuit, side, database, queries, level, hash_fraction, apl_gain, n_shuffles) -> dict`, `main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase2.py`:

```python
import numpy as np
from scipy import sparse

from flyhash.circuit import Circuit
from flyhash.phase2 import run_condition


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
    types = np.array(
        [f"G{c}_adPN" if c % 4 else "M_lPNm11D" for c in range(n_pn)], dtype=object
    )
    return Circuit(
        pn_to_kc=m,
        kc_ids=np.arange(n_kc, dtype=np.int64),
        kc_types=np.array(["KCg-m"] * n_kc, dtype=object),
        kc_sides=np.array(["L"] * n_kc, dtype=object),
        pn_ids=np.arange(n_pn, dtype=np.int64),
        pn_types=types,
        pn_sides=np.array(["L"] * n_pn, dtype=object),
        apl_to_kc=rng.uniform(6.0, 160.0, n_kc).astype(np.float32),
    )


def _run(level="pn", fraction=0.05, gain=0.0):
    rng = np.random.default_rng(0)
    db = rng.random((80, 20), dtype=np.float32)
    q = rng.random((10, 20), dtype=np.float32)
    return run_condition(
        make_small_circuit(), "L", db, q,
        level=level, hash_fraction=fraction, apl_gain=gain,
        n_shuffles=3, ground_truth_k=5,
    )


def test_run_condition_reports_its_own_settings():
    out = _run()
    assert out["level"] == "pn"
    assert out["hash_fraction"] == 0.05
    assert out["apl_gain"] == 0.0
    assert len(out["shuffled"]) == 3


def test_glomerulus_level_has_fewer_channels_than_pn_level():
    assert _run(level="glom")["n_channels"] < _run(level="pn")["n_channels"]


def test_hash_fraction_changes_the_hash_length():
    assert _run(fraction=0.20)["k"] > _run(fraction=0.05)["k"]


def test_zero_apl_gain_reproduces_the_plain_run():
    assert _run(gain=0.0)["fly"] == _run(gain=0.0)["fly"]
    assert _run(gain=0.0)["fly"] != _run(gain=0.02)["fly"]


def test_run_condition_rejects_an_unknown_level():
    import pytest

    with pytest.raises(ValueError, match="level"):
        _run(level="nonsense")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_phase2.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.phase2'`

- [ ] **Step 3: Write `src/flyhash/phase2.py`**

```python
"""Phase 2: does the Phase 1 null survive changes of dataset, input
granularity, hash length, and winner-take-all rule?"""

from __future__ import annotations

import argparse
import itertools
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
from flyhash.datasets import DATASETS
from flyhash.encode import encode_with_apl, make_compressor
from flyhash.glomeruli import aggregate_by_glomerulus
from flyhash.models import (
    dense_gaussian_projection,
    hash_length,
    shuffled_projection,
    uniform_projection,
)
from flyhash.phase1 import (
    COMPRESSOR_SEED,
    GAUSSIAN_SEED,
    SHUFFLE_SEED_BASE,
    UNIFORM_SEED,
    verdict,
)

LEVELS = ("pn", "glom")
HASH_FRACTIONS = (0.02, 0.05, 0.10, 0.20)
APL_GAINS = (0.0, 0.005, 0.02, 0.05)

# Baseline reproduces Phase 1 exactly.
BASELINE = {"level": "pn", "hash_fraction": 0.05, "apl_gain": 0.0}


def sweep_conditions() -> list[dict]:
    """One factor varied at a time from the baseline, not a full cross product.

    The question is whether each factor on its own can overturn the Phase 1
    answer, so the interactions between them are not worth a 4x larger run.
    """
    seen = [dict(BASELINE)]
    for level in LEVELS:
        for fraction in HASH_FRACTIONS:
            for gain in APL_GAINS:
                varied = sum(
                    [level != BASELINE["level"],
                     fraction != BASELINE["hash_fraction"],
                     gain != BASELINE["apl_gain"]]
                )
                if varied == 1:
                    seen.append(
                        {"level": level, "hash_fraction": fraction, "apl_gain": gain}
                    )
    return seen


def _projection_for_level(circuit: Circuit, level: str):
    if level == "pn":
        return circuit.binary()
    if level == "glom":
        aggregated, _ = aggregate_by_glomerulus(circuit)
        out = aggregated.copy()
        out.data = np.ones_like(out.data, dtype=np.float32)
        return out
    raise ValueError(f"level must be one of {LEVELS}, got {level!r}")


def run_condition(
    circuit: Circuit,
    side: str,
    database: np.ndarray,
    queries: np.ndarray,
    level: str = "pn",
    hash_fraction: float = 0.05,
    apl_gain: float = 0.0,
    n_shuffles: int = 100,
    ground_truth_k: int = GROUND_TRUTH_K,
) -> dict:
    fly = _projection_for_level(circuit, level)
    n_kc, n_channels = fly.shape
    k = hash_length(n_kc, hash_fraction)
    apl = circuit.apl_to_kc

    compressor = make_compressor(database.shape[1], n_channels, seed=COMPRESSOR_SEED)
    truth = true_neighbours(database, queries, k=ground_truth_k)

    def score(projection) -> float:
        db_codes = encode_with_apl(
            database, compressor, projection, k, apl, apl_gain
        )
        q_codes = encode_with_apl(queries, compressor, projection, k, apl, apl_gain)
        retrieved = rank_by_code_overlap(db_codes, q_codes, k=ground_truth_k)
        return mean_average_precision(retrieved, truth)

    mean_claws = max(1, int(round(fly.nnz / n_kc)))
    shuffles = [
        score(shuffled_projection(fly, seed=SHUFFLE_SEED_BASE + i))
        for i in range(n_shuffles)
    ]
    fly_map = score(fly)
    below = int(sum(1 for s in shuffles if s < fly_map))
    above = int(sum(1 for s in shuffles if s > fly_map))
    return {
        "side": side,
        "level": level,
        "hash_fraction": hash_fraction,
        "apl_gain": apl_gain,
        "n_kc": int(n_kc),
        "n_channels": int(n_channels),
        "k": int(k),
        "fly": fly_map,
        "shuffled_mean": float(np.mean(shuffles)),
        "shuffled": shuffles,
        "uniform": score(
            uniform_projection(n_kc, n_channels, mean_claws, UNIFORM_SEED)
        ),
        "gaussian": score(dense_gaussian_projection(n_kc, n_channels, GAUSSIAN_SEED)),
        "verdict": verdict(fly_map, shuffles),
        "n_shuffles_below": below,
        "n_shuffles_above": above,
        "empirical_p_two_sided": min(
            1.0, 2.0 * min(below + 1, above + 1) / (len(shuffles) + 1)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the Phase 2 robustness sweeps")
    p.add_argument("--circuit", default="data/circuit.npz")
    p.add_argument("--out", default="results/phase2-sweeps.json")
    p.add_argument("--shuffles", type=int, default=100)
    p.add_argument("--datasets", nargs="+", default=["mnist", "glove", "sift"])
    args = p.parse_args(argv)

    circuit = Circuit.load(args.circuit)
    results = []
    for name in args.datasets:
        database, queries = DATASETS[name]()
        for side, condition in itertools.product(("L", "R"), sweep_conditions()):
            row = run_condition(
                circuit.hemisphere(side), side, database, queries,
                n_shuffles=args.shuffles, **condition,
            )
            fraction, gain, level = (
                condition["hash_fraction"], condition["apl_gain"], condition["level"]
            )
            row["dataset"] = name
            results.append(row)
            print(
                f"{name:6s} {side} {level:5s} frac={fraction:.2f} gain={gain:.3f} "
                f"fly={row['fly']:.4f} shuf={row['shuffled_mean']:.4f} "
                f"{row['verdict']} p={row['empirical_p_two_sided']:.3f}",
                flush=True,
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    flips = [r for r in results if r["verdict"] == "fly_better"]
    print(f"\nconditions run: {len(results)}")
    print(f"conditions where the fly beat the shuffle: {len(flips)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_phase2.py -v`
Expected: 5 passed

- [ ] **Step 5: Verify the gain=0, fraction=0.05, level=pn condition reproduces Phase 1**

This is the correctness check for the whole phase. Run:
```bash
cd "G:/C#/flyhash" && python -c "
import json, numpy as np
from flyhash.circuit import Circuit
from flyhash.datasets import load_mnist
from flyhash.phase2 import run_condition
c = Circuit.load('data/circuit.npz')
db, q = load_mnist()
old = {r['side']: r for r in json.load(open('results/phase1-mnist.json'))}
for s in ('L','R'):
    r = run_condition(c.hemisphere(s), s, db, q, level='pn', hash_fraction=0.05, apl_gain=0.0, n_shuffles=5)
    print(s, 'phase2 fly=', round(r['fly'],6), ' phase1 fly=', round(old[s]['fly'],6),
          ' match=', abs(r['fly']-old[s]['fly']) < 1e-12)
"
```
Expected: `match= True` for both hemispheres. If it is False, stop and report — the sweep runner is not reproducing the pipeline it is supposed to extend.

- [ ] **Step 6: Run the full sweep**

Run: `cd "G:/C#/flyhash" && python -m flyhash.phase2`

`sweep_conditions()` yields 8 conditions — the baseline plus one factor varied at a time (3 other hash fractions, 3 other APL gains, the glomerulus level). Crossed with 2 hemispheres and 3 datasets that is **48 conditions**, each with 100 shuffles: about 4850 scorings. After Task 0 this should run in well under an hour. Run it in the background and do not interrupt it.

Report the wall-clock time. If it is heading past two hours, stop and report rather than letting it run — that would mean Task 0's speedup did not take effect.

**Do not tune anything based on what comes out.** The pre-registration in spec §6 is unchanged; these sweeps can only show whether the Phase 1 answer is stable, not replace it.

- [ ] **Step 7: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/phase2.py tests/test_phase2.py results/phase2-sweeps.json && git commit -m "feat: run phase 2 robustness sweeps"
```

---

### Task 5: Record the Phase 2 findings

**Files:**
- Modify: `docs/superpowers/FINDINGS.md`

- [ ] **Step 1: Append a Phase 2 section**

Summarise: how many of the 192 conditions produced each verdict; whether any condition made the fly beat the shuffle and if so which; how the glomerulus level compares to the PN level; how mAP moves with APL gain; whether the answer differs by dataset. State plainly whether the Phase 1 null held.

- [ ] **Step 2: Commit**

```bash
cd "G:/C#/flyhash" && git add docs/superpowers/FINDINGS.md && git commit -m "docs: record phase 2 robustness results"
```
