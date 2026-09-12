# FlyHash Phase 3 Implementation Plan — replication in a second animal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the Phase 1/2 null — measured PN→KC wiring gives no retrieval advantage over a degree-preserving shuffle — replicates in a second fly: a female, reconstructed by a different laboratory from different images.

**Architecture:** The whole Phase 1/2 pipeline is reused unchanged. Phase 3 adds one module that builds the existing `Circuit` container from FlyWire's three CSVs, so every downstream stage — models, encode, bench, the sweep runner — works on the female exactly as it does on the male, with no branching.

**Tech Stack:** Python 3.11+, numpy, scipy.sparse, pandas, pytest. No new dependencies.

## Why this is worth more than more sweeps

Phase 2 varied the conditions around one animal. This varies the animal. A null measured once can always be blamed on that specimen or that reconstruction; the same null in a second individual, of the other sex, segmented by a different group from a different electron-microscopy volume, cannot.

It also upgrades the study's own yardstick. Phase 1 used the gap between a fly's left and right mushroom body as a noise floor — two instances of one genetic program in one head. Phase 3 gives **four** such instances across two animals, so within-animal variation and between-animal variation can be compared against each other and both against the fly-versus-shuffle effect.

## Global Constraints

Phase 3 does **not** alter the pre-registration in spec §6. The primary comparison stays FLY vs SHUFFLED by mAP@100, and `verdict()` is used unchanged.

- Compressor seed **0**, one per condition, **shared by all five models**.
- Uniform seed **500**, Gaussian seed **600**, shuffle seeds **1000..1099**.
- Ground truth top-**100**; metric **mAP@100**; database **10000**.
- Condition held at the Phase 1 baseline throughout: PN input level, hash fraction **0.05**, APL gain **0**.

**The matched-threshold rule, and why it is not optional.** FlyWire's published connection table is effectively thresholded at 5 synapses: the raw per-neuropil counts run 82, 61, 48, 44 rows at weights 1–4 and then jump to 1097 at weight 5, and after merging neuropils the minimum PN→KC weight is exactly 5. The male dataset is a full edge list to which the study applied a threshold of 3. **Comparing the male at 3 against the female at 5 would confound sex with edge-inclusion criteria.** Every male-versus-female comparison in this phase therefore uses the male circuit extracted at `min_weight=5`.

Verified FlyWire circuit facts (measured 2026-09-12, threshold 5, in-degree ≥ 2 — assert against these):

| Hemisphere | KC | PN | Edges | Mean in-degree | Uniglomerular PN | Glomeruli | Ipsilateral PN |
|---|---|---|---|---|---|---|---|
| left | 2267 | 144 | 10912 | 4.81 | 129 | 63 | 138 of 144 |
| right | 2236 | 148 | 10377 | 4.64 | 132 | 63 | 141 of 148 |

For orientation, the male at threshold 3 was 1865/1875 Kenyon cells, 149/143 projection neurons, 62/61 glomeruli. The female has about 20% more Kenyon cells on a similar number of edges.

---

### Task 1: Build a `Circuit` from FlyWire

**Files:**
- Create: `src/flyhash/flywire.py`
- Test: `tests/test_flywire.py`

**Interfaces:**
- Consumes: `Circuit` from Phase 1
- Produces: `extract_flywire_circuit(classification_path, cell_types_path, connections_path, min_indegree=2) -> Circuit`, and `main(argv=None) -> int` for CLI use.

**Format differences from the male dataset, all of which this module absorbs:**

| | male-cns | FlyWire |
|---|---|---|
| files | 2 Feather | 3 gzipped CSV |
| id column | `bodyId` | `root_id` |
| weight column | `weight` | `syn_count` |
| side values | `L` / `R` | `left` / `right` |
| cell type | `type`, in the same table | `primary_type`, in a separate table joined on `root_id` |
| edges | one row per pair | one row per (pair, neuropil) — **must be summed per pair** |
| APL | `type == "APL"` | `primary_type == "APL"`, class `MBIN` |

`Circuit.hemisphere()` accepts only `"L"` and `"R"`, so sides must be mapped on the way in. Classes (`Kenyon_Cell`, `ALPN`) and type names (`DA1_lPN`, `M_lPNm11D`) use the same nomenclature in both datasets and need no translation — which is what makes `glomeruli.py` work unchanged on the female.

No synapse-weight threshold is applied here: the source already imposes one at 5.

- [ ] **Step 1: Write the failing test**

Create `tests/test_flywire.py`:

```python
import numpy as np
import pandas as pd
import pytest

from flyhash.flywire import extract_flywire_circuit


@pytest.fixture
def toy_files(tmp_path):
    """KC 100,101 left; KC 102 right; KC 103 under-connected and dropped.
    PN 200,201 reach KCs; PN 202 reaches nothing. 900 is APL."""
    classification = pd.DataFrame(
        {
            "root_id": [100, 101, 102, 103, 200, 201, 202, 900],
            "class": [
                "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell",
                "ALPN", "ALPN", "ALPN", "MBIN",
            ],
            "side": ["left", "left", "right", "left", "left", "right", "left", "left"],
        }
    )
    cell_types = pd.DataFrame(
        {
            "root_id": [100, 101, 102, 103, 200, 201, 202, 900],
            "primary_type": [
                "KCg-m", "KCab", "KCg-m", "KCab-p",
                "DA1_lPN", "VA1v_adPN", "DL3_lPN", "APL",
            ],
        }
    )
    # PN200 -> KC100 appears twice, in two neuropils: 4 + 3 = 7 after merging.
    connections = pd.DataFrame(
        {
            "pre_root_id": [200, 200, 201, 200, 201, 202, 900, 900, 100],
            "post_root_id": [100, 100, 100, 101, 102, 999, 100, 101, 101],
            "neuropil": ["MB_CA_L", "SCL_L", "MB_CA_L", "MB_CA_L", "MB_CA_R",
                         "SCL_L", "MB_CA_L", "MB_CA_L", "MB_CA_L"],
            "syn_count": [4, 3, 6, 8, 7, 9, 40, 41, 99],
        }
    )
    paths = []
    for frame, name in (
        (classification, "classification.csv.gz"),
        (cell_types, "consolidated_cell_types.csv.gz"),
        (connections, "connections.csv.gz"),
    ):
        path = tmp_path / name
        frame.to_csv(path, index=False)
        paths.append(path)
    return tuple(paths)


def test_neuropil_rows_are_summed_per_pair(toy_files):
    c = extract_flywire_circuit(*toy_files, min_indegree=1)
    kc = {int(b): i for i, b in enumerate(c.kc_ids)}
    pn = {int(b): i for i, b in enumerate(c.pn_ids)}
    dense = c.pn_to_kc.toarray()
    assert dense[kc[100], pn[200]] == 7.0  # 4 in MB_CA_L plus 3 in SCL_L
    assert dense[kc[100], pn[201]] == 6.0


def test_sides_are_mapped_to_single_letters(toy_files):
    c = extract_flywire_circuit(*toy_files, min_indegree=1)
    assert set(c.kc_sides) <= {"L", "R"}
    assert c.hemisphere("L").n_kc == 2


def test_indegree_filter_drops_under_connected_kcs(toy_files):
    c = extract_flywire_circuit(*toy_files, min_indegree=2)
    assert c.kc_ids.tolist() == [100]


def test_keeps_only_pns_that_reach_kcs(toy_files):
    c = extract_flywire_circuit(*toy_files, min_indegree=1)
    assert sorted(c.pn_ids.tolist()) == [200, 201]


def test_apl_is_separated_from_the_pn_matrix(toy_files):
    c = extract_flywire_circuit(*toy_files, min_indegree=1)
    kc = {int(b): i for i, b in enumerate(c.kc_ids)}
    assert 900 not in c.pn_ids.tolist()
    assert c.apl_to_kc[kc[100]] == 40.0
    assert c.apl_to_kc[kc[101]] == 41.0


def test_kc_to_kc_edges_are_ignored(toy_files):
    c = extract_flywire_circuit(*toy_files, min_indegree=1)
    assert 100 not in c.pn_ids.tolist()


def test_types_come_from_the_joined_table(toy_files):
    c = extract_flywire_circuit(*toy_files, min_indegree=1)
    by_id = dict(zip(c.pn_ids.tolist(), c.pn_types.tolist()))
    assert by_id[200] == "DA1_lPN"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_flywire.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.flywire'`

- [ ] **Step 3: Write `src/flyhash/flywire.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_flywire.py -v`
Expected: 7 passed

- [ ] **Step 5: Build the real female circuit and check it against the measured facts**

Run: `cd "G:/C#/flyhash" && python -m flyhash.flywire`

Expected, and these are verified ground truth — if they differ, stop and report rather than adjusting:
```
L: n_kc=2267 n_pn=144 edges=10912 mean_indeg=4.81
R: n_kc=2236 n_pn=148 edges=10377 mean_indeg=4.64
wrote data/circuit-flywire.npz
```

- [ ] **Step 6: Check that the glomerulus code works unchanged on the female**

This is the payoff of the shared nomenclature; if it fails, the two datasets are less compatible than believed. Run:
```bash
cd "G:/C#/flyhash" && python -c "
from flyhash.circuit import Circuit
from flyhash.glomeruli import aggregate_by_glomerulus
c = Circuit.load('data/circuit-flywire.npz')
for s in ('L','R'):
    h = c.hemisphere(s)
    m, names = aggregate_by_glomerulus(h)
    print(s, 'glomeruli=', len(names), 'shape=', m.shape)
"
```
Expected: 63 glomeruli on both sides.

- [ ] **Step 7: Commit**

```bash
cd "G:/C#/flyhash" && git add src/flyhash/flywire.py tests/test_flywire.py data/circuit-flywire.npz && git commit -m "feat: build the study circuit from the FlyWire female connectome"
```

`data/circuit-flywire.npz` is a few megabytes and is committed deliberately, as the male circuit is, so the replication reproduces without the 50 MB download.

---

### Task 2: The replication run

Runs the identical baseline condition on both animals, at a matched threshold, across all three datasets.

**Files:**
- Create: `src/flyhash/phase3.py`
- Test: `tests/test_phase3.py`

**Interfaces:**
- Consumes: `run_condition` from `phase2`, `Circuit`, `DATASETS`
- Produces: `ANIMALS: dict[str, str]` mapping an animal label to its circuit path, `run_replication(circuits, database, queries, dataset_name, n_shuffles=100) -> list[dict]`, `main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase3.py`:

```python
import numpy as np
from scipy import sparse

from flyhash.circuit import Circuit
from flyhash.phase3 import run_replication


def make_circuit(n_kc, seed):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(n_kc):
        for c in rng.choice(12, size=rng.integers(2, 6), replace=False):
            rows.append(i)
            cols.append(c)
    m = sparse.csr_array(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_kc, 12)
    )
    return Circuit(
        pn_to_kc=m,
        kc_ids=np.arange(n_kc, dtype=np.int64),
        kc_types=np.array(["KCg-m"] * n_kc, dtype=object),
        kc_sides=np.array(["L"] * n_kc, dtype=object),
        pn_ids=np.arange(12, dtype=np.int64),
        pn_types=np.array([f"G{c}_adPN" for c in range(12)], dtype=object),
        pn_sides=np.array(["L"] * 12, dtype=object),
        apl_to_kc=rng.uniform(6.0, 160.0, n_kc).astype(np.float32),
    )


def _run():
    rng = np.random.default_rng(0)
    db = rng.random((80, 20), dtype=np.float32)
    q = rng.random((10, 20), dtype=np.float32)
    circuits = {"male": make_circuit(60, 0), "female": make_circuit(70, 1)}
    return run_replication(circuits, db, q, "toy", n_shuffles=3, ground_truth_k=5)


def test_every_animal_and_hemisphere_is_reported():
    rows = _run()
    assert {(r["animal"], r["side"]) for r in rows} == {
        ("male", "L"), ("female", "L")
    }


def test_rows_carry_the_dataset_name():
    assert all(r["dataset"] == "toy" for r in _run())


def test_rows_carry_a_verdict_and_empirical_rank():
    for r in _run():
        assert r["verdict"] in {"fly_better", "fly_worse", "no_difference"}
        assert 0.0 <= r["empirical_p_two_sided"] <= 1.0


def test_replication_is_deterministic():
    assert [r["fly"] for r in _run()] == [r["fly"] for r in _run()]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_phase3.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'flyhash.phase3'`

- [ ] **Step 3: Write `src/flyhash/phase3.py`**

```python
"""Phase 3: does the null replicate in a second animal?

The male and female circuits are run through the identical baseline
condition. The male circuit must be the one extracted at min_weight=5, not
the study's default of 3, because FlyWire's published table is already
thresholded at 5 — comparing 3 against 5 would confound sex with the
edge-inclusion criterion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flyhash.bench import GROUND_TRUTH_K
from flyhash.circuit import Circuit
from flyhash.datasets import DATASETS
from flyhash.phase2 import run_condition

MATCHED_THRESHOLD = 5
ANIMALS = {
    "male": f"data/thresholds/circuit-minweight-{MATCHED_THRESHOLD}.npz",
    "female": "data/circuit-flywire.npz",
}


def run_replication(
    circuits: dict[str, Circuit],
    database: np.ndarray,
    queries: np.ndarray,
    dataset_name: str,
    n_shuffles: int = 100,
    ground_truth_k: int = GROUND_TRUTH_K,
) -> list[dict]:
    rows = []
    for animal, circuit in circuits.items():
        for side in sorted(set(circuit.kc_sides)):
            row = run_condition(
                circuit.hemisphere(side), side, database, queries,
                level="pn", hash_fraction=0.05, apl_gain=0.0,
                n_shuffles=n_shuffles, ground_truth_k=ground_truth_k,
            )
            row["animal"] = animal
            row["dataset"] = dataset_name
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the cross-animal replication")
    p.add_argument("--out", default="results/phase3-replication.json")
    p.add_argument("--shuffles", type=int, default=100)
    p.add_argument("--datasets", nargs="+", default=["mnist", "glove", "sift"])
    args = p.parse_args(argv)

    circuits = {name: Circuit.load(path) for name, path in ANIMALS.items()}
    results = []
    for name in args.datasets:
        database, queries = DATASETS[name]()
        for row in run_replication(circuits, database, queries, name, args.shuffles):
            results.append(row)
            print(
                f"{name:6s} {row['animal']:6s} {row['side']} "
                f"n_kc={row['n_kc']:<5d} n_pn={row['n_channels']:<4d} "
                f"fly={row['fly']:.4f} shuf={row['shuffled_mean']:.4f} "
                f"d={row['fly'] - row['shuffled_mean']:+.4f} "
                f"{row['verdict']} p={row['empirical_p_two_sided']:.3f}",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(out)

    print(f"\nconditions: {len(results)}")
    for animal in ANIMALS:
        rows = [r for r in results if r["animal"] == animal]
        deltas = [r["fly"] - r["shuffled_mean"] for r in rows]
        print(
            f"  {animal:6s}: mean fly-minus-shuffle {np.mean(deltas):+.4f}, "
            f"negative in {sum(d < 0 for d in deltas)}/{len(deltas)}, "
            f"fly_better in {sum(r['verdict'] == 'fly_better' for r in rows)}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note the results file is written to a temp path and renamed, so a failure mid-write cannot truncate a completed run's results. An earlier careless in-place write destroyed a 90-minute sweep; do not reintroduce that pattern.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "G:/C#/flyhash" && python -m pytest tests/test_phase3.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full suite**

Run: `cd "G:/C#/flyhash" && python -m pytest -q`
Expected: all tests pass; report the count.

- [ ] **Step 6: STOP — do not run the replication**

The replication takes roughly half an hour. Report that Step 5 passed and stop; the controller will run it.

---

### Task 3: Record the findings

**Files:**
- Modify: `docs/superpowers/FINDINGS.md`

- [ ] **Step 1: Append a Phase 3 section**

State: whether the null replicated in the female; the per-animal mean fly-minus-shuffle and verdict counts; the four-hemisphere comparison (within-animal versus between-animal variation, both against the fly-versus-shuffle effect); and the matched-threshold rule with the evidence for FlyWire's effective threshold of 5. Record any deviation from pre-registration that the replication introduces.

- [ ] **Step 2: Commit**

```bash
cd "G:/C#/flyhash" && git add docs/superpowers/FINDINGS.md results/phase3-replication.json && git commit -m "docs: record the cross-animal replication result"
```
