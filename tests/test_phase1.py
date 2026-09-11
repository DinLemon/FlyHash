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
