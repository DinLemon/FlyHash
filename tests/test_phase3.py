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
