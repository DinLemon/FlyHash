import numpy as np
from scipy import sparse

from flyhash.circuit import Circuit
from flyhash.phase2 import _projection_for_level, run_condition


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


def _make_circuit_with_shared_glomerulus():
    """KC0 is contacted by two sister PNs (PN0, PN1) of the same glomerulus
    (DA1), so the aggregated glom-level entry for KC0 is 2. KC1 is contacted
    by a single PN (PN2) of a different glomerulus (VA1)."""
    n_kc, n_pn = 2, 3
    rows = [0, 0, 1]
    cols = [0, 1, 2]
    m = sparse.csr_array(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_kc, n_pn)
    )
    types = np.array(["DA1_adPN", "DA1_lPN", "VA1_adPN"], dtype=object)
    return Circuit(
        pn_to_kc=m,
        kc_ids=np.arange(n_kc, dtype=np.int64),
        kc_types=np.array(["KCg-m"] * n_kc, dtype=object),
        kc_sides=np.array(["L"] * n_kc, dtype=object),
        pn_ids=np.arange(n_pn, dtype=np.int64),
        pn_types=types,
        pn_sides=np.array(["L"] * n_pn, dtype=object),
        apl_to_kc=np.full(n_kc, 50.0, dtype=np.float32),
    )


def test_glomerulus_projection_is_binarised():
    circuit = _make_circuit_with_shared_glomerulus()
    glom = _projection_for_level(circuit, "glom")
    pn = _projection_for_level(circuit, "pn")

    assert glom.nnz > 0
    assert np.all(glom.data == 1.0)
    assert glom.nnz < pn.nnz
