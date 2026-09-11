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
