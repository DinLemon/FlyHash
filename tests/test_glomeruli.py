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
