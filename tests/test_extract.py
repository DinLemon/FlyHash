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
