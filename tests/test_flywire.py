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


def test_rejects_a_neuron_with_an_unmapped_side(toy_files):
    """A KC with unmapped side (center) that survives the in-degree filter should raise ValueError."""
    # Build variant where KC 100 has unmapped side but meets in-degree threshold
    tmp_path = toy_files[0].parent
    classification = pd.DataFrame(
        {
            "root_id": [100, 101, 102, 103, 200, 201, 202, 900],
            "class": [
                "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell",
                "ALPN", "ALPN", "ALPN", "MBIN",
            ],
            "side": ["center", "left", "right", "left", "left", "right", "left", "left"],
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
    connections = pd.DataFrame(
        {
            "pre_root_id": [200, 201, 200],
            "post_root_id": [100, 100, 101],
            "syn_count": [6, 6, 8],
        }
    )
    paths = []
    for frame, name in (
        (classification, "classification_center.csv.gz"),
        (cell_types, "consolidated_cell_types_center.csv.gz"),
        (connections, "connections_center.csv.gz"),
    ):
        path = tmp_path / name
        frame.to_csv(path, index=False)
        paths.append(path)

    with pytest.raises(ValueError, match="contains 1 Kenyon cells and 0 projection neurons with unmapped side values"):
        extract_flywire_circuit(*paths, min_indegree=2)


def test_ignores_unmapped_side_neurons_below_indegree_threshold(toy_files):
    """A KC with unmapped side that does NOT survive the in-degree filter should be ignored silently."""
    # Build variant where KC 103 has unmapped side but is dropped by indegree filter
    tmp_path = toy_files[0].parent
    classification = pd.DataFrame(
        {
            "root_id": [100, 101, 102, 103, 200, 201, 202, 900],
            "class": [
                "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell",
                "ALPN", "ALPN", "ALPN", "MBIN",
            ],
            "side": ["left", "left", "right", "center", "left", "right", "left", "left"],
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
    # KC103 gets only 1 edge, won't survive min_indegree=2
    connections = pd.DataFrame(
        {
            "pre_root_id": [200, 200, 201, 200, 201, 202, 900, 900, 200],
            "post_root_id": [100, 100, 100, 101, 102, 999, 100, 101, 103],
            "neuropil": ["MB_CA_L", "SCL_L", "MB_CA_L", "MB_CA_L", "MB_CA_R",
                         "SCL_L", "MB_CA_L", "MB_CA_L", "MB_CA_L"],
            "syn_count": [4, 3, 6, 8, 7, 9, 40, 41, 5],
        }
    )
    paths = []
    for frame, name in (
        (classification, "classification_below.csv.gz"),
        (cell_types, "consolidated_cell_types_below.csv.gz"),
        (connections, "connections_below.csv.gz"),
    ):
        path = tmp_path / name
        frame.to_csv(path, index=False)
        paths.append(path)

    # This should succeed: KC103 with center side is excluded by indegree, not by the check
    c = extract_flywire_circuit(*paths, min_indegree=2)
    # Verify the circuit was built with only the valid neurons
    assert 103 not in c.kc_ids.tolist()
