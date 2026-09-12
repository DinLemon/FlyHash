from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

from flyhash.circuit import Circuit
from flyhash.encode import make_compressor
from flyhash.phase1 import run_hemisphere, verdict, verdict_margin_in_sd


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
    assert set(out) >= {"side", "n_kc", "n_pn", "k", "fly", "uniform", "gaussian",
                        "shuffled", "verdict"}
    assert len(out["shuffled"]) == 3
    assert 0.0 <= out["fly"] <= 1.0


def test_run_hemisphere_uses_the_same_compressor_for_every_model():
    """A different compressor per model would confound the comparison, so
    make_compressor must be called exactly once per run_hemisphere call and
    that one compressor must be reused for fly, uniform, gaussian and every
    shuffle -- not rebuilt (even with a fixed seed) per model."""
    rng = np.random.default_rng(0)
    db = rng.random((80, 20), dtype=np.float32)
    queries = rng.random((10, 20), dtype=np.float32)

    with patch(
        "flyhash.phase1.make_compressor", wraps=make_compressor
    ) as mock_compressor:
        out = run_hemisphere(
            make_small_circuit(), "L", db, queries, n_shuffles=2, ground_truth_k=5
        )

    assert mock_compressor.call_count == 1
    assert set(out) >= {"fly", "uniform", "gaussian", "shuffled"}
    assert len(out["shuffled"]) == 2


def test_run_hemisphere_is_deterministic():
    """Two runs with the same inputs must reproduce exactly."""
    rng = np.random.default_rng(0)
    db = rng.random((80, 20), dtype=np.float32)
    queries = rng.random((10, 20), dtype=np.float32)
    a = run_hemisphere(make_small_circuit(), "L", db, queries, n_shuffles=2,
                       ground_truth_k=5)
    b = run_hemisphere(make_small_circuit(), "L", db, queries, n_shuffles=2,
                       ground_truth_k=5)
    assert a["fly"] == b["fly"]
    assert a["shuffled"] == b["shuffled"]


def test_verdict_at_the_decision_boundary():
    # The fly value sits a hair below the interpolated 2.5th percentile of
    # the shuffle list, so verdict() must report "fly_worse" -- even though
    # only 3 of the 100 shuffles actually lie below it (empirical two-sided
    # p ~= 0.06). This is the pre-registered rule's real, knife-edge
    # behaviour: a verdict can flip on a margin far smaller than the shuffle
    # spread, which is exactly why the empirical rank is reported alongside
    # it (see run_hemisphere's n_shuffles_below/above and
    # empirical_p_two_sided) rather than folded into verdict() itself.
    shuffles = list(np.linspace(0.10, 0.20, 100))
    lower = float(np.percentile(shuffles, 2.5))
    fly_map = lower - 1e-6
    assert verdict(fly_map, shuffles) == "fly_worse"


def test_verdict_margin_flags_a_borderline_call():
    # A fly value sitting a hair below the 2.5th-percentile cutoff produces
    # a "fly_worse" verdict that is only barely decided: the margin, in
    # units of the shuffle distribution's standard deviation, should be
    # tiny and the call should be flagged as borderline.
    shuffles = list(np.linspace(0.10, 0.20, 100))
    lower = float(np.percentile(shuffles, 2.5))
    borderline_fly_map = lower - 1e-6

    label = verdict(borderline_fly_map, shuffles)
    assert label == "fly_worse"
    margin = verdict_margin_in_sd(label, borderline_fly_map, shuffles)
    assert abs(margin) < 0.5

    # A comfortably-decided case: fly_map sits far below every shuffle, so
    # the margin should be large and not borderline.
    comfortable_fly_map = 0.0
    comfortable_label = verdict(comfortable_fly_map, shuffles)
    assert comfortable_label == "fly_worse"
    comfortable_margin = verdict_margin_in_sd(
        comfortable_label, comfortable_fly_map, shuffles
    )
    assert abs(comfortable_margin) >= 0.5

    # And run_hemisphere must surface both derived fields on its result.
    rng = np.random.default_rng(0)
    db = rng.random((80, 20), dtype=np.float32)
    queries = rng.random((10, 20), dtype=np.float32)
    out = run_hemisphere(
        make_small_circuit(), "L", db, queries, n_shuffles=5, ground_truth_k=5
    )
    assert "verdict_margin_in_sd" in out
    assert "verdict_borderline" in out
    assert out["verdict_borderline"] == (abs(out["verdict_margin_in_sd"]) < 0.5)
