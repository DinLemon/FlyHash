"""Phase 1: the pre-registered MNIST comparison, one hemisphere at a time."""

from __future__ import annotations

import argparse
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
from flyhash.datasets import load_mnist
from flyhash.encode import encode, make_compressor
from flyhash.models import (
    dense_gaussian_projection,
    fly_projection,
    hash_length,
    shuffled_projection,
    uniform_projection,
)

COMPRESSOR_SEED = 0
UNIFORM_SEED = 500
GAUSSIAN_SEED = 600
SHUFFLE_SEED_BASE = 1000
SIGNIFICANCE_PERCENTILE = 97.5


def verdict(fly_map: float, shuffle_maps: list[float]) -> str:
    """Pre-registered two-sided test at alpha = 0.05."""
    upper = float(np.percentile(shuffle_maps, SIGNIFICANCE_PERCENTILE))
    lower = float(np.percentile(shuffle_maps, 100.0 - SIGNIFICANCE_PERCENTILE))
    if fly_map > upper:
        return "fly_better"
    if fly_map < lower:
        return "fly_worse"
    return "no_difference"


def verdict_margin_in_sd(
    verdict_label: str, fly_map: float, shuffle_maps: list[float]
) -> float:
    """How far fly_map sits from the cutoff that decided verdict_label,
    in units of the shuffle distribution's standard deviation.

    This is a derived reporting quantity only -- it never feeds back into
    verdict(), which remains the pre-registered decision exactly as written.

    Sign convention: positive means the verdict is comfortable (fly_map is
    further into the verdict's region, away from the cutoff that could flip
    it); negative or near zero means the call is marginal. For
    "fly_worse"/"fly_better" this is the signed distance from fly_map to
    the cutoff that produced that verdict, oriented so that being deeper in
    the verdict's region is positive. For "no_difference" it is the
    distance to whichever of the two cutoffs is nearer -- a small value
    there means fly_map is close to flipping the verdict.
    """
    std = float(np.std(shuffle_maps))
    upper = float(np.percentile(shuffle_maps, SIGNIFICANCE_PERCENTILE))
    lower = float(np.percentile(shuffle_maps, 100.0 - SIGNIFICANCE_PERCENTILE))

    if std == 0.0:
        # No spread to measure a margin against -- avoid dividing by zero.
        return 0.0

    if verdict_label == "fly_worse":
        raw = lower - fly_map
    elif verdict_label == "fly_better":
        raw = fly_map - upper
    else:
        raw = min(fly_map - lower, upper - fly_map)

    return raw / std


BORDERLINE_THRESHOLD_SD = 0.5


def verdict_with_margin(
    fly_map: float, shuffle_maps: list[float]
) -> tuple[str, float, bool]:
    """The pre-registered verdict plus its reporting-only margin and
    borderline flag, computed the same way for every caller."""
    verdict_label = verdict(fly_map, shuffle_maps)
    margin_in_sd = verdict_margin_in_sd(verdict_label, fly_map, shuffle_maps)
    borderline = abs(margin_in_sd) < BORDERLINE_THRESHOLD_SD
    return verdict_label, margin_in_sd, borderline


def run_hemisphere(
    circuit: Circuit,
    side: str,
    database: np.ndarray,
    queries: np.ndarray,
    n_shuffles: int = 100,
    ground_truth_k: int = GROUND_TRUTH_K,
) -> dict:
    fly = fly_projection(circuit)
    n_kc, n_pn = fly.shape
    k = hash_length(n_kc)

    # One compressor, shared by every model. This is what keeps the
    # comparison about the wiring rather than about the preprocessing.
    compressor = make_compressor(database.shape[1], n_pn, seed=COMPRESSOR_SEED)
    truth = true_neighbours(database, queries, k=ground_truth_k)

    def score(projection) -> float:
        db_codes = encode(database, compressor, projection, k)
        q_codes = encode(queries, compressor, projection, k)
        retrieved = rank_by_code_overlap(db_codes, q_codes, k=ground_truth_k)
        return mean_average_precision(retrieved, truth)

    mean_claws = int(round(fly.nnz / n_kc))
    shuffles = [
        score(shuffled_projection(fly, seed=SHUFFLE_SEED_BASE + i))
        for i in range(n_shuffles)
    ]
    fly_map = score(fly)

    # Empirical rank disclosure. This does NOT feed back into verdict(),
    # which stays exactly as pre-registered -- it exists only so a
    # borderline interpolated verdict can be read alongside how many of
    # the actual shuffle draws it beat.
    n_shuffles_below = int(sum(1 for s in shuffles if s < fly_map))
    n_shuffles_above = int(sum(1 for s in shuffles if s > fly_map))
    empirical_p_two_sided = min(
        2
        * min(n_shuffles_below + 1, n_shuffles_above + 1)
        / (len(shuffles) + 1),
        1.0,
    )

    verdict_label, margin_in_sd, borderline = verdict_with_margin(fly_map, shuffles)

    return {
        "side": side,
        "n_kc": int(n_kc),
        "n_pn": int(n_pn),
        "k": int(k),
        "mean_claws": mean_claws,
        "fly": fly_map,
        "uniform": score(uniform_projection(n_kc, n_pn, mean_claws, UNIFORM_SEED)),
        "gaussian": score(dense_gaussian_projection(n_kc, n_pn, GAUSSIAN_SEED)),
        "shuffled": shuffles,
        "shuffled_mean": float(np.mean(shuffles)),
        "verdict": verdict_label,
        "n_shuffles_below": n_shuffles_below,
        "n_shuffles_above": n_shuffles_above,
        "empirical_p_two_sided": float(empirical_p_two_sided),
        "verdict_margin_in_sd": margin_in_sd,
        "verdict_borderline": borderline,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the pre-registered Phase 1 comparison")
    p.add_argument("--circuit", default="data/circuit.npz")
    p.add_argument("--out", default="results/phase1-mnist.json")
    p.add_argument("--shuffles", type=int, default=100)
    args = p.parse_args(argv)

    circuit = Circuit.load(args.circuit)
    database, queries = load_mnist()
    results = [
        run_hemisphere(
            circuit.hemisphere(side), side, database, queries, args.shuffles
        )
        for side in ("L", "R")
    ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    for r in results:
        if r["verdict_borderline"]:
            marker = f" [BORDERLINE, {r['verdict_margin_in_sd']:.2f} SD]"
        else:
            marker = ""
        print(
            f"{r['side']}: n_kc={r['n_kc']} n_pn={r['n_pn']} k={r['k']}\n"
            f"   FLY      {r['fly']:.4f}\n"
            f"   SHUFFLED {r['shuffled_mean']:.4f} (mean of {len(r['shuffled'])})\n"
            f"   UNIFORM  {r['uniform']:.4f}\n"
            f"   GAUSSIAN {r['gaussian']:.4f}\n"
            f"   verdict: {r['verdict']}{marker} - empirical "
            f"{r['n_shuffles_below']}/{len(r['shuffled'])} shuffles "
            f"below, two-sided p={r['empirical_p_two_sided']:.4f}"
        )
    gap = abs(results[0]["fly"] - results[1]["fly"])
    print(f"\nhemisphere noise floor (|FLY-L - FLY-R|): {gap:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
