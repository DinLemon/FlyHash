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
    fly_projection,
    hash_length,
    lsh_projection,
    shuffled_projection,
    uniform_projection,
)

COMPRESSOR_SEED = 0
UNIFORM_SEED = 500
LSH_SEED = 600
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
    return {
        "side": side,
        "n_kc": int(n_kc),
        "n_pn": int(n_pn),
        "k": int(k),
        "mean_claws": mean_claws,
        "fly": fly_map,
        "uniform": score(uniform_projection(n_kc, n_pn, mean_claws, UNIFORM_SEED)),
        "lsh": score(lsh_projection(n_kc, n_pn, LSH_SEED)),
        "shuffled": shuffles,
        "shuffled_mean": float(np.mean(shuffles)),
        "verdict": verdict(fly_map, shuffles),
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
        print(
            f"{r['side']}: n_kc={r['n_kc']} n_pn={r['n_pn']} k={r['k']}\n"
            f"   FLY      {r['fly']:.4f}\n"
            f"   SHUFFLED {r['shuffled_mean']:.4f} (mean of {len(r['shuffled'])})\n"
            f"   UNIFORM  {r['uniform']:.4f}\n"
            f"   LSH      {r['lsh']:.4f}\n"
            f"   verdict: {r['verdict']}"
        )
    gap = abs(results[0]["fly"] - results[1]["fly"])
    print(f"\nhemisphere noise floor (|FLY-L - FLY-R|): {gap:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
