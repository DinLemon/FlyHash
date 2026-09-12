"""Phase 2: does the Phase 1 null survive changes of dataset, input
granularity, hash length, and winner-take-all rule?"""

from __future__ import annotations

import argparse
import itertools
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
from flyhash.datasets import DATASETS
from flyhash.encode import encode_with_apl, make_compressor
from flyhash.glomeruli import aggregate_by_glomerulus
from flyhash.models import (
    dense_gaussian_projection,
    hash_length,
    shuffled_projection,
    uniform_projection,
)
from flyhash.phase1 import (
    COMPRESSOR_SEED,
    GAUSSIAN_SEED,
    SHUFFLE_SEED_BASE,
    UNIFORM_SEED,
    verdict_with_margin,
)

LEVELS = ("pn", "glom")
HASH_FRACTIONS = (0.02, 0.05, 0.10, 0.20)
APL_GAINS = (0.0, 1e-5, 3e-5, 1e-4)  # calibrated on the real circuit, see below

# Baseline reproduces Phase 1 exactly.
BASELINE = {"level": "pn", "hash_fraction": 0.05, "apl_gain": 0.0}

# Phase 1's pre-registered verdict, per hemisphere. Phase 2 asks whether
# varying dataset, granularity, hash length, or WTA rule moves either
# hemisphere away from this.
PHASE1_VERDICT = {"L": "no_difference", "R": "fly_worse"}


def sweep_conditions() -> list[dict]:
    """One factor varied at a time from the baseline, not a full cross product.

    The question is whether each factor on its own can overturn the Phase 1
    answer, so the interactions between them are not worth a 4x larger run.
    """
    seen = [dict(BASELINE)]
    for level in LEVELS:
        for fraction in HASH_FRACTIONS:
            for gain in APL_GAINS:
                varied = sum(
                    [level != BASELINE["level"],
                     fraction != BASELINE["hash_fraction"],
                     gain != BASELINE["apl_gain"]]
                )
                if varied == 1:
                    seen.append(
                        {"level": level, "hash_fraction": fraction, "apl_gain": gain}
                    )
    return seen


def _projection_for_level(circuit: Circuit, level: str):
    if level == "pn":
        return circuit.binary()
    if level == "glom":
        aggregated, _ = aggregate_by_glomerulus(circuit)
        out = aggregated.copy()
        out.data = np.ones_like(out.data, dtype=np.float32)
        return out
    raise ValueError(f"level must be one of {LEVELS}, got {level!r}")


def run_condition(
    circuit: Circuit,
    side: str,
    database: np.ndarray,
    queries: np.ndarray,
    level: str = "pn",
    hash_fraction: float = 0.05,
    apl_gain: float = 0.0,
    n_shuffles: int = 100,
    ground_truth_k: int = GROUND_TRUTH_K,
) -> dict:
    fly = _projection_for_level(circuit, level)
    n_kc, n_channels = fly.shape
    k = hash_length(n_kc, hash_fraction)
    apl = circuit.apl_to_kc

    compressor = make_compressor(database.shape[1], n_channels, seed=COMPRESSOR_SEED)
    truth = true_neighbours(database, queries, k=ground_truth_k)

    def score(projection) -> float:
        db_codes = encode_with_apl(
            database, compressor, projection, k, apl, apl_gain
        )
        q_codes = encode_with_apl(queries, compressor, projection, k, apl, apl_gain)
        retrieved = rank_by_code_overlap(db_codes, q_codes, k=ground_truth_k)
        return mean_average_precision(retrieved, truth)

    mean_claws = max(1, int(round(fly.nnz / n_kc)))
    shuffles = [
        score(shuffled_projection(fly, seed=SHUFFLE_SEED_BASE + i))
        for i in range(n_shuffles)
    ]
    fly_map = score(fly)
    below = int(sum(1 for s in shuffles if s < fly_map))
    above = int(sum(1 for s in shuffles if s > fly_map))
    verdict_label, margin_in_sd, borderline = verdict_with_margin(fly_map, shuffles)
    return {
        "side": side,
        "level": level,
        "hash_fraction": hash_fraction,
        "apl_gain": apl_gain,
        "n_kc": int(n_kc),
        "n_channels": int(n_channels),
        "k": int(k),
        "mean_claws": mean_claws,
        "fly": fly_map,
        "shuffled_mean": float(np.mean(shuffles)),
        "shuffled": shuffles,
        "uniform": score(
            uniform_projection(n_kc, n_channels, mean_claws, UNIFORM_SEED)
        ),
        "gaussian": score(dense_gaussian_projection(n_kc, n_channels, GAUSSIAN_SEED)),
        "verdict": verdict_label,
        "verdict_margin_in_sd": margin_in_sd,
        "verdict_borderline": borderline,
        "n_shuffles_below": below,
        "n_shuffles_above": above,
        "empirical_p_two_sided": min(
            1.0, 2.0 * min(below + 1, above + 1) / (len(shuffles) + 1)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the Phase 2 robustness sweeps")
    p.add_argument("--circuit", default="data/circuit.npz")
    p.add_argument("--out", default="results/phase2-sweeps.json")
    p.add_argument("--shuffles", type=int, default=100)
    p.add_argument("--datasets", nargs="+", default=["mnist", "glove", "sift"])
    args = p.parse_args(argv)

    circuit = Circuit.load(args.circuit)
    results = []
    for name in args.datasets:
        database, queries = DATASETS[name]()
        for side, condition in itertools.product(("L", "R"), sweep_conditions()):
            row = run_condition(
                circuit.hemisphere(side), side, database, queries,
                n_shuffles=args.shuffles, **condition,
            )
            fraction, gain, level = (
                condition["hash_fraction"], condition["apl_gain"], condition["level"]
            )
            row["dataset"] = name
            results.append(row)
            if row["verdict_borderline"]:
                marker = f" [BORDERLINE, {row['verdict_margin_in_sd']:.2f} SD]"
            else:
                marker = ""
            print(
                f"{name:6s} {side} {level:5s} frac={fraction:.2f} gain={gain:.0e} "
                f"fly={row['fly']:.4f} shuf={row['shuffled_mean']:.4f} "
                f"{row['verdict']}{marker} p={row['empirical_p_two_sided']:.3f}",
                flush=True,
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    matches = [r for r in results if r["verdict"] == PHASE1_VERDICT[r["side"]]]
    differs = [r for r in results if r["verdict"] != PHASE1_VERDICT[r["side"]]]
    fly_better = [r for r in results if r["verdict"] == "fly_better"]

    print(f"\nconditions run: {len(results)}")
    print(f"conditions matching Phase 1's verdict for their hemisphere: {len(matches)}")
    print(f"conditions that differ from Phase 1's verdict for their hemisphere: {len(differs)}")
    for r in differs:
        print(
            f"  FLIP {r['dataset']:6s} {r['side']} {r['level']:5s} "
            f"frac={r['hash_fraction']:.2f} gain={r['apl_gain']:.0e} "
            f"phase1={PHASE1_VERDICT[r['side']]} phase2={r['verdict']}"
        )
    print(f"conditions where the fly beat the shuffle (fly_better): {len(fly_better)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
