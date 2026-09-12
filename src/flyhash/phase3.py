"""Phase 3: does the null replicate in a second animal?

The male and female circuits are run through the identical baseline
condition. The male circuit must be the one extracted at min_weight=5, not
the study's default of 3, because FlyWire's published table is already
thresholded at 5 — comparing 3 against 5 would confound sex with the
edge-inclusion criterion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flyhash.bench import GROUND_TRUTH_K
from flyhash.circuit import Circuit
from flyhash.datasets import DATASETS
from flyhash.phase2 import run_condition

MATCHED_THRESHOLD = 5
ANIMALS = {
    "male": f"data/thresholds/circuit-minweight-{MATCHED_THRESHOLD}.npz",
    "female": "data/circuit-flywire.npz",
}


def run_replication(
    circuits: dict[str, Circuit],
    database: np.ndarray,
    queries: np.ndarray,
    dataset_name: str,
    n_shuffles: int = 100,
    ground_truth_k: int = GROUND_TRUTH_K,
) -> list[dict]:
    rows = []
    for animal, circuit in circuits.items():
        for side in sorted(set(circuit.kc_sides)):
            row = run_condition(
                circuit.hemisphere(side), side, database, queries,
                level="pn", hash_fraction=0.05, apl_gain=0.0,
                n_shuffles=n_shuffles, ground_truth_k=ground_truth_k,
            )
            row["animal"] = animal
            row["dataset"] = dataset_name
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the cross-animal replication")
    p.add_argument("--out", default="results/phase3-replication.json")
    p.add_argument("--shuffles", type=int, default=100)
    p.add_argument("--datasets", nargs="+", default=["mnist", "glove", "sift"])
    args = p.parse_args(argv)

    circuits = {name: Circuit.load(path) for name, path in ANIMALS.items()}
    results = []
    for name in args.datasets:
        database, queries = DATASETS[name]()
        for row in run_replication(circuits, database, queries, name, args.shuffles):
            results.append(row)
            print(
                f"{name:6s} {row['animal']:6s} {row['side']} "
                f"n_kc={row['n_kc']:<5d} n_pn={row['n_channels']:<4d} "
                f"fly={row['fly']:.4f} shuf={row['shuffled_mean']:.4f} "
                f"d={row['fly'] - row['shuffled_mean']:+.4f} "
                f"{row['verdict']} p={row['empirical_p_two_sided']:.3f}",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(out)

    print(f"\nconditions: {len(results)}")
    for animal in ANIMALS:
        rows = [r for r in results if r["animal"] == animal]
        deltas = [r["fly"] - r["shuffled_mean"] for r in rows]
        print(
            f"  {animal:6s}: mean fly-minus-shuffle {np.mean(deltas):+.4f}, "
            f"negative in {sum(d < 0 for d in deltas)}/{len(deltas)}, "
            f"fly_better in {sum(r['verdict'] == 'fly_better' for r in rows)}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
