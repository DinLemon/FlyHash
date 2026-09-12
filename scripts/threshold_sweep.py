"""The synapse-weight-threshold sweep that Phase 2 left undone.

Spec section 7 assigns thresholds 1, 5 and 10 to Phase 2 as a robustness check
alongside the hash-length sweep; only the latter was run. The pre-registered
threshold is 3, used everywhere else.

Changing the threshold changes which edges survive, so each value needs its own
extracted circuit. Everything else is held at the Phase 1 baseline: PN input
level, 5% hash, no APL inhibition, MNIST.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyhash.circuit import Circuit
from flyhash.datasets import load_mnist
from flyhash.extract import extract_circuit
from flyhash.phase2 import run_condition

ANNOTATIONS = "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = "data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
PREREGISTERED_THRESHOLD = 3
THRESHOLDS = (1, 3, 5, 10)


def circuit_for(threshold: int, cache_dir: Path) -> Circuit:
    """Extract (or reuse) the circuit at one synapse-weight threshold."""
    path = cache_dir / f"circuit-minweight-{threshold}.npz"
    if path.exists():
        return Circuit.load(path)
    circuit = extract_circuit(ANNOTATIONS, WEIGHTS, min_weight=threshold)
    cache_dir.mkdir(parents=True, exist_ok=True)
    circuit.save(path)
    return circuit


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sweep the synapse-weight threshold")
    p.add_argument("--out", default="results/phase2-threshold-sweep.json")
    p.add_argument("--cache", default="data/thresholds")
    p.add_argument("--shuffles", type=int, default=100)
    args = p.parse_args(argv)

    database, queries = load_mnist()
    results = []
    for threshold in THRESHOLDS:
        circuit = circuit_for(threshold, Path(args.cache))
        for side in ("L", "R"):
            hemisphere = circuit.hemisphere(side)
            row = run_condition(
                hemisphere, side, database, queries,
                level="pn", hash_fraction=0.05, apl_gain=0.0,
                n_shuffles=args.shuffles,
            )
            row["min_weight"] = threshold
            row["preregistered"] = threshold == PREREGISTERED_THRESHOLD
            results.append(row)
            print(
                f"thr={threshold:<3d} {side} n_kc={row['n_kc']:<5d} "
                f"n_pn={row['n_channels']:<4d} k={row['k']:<4d} "
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

    verdicts = {r["verdict"] for r in results}
    print(f"\nthresholds swept: {THRESHOLDS}")
    print(f"distinct verdicts across the sweep: {sorted(verdicts)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
