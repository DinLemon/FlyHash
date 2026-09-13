"""Run from the repository root: `python scripts/wiring_structure.py`.

Is the wiring structured at all? A test that needs neither a task nor dynamics.

The retrieval benchmarks in this repository ask whether the measured wiring
*helps* at nearest-neighbour search. A fair objection is that the benchmark, or
the static model behind it, is the wrong readout.

This asks a narrower question that avoids both: do Kenyon cells sample
glomeruli in a pattern a degree-preserving shuffle would not produce? That is
about the wiring alone. No encoding, no metric, no model of neural dynamics —
just which glomeruli end up on the same cell, measured against the same null
the rest of the study uses.

Two statistics are reported:

  within-KC correlation
      the average pairwise correlation, across the 110 Hallem odorants, of the
      glomeruli each cell samples. High means cells pool inputs that respond
      alike (redundancy); low means they pool inputs that respond differently
      (decorrelation, what efficient coding would predict).

  pair over-representation
      how often each glomerulus pair lands on the same cell, against the null.
      This is where the structure actually lives.

The shuffle is restricted to the Hallem columns, so every cell keeps its exact
number of measured inputs and every glomerulus its exact number of targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import sparse, stats

from flyhash.circuit import Circuit
from flyhash.glomeruli import aggregate_by_glomerulus
from flyhash.models import shuffled_projection

# Importable both as a script (`python scripts/wiring_structure.py`) and as a
# module (`from scripts.wiring_structure import ...`), so put its own
# directory on the path rather than relying on how it was invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from odour_experiment import HEMISPHERES, load_odours  # noqa: E402

SHUFFLE_SEED_BASE = 7000


def within_kc_correlation(binary: np.ndarray, correlation: np.ndarray) -> float:
    """Mean pairwise correlation among the glomeruli each cell samples."""
    values = []
    for row in range(binary.shape[0]):
        cols = np.flatnonzero(binary[row])
        if len(cols) < 2:
            continue
        block = correlation[np.ix_(cols, cols)]
        values.append(block[np.triu_indices(len(cols), 1)].mean())
    return float(np.mean(values))


def pair_counts(binary: np.ndarray, n: int) -> np.ndarray:
    """How many cells sample each glomerulus pair together."""
    counts = np.zeros((n, n))
    for row in range(binary.shape[0]):
        cols = np.flatnonzero(binary[row])
        for i, j in combinations(cols, 2):
            counts[i, j] += 1
    return counts


def run(n_shuffles: int) -> dict:
    X, glomeruli, _ = load_odours()
    pearson = np.corrcoef(X.T)
    spearman = stats.spearmanr(X).statistic
    n = len(glomeruli)
    upper = np.triu_indices(n, 1)

    report: dict = {"glomeruli": glomeruli, "hemispheres": []}
    for animal, side, path in HEMISPHERES:
        circuit = Circuit.load(path).hemisphere(side)
        aggregated, names = aggregate_by_glomerulus(circuit)
        names = list(names)
        columns = [names.index(g) for g in glomeruli]
        binary = (aggregated.toarray() > 0).astype(np.float32)[:, columns]
        sparse_binary = sparse.csr_array(binary)

        shuffles = [
            shuffled_projection(sparse_binary, seed=SHUFFLE_SEED_BASE + i).toarray()
            for i in range(n_shuffles)
        ]

        entry: dict = {"animal": animal, "side": side, "n_shuffles": n_shuffles}
        for label, correlation in (("pearson", pearson), ("spearman", spearman)):
            observed = within_kc_correlation(binary, correlation)
            null = np.array(
                [within_kc_correlation(s, correlation) for s in shuffles]
            )
            entry[label] = {
                "fly": observed,
                "shuffled_mean": float(null.mean()),
                "shuffled_sd": float(null.std()),
                "z": float((observed - null.mean()) / null.std()),
                "n_below": int((null < observed).sum()),
            }

        real = pair_counts(binary, n)
        null_pairs = np.array([pair_counts(s, n) for s in shuffles])
        mean, spread = null_pairs.mean(0), null_pairs.std(0) + 1e-9
        excess_z = ((real - mean) / spread)[upper]
        entry["pair_vs_correlation_r"] = float(
            np.corrcoef((real - mean)[upper], pearson[upper])[0, 1]
        )
        order = np.argsort(-excess_z)
        names_upper = [(glomeruli[i], glomeruli[j]) for i, j in zip(*upper)]
        entry["top_pairs"] = [
            {
                "pair": list(names_upper[t]),
                "observed": float(real[upper][t]),
                "expected": float(mean[upper][t]),
                "z": float(excess_z[t]),
                "odour_correlation": float(pearson[upper][t]),
            }
            for t in order[:8]
        ]
        report["hemispheres"].append(entry)

        p = entry["pearson"]
        print(
            f"{animal:6s} {side}: within-KC correlation fly={p['fly']:+.4f} "
            f"shuffled={p['shuffled_mean']:+.4f} z={p['z']:+.2f} "
            f"({p['n_below']}/{n_shuffles})  "
            f"pair-vs-correlation r={entry['pair_vs_correlation_r']:+.3f}",
            flush=True,
        )
        for pair in entry["top_pairs"][:3]:
            print(
                f"         {pair['pair'][0]:5s}-{pair['pair'][1]:5s} "
                f"{pair['observed']:5.0f} vs {pair['expected']:5.1f} expected, "
                f"z={pair['z']:+.1f}",
                flush=True,
            )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Test the wiring for structure")
    p.add_argument("--out", default="results/wiring-structure.json")
    p.add_argument("--shuffles", type=int, default=200)
    args = p.parse_args(argv)

    report = run(args.shuffles)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
