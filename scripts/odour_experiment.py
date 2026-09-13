"""The decisive test: the circuit's own modality, with channel identity intact.

Every earlier phase pushed MNIST, GloVe or SIFT through the olfactory circuit
via a random compressor, which destroys any correspondence between an input
dimension and a particular projection neuron. That design cannot detect wiring
tuned to specific input channels, because it scrambles which channel is which.

Here the input is real odour data — Hallem & Carlson (2006), 110 odorants
against 24 receptors — and the assignment is exact: receptor Or22a feeds
glomerulus DM2, which feeds the projection neurons the connectome says are
DM2's. No compressor. If the measured PN->KC wiring is tuned for anything,
this is where it should show.

Responses are converted to firing rates as SFR + delta, clipped at zero, since
the published matrix is a change from the spontaneous rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from flyhash.bench import mean_average_precision, rank_by_code_overlap
from flyhash.circuit import Circuit
from flyhash.encode import normalize, winner_take_all
from flyhash.glomeruli import aggregate_by_glomerulus
from flyhash.models import hash_length, shuffled_projection

# Paths resolve from the repository root, so these scripts run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
HALLEM = REPO_ROOT / "data" / "hallem-carlson-2006.csv"
SHUFFLE_SEED_BASE = 3000

# Receptor -> glomerulus, from the DoOR mapping table. Or33b is co-expressed in
# two glomeruli and is therefore excluded; the remaining 23 are unambiguous and
# all present in every hemisphere of both connectomes.
RECEPTOR_TO_GLOMERULUS = {
    "Or10a": "DL1", "Or19a": "DC1", "Or22a": "DM2", "Or23a": "DA3",
    "Or2a": "DA4m", "Or35a": "VC3", "Or43a": "DA4l", "Or43b": "VM2",
    "Or47a": "DM3", "Or47b": "VA1v", "Or49b": "VA5", "Or59b": "DM4",
    "Or65a": "DL3", "Or67a": "DM6", "Or67c": "VC4", "Or7a": "DL5",
    "Or82a": "VA6", "Or85a": "DM5", "Or85b": "VM5d", "Or85f": "DL4",
    "Or88a": "VA1d", "Or98a": "VM5v", "Or9a": "VM3",
}

HEMISPHERES = [
    ("male", "L", REPO_ROOT / "data" / "circuit.npz"),
    ("male", "R", REPO_ROOT / "data" / "circuit.npz"),
    ("female", "L", REPO_ROOT / "data" / "circuit-flywire.npz"),
    ("female", "R", REPO_ROOT / "data" / "circuit-flywire.npz"),
]
HASH_FRACTIONS = (0.02, 0.05, 0.10, 0.20)
NEIGHBOURS = 10


def load_odours() -> tuple[np.ndarray, list[str], list[str]]:
    table = pd.read_csv(HALLEM, index_col=0)
    spontaneous = table.loc[table.odorant == "sfr"].iloc[0]
    odours = table[table.odorant != "sfr"]
    receptors = [r for r in RECEPTOR_TO_GLOMERULUS if r in odours.columns]
    rates = odours[receptors].to_numpy(float) + spontaneous[receptors].to_numpy(float)
    return (
        np.clip(rates, 0, None).astype(np.float32),
        [RECEPTOR_TO_GLOMERULUS[r] for r in receptors],
        odours["odorant"].tolist(),
    )


def true_neighbours_small(X: np.ndarray, k: int) -> np.ndarray:
    """Exact nearest neighbours within a small set, excluding self."""
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(d, idx, axis=1).argsort(axis=1)
    return np.take_along_axis(idx, order, axis=1)


def score(projection: np.ndarray, X: np.ndarray, k: int, truth: np.ndarray) -> float:
    activity = np.asarray(normalize(X) @ projection.T, dtype=np.float32)
    codes = winner_take_all(activity, k)
    retrieved = rank_by_code_overlap(codes, codes, k=truth.shape[1] + 1)[:, 1:]
    return mean_average_precision(retrieved, truth)


def run(n_shuffles: int) -> list[dict]:
    X, glomeruli, _ = load_odours()
    truth = true_neighbours_small(X, NEIGHBOURS)
    print(f"odorants={X.shape[0]}  channels={X.shape[1]}  "
          f"firing rate {X.min():.0f}..{X.max():.0f} spikes/s")

    rows = []
    for animal, side, path in HEMISPHERES:
        circuit = Circuit.load(path).hemisphere(side)
        aggregated, names = aggregate_by_glomerulus(circuit)
        names = list(names)
        full = (aggregated.toarray() > 0).astype(np.float32)
        columns = [names.index(g) for g in glomeruli]

        # One set of shuffles per hemisphere, reused across hash fractions, so
        # the fractions are compared on the same null draws.
        shuffles = [
            shuffled_projection(sparse.csr_array(full), seed=SHUFFLE_SEED_BASE + i)
            .toarray()[:, columns]
            for i in range(n_shuffles)
        ]
        fly = full[:, columns]

        for fraction in HASH_FRACTIONS:
            k = hash_length(full.shape[0], fraction)
            fly_score = score(fly, X, k, truth)
            null = np.array([score(s, X, k, truth) for s in shuffles])
            spread = float(null.std())
            rows.append({
                "animal": animal, "side": side, "hash_fraction": fraction,
                "k": int(k), "n_kc": int(full.shape[0]), "n_channels": len(columns),
                "fly": float(fly_score),
                "shuffled_mean": float(null.mean()), "shuffled_sd": spread,
                "shuffled": null.tolist(),
                "z": float((fly_score - null.mean()) / spread) if spread else 0.0,
                "n_below": int((null < fly_score).sum()),
                "n_shuffles": len(null),
            })
            r = rows[-1]
            print(f"  {animal:6s} {side} frac={fraction:.2f} k={k:<4d} "
                  f"fly={r['fly']:.4f} shuf={r['shuffled_mean']:.4f} "
                  f"z={r['z']:+.2f}  below={r['n_below']}/{len(null)}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the native-odour experiment")
    p.add_argument("--out", default="results/odour-experiment.json")
    p.add_argument("--shuffles", type=int, default=200)
    args = p.parse_args(argv)

    rows = run(args.shuffles)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
