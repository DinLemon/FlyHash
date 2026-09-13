"""Export the sparse codes the circuit produces, and look at them.

The pipeline has no memory: one input vector goes in, one sparse binary code
comes out, and the two hemispheres are encoded separately and never combined.
This script dumps those codes so they can be inspected directly, and draws a
PCA of them alongside the same view for a degree-preserving shuffle.

Digit identity is carried by the glyph, not by colour — ten categorical hues
would fail every colour-vision check.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flyhash.circuit import Circuit
from flyhash.datasets import load_mnist
from flyhash.encode import encode, make_compressor
from flyhash.models import fly_projection, hash_length, shuffled_projection

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
CLOUD = "#c9c8c2"
BLUE = "#2a78d6"
GRID = "#e4e3de"

LABELS = Path("data/datasets/train-labels.npy")


def codes_for(projection, database, compressor, k) -> np.ndarray:
    return encode(database, compressor, projection, k).toarray().astype(np.float32)


def write_csv(path: Path, codes: np.ndarray, labels: np.ndarray, k: int) -> None:
    """One row per input: its digit, then the indices of the active cells.

    Stored as indices rather than 1865 mostly-zero columns — the code is
    exactly k-sparse by construction, so this is lossless and ~20x smaller.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["digit"] + [f"active_{i}" for i in range(k)])
        for label, row in zip(labels, codes):
            writer.writerow([int(label)] + np.flatnonzero(row).tolist())
    print(f"wrote {path}  ({len(codes)} codes, {k} active cells each)")


def pca_2d(codes: np.ndarray) -> np.ndarray:
    centred = codes - codes.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    return centred @ components[:2].T


def panel(ax, points, labels, title) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(points[:, 0], points[:, 1], s=6, color=CLOUD, alpha=0.55,
               linewidths=0, zorder=2)
    for digit in range(10):
        centre = points[labels == digit].mean(axis=0)
        ax.text(centre[0], centre[1], str(digit), fontsize=19, color=BLUE,
                fontweight="bold", ha="center", va="center", zorder=4,
                path_effects=None)
    ax.set_title(title, fontsize=12, color=INK, pad=10, loc="left",
                 fontweight="bold")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export and visualise the sparse codes")
    p.add_argument("--circuit", default="data/circuit.npz")
    p.add_argument("--side", default="L")
    p.add_argument("--count", type=int, default=4000)
    p.add_argument("--out-dir", default="results/codes")
    args = p.parse_args(argv)

    if not LABELS.exists():
        raise SystemExit(
            f"{LABELS} is missing — it holds the MNIST digit labels used to "
            "colour the plot. Fetch train-labels-idx1-ubyte.gz from the MNIST "
            "mirror and save it as this .npy."
        )

    circuit = Circuit.load(args.circuit).hemisphere(args.side)
    fly = fly_projection(circuit)
    n_kc, n_pn = fly.shape
    k = hash_length(n_kc)

    database, _ = load_mnist()
    database = database[: args.count]
    labels = np.load(LABELS)[: args.count]
    compressor = make_compressor(database.shape[1], n_pn, seed=0)

    out = Path(args.out_dir)
    fly_codes = codes_for(fly, database, compressor, k)
    shuffled_codes = codes_for(
        shuffled_projection(fly, seed=1000), database, compressor, k
    )
    write_csv(out / f"codes-fly-{args.side}.csv", fly_codes, labels, k)
    write_csv(out / f"codes-shuffled-{args.side}.csv", shuffled_codes, labels, k)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    panel(axes[0], pca_2d(fly_codes), labels, "Measured wiring")
    panel(axes[1], pca_2d(shuffled_codes), labels, "Degree-preserving shuffle")
    fig.suptitle(
        f"The {k}-bit codes carry digit structure — and the shuffle carries it too",
        fontsize=13, color=INK, fontweight="bold", x=0.09, ha="left", y=1.00,
    )
    fig.text(
        0.09, -0.02,
        "PCA of the sparse codes for 4000 MNIST digits. Grey dots are individual "
        "codes; numerals mark each digit's centroid.",
        fontsize=9.5, color=INK_SOFT, ha="left",
    )
    figures = Path("docs/figures")
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "fig5-code-space.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figures / 'fig5-code-space.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
