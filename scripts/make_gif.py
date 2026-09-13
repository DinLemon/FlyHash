"""Animate the study's central method, and the fragility of its one spike.

Both panels are the same hemisphere of the same fly on the same odour data.
Only the sparsity differs. Shuffles accumulate one batch at a time into the
null distribution while the measured wiring's score stays fixed, so you watch
a convincing-looking result form on the left and dissolve on the right.

Written as a GIF because it is meant for a README, which runs no JavaScript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

RESULTS = Path("results/odour-experiment.json")
BATCH = 4
HOLD_FRAMES = 14


def panel_rows(rows, animal, side, fractions):
    picked = []
    for fraction in fractions:
        match = [
            r for r in rows
            if r["animal"] == animal and r["side"] == side
            and abs(r["hash_fraction"] - fraction) < 1e-9
        ]
        if not match:
            raise SystemExit(f"no row for {animal} {side} at {fraction}")
        picked.append(match[0])
    return picked


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Animate the null distribution forming")
    p.add_argument("--animal", default="male")
    p.add_argument("--side", default="R")
    p.add_argument("--out", default="docs/figures/null-forming.gif")
    args = p.parse_args(argv)

    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    left, right = panel_rows(rows, args.animal, args.side, [0.05, 0.10])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.05, right=0.97, wspace=0.16)

    span = np.concatenate([
        np.asarray(left["shuffled"]), np.asarray(right["shuffled"]),
        [left["fly"], right["fly"]],
    ])
    lo, hi = span.min() - 0.012, span.max() + 0.012
    bins = np.linspace(lo, hi, 34)

    def setup(ax, row, title):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=9, length=3)
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 42)
        ax.set_xlabel("mAP@10", fontsize=9.5, color=INK_SOFT)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left",
                     fontweight="bold", pad=10)
        ax.axvline(row["fly"], color=BLUE, linewidth=2.4, zorder=5)
        near_right = row["fly"] > lo + 0.62 * (hi - lo)
        ax.text(row["fly"], 34.0,
                "measured wiring " if near_right else " measured wiring",
                color=BLUE, fontsize=9.5, fontweight="bold", va="center",
                ha="right" if near_right else "left", zorder=6)

    setup(axes[0], left, f"{args.animal} {args.side} — 5% of cells active")
    setup(axes[1], right, f"{args.animal} {args.side} — 10% of cells active")

    caption = fig.text(0.5, 0.845, "", fontsize=10.5, color=INK_SOFT, ha="center")
    # Verdicts live inside their own axes, so they cannot collide with titles.
    verdicts = [
        ax.text(0.98, 0.95, "", transform=ax.transAxes, fontsize=10,
                ha="right", va="top", fontweight="bold", zorder=7)
        for ax in axes
    ]
    fig.suptitle(
        "The same fly, the same smells — only the sparsity changes",
        fontsize=13.5, color=INK, fontweight="bold", y=0.96,
    )

    n_total = min(len(left["shuffled"]), len(right["shuffled"]))
    steps = list(range(BATCH, n_total + 1, BATCH))
    frames = steps + [n_total] * HOLD_FRAMES

    def draw(count):
        for ax, row in zip(axes, (left, right)):
            for patch in list(ax.patches):
                patch.remove()
            drawn = np.asarray(row["shuffled"][:count])
            ax.hist(drawn, bins=bins, color=INK_MUTED, alpha=0.75, zorder=3)
        caption.set_text(f"{count} of {n_total} degree-preserving shuffles drawn")
        for text, row in zip(verdicts, (left, right)):
            drawn = np.asarray(row["shuffled"][:count])
            below = int((drawn < row["fly"]).sum())
            beats = below == count
            text.set_text(f"{below} of {count} shuffles below the fly")
            text.set_color(ORANGE if beats else INK_SOFT)
        return []

    animation = FuncAnimation(fig, draw, frames=frames, interval=110, blit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out, writer=PillowWriter(fps=9),
                   savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
