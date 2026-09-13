"""Render the README figures from the committed result files.

Static PNGs, because GitHub renders READMEs without JavaScript. Every number
comes from results/*.json — nothing here is typed by hand, so the figures cannot
drift from the data.

Palette: slots 1-3 of the validated categorical set, which are the three that
clear the all-pairs colour-vision floors. The aqua slot sits below 3:1 contrast
on this surface, so every series carrying it is also directly labelled.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGURES = Path("docs/figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

DATASET_COLOUR = {"mnist": BLUE, "glove": ORANGE, "sift": AQUA}
DATASET_LABEL = {"mnist": "MNIST", "glove": "GloVe", "sift": "SIFT"}


def load(name: str) -> list[dict]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def style(ax) -> None:
    """Recessive axes and grid; the data carries the emphasis."""
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(colors=INK_SOFT, labelsize=10, length=3, width=1.0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    return fig, ax


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES / name}")


def z_score(row: dict) -> float:
    """The effect in units of that condition's own null spread."""
    shuffles = np.asarray(row["shuffled"], dtype=float)
    spread = shuffles.std()
    return float((row["fly"] - shuffles.mean()) / spread) if spread else 0.0


# ---------------------------------------------------------------- figure 1

def fig_effect_against_null() -> None:
    """Every condition, expressed in units of its own shuffle distribution."""
    groups = [
        ("Phase 1 — MNIST", load("phase1-mnist.json")),
        ("Phase 2 — 48 sweep conditions", load("phase2-sweeps.json")),
        ("Phase 2 — synapse thresholds", load("phase2-threshold-sweep.json")),
        ("Phase 3 — two animals", load("phase3-replication.json")),
        ("Phase 4 — native odours", load("odour-experiment.json")),
    ]
    fig, ax = figure(9.5, 5.3)
    rng = np.random.default_rng(0)

    ax.axvspan(-1.96, 1.96, color=BLUE, alpha=0.07, zorder=0)
    for edge in (-1.96, 1.96):
        ax.axvline(edge, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=1)

    labels = []
    for position, (label, rows) in enumerate(groups):
        scores = [z_score(r) for r in rows]
        jitter = rng.uniform(-0.16, 0.16, len(scores))
        ax.scatter(
            scores, np.full(len(scores), position) + jitter,
            s=46, color=BLUE, alpha=0.75,
            edgecolor=SURFACE, linewidth=1.2, zorder=3,
        )
        labels.append(f"{label}\n{len(rows)} conditions")

    every = [z_score(r) for _, rows in groups for r in rows]
    below = sum(1 for z in every if z < -1.96)
    above = sum(1 for z in every if z > 1.96)
    median = float(np.median(every))
    ax.axvline(median, color=ORANGE, linewidth=2.0, zorder=2)

    ax.set_yticks(range(len(groups)), labels, fontsize=10, color=INK_SOFT)
    ax.set_ylim(-0.75, len(groups) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(
        "fly minus shuffled wiring, in standard deviations of that condition's own null",
        fontsize=10, color=INK_SOFT,
    )
    ax.grid(axis="y", visible=False)
    ax.text(0.0, -0.70, "  null band (95%)", fontsize=9, color=INK_MUTED,
            ha="left", va="center")
    ax.text(median - 0.12, -0.70, f"median {median:+.2f}  ", fontsize=9,
            color=ORANGE, ha="right", va="center", fontweight="bold")
    ax.text(
        0.01, 0.02,
        f"{below} of {len(every)} conditions fall below the band (fly worse)"
        f"  ·  {above} above it (fly better)",
        transform=ax.transAxes, fontsize=10, color=INK_SOFT, ha="left", va="bottom",
    )
    ax.set_title(
        "Where the wiring differs from random, it is worse — almost never better",
        fontsize=13, color=INK, pad=14, loc="left", fontweight="bold",
    )
    save(fig, "fig1-effect-against-null.png")


# ---------------------------------------------------------------- figure 2

def fig_effect_against_biological_noise() -> None:
    """The effect compared with the study's two measured noise floors."""
    rows = load("phase3-replication.json")
    fly = {(r["dataset"], r["animal"], r["side"]): r["fly"] for r in rows}
    datasets = ["mnist", "glove", "sift"]

    within, across, effect = [], [], []
    for d in datasets:
        within += [abs(fly[(d, a, "L")] - fly[(d, a, "R")]) for a in ("male", "female")]
        across += [
            abs(fly[(d, "male", a)] - fly[(d, "female", b)])
            for a in "LR" for b in "LR"
        ]
        effect.append(
            max(abs(r["fly"] - r["shuffled_mean"]) for r in rows if r["dataset"] == d)
        )

    bars = [
        ("Between the two hemispheres\nof one individual", float(np.mean(within)), INK_MUTED),
        ("Between two different animals\nof opposite sex", float(np.mean(across)), INK_MUTED),
        ("Measured wiring vs\ndegree-matched random", float(np.mean(effect)), BLUE),
    ]

    fig, ax = figure(9.0, 3.6)
    positions = np.arange(len(bars))
    for position, (_, value, colour) in zip(positions, bars):
        ax.barh(position, value, height=0.55, color=colour, zorder=3)
        ax.text(
            value + 0.0006, position, f"{value:.4f}",
            va="center", ha="left", fontsize=11, color=INK, fontweight="bold",
        )
    ax.set_yticks(positions, [b[0] for b in bars], fontsize=10, color=INK_SOFT)
    ax.invert_yaxis()
    ax.set_xlim(0, max(b[1] for b in bars) * 1.22)
    ax.set_xlabel("difference in mAP@100", fontsize=10, color=INK_SOFT)
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "The effect under test is smaller than the fly's own biological noise",
        fontsize=13, color=INK, pad=14, loc="left", fontweight="bold",
    )
    save(fig, "fig2-effect-vs-noise.png")


# ---------------------------------------------------------------- figure 3

def fig_apl_gain() -> None:
    """What the fly's measured inhibition does to retrieval quality."""
    rows = [
        r for r in load("phase2-sweeps.json")
        if r["level"] == "pn" and r["hash_fraction"] == 0.05
    ]
    fig, ax = figure(8.6, 4.4)

    for dataset in ("mnist", "glove", "sift"):
        subset = sorted(
            (r for r in rows if r["dataset"] == dataset), key=lambda r: r["apl_gain"]
        )
        gains = sorted({r["apl_gain"] for r in subset})
        means = [
            float(np.mean([r["fly"] for r in subset if r["apl_gain"] == g]))
            for g in gains
        ]
        colour = DATASET_COLOUR[dataset]
        ax.plot(range(len(gains)), means, color=colour, linewidth=2.0, zorder=3)
        ax.scatter(range(len(gains)), means, s=52, color=colour,
                   edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.text(
            len(gains) - 1 + 0.09, means[-1], DATASET_LABEL[dataset],
            color=colour, fontsize=11, fontweight="bold", va="center", ha="left",
        )

    gains = sorted({r["apl_gain"] for r in rows})
    ax.set_xticks(range(len(gains)), ["0", "1e-5", "3e-5", "1e-4"], fontsize=10)
    ax.set_xlim(-0.2, len(gains) - 0.55)
    ax.set_xlabel("strength of the fly's measured APL inhibition", fontsize=10, color=INK_SOFT)
    ax.set_ylabel("mAP@100  (mean of both hemispheres)", fontsize=10, color=INK_SOFT)
    ax.set_title(
        "Adding the fly's real inhibition makes retrieval steadily worse",
        fontsize=13, color=INK, pad=14, loc="left", fontweight="bold",
    )
    save(fig, "fig3-apl-gain.png")


# ---------------------------------------------------------------- figure 4

def fig_replication() -> None:
    """The same measurement in two animals of opposite sex."""
    rows = load("phase3-replication.json")
    fig, ax = figure(8.6, 3.8)

    ax.axvline(0, color=INK_MUTED, linewidth=1.2, zorder=2)
    offsets = {"male": -0.16, "female": 0.16}
    colours = {"male": BLUE, "female": ORANGE}
    datasets = ["mnist", "glove", "sift"]

    for position, dataset in enumerate(datasets):
        for animal, offset in offsets.items():
            deltas = [
                r["fly"] - r["shuffled_mean"]
                for r in rows if r["dataset"] == dataset and r["animal"] == animal
            ]
            ax.scatter(
                deltas, np.full(len(deltas), position + offset),
                s=80, color=colours[animal], alpha=0.85,
                edgecolor=SURFACE, linewidth=1.5, zorder=3,
                label=animal if position == 0 else None,
            )

    ax.set_yticks(range(len(datasets)), [DATASET_LABEL[d] for d in datasets],
                  fontsize=11, color=INK_SOFT)
    ax.set_ylim(-0.6, len(datasets) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("fly minus shuffled wiring, in mAP@100", fontsize=10, color=INK_SOFT)
    # Legend sits above the plot: inside it would collide with the SIFT points.
    legend = ax.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False,
        fontsize=10, labelcolor=INK_SOFT, handletextpad=0.4, columnspacing=1.6,
    )
    for text, animal in zip(legend.get_texts(), offsets):
        text.set_text({"male": "male — male-CNS", "female": "female — FlyWire"}[animal])
    ax.set_title(
        "The same slight disadvantage reappears in a second animal of the other sex",
        fontsize=13, color=INK, pad=34, loc="left", fontweight="bold",
    )
    save(fig, "fig4-replication.png")


def main() -> int:
    fig_effect_against_null()
    fig_effect_against_biological_noise()
    fig_apl_gain()
    fig_replication()
    fig_odour()
    return 0



# ---------------------------------------------------------------- figure 6

HEMI_COLOUR = {("male", "L"): BLUE, ("male", "R"): ORANGE,
               ("female", "L"): AQUA, ("female", "R"): "#eda100"}


def fig_odour() -> None:
    """The circuit's own modality, with channel identity preserved."""
    rows = json.loads((RESULTS / "odour-experiment.json").read_text(encoding="utf-8"))
    matched = json.loads(
        (RESULTS / "odour-vs-mnist-matched.json").read_text(encoding="utf-8")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=150,
                             gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style(ax)

    ax = axes[0]
    fractions = sorted({r["hash_fraction"] for r in rows})
    ax.axhspan(-1.96, 1.96, color=BLUE, alpha=0.07, zorder=0)
    for edge in (-1.96, 1.96):
        ax.axhline(edge, color=INK_MUTED, linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=1)
    for animal in ("male", "female"):
        for side in ("L", "R"):
            series = sorted(
                (r for r in rows if r["animal"] == animal and r["side"] == side),
                key=lambda r: r["hash_fraction"],
            )
            colour = HEMI_COLOUR[(animal, side)]
            zs = [r["z"] for r in series]
            ax.plot(range(len(zs)), zs, color=colour, linewidth=2.0, zorder=3)
            ax.scatter(range(len(zs)), zs, s=50, color=colour,
                       edgecolor=SURFACE, linewidth=1.4, zorder=4)
            ax.text(len(zs) - 1 + 0.08, zs[-1], f"{animal} {side}", color=colour,
                    fontsize=10, fontweight="bold", va="center", ha="left")
    ax.set_xticks(range(len(fractions)), [f"{int(f*100)}%" for f in fractions],
                  fontsize=10)
    ax.set_xlim(-0.15, len(fractions) - 0.35)
    ax.set_xlabel("fraction of Kenyon cells kept active", fontsize=10, color=INK_SOFT)
    ax.set_ylabel("fly minus shuffle, in null SDs", fontsize=10, color=INK_SOFT)
    ax.set_title("One spike, and it does not survive a change of sparsity",
                 fontsize=12, color=INK, pad=12, loc="left", fontweight="bold")

    ax = axes[1]
    labels, flies, shufs = [], [], []
    for key, label in (("odour", "odours\n(native, 23 channels)"),
                       ("mnist", "MNIST\n(via random compressor)")):
        labels.append(label)
        flies.append(matched[key]["fly"])
        shufs.append(float(np.mean(matched[key]["shuffled"])))
    positions = np.arange(len(labels))
    ax.bar(positions - 0.19, flies, width=0.34, color=BLUE, zorder=3,
           label="measured wiring")
    ax.bar(positions + 0.19, shufs, width=0.34, color=INK_MUTED, zorder=3,
           label="degree-preserving shuffle")
    for x, v in zip(positions - 0.19, flies):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, color=INK,
                fontweight="bold")
    for x, v in zip(positions + 0.19, shufs):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, color=INK_SOFT)
    ax.set_xticks(positions, labels, fontsize=10, color=INK_SOFT)
    ax.set_ylim(0, max(flies + shufs) * 1.22)
    ax.set_ylabel("mAP@10  (110 items, matched difficulty)", fontsize=10,
                  color=INK_SOFT)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK_SOFT)
    ax.set_title("Native data lifts both — it does not favour the fly",
                 fontsize=12, color=INK, pad=12, loc="left", fontweight="bold")

    fig.suptitle(
        "The decisive test: the circuit's own modality, with channel identity intact",
        fontsize=13.5, color=INK, fontweight="bold", x=0.075, ha="left", y=1.04,
    )
    save(fig, "fig6-odour.png")

if __name__ == "__main__":
    raise SystemExit(main())
