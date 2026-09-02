"""Extended Data graphics for defense-spillover sensitivity analyses."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.viz.style import apply_style


VERMILLION = "#B95038"
NAVY = "#2E5B82"
GRAY = "#777777"
LIGHT = "#D9D9D6"


def _forest(ax, frame, label_column, title, letter):
    y = np.arange(len(frame))[::-1]
    point = 100 * frame.directional_contrast.to_numpy()
    lo = 100 * frame.directional_lo.to_numpy()
    hi = 100 * frame.directional_hi.to_numpy()
    ax.errorbar(
        point, y, xerr=np.vstack([point - lo, hi - point]), fmt="o",
        color=VERMILLION, mfc="white", mew=1.1, ms=4.2,
        elinewidth=1.0, capsize=2.2,
    )
    ax.axvline(0, color=GRAY, lw=.8)
    ax.set_yticks(y, [
        f"{label.replace('<=', '≤')}\n({int(pairs):,} pairs)"
        for label, pairs in zip(frame[label_column], frame.pairs)
    ])
    ax.set_xlabel("Directional contrast (percentage points)")
    ax.grid(axis="x", color=LIGHT, lw=.45)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=8.4,
                 fontweight="bold", pad=5)


def _ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, len(values) + 1) / len(values)


def plot_defense_sensitivity(spatial, threshold, design, eligible, matched,
                             output_dir: Path):
    """Render a compact four-panel Extended Data sensitivity figure."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.15))
    _forest(
        axes[0, 0], spatial, "analysis",
        "Spatial dependence and defense isolation", "a",
    )

    threshold_plot = threshold.copy()
    threshold_plot["label"] = [
        f"{100 * value:.0f}% upstream destroyed" for value in
        threshold_plot.upstream_destroyed_threshold
    ]
    _forest(
        axes[0, 1], threshold_plot, "label",
        "Upstream-destruction threshold", "b",
    )

    _forest(
        axes[1, 0], design, "design",
        "Arrival window and neighbor radius", "c",
    )

    ax = axes[1, 1]
    styles = [
        (eligible[eligible.defended.eq(0)], "Eligible undefended", GRAY, "--"),
        (eligible[eligible.defended.eq(1)], "Eligible defended", VERMILLION, "--"),
        (matched[matched.defended.eq(0)], "Matched undefended", NAVY, "-"),
        (matched[matched.defended.eq(1)], "Matched defended", VERMILLION, "-"),
    ]
    for frame, label, color, linestyle in styles:
        x, y = _ecdf(frame.propensity)
        ax.plot(x, y, color=color, ls=linestyle,
                lw=1.25 if linestyle == "-" else .8, label=label)
    ax.set_xlabel("Estimated probability of documented defense")
    ax.set_ylabel("Cumulative share")
    ax.set_ylim(0, 1)
    ax.grid(color=LIGHT, lw=.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("d  Propensity-score overlap", loc="left", fontsize=8.4,
                 fontweight="bold", pad=5)
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")

    fig.text(
        .01, .005,
        "All sensitivity analyses restrict eligibility before rematching; "
        "intervals preserve matched pairs. Negative contrasts indicate fewer "
        "destroyed down-fire neighbors.",
        fontsize=5.8, color="#555555",
    )
    fig.subplots_adjust(left=.24, right=.985, top=.965, bottom=.085,
                        wspace=.58, hspace=.48)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "ED_fig_defense_spillover_sensitivity"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600,
                bbox_inches="tight", facecolor="white")
    return fig
