"""Extended Data figure for defense sensitivity to progression mapping."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"b": "#7B8FA1", "c": "#C24D32", "d": "#4C7C6B"}


def plot_progression_sensitivity(results, output_dir: Path):
    direct = results["direct"].query("sample == 'Pooled'").set_index("method")
    spill = results["spillover"].query("sample == 'Pooled'").set_index("method")
    order = ["b", "c", "d"]
    labels = [
        "b  Linear interpolation",
        "c  Kriging update (primary)",
        "d  Direct regression kriging",
    ]
    y = np.arange(3)[::-1]
    panels = [
        (direct, "survival_difference_pp", "ci_lo", "ci_hi", 1,
         "a  Focal survival difference", "percentage points"),
        (spill, "directional_contrast", "directional_contrast_lo",
         "directional_contrast_hi", 100,
         "b  Direction-adjusted neighbor contrast", "percentage points"),
        (spill, "local_iv", "local_iv_lo", "local_iv_hi", 100,
         "c  Local IV ratio", "percentage points"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35))
    for ax, (frame, estimate, low, high, scale, title, xlabel) in zip(axes, panels):
        values = scale * frame.loc[order, estimate].to_numpy(float)
        lower = scale * frame.loc[order, low].to_numpy(float)
        upper = scale * frame.loc[order, high].to_numpy(float)
        for i, code in enumerate(order):
            ax.errorbar(
                values[i], y[i],
                xerr=[[values[i] - lower[i]], [upper[i] - values[i]]],
                fmt="o", ms=5.2, color=COLORS[code], mfc=COLORS[code],
                mec="white", mew=.55, elinewidth=1.15, capsize=2.3, zorder=3,
            )
        ax.axvline(0, color="#777777", lw=.75, ls="--", zorder=1)
        ax.grid(axis="x", color="#DEDEDE", lw=.55, alpha=.75)
        ax.set_title(title, loc="left", fontsize=9.2, pad=5)
        ax.set_xlabel(xlabel, fontsize=8.2)
        ax.set_ylim(-.65, 2.65)
        ax.tick_params(axis="both", labelsize=7.7)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, labels, fontsize=7.7)
    for ax in axes[1:]:
        ax.set_yticks(y, [])
    axes[0].text(
        .01, -.29, "greater survival →", transform=axes[0].transAxes,
        fontsize=7.1, color="#555555",
    )
    axes[1].text(
        .01, -.29, "fewer later-arriving neighbor losses ←",
        transform=axes[1].transAxes, fontsize=7.1, color="#555555",
    )
    axes[2].text(
        .01, -.29, "fewer later-arriving neighbor losses ←",
        transform=axes[2].transAxes, fontsize=7.1, color="#555555",
    )
    fig.suptitle(
        "Defense estimates across Eaton fire-progression reconstructions",
        x=.055, ha="left", fontsize=10.5, y=.995,
    )
    fig.text(
        .055, .91,
        "Palisades arrival times and all non-progression analysis choices are fixed; points show estimates and 95% bootstrap intervals.",
        ha="left", fontsize=7.6, color="#4D4D4D",
    )
    fig.subplots_adjust(left=.235, right=.99, bottom=.25, top=.78, wspace=.30)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "ED_fig_defense_progression_map_sensitivity"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return fig
