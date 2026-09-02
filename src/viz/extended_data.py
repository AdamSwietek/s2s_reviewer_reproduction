"""Extended Data figure builders."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext

from src.analysis.fragility import FIRES
from src.viz.style import FIRE_COLORS, log_grid


def plot_neighbor_radius_sweep(binned: pd.DataFrame, smooth: pd.DataFrame,
                               radii_ft, output_dir: Path,
                               source_dir: Path):
    """Plot F* by damaged-neighbor count for two fires and four radii."""
    styles = {
        "no_damage": {
            "label": "No damage", "marker": "s", "line": (0, (3.2, 2)),
            "filled": False, "color": "#777777",
        },
        "destroyed": {
            "label": "Destroyed", "marker": "o", "line": "-",
            "filled": True,
        },
    }
    fig, axes = plt.subplots(
        len(FIRES), len(radii_ft), figsize=(9.2, 4.9),
        sharey="row", sharex="col",
    )
    letters = "abcdefgh"
    for row, fire in enumerate(FIRES):
        for column, radius in enumerate(radii_ft):
            ax = axes[row, column]
            for outcome in ["no_damage", "destroyed"]:
                style = styles[outcome]
                color = FIRE_COLORS[fire] if outcome == "destroyed" else style["color"]
                points = binned[
                    binned.fire.eq(fire) & binned.radius_ft.eq(radius)
                    & binned.outcome.eq(outcome)
                ]
                curve = smooth[
                    smooth.fire.eq(fire) & smooth.radius_ft.eq(radius)
                    & smooth.outcome.eq(outcome)
                ]
                ax.plot(
                    curve.damaged_neighbors, curve.F_star,
                    color=color, ls=style["line"], lw=1.6, zorder=2,
                )
                error = np.vstack([
                    points.F_star - points.F_star_lo,
                    points.F_star_hi - points.F_star,
                ])
                ax.errorbar(
                    points.damaged_neighbors, points.F_star,
                    yerr=np.clip(error, 0, None), fmt=style["marker"],
                    ms=3.2, mew=.7, color=color, ecolor=color,
                    elinewidth=.45, alpha=.82, capsize=1.3, lw=0,
                    mfc=color if style["filled"] else "white", zorder=3,
                )
            ax.set_yscale("log")
            ax.set_xlim(left=0)
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            log_grid(ax, axis="y")
            ax.text(.045, .94, letters[row * len(radii_ft) + column],
                    transform=ax.transAxes, fontweight="bold",
                    va="top", ha="left")
            if row == 0:
                ax.set_title(f"Radius = {radius:,} ft", pad=6)
            if row == len(FIRES) - 1:
                ax.set_xlabel("Damaged neighboring\nstructures", labelpad=4)
            if column == 0:
                ax.set_ylabel(
                    f"{fire.title()}\nRealized coupling, $F^*$", labelpad=5
                )
    handles = [
        Line2D([0], [0], color="#202020", lw=1.55, ls="-", marker="o",
               markersize=4.1, markerfacecolor="#202020", label="Destroyed"),
        Line2D([0], [0], color="#777777", lw=1.55, ls=(0, (3.2, 2)),
               marker="s", markersize=4.1, markerfacecolor="white",
               markeredgecolor="#777777", label="No damage"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(.5, -.02), handlelength=2,
               columnspacing=1.5, handletextpad=.45)
    fig.subplots_adjust(
        left=.075, right=.99, top=.93, bottom=.19, wspace=.08, hspace=.32
    )
    output_dir = Path(output_dir)
    source_dir = Path(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "ED_fig_neighbor_radius_sweep"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight",
                facecolor="white")
    pd.concat([
        binned.assign(series="empirical_bin"),
        smooth.assign(series="lowess"),
    ], ignore_index=True, sort=False).to_csv(
        source_dir / "ED_fig_neighbor_radius_sweep_source.csv", index=False
    )
    return fig

