"""Visualize four- versus five-parameter fragility sensitivity."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.fragility import FIRES, binned_proportions, logistic_5pl
from ..analysis.fragility_sensitivity import logistic_4pl
from .style import FIRE_COLORS, apply_style, log_grid


def plot_4pl_5pl_comparison(data: pd.DataFrame, full: pd.DataFrame,
                            cv: pd.DataFrame, figure_dir: Path,
                            source_dir: Path) -> Path:
    apply_style()
    figure_dir, source_dir = Path(figure_dir), Path(source_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    exposed = data[data.exposed.eq(1)]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65),
                             gridspec_kw={"width_ratios": [1, 1, .82]})
    source_rows = []
    for ax, fire, letter in zip(axes[:2], FIRES, ("a", "b")):
        frame = exposed[exposed.fire.eq(fire)]
        bins = binned_proportions(
            frame.F_destroyed_wmean, frame.is_destroyed, n_bins=12
        )
        ax.errorbar(
            bins.x, bins.probability,
            yerr=np.vstack([bins.probability - bins.ci_lo,
                            bins.ci_hi - bins.probability]),
            fmt="o", ms=3.2, color=FIRE_COLORS[fire], mfc="white",
            mew=.8, elinewidth=.65, capsize=1.4, zorder=4,
        )
        grid = np.geomspace(max(1e-4, frame.F_destroyed_wmean.min()),
                           frame.F_destroyed_wmean.max(), 500)
        for model, ls, lw in (("4PL", "--", 1.25), ("5PL", "-", 1.8)):
            row = full[(full.fire.eq(fire)) & (full.model.eq(model))].iloc[0]
            if model == "4PL":
                probability = logistic_4pl(
                    grid, row.p_min, row.p_max, row.k, row.hill
                )
            else:
                probability = logistic_5pl(
                    grid, row.p_min, row.p_max, row.k, row.hill, row.asymmetry
                )
            ax.plot(grid, probability, color=FIRE_COLORS[fire], ls=ls, lw=lw,
                    label=model)
            source_rows.extend({
                "fire": fire, "model": model, "F_star": xx,
                "probability": pp,
            } for xx, pp in zip(grid, probability))
        row4 = full[(full.fire.eq(fire)) & full.model.eq("4PL")].iloc[0]
        row5 = full[(full.fire.eq(fire)) & full.model.eq("5PL")].iloc[0]
        ax.text(.04, .96,
                f"$F_{{50}}$: {row4.f50:.3f} (4PL)\n"
                f"{row5.f50:.3f} (5PL)",
                transform=ax.transAxes, va="top", fontsize=6.8)
        ax.set_xscale("log"); ax.set_xlim(left=1e-4); ax.set_ylim(-.02, 1.02)
        ax.set_xlabel("Realized geometric coupling, $F^*$")
        ax.set_title(f"{letter}  {fire.title()}", loc="left", fontweight="bold")
        log_grid(ax)
    axes[0].set_ylabel("Buildings destroyed")
    axes[1].legend(loc="lower right")

    # c: held-out differences; negative favors the 5PL.
    pivot = cv.pivot(index="fire", columns="model",
                     values="spatial_cv_log_loss")
    difference = pivot["5PL"] - pivot["4PL"]
    colors = [FIRE_COLORS[fire] for fire in difference.index]
    axes[2].bar(difference.index.str.title(), difference, color=colors, width=.58)
    axes[2].axhline(0, color="#333333", lw=.7)
    for index, value in enumerate(difference):
        axes[2].text(index, value + (-.0003 if value < 0 else .0003),
                     f"{value:+.4f}", ha="center",
                     va="top" if value < 0 else "bottom", fontsize=7)
    margin = max(.002, float(np.abs(difference).max()) * 1.55)
    axes[2].set_ylim(-margin, margin)
    axes[2].set_ylabel("5PL − 4PL spatial-CV log loss")
    axes[2].set_title("c  Out-of-sample fit", loc="left", fontweight="bold")
    axes[2].text(.5, .04, "negative favors 5PL", transform=axes[2].transAxes,
                 ha="center", fontsize=6.7, color="#555555")
    axes[2].grid(axis="y", color="#E1E1E1", lw=.45)

    fig.subplots_adjust(left=.075, right=.99, bottom=.21, top=.9, wspace=.35)
    stem = figure_dir / "ED_fig_fragility_model_form"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(source_rows).to_csv(
        source_dir / "ED_fig_fragility_model_form_source.csv", index=False
    )
    return stem.with_suffix(".png")
