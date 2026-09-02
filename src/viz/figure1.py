"""Publication panels for geometric coupling and structural fragility."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, LogFormatterMathtext

from src.analysis.fragility import (
    FIRES,
    binned_proportions,
    logistic_5pl,
)
from src.viz.style import FIRE_COLORS, FIRE_MARKERS, log_grid


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.png", dpi=400, bbox_inches="tight",
                facecolor="white")


def plot_fragility(data: pd.DataFrame, fits: dict, output_dir: Path,
                   source_dir: Path):
    """Plot fire-specific destruction fragility curves and empirical bins."""
    exposed = data[data.exposed.eq(1)]
    x_min = 1e-4
    x_max = exposed.F_destroyed_wmean.max()
    # Sized for a half-width slot in the final composite so labels remain
    # legible after assembly rather than being scaled down from a wide panel.
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    source_rows = []
    for fire in FIRES:
        subset = exposed[exposed.fire.eq(fire)]
        fit = fits[fire]
        grid = np.geomspace(x_min, x_max, 400)
        probability = logistic_5pl(grid, *fit["parameters"])
        ax.plot(grid, probability, color=FIRE_COLORS[fire], lw=2.2,
                label=(f"{fire.title()}  ($F_{{50}}$={fit['f50']:.3f}, "
                       f"95% CI {fit['f50_lo']:.3f}–{fit['f50_hi']:.3f})"))
        bins = binned_proportions(
            subset.F_destroyed_wmean, subset.is_destroyed, n_bins=10
        )
        ax.errorbar(
            bins.x, bins.probability,
            yerr=[bins.probability - bins.ci_lo,
                  bins.ci_hi - bins.probability],
            fmt=FIRE_MARKERS[fire], ms=5.1, mfc="white",
            mec=FIRE_COLORS[fire], mew=1.05, color=FIRE_COLORS[fire],
            ecolor=FIRE_COLORS[fire], elinewidth=.7, capsize=1.6, lw=0,
            zorder=4,
        )
        ax.axvline(fit["f50"], color=FIRE_COLORS[fire],
                   ls=(0, (1, 1.7)), lw=1.0)
        source_rows.extend({
            "fire": fire, "series": "fitted_curve", "F_star": x,
            "probability": p, "ci_lo": np.nan, "ci_hi": np.nan,
            "n": np.nan,
        } for x, p in zip(grid, probability))
        source_rows.extend({
            "fire": fire, "series": "empirical_bin", "F_star": row.x,
            "probability": row.probability, "ci_lo": row.ci_lo,
            "ci_hi": row.ci_hi, "n": row.n,
        } for row in bins.itertuples(index=False))

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max * 1.1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    log_grid(ax, axis="x")
    ax.grid(True, axis="y", color="#E8E8E8", lw=.45)
    ax.set_xlabel(r"Realized geometric coupling, $F^*$")
    ax.set_ylabel("Probability of destruction")
    # Half-width panels are reduced to about 79% during assembly.  Explicit
    # sizing keeps their final panel headings aligned with panels a-e.
    ax.set_title("g  Structural fragility", loc="left", fontweight="bold",
                 fontsize=10.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, output_dir, "fig1c_fragility")
    pd.DataFrame(source_rows).to_csv(
        Path(source_dir) / "fig1c_fragility_source.csv", index=False
    )
    return fig


def plot_partial_damage(data: pd.DataFrame, destroyed_fits: dict,
                        any_damage_fits: dict, peak_summary: pd.DataFrame,
                        output_dir: Path, source_dir: Path):
    """Plot fitted and empirical partial-damage probability by exposure."""
    exposed = data[data.exposed.eq(1)]
    x_min = 1e-4
    x_max = exposed.F_destroyed_wmean.max()
    # Match the half-width structural-fragility panel in the composite.
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    source_rows = []
    for fire in FIRES:
        subset = exposed[exposed.fire.eq(fire)]
        grid = np.geomspace(x_min, x_max, 400)
        probability = (
            logistic_5pl(grid, *any_damage_fits[fire]["parameters"])
            - logistic_5pl(grid, *destroyed_fits[fire]["parameters"])
        )
        peak = peak_summary[peak_summary.fire.eq(fire)].iloc[0]
        ax.plot(grid, probability, color=FIRE_COLORS[fire], lw=2.2,
                label=(f"{fire.title()}  (peak "
                       f"{100*peak.peak_partial_probability:.1f}% at "
                       f"$F^*$={peak.peak_F_star:.3f})"))
        ax.axvline(peak.peak_F_star, color=FIRE_COLORS[fire],
                   ls=(0, (1, 1.7)), lw=1.0)
        empirical = binned_proportions(
            subset.F_destroyed_wmean, subset.outcome.eq("partial"), n_bins=10
        )
        ax.errorbar(
            empirical.x, empirical.probability,
            yerr=[empirical.probability - empirical.ci_lo,
                  empirical.ci_hi - empirical.probability],
            fmt=FIRE_MARKERS[fire], ms=5.0, mfc="white",
            mec=FIRE_COLORS[fire], mew=1.05, color=FIRE_COLORS[fire],
            ecolor=FIRE_COLORS[fire], elinewidth=.7, capsize=1.6, lw=0,
        )
        source_rows.extend({
            "fire": fire, "series": "implied_curve", "F_star": x,
            "probability": p, "ci_lo": np.nan, "ci_hi": np.nan,
            "n": np.nan,
        } for x, p in zip(grid, probability))
        source_rows.extend({
            "fire": fire, "series": "empirical_bin", "F_star": row.x,
            "probability": row.probability, "ci_lo": row.ci_lo,
            "ci_hi": row.ci_hi, "n": row.n,
        } for row in empirical.itertuples(index=False))

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max * 1.1)
    ax.set_ylim(0, .30)
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    log_grid(ax, axis="x")
    ax.grid(True, axis="y", color="#E8E8E8", lw=.45)
    ax.set_xlabel(r"Realized geometric coupling, $F^*$")
    ax.set_ylabel("Probability of partial damage")
    ax.set_title("h  Damage without total destruction", loc="left",
                 fontweight="bold", fontsize=10.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, output_dir, "fig1d_partial_damage")
    pd.DataFrame(source_rows).to_csv(
        Path(source_dir) / "fig1d_partial_damage_source.csv", index=False
    )
    return fig


def plot_distance_comparison(summaries: dict, output_dir: Path,
                             source_dir: Path):
    """Plot realized exposure against CCD and surface-to-surface distance."""
    settings = {
        "ccd_ft": {
            "title": "Centroid-to-centroid distance",
            "label": r"Nearest destroyed structure, $d_{CCD}$ (ft)",
            "ticks": [30, 50, 100, 300, 500, 800],
            "limits": (26, 820),
        },
        "ssd_ft": {
            "title": "Surface-to-surface distance",
            "label": r"Nearest visible destroyed surface, $d_{SSD}$ (ft)",
            "ticks": [5, 10, 25, 50, 100, 250, 500, 800],
            "limits": (5, 820),
        },
    }
    styles = {
        0: {"label": "Survived", "marker": "o", "line": (0, (3.2, 2)),
            "color": "#777777", "filled": False},
        1: {"label": "Destroyed", "marker": "s", "line": "-",
            "color": "#202020", "filled": True},
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharey=True)
    source_rows = []
    for panel, (ax, distance_col) in enumerate(zip(axes, settings)):
        binned, smooth, ratio = summaries[distance_col]
        for destroyed in (0, 1):
            style = styles[destroyed]
            points = binned[binned.is_destroyed.eq(destroyed)]
            curve = smooth[smooth.is_destroyed.eq(destroyed)]
            ax.plot(curve.distance_ft, curve.F_star, color=style["color"],
                    ls=style["line"], lw=1.65)
            ax.errorbar(
                points.distance_ft, points.F_star,
                yerr=[points.F_star - points.F_star_lo,
                      points.F_star_hi - points.F_star],
                fmt=style["marker"], ms=3.2, mew=.7,
                color=style["color"], ecolor=style["color"],
                elinewidth=.45, capsize=1.3, lw=0,
                mfc=style["color"] if style["filled"] else "white",
            )
            source_rows.extend({
                "distance_measure": distance_col,
                "is_destroyed": destroyed,
                "series": "empirical_bin",
                **row._asdict(),
            } for row in points.itertuples(index=False))
            source_rows.extend({
                "distance_measure": distance_col,
                "is_destroyed": destroyed,
                "series": "lowess",
                "distance_ft": row.distance_ft,
                "F_star": row.F_star,
            } for row in curve.itertuples(index=False))
        config = settings[distance_col]
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*config["limits"]); ax.set_ylim(1e-4, .6)
        ax.set_xticks(config["ticks"])
        ax.xaxis.set_major_formatter(FixedFormatter(
            [str(value) for value in config["ticks"]]
        ))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        log_grid(ax)
        ax.set_xlabel(config["label"])
        ax.set_title(f"{'ef'[panel]}  {config['title']}", loc="left",
                     fontweight="bold", fontsize=8.2)
        ax.text(.97, .95, f"{ratio:.2f}× outcome gap",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.2)
    axes[0].set_ylabel(r"Realized geometric coupling, $F^*$")
    handles = [Line2D(
        [0], [0], color=styles[value]["color"],
        ls=styles[value]["line"], marker=styles[value]["marker"],
        markersize=4.2,
        markerfacecolor=(styles[value]["color"] if styles[value]["filled"]
                          else "white"),
        label=styles[value]["label"],
    ) for value in (0, 1)]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(.5, -.005))
    fig.subplots_adjust(left=.10, right=.99, top=.91, bottom=.23, wspace=.08)
    _save(fig, output_dir, "fig1b_distance_comparison")
    pd.DataFrame(source_rows).to_csv(
        Path(source_dir) / "fig1b_distance_comparison_source.csv", index=False
    )
    return fig
