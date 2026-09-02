"""Extended Data graphics for SEN threshold and domain sensitivity."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.analysis.fragility import FIRES
from src.viz.style import (
    FIRE_COLORS, FIRE_MARKERS, WUI_INTERFACE_COLOR, apply_style,
)


INK = "#222222"
MID = "#777777"
MAGENTA = WUI_INTERFACE_COLOR


def _panel_title(ax, letter, title):
    ax.text(0, 1.06, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")
    ax.text(.06, 1.06, title, transform=ax.transAxes, fontsize=7.5,
            fontweight="bold", va="bottom")


def _threshold_line(ax):
    ax.axvline(.50, color="#8C8C89", lw=.75, ls=":", zorder=0)


def plot_sen_sensitivity(fire_sweep: pd.DataFrame,
                          calibration_comparison: pd.DataFrame,
                          regional_sweep: pd.DataFrame,
                          domain_sensitivity: pd.DataFrame,
                          output_dir: Path):
    """Draw six compact panels covering threshold and domain sensitivity."""
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.15))

    ax = axes[0, 0]
    for fire in FIRES:
        frame = fire_sweep[fire_sweep.fire.eq(fire)]
        ax.plot(frame.probability_equivalent, frame.largest_component,
                color=FIRE_COLORS[fire], marker=FIRE_MARKERS[fire],
                ms=3.5, lw=1.35, label=fire.title())
    ax.set_yscale("log"); _threshold_line(ax)
    ax.set_ylabel("Largest SEN (buildings)")
    ax.set_xlabel("Destruction-probability equivalent")
    ax.legend(fontsize=6.2, loc="upper right")
    _panel_title(ax, "a", "Fire-network fragmentation")

    ax = axes[0, 1]
    for fire in FIRES:
        frame = fire_sweep[fire_sweep.fire.eq(fire)]
        ax.plot(frame.probability_equivalent, frame.large_minus_isolated_pp,
                color=FIRE_COLORS[fire], marker=FIRE_MARKERS[fire],
                ms=3.5, lw=1.35, label=fire.title())
    ax.axhline(0, color=MID, lw=.65); _threshold_line(ax)
    ax.set_ylabel("Destruction difference (pp)")
    ax.set_xlabel("Destruction-probability equivalent")
    ax.text(.03, .04, "SENs ≥100 buildings minus isolated",
            transform=ax.transAxes, fontsize=5.6, color=MID)
    _panel_title(ax, "b", "Outcome contrast remains positive")

    ax = axes[0, 2]
    x = np.arange(len(FIRES)); width = .32
    for offset, (calibration, hatch) in enumerate([
        ("Common pooled", None), ("Fire-specific", "///")
    ]):
        frame = (calibration_comparison[
            calibration_comparison.calibration.eq(calibration)]
            .set_index("fire").loc[list(FIRES)])
        bars = ax.bar(x + (offset - .5) * width, frame.largest_component,
                      width=width, color=[FIRE_COLORS[f] for f in FIRES],
                      alpha=.9 if offset == 0 else .42, hatch=hatch,
                      edgecolor=[FIRE_COLORS[f] for f in FIRES], linewidth=.7,
                      label=calibration)
        for bar, value in zip(bars, frame.largest_component):
            ax.text(bar.get_x() + bar.get_width()/2, value * 1.06,
                    f"{int(value):,}", ha="center", va="bottom", fontsize=5.5)
    ax.set_yscale("log"); ax.set_xticks(x, [f.title() for f in FIRES])
    ax.set_ylabel("Largest SEN (buildings)")
    ax.legend(fontsize=5.6, loc="upper right")
    _panel_title(ax, "c", "Common versus fire-specific cutoff")

    ax = axes[1, 0]
    ax.plot(regional_sweep.probability_equivalent,
            regional_sweep.connected_building_share, color=MID,
            marker="o", ms=3.2, lw=1.15, label="any connected SEN")
    ax.plot(regional_sweep.probability_equivalent,
            regional_sweep.share_in_interface_spanning_SENs,
            color=MAGENTA, marker="s", ms=3.2, lw=1.45,
            label="Interface-spanning SEN")
    _threshold_line(ax); ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Share of regional buildings")
    ax.set_xlabel("Destruction-probability equivalent")
    ax.legend(fontsize=5.5, loc="lower left")
    _panel_title(ax, "d", "Regional connected share")

    ax = axes[1, 1]
    ax.plot(regional_sweep.probability_equivalent,
            regional_sweep.largest_SEN, color=INK, marker="o",
            ms=3.4, lw=1.5)
    ax.set_yscale("log"); _threshold_line(ax)
    ax.set_ylabel("Largest regional SEN (buildings)")
    ax.set_xlabel("Destruction-probability equivalent")
    _panel_title(ax, "e", "Regional component size")

    ax = axes[1, 2]
    domain = domain_sensitivity.copy()
    labels = ["0.50", "0.75", "1.00", "Production"]
    x = np.arange(len(domain))
    ax.plot(x, domain.share_in_interface_spanning_SENs,
            color=MAGENTA, marker="o", ms=3.6, lw=1.45)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(.45, .60); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Buildings in Interface-spanning SENs")
    ax.set_xlabel("Distance from Interface spine (km)")
    for position, row in enumerate(domain.itertuples(index=False)):
        ax.annotate(f"{int(row.buildings):,}",
                    (position, row.share_in_interface_spanning_SENs),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=5.3, color=INK)
    _panel_title(ax, "f", "Regional-domain sensitivity")

    for ax in axes.flat:
        ax.grid(True, which="major", color="#E2E2DF", lw=.45, zorder=0)
        ax.tick_params(labelsize=6.5)
    fig.subplots_adjust(left=.075, right=.985, bottom=.10, top=.94,
                        hspace=.42, wspace=.34)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "ED_fig_sen_sensitivity"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight",
                facecolor="white")
    return fig
