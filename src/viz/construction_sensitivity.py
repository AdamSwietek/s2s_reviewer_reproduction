"""Extended Data graphics for construction-attribute sensitivity."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

from src.viz.style import apply_style, log_grid


COLORS = {
    "Tile roofing": "#2E5B82",
    "Lower vegetation": "#3D6B45",
    "Newer construction": "#A66F00",
}


def plot_exposure_dependent_margins(curves, summary, output_dir: Path):
    """Plot standardized reductions in destruction across exposure."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    for attribute in COLORS:
        data = curves[curves.attribute.eq(attribute)].sort_values("F_star")
        color = COLORS[attribute]
        ax.fill_between(data.F_star, data.ci_lo, data.ci_hi,
                        color=color, alpha=.13, linewidth=0)
        ax.plot(data.F_star, data.margin_pp, color=color, lw=2.0,
                label=attribute)
        peak = summary[summary.attribute.eq(attribute)].iloc[0]
        ax.plot(peak.peak_F_star, peak.peak_margin_pp, "o", ms=5,
                color=color, mfc="white", mew=1.1, zorder=5)
        ax.annotate(
            f"{peak.peak_margin_pp:.1f} pp",
            (peak.peak_F_star, peak.peak_margin_pp),
            xytext=(4, 5), textcoords="offset points", fontsize=7,
            color=color,
        )
    ax.axhline(0, color="#4A4A4A", lw=.7)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlabel(r"Realized geometric coupling, $F^*$")
    ax.set_ylabel("Predicted reduction in destruction\n(percentage points)")
    ax.set_title("Construction attributes provide their largest margin near the fragility transition",
                 loc="left", fontsize=10, fontweight="bold")
    ax.text(.995, .02,
            "Lines: standardized predictions; shading: 95% cluster-robust confidence bands",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#555555")
    log_grid(ax, axis="x")
    ax.grid(True, axis="y", color="#E4E4E4", lw=.45)
    ax.legend(frameon=False, loc="upper left", ncol=3,
              handlelength=1.8, columnspacing=1.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=.105, right=.99, top=.88, bottom=.16)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "ED_fig_construction_exposure_margins"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight",
                facecolor="white")
    return fig
