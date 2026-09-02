"""Shared visual style for the publication reproduction notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt


FIRE_COLORS = {"EATON": "#C24D32", "PALISADES": "#2E5B82"}
FIRE_MARKERS = {"EATON": "o", "PALISADES": "s"}
WUI_INTERFACE_COLOR = "#C51B7D"
WUI_INFLUENCE_COLOR = "#D95F02"
FIRE_PERIMETER_COLOR = "#222222"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": .7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def log_grid(ax, axis: str = "both") -> None:
    ax.minorticks_on()
    ax.grid(True, which="major", axis=axis, color="#D8D8D8", lw=.5, zorder=0)
    ax.grid(True, which="minor", axis=axis, color="#EEEEEE", lw=.3, zorder=0)
