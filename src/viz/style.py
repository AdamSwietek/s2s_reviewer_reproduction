"""Shared visual style for the publication reproduction notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt


FIRE_COLORS = {"EATON": "#C24D32", "PALISADES": "#2E5B82"}
FIRE_MARKERS = {"EATON": "o", "PALISADES": "s"}
# WUI palette shared with the statewide/regional community figure.
WUI_INFLUENCE_COLOR = "#D5A23E"
WUI_INTERMIX_COLOR = "#6F9B8A"
WUI_INTERFACE_COLOR = "#3B78A3"
WUI_EXTENSION_COLOR = "#B24C63"
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
