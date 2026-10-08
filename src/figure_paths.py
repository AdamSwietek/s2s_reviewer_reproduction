"""Where a figure goes, decided per figure rather than per notebook.

A notebook usually produces figures for more than one destination, so the
tier is named at the point of saving. Keeping the roots here means moving a
whole tier later is one edit rather than one per notebook.

    from src.figure_paths import save_figure
    save_figure(fig, 'urban_depth_shared_loss', 'main', source=coverage)

``exploratory`` is deliberately outside the submission tiers: everything under
``main``, ``extended`` and ``supplementary`` should be cited somewhere, which
makes an inventory of those folders a useful check on its own.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import FIGURES_DIR


FIGURES = FIGURES_DIR
FIGURE_TIERS = {
    "main": FIGURES / "main",
    "extended": FIGURES / "extended_data",
    "supplementary": FIGURES / "supplementary",
    "exploratory": FIGURES / "exploratory",
}
SOURCE_DATA = FIGURES / "source_data"
FIGURE_DPI = 400


def figure_dir(tier: str) -> Path:
    """The directory for one tier, created on first use."""
    if tier not in FIGURE_TIERS:
        raise KeyError(
            f"Unknown figure tier {tier!r}; expected one of "
            f"{sorted(FIGURE_TIERS)}"
        )
    directory = FIGURE_TIERS[tier]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_figure(fig, stem: str, tier: str, *, source: pd.DataFrame | None = None,
                dpi: int = FIGURE_DPI, tight: bool = True, **savefig_kwargs) -> Path:
    """Write one figure as PDF and PNG into its tier, and its source data.

    ``source`` is the frame the figure is drawn from; submission tiers need it
    for Source Data, and writing it here keeps the pairing from drifting.
    Returns the stem path, so the caller can print or reuse it.
    """
    target = figure_dir(tier) / stem
    options = {"facecolor": "white", **savefig_kwargs}
    if tight:
        options.setdefault("bbox_inches", "tight")
    fig.savefig(target.with_suffix(".pdf"), **options)
    fig.savefig(target.with_suffix(".png"), dpi=dpi, **options)
    if source is not None:
        SOURCE_DATA.mkdir(parents=True, exist_ok=True)
        source.to_csv(SOURCE_DATA / f"{stem}.csv", index=False)
    return target
