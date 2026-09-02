"""Regional SEN screening graphics."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import Point, box

from src.viz.style import WUI_INTERFACE_COLOR, apply_style


INK = "#222222"
MID = "#777777"
WUI = WUI_INTERFACE_COLOR
OTHER = "#D8DAD8"
ISOLATED = "#EFEFEB"
CMAP = "viridis"
ZOOMS = {
    "Santa Monica–Palisades": (359500, 366000, 3765800, 3772300),
    "Silver Lake–Echo Park": (380000, 387500, 3770000, 3777500),
}
PLACES_WGS84 = {
    "Santa Monica": (-118.490, 34.040),
    "Beverly Hills": (-118.400, 34.073),
    "Hollywood": (-118.328, 34.101),
    "Silver Lake": (-118.271, 34.087),
    "Echo Park": (-118.261, 34.078),
}


def _full_interface(project_root: Path, crs, corridor) -> gpd.GeoSeries:
    wui = (gpd.read_file(Path(project_root) / "data" / "calfire_wui_la.gpkg")
           .to_crs(crs))
    # Show only the Interface boundary relevant to the screened corridor. A
    # small buffer preserves boundary strokes at the screen edge without
    # pulling unrelated WUI fragments into the overview extent.
    context = corridor.geometry.union_all().buffer(100)
    geometry = (wui.loc[wui.WUI_DESC.eq("Interface")]
                .geometry.boundary.union_all().intersection(context))
    return gpd.GeoSeries([geometry], crs=crs)


def _draw_buildings(ax, buildings, norm, interface, corridor,
                    bounds=None, labels=False):
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        subset = buildings.cx[xmin:xmax, ymin:ymax]
    else:
        subset = buildings
        xmin, ymin, xmax, ymax = buildings.total_bounds
    isolated = subset.component_size.eq(1)
    spanning = subset.spans_interface_class
    subset.loc[isolated].plot(
        ax=ax, color=ISOLATED, linewidth=0, rasterized=True, zorder=1,
    )
    subset.loc[~isolated & ~spanning].plot(
        ax=ax, color=OTHER, linewidth=0, rasterized=True, zorder=2,
    )
    subset.loc[spanning].plot(
        ax=ax, column="component_size", cmap=CMAP, norm=norm,
        linewidth=0, rasterized=True, zorder=3,
    )
    interface.plot(
        ax=ax, color=WUI, linewidth=.85, zorder=6,
        path_effects=[pe.Stroke(linewidth=1.8, foreground="white"), pe.Normal()],
    )
    corridor.boundary.plot(
        ax=ax, color="#8B8B88", linewidth=.45, linestyle=(0, (2, 2)),
        alpha=.75, zorder=5,
    )
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if labels:
        points = (gpd.GeoSeries([Point(xy) for xy in PLACES_WGS84.values()],
                                index=list(PLACES_WGS84), crs=4326)
                  .to_crs(buildings.crs))
        for name, point in points.items():
            if xmin <= point.x <= xmax and ymin <= point.y <= ymax:
                ax.scatter(point.x, point.y, s=8, color=INK, edgecolor="white",
                           linewidth=.45, zorder=8)
                ax.annotate(
                    name, (point.x, point.y), xytext=(3, -3),
                    textcoords="offset points", fontsize=5.7, color=INK,
                    ha="left", va="top", zorder=9,
                    path_effects=[pe.withStroke(linewidth=1.8,
                                                foreground="white")],
                )


def plot_regional_sen(result: dict, size_profile: pd.DataFrame,
                       category_summary: pd.DataFrame, project_root: Path,
                       output_dir: Path):
    """Draw a regional Interface-spanning SEN map, two zooms and size profile."""
    apply_style()
    buildings = result["buildings"]
    corridor = result["corridor"]
    summary = result["summary"].iloc[0]
    interface = _full_interface(project_root, buildings.crs, corridor)
    maximum = int(buildings.loc[buildings.spans_interface_class,
                                "component_size"].max())
    norm = LogNorm(vmin=2, vmax=maximum)

    fig = plt.figure(figsize=(7.2, 6.25))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.32, .82],
                           hspace=.30, wspace=.22)
    regional = fig.add_subplot(grid[0, :])

    # The two detail windows extend slightly beyond the building envelope.
    # Include them in the overview extent before adding a small visual margin;
    # otherwise Matplotlib clips the outer sides of their rectangles.
    bxmin, bymin, bxmax, bymax = buildings.total_bounds
    ixmin, iymin, ixmax, iymax = interface.total_bounds
    cxmin, cymin, cxmax, cymax = corridor.total_bounds
    zoom_xmin = min(bounds[0] for bounds in ZOOMS.values())
    zoom_xmax = max(bounds[1] for bounds in ZOOMS.values())
    zoom_ymin = min(bounds[2] for bounds in ZOOMS.values())
    zoom_ymax = max(bounds[3] for bounds in ZOOMS.values())
    xmin = min(bxmin, ixmin, cxmin, zoom_xmin)
    xmax = max(bxmax, ixmax, cxmax, zoom_xmax)
    ymin = min(bymin, iymin, cymin, zoom_ymin)
    ymax = max(bymax, iymax, cymax, zoom_ymax)
    xpad = .018 * (xmax - xmin)
    ypad = .035 * (ymax - ymin)
    overview_bounds = (xmin - xpad, xmax + xpad,
                       ymin - ypad, ymax + ypad)
    _draw_buildings(regional, buildings, norm, interface, corridor,
                    bounds=overview_bounds, labels=True)
    regional.text(0, 1.025, "a", transform=regional.transAxes,
                  fontsize=9, fontweight="bold", va="bottom")
    regional.text(.027, 1.025, "Regional Interface-spanning SENs",
                  transform=regional.transAxes, fontsize=8.5,
                  fontweight="bold", va="bottom")
    regional.text(
        .995, 1.018,
        f"{int(summary.interface_spanning_SENs):,} Interface-spanning SENs · "
        f"{100 * summary.share_in_interface_spanning_SENs:.1f}% of buildings\n"
        f"largest observed: {int(summary.largest_interface_spanning_SEN):,} "
        "buildings (screen-edge censored)",
        transform=regional.transAxes, ha="right", va="bottom", fontsize=5.7,
        linespacing=1.12, color=INK, clip_on=False,
    )
    zoom_colors = ["#2E5B82", "#C24D32"]
    for (name, bounds), color in zip(ZOOMS.items(), zoom_colors):
        xmin, xmax, ymin, ymax = bounds
        regional.add_patch(Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
            edgecolor=color, linewidth=.8, zorder=10,
        ))

    zoom_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    for ax, panel, ((name, bounds), color) in zip(
        zoom_axes, ["b", "c"], zip(ZOOMS.items(), zoom_colors)
    ):
        _draw_buildings(ax, buildings, norm, interface, corridor, bounds=bounds)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color(color); spine.set_linewidth(.8)
        ax.text(0, 1.055, panel, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")
        ax.text(.09, 1.055, name, transform=ax.transAxes, fontsize=7.4,
                fontweight="bold", va="bottom")

    profile = fig.add_subplot(grid[1, 2])
    profile.plot(size_profile.minimum_SEN_size,
                 size_profile.building_share, color=INK, lw=1.7,
                 marker="o", ms=2.8)
    profile.set_xscale("log")
    profile.set_xlim(1, max(size_profile.minimum_SEN_size) * 1.15)
    profile.set_ylim(0, 1.03)
    profile.set_xlabel("Minimum SEN size (buildings)")
    profile.set_ylabel("")
    profile.text(.025, .965, "Share of buildings", transform=profile.transAxes,
                 ha="left", va="top", fontsize=6.4, color=INK)
    profile.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(1, decimals=0)
    )
    profile.grid(True, which="major", color="#E0E0DD", lw=.45)
    profile.grid(True, which="minor", axis="x", color="#EEEEEB", lw=.3)
    profile.text(0, 1.055, "d", transform=profile.transAxes, fontsize=9,
                 fontweight="bold", va="bottom")
    profile.text(.09, 1.055, "Regional SEN size distribution",
                 transform=profile.transAxes, fontsize=7.4,
                 fontweight="bold", va="bottom")
    for cutoff in [100, 1000]:
        row = size_profile.loc[size_profile.minimum_SEN_size.eq(cutoff)].iloc[0]
        profile.annotate(
            f"{100 * row.building_share:.0f}% in ≥{cutoff:,}",
            (cutoff, row.building_share), xytext=(4, -12 if cutoff == 100 else 7),
            textcoords="offset points", fontsize=5.8, color=INK,
            arrowprops=dict(arrowstyle="-", color=MID, lw=.45),
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, ax=regional, orientation="horizontal",
                        fraction=.028, pad=.035, aspect=48)
    cbar.set_label("Buildings in Interface-spanning SEN", fontsize=6.6)
    cbar.ax.tick_params(labelsize=6, length=2)
    regional.legend(handles=[
        Patch(facecolor=OTHER, edgecolor="none", label="other connected SEN"),
        Patch(facecolor=ISOLATED, edgecolor="none", label="isolated building"),
        Line2D([0], [0], color=WUI, lw=1.1, label="WUI Interface boundary"),
        Line2D([0], [0], color="#8B8B88", lw=.6, ls=(0, (2, 2)),
               label="1-km analysis corridor"),
    ], loc="upper left", bbox_to_anchor=(.002, .875), ncol=1,
       fontsize=5.6, labelspacing=.35, handlelength=1.4,
       frameon=True, facecolor="white", edgecolor="none", framealpha=.82)
    fig.subplots_adjust(left=.055, right=.985, top=.935, bottom=.07)

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Fig5_regional_sen_extent"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight",
                facecolor="white")
    return fig
