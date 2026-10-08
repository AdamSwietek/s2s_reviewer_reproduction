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


def _full_interface(project_root: Path, crs, corridor, wui=None) -> gpd.GeoSeries:
    if wui is None:
        wui = gpd.read_file(
            Path(project_root) / "data" / "calfire_wui_la.gpkg"
        )
    wui = wui.to_crs(crs)
    # Show only the Interface boundary relevant to the screened corridor. A
    # small buffer preserves boundary strokes at the screen edge without
    # pulling unrelated WUI fragments into the overview extent.
    context = corridor.geometry.union_all().buffer(100)
    geometry = (wui.loc[wui.WUI_DESC.eq("Interface")]
                .geometry.boundary.union_all().intersection(context))
    return gpd.GeoSeries([geometry], crs=crs)


def _draw_buildings(ax, buildings, norm, interface, corridor,
                    bounds=None, labels=False, places_wgs84=None):
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
        places_wgs84 = PLACES_WGS84 if places_wgs84 is None else places_wgs84
        points = (gpd.GeoSeries([Point(xy) for xy in places_wgs84.values()],
                                index=list(places_wgs84), crs=4326)
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
                       output_dir: Path, *, zooms=None, places_wgs84=None,
                       overview_title: str = "Regional Interface-spanning SENs",
                       screen_legend_label: str = "1-km analysis corridor",
                       output_stem: str = "Fig5_regional_sen_extent"):
    """Draw a regional Interface-spanning SEN map, two zooms and size profile."""
    apply_style()
    buildings = result["buildings"]
    corridor = result["corridor"]
    summary = result["summary"].iloc[0]
    zooms = ZOOMS if zooms is None else zooms
    interface = _full_interface(
        project_root, buildings.crs, corridor, wui=result.get("wui")
    )
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
    zoom_xmin = min(bounds[0] for bounds in zooms.values())
    zoom_xmax = max(bounds[1] for bounds in zooms.values())
    zoom_ymin = min(bounds[2] for bounds in zooms.values())
    zoom_ymax = max(bounds[3] for bounds in zooms.values())
    xmin = min(bxmin, ixmin, cxmin, zoom_xmin)
    xmax = max(bxmax, ixmax, cxmax, zoom_xmax)
    ymin = min(bymin, iymin, cymin, zoom_ymin)
    ymax = max(bymax, iymax, cymax, zoom_ymax)
    xpad = .018 * (xmax - xmin)
    ypad = .035 * (ymax - ymin)
    overview_bounds = (xmin - xpad, xmax + xpad,
                       ymin - ypad, ymax + ypad)
    _draw_buildings(regional, buildings, norm, interface, corridor,
                    bounds=overview_bounds, labels=True,
                    places_wgs84=places_wgs84)
    regional.text(0, 1.025, "a", transform=regional.transAxes,
                  fontsize=9, fontweight="bold", va="bottom")
    regional.text(.027, 1.025, overview_title,
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
    for (name, bounds), color in zip(zooms.items(), zoom_colors):
        xmin, xmax, ymin, ymax = bounds
        regional.add_patch(Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
            edgecolor=color, linewidth=.8, zorder=10,
        ))

    zoom_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    for ax, panel, ((name, bounds), color) in zip(
        zoom_axes, ["b", "c"], zip(zooms.items(), zoom_colors)
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
               label=screen_legend_label),
    ], loc="upper left", bbox_to_anchor=(.002, .875), ncol=1,
       fontsize=5.6, labelspacing=.35, handlelength=1.4,
       frameon=True, facecolor="white", edgecolor="none", framealpha=.82)
    fig.subplots_adjust(left=.055, right=.985, top=.935, bottom=.07)

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight",
                facecolor="white")
    return fig


def plot_mountains_fire_details(
    buildings: gpd.GeoDataFrame, wui: gpd.GeoDataFrame,
    fire_perimeters: gpd.GeoDataFrame, defended_ids: set, fire_buildings_dir: Path,
    *, viewports: dict, window_lonlat: tuple, norm, cmap, bounds, ticks, ticklabels,
    overview_title: str = "a   Building connectivity across the Santa Monica Mountains",
    colorbar_label: str = "Connected-network size (buildings)",
    wui_color: str = "#3B78A3", defense_color: str = "#366458FF",
    defense_edge: str = "#FFFFFF", color_column: str = "component_size",
    missing_color: str = "#D9D9D6", missing_label: str | None = None,
):
    """Santa Monica Mountains overview above Palisades and Eaton details.

    Buildings are coloured by ``color_column`` (default ``component_size``,
    under whatever link rule produced it); missing values draw in
    ``missing_color`` and, with ``missing_label``, get a legend entry.
    Defended structures (DINS) and the 2025 perimeters overlay the fire
    panels. Mirrors the notebook 05 publication figure so any SEN
    assignment can be drawn the same way. Returns the figure and a summary.
    """
    cmap = plt.get_cmap(cmap).with_extremes(bad=missing_color)
    def in_view(xlim, ylim):
        (xmin, xmax), (ymin, ymax) = xlim, ylim
        candidates = buildings.cx[xmin:xmax, ymin:ymax]
        points = candidates.geometry.centroid
        inside = points.x.between(xmin, xmax) & points.y.between(ymin, ymax)
        return candidates.loc[inside], points.loc[inside]

    overview_window = gpd.GeoSeries([box(*window_lonlat)], crs="EPSG:4326").to_crs(buildings.crs)
    overview_geometry = overview_window.iloc[0]
    oxmin, oymin, oxmax, oymax = overview_window.total_bounds
    candidates = buildings.cx[oxmin:oxmax, oymin:oymax]
    candidate_points = candidates.geometry.centroid
    inside = candidate_points.intersects(overview_geometry)
    overview, overview_points = candidates.loc[inside], candidate_points.loc[inside]
    wui_boundary = wui.to_crs(buildings.crs).geometry.union_all().boundary

    fig = plt.figure(figsize=(7.2, 6.55))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.08, .82), hspace=.16, wspace=.09)
    ax = fig.add_subplot(grid[0, :])
    ax.scatter(overview_points.x, overview_points.y, c=overview[color_column], cmap=cmap,
               norm=norm, s=.20, linewidths=0, alpha=.90, rasterized=True, zorder=2,
               plotnonfinite=True)
    perimeter = gpd.GeoSeries([wui_boundary.intersection(overview_geometry)], crs=buildings.crs)
    perimeter.plot(ax=ax, color="white", linewidth=1.8, zorder=5)
    perimeter.plot(ax=ax, color=wui_color, linewidth=.7, zorder=6)
    for label, (xlim, ylim) in zip(("b", "c"), viewports.values()):
        (xmin, xmax), (ymin, ymax) = xlim, ylim
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                               edgecolor="#202020", linewidth=.75, zorder=8))
        ax.text(xmin + 110, ymax - 110, label, ha="left", va="top", fontsize=7,
                fontweight="bold", color="#202020", zorder=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=.8))
    ax.set(xlim=(oxmin, oxmax), ylim=(oymin, oymax))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(overview_title, loc="left", fontsize=8.5, fontweight="bold", pad=5)

    rows = []
    for label, column, (fire, (xlim, ylim)) in zip(("b", "c"), (0, 1), viewports.items()):
        (xmin, xmax), (ymin, ymax) = xlim, ylim
        ax = fig.add_subplot(grid[1, column])
        subset, points = in_view(xlim, ylim)
        ax.scatter(points.x, points.y, c=subset[color_column], cmap=cmap, norm=norm,
                   s=.62, linewidths=0, alpha=.91, rasterized=True, zorder=2,
                   plotnonfinite=True)
        fire_buildings = gpd.read_parquet(
            Path(fire_buildings_dir) / f"{fire}_buildings.parquet",
            columns=["BLD_ID", "geometry"]).to_crs(buildings.crs)
        fire_buildings["BLD_ID"] = fire_buildings.BLD_ID.astype(str)
        defended = (fire_buildings.loc[fire_buildings.BLD_ID.isin(defended_ids)]
                    .drop_duplicates("BLD_ID").geometry.representative_point())
        visible = defended.loc[defended.x.between(xmin, xmax) & defended.y.between(ymin, ymax)]
        detail_wui = gpd.GeoSeries([wui_boundary.intersection(box(xmin, ymin, xmax, ymax))],
                                   crs=buildings.crs)
        detail_wui.plot(ax=ax, color="white", linewidth=1.25, zorder=3)
        detail_wui.plot(ax=ax, color=wui_color, linewidth=.5, zorder=4)
        fire_perimeters.loc[fire_perimeters.FIRE_NAME.eq(fire)].to_crs(buildings.crs).boundary.plot(
            ax=ax, color="#303030", linewidth=.9, zorder=5)
        ax.scatter(visible.x, visible.y, s=7.5, marker="o", facecolors=defense_color,
                   edgecolors="white", linewidths=.25, alpha=.85, rasterized=True, zorder=6)
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0, 1.055, f"{label}   {fire.title()} fire", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.8, fontweight="bold")
        ax.text(0, 1.012, f"n = {len(visible):,} recorded defensive actions",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, color="#5F5F5B")
        scale_x, scale_y = xmin + .055 * (xmax - xmin), ymin + .055 * (ymax - ymin)
        ax.plot([scale_x, scale_x + 1000], [scale_y, scale_y], color="white",
                linewidth=2.2, solid_capstyle="butt", zorder=8)
        ax.plot([scale_x, scale_x + 1000], [scale_y, scale_y], color="#303030",
                linewidth=.8, solid_capstyle="butt", zorder=9)
        ax.text(scale_x + 500, scale_y + .018 * (ymax - ymin), "1 km", ha="center",
                va="bottom", fontsize=5.3, color="#303030", zorder=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.72, pad=.4))
        row = {"fire": fire, "buildings_in_view": len(subset),
               "SENs_in_view": subset.component_id.nunique(),
               "isolated_share": float(subset.component_size.eq(1).mean()),
               "median_SEN_size": float(subset.component_size.median()),
               "largest_SEN_in_view": int(subset.component_size.max()),
               "defended_in_view": len(visible)}
        if color_column != "component_size":
            row[f"median_{color_column}"] = float(subset[color_column].median())
        rows.append(row)

    fig.legend(handles=[
        Line2D([], [], color=wui_color, lw=1.05, label="Mapped WUI"),
        Line2D([], [], color="#303030", lw=.9, label="2025 fire perimeter"),
    ], title="Boundaries", loc="lower center", bbox_to_anchor=(.39, .078), ncol=2,
        frameon=False, fontsize=5.9, title_fontsize=6.1, columnspacing=1.25, handlelength=1.7)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor=defense_color,
               markeredgecolor=defense_edge, markeredgewidth=.25, markersize=4.2,
               alpha=.85, label="defensive action"),
    ], title="Observed response", loc="lower center", bbox_to_anchor=(.76, .078),
        frameon=False, fontsize=5.9, title_fontsize=6.1, handlelength=1.2)
    if missing_label is not None:
        fig.legend(handles=[
            Line2D([], [], marker="s", ls="", markerfacecolor=missing_color,
                   markeredgecolor="none", markersize=4.2, label=missing_label),
        ], loc="lower left", bbox_to_anchor=(.80, .018), frameon=False,
            fontsize=5.7, handlelength=1.0, handletextpad=.4)
    cbar_ax = fig.add_axes([.22, .036, .56, .014])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax,
                        orientation="horizontal", boundaries=bounds, ticks=ticks,
                        spacing="uniform")
    cbar.set_ticklabels(ticklabels)
    cbar.set_label(colorbar_label, fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.7, length=2)
    fig.subplots_adjust(left=.018, right=.988, top=.965, bottom=.135)
    return fig, pd.DataFrame(rows)
