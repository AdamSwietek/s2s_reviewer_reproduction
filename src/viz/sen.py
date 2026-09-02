"""Publication graphics for Structure Exposure Networks (SENs)."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.analysis.fragility import FIRES
from src.viz.style import (
    FIRE_COLORS, FIRE_MARKERS, FIRE_PERIMETER_COLOR,
    WUI_INFLUENCE_COLOR, WUI_INTERFACE_COLOR, apply_style,
)


INK = FIRE_PERIMETER_COLOR
MID = "#747474"
OTHER = "#E8E8E4"
WUI_INTERFACE = WUI_INTERFACE_COLOR
WUI_INFLUENCE = WUI_INFLUENCE_COLOR
VIEWPORTS = {
    "EATON": ((392700, 398900), (3780900, 3785500)),
    # Match the Eaton viewport aspect so the two maps align as paired panels.
    "PALISADES": ((355460, 362340), (3766200, 3771300)),
}


def _plot_network_map(ax, state: dict, fire: str, wui_interface,
                      wui_influence, perimeter,
                      norm: LogNorm, cmap, panel: str) -> None:
    nodes = state["nodes"]
    singleton = nodes.component_size.eq(1)
    nodes.loc[singleton].plot(
        ax=ax, color=OTHER, linewidth=0, rasterized=True, zorder=1,
    )
    nodes.loc[~singleton].plot(
        ax=ax, column="component_size", cmap=cmap, norm=norm,
        linewidth=0, rasterized=True, zorder=2,
    )
    wui_influence.to_crs(nodes.crs).boundary.plot(
        ax=ax, color=WUI_INFLUENCE, linewidth=.75, zorder=4,
        path_effects=[pe.Stroke(linewidth=1.65, foreground="white"), pe.Normal()],
    )
    wui_interface.to_crs(nodes.crs).boundary.plot(
        ax=ax, color=WUI_INTERFACE, linewidth=.8, zorder=5,
        path_effects=[pe.Stroke(linewidth=1.8, foreground="white"), pe.Normal()],
    )
    boundary = gpd.GeoSeries(
        [perimeter], crs=wui_interface.crs).to_crs(nodes.crs).boundary
    boundary.plot(
        ax=ax, color=INK, linewidth=1.15, zorder=6,
        path_effects=[pe.Stroke(linewidth=2.25, foreground="white"), pe.Normal()],
    )
    ax.set_xlim(*VIEWPORTS[fire][0]); ax.set_ylim(*VIEWPORTS[fire][1])
    ax.set_aspect("equal"); ax.set_anchor("S")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    summary = state["summary"]
    ax.text(0, 1.075, panel, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", clip_on=False)
    ax.text(.037, 1.075, fire.title(), transform=ax.transAxes, fontsize=8.4,
            fontweight="bold", va="bottom", clip_on=False)
    ax.text(
        .037, 1.028,
        f"{summary['active_bonds']:,} bonds · {summary['components']:,} SENs · "
        f"largest: {summary['largest_component']:,} buildings",
        transform=ax.transAxes, fontsize=6.1, va="bottom", color="#444444",
        clip_on=False,
    )


def plot_sen_composite(states: dict[str, dict], shared_fate: pd.DataFrame,
                        size_count_summary: pd.DataFrame,
                        threshold: float, project_root: Path,
                        output_dir: Path, map_cmaps=None):
    """Draw the main SEN maps, shared-fate curve and outcome-size profile."""
    apply_style()
    project_root, output_dir = Path(project_root), Path(output_dir)
    if map_cmaps is None:
        map_cmaps = {fire: "viridis" for fire in FIRES}
    wui = gpd.read_file(project_root / "data" / "calfire_wui_la.gpkg")
    wui_interface = wui[wui.WUI_DESC.eq("Interface")]
    wui_influence = wui[wui.WUI_DESC.eq("Influence Zone")]
    perimeters = (gpd.read_parquet(project_root / "data" / "nx" /
                                   "fire_perims.parquet")
                  .set_index("FIRE_NAME"))

    # A single scale lets a color denote the same SEN size in both fires and
    # links the mapped components directly to the size axis in panel d.
    largest_component = max(
        states[fire]["summary"]["largest_component"] for fire in FIRES
    )
    size_scale_max = int(2 ** np.ceil(np.log2(largest_component)))
    size_norm = LogNorm(vmin=2, vmax=size_scale_max)
    size_cmap = map_cmaps[FIRES[0]]
    fig = plt.figure(figsize=(7.35, 6.85))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.20, .82],
                           hspace=.37, wspace=.17)
    map_axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    for ax, fire, panel in zip(map_axes, FIRES, ["a", "b"]):
        _plot_network_map(ax, states[fire], fire, wui_interface, wui_influence,
                          perimeters.loc[fire].geometry, size_norm,
                          size_cmap, panel)

    sm = plt.cm.ScalarMappable(norm=size_norm, cmap=size_cmap)
    cbar = fig.colorbar(sm, ax=map_axes, orientation="horizontal",
                        fraction=.035, pad=.02, aspect=55)
    color_ticks = 2 ** np.arange(1, int(np.log2(size_scale_max)) + 1, 2)
    if color_ticks[-1] != size_scale_max:
        color_ticks = np.append(color_ticks, size_scale_max)
    cbar.set_ticks(color_ticks)
    cbar.set_ticklabels([f"{value:g}" for value in color_ticks])
    cbar.set_label("Buildings per SEN", fontsize=6.6, labelpad=1.5)
    cbar.ax.tick_params(labelsize=6.0, length=2, pad=1.5)
    map_legend = [
        Patch(facecolor=OTHER, edgecolor="none", label="isolated building"),
        Line2D([0], [0], color=WUI_INTERFACE, lw=1.1,
               label="WUI Interface boundary"),
        Line2D([0], [0], color=WUI_INFLUENCE, lw=1.1,
               label="WUI Influence boundary"),
        Line2D([0], [0], color=INK, lw=1.2, label="fire perimeter"),
    ]
    fig.legend(handles=map_legend, loc="upper center", ncol=4,
               bbox_to_anchor=(.5, .552), fontsize=5.8,
               handlelength=1.35, columnspacing=1.1, frameon=False)

    fate_ax = fig.add_subplot(grid[1, 0])
    for fire in FIRES:
        frame = shared_fate[shared_fate.fire.eq(fire)]
        fate_ax.plot(frame.p_destroyed_equivalent, frame.observed_diversity,
                     color=FIRE_COLORS[fire], lw=1.35,
                     marker=FIRE_MARKERS[fire], ms=2.7, markevery=4,
                     label=fire.title())
    pooled = shared_fate[shared_fate.fire.eq("POOLED")]
    fate_ax.fill_between(
        pooled.p_destroyed_equivalent, pooled.null_lo, pooled.null_hi,
        color="#D5D5D2", alpha=.75, linewidth=0, label="within-fire shuffle, 95%",
    )
    fate_ax.plot(pooled.p_destroyed_equivalent, pooled.null_mean,
                 color=MID, lw=.9, ls="--")
    fate_ax.plot(pooled.p_destroyed_equivalent, pooled.observed_diversity,
                 color=INK, lw=2.0, label="Pooled")
    fate_ax.axvline(.5, color="#999999", lw=.75, ls=":")
    fate_ax.text(
        .515, .965,
        f"analysis threshold: $P(\\mathrm{{destroyed}})=0.50$\n"
        f"$F_{{ij}}={threshold:.4f}$",
        transform=fate_ax.get_xaxis_transform(), fontsize=6.15,
        color=INK, va="top", ha="left",
    )
    fate_ax.set_xlabel("Single-emitter $P(\\mathrm{destroyed})$ equivalent")
    fate_ax.set_ylabel("Within-SEN outcome diversity")
    fate_ax.set_xlim(.05, .90); fate_ax.set_ylim(0, 1.02)
    fate_ax.grid(color="#E5E5E2", lw=.45)
    fate_ax.text(0, 1.07, "c", transform=fate_ax.transAxes, fontsize=9,
                 fontweight="bold")
    fate_ax.text(.055, 1.07, "Outcome clustering within SENs",
                 transform=fate_ax.transAxes, fontsize=8.2,
                 fontweight="bold")
    fate_ax.legend(loc="lower right", fontsize=5.9, ncol=2,
                   columnspacing=.8, handlelength=1.6)

    size_ax = fig.add_subplot(grid[1, 1])
    for fire in FIRES:
        frame = size_count_summary[size_count_summary.fire.eq(fire)].sort_values(
            "size_bin_id"
        )
        color = FIRE_COLORS[fire]
        size_ax.fill_between(frame.size_position, frame.null_lo, frame.null_hi,
                             color=color, alpha=.09, linewidth=0)
        size_ax.plot(frame.size_position, frame.null_mean,
                     color=color, lw=.7, ls=":")
        yerr = np.vstack((frame.destroyed_share - frame.ci_lo,
                          frame.ci_hi - frame.destroyed_share))
        size_ax.errorbar(
            frame.size_position, frame.destroyed_share, yerr=yerr, color=color,
            marker=FIRE_MARKERS[fire], ms=3.8, mfc="white", mew=.9,
            lw=1.3, elinewidth=.65, capsize=1.7, label=fire.title(),
        )
    pooled_size = size_count_summary[
        size_count_summary.fire.eq("POOLED")
    ].sort_values("size_bin_id")
    yerr = np.vstack((pooled_size.destroyed_share - pooled_size.ci_lo,
                      pooled_size.ci_hi - pooled_size.destroyed_share))
    size_ax.errorbar(
        pooled_size.size_position, pooled_size.destroyed_share,
        yerr=yerr, color=INK, marker="o", ms=3.5, lw=1.9,
        elinewidth=.65, capsize=1.7, label="Pooled",
    )
    max_size = int(size_count_summary.bin_upper.max())
    size_ticks = 2 ** np.arange(0, int(np.log2(max_size)) + 1)
    size_ax.set_xscale("log", base=2)
    size_ax.set_xticks(size_ticks, [f"{value:g}" for value in size_ticks])
    size_ax.xaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
    size_ax.set_ylim(.35, .96)
    size_ax.set_xlim(.85, max_size * 1.12)
    # Repeat the map's component-size encoding as a narrow ribbon inside the
    # otherwise empty lower edge of panel d. The grey interval identifies the
    # isolated-building class; connected SENs use the common logarithmic map.
    ribbon_y0, ribbon_y1 = .35, .361
    connected_left = np.sqrt(2)
    ribbon_edges = np.geomspace(
        connected_left, size_scale_max * np.sqrt(2), 257
    )
    ribbon_values = np.sqrt(ribbon_edges[:-1] * ribbon_edges[1:])
    size_ax.fill_between(
        [.85, connected_left], ribbon_y0, ribbon_y1,
        color=OTHER, linewidth=0, zorder=.2,
    )
    size_ax.pcolormesh(
        ribbon_edges, [ribbon_y0, ribbon_y1], ribbon_values[np.newaxis, :],
        cmap=size_cmap, norm=size_norm, shading="flat", rasterized=True,
        zorder=.2,
    )
    size_ax.set_xlabel("Buildings per SEN")
    size_ax.set_ylabel("Buildings destroyed")
    size_ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    size_ax.grid(axis="y", color="#E5E5E2", lw=.45)
    size_ax.text(0, 1.07, "d", transform=size_ax.transAxes, fontsize=9,
                 fontweight="bold")
    size_ax.text(.055, 1.07, "Destruction is elevated in larger SENs",
                 transform=size_ax.transAxes, fontsize=8.2,
                 fontweight="bold")
    size_ax.legend(loc="lower right", fontsize=6.1, ncol=3,
                   handlelength=1.4, columnspacing=.8)

    fig.subplots_adjust(left=.075, right=.985, top=.925, bottom=.075)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Fig4_structure_exposure_networks"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight",
                facecolor="white")
    return fig
