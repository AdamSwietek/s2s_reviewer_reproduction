"""Publication graphics for Structure Exposure Networks (SENs)."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.analysis.fragility import FIRES
from src.viz.style import (
    FIRE_COLORS, FIRE_MARKERS, FIRE_PERIMETER_COLOR,
    WUI_INFLUENCE_COLOR, WUI_INTERMIX_COLOR, WUI_INTERFACE_COLOR,
    apply_style,
)


INK = FIRE_PERIMETER_COLOR
MID = "#747474"
OTHER = "#E8E8E4"
WUI_INTERFACE = WUI_INTERFACE_COLOR
WUI_INTERMIX = WUI_INTERMIX_COLOR
WUI_INFLUENCE = WUI_INFLUENCE_COLOR
VIEWPORTS = {
    "EATON": ((392700, 398900), (3780900, 3785500)),
    # Match the Eaton viewport aspect so the two maps align as paired panels.
    "PALISADES": ((355460, 362340), (3766200, 3771300)),
}


def _plot_network_map(ax, state: dict, fire: str, wui_interface,
                      wui_intermix, wui_influence, perimeter,
                      norm: LogNorm, cmap, panel: str) -> None:
    nodes = state["nodes"]
    nodes = nodes.set_geometry(nodes.geometry.centroid)
    unassessed = ~nodes.assessed
    nodes.loc[unassessed].plot(
        ax=ax, column="component_size", cmap="Greys", norm=norm,
        markersize = 1,
        linewidth=0, rasterized=True, zorder=1,
    )
    nodes.loc[~unassessed].plot(
        ax=ax, column="component_size", cmap=cmap, norm=norm,
        markersize = .5,
        linewidth=0, rasterized=True, zorder=2,
    )
    perimeter_map = gpd.GeoSeries(
        [perimeter], crs=nodes.crs,
    )
    # Intersect the source boundary lines with the fire perimeter. Clipping
    # polygons first would manufacture colored linework along the perimeter.
    for layer, color, zorder in (
        (wui_influence, WUI_INFLUENCE, 4),
        (wui_intermix, WUI_INTERMIX, 5),
        (wui_interface, WUI_INTERFACE, 6),
    ):
        boundary = layer.to_crs(nodes.crs).boundary.intersection(perimeter)
        boundary = boundary.loc[~boundary.is_empty]
        if len(boundary):
            boundary.plot(
                ax=ax, color=color, linewidth=.48, alpha=.72, zorder=zorder,
            )
    boundary = perimeter_map.boundary
    boundary.plot(
        ax=ax, color=INK, linewidth=.85, zorder=7,
    )
    # The graph can contain useful connectors outside the perimeter, but the
    # fire panels use the perimeter as their geographic frame.
    xmin, ymin, xmax, ymax = perimeter.bounds
    pad = max(xmax - xmin, ymax - ymin) * .035
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box"); ax.set_anchor("N")
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
                        output_dir: Path, map_cmaps=None,
                        threshold_expression: str | None = None,
                        fate_xlabel: str | None = None,
                        output_stem: str = "Fig4_structure_exposure_networks",
                        wui: gpd.GeoDataFrame | None = None):
    """Draw the main SEN maps, shared-fate curve and outcome-size profile."""
    apply_style()
    project_root, output_dir = Path(project_root), Path(output_dir)
    if map_cmaps is None:
        map_cmaps = {fire: "viridis" for fire in FIRES}
    if wui is None:
        wui = gpd.read_file(project_root / "data" / "calfire_wui_la.gpkg")
    wui_interface = wui[wui.WUI_DESC.eq("Interface")]
    wui_intermix = wui[wui.WUI_DESC.eq("Intermix")]
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
    size_norm = LogNorm(vmin=1, vmax=size_scale_max)
    size_cmap = map_cmaps[FIRES[0]]
    fig = plt.figure(figsize=(7.35, 6.85))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.20, .82],
                           hspace=.37, wspace=.17)
    map_axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    for ax, fire, panel in zip(map_axes, FIRES, ["a", "b"]):
        _plot_network_map(ax, states[fire], fire, wui_interface, wui_intermix,
                          wui_influence, perimeters.loc[fire].geometry, size_norm,
                          size_cmap, panel)

    sm = plt.cm.ScalarMappable(norm=size_norm, cmap=size_cmap)
    cbar = fig.colorbar(sm, ax=map_axes, orientation="horizontal",
                        fraction=.035, pad=.02, aspect=55)
    color_ticks = 2 ** np.arange(0, int(np.log2(size_scale_max)) + 1, 2)
    if color_ticks[-1] != size_scale_max:
        color_ticks = np.append(color_ticks, size_scale_max)
    cbar.set_ticks(color_ticks)
    cbar.set_ticklabels([f"{value:g}" for value in color_ticks])
    cbar.set_label("Buildings per SEN", fontsize=6.6, labelpad=1.5)
    cbar.ax.tick_params(labelsize=6.0, length=2, pad=1.5)
    map_legend = [
        Patch(facecolor=plt.cm.viridis(.48), edgecolor="none",
              label="DINS-assessed building"),
        Patch(facecolor=plt.cm.Greys(.48), edgecolor="none",
              label="unassessed connector"),
        Line2D([0], [0], color=WUI_INFLUENCE, lw=1.1,
               label="Influence"),
        Line2D([0], [0], color=WUI_INTERMIX, lw=1.1,
               label="Intermix"),
        Line2D([0], [0], color=WUI_INTERFACE, lw=1.1,
               label="Interface"),
        Line2D([0], [0], color=INK, lw=1.2, label="fire perimeter"),
    ]
    map_legend_artist = fig.legend(
        handles=map_legend, loc="upper center", ncol=6,
        bbox_to_anchor=(.5, .565), fontsize=5.55,
        handlelength=1.25, columnspacing=.9, frameon=False,
    )

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
    if threshold_expression is None:
        threshold_expression = f"$F_{{ij}}={threshold:.4f}$"
    fate_ax.text(
        .515, .965,
        f"analysis threshold: $P(\\mathrm{{destroyed}})=0.50$\n"
        f"{threshold_expression}",
        transform=fate_ax.get_xaxis_transform(), fontsize=6.15,
        color=INK, va="top", ha="left",
    )
    if fate_xlabel is None:
        fate_xlabel = "Single-emitter $P(\\mathrm{destroyed})$ equivalent"
    fate_ax.set_xlabel(fate_xlabel)
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
    # Repeat the assessed-building component-size encoding as a narrow ribbon
    # inside the otherwise empty lower edge of panel d.
    ribbon_y0, ribbon_y1 = .35, .361
    ribbon_edges = np.geomspace(
        1 / np.sqrt(2), size_scale_max * np.sqrt(2), 257
    )
    ribbon_values = np.sqrt(ribbon_edges[:-1] * ribbon_edges[1:])
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
    # Reserve explicit bands for maps, legend and color scale. Perimeter-shaped
    # map axes otherwise expand downward and collide with the shared legend.
    for ax, x0 in zip(map_axes, (.075, .535)):
        ax.set_position([x0, .605, .405, .285])
    map_legend_artist.set_bbox_to_anchor((.5, .575), transform=fig.transFigure)
    cbar.ax.set_position([.16, .525, .68, .014])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight",
                facecolor="white")
    return fig


def plot_pooled_top_n_comparison(shared_fate: pd.DataFrame,
                                 size_summary: pd.DataFrame,
                                 output_dir: Path):
    """Compare pooled panel-c and panel-d results for cumulative top-N SENs."""
    apply_style()
    output_dir = Path(output_dir)
    neighbor_counts = sorted(shared_fate.n_neighbors.unique())
    colors = plt.get_cmap("viridis")(
        np.linspace(.08, .88, len(neighbor_counts))
    )
    markers = ["o", "s", "^", "D"]
    fig, (fate_ax, size_ax) = plt.subplots(
        1, 2, figsize=(7.35, 3.35), gridspec_kw={"wspace": .28}
    )

    for color, marker, n_neighbors in zip(colors, markers, neighbor_counts):
        fate = shared_fate[
            shared_fate.fire.eq("POOLED")
            & shared_fate.n_neighbors.eq(n_neighbors)
        ].sort_values("p_destroyed_equivalent")
        fate_ax.plot(
            fate.p_destroyed_equivalent, fate.observed_diversity,
            color=color, lw=1.8, marker=marker, ms=3.0, markevery=4,
            label=f"N = {n_neighbors}",
        )
        fate_ax.plot(
            fate.p_destroyed_equivalent, fate.null_mean,
            color=color, lw=.85, ls="--", alpha=.65,
        )

        size = size_summary[
            size_summary.fire.eq("POOLED")
            & size_summary.n_neighbors.eq(n_neighbors)
        ].sort_values("size_bin_id")
        yerr = np.vstack((size.destroyed_share - size.ci_lo,
                          size.ci_hi - size.destroyed_share))
        size_ax.errorbar(
            size.size_position, size.destroyed_share, yerr=yerr,
            color=color, marker=marker, ms=3.5, lw=1.55,
            elinewidth=.55, capsize=1.4, label=f"N = {n_neighbors}",
        )

    fate_ax.axvline(.5, color="#999999", lw=.75, ls=":")
    fate_ax.set(
        xlabel="Cumulative $F_{score}$ threshold, "
               "$P(\\mathrm{destroyed})$ equivalent",
        ylabel="Within-SEN outcome diversity", xlim=(.05, .90), ylim=(0, 1.02),
    )
    fate_ax.grid(color="#E5E5E2", lw=.45)
    fate_ax.text(0, 1.07, "c", transform=fate_ax.transAxes,
                 fontsize=9, fontweight="bold")
    fate_ax.text(.055, 1.07, "Pooled outcome clustering by top-$N$ rule",
                 transform=fate_ax.transAxes, fontsize=8.2,
                 fontweight="bold")
    n_legend = fate_ax.legend(loc="lower left", fontsize=6.2, ncol=2,
                              columnspacing=.8, handlelength=1.6)
    fate_ax.add_artist(n_legend)
    fate_ax.legend(handles=[
        Line2D([0], [0], color=INK, lw=1.7, label="observed"),
        Line2D([0], [0], color=INK, lw=.85, ls="--",
               label="within-fire shuffle"),
    ], loc="lower right", fontsize=5.9, frameon=False)

    pooled = size_summary[size_summary.fire.eq("POOLED")]
    baseline = pooled.groupby("n_neighbors").apply(
        lambda frame: frame.destroyed.sum() / frame.assessed.sum(),
        include_groups=False,
    ).mean()
    size_ax.axhline(baseline, color=MID, lw=.8, ls=":")
    max_size = int(pooled.bin_upper.max())
    size_ticks = 2 ** np.arange(0, int(np.log2(max_size)) + 1, 2)
    size_ax.set_xscale("log", base=2)
    size_ax.set_xticks(size_ticks, [f"{value:g}" for value in size_ticks])
    size_ax.xaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
    y_min = max(0, pooled.ci_lo.min() - .04)
    y_max = min(1, pooled.ci_hi.max() + .04)
    size_ax.set(
        xlabel="Buildings per SEN", ylabel="Buildings destroyed",
        xlim=(.85, max_size * 1.12), ylim=(y_min, y_max),
    )
    size_ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(1)
    )
    size_ax.grid(axis="y", color="#E5E5E2", lw=.45)
    size_ax.text(0, 1.07, "d", transform=size_ax.transAxes,
                 fontsize=9, fontweight="bold")
    size_ax.text(.055, 1.07, "Pooled destruction by top-$N$ SEN size",
                 transform=size_ax.transAxes, fontsize=8.2,
                 fontweight="bold")
    size_ax.legend(loc="lower right", fontsize=6.2, ncol=2,
                   columnspacing=.8, handlelength=1.5)

    fig.subplots_adjust(left=.09, right=.985, top=.88, bottom=.19)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Fig4_pooled_cumulative_topN_comparison"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight",
                facecolor="white")
    return fig
