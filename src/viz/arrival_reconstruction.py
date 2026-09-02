"""Extended Data figure for the rebuilt per-building arrival estimates.

Follows the arrival-estimation ladder: observed fire fronts, the boundary
interpolation they imply, and the timeline-updated kriged field that becomes
``T_arrival_hrs``. Eaton (top row) carries all three stages; Palisades (bottom
row) has no timeline update, so its interpolation is the final clock.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from .style import apply_style

METRIC_EPSG = 32611
# Eaton ignition (first dispatch); VIIRS acquisition times are expressed
# relative to it, matching the hours-since-ignition stamps on the isochrones.
EATON_IGNITION = pd.Timestamp("2025-01-07 18:11:00")
EATON_BOUNDS = {"lat_min": 34.10, "lat_max": 34.30,
                "lon_min": -118.25, "lon_max": -117.95}
CONTEXT_FACE = "#F1F1F1"
CONTEXT_EDGE = "#CCCCCC"
CONTROL_COLOR = "#111111"


def _rank_scale(reference: pd.Series):
    """Map hours since ignition to arrival-order percentile within a fire.

    Arrival times are heavily skewed, so an hours colour scale compresses most
    structures into a narrow band. Ranking against the fire's own arrival
    distribution spreads the progression evenly across the ramp; fronts,
    detections and structures all pass through the same mapping, so they stay
    comparable within a panel.
    """
    ordered = np.sort(np.asarray(reference.dropna(), dtype=float))

    def to_rank(values):
        position = np.searchsorted(ordered, np.asarray(values, dtype=float),
                                   side="right")
        return 100. * position / max(len(ordered), 1)

    return to_rank


def _panel_note(ax, text: str) -> None:
    ax.text(.03, .97, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=6.4, linespacing=1.25, zorder=20,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                  "edgecolor": "#D9D9D9", "linewidth": .45, "alpha": .92})


def _north_arrow(ax) -> None:
    ax.annotate("", xy=(.94, .91), xytext=(.94, .79), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": "#1A1A1A",
                            "linewidth": .8, "mutation_scale": 8,
                            "shrinkA": 0, "shrinkB": 0}, zorder=25)
    ax.text(.94, .935, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.6, fontweight="bold", color="#1A1A1A", zorder=25)


def _draw_fronts(ax, isochrones: gpd.GeoDataFrame, norm, cmap,
                 colorize: bool = True, to_rank=None) -> None:
    """Outline each observed front; synthetic early hulls are dashed."""
    for row in isochrones.sort_values("hrs_since_ignition").itertuples():
        hours = float(row.hrs_since_ignition)
        value = float(to_rank([hours])[0]) if to_rank is not None else hours
        color = cmap(norm(value)) if colorize else "#5A5A5A"
        style = (0, (3.5, 2.)) if row.source == "SYNTHETIC" else "-"
        boundary = gpd.GeoSeries([row.geometry], crs=isochrones.crs).boundary
        boundary.plot(ax=ax, color="white", linewidth=2., alpha=.75, zorder=5)
        boundary.plot(ax=ax, color=color, linewidth=1.05 if colorize else .75,
                      linestyle=style, alpha=.95 if colorize else .55, zorder=6)


def _viirs_detections(project_root: Path) -> gpd.GeoDataFrame:
    """High/nominal-confidence VIIRS detections in the Eaton window."""
    firms = pd.read_parquet(
        Path(project_root) / "data" / "arrival"
        / "firms_viirs_eaton_palisades_jan2025.parquet")
    acquired = pd.to_datetime(firms.acq_dt)
    selected = firms[
        firms.confidence.isin(["h", "n"])
        & firms.latitude.between(EATON_BOUNDS["lat_min"], EATON_BOUNDS["lat_max"])
        & firms.longitude.between(EATON_BOUNDS["lon_min"], EATON_BOUNDS["lon_max"])
        & acquired.ge(EATON_IGNITION)
    ].copy()
    selected["hrs_since_ignition"] = (
        pd.to_datetime(selected.acq_dt) - EATON_IGNITION
    ).dt.total_seconds() / 3600
    return gpd.GeoDataFrame(
        selected,
        geometry=gpd.points_from_xy(selected.longitude, selected.latitude),
        crs=4326).to_crs(METRIC_EPSG)


def _timeline_controls(project_root: Path) -> gpd.GeoDataFrame:
    """Post-ignition timeline events the Eaton posterior is fitted to."""
    events = pd.read_parquet(
        Path(project_root) / "data" / "arrival"
        / "eaton_fire_events_recovered.parquet")
    events = events[events.hrs_since_ignition.ge(0)]
    return gpd.GeoDataFrame(
        events, geometry=gpd.points_from_xy(events.lon, events.lat),
        crs=4326).to_crs(METRIC_EPSG)


def _fire_layers(project_root: Path, reconstruction: pd.DataFrame, fire: str):
    """Structure points, footprints and fronts for one fire, in metric CRS."""
    root = Path(project_root)
    rows = reconstruction[reconstruction.fire.eq(fire)].copy()
    structures = gpd.GeoDataFrame(
        rows, geometry=gpd.points_from_xy(rows.lon_wgs84, rows.lat_wgs84),
        crs=4326).to_crs(METRIC_EPSG)
    structures["x"] = structures.geometry.x
    structures["y"] = structures.geometry.y
    footprints = gpd.read_parquet(
        root / "data" / "nx" / f"{fire}_buildings.parquet",
        columns=["geometry"]).to_crs(METRIC_EPSG)
    fronts = gpd.read_file(
        root / "data" / "arrival" / f"isochrones_{fire.lower()}.gpkg"
    ).to_crs(METRIC_EPSG)
    return structures, footprints, fronts


def _extent(structures: gpd.GeoDataFrame, fronts: gpd.GeoDataFrame,
            cutoff_hrs: float):
    """Map extent covering mapped structures and the fronts drawn with them."""
    mapped = structures[structures.T_arrival_hrs.notna()
                        | structures.T_arrival_interp_hrs.notna()]
    minx, miny, maxx, maxy = mapped.total_bounds
    early = fronts[fronts.hrs_since_ignition.le(cutoff_hrs)]
    if len(early):
        front_minx, front_miny, front_maxx, front_maxy = early.total_bounds
        minx, miny = min(minx, front_minx), min(miny, front_miny)
        maxx, maxy = max(maxx, front_maxx), max(maxy, front_maxy)
    margin_x, margin_y = (maxx - minx) * .045, (maxy - miny) * .055
    xlim = (minx - margin_x, maxx + margin_x)
    ylim = (miny - margin_y, maxy + margin_y)
    return xlim, ylim, early


def _frame(axes, footprints: gpd.GeoDataFrame, xlim, ylim,
           shared_xlabel: bool = False) -> None:
    """Context footprints and local-kilometre axes for one row of maps.

    ``shared_xlabel`` labels only the leftmost panel: every map in a row shares
    one coordinate frame, so repeating the label under each is redundant ink.
    """
    origin_x, origin_y = xlim[0], ylim[0]
    context = footprints.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]
    for ax in axes:
        context.plot(ax=ax, facecolor=CONTEXT_FACE, edgecolor=CONTEXT_EDGE,
                     linewidth=.06, alpha=.4, rasterized=True, zorder=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(axis="both", which="major", length=2.5, pad=1.5,
                       labelsize=6.6)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{(value - origin_x) / 1000:.0f}"))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{(value - origin_y) / 1000:.0f}"))
        if not shared_xlabel:
            ax.set_xlabel("Local easting (km)", fontsize=7)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    if shared_xlabel:
        axes[0].set_xlabel("Local easting (km)", fontsize=7)
    axes[0].set_ylabel("Local northing (km)", fontsize=7)


def make_arrival_reconstruction_figure(project_root: Path, results_dir: Path,
                                       figure_dir: Path,
                                       source_dir: Path) -> Path:
    """Plot the rebuilt arrival ladder for both fires.

    ``results_dir`` is retained for a stable call signature; the rebuild-versus-
    packaged agreement it holds is reported in the ED01 tables, not on the plot.
    """
    root = Path(project_root)
    figures, source = Path(figure_dir), Path(source_dir)
    figures.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    apply_style()

    reconstruction = pd.read_parquet(
        root / "data" / "derived" / "arrival_reconstructed.parquet")

    arrival_values = pd.concat([reconstruction.T_arrival_interp_hrs,
                                reconstruction.T_arrival_hrs]).dropna()
    vmax = float(np.ceil(arrival_values.quantile(.98)))
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=0, vmax=100, clip=True)
    rank_of = {
        fire: _rank_scale(
            reconstruction.loc[reconstruction.fire.eq(fire), "T_arrival_hrs"]
            .fillna(reconstruction.T_arrival_interp_hrs))
        for fire in ("EATON", "PALISADES")}

    cutoff = max(vmax * 1.05, 27.)
    eaton, eaton_footprints, eaton_fronts = _fire_layers(
        root, reconstruction, "EATON")
    palisades, palisades_footprints, palisades_fronts = _fire_layers(
        root, reconstruction, "PALISADES")
    eaton_xlim, eaton_ylim, eaton_early = _extent(eaton, eaton_fronts, cutoff)
    palisades_xlim, palisades_ylim, palisades_early = _extent(
        palisades, palisades_fronts, cutoff)

    # Row heights track each fire's mapped aspect ratio so the equal-aspect
    # maps fill their axes instead of floating in whitespace.
    aspects = [(ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
               for xlim, ylim in ((eaton_xlim, eaton_ylim),
                                  (palisades_xlim, palisades_ylim))]
    panel_width_in = 2.15
    fig, axes = plt.subplots(
        2, 3, figsize=(7.35, 1.15 + panel_width_in * sum(aspects)),
        gridspec_kw={"height_ratios": aspects})
    titles = [
        "a  Eaton — observed fronts",
        "b  Eaton — boundary interpolation",
        "c  Eaton — timeline update ($T_{arrival}$)",
        "d  Palisades — observed fronts",
        "e  Palisades — boundary interpolation ($T_{arrival}$)",
    ]
    for ax, title in zip(axes.ravel()[:5], titles):
        ax.set_title(title, loc="left", fontweight="bold", pad=5, fontsize=7.5)

    # Eaton: observations, the interpolation they imply, the timeline update.
    _frame(axes[0], eaton_footprints, eaton_xlim, eaton_ylim,
           shared_xlabel=True)

    detections = _viirs_detections(root)
    detections = detections[detections.hrs_since_ignition.le(cutoff)]
    axes[0, 0].scatter(detections.geometry.x, detections.geometry.y,
                       c=rank_of["EATON"](detections.hrs_since_ignition),
                       cmap=cmap, norm=norm,
                       s=8, marker="s", edgecolors="white", linewidths=.15,
                       alpha=.92, rasterized=True, zorder=4)
    _draw_fronts(axes[0, 0], eaton_early, norm, cmap, colorize=True,
                 to_rank=rank_of["EATON"])
    _north_arrow(axes[0, 0])

    _draw_fronts(axes[0, 1], eaton_early, norm, cmap, colorize=False)
    interpolated = eaton[eaton.T_arrival_interp_hrs.notna()]
    axes[0, 1].scatter(interpolated.x, interpolated.y,
                       c=rank_of["EATON"](interpolated.T_arrival_interp_hrs),
                       cmap=cmap, norm=norm, s=2.2, linewidths=0, alpha=.9,
                       rasterized=True, zorder=4)

    assigned = eaton[eaton.T_arrival_hrs.notna()]
    axes[0, 2].scatter(assigned.x, assigned.y,
                       c=rank_of["EATON"](assigned.T_arrival_hrs),
                       cmap=cmap, norm=norm, s=2.2, linewidths=0, alpha=.9,
                       rasterized=True, zorder=4)
    controls = _timeline_controls(root)
    axes[0, 2].scatter(controls.geometry.x, controls.geometry.y, s=9,
                       marker="+", c=CONTROL_COLOR, linewidths=.55, alpha=.62,
                       rasterized=True, zorder=8)

    kriged = eaton[eaton.arrival_method.eq("kriged")]
    shift = float((kriged.T_arrival_hrs - kriged.T_arrival_interp_hrs).median())
    _panel_note(axes[0, 0], f"{len(detections):,} VIIRS detections\n"
                            f"{len(eaton_early):,} fronts")
    _panel_note(axes[0, 1], f"{len(interpolated):,} structures")
    _panel_note(axes[0, 2], f"{len(assigned):,} structures\n"
                            f"{len(kriged):,} updated, median {shift:+.1f} h")

    # Palisades: the interpolation is the final clock.
    _frame([axes[1, 0], axes[1, 1]], palisades_footprints, palisades_xlim,
           palisades_ylim, shared_xlabel=True)
    _draw_fronts(axes[1, 0], palisades_early, norm, cmap, colorize=True,
                 to_rank=rank_of["PALISADES"])
    _draw_fronts(axes[1, 1], palisades_early, norm, cmap, colorize=False)
    palisades_assigned = palisades[palisades.T_arrival_hrs.notna()]
    axes[1, 1].scatter(palisades_assigned.x, palisades_assigned.y,
                       c=rank_of["PALISADES"](palisades_assigned.T_arrival_hrs),
                       cmap=cmap, norm=norm,
                       s=2.2, linewidths=0, alpha=.9, rasterized=True, zorder=4)
    _panel_note(axes[1, 0],
                f"{len(palisades_early):,} of {len(palisades_fronts):,} "
                "progression tiles")
    _panel_note(axes[1, 1], f"{len(palisades_assigned):,} structures\n"
                            "no timeline update")

    # The vacated bottom-right cell carries the shared scale and key, so no
    # panel has to host a legend on top of its own data.
    fig.subplots_adjust(left=.068, right=.992, top=.945, bottom=.075,
                        wspace=.1, hspace=.28)
    slot = axes[1, 2].get_position()
    fig.delaxes(axes[1, 2])

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(
        mappable,
        cax=fig.add_axes([slot.x0 + .035, slot.y0 + slot.height * .82,
                          slot.width - .07, .020]),
        orientation="horizontal")
    colorbar.set_label("Arrival order (percentile within fire)", fontsize=7.2)
    colorbar.ax.tick_params(length=2.5, pad=1.5, labelsize=6.6)

    fig.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor="#777777", markeredgecolor="white",
                   markeredgewidth=.3, markersize=4.5,
                   label="VIIRS detection"),
            Line2D([0], [0], color="#555555", lw=1., label="observed front"),
            Line2D([0], [0], color="#555555", lw=1., linestyle=(0, (3.5, 2.)),
                   label="synthetic early hull"),
            Line2D([0], [0], marker="+", color=CONTROL_COLOR, lw=0,
                   markersize=5, label="timeline control event"),
        ],
        loc="upper left",
        bbox_to_anchor=(slot.x0 + .035, slot.y0 + slot.height * .64),
        frameon=False, handlelength=1.7, labelspacing=.6, fontsize=6.6)

    reconstruction[[
        "BLD_ID", "fire", "lon_wgs84", "lat_wgs84", "T_arrival_snap_hrs",
        "T_arrival_interp_hrs", "T_arrival_hrs", "T_arrival_sd_hrs",
        "arrival_method",
    ]].to_csv(source / "ED_fig_arrival_reconstruction_source.csv", index=False)

    stem = figures / "ED_fig_arrival_reconstruction"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def make_eaton_arrival_steps_figure(project_root: Path, figure_dir: Path,
                                    source_dir: Path) -> Path:
    """Draw the standalone Eaton progression-reconstruction sequence.

    The three panels distinguish the observations from the two modeled stages:
    VIIRS-derived fronts, signed-distance interpolation between those fronts,
    and the incident-timeline regression-kriging update used in the analysis.
    """
    root = Path(project_root)
    figures, source = Path(figure_dir), Path(source_dir)
    figures.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    apply_style()

    reconstruction = pd.read_parquet(
        root / "data" / "derived" / "arrival_reconstructed.parquet")
    eaton, footprints, fronts = _fire_layers(root, reconstruction, "EATON")
    arrival_values = pd.concat([
        eaton.T_arrival_interp_hrs, eaton.T_arrival_hrs]).dropna()
    vmax = float(np.ceil(arrival_values.quantile(.98)))
    cutoff = max(vmax * 1.05, 27.)
    xlim, ylim, early_fronts = _extent(eaton, fronts, cutoff)
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax, clip=True)

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.28))
    titles = [
        "a  Fire-front observations",
        "b  Boundary interpolation",
        "c  Timeline update ($T_{arrival}$)",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc="left", fontweight="bold", pad=5,
                     fontsize=8.2)
    _frame(axes, footprints, xlim, ylim)

    detections = _viirs_detections(root)
    detections = detections[detections.hrs_since_ignition.le(cutoff)]
    axes[0].scatter(
        detections.geometry.x, detections.geometry.y,
        c=detections.hrs_since_ignition, cmap=cmap, norm=norm,
        s=8, marker="s", edgecolors="white", linewidths=.15,
        alpha=.92, rasterized=True, zorder=4)
    _draw_fronts(axes[0], early_fronts, norm, cmap, colorize=True)
    _north_arrow(axes[0])

    _draw_fronts(axes[1], early_fronts, norm, cmap, colorize=False)
    interpolated = eaton[eaton.T_arrival_interp_hrs.notna()]
    axes[1].scatter(
        interpolated.x, interpolated.y,
        c=interpolated.T_arrival_interp_hrs, cmap=cmap, norm=norm,
        s=2.2, linewidths=0, alpha=.9, rasterized=True, zorder=4)

    assigned = eaton[eaton.T_arrival_hrs.notna()]
    axes[2].scatter(
        assigned.x, assigned.y, c=assigned.T_arrival_hrs,
        cmap=cmap, norm=norm, s=2.2, linewidths=0, alpha=.9,
        rasterized=True, zorder=4)
    controls = _timeline_controls(root)
    axes[2].scatter(
        controls.geometry.x, controls.geometry.y, s=9, marker="+",
        c=CONTROL_COLOR, linewidths=.55, alpha=.62, rasterized=True, zorder=8)

    kriged = eaton[eaton.arrival_method.eq("kriged")]
    shift = float(
        (kriged.T_arrival_hrs - kriged.T_arrival_interp_hrs).median())
    _panel_note(axes[0], f"{len(detections):,} VIIRS detections\n"
                         f"{len(early_fronts):,} fronts through {cutoff:.0f} h")
    _panel_note(axes[1], f"{len(interpolated):,} structures\n"
                         "distance-weighted between fronts")
    _panel_note(axes[2], f"{len(assigned):,} structures\n"
                         f"{len(kriged):,} kriged (timeline-updated)\n"
                         f"median shift {shift:+.1f} h")

    axes[0].legend(handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markeredgewidth=.3, markersize=4.5,
               label="VIIRS detection"),
        Line2D([0], [0], color="#555555", lw=1., label="observed front"),
        Line2D([0], [0], color="#555555", lw=1., linestyle=(0, (3.5, 2.)),
               label="synthetic early hull"),
    ], loc="lower left", bbox_to_anchor=(.02, .02), borderaxespad=0,
       handlelength=1.7, fontsize=6.2)
    axes[2].legend(handles=[
        Line2D([0], [0], marker="+", color=CONTROL_COLOR, lw=0,
               markersize=5, label="timeline control event")
    ], loc="lower left", bbox_to_anchor=(.02, .02), borderaxespad=0,
       handlelength=1., fontsize=6.2)

    fig.subplots_adjust(left=.068, right=.992, top=.90, bottom=.32, wspace=.12)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(
        mappable, cax=fig.add_axes([.23, .19, .54, .035]),
        orientation="horizontal", extend="max")
    colorbar.set_label("Hours since Eaton ignition", fontsize=7.4)
    colorbar.ax.tick_params(length=2.5, pad=1.5, labelsize=6.6)

    note = textwrap.fill(
        "Panel a shows high- and nominal-confidence VIIRS detections and the "
        "resulting cumulative fire fronts. Panel b assigns a first-stage arrival "
        "time to each structure by signed-distance interpolation between fronts. "
        "Panel c shows the analysis clock: the timeline-informed regression-kriging "
        "update where supported, with boundary interpolation elsewhere. Colour is "
        "clipped at the 98th percentile of Eaton building arrival times.",
        width=165)
    fig.text(.068, .072, note, ha="left", va="top", fontsize=6.35,
             color="#4D4D4D", linespacing=1.3)

    eaton[[
        "BLD_ID", "lon_wgs84", "lat_wgs84", "T_arrival_interp_hrs",
        "T_arrival_hrs", "T_arrival_sd_hrs", "arrival_method",
    ]].to_csv(source / "ED_fig_eaton_arrival_steps_buildings.csv", index=False)
    detections.assign(
        easting_m=detections.geometry.x,
        northing_m=detections.geometry.y,
    )[["acq_dt", "hrs_since_ignition", "confidence", "easting_m",
       "northing_m"]].to_csv(
           source / "ED_fig_eaton_arrival_steps_viirs.csv", index=False)
    controls.assign(
        easting_m=controls.geometry.x,
        northing_m=controls.geometry.y,
    )[["hrs_since_ignition", "easting_m", "northing_m"]].to_csv(
           source / "ED_fig_eaton_arrival_steps_controls.csv", index=False)

    stem = figures / "ED_fig_eaton_arrival_steps"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")
