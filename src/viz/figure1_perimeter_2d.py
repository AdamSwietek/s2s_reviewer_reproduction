"""Two-dimensional accumulated exposure for the Figure 1 Eaton receiver.

The three-dimensional panels resolve exposure on sampled facade patches.  The
two-dimensional model instead segments each footprint perimeter and exchanges
visible length between segments, so accumulated exposure is the share of the
receiver perimeter that destroyed neighbours can see.

The statewide 2D graph is built on OSM footprints, which omit several LARIAC
structures next to this receiver, so the panel reads a small LARIAC-footprint
2D graph over the same scene the other panels draw
(``scripts/build_eaton_scene_2d.py``).  Building identifiers, footprints and
projection are therefore shared with panels a, b and d.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon

from src.viz.figure1_composite import FIG_RC, INK, MM, _panel_title
from src.viz.figure1_eaton_example import (
    CONTEXT, CONTEXT_EDGE, EMITTER, EXCHANGE_STRONG, RECEIVER, TERRAIN,
    TERRAIN_GRID, _draw_plan_building, _plan_panel, _resolve_building_id,
    _three_d_panel, load_eaton_example,
)

PERIMETER_QUIET = "#AFB5B8"   # perimeter no destroyed neighbour can see
PERIMETER_EXPOSED = "#8F2F21"  # deeper than the emitter fill, so it reads on it
VIEW_HALF_WIDTH_M = 26.0      # a little wider than a-b: the block, not the pair


def _segment_endpoints(patches: pd.DataFrame) -> np.ndarray:
    """Endpoints of each perimeter segment from its centre, normal and length."""
    tangent = np.column_stack([-patches.ny.to_numpy(), patches.nx.to_numpy()])
    centre = patches[["x_m", "y_m"]].to_numpy()
    half = (patches.length_m.to_numpy() / 2)[:, None] * tangent
    return np.stack([centre - half, centre + half], axis=1)


def load_two_dimensional_exposure(
    run_2d: Path,
    building_cache: Path,
    analysis_path: Path,
    *,
    receiver_id: int,
    fire: str = "EATON",
):
    """Receiver perimeter, destroyed emitters, and their visible exchange."""
    run_2d = Path(run_2d)
    required = [
        run_2d / name for name in (
            "building_metrics.parquet", "patch_exchange.parquet",
            "building_ids.parquet",
        )
    ]
    if any(not path.exists() for path in required):
        raise FileNotFoundError(
            f"The Eaton 2D scene graph is missing from {run_2d}. Build it with "
            "python scripts/build_eaton_scene_2d.py"
        )
    receiver_id = int(receiver_id)

    with duckdb.connect() as con:
        metrics = con.execute(
            "SELECT perimeter_m, neighbor_count, bvf_all_edges "
            "FROM read_parquet(?) WHERE building_id = ?",
            [str(run_2d / "building_metrics.parquet"), receiver_id],
        ).fetchdf()
        if metrics.empty:
            raise ValueError(f"Receiver {receiver_id} is not a node of {run_2d}")
        exchange = con.execute(
            """
            SELECT patch_p, patch_q, building_p, building_q,
                   distance_m, exchange_pq_m
            FROM read_parquet(?)
            WHERE building_p = ? OR building_q = ?
            """,
            [str(run_2d / "patch_exchange.parquet"), receiver_id, receiver_id],
        ).fetchdf()
        patches = con.execute(
            "SELECT patch_id, building_id, x_m, y_m, nx, ny, length_m "
            "FROM read_parquet(?)",
            [str(run_2d / "patches" / "*.parquet")],
        ).fetchdf()

    owns = exchange.building_p.eq(receiver_id)
    exchange["emitter"] = np.where(owns, exchange.building_q, exchange.building_p)
    exchange["receiver_patch"] = np.where(owns, exchange.patch_p, exchange.patch_q)
    exchange["emitter_patch"] = np.where(owns, exchange.patch_q, exchange.patch_p)

    outcomes = pd.read_parquet(
        analysis_path, columns=["graph_id", "fire", "is_destroyed"],
    )
    destroyed = set(outcomes.loc[
        outcomes.fire.eq(fire) & outcomes.is_destroyed.eq(1), "graph_id"
    ].astype(int))
    exchange["emitter_destroyed"] = exchange.emitter.isin(destroyed)
    destroyed_exchange = exchange.loc[exchange.emitter_destroyed]
    if destroyed_exchange.empty:
        raise ValueError(
            f"Receiver {receiver_id} sees no destroyed {fire} neighbour in 2D"
        )

    receiver_patches = patches.loc[patches.building_id.eq(receiver_id)].copy()
    segment_exposure = destroyed_exchange.groupby(
        "receiver_patch",
    ).exchange_pq_m.sum()
    receiver_patches["exposed_m"] = np.minimum(
        receiver_patches.patch_id.map(segment_exposure).fillna(0.0),
        receiver_patches.length_m,
    )
    receiver_patches["exposed_share"] = (
        receiver_patches.exposed_m / receiver_patches.length_m
    )

    emitter_exposure = (
        destroyed_exchange.groupby("emitter").exchange_pq_m.sum()
        .sort_values(ascending=False)
    )
    perimeter_m = float(metrics.perimeter_m.iloc[0])
    exposed_m = float(receiver_patches.exposed_m.sum())
    footprints = gpd.read_parquet(
        building_cache, columns=["building_id", "geometry"],
        filters=[[("building_id", "in", patches.building_id.unique().tolist())]],
    ).set_index("building_id")

    summary = pd.DataFrame([{
        "receiver_id": receiver_id,
        "fire": fire,
        "perimeter_m": perimeter_m,
        "perimeter_segments": len(receiver_patches),
        "neighbors_2d": int(metrics.neighbor_count.iloc[0]),
        "destroyed_emitters_2d": int(len(emitter_exposure)),
        "exposed_perimeter_m": exposed_m,
        "F_i_star_2d": exposed_m / perimeter_m,
        "strongest_emitter": int(emitter_exposure.index[0]),
        "strongest_emitter_m": float(emitter_exposure.iloc[0]),
    }])
    scene = {
        "receiver_id": receiver_id,
        "receiver_patches": receiver_patches,
        "patches": patches,
        "exchange": exchange,
        "emitter_exposure": emitter_exposure,
        "footprints": footprints,
        "destroyed": destroyed,
    }
    return scene, summary


def plot_perimeter_exposure_panel(
    ax, scene, summary, *, view_half_width_m: float = VIEW_HALF_WIDTH_M,
    aspect_ratio: float = 1.43, sight_lines: int = 10, show_value: bool = True,
):
    """Plan panel: the exposed share of the receiver perimeter in 2D."""
    receiver_id = scene["receiver_id"]
    footprints = scene["footprints"]
    receiver_patches = scene["receiver_patches"]
    destroyed = scene["destroyed"]
    emitters = set(scene["emitter_exposure"].index.astype(int))

    receiver_centroid = footprints.loc[receiver_id].geometry.centroid
    centre = np.array([receiver_centroid.x, receiver_centroid.y])
    half_height = view_half_width_m / aspect_ratio
    xlim = (centre[0] - view_half_width_m, centre[0] + view_half_width_m)
    ylim = (centre[1] - half_height, centre[1] + half_height)

    for building_id, row in footprints.iterrows():
        if building_id == receiver_id:
            continue
        minx, miny, maxx, maxy = row.geometry.bounds
        if maxx < xlim[0] or minx > xlim[1] or maxy < ylim[0] or miny > ylim[1]:
            continue
        if building_id in emitters:
            _draw_plan_building(ax, row.geometry, EMITTER, zorder=3)
        elif building_id in destroyed:
            _draw_plan_building(ax, row.geometry, EMITTER, alpha=.45, zorder=2)
        else:
            _draw_plan_building(ax, row.geometry, CONTEXT, edge=CONTEXT_EDGE,
                                alpha=.58, zorder=1)
    _draw_plan_building(ax, footprints.loc[receiver_id].geometry, RECEIVER,
                        zorder=4)

    # Sight lines run under the perimeter so the highlight stays readable.
    exchange = scene["exchange"]
    strongest = exchange.loc[exchange.emitter_destroyed].nlargest(
        sight_lines, "exchange_pq_m",
    )
    patch_centres = scene["patches"].set_index("patch_id")[["x_m", "y_m"]]
    for pair in strongest.itertuples():
        start = patch_centres.loc[pair.emitter_patch].to_numpy(float)
        end = patch_centres.loc[pair.receiver_patch].to_numpy(float)
        ax.plot(
            [start[0], end[0]], [start[1], end[1]], color=EXCHANGE_STRONG,
            ls=(0, (2.2, 1.5)), lw=.62, alpha=.85, zorder=6,
            dash_capstyle="round",
        )

    # Each wall carries its exposed length as a centred highlight, so the
    # highlighted perimeter sums to the exposed perimeter in the label.
    endpoints = _segment_endpoints(receiver_patches)
    for (start, end), row in zip(endpoints, receiver_patches.itertuples()):
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=PERIMETER_QUIET, lw=1.1, solid_capstyle="butt", zorder=7)
        if row.exposed_m <= 0:
            continue
        midpoint = (start + end) / 2
        direction = (end - start) / max(np.linalg.norm(end - start), 1e-9)
        offset = direction * (row.exposed_m / 2)
        ax.plot(*np.column_stack([midpoint - offset, midpoint + offset]),
                color="white", lw=4.2, solid_capstyle="butt", zorder=8)
        ax.plot(*np.column_stack([midpoint - offset, midpoint + offset]),
                color=PERIMETER_EXPOSED, lw=2.6, solid_capstyle="butt",
                zorder=9)

    if show_value:
        row = summary.iloc[0]
        ax.annotate(
            rf"$F_i^*$(2D) $=$ {row.F_i_star_2d:.3f}"
            "\n"
            f"{row.exposed_perimeter_m:.1f} of {row.perimeter_m:.1f} m exposed"
            "\n"
            f"{int(row.destroyed_emitters_2d)} destroyed neighbours",
            (.018, .022), xycoords="axes fraction", ha="left", va="bottom",
            fontsize=5.4, color=INK, linespacing=1.35, zorder=12,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.80, pad=.5),
        )

    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAF9F5")
    ax.axis("off")


def _total_three_dimensional_exposure(run_3d, analysis_path, receiver_id, fire):
    """Receiver exposure summed over every destroyed neighbour in the 3D run."""
    with duckdb.connect() as con:
        edges = con.execute(
            """
            SELECT building_i, building_j, vf_i_to_j, vf_j_to_i
            FROM read_parquet(?)
            WHERE building_i = ? OR building_j = ?
            """,
            [str(Path(run_3d) / "building_edges.parquet"),
             int(receiver_id), int(receiver_id)],
        ).fetchdf()
    outcomes = pd.read_parquet(
        analysis_path, columns=["graph_id", "fire", "is_destroyed"],
    )
    destroyed = set(outcomes.loc[
        outcomes.fire.eq(fire) & outcomes.is_destroyed.eq(1), "graph_id"
    ].astype(int))
    owns = edges.building_i.eq(int(receiver_id))
    emitter = np.where(owns, edges.building_j, edges.building_i)
    exposure = np.where(owns, edges.vf_i_to_j, edges.vf_j_to_i)
    return float(exposure[pd.Series(emitter).isin(destroyed).to_numpy()].sum())


def build_perimeter_exposure_row(
    run_3d: Path,
    building_cache: Path,
    analysis_path: Path,
    run_2d: Path,
    output_dir: Path,
    *,
    receiver_id: int | str,
    primary_emitter_id: int | str | None = None,
    fire: str = "EATON",
    show_values: bool = True,
    stem: str = "Fig1a-d_two_dimensional_exposure",
    patch_run_dir: Path | None = None,
):
    """Draw a-d with 2D accumulated exposure in place of the pairwise panel."""
    receiver_graph_id = _resolve_building_id(building_cache, receiver_id)
    if primary_emitter_id is not None:
        primary_emitter_id = _resolve_building_id(
            building_cache, primary_emitter_id,
        )
    buildings, emitters, patches, exchange, summary_3d = load_eaton_example(
        run_3d, building_cache, analysis_path, receiver_id=receiver_graph_id,
        patch_run_dir=patch_run_dir,
    )
    emitter_ids = emitters.emitter.astype(int).tolist()
    primary = (
        emitter_ids[0] if primary_emitter_id is None else int(primary_emitter_id)
    )
    if primary not in emitter_ids:
        raise ValueError(
            f"Requested emitter {primary} is not among receiver "
            f"{receiver_graph_id}'s three strongest destroyed contributors"
        )
    scene, summary_2d = load_two_dimensional_exposure(
        run_2d, building_cache, analysis_path,
        receiver_id=receiver_graph_id, fire=fire,
    )

    width, height = 180 * MM, 52 * MM
    fx, fy = lambda value: value / width, lambda value: value / height
    panel_height, panel_bottom, gap = 1.28, .48, .10
    panel_widths = (1.43, 1.43, 1.83, 1.83)
    panel_lefts = [.20]
    for panel_width in panel_widths[:-1]:
        panel_lefts.append(panel_lefts[-1] + panel_width + gap)

    with plt.rc_context(FIG_RC):
        fig = plt.figure(figsize=(width, height))
        axes = []
        for index, (left, panel_width) in enumerate(
            zip(panel_lefts, panel_widths)
        ):
            kwargs = (
                {"projection": "3d", "computed_zorder": False}
                if index == 3 else {}
            )
            axes.append(fig.add_axes([
                fx(left), fy(panel_bottom), fx(panel_width), fy(panel_height),
            ], **kwargs))

        _plan_panel(axes[0], buildings, receiver_graph_id, primary,
                    surface=False, show_value=show_values)
        _plan_panel(axes[1], buildings, receiver_graph_id, primary,
                    surface=True, show_value=show_values)
        plot_perimeter_exposure_panel(
            axes[2], scene, summary_2d,
            aspect_ratio=panel_widths[2] / panel_height,
            show_value=show_values,
        )
        _three_d_panel(axes[3], buildings, patches, exchange,
                       receiver_graph_id, emitter_ids, links_per_emitter=2)
        if show_values:
            axes[3].text2D(
                .50, .02,
                rf"$F_i^*$(3D) $=\sum_j F_{{ij}}={emitters.F_ij.sum():.3f}$",
                transform=axes[3].transAxes, ha="center", va="bottom",
                fontsize=6.0, color=INK, zorder=30,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82,
                          pad=.4),
            )

        titles = [
            r"Centroid separation  $d_{CC}$",
            r"Surface separation  $d_{SS}$",
            r"Exposed perimeter  $F_i^*$ (2D)",
            r"Accumulated exposure  $F_i^*$ (3D)",
        ]
        for ax, letter, title in zip(axes, "abcd", titles):
            _panel_title(ax, letter, title, size=7.3, rise=.025, gap_pt=7.2)

        fig.legend(handles=[
            Patch(facecolor=EMITTER, edgecolor="none",
                  label=r"Destroyed emitter $j$"),
            Patch(facecolor=RECEIVER, edgecolor="none", label=r"Receiver $i$"),
            Patch(facecolor=CONTEXT, edgecolor=CONTEXT_EDGE, lw=.4,
                  label="Neighbor"),
            Patch(facecolor=TERRAIN, edgecolor=TERRAIN_GRID, lw=.4,
                  label="Local terrain"),
            Line2D([0], [0], color=PERIMETER_EXPOSED, lw=2.4,
                   label="Exposed perimeter"),
            Line2D([0], [0], color=PERIMETER_QUIET, lw=1.1,
                   label="Unexposed perimeter"),
            Line2D([0], [0], color=EXCHANGE_STRONG, lw=.9, ls=(0, (2.2, 1.5)),
                   label="Visible surface exchange"),
        ], loc="lower center", bbox_to_anchor=(.5, .040), ncol=7,
           frameon=False, fontsize=5.4, handlelength=1.0, handleheight=.70,
           handletextpad=.32, columnspacing=.7, labelspacing=.20, borderpad=0)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{stem}.pdf", facecolor="white")
        fig.savefig(output_dir / f"{stem}.png", dpi=400, facecolor="white")

    # Panel d draws the three strongest contributors; the 2D panel accumulates
    # over every destroyed neighbour, so the full 3D sum is reported alongside.
    summary_3d = summary_3d.assign(
        F_i_star_3d=float(emitters.F_ij.sum()),
        F_i_star_3d_all_destroyed=_total_three_dimensional_exposure(
            run_3d, analysis_path, receiver_graph_id, fire,
        ),
    )
    summary = pd.concat([
        summary_3d,
        summary_2d.drop(columns=summary_2d.columns.intersection(summary_3d.columns)),
    ], axis=1)
    return fig, summary, scene
