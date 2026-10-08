"""Observed Eaton geometry for Figure 1 panels a-d.

The drawing is reconstructed from the completed OpenView 3D products: LARIAC
footprints, completed building heights and base elevations, sampled facade
patches, and visible patch-to-patch exchange.  No schematic building geometry
is introduced.
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import nearest_points

from src.viz.figure1_composite import (
    FIG_RC, INK, MM, _panel_title,
)


DEFAULT_EATON_RECEIVER = 2_133_192
DEFAULT_EATON_STREET_EMITTER = 1_989_387
EMITTER = "#9A6858"       # muted fired clay
RECEIVER = "#AEC3CD"      # light architectural blue-grey; keeps the rays legible
CONTEXT = "#DDD9CF"       # warm stone
CONTEXT_EDGE = "#AAA79F"
TERRAIN = "#E3E6D8"       # pale sage ground
TERRAIN_GRID = "#C1C7B6"
EXCHANGE_STRONG = "#A86459"  # desaturated clay-red analytical linework


def _polygons(geometry):
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [part for part in geometry.geoms if isinstance(part, Polygon)]


def _sql_ids(values) -> str:
    return ", ".join(str(int(value)) for value in values)


def _resolve_building_id(building_cache: Path, value: int | str) -> int:
    """Resolve either an internal graph ID or a LARIAC source ID."""
    cache_path = str(Path(building_cache)).replace("'", "''")
    con = duckdb.connect()
    try:
        graph_match = con.execute(
            f"SELECT building_id FROM read_parquet('{cache_path}') "
            "WHERE building_id = ? LIMIT 1",
            [int(value)],
        ).fetchone()
        if graph_match is not None:
            return int(graph_match[0])

        source_match = con.execute(
            f"SELECT building_id FROM read_parquet('{cache_path}') "
            "WHERE LARIAC_BLD_ID = ? LIMIT 1",
            [str(value)],
        ).fetchone()
        if source_match is not None:
            return int(source_match[0])
    finally:
        con.close()
    raise ValueError(f"Building identifier {value!r} was not found in {building_cache}")


def load_eaton_example(
    run_dir: Path,
    building_cache: Path,
    analysis_path: Path,
    *,
    receiver_id: int = DEFAULT_EATON_RECEIVER,
    n_emitters: int = 3,
    patch_run_dir: Path | None = None,
):
    """Load one receiver and its strongest destroyed Eaton emitters.

    Emitters are ranked from ``run_dir``'s building edges. Facade patches and
    their exchange come from ``patch_run_dir`` when given: a companion run
    with the same configuration that kept the patch table, which county runs
    do not.
    """
    run_dir = Path(run_dir)
    patch_run_dir = Path(patch_run_dir) if patch_run_dir is not None else run_dir
    analysis = pd.read_parquet(
        analysis_path,
        columns=["graph_id", "fire", "outcome", "is_destroyed"],
    )
    eaton = analysis.loc[analysis.fire.eq("EATON")].copy()

    con = duckdb.connect()
    con.register("eaton", eaton)
    edge_path = str(run_dir / "building_edges.parquet").replace("'", "''")
    try:
        emitters = con.execute(f"""
            WITH directed AS (
                SELECT building_i AS receiver, building_j AS emitter,
                       vf_i_to_j AS F_ij, has_contact
                FROM read_parquet('{edge_path}')
                UNION ALL
                SELECT building_j AS receiver, building_i AS emitter,
                       vf_j_to_i AS F_ij, has_contact
                FROM read_parquet('{edge_path}')
            )
            SELECT d.emitter, d.F_ij, d.has_contact, a.outcome
            FROM directed d
            INNER JOIN eaton a ON d.emitter = a.graph_id
            WHERE d.receiver = {int(receiver_id)}
              AND a.is_destroyed = 1 AND d.F_ij > 0
            ORDER BY d.F_ij DESC
            LIMIT {int(n_emitters)}
        """).fetchdf()
        if len(emitters) != n_emitters:
            raise ValueError(
                f"Receiver {receiver_id} has only {len(emitters)} positive "
                "destroyed-neighbor contributions."
            )
        building_ids = [int(receiver_id), *emitters.emitter.astype(int)]
        id_sql = _sql_ids(building_ids)
        patches = con.execute(f"""
            SELECT patch_id, building_id, x_m, y_m, z_m,
                   nx, ny, nz, width_m, height_m, area_m2,
                   z_low_m, z_high_m
            FROM read_parquet(
                '{str(patch_run_dir / 'patches' / '*.parquet').replace("'", "''")}',
                union_by_name=true
            )
            WHERE building_id IN ({id_sql})
        """).fetchdf()
        exchange = con.execute(f"""
            SELECT patch_p, patch_q, building_p, building_q,
                   distance_m, exchange_pq_m2
            FROM read_parquet(
                '{str(patch_run_dir / 'patch_exchange.parquet').replace("'", "''")}'
            )
            WHERE building_p IN ({id_sql}) AND building_q IN ({id_sql})
              AND (building_p = {int(receiver_id)}
                   OR building_q = {int(receiver_id)})
        """).fetchdf()
    finally:
        con.unregister("eaton")
        con.close()

    buildings = gpd.read_parquet(
        building_cache,
        columns=["building_id", "LARIAC_BLD_ID", "height_m", "base_z_m",
                 "geometry"],
        filters=[[('building_id', 'in', building_ids)]],
    ).set_index("building_id").reindex(building_ids)
    if buildings.geometry.isna().any():
        missing = buildings.index[buildings.geometry.isna()].tolist()
        raise ValueError(f"Missing observed geometry for buildings: {missing}")
    if set(patches.building_id.astype(int)) != set(building_ids):
        raise ValueError("The selected Eaton buildings do not all have 3D patches")

    emitters = emitters.assign(rank=np.arange(1, len(emitters) + 1))
    summary = pd.DataFrame([{
        "fire": "Eaton",
        "receiver_id": int(receiver_id),
        "receiver_outcome": eaton.set_index("graph_id").loc[
            receiver_id, "outcome"
        ],
        "emitter_ids": ", ".join(emitters.emitter.astype(str)),
        "pairwise_F_ij": float(emitters.F_ij.iloc[0]),
        "cumulative_F_i_star": float(emitters.F_ij.sum()),
        "destroyed_emitters": len(emitters),
    }])
    return buildings, emitters, patches, exchange, summary


def _scene_limits(buildings, pad=0.75):
    minx, miny, maxx, maxy = buildings.total_bounds
    return (minx - pad, maxx + pad), (miny - pad, maxy + pad)


def _draw_plan_building(ax, geometry, colour, *, edge="#50555A",
                        alpha=1.0, zorder=2):
    for polygon in _polygons(geometry):
        ax.add_patch(MplPolygon(
            np.asarray(polygon.exterior.coords), closed=True,
            facecolor=colour, edgecolor=edge, linewidth=.45,
            alpha=alpha, zorder=zorder,
        ))


def _plan_panel(
    ax, buildings, receiver_id, primary_emitter, *, surface=False,
    show_value=False,
):
    other_ids = [
        value for value in buildings.index
        if value not in (receiver_id, primary_emitter)
    ]
    for building_id in other_ids:
        _draw_plan_building(
            ax, buildings.loc[building_id].geometry, CONTEXT,
            edge=CONTEXT_EDGE, alpha=.58, zorder=1,
        )
    _draw_plan_building(
        ax, buildings.loc[primary_emitter].geometry, EMITTER, zorder=3,
    )
    _draw_plan_building(
        ax, buildings.loc[receiver_id].geometry, RECEIVER, zorder=3,
    )

    source = buildings.loc[primary_emitter].geometry
    target = buildings.loc[receiver_id].geometry
    if surface:
        p0, p1 = nearest_points(source, target)
        xy0, xy1 = np.asarray(p0.coords[0]), np.asarray(p1.coords[0])
        distance = source.distance(target)
        linestyle = "-"
        for point in (xy0, xy1):
            ax.plot(*point, "o", ms=2.6, mfc=EXCHANGE_STRONG, mec="white",
                    mew=.35, zorder=8)
    else:
        xy0 = np.array([source.centroid.x, source.centroid.y])
        xy1 = np.array([target.centroid.x, target.centroid.y])
        distance = source.centroid.distance(target.centroid)
        linestyle = "-"
        for point in (xy0, xy1):
            ax.plot(*point, "o", ms=3.2, mfc="white", mec=INK,
                    mew=.65, zorder=8)

    ax.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]], color="white", lw=2.7,
            zorder=6, solid_capstyle="round")
    ax.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]], color=INK, lw=1.0,
            ls=linestyle, zorder=7, solid_capstyle="round")
    if show_value:
        midpoint = (xy0 + xy1) / 2
        symbol = r"$d_{SS}$" if surface else r"$d_{CC}$"
        ax.annotate(
            rf"{symbol} = {distance:.1f} m", midpoint, xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=6.5, color=INK, path_effects=[], zorder=9,
        )
    xlim, ylim = _scene_limits(buildings)
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAF9F5")
    ax.axis("off")


def _prism_faces(geometry, base_z, height, origin):
    x0, y0, z0 = origin
    for polygon in _polygons(geometry):
        coords = np.asarray(polygon.exterior.coords)[:-1]
        bottom = np.column_stack([
            coords[:, 0] - x0, coords[:, 1] - y0,
            np.full(len(coords), base_z - z0),
        ])
        top = bottom.copy()
        top[:, 2] += height
        walls = [
            [bottom[i], bottom[(i + 1) % len(bottom)],
             top[(i + 1) % len(top)], top[i]]
            for i in range(len(bottom))
        ]
        yield walls, top


def _draw_prism(ax, row, origin, colour, *, alpha=1.0, zorder=3):
    for walls, roof in _prism_faces(
        row.geometry, float(row.base_z_m), float(row.height_m), origin,
    ):
        wall = tuple(np.asarray(plt.matplotlib.colors.to_rgb(colour)) * .82)
        ax.add_collection3d(Poly3DCollection(
            walls + [roof], facecolors=[wall] * len(walls) + [colour],
            edgecolors="#45494D", linewidths=.34, alpha=alpha,
            zorder=zorder,
        ))


def _draw_terrain(ax, buildings, origin):
    """Draw a restrained local terrain plane fitted to building base elevations."""
    centres = np.array([
        [row.geometry.centroid.x - origin[0],
         row.geometry.centroid.y - origin[1],
         float(row.base_z_m) - origin[2]]
        for _, row in buildings.iterrows()
    ])
    design = np.column_stack([centres[:, :2], np.ones(len(centres))])
    slope_x, slope_y, intercept = np.linalg.lstsq(
        design, centres[:, 2], rcond=None,
    )[0]
    minx, miny, maxx, maxy = buildings.total_bounds
    x = np.linspace(minx - origin[0] - 2, maxx - origin[0] + 2, 9)
    y = np.linspace(miny - origin[1] - 2, maxy - origin[1] + 2, 9)
    xx, yy = np.meshgrid(x, y)
    zz = slope_x * xx + slope_y * yy + intercept - .10
    ax.plot_surface(
        xx, yy, zz, color=TERRAIN, edgecolor=TERRAIN_GRID,
        linewidth=.18, alpha=.88, shade=False, zorder=0,
        rstride=2, cstride=2,
    )


def _link_coordinates(exchange, patches, receiver_id, emitter_ids,
                      *, links_per_emitter):
    patch_xyz = patches.set_index("patch_id")[["x_m", "y_m", "z_m"]]
    rows = []
    for emitter_id in emitter_ids:
        pair = exchange.loc[
            ((exchange.building_p.eq(receiver_id)
              & exchange.building_q.eq(emitter_id))
             | (exchange.building_q.eq(receiver_id)
                & exchange.building_p.eq(emitter_id)))
        ].nlargest(links_per_emitter, "exchange_pq_m2")
        for item in pair.itertuples():
            if item.patch_p not in patch_xyz.index or item.patch_q not in patch_xyz.index:
                continue
            rows.append((
                patch_xyz.loc[item.patch_p].to_numpy(float),
                patch_xyz.loc[item.patch_q].to_numpy(float),
                float(item.exchange_pq_m2), int(emitter_id),
            ))
    return rows


def _three_d_panel(ax, buildings, patches, exchange, receiver_id,
                   emitter_ids, *, links_per_emitter):
    minx, miny, maxx, maxy = buildings.total_bounds
    z_min = float(buildings.base_z_m.min())
    z_max = float((buildings.base_z_m + buildings.height_m).max())
    origin = ((minx + maxx) / 2, (miny + maxy) / 2, z_min)

    _draw_terrain(ax, buildings, origin)

    active = {receiver_id, *emitter_ids}
    for building_id, row in buildings.iterrows():
        if building_id == receiver_id:
            colour, alpha, zorder = RECEIVER, 1.0, 9
        elif building_id in emitter_ids:
            colour, alpha, zorder = EMITTER, 1.0, 10
        else:
            colour, alpha, zorder = CONTEXT, .22, 2
        _draw_prism(ax, row, origin, colour, alpha=alpha, zorder=zorder)

    links = _link_coordinates(
        exchange, patches, receiver_id, emitter_ids,
        links_per_emitter=links_per_emitter,
    )
    if links:
        peak = max(weight for _, _, weight, _ in links)
        for start, end, weight, _ in links:
            start = start - np.asarray(origin)
            end = end - np.asarray(origin)
            ax.plot(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=EXCHANGE_STRONG, ls=(0, (2.2, 1.5)),
                lw=.55 + .55 * np.sqrt(weight / peak),
                alpha=.92, zorder=7, dash_capstyle="round",
            )

    dx, dy, dz = maxx - minx, maxy - miny, z_max - z_min
    ax.set_xlim(-dx / 2 - 1.5, dx / 2 + 1.5)
    ax.set_ylim(-dy / 2 - 1.5, dy / 2 + 1.5)
    ax.set_zlim(-.3, dz + 1.0)
    ax.set_box_aspect((dx, dy, max(dz * 1.7, min(dx, dy) * .32)), zoom=1.55)
    ax.set_proj_type("ortho")
    ax.view_init(elev=28, azim=-58)
    ax.set_axis_off()
    ax.patch.set_alpha(0)


def build_eaton_geometry_panels(
    run_dir: Path,
    building_cache: Path,
    analysis_path: Path,
    output_dir: Path,
    *,
    receiver_id: int | str = DEFAULT_EATON_RECEIVER,
    primary_emitter_id: int | str | None = None,
    show_values: bool = False,
    stem: str = "Fig1a-d_eaton_observed_geometry",
    patch_run_dir: Path | None = None,
):
    """Render the observed Eaton a-d sequence and return its source summary."""
    receiver_id = _resolve_building_id(building_cache, receiver_id)
    if primary_emitter_id is not None:
        primary_emitter_id = _resolve_building_id(
            building_cache, primary_emitter_id,
        )
    buildings, emitters, patches, exchange, summary = load_eaton_example(
        run_dir, building_cache, analysis_path, receiver_id=receiver_id,
        patch_run_dir=patch_run_dir,
    )
    primary = (
        int(emitters.emitter.iloc[0])
        if primary_emitter_id is None else int(primary_emitter_id)
    )
    emitter_ids = emitters.emitter.astype(int).tolist()
    if primary not in emitter_ids:
        raise ValueError(
            f"Requested pairwise emitter {primary} is not among receiver "
            f"{receiver_id}'s three strongest destroyed contributors"
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
                if index >= 2 else {}
            )
            axes.append(fig.add_axes([
                fx(left), fy(panel_bottom), fx(panel_width), fy(panel_height),
            ], **kwargs))

        _plan_panel(
            axes[0], buildings, receiver_id, primary,
            surface=False, show_value=show_values,
        )
        _plan_panel(
            axes[1], buildings, receiver_id, primary,
            surface=True, show_value=show_values,
        )
        _three_d_panel(
            axes[2], buildings, patches, exchange, receiver_id, [primary],
            links_per_emitter=4,
        )
        _three_d_panel(
            axes[3], buildings, patches, exchange, receiver_id, emitter_ids,
            links_per_emitter=2,
        )

        pairwise_value = float(
            emitters.loc[emitters.emitter.eq(primary), "F_ij"].iloc[0]
        )
        cumulative_value = float(emitters.F_ij.sum())
        summary.loc[:, "pairwise_emitter_id"] = primary
        summary.loc[:, "pairwise_F_ij"] = pairwise_value
        if show_values:
            value_box = dict(
                facecolor="white", edgecolor="none", alpha=.82, pad=.4,
            )
            axes[2].text2D(
                .50, .02, rf"$F_{{ij}}={pairwise_value:.3f}$",
                transform=axes[2].transAxes, ha="center", va="bottom",
                fontsize=6.5, color=INK, bbox=value_box, zorder=30,
            )
            axes[3].text2D(
                .50, .02,
                rf"$F_i^*=\sum_j F_{{ij}}={cumulative_value:.3f}$",
                transform=axes[3].transAxes, ha="center", va="bottom",
                fontsize=6.5, color=INK, bbox=value_box, zorder=30,
            )

        titles = [
            r"Centroid separation  $d_{CC}$",
            r"Surface separation  $d_{SS}$",
            r"Pairwise exposure  $F_{ij}$",
            r"Accumulated exposure  $F_i^*$",
        ]
        for ax, letter, title in zip(axes, "abcd", titles):
            _panel_title(ax, letter, title, size=7.3, rise=.025, gap_pt=7.2)

        legend_handles = [
            Patch(facecolor=EMITTER, edgecolor="none",
                  label=r"Destroyed emitter $j$"),
            Patch(facecolor=RECEIVER, edgecolor="none",
                  label=r"Receiver $i$"),
            Patch(facecolor=CONTEXT, edgecolor=CONTEXT_EDGE, lw=.4,
                  label="Neighbor"),
            Patch(facecolor=TERRAIN, edgecolor=TERRAIN_GRID, lw=.4,
                  label="Local terrain"),
            Line2D([0], [0], color=EXCHANGE_STRONG, lw=.9,
                   ls=(0, (2.2, 1.5)), label="Visible surface exchange"),
        ]
        fig.legend(
            handles=legend_handles, loc="lower center",
            bbox_to_anchor=(.5, .055), ncol=5, frameon=False,
            fontsize=5.9, handlelength=.95, handleheight=.70,
            handletextpad=.35, columnspacing=.8, labelspacing=.20,
            borderpad=0,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{stem}.pdf", facecolor="white")
        fig.savefig(output_dir / f"{stem}.png", dpi=400, facecolor="white")
    return fig, summary, emitters
