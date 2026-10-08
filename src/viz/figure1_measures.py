"""The Figure 1 measurement row: distance in a-b, surface exposure in c-d.

One Eaton scene carries all four panels. The plan panels measure centroid and
surface separation, panel c shades the share of the receiver perimeter that
destroyed neighbours can see in the 2D graph, and panel d extrudes the same
footprints and shades each receiver facade by the exposure it accumulates in
the 3D graph. Panels are drawn into axes the caller has already positioned, so
the standalone row and the Figure 1 composite share one implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb, to_rgba
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import shapely
from shapely.geometry import MultiPoint
from shapely.ops import nearest_points, triangulate

from src.viz.figure1_eaton_example import (
    _polygons, _resolve_building_id, load_eaton_example,
)
from src.viz.figure1_perimeter_2d import (
    _segment_endpoints, load_two_dimensional_exposure,
)

RECEIVER, SOURCE, CONTEXT = "#A9C2CE", "#96594B", "#F1EFEB"
LOW, MEDIUM, HIGH = "#E7A18F", "#B85C4A", "#743126"
QUIET, INK, CONTEXT_EDGE, CORRIDOR = "#8FA3AD", "#333333", "#D5D2CB", "#B85C4A"
OUTLINE, CONTEXT_OUTLINE, MEASURE, EDGE, PERIMETER = .7, .4, 1.15, 1.8, .65
PANEL_TITLES = ["Centroid distance", "Surface distance",
                "2D exposed perimeter", "3D accumulated exposure"]
SHADE = LinearSegmentedColormap.from_list("exposure", [LOW, MEDIUM, HIGH])
# A few faces take most of the exchange, so the ramp is eased to keep the rest
# from collapsing into the palest class.
EXPOSURE_NORM = PowerNorm(gamma=.6, vmin=0, vmax=1)
# Point-to-point rendering of c-d: evenly sampled wall points, links between
# facing, mutually visible points, link opacity eased by view-factor density.
LINK = "#8A3A2A"
LINK_ALPHA = (0., .45)
LINK_GAMMA = 1.0
LINK_WIDTH, LINK_WIDTH_3D = .3, .08
SPACING_2D, SPACING_3D = 2.0, 2.0
RAY_OFFSET, MIN_CROSSING, MAX_DISTANCE = .05, .25, 152.4


@dataclass
class MeasureScene:
    """Everything the four panels draw, resolved once."""

    receiver: int
    source: int
    emitter_ids: list
    footprints: gpd.GeoDataFrame
    in_window: gpd.GeoDataFrame
    frame_block: gpd.GeoDataFrame
    centre: np.ndarray
    half_height: float
    plan_reach: float
    exchange_2d: pd.DataFrame
    scene_2d: dict
    patches_3d: pd.DataFrame
    exchange_3d: pd.DataFrame
    blockers: object
    summary: pd.DataFrame
    origin: np.ndarray = field(default=None)


def load_measure_scene(run_3d: Path, building_cache: Path, analysis_path: Path,
                       run_2d: Path, *, receiver_id, source_id,
                       plan_aspect: float, fire: str = "EATON",
                       patch_run: Path | None = None) -> MeasureScene:
    """Resolve the scene shared by the four measurement panels.

    ``plan_aspect`` is the width-to-height ratio of the plan axes, which sets
    how much of the block the panels reach across at equal map scale.
    """
    receiver = _resolve_building_id(building_cache, receiver_id)
    source = _resolve_building_id(building_cache, source_id)
    _, emitters, patches_3d, exchange_3d, _ = load_eaton_example(
        run_3d, building_cache, analysis_path, receiver_id=receiver,
        patch_run_dir=patch_run)
    emitter_ids = emitters.emitter.astype(int).tolist()
    scene_2d, summary = load_two_dimensional_exposure(
        run_2d, building_cache, analysis_path, receiver_id=receiver, fire=fire)

    footprints = gpd.read_parquet(
        building_cache,
        columns=["building_id", "geometry", "height_m", "base_z_m"],
    ).set_index("building_id")
    focal = footprints.loc[[receiver, *emitter_ids]]
    centre = np.array([focal.total_bounds[[0, 2]].mean(),
                       focal.total_bounds[[1, 3]].mean()])
    half_height = max(focal.total_bounds[3] - focal.total_bounds[1],
                      focal.total_bounds[2] - focal.total_bounds[0]) / 2 + 9
    plan_reach = half_height * plan_aspect
    in_window = footprints.cx[
        centre[0] - 2.2 * plan_reach:centre[0] + 2.2 * plan_reach,
        centre[1] - half_height:centre[1] + half_height]
    # Panel d draws the same context as a-c but frames on the focal group, so
    # the receiver and its emitters keep their visual weight.
    frame_block = in_window.loc[
        in_window.geometry.centroid.distance(
            footprints.loc[receiver].geometry.centroid).le(.72 * half_height)
        | in_window.index.isin([receiver, *emitter_ids])]
    exchange = scene_2d["exchange"]
    data = MeasureScene(
        receiver=receiver, source=source, emitter_ids=emitter_ids,
        footprints=footprints, in_window=in_window, frame_block=frame_block,
        centre=centre, half_height=half_height, plan_reach=plan_reach,
        exchange_2d=exchange.loc[exchange.emitter_destroyed], scene_2d=scene_2d,
        patches_3d=patches_3d, exchange_3d=exchange_3d,
        blockers=in_window.geometry.union_all(), summary=summary,
    )
    data.origin = np.array([centre[0], centre[1],
                            float(in_window.base_z_m.min())])
    return data


def _building_style(data: MeasureScene, building_id):
    """Fill, outline colour and outline weight for one building."""
    if building_id == data.receiver:
        return RECEIVER, INK, OUTLINE
    if building_id in data.emitter_ids:
        return SOURCE, INK, OUTLINE
    return CONTEXT, CONTEXT_EDGE, CONTEXT_OUTLINE


def _exchange_envelope(data: MeasureScene, emitter_id):
    """Plan envelope between one neighbour's facades and the receiver's.

    The hull of the two exchanging surfaces, with every footprint removed, so
    the corridor occupies only the open space the exchange crosses.
    """
    pairs = (data.exchange_2d.loc[data.exchange_2d.emitter.eq(emitter_id)]
             .nlargest(3, "exchange_pq_m"))
    patches = data.scene_2d["patches"]
    exchanging = patches.loc[patches.patch_id.isin(
        set(pairs.receiver_patch) | set(pairs.emitter_patch))]
    corners = _segment_endpoints(exchanging).reshape(-1, 2)
    return MultiPoint(corners).convex_hull.difference(data.blockers)


def _facade_quad(face, origin):
    """The four corners of one sampled facade patch, in local coordinates."""
    tangent = np.array([-face.ny, face.nx]) / np.hypot(face.nx, face.ny)
    offset = tangent * face.width_m / 2
    middle = np.array([face.x_m, face.y_m]) - origin[:2]
    low, high = face.z_low_m - origin[2], face.z_high_m - origin[2]
    return np.array([[*(middle - offset), low], [*(middle + offset), low],
                     [*(middle + offset), high], [*(middle - offset), high]])


def _prism(geometry, base_z, height, origin):
    """Wall quads and the roof polygon of an extruded footprint."""
    for polygon in _polygons(geometry):
        coords = np.asarray(polygon.exterior.coords)[:-1] - origin[:2]
        base = np.column_stack([coords, np.full(len(coords), base_z - origin[2])])
        top = base + [0, 0, height]
        yield ([[base[index], base[(index + 1) % len(base)],
                 top[(index + 1) % len(top)], top[index]]
                for index in range(len(base))], top)


def draw_plan_panel(ax, data: MeasureScene, measure: str | None = None):
    """Footprints in plan, optionally with one distance measurement drawn."""
    for building_id, row in data.in_window.iterrows():
        fill, edge, weight = _building_style(data, building_id)
        for polygon in _polygons(row.geometry):
            ax.add_patch(MplPolygon(
                np.asarray(polygon.exterior.coords), closed=True,
                facecolor=fill, edgecolor=edge, linewidth=weight,
                zorder=2 + (building_id == data.receiver)))
    if measure is not None:
        emitter = data.footprints.loc[data.source].geometry
        receiver = data.footprints.loc[data.receiver].geometry
        if measure == "surface":
            start, end = (np.asarray(point.coords[0])
                          for point in nearest_points(emitter, receiver))
        else:
            start = np.array([emitter.centroid.x, emitter.centroid.y])
            end = np.array([receiver.centroid.x, receiver.centroid.y])
        line = np.column_stack([start, end])
        ax.plot(*line, color="white", lw=MEASURE + 1.6, zorder=6)
        ax.plot(*line, color=INK, lw=MEASURE, zorder=7)
        ax.plot(*line, ls="", marker="o", ms=3.4, mfc="white", mec=INK,
                mew=.8, zorder=8)
    frame_plan_panel(ax, data)


def draw_perimeter_panel(ax, data: MeasureScene):
    """Exposed receiver perimeter, over the corridors that expose it."""
    for emitter_id in data.emitter_ids:
        for polygon in _polygons(_exchange_envelope(data, emitter_id)):
            ax.add_patch(MplPolygon(
                np.asarray(polygon.exterior.coords), closed=True,
                facecolor=CORRIDOR, edgecolor="none", alpha=.16, zorder=0))
    draw_plan_panel(ax, data)
    segments = data.scene_2d["receiver_patches"]
    for (start, end), row in zip(_segment_endpoints(segments),
                                 segments.itertuples()):
        exposed = row.exposed_m > 0
        ax.plot(*np.column_stack([start, end]),
                color=HIGH if exposed else QUIET,
                lw=EDGE if exposed else PERIMETER, solid_capstyle="butt",
                zorder=9 if exposed else 8)


def frame_plan_panel(ax, data: MeasureScene):
    """Equal-scale limits shared by the three plan panels."""
    ax.set(xlim=(data.centre[0] - data.plan_reach,
                 data.centre[0] + data.plan_reach),
           ylim=(data.centre[1] - data.half_height,
                 data.centre[1] + data.half_height))
    ax.set_aspect("equal")
    ax.axis("off")


def draw_exposure_panel(ax, data: MeasureScene, *, frame_aspect: float,
                        zoom: float = 2.05, elev: float = 34,
                        azim: float = -58):
    """The same footprints extruded, receiver facades shaded by exposure."""
    origin = data.origin
    for building_id, row in data.in_window.iterrows():
        if np.isnan(row.height_m):
            continue
        fill, edge, weight = _building_style(data, building_id)
        wall = tuple(np.asarray(to_rgb(fill)) * .88)
        contributor = (building_id in data.emitter_ids
                       or building_id == data.receiver)
        for walls, roof in _prism(row.geometry, row.base_z_m, row.height_m,
                                  origin):
            ax.add_collection3d(Poly3DCollection(
                walls, facecolors=[wall] * len(walls), edgecolors=edge,
                linewidths=weight * .6,
                zorder=10 if building_id == data.receiver else
                (3 if contributor else 2)))
            ax.add_collection3d(Poly3DCollection(
                [roof], facecolors=fill, edgecolors=edge,
                linewidths=weight * .6,
                zorder=12 if building_id == data.receiver else
                (3 if contributor else 2)))

    # Receiver facades carry accumulated exposure from the destroyed emitters.
    exchange = data.exchange_3d
    received = (exchange.assign(face=np.where(
        exchange.building_p.eq(data.receiver),
        exchange.patch_p, exchange.patch_q))
        .groupby("face").exchange_pq_m2.sum())
    faces = data.patches_3d[data.patches_3d.building_id.eq(data.receiver)].copy()
    faces["exposure"] = faces.patch_id.map(received).fillna(0.)
    peak = max(faces.exposure.max(), 1e-9)
    quads = [_facade_quad(face, origin) for face in faces.itertuples()]
    ax.add_collection3d(Poly3DCollection(
        quads, facecolors=RECEIVER, edgecolors=INK, linewidths=.35, alpha=1,
        zorder=13))
    for face, quad in zip(faces.itertuples(), quads):
        if face.exposure <= 0:
            continue
        ax.add_collection3d(Poly3DCollection(
            [quad], facecolors=SHADE(EXPOSURE_NORM(face.exposure / peak)),
            edgecolors=INK, linewidths=.35, alpha=.78, zorder=14))

    # One exchange volume per emitter, between its facade and the receiver
    # facade it illuminates, so the exchange reads as a surface pair.
    reaching = exchange.assign(emitter=np.where(
        exchange.building_p.eq(data.receiver),
        exchange.building_q, exchange.building_p))
    patches = data.patches_3d.set_index("patch_id")
    for pair in (reaching.sort_values("exchange_pq_m2", ascending=False)
                 .groupby("emitter").head(1).itertuples()):
        face_p, face_q = patches.loc[pair.patch_p], patches.loc[pair.patch_q]
        near, far = _facade_quad(face_p, origin), _facade_quad(face_q, origin)
        if (np.linalg.norm(near - far, axis=1).sum()
                > np.linalg.norm(near - far[[1, 0, 3, 2]], axis=1).sum()):
            far = far[[1, 0, 3, 2]]
        ax.add_collection3d(Poly3DCollection(
            [[near[index], near[(index + 1) % 4], far[(index + 1) % 4],
              far[index]] for index in range(4)],
            facecolors=CORRIDOR, edgecolors="none", alpha=.08, zorder=1))
        source_quad, target_quad = (
            (near, far) if int(face_p.building_id) == data.receiver
            else (far, near))
        ax.add_collection3d(Line3DCollection(
            [[source_quad.mean(axis=0), corner] for corner in target_quad],
            colors=CORRIDOR, linewidths=.65, linestyles=(0, (3, 2)),
            alpha=.82, zorder=15))
    relief = float((data.frame_block.base_z_m
                    + data.frame_block.height_m).max() - origin[2])
    minx, miny, maxx, maxy = data.frame_block.total_bounds
    block = np.array([(minx + maxx) / 2, (miny + maxy) / 2]) - origin[:2]
    span_y = max(maxy - miny, (maxx - minx) / frame_aspect) / 2 + 3
    span_x = span_y * frame_aspect
    ax.set(xlim=(block[0] - span_x, block[0] + span_x),
           ylim=(block[1] - span_y, block[1] + span_y), zlim=(0, relief))
    ax.set_box_aspect((2 * span_x, 2 * span_y,
                       max(relief * 1.9, span_y * .55)), zoom=zoom)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")


def _outward_edges(geometry):
    """Footprint edges with outward unit normals: start, end, normal."""
    for polygon in _polygons(geometry):
        coords = np.asarray(polygon.exterior.coords)[:, :2]
        for a, b in zip(coords[:-1], coords[1:]):
            length = float(np.hypot(*(b - a)))
            if length < 1e-6:
                continue
            normal = np.array([b[1] - a[1], a[0] - b[0]]) / length
            if polygon.covers(shapely.Point(*((a + b) / 2 + normal * .05))):
                normal = -normal
            yield a, b, normal, length


def _wall_points_2d(geometry, spacing):
    """Evenly spaced perimeter points: xy, outward normal, represented length."""
    rows = []
    for a, b, normal, length in _outward_edges(geometry):
        n = max(1, int(np.ceil(length / spacing)))
        for k in range(n):
            rows.append((*(a + (b - a) * (k + .5) / n), *normal, length / n))
    return np.asarray(rows, dtype=float).reshape(-1, 5)


def _wall_points_3d(geometry, base_z, height, spacing):
    """Evenly spaced facade points: xyz, outward normal (horizontal), area."""
    rows = []
    n_v = max(1, int(np.ceil(height / spacing)))
    for a, b, normal, length in _outward_edges(geometry):
        n_h = max(1, int(np.ceil(length / spacing)))
        for k in range(n_h):
            xy = a + (b - a) * (k + .5) / n_h
            for v in range(n_v):
                rows.append((*xy, base_z + height * (v + .5) / n_v, *normal, 0.,
                             (length / n_h) * (height / n_v)))
    return np.asarray(rows, dtype=float).reshape(-1, 7)


def _link_rgba(weight):
    """Link colour with opacity eased by its share of the strongest link."""
    alpha = LINK_ALPHA[0] + (LINK_ALPHA[1] - LINK_ALPHA[0]) * (
        weight / weight.max()) ** LINK_GAMMA
    rgba = np.tile(to_rgba(LINK), (len(weight), 1))
    rgba[:, 3] = alpha
    return rgba


def plan_links(data: MeasureScene, spacing: float = SPACING_2D):
    """2D links from each emitter's perimeter points to the receiver's.

    The line kernel cos(a)cos(b)/(2d) of the 2D model, kept where both points
    face each other and the sight line crosses no footprint by more than
    MIN_CROSSING (the 2D model's visibility rule).
    """
    if getattr(data, "_plan_links", None) is not None:
        return data._plan_links
    receiver = _wall_points_2d(data.footprints.loc[data.receiver].geometry, spacing)
    blockers = data.in_window.geometry.to_numpy()
    tree = shapely.STRtree(blockers)
    parts = []
    for emitter in data.emitter_ids:
        source = _wall_points_2d(data.footprints.loc[emitter].geometry, spacing)
        i, j = np.meshgrid(np.arange(len(source)), np.arange(len(receiver)), indexing="ij")
        i, j = i.ravel(), j.ravel()
        delta = receiver[j, :2] - source[i, :2]
        distance = np.hypot(delta[:, 0], delta[:, 1])
        cos_a = np.einsum("ij,ij->i", delta, source[i, 2:4]) / distance
        cos_b = -np.einsum("ij,ij->i", delta, receiver[j, 2:4]) / distance
        keep = (cos_a > 0) & (cos_b > 0) & (distance <= MAX_DISTANCE)
        i, j, distance, cos_a, cos_b = i[keep], j[keep], distance[keep], cos_a[keep], cos_b[keep]
        start = source[i, :2] + RAY_OFFSET * source[i, 2:4]
        end = receiver[j, :2] + RAY_OFFSET * receiver[j, 2:4]
        lines = shapely.linestrings(np.stack([start, end], axis=1))
        line_idx, poly_idx = tree.query(lines, predicate="intersects")
        blocked = np.zeros(len(lines), dtype=bool)
        crossing = shapely.length(shapely.intersection(lines[line_idx], blockers[poly_idx]))
        np.logical_or.at(blocked, line_idx[crossing >= MIN_CROSSING], True)
        visible = ~blocked
        kernel = cos_a * cos_b / (2 * distance)
        parts.append(pd.DataFrame({
            "emitter": emitter, "x0": start[:, 0], "y0": start[:, 1],
            "x1": end[:, 0], "y1": end[:, 1], "receiver_point": j,
            "kernel": kernel, "exchange": kernel * source[i, 4] * receiver[j, 4],
        })[visible])
    links = pd.concat(parts, ignore_index=True)
    data._plan_links = (links, receiver)
    return data._plan_links


def _prism_mesh(buildings, origin):
    """Triangle mesh of extruded footprints (walls and roofs), local frame."""
    vertices, faces = [], []
    for row in buildings.itertuples():
        if not np.isfinite(row.height_m):
            continue
        z0, z1 = row.base_z_m - origin[2], row.base_z_m + row.height_m - origin[2]
        for polygon in _polygons(row.geometry):
            ring = np.asarray(polygon.exterior.coords)[:, :2] - origin[:2]
            for a, b in zip(ring[:-1], ring[1:]):
                base = len(vertices)
                vertices += [(*a, z0), (*b, z0), (*b, z1), (*a, z1)]
                faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            for triangle in triangulate(polygon):
                if polygon.covers(triangle.representative_point()):
                    base = len(vertices)
                    vertices += [(x - origin[0], y - origin[1], z1)
                                 for x, y in np.asarray(triangle.exterior.coords)[:3]]
                    faces.append((base, base + 1, base + 2))
    return (np.ascontiguousarray(vertices, dtype=np.float32),
            np.ascontiguousarray(faces, dtype=np.int32))


def facade_links(data: MeasureScene, spacing: float = SPACING_3D):
    """3D links from each emitter's facade points to the receiver's.

    The view-factor kernel cos(a)cos(b)/(pi d^2), kept where the points face
    each other and a ray between them reaches the target unobstructed by any
    extruded building in the window.
    """
    if getattr(data, "_facade_links", None) is not None:
        return data._facade_links
    import point_cloud_utils as pcu

    origin = data.origin
    vertices, faces = _prism_mesh(data.in_window, origin)
    intersector = pcu.RayMeshIntersector(vertices, faces)
    row = data.footprints.loc[data.receiver]
    receiver = _wall_points_3d(row.geometry, row.base_z_m, row.height_m, spacing)
    receiver[:, :3] -= origin
    parts = []
    for emitter in data.emitter_ids:
        row = data.footprints.loc[emitter]
        source = _wall_points_3d(row.geometry, row.base_z_m, row.height_m, spacing)
        source[:, :3] -= origin
        i, j = np.meshgrid(np.arange(len(source)), np.arange(len(receiver)), indexing="ij")
        i, j = i.ravel(), j.ravel()
        delta = receiver[j, :3] - source[i, :3]
        distance = np.linalg.norm(delta, axis=1)
        cos_a = np.einsum("ij,ij->i", delta, source[i, 3:6]) / distance
        cos_b = -np.einsum("ij,ij->i", delta, receiver[j, 3:6]) / distance
        keep = (cos_a > 0) & (cos_b > 0) & (distance <= MAX_DISTANCE)
        i, j, cos_a, cos_b = i[keep], j[keep], cos_a[keep], cos_b[keep]
        start = source[i, :3] + RAY_OFFSET * source[i, 3:6]
        end = receiver[j, :3] + RAY_OFFSET * receiver[j, 3:6]
        ray = end - start
        length = np.linalg.norm(ray, axis=1)
        face_id, _, hit = intersector.intersect_rays(
            np.ascontiguousarray(start, dtype=np.float32),
            np.ascontiguousarray(ray / length[:, None], dtype=np.float32),
            ray_near=1e-4, ray_far=float(length.max() + .01))
        face_id, hit = np.asarray(face_id).ravel(), np.asarray(hit).ravel()
        visible = (face_id < 0) | (hit >= length - .01)
        kernel = cos_a * cos_b / (np.pi * length ** 2)
        parts.append(pd.DataFrame({
            "emitter": emitter, "x0": start[:, 0], "y0": start[:, 1], "z0": start[:, 2],
            "x1": end[:, 0], "y1": end[:, 1], "z1": end[:, 2], "receiver_point": j,
            "kernel": kernel, "exchange": kernel * source[i, 6] * receiver[j, 6],
        })[visible])
    links = pd.concat(parts, ignore_index=True)
    data._facade_links = (links, receiver)
    return data._facade_links


def _receiver_point_colours(links, receiver, size_column):
    """Exposure ramp for receiver points: exchange received per unit size."""
    received = links.groupby("receiver_point").exchange.sum().reindex(
        range(len(receiver))).fillna(0.).to_numpy() / receiver[:, size_column]
    peak = max(received.max(), 1e-12)
    return np.array([SHADE(EXPOSURE_NORM(value / peak)) if value > 0 else to_rgba(QUIET)
                     for value in received]), received > 0


def draw_perimeter_links_panel(ax, data: MeasureScene, spacing: float = SPACING_2D):
    """Panel c, point to point: perimeter samples and visible 2D links."""
    draw_plan_panel(ax, data)
    links, receiver = plan_links(data, spacing)
    links = links.sort_values("kernel")
    ax.add_collection(LineCollection(
        np.stack([links[["x0", "y0"]].to_numpy(), links[["x1", "y1"]].to_numpy()], axis=1),
        colors=_link_rgba(links.kernel.to_numpy()), linewidths=LINK_WIDTH, zorder=5))
    # Emitter points are drawn where they reach the receiver.
    sources = links[["x0", "y0"]].drop_duplicates().to_numpy()
    ax.scatter(sources[:, 0], sources[:, 1], s=1.6, color=INK, linewidths=0, zorder=9)
    colours, exposed = _receiver_point_colours(links, receiver, 4)
    ax.scatter(receiver[:, 0], receiver[:, 1], s=np.where(exposed, 3.2, 1.6),
               c=colours, linewidths=0, zorder=10)


def draw_exposure_links_panel(ax, data: MeasureScene, *, frame_aspect: float,
                              zoom: float = 2.05, elev: float = 34, azim: float = -58,
                              spacing: float = SPACING_3D):
    """Panel d, point to point: facade samples and visible 3D links."""
    origin = data.origin
    for building_id, row in data.in_window.iterrows():
        if np.isnan(row.height_m):
            continue
        fill, edge, weight = _building_style(data, building_id)
        wall = tuple(np.asarray(to_rgb(fill)) * .88)
        contributor = (building_id in data.emitter_ids
                       or building_id == data.receiver)
        for walls, roof in _prism(row.geometry, row.base_z_m, row.height_m, origin):
            ax.add_collection3d(Poly3DCollection(
                walls, facecolors=[wall] * len(walls), edgecolors=edge,
                linewidths=weight * .6, alpha=.8 if contributor else 1,
                zorder=10 if building_id == data.receiver else (3 if contributor else 2)))
            ax.add_collection3d(Poly3DCollection(
                [roof], facecolors=fill, edgecolors=edge, linewidths=weight * .6,
                alpha=.8 if contributor else 1,
                zorder=12 if building_id == data.receiver else (3 if contributor else 2)))
    links, receiver = facade_links(data, spacing)
    links = links.sort_values("kernel")
    ax.add_collection3d(Line3DCollection(
        np.stack([links[["x0", "y0", "z0"]].to_numpy(), links[["x1", "y1", "z1"]].to_numpy()], axis=1),
        colors=_link_rgba(links.kernel.to_numpy()), linewidths=LINK_WIDTH_3D,
        zorder=15))
    sources = links[["x0", "y0", "z0"]].drop_duplicates().to_numpy()
    ax.scatter(sources[:, 0], sources[:, 1], sources[:, 2], s=1.2, color=INK,
               linewidths=0, depthshade=False, zorder=16)
    colours, exposed = _receiver_point_colours(links, receiver, 6)
    ax.scatter(receiver[:, 0], receiver[:, 1], receiver[:, 2],
               s=np.where(exposed, 2.6, 1.2), c=colours, linewidths=0,
               depthshade=False, zorder=17)
    relief = float((data.frame_block.base_z_m + data.frame_block.height_m).max() - origin[2])
    minx, miny, maxx, maxy = data.frame_block.total_bounds
    block = np.array([(minx + maxx) / 2, (miny + maxy) / 2]) - origin[:2]
    span_y = max(maxy - miny, (maxx - minx) / frame_aspect) / 2 + 3
    span_x = span_y * frame_aspect
    ax.set(xlim=(block[0] - span_x, block[0] + span_x),
           ylim=(block[1] - span_y, block[1] + span_y), zlim=(0, relief))
    ax.set_box_aspect((2 * span_x, 2 * span_y, max(relief * 1.9, span_y * .55)), zoom=zoom)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")


def draw_measure_row(axes, data: MeasureScene, *, frame_aspect: float,
                     zoom: float = 2.05, links: bool = False):
    """Draw a-d into four positioned axes.

    ``links`` draws c-d point to point: evenly sampled wall points joined by
    every visible facing link, link opacity eased by view-factor density.
    """
    draw_plan_panel(axes[0], data, "centroid")
    draw_plan_panel(axes[1], data, "surface")
    if links:
        draw_perimeter_links_panel(axes[2], data)
        draw_exposure_links_panel(axes[3], data, frame_aspect=frame_aspect, zoom=zoom)
    else:
        draw_perimeter_panel(axes[2], data)
        draw_exposure_panel(axes[3], data, frame_aspect=frame_aspect, zoom=zoom)


def draw_exposure_key(fig, ax, *, swatch_y: float, caption_y: float | None,
                      swatch=(.028, .038), step: float = .092,
                      fontsize: float = 6.6):
    """Three-step key for the receiver facades, centred under one panel."""
    box = ax.get_position()
    left = box.x0 + .5 * (box.width - 3 * step + (step - swatch[0]))
    for index, (colour, label) in enumerate(
            [(LOW, "Low"), (MEDIUM, "Medium"), (HIGH, "High")]):
        x = left + index * step
        fig.add_artist(Rectangle(
            (x, swatch_y), *swatch, facecolor=colour, edgecolor="none",
            transform=fig.transFigure))
        fig.text(x + swatch[0] + .005, swatch_y + .010, label,
                 fontsize=fontsize, color=INK, va="baseline")
    if caption_y is not None:
        fig.text(box.x0 + .5 * box.width, caption_y,
                 "Accumulated exposure on receiver surfaces",
                 fontsize=fontsize, color="#5A5E63", ha="center",
                 va="baseline")
