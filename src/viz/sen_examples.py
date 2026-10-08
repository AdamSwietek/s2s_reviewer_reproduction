"""One SEN per quartile of the WUI size distribution, drawn as a sequence.

Quartiles are building-weighted over mapped-WUI buildings (Influence Zone,
Intermix, Interface): a quartile is the SEN size experienced by a quarter of
WUI buildings. Each panel shows a SEN at the median size of its quartile,
drawn with the links that formed it, and one focal building with its two
strongest pairwise couplings so the threshold rule can be read directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.viz.style import (
    WUI_INFLUENCE_COLOR, WUI_INTERFACE_COLOR, WUI_INTERMIX_COLOR,
)

WUI_CLASSES = ("Influence Zone", "Intermix", "Interface")
QUARTILE_COLORS = ("#7A7A76", "#4575B4", "#E6A817", "#C0392B")
ISOLATED_COLOR = "#A9A9A4"
CONNECTED_COLORS = ("#4575B4", "#41AB5D", "#E6A817", "#C0392B")
FOCAL = "#8E44AD"
DESTROYED_HATCH = "#5B1A12"
METHODS = {
    "top2": "Cumulative Top-2",
    "eeat": "EEAT",
    "distance": "Distance-only EEAT",
}


@dataclass
class SenExample:
    label: str
    size_range: str
    color: str
    members: gpd.GeoDataFrame
    links: set
    focal: int
    focal_top: pd.DataFrame
    focal_score: float
    focal_degree: int
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.members)


def incident_edges(run_dir: Path, ids) -> pd.DataFrame:
    """Every coupled pair touching ``ids``, scored as in the SEN builders."""
    edge_path = str(Path(run_dir) / "building_edges.parquet").replace("'", "''")
    con = duckdb.connect()
    con.register("ids", pd.DataFrame({"building_id": sorted(ids)}))
    try:
        return con.execute(f"""
            SELECT building_i, building_j,
                   LEAST(1.0, GREATEST(vf_i_to_j, vf_j_to_i)) AS F_ij
            FROM read_parquet('{edge_path}')
            WHERE building_i IN (SELECT building_id FROM ids)
               OR building_j IN (SELECT building_id FROM ids)
        """).fetchdf()
    finally:
        con.close()


def _incidence(edges: pd.DataFrame) -> pd.DataFrame:
    """Both directions, ranked by coupling with the builders' tie-break."""
    directed = pd.concat([
        edges.rename(columns={"building_i": "node", "building_j": "other"}),
        edges.rename(columns={"building_j": "node", "building_i": "other"}),
    ], ignore_index=True)
    directed = directed.sort_values(["node", "F_ij", "other"],
                                    ascending=[True, False, True])
    directed["rank"] = directed.groupby("node").cumcount() + 1
    return directed


def strongest(edges: pd.DataFrame, node: int, n: int = 2) -> pd.DataFrame:
    incidence = _incidence(edges)
    return incidence[incidence.node.eq(node)].head(n)


def method_links(edges, member_ids, threshold, method, *, n_neighbors=2,
                 edge_floor_fraction=0.75) -> set:
    """Links among ``member_ids`` selected by ``method``.

    ``edges`` must hold every pair touching the members, so each member's
    Top-N ranking sees all of its neighbors, as in the county builder.
    """
    if method in ("eeat", "distance"):
        inside = edges[edges.building_i.isin(member_ids)
                       & edges.building_j.isin(member_ids)
                       & edges.F_ij.ge(threshold)]
        return set(zip(inside.building_i, inside.building_j))
    if method != "top2":
        raise ValueError(f"Unknown method {method!r}")
    top = _incidence(edges)
    top = top[top["rank"].le(n_neighbors) & top.node.isin(member_ids)]
    totals = top.groupby("node").F_ij.agg(["size", "sum"])
    qualified = set(totals.index[totals["size"].eq(n_neighbors)
                                 & totals["sum"].ge(threshold)])
    chosen = top[top.node.isin(qualified) & top.other.isin(qualified)
                 & top.F_ij.ge(edge_floor_fraction * threshold)]
    return {(min(u, v), max(u, v)) for u, v in zip(chosen.node, chosen.other)}


def _size_class(label, sizes, color):
    low, high = int(sizes.min()), int(sizes.max())
    return dict(
        label=label, target=int(sizes.median()), low=low, high=high, color=color,
        size_range=(f"{low:,} building{'' if high == 1 else 's'}" if low == high
                    else f"{low:,}–{high:,} buildings"),
    )


def wui_size_quartiles(buildings: gpd.GeoDataFrame, split_isolated: bool = False):
    """Building-weighted quartiles of SEN size over mapped-WUI buildings."""
    return size_quartiles(
        buildings.loc[buildings.wui_class.isin(WUI_CLASSES), "component_size"],
        split_isolated=split_isolated)


def size_quartiles(sizes: pd.Series, split_isolated: bool = False):
    """Building-weighted quartile cuts of the SEN sizes given (one per building).

    With ``split_isolated``, isolated buildings form their own class and the
    quartiles are taken over connected WUI buildings only, for graphs where
    so many buildings are isolated that the lower quartiles would coincide.
    """
    sizes = pd.Series(sizes)
    classes = []
    if split_isolated:
        classes.append(_size_class("Isolated", sizes[sizes.eq(1)], ISOLATED_COLOR))
        sizes = sizes[sizes.gt(1)]
    cuts = sizes.quantile([.25, .5, .75]).to_numpy()
    bounds = [0, *cuts, np.inf]
    for index in range(4):
        inside = sizes[(sizes > bounds[index]) & (sizes <= bounds[index + 1])]
        if inside.empty:
            raise ValueError(
                f"Quartile Q{index + 1} is empty (cuts {cuts.tolist()}): too many "
                "ties; use split_isolated=True")
        palette = CONNECTED_COLORS if split_isolated else QUARTILE_COLORS
        classes.append(_size_class(f"Q{index + 1}", inside, palette[index]))
    return cuts, classes


def quartile_examples(buildings: gpd.GeoDataFrame, run_dir: Path,
                      threshold: float, method: str, window_lonlat, *,
                      seed: int = 20251001, n_neighbors: int = 2,
                      edge_floor_fraction: float = 0.75, quartiles=None,
                      edge_fn=None):
    """Pick one SEN at (or nearest) the median size of each WUI quartile.

    Candidates lie wholly inside the window, are at least 90% Interface or
    Intermix buildings, and do not touch the screen edge, where component
    sizes are censored. An isolated example must be a near miss: a neighbor
    coupled at no less than 0.4 F*, but a rule score below F*.

    ``edge_fn(ids)`` replaces the run's edge table, returning every pair
    touching ``ids`` as building_i, building_j, F_ij (for the distance rule,
    F_ij = linking distance / gap, so the threshold is 1).
    ``quartiles`` takes ``wui_size_quartiles`` output computed over a wider
    population (e.g. statewide), so ``buildings`` can be just a regional
    extract; only SENs with every member present in it are eligible.
    """
    cuts, quartiles = quartiles if quartiles is not None else wui_size_quartiles(buildings)
    # A lon/lat bounding box, or a polygon already in the buildings' CRS.
    window = (window_lonlat if isinstance(window_lonlat, shapely.Geometry)
              else gpd.GeoSeries([shapely.geometry.box(*window_lonlat)],
                                 crs=4326).to_crs(buildings.crs).iloc[0])
    xmin, ymin, xmax, ymax = window.bounds
    candidates = buildings.cx[xmin:xmax, ymin:ymax]
    in_window = candidates.geometry.representative_point().within(window)
    view_ids = set(candidates.loc[in_window.to_numpy(), "component_id"])
    pool = buildings[buildings.component_id.isin(view_ids)]
    components = (pool.assign(
        core_wui=pool.wui_class.isin(["Interface", "Intermix"]),
        in_window=pool.geometry.representative_point().within(window).to_numpy())
        .groupby("component_id")
        .agg(size=("component_size", "first"), loaded=("building_id", "size"),
             wui_share=("core_wui", "mean"), window_share=("in_window", "mean"),
             edge=("near_screen_edge", "max")))
    eligible = components[components.wui_share.ge(.9)
                          & components.window_share.eq(1) & ~components.edge
                          & components.loaded.eq(components["size"])]

    rng = np.random.default_rng(seed)
    examples = []
    for quartile in quartiles:
        within = eligible[eligible["size"].between(quartile["low"], quartile["high"])]
        gap = (within["size"] - quartile["target"]).abs()
        for gap_value in np.sort(gap.unique()):
            chosen = None
            for component_id in rng.permutation(within.index[gap.eq(gap_value)].to_numpy()):
                members = pool[pool.component_id.eq(component_id)]
                member_ids = set(members.building_id)
                edges = (edge_fn(member_ids) if edge_fn is not None
                         else incident_edges(run_dir, member_ids))
                incidence = _incidence(edges)
                if len(members) == 1:
                    top = incidence[incidence.node.eq(members.building_id.iloc[0])]
                    score = (top.F_ij.max() if method in ("eeat", "distance")
                             else top[top["rank"].le(2)].F_ij.sum())
                    if (len(top) < 2 or top.F_ij.max() < .4 * threshold
                            or score >= threshold):
                        continue
                chosen = (members, member_ids, edges, incidence)
                break
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError(f"No eligible {quartile['label']} SEN in the window")
        members, member_ids, edges, incidence = chosen
        links = method_links(edges, member_ids, threshold, method,
                             n_neighbors=n_neighbors,
                             edge_floor_fraction=edge_floor_fraction)
        top = incidence[incidence["rank"].le(2)]
        # The score each rule compares with F*: the strongest single coupling
        # for EEAT, the sum of the two strongest for cumulative Top-2.
        score = (top.groupby("node").F_ij.max() if method in ("eeat", "distance")
                 else top.groupby("node").F_ij.sum())
        score = score.reindex(sorted(member_ids)).fillna(0.)
        focal = int(score.sort_values(kind="stable").index[len(score) // 2])
        examples.append(SenExample(
            label=quartile["label"], size_range=quartile["size_range"],
            color=quartile["color"], members=members, links=links, focal=focal,
            focal_top=top[top.node.eq(focal)], focal_score=float(score[focal]),
            focal_degree=sum(focal in link for link in links),
        ))
    return cuts, examples


def _extent(frame, pad=45.):
    xmin, ymin, xmax, ymax = frame.total_bounds
    return (xmin + xmax) / 2, (ymin + ymax) / 2, max(xmax - xmin, ymax - ymin) / 2 + pad


def draw_example(ax, example: SenExample, buildings, wui, threshold, method,
                 *, title=None, distance_ft=None, extent=None, destroyed=None):
    centroid = buildings.set_index("building_id").geometry.centroid
    members = example.members
    cx, cy, half = extent or _extent(members)
    view = shapely.geometry.box(cx - half, cy - half, cx + half, cy + half)
    nearby = buildings.cx[cx - half:cx + half, cy - half:cy + half]
    for desc, color in [("Influence Zone", WUI_INFLUENCE_COLOR),
                        ("Intermix", WUI_INTERMIX_COLOR),
                        ("Interface", WUI_INTERFACE_COLOR)]:
        boundary = wui[wui.WUI_DESC.eq(desc)].boundary.intersection(view)
        boundary = boundary[~boundary.is_empty]
        if len(boundary):
            boundary.plot(ax=ax, color=color, lw=.8, alpha=.7, zorder=1)
    nearby[~nearby.building_id.isin(members.building_id)].plot(
        ax=ax, color="#D9D9D5", edgecolor="#B5B5B0", lw=.3, zorder=2)
    members.plot(ax=ax, color=example.color, alpha=.75, edgecolor="#333333",
                 lw=.4, zorder=3)
    for u, v in example.links:
        (x0, y0), (x1, y1) = centroid[u].coords[0], centroid[v].coords[0]
        ax.plot([x0, x1], [y0, y1], color="#222222", lw=.9, zorder=4)
    for order, row in enumerate(example.focal_top.itertuples()):
        (x0, y0), (x1, y1) = (centroid[example.focal].coords[0],
                              centroid[row.other].coords[0])
        ax.plot([x0, x1], [y0, y1], color=FOCAL, lw=1.3, ls=(0, (3, 1.5)), zorder=5)
        # Past the midpoint and nudged to opposite sides of each link, so the
        # two labels stay apart even for short links in similar directions.
        length = max(np.hypot(x1 - x0, y1 - y0), 1e-9)
        nudge = (1 if order == 0 else -1) * .035 * half
        label = (f"{distance_ft / row.F_ij:.0f} ft" if method == "distance"
                 else f"{row.F_ij:.3f}")
        ax.text(x0 + .62 * (x1 - x0) - nudge * (y1 - y0) / length,
                y0 + .62 * (y1 - y0) + nudge * (x1 - x0) / length, label,
                fontsize=5.5,
                color=FOCAL, ha="center", va="center", zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=.4, alpha=.8))
    if destroyed is not None:
        lost = nearby[nearby.building_id.isin(destroyed)]
        if len(lost):
            lost.plot(ax=ax, facecolor="none", edgecolor=DESTROYED_HATCH, hatch="//////",
                      lw=0, zorder=3.5)
    members[members.building_id.eq(example.focal)].plot(
        ax=ax, facecolor="none", edgecolor=FOCAL, lw=1.6, zorder=6)
    ax.plot([cx - half + 10, cx - half + 60], [cy - half + 10] * 2,
            color="#111111", lw=1.6, zorder=8)
    ax.text(cx - half + 35, cy - half + 14, "50 m", ha="center", va="bottom",
            fontsize=6, zorder=8)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
        spine.set_linewidth(.5)
    connected = ("isolated building" if example.size == 1
                 else f"SEN of {example.size} buildings")
    ax.set_title(title or f"{example.label} · {example.size_range}\n{connected}",
                 loc="left", fontsize=8.5, fontweight="bold")
    lonlat = gpd.GeoSeries([shapely.geometry.Point(cx, cy)],
                           crs=buildings.crs).to_crs(4326).iloc[0]
    links = f"{example.focal_degree} link{'' if example.focal_degree == 1 else 's'}"
    if method == "distance":
        nearest = distance_ft / example.focal_score
        rule = (f"focal nearest neighbor {nearest:.0f} ft "
                f"{'≤' if nearest <= distance_ft else '>'} {distance_ft:g} ft · {links}")
    else:
        score_name = ("strongest coupling" if method == "eeat"
                      else "top-2 coupling sum")
        verdict = "≥" if example.focal_score >= threshold else "<"
        rule = (f"focal {score_name} {example.focal_score:.3f} {verdict} "
                f"F* = {threshold:.3f} · {links}")
    ax.text(.01, .99,
            f"{members.wui_class.mode().iloc[0]} · {lonlat.y:.4f}°N, "
            f"{-lonlat.x:.4f}°W\n{rule}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.5),
            zorder=9)


def plot_size_sequence(examples, buildings, wui, threshold, method, *,
                       rule: str, edge_floor_fraction: float = 0.75,
                       title: str | None = None, distance_ft: float | None = None):
    """Size classes left to right, so connectivity reads as a sequence."""
    fig, axes = plt.subplots(1, len(examples), figsize=(3.55 * len(examples), 4.55))
    for ax, example in zip(np.atleast_1d(axes), examples):
        draw_example(ax, example, buildings, wui, threshold, method,
                     distance_ft=distance_ft)
    if method == "distance":
        link_label = f"link (footprints ≤ {distance_ft:g} ft apart)"
    elif method == "eeat":
        link_label = "EEAT link (F_ij ≥ F*)"
    else:
        link_label = f"Top-2 link (both ends qualify, F_ij ≥ {edge_floor_fraction:g} F*)"
    focal_label = ("focal building's two nearest neighbors" if method == "distance"
                   else "focal building's two strongest couplings (F_ij)")
    handles = [
        Patch(facecolor="#9A9A96", alpha=.75, edgecolor="#333333",
              label="SEN members (color by quartile)"),
        Patch(facecolor="#D9D9D5", edgecolor="#B5B5B0", label="other buildings"),
        Line2D([0], [0], color="#222222", lw=.9, label=link_label),
        Line2D([0], [0], color=FOCAL, lw=1.3, ls=(0, (3, 1.5)), label=focal_label),
        Line2D([0], [0], color=WUI_INTERFACE_COLOR, lw=.8, label="Interface"),
        Line2D([0], [0], color=WUI_INTERMIX_COLOR, lw=.8, label="Intermix"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=6.4, frameon=False, bbox_to_anchor=(.5, .005))
    fig.suptitle(title or (f"{METHODS[method]} SEN examples at the median size "
                           f"of each WUI quartile ({rule}; F* = {threshold:.4f}, P50)"),
                 x=.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, .06, 1, .955), w_pad=1.2)
    return fig


def example_for_focal(buildings, focal, threshold, method, *, run_dir=None,
                      edge_fn=None, label="", size_range="", color=QUARTILE_COLORS[1],
                      n_neighbors=2, edge_floor_fraction=0.75) -> SenExample:
    """The SEN containing ``focal`` under one rule, drawn around that building.

    Used to show the same focal building under different link rules; the
    focal building is fixed rather than chosen by its rule score.
    """
    component = buildings.loc[buildings.building_id.eq(focal), "component_id"].iloc[0]
    members = buildings[buildings.component_id.eq(component)]
    member_ids = set(members.building_id)
    edges = (edge_fn(member_ids) if edge_fn is not None
             else incident_edges(run_dir, member_ids))
    links = method_links(edges, member_ids, threshold, method, n_neighbors=n_neighbors,
                         edge_floor_fraction=edge_floor_fraction)
    incidence = _incidence(edges)
    focal_top = incidence[incidence.node.eq(focal) & incidence["rank"].le(2)]
    score = (0.0 if focal_top.empty else
             float(focal_top.F_ij.max() if method in ("eeat", "distance")
                   else focal_top.F_ij.sum()))
    return SenExample(label=label, size_range=size_range, color=color, members=members,
                      links=links, focal=int(focal), focal_top=focal_top,
                      focal_score=score, focal_degree=sum(focal in link for link in links))


def focal_comparison_extents(rows):
    """One shared map extent per focal-building column across all rows."""
    n_cols = len(rows[0]["examples"])
    if any(len(row["examples"]) != n_cols for row in rows):
        raise ValueError("Every comparison row must contain the same columns")
    return [
        _extent(pd.concat([row["examples"][col].members for row in rows]))
        for col in range(n_cols)
    ]


def plot_focal_comparison(rows, wui, *, title, distance_ft=None, destroyed=None,
                          extents=None):
    """One column per focal building, one row per link rule, each column on a
    shared extent so the SENs can be compared building for building.

    ``rows``: dicts with name, examples, buildings, threshold and method.
    """
    n_cols = len(rows[0]["examples"])
    extents = focal_comparison_extents(rows) if extents is None else extents
    if len(extents) != n_cols:
        raise ValueError("extents must contain one extent per comparison column")
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(3.55 * n_cols, 4.35 * len(rows)), squeeze=False)
    for col in range(n_cols):
        extent = extents[col]
        for r, row in enumerate(rows):
            example = row["examples"][col]
            size = ("isolated building" if example.size == 1
                    else f"SEN of {example.size} buildings")
            draw_example(axes[r, col], example, row["buildings"], wui, row["threshold"],
                         row["method"], distance_ft=distance_ft, extent=extent,
                         destroyed=destroyed,
                         title=f"{example.label} · {row['name']}\n{size}")
    handles = [
        Patch(facecolor="#9A9A96", alpha=.75, edgecolor="#333333", label="SEN members"),
        Patch(facecolor="#D9D9D5", edgecolor="#B5B5B0", label="other buildings"),
        Line2D([0], [0], color="#222222", lw=.9, label="link under that row's rule"),
        Line2D([0], [0], color=FOCAL, lw=1.3, ls=(0, (3, 1.5)),
               label="focal building's two nearest / strongest neighbors"),
        Line2D([0], [0], color=WUI_INTERFACE_COLOR, lw=.8, label="Interface"),
        Line2D([0], [0], color=WUI_INTERMIX_COLOR, lw=.8, label="Intermix"),
    ]
    if destroyed is not None:
        handles.insert(2, Patch(facecolor="white", edgecolor=DESTROYED_HATCH,
                                hatch="//////", lw=0, label="destroyed (DINS)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=6.4,
               frameon=False, bbox_to_anchor=(.5, .005))
    fig.suptitle(title, x=.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, .035, 1, .965), w_pad=1.2, h_pad=2.6)
    return fig
