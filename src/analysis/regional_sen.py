"""Regional Structure Exposure Network screening."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse


CORRIDOR_RELATIVE = Path("nx") / "samo_silverlake"
WUI_PRIORITY = {"Interface": 3, "Intermix": 2, "Influence Zone": 1}


def audit_corridor_inputs(project_root: Path) -> pd.DataFrame:
    """Summarize the completed high-compute tiled reconstruction."""
    project_root = Path(project_root)
    base = project_root / "data" / CORRIDOR_RELATIVE
    manifest = pd.read_csv(base / "tile_manifest.csv")
    buildings = gpd.read_parquet(base / "corridor_buildings.parquet")
    pairs = pd.read_parquet(base / "corridor_geom_pairs.parquet")
    corridor = gpd.read_parquet(base / "corridor.parquet")
    return pd.DataFrame([{
        "corridor": "Santa Monica Mountains–Silver Lake",
        "corridor_area_km2": float(corridor.geometry.area.sum() / 1e6),
        "tiles_total": len(manifest),
        "tiles_complete": int(manifest.status.eq("complete").sum()),
        "buildings": len(buildings),
        "candidate_building_pairs": len(pairs),
        "maximum_pair_distance_m": 100.0,
        "tile_core_m": 2500.0,
        "tile_halo_m": float(manifest.halo_m.iloc[0]),
    }])


def _wui_classes(points: gpd.GeoDataFrame, project_root: Path) -> pd.Series:
    """Assign one CAL FIRE WUI class, prioritizing Interface if layers overlap."""
    wui = (gpd.read_file(Path(project_root) / "data" / "calfire_wui_la.gpkg")
           .to_crs(points.crs))
    joined = gpd.sjoin(
        points[["node", "geometry"]], wui[["WUI_DESC", "geometry"]],
        how="left", predicate="within",
    )
    joined["priority"] = joined.WUI_DESC.map(WUI_PRIORITY).fillna(0)
    classes = (joined.sort_values("priority")
               .drop_duplicates("node", keep="last")
               .set_index("node").WUI_DESC
               .reindex(points.node))
    return classes.fillna("Outside mapped WUI").reset_index(drop=True)


def build_regional_sen(project_root: Path, threshold: float,
                        max_spine_distance_m: float | None = None) -> dict:
    """Build the corridor-wide SEN from frozen building and pair products."""
    project_root = Path(project_root)
    base = project_root / "data" / CORRIDOR_RELATIVE
    buildings = gpd.read_parquet(base / "corridor_buildings.parquet")
    pairs = pd.read_parquet(base / "corridor_geom_pairs.parquet")
    corridor = gpd.read_parquet(base / "corridor.parquet")
    interface_spine = gpd.read_parquet(base / "interface_boundary.parquet")
    buildings["LARIAC_BLD_ID"] = buildings.LARIAC_BLD_ID.astype(str)
    pairs[["bld_a", "bld_b"]] = pairs[["bld_a", "bld_b"]].astype(str)
    if max_spine_distance_m is not None:
        if max_spine_distance_m <= 0:
            raise ValueError("max_spine_distance_m must be positive")
        spine = interface_spine.geometry.union_all()
        keep = buildings.geometry.representative_point().distance(
            spine
        ).le(max_spine_distance_m)
        buildings = buildings.loc[keep].reset_index(drop=True)
        keep_ids = set(buildings.LARIAC_BLD_ID)
        pairs = pairs.loc[
            pairs.bld_a.isin(keep_ids) & pairs.bld_b.isin(keep_ids)
        ].reset_index(drop=True)

    node_index = pd.Series(
        np.arange(len(buildings), dtype=np.int64),
        index=buildings.LARIAC_BLD_ID,
    )
    active = pairs.loc[pairs.w.ge(threshold)].copy()
    active["u"] = active.bld_a.map(node_index)
    active["v"] = active.bld_b.map(node_index)
    active = active.dropna(subset=["u", "v"])
    active[["u", "v"]] = active[["u", "v"]].astype(np.int64)
    u, v = active.u.to_numpy(), active.v.to_numpy()
    graph = sparse.coo_matrix(
        (np.ones(2 * len(active), dtype=np.int8),
         (np.r_[u, v], np.r_[v, u])),
        shape=(len(buildings), len(buildings)),
    ).tocsr()
    n_components, labels = sparse.csgraph.connected_components(
        graph, directed=False,
    )
    sizes = np.bincount(labels)
    stable_ids = pd.Series(buildings.LARIAC_BLD_ID).groupby(labels).min()
    buildings["component_id"] = labels
    buildings["sen_id"] = stable_ids.loc[labels].to_numpy()
    buildings["component_size"] = sizes[labels]

    points = gpd.GeoDataFrame(
        {"node": np.arange(len(buildings), dtype=np.int64)},
        geometry=buildings.geometry.representative_point(), crs=buildings.crs,
    )
    buildings["wui_class"] = _wui_classes(points, project_root).to_numpy()
    buildings["is_interface"] = buildings.wui_class.eq("Interface")
    if max_spine_distance_m is None:
        screen_geometry = corridor.geometry.union_all()
    else:
        screen_geometry = interface_spine.geometry.union_all().buffer(
            max_spine_distance_m
        )
    corridor_boundary = screen_geometry.boundary
    buildings["near_screen_edge"] = points.geometry.distance(
        corridor_boundary
    ).le(50).to_numpy()

    node_frame = pd.DataFrame({
        "component_id": labels,
        "component_size": sizes[labels],
        "is_interface": buildings.is_interface.to_numpy(),
        "near_screen_edge": buildings.near_screen_edge.to_numpy(),
        "x": points.geometry.x.to_numpy(),
        "y": points.geometry.y.to_numpy(),
    })
    components = (node_frame.groupby("component_id", as_index=False)
                  .agg(component_size=("component_size", "first"),
                       interface_buildings=("is_interface", "sum"),
                       near_screen_edge=("near_screen_edge", "max"),
                       xmin=("x", "min"), xmax=("x", "max"),
                       ymin=("y", "min"), ymax=("y", "max")))
    components["other_buildings"] = (
        components.component_size - components.interface_buildings
    )
    components["spans_interface_class"] = (
        components.interface_buildings.gt(0)
        & components.other_buildings.gt(0)
    )
    components["planar_span_km"] = np.hypot(
        components.xmax - components.xmin,
        components.ymax - components.ymin,
    ) / 1000
    component_lookup = components.set_index("component_id")
    buildings["spans_interface_class"] = component_lookup.loc[
        labels, "spans_interface_class"
    ].to_numpy(bool)
    buildings["network_category"] = np.select(
        [buildings.component_size.eq(1), buildings.spans_interface_class],
        ["Isolated", "Interface-spanning SEN"],
        default="Other connected SEN",
    )

    # Attach mean within-component SSD for interpretable bond separation.
    active_roots = labels[active.u.to_numpy()]
    mean_ssd = pd.Series(active.ssd_m.to_numpy()).groupby(active_roots).mean()
    components["mean_bond_ssd_ft"] = (
        components.component_id.map(mean_ssd) * 3.28084
    )

    spanning = components[components.spans_interface_class]
    connected = components[components.component_size.gt(1)]
    summary = pd.DataFrame([{
        "corridor": "Santa Monica Mountains–Silver Lake",
        "screen_buffer_m": (float(max_spine_distance_m)
                            if max_spine_distance_m is not None else np.nan),
        "F_ij_threshold": float(threshold),
        "P_destroyed_equivalent": .50,
        "buildings": len(buildings),
        "candidate_pairs": len(pairs),
        "active_bonds": len(active),
        "SENs": n_components,
        "connected_SENs": len(connected),
        "connected_buildings": int(buildings.component_size.gt(1).sum()),
        "connected_building_share": float(buildings.component_size.gt(1).mean()),
        "largest_SEN": int(sizes.max()),
        "interface_buildings": int(buildings.is_interface.sum()),
        "interface_spanning_SENs": len(spanning),
        "buildings_in_interface_spanning_SENs": int(spanning.component_size.sum()),
        "share_in_interface_spanning_SENs": float(
            spanning.component_size.sum() / len(buildings)
        ),
        "largest_interface_spanning_SEN": int(spanning.component_size.max()),
        "maximum_interface_spanning_span_km": float(spanning.planar_span_km.max()),
        "SENs_at_least_100_buildings": int(components.component_size.ge(100).sum()),
        "buildings_in_SENs_at_least_100": int(
            components.loc[components.component_size.ge(100), "component_size"].sum()
        ),
        "SENs_at_least_1000_buildings": int(components.component_size.ge(1000).sum()),
        "buildings_in_SENs_at_least_1000": int(
            components.loc[components.component_size.ge(1000), "component_size"].sum()
        ),
        "edge_censored_interface_spanning_SENs": int(
            spanning.near_screen_edge.sum()
        ),
    }])
    return {
        "buildings": buildings,
        "pairs": pairs,
        "active_edges": active,
        "components": components,
        "corridor": corridor,
        "interface_spine": interface_spine,
        "summary": summary,
    }


def regional_size_profile(components: pd.DataFrame) -> pd.DataFrame:
    """Cumulative share of buildings contained in SENs above each size."""
    total = int(components.component_size.sum())
    cutoffs = np.unique(np.r_[
        1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
        int(components.component_size.max()),
    ])
    rows = []
    for cutoff in cutoffs:
        retained = components.component_size.ge(cutoff)
        buildings = int(components.loc[retained, "component_size"].sum())
        rows.append({
            "minimum_SEN_size": int(cutoff),
            "SENs": int(retained.sum()),
            "buildings": buildings,
            "building_share": buildings / total,
        })
    return pd.DataFrame(rows)


def regional_category_summary(buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """Summarize isolated, other-connected and Interface-spanning buildings."""
    order = ["Isolated", "Other connected SEN", "Interface-spanning SEN"]
    counts = buildings.network_category.value_counts().reindex(order, fill_value=0)
    return pd.DataFrame({
        "network_category": order,
        "buildings": counts.to_numpy(int),
        "building_share": counts.to_numpy() / len(buildings),
    })
