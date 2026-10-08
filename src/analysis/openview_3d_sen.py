"""Adapters from an OpenView regional 3D BVF run to reproduction SEN analyses."""
from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import CRS, Transformer
from scipy import sparse
from scipy.spatial import cKDTree
from shapely.geometry import box


WUI_PRIORITY = {"Interface": 3, "Intermix": 2, "Influence Zone": 1}
WUI_BUFFER_M = 3_218.688
WUI_SIMPLIFY_M = 25.0


def read_wui_for_bounds(wui_path: Path, bounds, bounds_crs) -> gpd.GeoDataFrame:
    """Read and clip a GeoParquet or GeoPackage WUI layer to data bounds."""
    wui_path = Path(wui_path)
    if wui_path.suffix.lower() in {".parquet", ".pq"}:
        geo_metadata = json.loads(
            pq.read_metadata(wui_path).metadata[b"geo"].decode("utf-8")
        )
        primary = geo_metadata["primary_column"]
        wui_crs = CRS.from_json_dict(
            geo_metadata["columns"][primary]["crs"]
        )
        transformer = Transformer.from_crs(
            bounds_crs, wui_crs, always_xy=True
        )
        source_bbox = transformer.transform_bounds(*bounds, densify_pts=21)
        try:
            wui = gpd.read_parquet(
                wui_path, columns=["WUI_DESC", primary], bbox=source_bbox,
            )
        except ValueError as error:
            # Some ArcGIS exports lack the GeoParquet ``covering`` column
            # required for Arrow bbox predicate pushdown.
            if "Specifying 'bbox' not supported" not in str(error):
                raise
            wui = gpd.read_parquet(
                wui_path, columns=["WUI_DESC", primary]
            )
            wui = wui.cx[
                source_bbox[0]:source_bbox[2],
                source_bbox[1]:source_bbox[3],
            ]
    else:
        import pyogrio

        info = pyogrio.read_info(wui_path)
        wui_crs = CRS.from_user_input(info["crs"])
        transformer = Transformer.from_crs(
            bounds_crs, wui_crs, always_xy=True
        )
        source_bbox = transformer.transform_bounds(*bounds, densify_pts=21)
        wui = pyogrio.read_dataframe(
            wui_path, columns=["WUI_DESC"], bbox=source_bbox,
            use_arrow=True,
        )
    wui = wui.to_crs(bounds_crs)
    wui = wui.loc[wui.WUI_DESC.isin(WUI_PRIORITY)].copy()
    # Keep complete intersecting polygons. Cutting them to the rectangle would
    # create artificial straight segments when their boundaries are plotted;
    # the map axes provide the exact visual crop instead.
    return wui.loc[wui.geometry.intersects(box(*bounds))].reset_index(drop=True)


def _manifest(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    required = [
        run_dir / "manifest.json",
        run_dir / "building_ids.parquet",
        run_dir / "building_edges.parquet",
        run_dir / "building_metrics.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete OpenView 3D run; missing: {missing}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("model") != "differential_area_3d":
        raise ValueError(
            f"Expected differential_area_3d, found {manifest.get('model')!r}"
        )
    return manifest


def audit_openview_3d_run(run_dir: Path) -> pd.DataFrame:
    """Report the finalized county graph products used by the new notebooks."""
    run_dir = Path(run_dir)
    manifest = _manifest(run_dir)
    con = duckdb.connect()
    try:
        buildings = con.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(run_dir / "building_ids.parquet")],
        ).fetchone()[0]
        edges = con.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(run_dir / "building_edges.parquet")],
        ).fetchone()[0]
    finally:
        con.close()
    cfg = manifest["config"]
    return pd.DataFrame([{
        "model": manifest["model"],
        "tiles": manifest["tile_count"],
        "buildings": buildings,
        "candidate_pairs": edges,
        "maximum_pair_distance_m": cfg["max_distance_m"],
        "wui_buffer_m": WUI_BUFFER_M,
        "terrain_resolution_m": cfg.get("terrain_resolution_m"),
        "source": manifest["input"],
    }])


def _edge_query(run_dir: Path, node_map: pd.DataFrame,
                threshold: float | None = None) -> pd.DataFrame:
    """Read only graph edges whose endpoints occur in ``node_map``."""
    edge_path = str(Path(run_dir) / "building_edges.parquet")
    predicate = "" if threshold is None else (
        f"WHERE GREATEST(e.vf_i_to_j, e.vf_j_to_i) >= {float(threshold)}"
    )
    con = duckdb.connect()
    con.register("node_map", node_map)
    try:
        edges = con.execute(f"""
            SELECT left_node.BLD_ID AS bld_a,
                   right_node.BLD_ID AS bld_b,
                   left_node.node::BIGINT AS u,
                   right_node.node::BIGINT AS v,
                   LEAST(1.0, GREATEST(e.vf_i_to_j, e.vf_j_to_i)) AS F_ij,
                   CAST(NULL AS DOUBLE) AS ssd_m,
                   e.has_contact
            FROM read_parquet('{edge_path.replace("'", "''")}') e
            JOIN node_map left_node ON e.building_i = left_node.building_id
            JOIN node_map right_node ON e.building_j = right_node.building_id
            {predicate}
        """).fetchdf()
    finally:
        con.unregister("node_map")
        con.close()
    edges[["u", "v"]] = edges[["u", "v"]].astype(np.int64)
    return edges


def _cumulative_top_n_edge_query(
    run_dir: Path, node_map: pd.DataFrame, threshold: float,
    n_neighbors: int, edge_floor_fraction: float = 0.25,
    restrict_to_node_map: bool = False,
) -> pd.DataFrame:
    """Select induced cumulative Top-N links without loading every edge.

    Each node qualifies when it has at least ``n_neighbors`` incident links
    and the sum of its N strongest undirected pair couplings reaches
    ``threshold``. The returned graph is the union of those selected Top-N
    links, restricted to links whose two endpoints qualify and whose
    individual coupling reaches ``edge_floor_fraction * threshold``.
    With ``restrict_to_node_map``, pairs touching a building absent from
    ``node_map`` are dropped before ranking, so an excluded structure can
    neither be selected as a neighbor nor qualify a node.
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least one")
    if not 0 <= edge_floor_fraction <= 1:
        raise ValueError("edge_floor_fraction must lie between zero and one")
    edge_path = str(Path(run_dir) / "building_edges.parquet")
    escaped = edge_path.replace("'", "''")
    membership = (
        "WHERE building_i IN (SELECT building_id FROM node_map) "
        "AND building_j IN (SELECT building_id FROM node_map)"
        if restrict_to_node_map else ""
    )
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.register("node_map", node_map)
    try:
        con.execute(f"""
            CREATE TEMP TABLE top_n_incidence AS
            WITH pair_scores AS (
                SELECT building_i, building_j,
                       LEAST(1.0, GREATEST(vf_i_to_j, vf_j_to_i)) AS F_ij,
                       has_contact
                FROM read_parquet('{escaped}')
                {membership}
            ),
            incidence AS (
                SELECT building_i AS node_id, building_j AS neighbor_id,
                       F_ij, has_contact
                FROM pair_scores
                UNION ALL
                SELECT building_j AS node_id, building_i AS neighbor_id,
                       F_ij, has_contact
                FROM pair_scores
            )
            SELECT node_id, neighbor_id, F_ij, has_contact
            FROM incidence
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY node_id ORDER BY F_ij DESC, neighbor_id
            ) <= {int(n_neighbors)}
        """)
        con.execute(f"""
            CREATE TEMP TABLE qualified_nodes AS
            SELECT node_id
            FROM top_n_incidence
            GROUP BY node_id
            HAVING COUNT(*) = {int(n_neighbors)}
               AND SUM(F_ij) >= {float(threshold)}
        """)
        edges = con.execute(f"""
            WITH selected AS (
                SELECT LEAST(i.node_id, i.neighbor_id) AS building_i,
                       GREATEST(i.node_id, i.neighbor_id) AS building_j,
                       MAX(i.F_ij) AS F_ij,
                       BOOL_OR(i.has_contact) AS has_contact
                FROM top_n_incidence i
                JOIN qualified_nodes focal ON i.node_id = focal.node_id
                JOIN qualified_nodes neighbor
                  ON i.neighbor_id = neighbor.node_id
                WHERE i.F_ij >= {float(edge_floor_fraction * threshold)}
                GROUP BY 1, 2
            )
            SELECT left_node.BLD_ID AS bld_a,
                   right_node.BLD_ID AS bld_b,
                   left_node.node::BIGINT AS u,
                   right_node.node::BIGINT AS v,
                   selected.F_ij,
                   CAST(NULL AS DOUBLE) AS ssd_m,
                   selected.has_contact
            FROM selected
            JOIN node_map left_node
              ON selected.building_i = left_node.building_id
            JOIN node_map right_node
              ON selected.building_j = right_node.building_id
        """).fetchdf()
    finally:
        con.unregister("node_map")
        con.close()
    edges[["u", "v"]] = edges[["u", "v"]].astype(np.int64)
    return edges


def build_fire_network_from_openview_3d(
    project_root: Path, run_dir: Path, fire: str, *,
    restrict_to_perimeter: bool = False,
) -> dict:
    """Subset the county graph to a fire scene and attach DINS-compatible IDs.

    By default the complete fire scene is retained, so every assessed DINS
    building present in the regional graph can receive an outcome label and
    unassessed buildings can remain structural connectors. The historical
    within-perimeter population is available only as an explicit option.
    """
    project_root, run_dir = Path(project_root), Path(run_dir)
    manifest = _manifest(run_dir)
    ids = pd.read_parquet(
        run_dir / "building_ids.parquet",
        columns=["building_id", "source_building_id"],
    ).rename(columns={"source_building_id": "BLD_ID"})
    ids["BLD_ID"] = ids.BLD_ID.astype(str)

    scene = gpd.read_parquet(
        project_root / "data" / "nx" / f"{fire}_buildings.parquet",
        columns=["BLD_ID", "geometry"],
    )
    scene["BLD_ID"] = scene.BLD_ID.astype(str)
    scene = scene.drop_duplicates("BLD_ID")
    if restrict_to_perimeter:
        perimeters = gpd.read_parquet(
            project_root / "data" / "nx" / "fire_perims.parquet"
        ).set_index("FIRE_NAME")
        perimeter = gpd.GeoSeries(
            [perimeters.loc[fire].geometry], crs=perimeters.crs
        ).to_crs(scene.crs).iloc[0]
        keep = scene.geometry.representative_point().within(perimeter)
        scene = scene.loc[keep]
    nodes = scene.merge(ids, on="BLD_ID", how="inner")
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=scene.crs)
    nodes = nodes.reset_index(drop=True)
    nodes["node"] = np.arange(len(nodes), dtype=np.int64)
    node_map = nodes[["building_id", "BLD_ID", "node"]].copy()
    edges = _edge_query(run_dir, node_map)
    return {
        "fire": fire,
        "nodes": nodes,
        "edges": edges,
        "max_distance_m": manifest["config"]["max_distance_m"],
        "source_run": str(run_dir),
        "restrict_to_perimeter": restrict_to_perimeter,
        "cached": False,
    }


def build_fire_networks_from_openview_3d(
    project_root: Path, run_dir: Path, fires, *,
    restrict_to_perimeter: bool = False,
) -> dict:
    return {
        fire: build_fire_network_from_openview_3d(
            project_root, run_dir, fire,
            restrict_to_perimeter=restrict_to_perimeter,
        )
        for fire in fires
    }


def load_county_buildings(
    run_dir: Path, cache_path: Path, *, force: bool = False,
) -> gpd.GeoDataFrame:
    """Attach LARIAC footprints to OpenView IDs, caching the costly GDB read."""
    run_dir, cache_path = Path(run_dir), Path(cache_path)
    if cache_path.exists() and not force:
        return gpd.read_parquet(cache_path)

    manifest = _manifest(run_dir)
    cfg = manifest["config"]
    ids = pd.read_parquet(run_dir / "building_ids.parquet")
    import pyogrio
    import shapely

    # One Arrow scan of the geometry column, filtered by FID in memory. FID
    # lists (OpenFileGDB caps them at ~5,000) are evaluated by rescanning the
    # whole layer per batch: hours for a county graph, seconds this way.
    # Only selected WKB is parsed; the layer holds curve geometries outside
    # the graph that shapely cannot read.
    meta, table = pyogrio.read_arrow(
        manifest["input"], layer=cfg.get("layer"), columns=[], return_fids=True,
    )
    fids = table.column(meta["fid_column"] or 0).to_numpy()
    keep = np.flatnonzero(np.isin(fids, ids.building_id.to_numpy(np.int64)))
    footprints = gpd.GeoDataFrame(
        {"building_id": fids[keep].astype(np.int64)},
        geometry=shapely.from_wkb(
            table.column(meta["geometry_name"]).take(keep)
            .to_numpy(zero_copy_only=False)
        ),
        crs=meta["crs"],
    ).to_crs(cfg["target_crs"])
    buildings = footprints[["building_id", "geometry"]].merge(
        ids, on="building_id", how="inner", validate="one_to_one",
    )
    buildings = gpd.GeoDataFrame(
        buildings, geometry="geometry", crs=footprints.crs
    ).rename(columns={"source_building_id": "LARIAC_BLD_ID"})
    buildings["LARIAC_BLD_ID"] = buildings.LARIAC_BLD_ID.astype(str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_parquet(cache_path, index=False)
    return buildings


def build_analysis_from_openview_3d(
    run_dir: Path,
    buildings: gpd.GeoDataFrame,
    dins: pd.DataFrame,
    output_path: Path,
    *,
    vegetation_path: Path | None = None,
) -> pd.DataFrame:
    """Build a DINS-assessed modeling table from building-level 3D view factors.

    ``dins`` must be the output of ``src.features.dins.build`` with a ``fire``
    column added by the caller. Prefer matching with the unique OpenView
    ``building_id`` and renaming that result to ``graph_id``; raw LARIAC source
    IDs are not globally unique in the county file. For compatibility with the existing fragility
    code, ``F_destroyed_wmean`` is retained as an alias of ``dose_sum``. No
    second area average is needed: each directional building view factor is
    already exchange area divided by the receiving building's facade area.
    """
    run_dir, output_path = Path(run_dir), Path(output_path)
    required = {"fire", "damage"}
    if "graph_id" not in dins and "BLD_ID" not in dins:
        required.add("graph_id or BLD_ID")
    missing = required.difference(dins.columns)
    if missing:
        raise ValueError(f"DINS table is missing columns: {sorted(missing)}")

    outcome_map = {
        "Destroyed (>50%)": "destroyed",
        "Major (26-50%)": "partial",
        "Minor (10-25%)": "partial",
        "Affected (1-9%)": "partial",
        "No Damage": "no_damage",
        "Inaccessible": "unassessed",
    }
    assessed = dins.copy()
    assessed["outcome"] = assessed.damage.map(outcome_map).fillna("unassessed")
    assessed = assessed.loc[
        assessed.fire.isin(["EATON", "PALISADES"])
        & assessed.outcome.ne("unassessed")
    ]
    assessed["any_damage"] = assessed.outcome.isin(
        ["partial", "destroyed"]
    ).astype(np.int8)
    assessed["is_destroyed"] = assessed.outcome.eq("destroyed").astype(np.int8)

    graph = buildings[["building_id", "LARIAC_BLD_ID", "geometry"]].copy()
    graph = graph.rename(columns={
        "building_id": "graph_id", "LARIAC_BLD_ID": "source_BLD_ID",
    })
    graph["source_BLD_ID"] = graph.source_BLD_ID.astype(str)
    if "graph_id" in assessed:
        assessed["graph_id"] = assessed.graph_id.astype(np.int64)
        assessed = assessed.drop(columns=["BLD_ID"], errors="ignore").merge(
            graph[["graph_id", "source_BLD_ID"]], on="graph_id", how="inner",
            validate="many_to_one",
        ).drop_duplicates(["fire", "graph_id"])
    else:
        assessed["BLD_ID"] = assessed.BLD_ID.astype(str)
        unique_source = graph.loc[
            ~graph.source_BLD_ID.duplicated(keep=False)
            & graph.source_BLD_ID.ne("None")
        ]
        assessed = assessed.merge(
            unique_source[["graph_id", "source_BLD_ID"]],
            left_on="BLD_ID", right_on="source_BLD_ID", how="inner",
            validate="many_to_one",
        ).drop(columns="BLD_ID").drop_duplicates(["fire", "graph_id"])

    # Retain ordinary LARIAC IDs where they identify one graph node. Use a
    # stable graph-prefixed key for missing or duplicated source IDs.
    source_counts = graph.source_BLD_ID.value_counts()
    source_is_unique = assessed.source_BLD_ID.map(source_counts).eq(1)
    source_is_valid = assessed.source_BLD_ID.notna() & assessed.source_BLD_ID.ne("None")
    assessed["BLD_ID"] = assessed.source_BLD_ID.where(
        source_is_unique & source_is_valid,
        "graph:" + assessed.graph_id.astype(str),
    )

    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.register(
        "assessed",
        assessed[["graph_id", "fire", "is_destroyed", "outcome"]],
    )
    edge_path = str(run_dir / "building_edges.parquet").replace("'", "''")
    try:
        exposure = con.execute(f"""
            WITH directed AS (
                SELECT building_i AS receiver, building_j AS emitter,
                       vf_i_to_j AS vf
                FROM read_parquet('{edge_path}')
                UNION ALL
                SELECT building_j AS receiver, building_i AS emitter,
                       vf_j_to_i AS vf
                FROM read_parquet('{edge_path}')
            ), dose AS (
                SELECT r.graph_id,
                       COUNT(d.emitter)::BIGINT AS graph_neighbors,
                       COUNT(e.graph_id)::BIGINT AS outcome_known_neighbors,
                       COUNT(DISTINCT d.emitter) FILTER (
                           WHERE e.is_destroyed = 1
                       )::BIGINT AS n_destroyed_bldgs,
                       COALESCE(SUM(d.vf), 0.0) AS F_total_wmean,
                       COALESCE(SUM(d.vf) FILTER (
                           WHERE e.is_destroyed = 1
                       ), 0.0) AS dose_sum,
                       COALESCE(MAX(d.vf) FILTER (
                           WHERE e.is_destroyed = 1
                       ), 0.0) AS dose_max_pair,
                       COALESCE(SUM(d.vf) FILTER (
                           WHERE e.outcome = 'partial'
                       ), 0.0) AS F_partial_wmean,
                       COALESCE(SUM(d.vf) FILTER (
                           WHERE e.outcome = 'no_damage'
                       ), 0.0) AS F_no_damage_wmean
                FROM assessed r
                LEFT JOIN directed d ON r.graph_id = d.receiver
                LEFT JOIN assessed e
                  ON d.emitter = e.graph_id AND r.fire = e.fire
                GROUP BY r.graph_id
            )
            SELECT * FROM dose
        """).fetchdf()
    finally:
        con.unregister("assessed")
        con.close()

    out = assessed.merge(exposure, on="graph_id", how="left", validate="one_to_one")
    out["F_destroyed_wmean"] = out.dose_sum
    out["log_dose"] = np.log(out.dose_sum.where(out.dose_sum > 0))
    out["exposed"] = out.dose_sum.gt(0).astype(np.int8)
    roof_pool = {
        "Wood": "Combustible", "Combustible": "Combustible",
        "Asphalt": "Asphalt", "Tile": "Tile", "Metal": "Metal",
        "Concrete": "Concrete", "Non Combustible": "Other",
        "Other": "Other", "Unknown": "Unknown",
    }
    if "roof_construction" in out:
        out["roof_class"] = pd.Categorical(
            out.roof_construction.map(roof_pool).fillna("Unknown"),
            categories=["Asphalt", "Tile", "Concrete", "Metal",
                        "Combustible", "Other", "Unknown"],
        )
    if "is_defended" in out:
        out["defended"] = out.is_defended.fillna(False).astype(bool)

    locations = graph.loc[graph.graph_id.isin(out.graph_id)].copy()
    locations["geometry"] = locations.geometry.representative_point()
    locations = locations.to_crs(4326)
    locations["lon_wgs84"] = locations.geometry.x
    locations["lat_wgs84"] = locations.geometry.y
    out = out.merge(
        locations[["graph_id", "lon_wgs84", "lat_wgs84"]],
        on="graph_id", how="left", validate="many_to_one",
    )
    dlat, dlon = 250 / 111_000, 250 / 92_000
    out["grid_id"] = (
        np.round(out.lat_wgs84 / dlat).astype("Int64").astype(str)
        + "_" + np.round(out.lon_wgs84 / dlon).astype("Int64").astype(str)
    )

    if vegetation_path is not None and Path(vegetation_path).exists():
        vegetation = pd.read_parquet(vegetation_path)
        vegetation["BLD_ID"] = vegetation.BLD_ID.astype(str)
        columns = [
            c for c in ["BLD_ID", "ndvi_mean", "veg_frac", "shrub_frac"]
            if c in vegetation.columns
        ]
        out = out.merge(
            vegetation[columns].drop_duplicates("BLD_ID"),
            left_on="source_BLD_ID", right_on="BLD_ID", how="left",
            validate="many_to_one", suffixes=("", "_vegetation"),
        ).drop(columns="BLD_ID_vegetation", errors="ignore")
        if "ndvi_mean" in out:
            out["has_ndvi"] = out.ndvi_mean.notna().astype(np.int8)

    out = out.sort_values(["fire", "graph_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out


def prepare_openview_3d_distance_comparison(
    run_dir: Path,
    buildings: gpd.GeoDataFrame,
    analysis: pd.DataFrame,
    cache_path: Path | None = None,
    *,
    rebuild: bool = False,
    patch_run_dir: Path | None = None,
) -> pd.DataFrame:
    """Attach nearest-destroyed CCD and visible-patch SSD to analysis rows.

    CCD is the planar centroid distance to the nearest *other* assessed,
    destroyed building in the same fire. SSD is the shortest 3D distance
    between a focal facade patch and a visible facade patch belonging to an
    assessed, destroyed building in the same fire. The latter is read from
    ``patch_exchange.parquet`` and can be cached because that county-scale
    table is large.

    The cache stores distances and identifiers only; current outcomes and
    exposure values are always taken from ``analysis`` when it is returned.
    County runs do not keep patch partitions, so ``patch_run_dir`` may name a
    fire-area companion run with the same configuration that does.
    """
    run_dir = Path(run_dir)
    patch_run_dir = Path(patch_run_dir) if patch_run_dir is not None else run_dir
    cache_path = Path(cache_path) if cache_path is not None else None
    required_analysis = {
        "graph_id", "fire", "is_destroyed", "F_destroyed_wmean",
    }
    missing = required_analysis.difference(analysis.columns)
    if missing:
        raise ValueError(f"Analysis table is missing columns: {sorted(missing)}")

    if cache_path is not None and cache_path.exists() and not rebuild:
        distances = pd.read_parquet(cache_path)
    else:
        assessed = analysis[["graph_id", "fire", "is_destroyed"]].copy()
        assessed["graph_id"] = assessed.graph_id.astype(np.int64)

        geometry = buildings[["building_id", "geometry"]].rename(
            columns={"building_id": "graph_id"}
        )
        points = geometry.loc[geometry.graph_id.isin(assessed.graph_id)].copy()
        points["geometry"] = points.geometry.centroid
        points["x"] = points.geometry.x
        points["y"] = points.geometry.y
        points = points.merge(
            assessed, on="graph_id", how="inner", validate="one_to_one",
        )

        ccd_parts = []
        for fire, fire_points in points.groupby("fire", sort=False):
            destroyed_xy = fire_points.loc[
                fire_points.is_destroyed.eq(1), ["x", "y"]
            ].to_numpy(float)
            if len(destroyed_xy) < 2:
                raise ValueError(f"{fire} has fewer than two destroyed buildings")
            tree = cKDTree(destroyed_xy)
            xy = fire_points[["x", "y"]].to_numpy(float)
            nearest = tree.query(xy, k=2)[0]
            ccd = np.where(
                fire_points.is_destroyed.to_numpy() == 1,
                nearest[:, 1], nearest[:, 0],
            )
            ccd_parts.append(pd.DataFrame({
                "graph_id": fire_points.graph_id.to_numpy(np.int64),
                "fire": fire,
                "ccd_ft": ccd * 3.28084,
            }))
        ccd = pd.concat(ccd_parts, ignore_index=True)

        # Raw patch partitions retain the same visibility-tested patch pairs
        # and distances as the reduced file. Read only tiles containing an
        # assessed facade: every qualifying pair has an assessed receiver and
        # an assessed destroyed emitter, so its canonical owner must be in one
        # of these tiles. This avoids two full scans of the 12 GB county file.
        manifest = _manifest(patch_run_dir)
        tile_size = float(manifest["config"]["tile_size_m"])
        tile_keys = set()
        for xmin, ymin, xmax, ymax in geometry.loc[
            geometry.graph_id.isin(assessed.graph_id), "geometry"
        ].bounds.itertuples(index=False, name=None):
            for ix in range(math.floor(xmin / tile_size),
                            math.floor(xmax / tile_size) + 1):
                for iy in range(math.floor(ymin / tile_size),
                                math.floor(ymax / tile_size) + 1):
                    tile_keys.add(f"{ix}_{iy}")
        raw_dir = patch_run_dir / "patch_exchange_raw"
        patch_files = sorted(
            path for key in tile_keys
            if (path := raw_dir / f"{key}.parquet").exists()
        )
        if not patch_files:
            raise FileNotFoundError(
                f"No patch-exchange partitions found for assessed tiles in {raw_dir}"
            )
        print(
            f"SSD: scanning {len(patch_files):,} assessed-area patch partitions "
            f"of {manifest['tile_count']:,} county tiles"
        )
        patch_source = "[" + ",".join(
            "'" + str(path).replace("'", "''") + "'" for path in patch_files
        ) + "]"
        con = duckdb.connect()
        con.execute("SET threads=4")
        con.execute("SET memory_limit='8GB'")
        con.register("assessed", assessed)
        try:
            ssd = con.execute(f"""
                WITH directed_visible AS (
                    SELECT r.graph_id, r.fire, p.distance_m
                    FROM read_parquet({patch_source}) p
                    JOIN assessed r ON p.building_p = r.graph_id
                    JOIN assessed e
                      ON p.building_q = e.graph_id
                     AND r.fire = e.fire
                     AND e.is_destroyed = 1
                    UNION ALL
                    SELECT r.graph_id, r.fire, p.distance_m
                    FROM read_parquet({patch_source}) p
                    JOIN assessed r ON p.building_q = r.graph_id
                    JOIN assessed e
                      ON p.building_p = e.graph_id
                     AND r.fire = e.fire
                     AND e.is_destroyed = 1
                )
                SELECT graph_id, fire, MIN(distance_m) * 3.28084 AS ssd_ft
                FROM directed_visible
                GROUP BY graph_id, fire
            """).fetchdf()
        finally:
            con.unregister("assessed")
            con.close()

        distances = ccd.merge(
            ssd, on=["graph_id", "fire"], how="left", validate="one_to_one",
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            distances.to_parquet(cache_path, index=False)

    if distances.duplicated(["graph_id", "fire"]).any():
        raise ValueError("Distance table is not unique by graph_id and fire")
    return analysis.merge(
        distances, on=["graph_id", "fire"], how="left", validate="one_to_one",
    )


def _wui_classes(points: gpd.GeoDataFrame, wui: gpd.GeoDataFrame) -> pd.Series:
    joined = gpd.sjoin(
        points[["node", "geometry"]], wui[["WUI_DESC", "geometry"]],
        how="left", predicate="within",
    )
    joined["priority"] = joined.WUI_DESC.map(WUI_PRIORITY).fillna(0)
    classes = (
        joined.sort_values("priority").drop_duplicates("node", keep="last")
        .set_index("node").WUI_DESC.reindex(points.node)
    )
    return classes.fillna("Outside mapped WUI").reset_index(drop=True)


def build_county_regional_sen(
    project_root: Path, run_dir: Path, threshold: float, cache_path: Path,
    wui_path: Path | None = None, *, n_neighbors: int | None = None,
    edge_floor_fraction: float = 0.25,
    exclude_building_ids: set | None = None,
) -> dict:
    """Construct a countywide EEAT or cumulative Top-N OpenView SEN.

    ``exclude_building_ids`` removes structures, such as LARIAC
    free-standing solar arrays, from the graph before any link is selected,
    so they neither join components nor count as isolated buildings.
    """
    project_root, run_dir = Path(project_root), Path(run_dir)
    manifest = _manifest(run_dir)
    buildings = load_county_buildings(run_dir, cache_path)
    excluded = 0
    if exclude_building_ids:
        keep = ~buildings.building_id.isin(exclude_building_ids)
        excluded = int((~keep).sum())
        buildings = buildings.loc[keep]
    buildings = buildings.reset_index(drop=True)
    buildings["node"] = np.arange(len(buildings), dtype=np.int64)
    node_map = buildings[["building_id", "LARIAC_BLD_ID", "node"]].rename(
        columns={"LARIAC_BLD_ID": "BLD_ID"}
    )
    if n_neighbors is None:
        # The EEAT query keeps only pairs whose endpoints are in node_map.
        active = _edge_query(run_dir, node_map, threshold=threshold)
        clustering_method = "EEAT"
    else:
        active = _cumulative_top_n_edge_query(
            run_dir, node_map, threshold, n_neighbors, edge_floor_fraction,
            restrict_to_node_map=bool(excluded),
        )
        clustering_method = f"cumulative_top_{n_neighbors}"

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

    if wui_path is None:
        wui = gpd.read_file(
            project_root / "data" / "calfire_wui_la.gpkg"
        ).to_crs(buildings.crs)
        wui = gpd.clip(wui, box(*buildings.total_bounds), keep_geom_type=True)
    else:
        wui = read_wui_for_bounds(
            wui_path, buildings.total_bounds, buildings.crs,
        )
    wui_union = wui.geometry.union_all()
    screen_geometry = wui_union.simplify(
        WUI_SIMPLIFY_M, preserve_topology=True
    ).buffer(WUI_BUFFER_M, quad_segs=4).intersection(
        box(*buildings.total_bounds)
    )
    screen = gpd.GeoDataFrame(
        {"screen": ["All CAL FIRE WUI classes + 2 mi"]},
        geometry=[screen_geometry], crs=buildings.crs,
    )
    points = gpd.GeoDataFrame(
        {"node": buildings.node},
        geometry=buildings.geometry.representative_point(), crs=buildings.crs,
    )
    buildings["wui_class"] = _wui_classes(points, wui).to_numpy()
    buildings["is_interface"] = buildings.wui_class.eq("Interface")
    buildings["near_screen_edge"] = points.geometry.distance(
        screen_geometry.boundary
    ).le(50).to_numpy()

    node_frame = pd.DataFrame({
        "component_id": labels,
        "component_size": sizes[labels],
        "is_interface": buildings.is_interface.to_numpy(),
        "near_screen_edge": buildings.near_screen_edge.to_numpy(),
        "x": points.geometry.x.to_numpy(),
        "y": points.geometry.y.to_numpy(),
    })
    components = (
        node_frame.groupby("component_id", as_index=False)
        .agg(component_size=("component_size", "first"),
             interface_buildings=("is_interface", "sum"),
             near_screen_edge=("near_screen_edge", "max"),
             xmin=("x", "min"), xmax=("x", "max"),
             ymin=("y", "min"), ymax=("y", "max"))
    )
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
    lookup = components.set_index("component_id")
    buildings["spans_interface_class"] = lookup.loc[
        labels, "spans_interface_class"
    ].to_numpy(bool)
    buildings["network_category"] = np.select(
        [buildings.component_size.eq(1), buildings.spans_interface_class],
        ["Isolated", "Interface-spanning SEN"],
        default="Other connected SEN",
    )
    components["mean_bond_ssd_ft"] = np.nan

    spanning = components[components.spans_interface_class]
    connected = components[components.component_size.gt(1)]
    con = duckdb.connect()
    try:
        candidate_pairs = con.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(run_dir / "building_edges.parquet")],
        ).fetchone()[0]
    finally:
        con.close()
    largest_spanning = int(spanning.component_size.max()) if len(spanning) else 0
    max_span = float(spanning.planar_span_km.max()) if len(spanning) else np.nan
    summary = pd.DataFrame([{
        "corridor": "Los Angeles County combined-WUI two-mile screen",
        "clustering_method": clustering_method,
        "n_neighbors": n_neighbors,
        "edge_floor_fraction": (
            edge_floor_fraction if n_neighbors is not None else np.nan
        ),
        "screen_buffer_m": WUI_BUFFER_M,
        "F_ij_threshold": float(threshold),
        "P_destroyed_equivalent": .50,
        "buildings": len(buildings),
        "excluded_structures": excluded,
        "candidate_pairs": candidate_pairs,
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
        "largest_interface_spanning_SEN": largest_spanning,
        "maximum_interface_spanning_span_km": max_span,
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
        "maximum_pair_distance_m": manifest["config"]["max_distance_m"],
    }])
    interface = wui.loc[wui.WUI_DESC.eq("Interface"), ["geometry"]]
    return {
        "buildings": buildings,
        "pairs": None,
        "active_edges": active,
        "components": components,
        "corridor": screen,
        "interface_spine": interface,
        "wui": wui,
        "summary": summary,
    }


def build_county_threshold_persistence(
    run_dir: Path,
    base_map_path: Path,
    probability_thresholds: pd.DataFrame,
    fmax_probability_thresholds: pd.DataFrame,
    bounds_lon_lat: tuple[float, float, float, float],
    output_map_path: Path,
    output_summary_path: Path,
    *,
    n_neighbors: int = 2,
    edge_floor_fraction: float = 0.25,
    force: bool = False,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Sweep cumulative Top-2, EEAT, and Fmax Top-2-pruned EEAT.

    ``probability_thresholds`` must contain increasing ``probability`` and
    ``F_ij_threshold`` columns for the cumulative-dose fragility curve.
    ``fmax_probability_thresholds`` has the same schema but is calibrated to
    the maximum single-pair dose. The returned map contains buildings inside
    the requested window and, for each method, the highest probability
    threshold at which that building remains in an Interface-spanning
    full-county component. Edges and ranked incidence are loaded once,
    avoiding repeated county graph/WUI reconstructions for the full sweep.
    """
    run_dir = Path(run_dir)
    base_map_path = Path(base_map_path)
    output_map_path = Path(output_map_path)
    output_summary_path = Path(output_summary_path)
    def _validated_thresholds(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        selected = frame[["probability", "F_ij_threshold"]].copy()
        selected = selected.sort_values("probability")
        if selected.empty or selected.duplicated("probability").any():
            raise ValueError(f"{name} thresholds must be nonempty and unique")
        if not selected.probability.between(0, 1, inclusive="neither").all():
            raise ValueError("Probabilities must lie strictly between zero and one")
        if not selected.F_ij_threshold.is_monotonic_increasing:
            raise ValueError(f"{name} coupling thresholds must increase with probability")
        return selected

    thresholds = _validated_thresholds(
        probability_thresholds, "Cumulative-dose",
    )
    fmax_thresholds = _validated_thresholds(
        fmax_probability_thresholds, "Fmax",
    )
    if not np.allclose(
        thresholds.probability.to_numpy(),
        fmax_thresholds.probability.to_numpy(),
    ):
        raise ValueError("Cumulative-dose and Fmax probabilities must match")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least two for Top-2 construction")
    if not 0 <= edge_floor_fraction <= 1:
        raise ValueError("edge_floor_fraction must lie between zero and one")
    methods = ("top2", "eeat", "eeat_top2_fmax")
    thresholds_by_method = {
        "top2": thresholds,
        "eeat": thresholds,
        "eeat_top2_fmax": fmax_thresholds,
    }
    if output_map_path.exists() and output_summary_path.exists() and not force:
        cached_summary = pd.read_csv(output_summary_path)
        expected = np.tile(thresholds.probability.to_numpy(), len(methods))
        required_columns = {
            f"{method}_{suffix}"
            for method in methods
            for suffix in (
                "max_probability", "component_size_at_max_probability",
            )
        }
        if (
            len(cached_summary) == len(methods) * len(thresholds)
            and np.allclose(cached_summary.probability.to_numpy(), expected)
            and "edge_floor_fraction" in cached_summary.columns
            and np.allclose(
                cached_summary.loc[
                    cached_summary.method.eq("top2"), "edge_floor_fraction"
                ].to_numpy(float),
                edge_floor_fraction,
            )
            and required_columns.issubset(gpd.read_parquet(output_map_path).columns)
        ):
            return gpd.read_parquet(output_map_path), cached_summary

    buildings = gpd.read_parquet(
        base_map_path,
        columns=["building_id", "wui_class", "geometry"],
    ).reset_index(drop=True)
    if buildings.building_id.duplicated().any():
        raise ValueError("Base county map must be unique by building_id")
    node_map = pd.DataFrame({
        "building_id": buildings.building_id.to_numpy(np.int64),
        "node": np.arange(len(buildings), dtype=np.int64),
    })
    projected_window = gpd.GeoSeries(
        [box(*bounds_lon_lat)], crs="EPSG:4326",
    ).to_crs(buildings.crs)
    window_geometry = projected_window.iloc[0]
    points = buildings.geometry.centroid
    in_window = points.intersects(window_geometry).to_numpy()
    window_nodes = np.flatnonzero(in_window)
    window = buildings.loc[in_window].copy()
    window["x"] = points.loc[in_window].x.to_numpy()
    window["y"] = points.loc[in_window].y.to_numpy()
    is_interface = buildings.wui_class.eq("Interface").to_numpy()

    edge_path = str(run_dir / "building_edges.parquet").replace("'", "''")
    minimum_threshold = float(min(
        thresholds.F_ij_threshold.min(),
        fmax_thresholds.F_ij_threshold.min(),
    ))
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.register("node_map", node_map)
    try:
        eeat = con.execute(f"""
            SELECT left_node.node::BIGINT AS u,
                   right_node.node::BIGINT AS v,
                   LEAST(1.0, GREATEST(e.vf_i_to_j, e.vf_j_to_i)) AS F_ij
            FROM read_parquet('{edge_path}') e
            JOIN node_map left_node ON e.building_i = left_node.building_id
            JOIN node_map right_node ON e.building_j = right_node.building_id
            WHERE GREATEST(e.vf_i_to_j, e.vf_j_to_i) >= {minimum_threshold}
        """).fetchdf()
        top_n = con.execute(f"""
            WITH pair_scores AS (
                SELECT building_i, building_j,
                       LEAST(1.0, GREATEST(vf_i_to_j, vf_j_to_i)) AS F_ij
                FROM read_parquet('{edge_path}')
            ), incidence AS (
                SELECT building_i AS node_id, building_j AS neighbor_id, F_ij
                FROM pair_scores
                UNION ALL
                SELECT building_j AS node_id, building_i AS neighbor_id, F_ij
                FROM pair_scores
            ), ranked AS (
                SELECT node_id, neighbor_id, F_ij,
                       ROW_NUMBER() OVER (
                           PARTITION BY node_id ORDER BY F_ij DESC, neighbor_id
                       ) AS neighbor_rank
                FROM incidence
                QUALIFY neighbor_rank <= {int(n_neighbors)}
            )
            SELECT focal.node::BIGINT AS u, neighbor.node::BIGINT AS v,
                   r.F_ij, r.neighbor_rank
            FROM ranked r
            JOIN node_map focal ON r.node_id = focal.building_id
            JOIN node_map neighbor ON r.neighbor_id = neighbor.building_id
        """).fetchdf()
    finally:
        con.unregister("node_map")
        con.close()

    n_nodes = len(buildings)
    top_u = top_n.u.to_numpy(np.int64)
    top_v = top_n.v.to_numpy(np.int64)
    top_f = top_n.F_ij.to_numpy(float)
    top_rank = top_n.neighbor_rank.to_numpy(np.int8)
    cumulative_top = {}
    for top_k in (1, 2):
        ranked = top_rank <= top_k
        ranked_u, ranked_v, ranked_f = (
            top_u[ranked], top_v[ranked], top_f[ranked]
        )
        cumulative_top[top_k] = (
            ranked_u,
            ranked_v,
            np.bincount(ranked_u, minlength=n_nodes),
            np.bincount(ranked_u, weights=ranked_f, minlength=n_nodes),
        )
    eeat_u = eeat.u.to_numpy(np.int64)
    eeat_v = eeat.v.to_numpy(np.int64)
    eeat_f = eeat.F_ij.to_numpy(float)
    del top_n, eeat

    persistence = {method: np.full(len(window), np.nan) for method in methods}
    persistent_size = {
        method: np.zeros(len(window), dtype=np.int32) for method in methods
    }
    rows = []
    for method in methods:
        for row in thresholds_by_method[method].itertuples(index=False):
            threshold = float(row.F_ij_threshold)
            if method == "top2":
                top_k = 2
                ranked_u, ranked_v, top_count, top_sum = cumulative_top[top_k]
                qualifies = (top_count == top_k) & (top_sum >= threshold)
                keep = (
                    qualifies[ranked_u] & qualifies[ranked_v]
                    & (ranked_f >= edge_floor_fraction * threshold)
                )
                u, v = ranked_u[keep], ranked_v[keep]
            elif method == "eeat_top2_fmax":
                keep = top_f >= threshold
                u, v = top_u[keep], top_v[keep]
            else:
                keep = eeat_f >= threshold
                u, v = eeat_u[keep], eeat_v[keep]
            graph = sparse.coo_matrix(
                (np.ones(2 * len(u), dtype=np.int8),
                 (np.r_[u, v], np.r_[v, u])),
                shape=(n_nodes, n_nodes),
            ).tocsr()
            n_components, labels = sparse.csgraph.connected_components(
                graph, directed=False,
            )
            sizes = np.bincount(labels)
            interface_counts = np.bincount(
                labels, weights=is_interface, minlength=n_components,
            )
            spanning_components = (
                (interface_counts > 0) & (interface_counts < sizes)
            )
            spanning_nodes = spanning_components[labels]
            shown = spanning_nodes[window_nodes]
            persistence[method][shown] = float(row.probability)
            persistent_size[method][shown] = sizes[labels[window_nodes[shown]]]
            spanning_sizes = sizes[spanning_components]
            rows.append({
                "method": method,
                "probability": float(row.probability),
                "F_ij_threshold": threshold,
                "edge_floor_fraction": (
                    edge_floor_fraction if method == "top2" else np.nan
                ),
                "full_county_components": int(n_components),
                "interface_spanning_components": int(spanning_components.sum()),
                "buildings_in_interface_spanning_components": int(spanning_sizes.sum()),
                "share_in_interface_spanning_components": float(
                    spanning_sizes.sum() / n_nodes
                ),
                "largest_interface_spanning_component": int(
                    spanning_sizes.max() if len(spanning_sizes) else 0
                ),
                "interface_spanning_buildings_in_window": int(shown.sum()),
                "interface_spanning_components_in_window": int(
                    np.unique(labels[window_nodes[shown]]).size
                ),
            })
            del graph, labels, sizes, interface_counts, spanning_nodes

    for method in methods:
        window[f"{method}_max_probability"] = persistence[method]
        window[f"{method}_component_size_at_max_probability"] = persistent_size[method]
    summary = pd.DataFrame(rows)
    output_map_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    window.to_parquet(output_map_path, index=False)
    summary.to_csv(output_summary_path, index=False)
    return window, summary


def save_county_regional_sen_cache(result: dict, map_path: Path) -> None:
    """Cache the complete per-building regional assignment and sidecars."""
    map_path = Path(map_path)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    required = {
        "building_id", "component_id", "sen_id", "component_size",
        "wui_class", "is_interface", "near_screen_edge",
        "spans_interface_class", "network_category", "geometry",
    }
    missing = required.difference(result["buildings"].columns)
    if missing:
        raise ValueError(f"Regional cache is missing fields: {sorted(missing)}")
    result["buildings"].to_parquet(map_path, index=False)
    result["components"].to_parquet(
        map_path.with_name(f"{map_path.stem}_components.parquet"), index=False,
    )
    result["summary"].to_parquet(
        map_path.with_name(f"{map_path.stem}_summary.parquet"), index=False,
    )


def pair_coupling(run_dir: Path, building_i, building_j) -> pd.DataFrame:
    """3D coupling ``F_ij`` for arbitrary building pairs.

    ``F_ij`` is the graph's score, ``min(1, max(vf_i_to_j, vf_j_to_i))``;
    a pair with no edge in the run (no visible exchange) scores zero.
    Returns one row per input pair, in input order, with ``F_ij``,
    ``has_edge`` and ``has_contact``.
    """
    pairs = pd.DataFrame({
        "position": np.arange(len(building_i)),
        "a": np.minimum(building_i, building_j),
        "b": np.maximum(building_i, building_j),
    })
    edge_path = str(Path(run_dir) / "building_edges.parquet").replace("'", "''")
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.register("pairs", pairs)
    try:
        scored = con.execute(f"""
            SELECT p.position,
                   -- LEAST/GREATEST skip NULLs, so test for the edge explicitly.
                   CASE WHEN e.building_i IS NULL THEN 0.0
                        ELSE LEAST(1.0, GREATEST(e.vf_i_to_j, e.vf_j_to_i)) END AS F_ij,
                   e.building_i IS NOT NULL AS has_edge,
                   COALESCE(e.has_contact, FALSE) AS has_contact
            FROM pairs p
            LEFT JOIN read_parquet('{edge_path}') e
              ON LEAST(e.building_i, e.building_j) = p.a
             AND GREATEST(e.building_i, e.building_j) = p.b
        """).df()
    finally:
        con.close()
    if not scored.position.is_unique:
        raise ValueError("building_edges.parquet holds duplicate pairs")
    return scored.sort_values("position").drop(columns="position").reset_index(drop=True)


def load_county_regional_sen_cache(
    project_root: Path, map_path: Path, wui_path: Path | None = None,
) -> dict:
    """Load a cached county regional result without scanning graph edges."""
    project_root, map_path = Path(project_root), Path(map_path)
    component_path = map_path.with_name(
        f"{map_path.stem}_components.parquet"
    )
    summary_path = map_path.with_name(f"{map_path.stem}_summary.parquet")
    missing = [
        str(path) for path in (map_path, component_path, summary_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete regional SEN cache: {missing}")

    buildings = gpd.read_parquet(map_path)
    components = pd.read_parquet(component_path)
    summary = pd.read_parquet(summary_path)
    if wui_path is None:
        wui = gpd.read_file(
            project_root / "data" / "calfire_wui_la.gpkg"
        ).to_crs(buildings.crs)
        wui = gpd.clip(
            wui, box(*buildings.total_bounds), keep_geom_type=True,
        )
    else:
        wui = read_wui_for_bounds(
            wui_path, buildings.total_bounds, buildings.crs,
        )
    screen_geometry = wui.geometry.union_all().simplify(
        WUI_SIMPLIFY_M, preserve_topology=True
    ).buffer(WUI_BUFFER_M, quad_segs=4).intersection(
        box(*buildings.total_bounds)
    )
    screen = gpd.GeoDataFrame(
        {"screen": ["All CAL FIRE WUI classes + 2 mi"]},
        geometry=[screen_geometry], crs=buildings.crs,
    )
    interface = wui.loc[wui.WUI_DESC.eq("Interface"), ["geometry"]]
    return {
        "buildings": buildings,
        "pairs": None,
        "active_edges": None,
        "components": components,
        "corridor": screen,
        "interface_spine": interface,
        "wui": wui,
        "summary": summary,
        "cached": True,
    }
