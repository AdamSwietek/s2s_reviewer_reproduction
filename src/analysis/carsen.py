"""Validation helpers for the statewide 2D California SEN graph."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq, curve_fit
from scipy.spatial import cKDTree
from shapely import intersects_xy
from shapely.geometry import box


def _manifest(run_dir: Path) -> dict:
    with (Path(run_dir) / "manifest.json").open() as stream:
        return json.load(stream)


def build_county_extent_crosswalk(
    county_buildings_path: Path,
    state_2d_run: Path,
    extent_buildings_path: Path,
    crosswalk_path: Path,
    *,
    maximum_centroid_distance_m: float = 5.0,
    force: bool = False,
) -> tuple[Path, Path, pd.DataFrame]:
    """Extract statewide 2D buildings in the 3D extent and match footprints.

    The two graph runs use different source inventories (LARIAC and OSM), so
    raw node IDs are not comparable.  We form a conservative one-to-one
    crosswalk from mutual nearest footprint centroids, retaining pairs no more
    than ``maximum_centroid_distance_m`` apart.
    """
    county_buildings_path = Path(county_buildings_path)
    state_2d_run = Path(state_2d_run)
    extent_buildings_path = Path(extent_buildings_path)
    crosswalk_path = Path(crosswalk_path)
    extent_buildings_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(state_2d_run)
    target_crs = manifest["config"]["target_crs"]

    county = gpd.read_parquet(
        county_buildings_path,
        columns=["building_id", "LARIAC_BLD_ID", "geometry"],
    )
    county_extent = gpd.GeoSeries(
        [box(*county.total_bounds)], crs=county.crs,
    ).to_crs(target_crs).iloc[0]

    if force or not extent_buildings_path.exists():
        state = gpd.read_file(
            manifest["input"],
            bbox=tuple(county_extent.bounds),
            columns=["building_id", "source_building_id"],
        )
        state_points = state.geometry.centroid
        state = state.loc[state_points.intersects(county_extent)].copy()
        state.to_parquet(extent_buildings_path, index=False)
    else:
        state = gpd.read_parquet(extent_buildings_path)

    if force or not crosswalk_path.exists():
        county = county.to_crs(target_crs)
        county_points = county.geometry.centroid
        state_points = state.geometry.centroid
        county_xy = np.column_stack((county_points.x, county_points.y))
        state_xy = np.column_stack((state_points.x, state_points.y))
        distance_3d_to_2d, nearest_2d = cKDTree(state_xy).query(county_xy)
        _, nearest_3d = cKDTree(county_xy).query(state_xy)
        mutual = nearest_3d[nearest_2d] == np.arange(len(county))
        keep = mutual & (distance_3d_to_2d <= maximum_centroid_distance_m)
        county_position = np.flatnonzero(keep)
        state_position = nearest_2d[keep]
        crosswalk = pd.DataFrame({
            "building_3d": county.iloc[county_position].building_id.to_numpy(),
            "building_2d": state.iloc[state_position].building_id.to_numpy(),
            "source_building_id_3d": (
                county.iloc[county_position].LARIAC_BLD_ID.astype(str).to_numpy()
            ),
            "source_building_id_2d": (
                state.iloc[state_position].source_building_id.astype(str).to_numpy()
            ),
            "centroid_distance_m": distance_3d_to_2d[keep],
        })
        crosswalk.to_parquet(crosswalk_path, index=False)
    else:
        crosswalk = pd.read_parquet(crosswalk_path)

    summary = pd.DataFrame([{
        "county_3d_buildings": len(county),
        "state_2d_buildings_in_3d_extent": len(state),
        "mutual_matches_within_distance": len(crosswalk),
        "maximum_centroid_distance_m": maximum_centroid_distance_m,
        "share_of_3d_buildings_matched": len(crosswalk) / len(county),
        "share_of_extent_2d_buildings_matched": len(crosswalk) / len(state),
        "median_centroid_distance_m": crosswalk.centroid_distance_m.median(),
        "p95_centroid_distance_m": crosswalk.centroid_distance_m.quantile(.95),
        "county_extent_crs": target_crs,
        "county_extent_xmin": county_extent.bounds[0],
        "county_extent_ymin": county_extent.bounds[1],
        "county_extent_xmax": county_extent.bounds[2],
        "county_extent_ymax": county_extent.bounds[3],
    }])
    return extent_buildings_path, crosswalk_path, summary


def build_shared_pair_comparison(
    state_2d_run: Path,
    county_3d_run: Path,
    crosswalk_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> Path:
    """Join 2D and 3D directional-max F scores for common matched pairs."""
    state_2d_run = Path(state_2d_run)
    county_3d_run = Path(county_3d_run)
    crosswalk_path = Path(crosswalk_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    con = duckdb.connect()
    try:
        con.execute("PRAGMA threads=4")
        con.execute("PRAGMA preserve_insertion_order=false")
        con.execute(
            "CREATE TEMP TABLE crosswalk AS "
            "SELECT building_2d, building_3d FROM read_parquet(?)",
            [str(crosswalk_path)],
        )
        con.execute("CREATE INDEX crosswalk_2d ON crosswalk(building_2d)")
        con.execute("CREATE INDEX crosswalk_3d ON crosswalk(building_3d)")
        output_sql = str(output_path).replace("'", "''")
        query = f"""
            COPY (
                WITH edges_2d AS (
                    SELECT
                        LEAST(ci.building_3d, cj.building_3d) AS building_a,
                        GREATEST(ci.building_3d, cj.building_3d) AS building_b,
                        LEAST(1.0, GREATEST(e.vf_i_to_j, e.vf_j_to_i)) AS F_2d
                    FROM read_parquet(?) e
                    JOIN crosswalk ci ON e.building_i = ci.building_2d
                    JOIN crosswalk cj ON e.building_j = cj.building_2d
                ),
                edges_3d AS (
                    SELECT
                        LEAST(e.building_i, e.building_j) AS building_a,
                        GREATEST(e.building_i, e.building_j) AS building_b,
                        LEAST(1.0, GREATEST(e.vf_i_to_j, e.vf_j_to_i)) AS F_3d
                    FROM read_parquet(?) e
                )
                SELECT e2.building_a, e2.building_b, e2.F_2d, e3.F_3d
                FROM edges_2d e2
                INNER JOIN edges_3d e3 USING (building_a, building_b)
                WHERE e2.F_2d > 0 AND e3.F_3d > 0
            ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(query, [
            str(state_2d_run / "building_edges.parquet"),
            str(county_3d_run / "building_edges.parquet"),
        ])
    finally:
        con.close()
    return output_path


def pair_comparison_summary(comparison_path: Path) -> pd.DataFrame:
    """Return raw- and log-scale Pearson correlations for shared pairs."""
    con = duckdb.connect()
    try:
        row = con.execute(
            """
            SELECT
                count(*) AS shared_positive_pairs,
                corr(F_2d, F_3d) AS pearson_r_raw,
                corr(log10(F_2d), log10(F_3d)) AS pearson_r_log10,
                regr_slope(log10(F_3d), log10(F_2d)) AS log10_slope,
                regr_intercept(log10(F_3d), log10(F_2d)) AS log10_intercept,
                median(F_2d) AS median_F_2d,
                median(F_3d) AS median_F_3d
            FROM read_parquet(?)
            """,
            [str(comparison_path)],
        ).fetchdf()
    finally:
        con.close()
    return row


def build_2d_fire_dose(
    analysis_path: Path,
    county_2d_run: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> Path:
    """Reconstruct cumulative exposure from destroyed 2D graph neighbors."""
    analysis_path = Path(analysis_path)
    county_2d_run = Path(county_2d_run)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path
    output_sql = str(output_path).replace("'", "''")
    con = duckdb.connect()
    try:
        con.execute("PRAGMA threads=4")
        con.execute("PRAGMA preserve_insertion_order=false")
        query = f"""
            COPY (
                WITH outcomes AS (
                    SELECT CAST(BLD_ID AS VARCHAR) AS BLD_ID,
                           fire, outcome, is_destroyed, grid_id
                    FROM read_parquet(?)
                    WHERE fire IN ('EATON', 'PALISADES')
                ), id_map AS (
                    SELECT building_id,
                           CAST(source_building_id AS VARCHAR) AS source_id
                    FROM read_parquet(?)
                    WHERE source_building_id IS NOT NULL
                ), matched AS (
                    SELECT o.*, m.building_id AS graph_id
                    FROM outcomes o
                    JOIN id_map m ON o.BLD_ID = m.source_id
                ), directed AS (
                    SELECT building_i AS emitter, building_j AS receiver,
                           vf_i_to_j AS vf
                    FROM read_parquet(?)
                    UNION ALL
                    SELECT building_j, building_i, vf_j_to_i
                    FROM read_parquet(?)
                ), dose AS (
                    SELECT r.graph_id,
                           count(d.emitter) AS graph_neighbors,
                           count(e.graph_id) AS outcome_known_neighbors,
                           count(*) FILTER (WHERE e.is_destroyed = 1)
                               AS destroyed_neighbors,
                           sum(CASE WHEN e.is_destroyed = 1 THEN d.vf ELSE 0 END)
                               AS dose_sum,
                           max(CASE WHEN e.is_destroyed = 1 THEN d.vf ELSE 0 END)
                               AS dose_max_pair
                    FROM matched r
                    LEFT JOIN directed d ON r.graph_id = d.receiver
                    LEFT JOIN matched e ON d.emitter = e.graph_id
                    GROUP BY r.graph_id
                )
                SELECT m.*,
                       coalesce(d.graph_neighbors, 0) AS graph_neighbors,
                       coalesce(d.outcome_known_neighbors, 0)
                           AS outcome_known_neighbors,
                       coalesce(d.destroyed_neighbors, 0) AS destroyed_neighbors,
                       coalesce(d.dose_sum, 0) AS dose_sum,
                       coalesce(d.dose_max_pair, 0) AS dose_max_pair
                FROM matched m LEFT JOIN dose d USING (graph_id)
            ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        edge_path = county_2d_run / "building_edges.parquet"
        con.execute(query, [
            str(analysis_path),
            str(county_2d_run / "building_ids.parquet"),
            str(edge_path),
            str(edge_path),
        ])
    finally:
        con.close()
    return output_path


def logistic_5pl(x, p_min, p_max, k, hill, asymmetry):
    """Five-parameter logistic curve on positive exposure."""
    x = np.maximum(np.asarray(x, float), 1e-300)
    with np.errstate(over="ignore", invalid="ignore"):
        return p_min + (p_max - p_min) / (
            1 + (k / x) ** hill
        ) ** asymmetry


def fit_2d_fragility(
    dose_path: Path, *, metric: str = "dose_sum",
) -> tuple[pd.DataFrame, dict]:
    """Fit fire-specific and pooled 5PL curves to a 2D exposure metric.

    ``metric`` selects the exposure column: ``dose_sum`` is the cumulative
    destroyed-neighbor dose used for EEAT and cumulative Top-N, and
    ``dose_max_pair`` is the single strongest destroyed-neighbor pair used
    for the Fmax-calibrated EEAT variant.
    """
    dose = pd.read_parquet(dose_path)
    if metric not in dose.columns:
        raise ValueError(f"Dose table has no {metric!r} column")
    fits, rows = {}, []
    for fire in ("EATON", "PALISADES", "POOLED"):
        sample = dose if fire == "POOLED" else dose[dose.fire.eq(fire)]
        x = sample[metric].to_numpy(float)
        y = sample.is_destroyed.to_numpy(float)
        use = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[use], y[use]
        base = y[x < np.quantile(x, .1)].mean()
        top = y[x > np.quantile(x, .9)].mean()
        p0 = [
            np.clip(base, 1e-3, .4), np.clip(top, .1, .99),
            np.median(x), 1., 1.,
        ]
        bounds = (
            [0, 0, 1e-12, .1, .1],
            [.5, 1., x.max() * 5, 20., 10.],
        )
        parameters, _ = curve_fit(
            logistic_5pl, x, y, p0=p0, bounds=bounds, maxfev=100_000,
        )
        target = (parameters[0] + parameters[1]) / 2
        lo, hi = x.min() / 10, x.max() * 10
        asymptotic_f50 = brentq(
            lambda value: logistic_5pl(value, *parameters) - target,
            lo, hi,
        )
        absolute_p50 = (
            brentq(
                lambda value: logistic_5pl(value, *parameters) - .5,
                lo, hi,
            )
            if parameters[0] < .5 < parameters[1] else np.nan
        )
        prediction = np.clip(logistic_5pl(x, *parameters), 1e-9, 1 - 1e-9)
        at_bound = bool(
            np.any(np.isclose(parameters, bounds[0], rtol=0, atol=1e-5))
            or np.any(np.isclose(parameters, bounds[1], rtol=0, atol=1e-5))
        )
        fit = {
            "fn": logistic_5pl,
            "parameters": parameters,
            "n": len(x),
            "p_min": parameters[0],
            "p_max": parameters[1],
            "k": parameters[2],
            "hill": parameters[3],
            "asymmetry": parameters[4],
            "asymptotic_f50": asymptotic_f50,
            "absolute_p50": absolute_p50,
            "brier": float(np.mean((prediction - y) ** 2)),
            "log_loss": float(-np.mean(
                y * np.log(prediction) + (1 - y) * np.log(1 - prediction)
            )),
            "parameter_at_bound": at_bound,
        }
        fits[fire] = fit
        rows.append({
            "metric": metric, "model": "5PL", "fire": fire,
            **{key: value for key, value in fit.items()
               if key not in {"fn", "parameters"}},
        })
    return pd.DataFrame(rows), fits


def _atomic_copy(con, query: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    partial_sql = str(partial).replace("'", "''")
    con.execute(
        f"COPY ({query}) TO '{partial_sql}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    partial.replace(destination)


def build_statewide_ranked_incidence(
    state_2d_run: Path,
    ranked_path: Path,
    *,
    maximum_rank: int = 2,
    force: bool = False,
) -> Path:
    """Cache each statewide building's strongest incident pair scores."""
    state_2d_run, ranked_path = Path(state_2d_run), Path(ranked_path)
    ranked_path.parent.mkdir(parents=True, exist_ok=True)
    if ranked_path.exists() and not force:
        return ranked_path
    con = duckdb.connect()
    try:
        con.execute("PRAGMA threads=4")
        con.execute("PRAGMA memory_limit='6GB'")
        con.execute("PRAGMA preserve_insertion_order=false")
        edge_sql = str(state_2d_run / "building_edges.parquet").replace("'", "''")
        query = f"""
            WITH pairs AS (
                SELECT building_i, building_j,
                       greatest(vf_i_to_j, vf_j_to_i) AS bond
                FROM read_parquet('{edge_sql}')
            ), incidence AS (
                SELECT building_i AS node, building_j AS neighbor, bond FROM pairs
                UNION ALL
                SELECT building_j AS node, building_i AS neighbor, bond FROM pairs
            ), ranked AS (
                SELECT node, neighbor, bond,
                       row_number() OVER (
                           PARTITION BY node ORDER BY bond DESC, neighbor
                       ) AS neighbor_rank
                FROM incidence
                QUALIFY neighbor_rank <= {int(maximum_rank)}
            )
            SELECT node, neighbor, bond,
                   neighbor_rank::TINYINT AS neighbor_rank
            FROM ranked
        """
        _atomic_copy(con, query, ranked_path)
    finally:
        con.close()
    return ranked_path


def _component_paths(output_dir: Path, method: str) -> dict[str, Path]:
    stem = Path(output_dir) / f"california_{method}"
    return {
        "scores": Path(f"{stem}_node_scores.parquet"),
        "edges": Path(f"{stem}_active_edges.parquet"),
        "assignments": Path(f"{stem}_assignments.parquet"),
        "summary": Path(f"{stem}_summary.csv"),
        "sizes": Path(f"{stem}_component_size_frequency.csv"),
    }


def build_statewide_components(
    state_2d_run: Path,
    output_dir: Path,
    threshold: float,
    method: str,
    *,
    ranked_path: Path | None = None,
    n_neighbors: int = 2,
    edge_floor_fraction: float = .25,
    force: bool = False,
) -> dict[str, Path]:
    """Build full-node statewide EEAT or cumulative Top-N assignments.

    For cumulative Top-N, nodes qualify on the sum of their strongest N
    incident couplings. Retained links must connect two qualifying nodes and
    individually reach ``edge_floor_fraction * threshold``.
    """
    if method not in {"eeat", "top2"}:
        raise ValueError("method must be 'eeat' or 'top2'")
    if not 0 <= edge_floor_fraction <= 1:
        raise ValueError("edge_floor_fraction must lie between zero and one")
    state_2d_run, output_dir = Path(state_2d_run), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _component_paths(output_dir, method)
    required = (paths["edges"], paths["assignments"], paths["summary"])
    if all(path.exists() for path in required) and not force:
        cached = pd.read_csv(paths["summary"]).iloc[0]
        same_n = (
            pd.isna(cached.neighbors_summed) if method == "eeat"
            else int(cached.neighbors_summed) == int(n_neighbors)
        )
        same_floor = (
            method == "eeat"
            or (
                "edge_floor_fraction" in cached.index
                and np.isclose(
                    float(cached.edge_floor_fraction), edge_floor_fraction,
                )
            )
        )
        if (
            cached.method == method
            and np.isclose(cached.threshold, threshold)
            and same_n
            and same_floor
        ):
            return paths
        force = True

    edge_source = state_2d_run / "building_edges.parquet"
    con = duckdb.connect()
    try:
        con.execute("PRAGMA threads=4")
        con.execute("PRAGMA memory_limit='6GB'")
        con.execute("PRAGMA preserve_insertion_order=false")
        if method == "eeat":
            edge_sql = str(edge_source).replace("'", "''")
            if force or not paths["edges"].exists():
                _atomic_copy(con, f"""
                    SELECT building_i, building_j,
                           least(1.0, greatest(vf_i_to_j, vf_j_to_i)) AS bond
                    FROM read_parquet('{edge_sql}')
                    WHERE greatest(vf_i_to_j, vf_j_to_i) >= {float(threshold)}
                """, paths["edges"])
        else:
            if ranked_path is None:
                raise ValueError("ranked_path is required for top2")
            ranked_path = Path(ranked_path)
            ranked_sql = str(ranked_path).replace("'", "''")
            if force or not paths["scores"].exists():
                _atomic_copy(con, f"""
                    SELECT node AS building_id,
                           count(*)::SMALLINT AS top_n_neighbor_count,
                           sum(bond) AS top_n_sum
                    FROM read_parquet('{ranked_sql}')
                    WHERE neighbor_rank <= {int(n_neighbors)}
                    GROUP BY node
                """, paths["scores"])
            score_sql = str(paths["scores"]).replace("'", "''")
            if force or not paths["edges"].exists():
                _atomic_copy(con, f"""
                    WITH qualifying AS (
                        SELECT building_id
                        FROM read_parquet('{score_sql}')
                        WHERE top_n_neighbor_count = {int(n_neighbors)}
                          AND top_n_sum >= {float(threshold)}
                    ), selected AS (
                        SELECT
                               least(t.node, t.neighbor) AS building_i,
                               greatest(t.node, t.neighbor) AS building_j,
                               max(t.bond) AS bond
                        FROM read_parquet('{ranked_sql}') t
                        JOIN qualifying q ON t.node = q.building_id
                        WHERE t.neighbor_rank <= {int(n_neighbors)}
                        GROUP BY 1, 2
                    )
                    SELECT s.building_i, s.building_j
                    FROM selected s
                    JOIN qualifying qi ON s.building_i = qi.building_id
                    JOIN qualifying qj ON s.building_j = qj.building_id
                    WHERE s.bond >= {float(edge_floor_fraction * threshold)}
                """, paths["edges"])
        ids = con.execute(
            "SELECT building_id FROM read_parquet(?) ORDER BY building_id",
            [str(state_2d_run / "building_ids.parquet")],
        ).fetchdf().building_id.to_numpy(np.int64)
        edges = con.execute(
            "SELECT building_i, building_j FROM read_parquet(?)",
            [str(paths["edges"])],
        ).fetchdf()
        candidate_pairs = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(edge_source)]
        ).fetchone()[0]
        if method == "top2":
            qualifying_ids = con.execute(f"""
                SELECT building_id FROM read_parquet('{score_sql}')
                WHERE top_n_neighbor_count = {int(n_neighbors)}
                  AND top_n_sum >= {float(threshold)}
            """).fetchnumpy()["building_id"]
        else:
            qualifying_ids = np.unique(np.r_[
                edges.building_i.to_numpy(np.int64),
                edges.building_j.to_numpy(np.int64),
            ])
    finally:
        con.close()

    u = np.searchsorted(ids, edges.building_i.to_numpy(np.int64))
    v = np.searchsorted(ids, edges.building_j.to_numpy(np.int64))
    if len(edges) and (
        np.any(u == len(ids)) or np.any(v == len(ids))
        or not np.array_equal(ids[u], edges.building_i.to_numpy(np.int64))
        or not np.array_equal(ids[v], edges.building_j.to_numpy(np.int64))
    ):
        raise RuntimeError("Active edge endpoint is absent from building IDs")
    graph = sparse.coo_matrix(
        (np.ones(2 * len(u), dtype=np.int8), (np.r_[u, v], np.r_[v, u])),
        shape=(len(ids), len(ids)),
    ).tocsr()
    _, labels = sparse.csgraph.connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    qualifying = np.zeros(len(ids), dtype=bool)
    qualifying[np.searchsorted(ids, np.asarray(qualifying_ids, np.int64))] = True
    assignments = pd.DataFrame({
        "building_id": ids,
        "component_id": labels.astype(np.int32),
        "component_size": sizes[labels].astype(np.int32),
        "qualifies": qualifying,
    })
    assignments.to_parquet(paths["assignments"], index=False, compression="zstd")

    size_values, component_counts = np.unique(sizes, return_counts=True)
    pd.DataFrame({
        "component_size": size_values,
        "component_count": component_counts,
        "buildings": size_values * component_counts,
    }).to_csv(paths["sizes"], index=False)
    summary = pd.DataFrame([{
        "region": "California WUI sample",
        "method": method,
        "threshold": float(threshold),
        "neighbors_summed": n_neighbors if method == "top2" else np.nan,
        "edge_floor_fraction": (
            edge_floor_fraction if method == "top2" else np.nan
        ),
        "individual_edge_floor": (
            edge_floor_fraction * threshold if method == "top2" else np.nan
        ),
        "nodes": len(ids),
        "candidate_pairs": int(candidate_pairs),
        "qualifying_buildings": int(qualifying.sum()),
        "active_edges": len(edges),
        "all_node_components": len(sizes),
        "connected_components": int(np.count_nonzero(sizes > 1)),
        "largest_component": int(sizes.max()),
    }])
    summary.to_csv(paths["summary"], index=False)
    return paths


def interface_buffer_sensitivity(
    assignments_path: Path,
    centroids_path: Path,
    interface: gpd.GeoDataFrame,
    county_boundary: gpd.GeoDataFrame,
    *,
    buffer_miles: tuple[float, ...] = (0, .5, 1, 1.5, 2),
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """Measure fixed SEN components interacting with expanded Interface zones.

    The graph (EEAT or cumulative Top-N) and its component labels remain
    fixed. At each distance, this function expands the mapped Interface
    geometry outward, clips it to the county, and reclassifies which
    components have at least one county building centroid inside that zone.
    """
    assignments_path, centroids_path = map(Path, (assignments_path, centroids_path))
    distances = tuple(float(value) for value in buffer_miles)
    if not distances or any(value < 0 for value in distances):
        raise ValueError("buffer_miles must contain nonnegative distances")
    if tuple(sorted(set(distances))) != distances:
        raise ValueError("buffer_miles must be unique and increasing")
    if interface.crs is None or county_boundary.crs is None:
        raise ValueError("Interface and county geometries require projected CRS metadata")

    target_crs = interface.crs
    county = county_boundary.to_crs(target_crs)
    county_geometry = county.geometry.union_all()
    xmin, ymin, xmax, ymax = county_geometry.bounds
    local_interface = interface.to_crs(target_crs).cx[xmin:xmax, ymin:ymax]
    interface_geometry = local_interface.geometry.union_all().intersection(
        county_geometry
    )

    con = duckdb.connect()
    try:
        points = con.execute(
            """
            SELECT a.component_id, a.component_size, c.x, c.y
            FROM read_parquet(?) a
            JOIN read_parquet(?) c USING (building_id)
            WHERE c.x BETWEEN ? AND ? AND c.y BETWEEN ? AND ?
            """,
            [str(assignments_path), str(centroids_path), xmin, xmax, ymin, ymax],
        ).fetchdf()
    finally:
        con.close()
    inside_county = intersects_xy(
        county_geometry, points.x.to_numpy(), points.y.to_numpy()
    )
    points = points.loc[inside_county].reset_index(drop=True)
    components = (
        points.groupby("component_id", as_index=False)
        .agg(
            global_component_size=("component_size", "first"),
            la_county_buildings=("component_size", "size"),
            x=("x", "mean"),
            y=("y", "mean"),
        )
    )
    components = components.loc[components.global_component_size.gt(1)].copy()
    component_lookup = components.set_index("component_id")

    summary_rows, detail_frames, buffer_rows = [], [], []
    previous_ids: set[int] = set()
    meters_per_mile = 1609.344
    for miles in distances:
        expanded = interface_geometry.buffer(miles * meters_per_mile).intersection(
            county_geometry
        )
        hit = intersects_xy(expanded, points.x.to_numpy(), points.y.to_numpy())
        hit_counts = points.loc[hit, "component_id"].value_counts()
        interacting_ids = set(hit_counts.index).intersection(component_lookup.index)
        captured = component_lookup.loc[list(interacting_ids)].reset_index()
        captured["buildings_inside_expanded_interface"] = (
            captured.component_id.map(hit_counts).astype(np.int64)
        )
        captured["interface_expansion_miles"] = miles
        captured["newly_interacting"] = captured.component_id.isin(
            interacting_ids.difference(previous_ids)
        )
        detail_frames.append(captured)
        newly_captured = captured.loc[captured.newly_interacting]
        summary_rows.append({
            "interface_expansion_miles": miles,
            "expanded_interface_area_km2": expanded.area / 1_000_000,
            "buildings_inside_expanded_interface": int(hit.sum()),
            "interacting_connected_components": len(captured),
            "interacting_components_ge_10": int(
                captured.global_component_size.ge(10).sum()
            ),
            "interacting_components_ge_25": int(
                captured.global_component_size.ge(25).sum()
            ),
            "interacting_components_ge_50": int(
                captured.global_component_size.ge(50).sum()
            ),
            "interacting_components_ge_100": int(
                captured.global_component_size.ge(100).sum()
            ),
            "la_buildings_in_interacting_components": int(
                captured.la_county_buildings.sum()
            ),
            "largest_interacting_global_component": int(
                captured.global_component_size.max() if len(captured) else 0
            ),
            "largest_interacting_la_component": int(
                captured.la_county_buildings.max() if len(captured) else 0
            ),
            "newly_interacting_components": len(newly_captured),
            "largest_new_global_component": int(
                newly_captured.global_component_size.max()
                if len(newly_captured) else 0
            ),
        })
        buffer_rows.append({
            "interface_expansion_miles": miles,
            "geometry": expanded,
        })
        previous_ids = interacting_ids

    detail = pd.concat(detail_frames, ignore_index=True)
    buffers = gpd.GeoDataFrame(buffer_rows, geometry="geometry", crs=target_crs)
    return pd.DataFrame(summary_rows), detail, buffers


eeat_interface_buffer_sensitivity = interface_buffer_sensitivity


def build_fire_dose_2d(run_dir: Path, analysis_path: Path, output_path: Path) -> Path:
    """Destroyed-neighbor 2D dose per assessed structure, matched on LARIAC
    ``BLD_ID`` (the query of the original 2D fragility notebook), for
    ``fit_2d_fragility``."""
    run_dir, output_path = Path(run_dir), Path(output_path)
    q = lambda path: str(path).replace("'", "''")
    ids, edges = q(run_dir / "building_ids.parquet"), q(run_dir / "building_edges.parquet")
    with duckdb.connect() as con:
        con.execute(f"""COPY (
            WITH outcomes AS (
                SELECT CAST(BLD_ID AS VARCHAR) AS BLD_ID, fire, outcome, is_destroyed, grid_id
                FROM read_parquet('{q(analysis_path)}')),
            id_map AS (
                SELECT building_id, CAST(source_building_id AS VARCHAR) AS source_id
                FROM read_parquet('{ids}') WHERE source_building_id IS NOT NULL),
            matched AS (SELECT o.*, m.building_id AS graph_id FROM outcomes o
                        JOIN id_map m ON o.BLD_ID = m.source_id),
            directed AS (
                SELECT building_i AS emitter, building_j AS receiver, vf_i_to_j AS vf FROM read_parquet('{edges}')
                UNION ALL SELECT building_j, building_i, vf_j_to_i FROM read_parquet('{edges}')),
            dose AS (
                SELECT r.graph_id, COUNT(d.emitter) AS graph_neighbors,
                       COUNT(e.graph_id) AS outcome_known_neighbors,
                       COUNT(*) FILTER (WHERE e.is_destroyed = 1) AS destroyed_neighbors,
                       SUM(CASE WHEN e.is_destroyed = 1 THEN d.vf ELSE 0 END) AS dose_sum,
                       MAX(CASE WHEN e.is_destroyed = 1 THEN d.vf ELSE 0 END) AS dose_max_pair
                FROM matched r LEFT JOIN directed d ON r.graph_id = d.receiver
                LEFT JOIN matched e ON d.emitter = e.graph_id GROUP BY r.graph_id)
            SELECT m.*, COALESCE(d.graph_neighbors, 0) AS graph_neighbors,
                   COALESCE(d.outcome_known_neighbors, 0) AS outcome_known_neighbors,
                   COALESCE(d.destroyed_neighbors, 0) AS destroyed_neighbors,
                   COALESCE(d.dose_sum, 0) AS dose_sum,
                   COALESCE(d.dose_max_pair, 0) AS dose_max_pair
            FROM matched m LEFT JOIN dose d USING (graph_id)
        ) TO '{q(output_path)}' (FORMAT PARQUET)""")
    return output_path
