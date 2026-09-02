"""Build Eaton arrival-map sensitivity inputs for the defense analysis.

Methods b and c are already present in the frozen arrival table. Method d is
reconstructed from advancing isochrone boundaries and timeline observations.
The direct-map area-weighted up-fire exposure requires the excluded patch-level
LOS table and is therefore computed by this release-building utility, then
distributed as a compact building-level column.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


ORIGIN_LL = (-118.0953, 34.1822)
BEARING_DEG = 192.0
NUGGET_H = 2.5
ORIGIN_SD_H = 0.3
CRS = 32611
DEDUP_M = 30
BOUNDARY_SPACING_M = 150.0
EDGE_TOL_M = 25.0
SIGMA_BOUNDARY = {"VIIRS": 1.5, "SYNTHETIC": 3.0}


def rotate(xy: np.ndarray, offset: np.ndarray) -> np.ndarray:
    bearing = np.deg2rad(BEARING_DEG)
    along = np.array([np.sin(bearing), np.cos(bearing)])
    cross = np.array([np.cos(bearing), -np.sin(bearing)])
    shifted = xy - offset
    return np.column_stack([shifted @ along, shifted @ cross])


def design(xy: np.ndarray, origin_xy: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(xy - origin_xy, axis=1)
    return np.column_stack([np.ones(len(distance)), distance])


def direct_arrivals(data: Path) -> pd.DataFrame:
    arrivals = pd.read_parquet(data / "arrival" / "arrival_interp.parquet")
    buildings = arrivals[arrivals.fire.eq("EATON")].reset_index(drop=True)
    points = gpd.GeoSeries(
        gpd.points_from_xy(buildings.lon_wgs84, buildings.lat_wgs84), crs=4326
    ).to_crs(CRS)
    buildings["x"], buildings["y"] = points.x.to_numpy(), points.y.to_numpy()

    perimeters = gpd.read_parquet(data / "nx" / "fire_perims.parquet")
    perimeter = (perimeters[perimeters.FIRE_NAME.eq("EATON")]
                 .to_crs(CRS).geometry.union_all())
    rings = (gpd.read_file(data / "arrival" / "isochrones_eaton.gpkg")
             .to_crs(CRS).sort_values("hrs_since_ignition").reset_index(drop=True))
    recovered = pd.read_parquet(
        data / "arrival" / "eaton_fire_events_recovered.parquet")
    events = gpd.GeoDataFrame(
        recovered,
        geometry=gpd.points_from_xy(recovered.lon, recovered.lat), crs=4326,
    ).to_crs(CRS)
    events = events[
        events.hrs_since_ignition.ge(0) & events.within(perimeter.buffer(150))
    ].copy()
    events["cx"] = (events.geometry.x // DEDUP_M).astype(int)
    events["cy"] = (events.geometry.y // DEDUP_M).astype(int)
    events = (events.sort_values("hrs_since_ignition")
              .drop_duplicates(["cx", "cy"], keep="first").reset_index(drop=True))

    origin_xy = np.asarray(gpd.GeoSeries(
        [Point(*ORIGIN_LL)], crs=4326).to_crs(CRS).iloc[0].coords[0])
    boundary_rows, previous = [], None
    for ring in rings.itertuples():
        cumulative = (ring.geometry if previous is None
                      else unary_union([previous, ring.geometry]))
        edge = (cumulative.boundary if previous is None else
                cumulative.boundary.difference(previous.buffer(EDGE_TOL_M)))
        if not edge.is_empty:
            for distance in np.arange(0, edge.length, BOUNDARY_SPACING_M):
                point = edge.interpolate(distance)
                boundary_rows.append(
                    (point.x, point.y, ring.hrs_since_ignition, ring.source))
        previous = cumulative
    boundary = pd.DataFrame(
        boundary_rows, columns=["x", "y", "t", "source"])
    keep = gpd.GeoSeries(
        gpd.points_from_xy(boundary.x, boundary.y), crs=CRS
    ).within(perimeter.buffer(300)).to_numpy()
    boundary = boundary[keep].reset_index(drop=True)
    boundary["sigma"] = boundary.source.map(SIGMA_BOUNDARY).fillna(2.5)

    event_xy = np.column_stack([events.geometry.x, events.geometry.y])
    boundary_xy = boundary[["x", "y"]].to_numpy()
    train_xy = np.vstack([event_xy, boundary_xy, origin_xy[None, :]])
    train_t = np.concatenate([
        events.hrs_since_ignition.to_numpy(float),
        boundary.t.to_numpy(float), [0.0],
    ])
    train_var = np.concatenate([
        np.full(len(events), NUGGET_H ** 2),
        boundary.sigma.to_numpy(float) ** 2, [ORIGIN_SD_H ** 2],
    ])
    matrix = design(train_xy, origin_xy)
    beta, *_ = np.linalg.lstsq(matrix, train_t, rcond=None)
    residual = train_t - matrix @ beta
    kernel = ConstantKernel(9.0, (1e-2, 1e4)) * Matern(
        [3000.0, 1500.0], length_scale_bounds=(300.0, 40000.0), nu=1.5)
    model = GaussianProcessRegressor(
        kernel, alpha=train_var, normalize_y=False,
        n_restarts_optimizer=2, random_state=0,
    ).fit(rotate(train_xy, origin_xy), residual)
    prediction_xy = buildings[["x", "y"]].to_numpy()
    residual_mean, residual_sd = model.predict(
        rotate(prediction_xy, origin_xy), return_std=True)
    direct = np.clip(design(prediction_xy, origin_xy) @ beta + residual_mean,
                     0, None)

    output = buildings[[
        "BLD_ID", "T_arrival_interp_hrs", "T_arrival_hrs"
    ]].copy()
    output["BLD_ID"] = output.BLD_ID.astype(str)
    output["T_arrival_direct_hrs"] = direct
    output["T_arrival_direct_sd_hrs"] = residual_sd
    output = (output[output.BLD_ID.ne("None")]
              .drop_duplicates("BLD_ID").reset_index(drop=True))
    print("direct GP kernel:", model.kernel_)
    print("Eaton structures with identifiers:", f"{len(output):,}")
    return output


def add_upfire_exposures(output: pd.DataFrame, data: Path,
                         los_path: Path) -> pd.DataFrame:
    enriched = pd.read_parquet(
        data / "buildings_enriched.parquet",
        columns=["BLD_ID", "fire", "is_destroyed"],
    )
    attributes = enriched[enriched.fire.eq("EATON")][
        ["BLD_ID", "is_destroyed"]].copy()
    attributes["BLD_ID"] = attributes.BLD_ID.astype(str)
    attributes = attributes[attributes.BLD_ID.ne("None")].drop_duplicates("BLD_ID")
    attributes = attributes.merge(
        output[["BLD_ID", "T_arrival_interp_hrs", "T_arrival_hrs",
                "T_arrival_direct_hrs"]], on="BLD_ID", how="inner")
    attributes = attributes.rename(columns={"is_destroyed": "destroyed"})

    escaped = str(los_path.resolve()).replace("'", "''")
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    connection.execute("SET memory_limit='6GB'")
    scratch = Path(tempfile.gettempdir()) / "wildfire_arrival_sensitivity_duckdb"
    scratch.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory='{str(scratch)}'")
    connection.register("attributes", attributes)
    exposure = connection.execute(f"""
        WITH valid AS (
            SELECT CAST(l.source_BLD_ID AS VARCHAR) AS BLD_ID,
                   CAST(l.target_BLD_ID AS VARCHAR) AS target_id,
                   l.source_uuid,
                   ANY_VALUE(l.A_recv) AS A_recv,
                   SUM(CASE WHEN t.destroyed = 1
                                  AND t.T_arrival_interp_hrs
                                      < s.T_arrival_interp_hrs
                       THEN LEAST(
                           GREATEST(l.cos_theta_recv, 0.0)
                           * GREATEST(l.cos_theta_emit, 0.0)
                           / (PI() * l.dist * l.dist) * l.A_emit, 1.0)
                       ELSE 0.0 END) AS E_upfire_interp,
                   SUM(CASE WHEN t.destroyed = 1
                                  AND t.T_arrival_hrs < s.T_arrival_hrs
                       THEN LEAST(
                           GREATEST(l.cos_theta_recv, 0.0)
                           * GREATEST(l.cos_theta_emit, 0.0)
                           / (PI() * l.dist * l.dist) * l.A_emit, 1.0)
                       ELSE 0.0 END) AS E_upfire_update,
                   SUM(CASE WHEN t.destroyed = 1
                                  AND t.T_arrival_direct_hrs
                                      < s.T_arrival_direct_hrs
                       THEN LEAST(
                           GREATEST(l.cos_theta_recv, 0.0)
                           * GREATEST(l.cos_theta_emit, 0.0)
                           / (PI() * l.dist * l.dist) * l.A_emit, 1.0)
                       ELSE 0.0 END) AS E_upfire_direct
            FROM read_parquet('{escaped}') AS l
            INNER JOIN attributes AS s
                ON CAST(l.source_BLD_ID AS VARCHAR) = s.BLD_ID
            LEFT JOIN attributes AS t
                ON CAST(l.target_BLD_ID AS VARCHAR) = t.BLD_ID
            WHERE COALESCE(CAST(l.vis AS BOOLEAN), FALSE)
              AND CAST(l.source_BLD_ID AS VARCHAR)
                  <> CAST(l.target_BLD_ID AS VARCHAR)
              AND l.dist > 0 AND l.A_emit > 0
              AND l.cos_theta_recv IS NOT NULL
              AND l.cos_theta_emit IS NOT NULL
            GROUP BY 1, 2, 3
        ), viewpoint AS (
            SELECT BLD_ID, source_uuid,
                   ANY_VALUE(A_recv) AS A_recv,
                   SUM(E_upfire_interp) AS E_upfire_interp,
                   SUM(E_upfire_update) AS E_upfire_update,
                   SUM(E_upfire_direct) AS E_upfire_direct
            FROM valid GROUP BY 1, 2
        )
        SELECT BLD_ID,
               SUM(E_upfire_interp * A_recv) / NULLIF(SUM(A_recv), 0)
                   AS F_upfire_interp_wmean_check,
               SUM(E_upfire_update * A_recv) / NULLIF(SUM(A_recv), 0)
                   AS F_upfire_update_wmean_check,
               SUM(E_upfire_direct) AS F_upfire_direct,
               SUM(E_upfire_direct * A_recv) / NULLIF(SUM(A_recv), 0)
                   AS F_upfire_direct_wmean
        FROM viewpoint GROUP BY 1
    """).fetchdf()
    connection.close()
    result = output.merge(exposure, on="BLD_ID", how="left")

    # Methods b and c were previously aggregated by the production pipeline.
    # Reproducing them from the raw patch table validates the identical
    # aggregation used to create the new method-d exposure column.
    radex = pd.read_parquet(
        data / "radex.parquet",
        columns=["BLD_ID", "fire", "F_upfire_interp_wmean", "F_upfire_wmean"],
    )
    radex["BLD_ID"] = radex.BLD_ID.astype(str)
    check = result.merge(
        radex[radex.fire.eq("EATON")].drop_duplicates("BLD_ID"),
        on="BLD_ID", how="inner",
    )
    comparisons = [
        ("linear", "F_upfire_interp_wmean_check", "F_upfire_interp_wmean"),
        ("kriging update", "F_upfire_update_wmean_check", "F_upfire_wmean"),
    ]
    for label, rebuilt, frozen in comparisons:
        valid = check[[rebuilt, frozen]].dropna()
        max_error = float(np.max(np.abs(valid[rebuilt] - valid[frozen])))
        print(f"{label} exposure validation max |error|: {max_error:.3e}")
        if not np.allclose(valid[rebuilt], valid[frozen], rtol=2e-6, atol=1e-10):
            raise RuntimeError(
                f"Raw LOS aggregation did not reproduce {label} exposure")
    return result.drop(columns=[
        "F_upfire_interp_wmean_check", "F_upfire_update_wmean_check"
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root", type=Path,
        default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--los-path", type=Path, required=True,
        help="Development-only Eaton patch-level LOS Parquet file")
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    output = direct_arrivals(package_root / "data")
    output = add_upfire_exposures(
        output, package_root / "data", args.los_path)
    destination = (package_root / "data" / "arrival" /
                   "eaton_progression_sensitivity.parquet")
    output.to_parquet(destination, index=False)
    print("saved", destination)


if __name__ == "__main__":
    main()
