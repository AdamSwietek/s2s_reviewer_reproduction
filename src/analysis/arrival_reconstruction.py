"""Reconstruction of the packaged per-building fire-arrival estimates.

The distributed `data/arrival/arrival_interp.parquet` and
`data/arrival/eaton_arrival_posterior_10m.tif` are derived products. This
module rebuilds both from the packaged primary inputs so a reviewer can
regenerate the arrival clock rather than accept it as a frozen field:

1. Cumulative-front signed-distance interpolation of the progression
   isochrones for both fires, giving `T_arrival_interp_hrs`.
2. Origin-anchored regression kriging of the Eaton timeline events over the
   interpolated drift, giving the three-band 10-m posterior raster.
3. Assembly of `T_arrival_hrs` — the kriged posterior where it covers a
   building, the interpolation elsewhere — and reconciliation against the
   packaged fields.

The rebuilt raster is written beside the frozen one under `data/derived/`;
the distributed raster is never overwritten.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, map_coordinates, minimum_filter
from scipy.stats import spearmanr
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Shared with the block-validation module so both use one specification of the
# origin, spread bearing and observation noise.
from src.analysis.population_scene_arrival import (
    BEARING_DEG,
    NUGGET_H,
    ORIGIN_LL,
    ORIGIN_SD_H,
    _coverage_inflation,
    _crps,
    _rotated,
)

FIRES = ("EATON", "PALISADES")
METRIC_EPSG = 32611
RASTER_RES_M = 10.
RING_SPACING_M = 150.
DEDUP_CELL_M = 30.
DRIFT_SMOOTH_CELLS = 4.
PREDICT_CHUNK = 20_000
# Ignition origin for the first arrival interval; None falls back to the
# representative point of the first cumulative front.
IGNITION_PT = {"EATON": (-118.098, 34.190), "PALISADES": None}
LENGTH_SCALE_SETTINGS = {
    "optimized": None,
    "2,000/1,000 m": (2000., 1000.),
    "4,000/2,000 m": (4000., 2000.),
}
CRPS_TOLERANCE_H = .05


def load_building_spine(project_root: Path) -> pd.DataFrame:
    """Rebuild the LOS building spine whose row order indexes arrival tables."""
    frames = []
    for fire in FIRES:
        scene = gpd.read_parquet(
            Path(project_root) / "data" / "nx" / f"{fire}_buildings.parquet",
            columns=["BLD_ID", "geometry"],
        )
        centroid = scene.geometry.centroid.to_crs(4326)
        frames.append(pd.DataFrame({
            "BLD_ID": scene.BLD_ID.astype(str).to_numpy(),
            "fire": fire,
            "lon_wgs84": centroid.x.to_numpy(),
            "lat_wgs84": centroid.y.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def assign_isochrones(lons: np.ndarray, lats: np.ndarray,
                      isochrones: gpd.GeoDataFrame,
                      ignition_lonlat: tuple[float, float] | None = None,
                      index=None) -> pd.DataFrame:
    """Cumulative-front signed-distance arrival interpolation (WGS84 inputs).

    Isochrones arrive either as incremental burn tiles (official progression
    maps, where each polygon is the area newly burned in that interval) or as
    per-pass active-fire hulls (VIIRS). Interpolating against raw previous
    polygons breaks for disjoint increments, so fronts are unioned into
    cumulative perimeters, nested by construction. A building's arrival
    interval is the first cumulative front containing it, and within that
    interval arrival is interpolated by signed distance to the bracketing
    fronts, so buildings deeper inside rank earlier. Dwell fields keep using
    the raw per-pass hulls.
    """
    points = shapely.points(lons, lats)
    n_buildings = len(points)

    ordered = isochrones.sort_values("acq_dt").reset_index(drop=True)
    hours = ordered.hrs_since_ignition.to_numpy(float)
    sources = ordered.source.to_numpy()
    n_rings = len(ordered)

    cumulative, accumulated = [], None
    for geometry in ordered.geometry:
        accumulated = (geometry if accumulated is None
                       else shapely.union_all([accumulated, geometry]))
        cumulative.append(shapely.make_valid(accumulated))
    cumulative_boundary = [front.boundary for front in cumulative]

    in_cumulative = np.zeros((n_buildings, n_rings), dtype=bool)
    in_raw = np.zeros((n_buildings, n_rings), dtype=bool)
    for ring in range(n_rings):
        in_cumulative[:, ring] = shapely.contains(cumulative[ring], points)
        in_raw[:, ring] = shapely.contains(ordered.geometry.iloc[ring], points)

    covered = in_cumulative.any(axis=1)
    arrival_ring = np.where(covered, in_cumulative.argmax(axis=1), -1)

    if ignition_lonlat is None:
        origin = cumulative[0].representative_point()
        ignition_lon, ignition_lat = origin.x, origin.y
    else:
        ignition_lon, ignition_lat = ignition_lonlat

    interpolated = np.full(n_buildings, np.nan)
    for ring in range(n_rings):
        selected = arrival_ring == ring
        if not selected.any():
            continue
        ring_points = points[selected]
        distance_in = shapely.distance(cumulative_boundary[ring], ring_points)
        if ring == 0:
            distance_out = np.hypot(lons[selected] - ignition_lon,
                                    lats[selected] - ignition_lat)
            previous_hours = 0.
        else:
            distance_out = shapely.distance(
                cumulative_boundary[ring - 1], ring_points)
            previous_hours = hours[ring - 1]
        total = distance_out + distance_in
        fraction = np.where(total > 1e-12, distance_out / total, .5)
        interpolated[selected] = (
            previous_hours + fraction * (hours[ring] - previous_hours))

    active_passes = in_raw.sum(axis=1)
    last_raw = n_rings - 1 - in_raw[:, ::-1].argmax(axis=1)
    ring_index = np.clip(arrival_ring, 0, n_rings - 1)
    snapped = hours[ring_index]
    previous = np.where(arrival_ring > 0,
                        hours[np.clip(arrival_ring - 1, 0, n_rings - 1)], 0.)

    return pd.DataFrame({
        "T_arrival_snap_hrs": np.where(covered, snapped, np.nan),
        "T_arrival_interp_hrs": interpolated,
        "arrival_window_hrs": np.where(covered, snapped - previous, np.nan),
        "T_exit_hrs": np.where(
            covered & (active_passes > 0), hours[last_raw],
            np.where(covered, snapped, np.nan)),
        "n_rings": np.where(covered, np.maximum(active_passes, 1), 0),
        "arrival_source": np.where(covered, sources[ring_index], "none"),
    }, index=index)


def interpolate_building_arrival(project_root: Path,
                                 spine: pd.DataFrame | None = None
                                 ) -> pd.DataFrame:
    """Interpolate arrival for every building from the packaged isochrones."""
    root = Path(project_root)
    spine = load_building_spine(root) if spine is None else spine
    parts = []
    for fire, group in spine.groupby("fire", sort=False):
        isochrones = gpd.read_file(
            root / "data" / "arrival" / f"isochrones_{fire.lower()}.gpkg")
        arrival = assign_isochrones(
            group.lon_wgs84.to_numpy(float), group.lat_wgs84.to_numpy(float),
            isochrones, IGNITION_PT.get(fire), index=group.index)
        parts.append(group.join(arrival))
    return pd.concat(parts).reindex(spine.index)


def summarize_interpolation(interpolated: pd.DataFrame) -> pd.DataFrame:
    """Ring coverage and interpolated arrival, by fire and isochrone source."""
    rows = []
    for fire, group in interpolated.groupby("fire", sort=False):
        covered = group.T_arrival_snap_hrs.notna()
        rows.append({
            "fire": fire,
            "buildings": len(group),
            "covered_by_isochrones": int(covered.sum()),
            "coverage_share": float(covered.mean()),
            "median_interp_h": float(group.T_arrival_interp_hrs.median()),
            "median_window_h": float(group.arrival_window_hrs.median()),
            "median_dwell_rings": float(group.n_rings.median()),
            "sources": ", ".join(sorted(
                group.loc[covered, "arrival_source"].astype(str).unique())),
        })
    return pd.DataFrame(rows)


def _grid_specification(project_root: Path):
    """Eaton perimeter, 10-m grid geometry and inside mask."""
    perimeters = gpd.read_parquet(
        Path(project_root) / "data" / "nx" / "fire_perims.parquet")
    perimeter = (perimeters[perimeters.FIRE_NAME.eq("EATON")]
                 .to_crs(METRIC_EPSG).geometry.union_all())
    minx, miny, maxx, maxy = perimeter.bounds
    minx = np.floor(minx / RASTER_RES_M) * RASTER_RES_M
    miny = np.floor(miny / RASTER_RES_M) * RASTER_RES_M
    maxx = np.ceil(maxx / RASTER_RES_M) * RASTER_RES_M
    maxy = np.ceil(maxy / RASTER_RES_M) * RASTER_RES_M
    width = int((maxx - minx) / RASTER_RES_M)
    height = int((maxy - miny) / RASTER_RES_M)
    transform = from_origin(minx, maxy, RASTER_RES_M, RASTER_RES_M)
    grid_x, grid_y = np.meshgrid(
        minx + (np.arange(width) + .5) * RASTER_RES_M,
        maxy - (np.arange(height) + .5) * RASTER_RES_M,
    )
    inside = ~geometry_mask([perimeter], out_shape=(height, width),
                            transform=transform, invert=False)
    return perimeter, transform, (minx, maxy), (height, width), grid_x, grid_y, inside


def _drift_surface(project_root: Path, interpolated: pd.DataFrame,
                   origin_xy: np.ndarray, grid_x: np.ndarray,
                   grid_y: np.ndarray) -> np.ndarray:
    """Smoothed first-stage drift from origin, synthetic rings and structures."""
    root = Path(project_root)
    eaton = interpolated[
        interpolated.fire.eq("EATON") & interpolated.T_arrival_interp_hrs.notna()]
    structures = gpd.GeoSeries(
        gpd.points_from_xy(eaton.lon_wgs84, eaton.lat_wgs84), crs=4326
    ).to_crs(METRIC_EPSG)

    rings = gpd.read_file(root / "data" / "arrival" / "isochrones_eaton.gpkg")
    rings = rings[rings.fire.eq("EATON")
                  & rings.source.eq("SYNTHETIC")].to_crs(METRIC_EPSG)
    ring_x, ring_y, ring_hours = [], [], []
    for ring in rings.itertuples():
        boundary = ring.geometry.boundary
        for distance in np.arange(0, boundary.length, RING_SPACING_M):
            point = boundary.interpolate(distance)
            ring_x.append(point.x)
            ring_y.append(point.y)
            ring_hours.append(ring.hrs_since_ignition)

    anchors = np.column_stack([
        np.concatenate([[origin_xy[0]], ring_x, structures.x.to_numpy()]),
        np.concatenate([[origin_xy[1]], ring_y, structures.y.to_numpy()]),
    ])
    values = np.concatenate([
        [0.], ring_hours, eaton.T_arrival_interp_hrs.to_numpy(float)])

    drift = griddata(anchors, values, (grid_x.ravel(), grid_y.ravel()),
                     method="linear").reshape(grid_x.shape)
    gaps = np.isnan(drift)
    if gaps.any():
        drift[gaps] = griddata(anchors, values, (grid_x[gaps], grid_y[gaps]),
                               method="nearest")
    return gaussian_filter(drift, sigma=DRIFT_SMOOTH_CELLS).astype("float32")


def _timeline_events(project_root: Path, perimeter) -> gpd.GeoDataFrame:
    """First-arrival timeline events inside the Eaton perimeter."""
    recovered = pd.read_parquet(
        Path(project_root) / "data" / "arrival"
        / "eaton_fire_events_recovered.parquet")
    events = gpd.GeoDataFrame(
        recovered,
        geometry=gpd.points_from_xy(recovered.lon, recovered.lat),
        crs=4326,
    ).to_crs(METRIC_EPSG)
    events = events[events.hrs_since_ignition.ge(0)
                    & events.within(perimeter.buffer(150))].copy()
    events["cell_x"] = (events.geometry.x // DEDUP_CELL_M).astype(int)
    events["cell_y"] = (events.geometry.y // DEDUP_CELL_M).astype(int)
    return (events.sort_values("hrs_since_ignition")
            .drop_duplicates(["cell_x", "cell_y"], keep="first")
            .reset_index(drop=True))


def _krige(train_xy: np.ndarray, train_t: np.ndarray, train_var: np.ndarray,
           predict_xy: np.ndarray, sample_drift, origin_xy: np.ndarray,
           length_scale: tuple[float, float] | None):
    """Regression kriging: drift + distance-from-origin trend + anisotropic GP."""
    residual = train_t - sample_drift(train_xy)
    distance = np.linalg.norm(train_xy - origin_xy, axis=1)
    design = np.column_stack([np.ones(len(distance)), distance])
    beta, *_ = np.linalg.lstsq(design, residual, rcond=None)
    gp_residual = residual - design @ beta

    if length_scale is None:
        matern = Matern(length_scale=[3000., 1500.],
                        length_scale_bounds=(500., 20000.), nu=1.5)
        restarts = 2
    else:
        matern = Matern(length_scale=list(length_scale),
                        length_scale_bounds="fixed", nu=1.5)
        restarts = 0
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(3., (1e-2, 1e3)) * matern, alpha=train_var,
        normalize_y=False, n_restarts_optimizer=restarts, random_state=0,
    ).fit(_rotated(train_xy, origin_xy), gp_residual)

    predict_distance = np.linalg.norm(predict_xy - origin_xy, axis=1)
    trend = np.column_stack(
        [np.ones(len(predict_distance)), predict_distance]) @ beta
    rotated = _rotated(predict_xy, origin_xy)
    mean = np.empty(len(predict_xy))
    sd = np.empty(len(predict_xy))
    for start in range(0, len(predict_xy), PREDICT_CHUNK):
        stop = start + PREDICT_CHUNK
        mean[start:stop], sd[start:stop] = gp.predict(
            rotated[start:stop], return_std=True)
    return sample_drift(predict_xy) + trend + mean, sd, gp.kernel_, beta


def rebuild_eaton_posterior(project_root: Path, interpolated: pd.DataFrame,
                            output_path: Path, n_folds: int = 10
                            ) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Refit the Eaton arrival posterior and write a rebuilt 10-m raster.

    Returns the raster path, the length-scale selection table (spatial block
    CV with nested coverage calibration) and physical sanity diagnostics.
    """
    root = Path(project_root)
    (perimeter, transform, (minx, maxy), (height, width),
     grid_x, grid_y, inside) = _grid_specification(root)
    origin = gpd.GeoSeries([Point(*ORIGIN_LL)], crs=4326).to_crs(METRIC_EPSG).iloc[0]
    origin_xy = np.array([origin.x, origin.y])

    drift = _drift_surface(root, interpolated, origin_xy, grid_x, grid_y)

    def sample_drift(xy: np.ndarray) -> np.ndarray:
        columns = (xy[:, 0] - minx) / RASTER_RES_M - .5
        rows = (maxy - xy[:, 1]) / RASTER_RES_M - .5
        return map_coordinates(drift, np.vstack([rows, columns]), order=1,
                               mode="nearest")

    events = _timeline_events(root, perimeter)
    event_xy = np.column_stack([events.geometry.x, events.geometry.y])
    event_t = events.hrs_since_ignition.to_numpy(float)
    event_var = np.full(len(events), NUGGET_H ** 2)
    folds = KMeans(n_clusters=n_folds, n_init=5,
                   random_state=0).fit_predict(event_xy)

    selection_rows, fold_predictions = [], {}
    for name, length_scale in LENGTH_SCALE_SETTINGS.items():
        mean = np.full(len(events), np.nan)
        sd = np.full(len(events), np.nan)
        for fold in range(n_folds):
            train, test = folds != fold, folds == fold
            train_xy = np.vstack([event_xy[train], origin_xy])
            train_t = np.append(event_t[train], 0.)
            train_var = np.append(event_var[train], ORIGIN_SD_H ** 2)
            mean[test], sd[test], _, _ = _krige(
                train_xy, train_t, train_var, event_xy[test], sample_drift,
                origin_xy, length_scale)
        fold_predictions[name] = (mean, sd)
        selection_rows.append({
            "model": name,
            "n_events": len(events),
            "crps_h": _crps(event_t, mean, sd),
            "rmse_h": float(np.sqrt(np.mean((event_t - mean) ** 2))),
            "mae_h": float(np.mean(np.abs(event_t - mean))),
            "bias_h": float(np.mean(mean - event_t)),
        })
    selection = pd.DataFrame(selection_rows)

    optimized_crps = float(
        selection.loc[selection.model.eq("optimized"), "crps_h"].iloc[0])
    eligible = selection[selection.model.ne("optimized")
                         & selection.crps_h.le(optimized_crps + CRPS_TOLERANCE_H)]
    chosen = (eligible.sort_values("crps_h").model.iloc[0]
              if len(eligible) else "optimized")
    mean, sd = fold_predictions[chosen]
    sigma_mu = _coverage_inflation(event_t, mean, sd)
    covered = np.abs(event_t - mean) <= 1.645 * np.sqrt(sd ** 2 + sigma_mu ** 2)
    nested = []
    for fold in range(n_folds):
        train, test = folds != fold, folds == fold
        fold_sigma = _coverage_inflation(event_t[train], mean[train], sd[train])
        nested.append(np.abs(event_t[test] - mean[test])
                      <= 1.645 * np.sqrt(sd[test] ** 2 + fold_sigma ** 2))
    selection["selected"] = selection.model.eq(chosen)
    selection["sigma_mu_h"] = np.where(selection.selected, sigma_mu, np.nan)
    selection["calibrated_90_coverage"] = np.where(
        selection.selected, float(np.mean(covered)), np.nan)
    selection["nested_cv_90_coverage"] = np.where(
        selection.selected, float(np.mean(np.concatenate(nested))), np.nan)

    # Production fit on every event plus the origin anchor.
    train_xy = np.vstack([event_xy, origin_xy])
    train_t = np.append(event_t, 0.)
    train_var = np.append(event_var, ORIGIN_SD_H ** 2)
    grid_xy = np.column_stack([grid_x[inside], grid_y[inside]])
    posterior_mean, posterior_sd, kernel, beta = _krige(
        train_xy, train_t, train_var, grid_xy, sample_drift, origin_xy,
        LENGTH_SCALE_SETTINGS[chosen])

    prior = np.where(inside, drift, np.nan).astype("float32")
    mean_grid = np.full((height, width), np.nan, "float32")
    mean_grid[inside] = np.clip(posterior_mean, 0, None)
    sd_grid = np.full((height, width), np.nan, "float32")
    sd_grid[inside] = posterior_sd
    total_sd = np.sqrt(sd_grid ** 2 + sigma_mu ** 2).astype("float32")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w", driver="GTiff", height=height, width=width, count=3,
        dtype="float32", crs=pyproj.CRS.from_epsg(METRIC_EPSG).to_wkt(),
        transform=transform, nodata=np.nan, compress="lzw",
    ) as raster:
        raster.write(prior, 1)
        raster.set_band_description(1, "prior_mu_hrs")
        raster.write(mean_grid, 2)
        raster.set_band_description(2, "posterior_mean_hrs")
        raster.write(total_sd, 3)
        raster.set_band_description(3, "posterior_sd_hrs")

    bearing = np.deg2rad(BEARING_DEG)
    along_wind = ((grid_x - origin_xy[0]) * np.sin(bearing)
                  + (grid_y - origin_xy[1]) * np.cos(bearing))
    gradient_y, gradient_x = np.gradient(mean_grid, RASTER_RES_M, RASTER_RES_M)
    spread_kmh = (1. / np.maximum(np.hypot(gradient_x, gradient_y), 1e-9)) / 1000.
    finite_spread = spread_kmh[inside & np.isfinite(spread_kmh)]
    local_minimum = (mean_grid == minimum_filter(
        np.where(np.isnan(mean_grid), np.inf, mean_grid), size=5)) & inside
    sanity = pd.DataFrame([{
        "n_events": len(events),
        "length_scale": chosen,
        "kernel": str(kernel),
        "trend_intercept_h": float(beta[0]),
        "trend_slope_h_per_m": float(beta[1]),
        "sigma_mu_h": sigma_mu,
        "grid_cells_inside": int(inside.sum()),
        "spearman_along_wind": float(
            spearmanr(mean_grid[inside], along_wind[inside]).statistic),
        "median_spread_kmh": float(np.median(finite_spread)),
        "p95_spread_kmh": float(np.percentile(finite_spread, 95)),
        "local_minimum_share": float(np.mean(local_minimum[inside])),
        "max_arrival_h": float(np.nanmax(mean_grid)),
        "median_posterior_minus_prior_h": float(
            np.nanmedian(np.abs(mean_grid - prior))),
    }])
    return output_path, selection, sanity


def compare_posterior_rasters(rebuilt_path: Path, frozen_path: Path
                              ) -> pd.DataFrame:
    """Per-band agreement between the rebuilt and distributed rasters."""
    with rasterio.open(rebuilt_path) as rebuilt, rasterio.open(frozen_path) as frozen:
        same_grid = (rebuilt.shape == frozen.shape
                     and np.allclose(np.asarray(rebuilt.transform)[:6],
                                     np.asarray(frozen.transform)[:6]))
        if not same_grid:
            raise AssertionError(
                "rebuilt raster grid differs from the distributed raster: "
                f"{rebuilt.shape} at {rebuilt.transform} vs "
                f"{frozen.shape} at {frozen.transform}")
        rows = []
        for band, label in enumerate(
            ("prior_mu", "posterior_mean", "posterior_sd"), start=1
        ):
            new = rebuilt.read(band).astype(float)
            old = frozen.read(band).astype(float)
            shared = np.isfinite(new) & np.isfinite(old)
            difference = np.abs(new[shared] - old[shared])
            rows.append({
                "band": label,
                "same_grid": bool(same_grid),
                "compared_cells": int(shared.sum()),
                "median_abs_diff_h": float(np.median(difference)),
                "p95_abs_diff_h": float(np.percentile(difference, 95)),
                "max_abs_diff_h": float(difference.max()),
                "spearman": float(
                    spearmanr(new[shared], old[shared]).statistic),
            })
    return pd.DataFrame(rows)


def sample_posterior(raster_path: Path, lon, lat) -> np.ndarray:
    """Sample posterior mean and sd at WGS84 points; NaN outside the clip."""
    with rasterio.open(raster_path) as raster:
        transformer = pyproj.Transformer.from_crs(
            4326, raster.crs.to_wkt(), always_xy=True)
        x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
        values = np.array(
            list(raster.sample(np.column_stack([x, y]), indexes=[2, 3])),
            dtype=float)
        nodata = raster.nodata
    if nodata is not None and np.isfinite(nodata):
        values[values == nodata] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def assemble_arrival(interpolated: pd.DataFrame, posterior_path: Path
                     ) -> pd.DataFrame:
    """Combine the kriged posterior and the interpolation into T_arrival_hrs."""
    arrival = interpolated.copy()
    kriged_mean = np.full(len(arrival), np.nan)
    kriged_sd = np.full(len(arrival), np.nan)
    eaton = arrival.fire.eq("EATON").to_numpy()
    if eaton.any():
        sampled = sample_posterior(
            posterior_path, arrival.loc[eaton, "lon_wgs84"].to_numpy(float),
            arrival.loc[eaton, "lat_wgs84"].to_numpy(float))
        kriged_mean[eaton], kriged_sd[eaton] = sampled[:, 0], sampled[:, 1]
    has_kriged = np.isfinite(kriged_mean)
    arrival["T_arrival_hrs"] = np.where(
        has_kriged, kriged_mean, arrival.T_arrival_interp_hrs)
    arrival["T_arrival_sd_hrs"] = kriged_sd
    arrival["arrival_method"] = np.where(
        has_kriged, "kriged",
        np.where(arrival.arrival_source.eq("none"), "none",
                 "interp:" + arrival.arrival_source.astype(str)))
    return arrival.drop(columns="arrival_source")


def summarize_assignment(arrival: pd.DataFrame) -> pd.DataFrame:
    """Assignment counts by fire and method, matching the packaged summary."""
    rows = []
    for fire, group in arrival.groupby("fire", observed=True):
        method = group.arrival_method.fillna("unassigned").astype(str)
        rows.append({
            "fire": fire,
            "buildings": len(group),
            "assigned": int(group.T_arrival_hrs.notna().sum()),
            "boundary_interpolation": int(method.str.startswith("interp:").sum()),
            "regression_kriging": int(method.eq("kriged").sum()),
            "unassigned": int(group.T_arrival_hrs.isna().sum()),
            "median_arrival_h": float(group.T_arrival_hrs.median()),
            "median_sd_h": float(group.T_arrival_sd_hrs.median()),
        })
    return pd.DataFrame(rows)


def compare_with_packaged(project_root: Path, arrival: pd.DataFrame
                          ) -> pd.DataFrame:
    """Reconcile the reconstruction against the distributed arrival fields."""
    packaged = pd.read_parquet(
        Path(project_root) / "data" / "arrival" / "arrival_interp.parquet")
    if len(packaged) != len(arrival):
        raise AssertionError(
            f"row mismatch: packaged {len(packaged)}, rebuilt {len(arrival)}")
    offset = np.hypot(packaged.lon_wgs84.to_numpy() - arrival.lon_wgs84.to_numpy(),
                      packaged.lat_wgs84.to_numpy() - arrival.lat_wgs84.to_numpy())
    if offset.max() > 1e-9:
        raise AssertionError("rebuilt spine is not row-aligned with the package")

    rows = []
    for field in ("T_arrival_interp_hrs", "T_arrival_hrs", "T_arrival_sd_hrs"):
        new = arrival[field].to_numpy(float)
        old = packaged[field].to_numpy(float)
        shared = np.isfinite(new) & np.isfinite(old)
        difference = np.abs(new[shared] - old[shared])
        rows.append({
            "field": field,
            "packaged_non_null": int(np.isfinite(old).sum()),
            "rebuilt_non_null": int(np.isfinite(new).sum()),
            "compared": int(shared.sum()),
            "median_abs_diff_h": float(np.median(difference)),
            "p95_abs_diff_h": float(np.percentile(difference, 95)),
            "max_abs_diff_h": float(difference.max()),
            "spearman": float(spearmanr(new[shared], old[shared]).statistic),
            "exact_match_share": float(np.mean(difference == 0)),
        })
    method_match = (arrival.arrival_method.to_numpy()
                    == packaged.arrival_method.to_numpy())
    rows.append({
        "field": "arrival_method",
        "packaged_non_null": int(packaged.arrival_method.notna().sum()),
        "rebuilt_non_null": int(arrival.arrival_method.notna().sum()),
        "compared": len(arrival),
        "median_abs_diff_h": np.nan,
        "p95_abs_diff_h": np.nan,
        "max_abs_diff_h": np.nan,
        "spearman": np.nan,
        "exact_match_share": float(np.mean(method_match)),
    })
    return pd.DataFrame(rows)


def build_arrival_reconstruction(project_root: Path, results_dir: Path,
                                 derived_dir: Path | None = None
                                 ) -> dict[str, pd.DataFrame]:
    """Rebuild arrival end to end and persist reconciliation tables."""
    root = Path(project_root)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    derived = Path(derived_dir) if derived_dir else root / "data" / "derived"

    interpolated = interpolate_building_arrival(root)
    rebuilt_path, selection, sanity = rebuild_eaton_posterior(
        root, interpolated, derived / "eaton_arrival_posterior_10m_rebuilt.tif")
    arrival = assemble_arrival(interpolated, rebuilt_path)

    tables = {
        "isochrone_interpolation": summarize_interpolation(interpolated),
        "posterior_selection": selection,
        "posterior_sanity": sanity,
        "posterior_raster_agreement": compare_posterior_rasters(
            rebuilt_path, root / "data" / "arrival"
            / "eaton_arrival_posterior_10m.tif"),
        "reconstructed_assignment": summarize_assignment(arrival),
        "packaged_agreement": compare_with_packaged(root, arrival),
    }
    for name, frame in tables.items():
        frame.to_csv(out / f"ED01_arrival_{name}.csv", index=False)
    arrival.to_parquet(derived / "arrival_reconstructed.parquet", index=False)
    tables["arrival"] = arrival
    return tables
