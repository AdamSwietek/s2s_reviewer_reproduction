"""Population, scene and Eaton arrival-time audits for Extended Data 1."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import kendalltau, norm, spearmanr
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


ORIGIN_LL = (-118.0953, 34.1822)
BEARING_DEG = 192.0
NUGGET_H = 2.5
ORIGIN_SD_H = 0.3


def summarize_arrival_assignment(project_root: Path) -> pd.DataFrame:
    """Count final building arrival assignments by fire and method."""
    arr = pd.read_parquet(Path(project_root) / "data/arrival/arrival_interp.parquet")
    rows = []
    for fire, frame in arr.groupby("fire", observed=True):
        method = frame["arrival_method"].fillna("unassigned").astype(str)
        rows.append({
            "fire": fire,
            "buildings": len(frame),
            "assigned": int(frame.T_arrival_hrs.notna().sum()),
            "boundary_interpolation": int(method.str.startswith("interp:").sum()),
            "regression_kriging": int(method.eq("kriged").sum()),
            "unassigned": int(frame.T_arrival_hrs.isna().sum()),
            "median_arrival_h": float(frame.T_arrival_hrs.median()),
            "median_sd_h": float(frame.T_arrival_sd_hrs.median()),
        })
    return pd.DataFrame(rows)


def summarize_eaton_posterior(project_root: Path) -> pd.DataFrame:
    """Summarize the frozen three-band Eaton posterior raster."""
    path = Path(project_root) / "data/arrival/eaton_arrival_posterior_10m.tif"
    rows = []
    with rasterio.open(path) as src:
        for band, label in enumerate(
            ("baseline_drift", "posterior_mean", "posterior_sd"), start=1
        ):
            values = src.read(band, masked=True).compressed()
            rows.append({
                "surface": label,
                "cells": len(values),
                "median_h": float(np.median(values)),
                "p05_h": float(np.quantile(values, .05)),
                "p95_h": float(np.quantile(values, .95)),
            })
    return pd.DataFrame(rows)


def _eaton_baseline_sampler(root: Path, origin_xy: np.ndarray):
    """Rebuild the unmasked 40-m-smoothed drift used by the production GP."""
    raster_path = root / "data/arrival/eaton_arrival_posterior_10m.tif"
    with rasterio.open(raster_path) as src:
        minx, miny, maxx, maxy = src.bounds
        width, height = src.width, src.height
        res = float(src.res[0])
    gx, gy = np.meshgrid(
        minx + (np.arange(width) + .5) * res,
        maxy - (np.arange(height) + .5) * res,
    )

    buildings = pd.read_parquet(root / "data/arrival/arrival_interp.parquet")
    buildings = buildings[
        buildings.fire.eq("EATON") & buildings.T_arrival_interp_hrs.notna()
    ]
    building_points = gpd.GeoSeries(
        gpd.points_from_xy(buildings.lon_wgs84, buildings.lat_wgs84), crs=4326
    ).to_crs(32611)
    rings = gpd.read_file(root / "data/arrival/isochrones_eaton.gpkg")
    rings = rings[
        rings.fire.eq("EATON") & rings.source.eq("SYNTHETIC")
    ].to_crs(32611)
    ring_x, ring_y, ring_t = [], [], []
    for row in rings.itertuples():
        boundary = row.geometry.boundary
        for distance in np.arange(0, boundary.length, 150.):
            point = boundary.interpolate(distance)
            ring_x.append(point.x); ring_y.append(point.y)
            ring_t.append(row.hrs_since_ignition)
    anchors = np.column_stack([
        np.concatenate([[origin_xy[0]], ring_x, building_points.x.to_numpy()]),
        np.concatenate([[origin_xy[1]], ring_y, building_points.y.to_numpy()]),
    ])
    values = np.concatenate([
        [0.], ring_t, buildings.T_arrival_interp_hrs.to_numpy(float)
    ])
    baseline = griddata(
        anchors, values, (gx.ravel(), gy.ravel()), method="linear"
    ).reshape(height, width)
    miseng = np.isnan(baseline)
    if miseng.any():
        baseline[miseng] = griddata(
            anchors, values, (gx[miseng], gy[miseng]), method="nearest"
        )
    baseline = gaussian_filter(baseline, sigma=4.)

    def sample(xy: np.ndarray) -> np.ndarray:
        columns = (xy[:, 0] - minx) / res - .5
        rows = (maxy - xy[:, 1]) / res - .5
        return map_coordinates(
            baseline, np.vstack([rows, columns]), order=1, mode="nearest"
        )

    return sample


def _prepare_eaton_validation(project_root: Path) -> tuple[pd.DataFrame, np.ndarray, float]:
    root = Path(project_root)
    perims = gpd.read_parquet(root / "data/nx/fire_perims.parquet")
    perimeter = (perims[perims.FIRE_NAME.eq("EATON")]
                 .to_crs(32611).geometry.union_all())
    recovered = pd.read_parquet(
        root / "data/arrival/eaton_fire_events_recovered.parquet"
    )
    events = gpd.GeoDataFrame(
        recovered,
        geometry=gpd.points_from_xy(recovered.lon, recovered.lat),
        crs=4326,
    ).to_crs(32611)
    events = events[
        events.hrs_since_ignition.ge(0) & events.within(perimeter.buffer(150))
    ].copy()
    events["cell_x"] = (events.geometry.x // 30).astype(int)
    events["cell_y"] = (events.geometry.y // 30).astype(int)
    events = (events.sort_values("hrs_since_ignition")
              .drop_duplicates(["cell_x", "cell_y"], keep="first")
              .reset_index(drop=True))

    xy = np.column_stack([events.geometry.x, events.geometry.y])
    origin = gpd.GeoSeries([Point(*ORIGIN_LL)], crs=4326).to_crs(32611).iloc[0]
    origin_xy = np.array([origin.x, origin.y])
    sample_baseline = _eaton_baseline_sampler(root, origin_xy)
    events["baseline_h"] = sample_baseline(xy)
    origin_mu = float(sample_baseline(origin_xy[None, :])[0])
    return events, origin_xy, origin_mu


def _rotated(xy: np.ndarray, offset: np.ndarray) -> np.ndarray:
    bearing = np.deg2rad(BEARING_DEG)
    along = np.array([np.sin(bearing), np.cos(bearing)])
    cross = np.array([np.cos(bearing), -np.sin(bearing)])
    shifted = xy - offset
    return np.column_stack([shifted @ along, shifted @ cross])


def _krige_fold(
    train_xy: np.ndarray,
    train_t: np.ndarray,
    train_mu: np.ndarray,
    train_var: np.ndarray,
    test_xy: np.ndarray,
    test_mu: np.ndarray,
    origin_xy: np.ndarray,
    length_scale: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    residual = train_t - train_mu
    distance = np.linalg.norm(train_xy - origin_xy, axis=1)
    design = np.column_stack([np.ones(len(distance)), distance])
    beta, *_ = np.linalg.lstsq(design, residual, rcond=None)
    gp_residual = residual - design @ beta

    if length_scale is None:
        matern = Matern(
            length_scale=[3000., 1500.],
            length_scale_bounds=(500., 20000.), nu=1.5,
        )
        restarts = 2
    else:
        matern = Matern(length_scale=length_scale,
                        length_scale_bounds="fixed", nu=1.5)
        restarts = 0
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(3., (1e-2, 1e3)) * matern,
        alpha=train_var, normalize_y=False,
        n_restarts_optimizer=restarts, random_state=0,
    ).fit(_rotated(train_xy, origin_xy), gp_residual)

    test_distance = np.linalg.norm(test_xy - origin_xy, axis=1)
    trend = np.column_stack([np.ones(len(test_distance)), test_distance]) @ beta
    mean, sd = gp.predict(_rotated(test_xy, origin_xy), return_std=True)
    return test_mu + trend + mean, sd


def _crps(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> float:
    z = (y - mean) / sd
    return float(np.mean(sd * (
        z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
    )))


def _coverage_inflation(y: np.ndarray, mean: np.ndarray,
                        sd: np.ndarray, target: float = .90) -> float:
    grid = np.linspace(0, 10, 101)
    coverage = [
        np.mean(np.abs(y - mean) <= 1.645 * np.sqrt(sd ** 2 + extra ** 2))
        for extra in grid
    ]
    return float(grid[np.argmin(np.abs(np.asarray(coverage) - target))])


def validate_eaton_regression_kriging(
    project_root: Path, n_folds: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce spatial block CV for the Eaton regression-kriging update."""
    events, origin_xy, origin_mu = _prepare_eaton_validation(project_root)
    xy = np.column_stack([events.geometry.x, events.geometry.y])
    observed = events.hrs_since_ignition.to_numpy(float)
    baseline = events.baseline_h.to_numpy(float)
    folds = KMeans(n_clusters=n_folds, n_init=5, random_state=0).fit_predict(xy)
    settings = {
        "optimized": None,
        "2,000/1,000 m": (2000., 1000.),
        "4,000/2,000 m": (4000., 2000.),
    }
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    summaries = []
    for name, length_scale in settings.items():
        mean = np.full(len(events), np.nan)
        sd = np.full(len(events), np.nan)
        for fold in range(n_folds):
            train, test = folds != fold, folds == fold
            train_xy = np.vstack([xy[train], origin_xy])
            train_t = np.append(observed[train], 0.)
            train_mu = np.append(baseline[train], origin_mu)
            train_var = np.append(np.full(train.sum(), NUGGET_H ** 2), ORIGIN_SD_H ** 2)
            mean[test], sd[test] = _krige_fold(
                train_xy, train_t, train_mu, train_var,
                xy[test], baseline[test], origin_xy, length_scale,
            )
        predictions[name] = (mean, sd)
        summaries.append({
            "model": name,
            "n_events": len(events),
            "crps_h": _crps(observed, mean, sd),
            "rmse_h": float(np.sqrt(np.mean((observed - mean) ** 2))),
            "mae_h": float(np.mean(np.abs(observed - mean))),
            "bias_h": float(np.mean(mean - observed)),
            "spearman": float(spearmanr(observed, mean).statistic),
            "kendall": float(kendalltau(observed, mean).statistic),
        })
    summary = pd.DataFrame(summaries)
    optimized_crps = float(summary.loc[summary.model.eq("optimized"), "crps_h"].iloc[0])
    eligible = summary[
        summary.model.ne("optimized") & summary.crps_h.le(optimized_crps + .05)
    ]
    chosen = (eligible.sort_values("crps_h").model.iloc[0]
              if len(eligible) else "optimized")
    mean, sd = predictions[chosen]
    sigma = _coverage_inflation(observed, mean, sd)
    total_sd = np.sqrt(sd ** 2 + sigma ** 2)
    coverage = float(np.mean(np.abs(observed - mean) <= 1.645 * total_sd))
    nested_covered = []
    for fold in range(n_folds):
        train, test = folds != fold, folds == fold
        fold_sigma = _coverage_inflation(observed[train], mean[train], sd[train])
        nested_covered.append(
            np.abs(observed[test] - mean[test])
            <= 1.645 * np.sqrt(sd[test] ** 2 + fold_sigma ** 2)
        )
    nested_coverage = float(np.mean(np.concatenate(nested_covered)))
    summary["selected"] = summary.model.eq(chosen)
    summary["sigma_mu_h"] = np.where(summary.model.eq(chosen), sigma, np.nan)
    summary["calibrated_90_coverage"] = np.where(
        summary.model.eq(chosen), coverage, np.nan
    )
    summary["nested_cv_90_coverage"] = np.where(
        summary.model.eq(chosen), nested_coverage, np.nan
    )

    prediction = pd.DataFrame({
        "event_id": np.arange(len(events)),
        "fold": folds,
        "observed_h": observed,
        "baseline_h": baseline,
        "predicted_h": mean,
        "posterior_sd_h": sd,
        "calibrated_sd_h": total_sd,
        "longitude": events.to_crs(4326).geometry.x,
        "latitude": events.to_crs(4326).geometry.y,
    })
    baseline_summary = pd.DataFrame([{
        "model": "baseline drift",
        "n_events": len(events),
        "rmse_h": float(np.sqrt(np.mean((observed - baseline) ** 2))),
        "mae_h": float(np.mean(np.abs(observed - baseline))),
        "bias_h": float(np.mean(baseline - observed)),
        "spearman": float(spearmanr(observed, baseline).statistic),
        "kendall": float(kendalltau(observed, baseline).statistic),
    }])
    return summary, prediction, baseline_summary


def build_ed01_audit(project_root: Path, results_dir: Path) -> dict[str, pd.DataFrame]:
    """Build all ED01 result tables and persist machine-readable outputs."""
    root, out = Path(project_root), Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "arrival_assignment": summarize_arrival_assignment(root),
        "eaton_posterior_raster": summarize_eaton_posterior(root),
    }
    cv, prediction, baseline = validate_eaton_regression_kriging(root)
    tables.update({
        "eaton_arrival_cv": cv,
        "eaton_arrival_cv_predictions": prediction,
        "eaton_arrival_baseline": baseline,
    })
    for name, frame in tables.items():
        frame.to_csv(out / f"ED01_{name}.csv", index=False)
    return tables
