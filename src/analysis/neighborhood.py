"""Neighborhood-density sensitivity analysis for Extended Data Figure 2."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.analysis.fragility import FIRES


DEFAULT_RADII_FT = (100, 250, 500, 1000)


def attach_damaged_neighbor_counts(
    analysis: pd.DataFrame,
    radii_ft=DEFAULT_RADII_FT,
) -> pd.DataFrame:
    """Count other damaged structures around every assessed focal structure.

    Coordinates are transformed to EPSG:2229, whose horizontal units are US
    survey feet. A damaged focal structure is explicitly removed from its own
    count; the archived exploratory implementation did not make that correction.
    """
    required = [
        "BLD_ID", "fire", "outcome", "any_damage", "F_destroyed_wmean",
        "lon_wgs84", "lat_wgs84",
    ]
    data = analysis[required].copy()
    rows = []
    for fire in FIRES:
        fire_data = data[data.fire.eq(fire)].copy()
        points = gpd.GeoDataFrame(
            fire_data,
            geometry=gpd.points_from_xy(
                fire_data.lon_wgs84, fire_data.lat_wgs84
            ),
            crs="EPSG:4326",
        ).to_crs("EPSG:2229")
        coordinates = np.column_stack([
            points.geometry.x.to_numpy(), points.geometry.y.to_numpy()
        ])
        damaged = points.any_damage.fillna(0).astype(bool).to_numpy()
        tree = cKDTree(coordinates[damaged])
        for radius in radii_ft:
            counts = tree.query_ball_point(
                coordinates, r=float(radius), workers=-1, return_length=True
            ).astype(int)
            # The query returns the focal point itself when it is damaged.
            points[f"damaged_neighbors_{radius}ft"] = counts - damaged.astype(int)
        rows.append(pd.DataFrame(points.drop(columns="geometry")))
    output = pd.concat(rows, ignore_index=True)
    count_columns = [f"damaged_neighbors_{radius}ft" for radius in radii_ft]
    if output[count_columns].lt(0).any().any():
        raise AssertionError("A damaged-neighbor count is negative")
    return output


def _binned_exposure(data: pd.DataFrame, count_col: str,
                     n_bins: int = 16) -> pd.DataFrame:
    subset = data[
        data[count_col].gt(0) & data.F_destroyed_wmean.gt(0)
    ].copy()
    subset["log_count"] = np.log(subset[count_col])
    subset["log_exposure"] = np.log(subset.F_destroyed_wmean)
    bin_count = min(n_bins, subset.log_count.nunique())
    quantiles = pd.qcut(subset.log_count, q=bin_count, duplicates="drop")
    rows = []
    for _, group in subset.groupby(quantiles, observed=True):
        if len(group) < 15:
            continue
        mean = group.log_exposure.mean()
        standard_error = group.log_exposure.std(ddof=1) / np.sqrt(len(group))
        rows.append({
            "damaged_neighbors": float(np.exp(group.log_count.mean())),
            "F_star": float(np.exp(mean)),
            "F_star_lo": float(np.exp(mean - 1.96 * standard_error)),
            "F_star_hi": float(np.exp(mean + 1.96 * standard_error)),
            "n": len(group),
            "log_count": float(group.log_count.mean()),
            "log_exposure": float(mean),
        })
    return pd.DataFrame(rows).sort_values("damaged_neighbors")


def neighborhood_exposure_summary(
    data: pd.DataFrame,
    radii_ft=DEFAULT_RADII_FT,
    n_bins: int = 16,
    lowess_fraction: float = .5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return empirical bins, smooth curves and matched outcome-gap summaries."""
    plot_data = data[data.outcome.isin(["destroyed", "no_damage"])].copy()
    binned_rows = []
    smooth_rows = []
    gap_rows = []
    for fire in FIRES:
        for radius in radii_ft:
            count_col = f"damaged_neighbors_{radius}ft"
            panel = plot_data[plot_data.fire.eq(fire)]
            for outcome in ["no_damage", "destroyed"]:
                subset = panel[panel.outcome.eq(outcome)]
                bins = _binned_exposure(subset, count_col, n_bins=n_bins)
                bins["fire"] = fire
                bins["radius_ft"] = radius
                bins["outcome"] = outcome
                binned_rows.append(bins)
                if not bins.empty:
                    smooth = lowess(
                        bins.log_exposure, bins.log_count,
                        frac=lowess_fraction, return_sorted=True,
                    )
                    smooth_rows.extend({
                        "fire": fire,
                        "radius_ft": radius,
                        "outcome": outcome,
                        "damaged_neighbors": float(np.exp(x_value)),
                        "F_star": float(np.exp(y_value)),
                    } for x_value, y_value in smooth)

            # Match outcomes approximately within deciles of neighbor count.
            matched = panel[
                panel[count_col].gt(0) & panel.F_destroyed_wmean.gt(0)
            ].copy()
            matched["log_count_plus_one"] = np.log1p(matched[count_col])
            matched["log10_exposure"] = np.log10(matched.F_destroyed_wmean)
            strata = pd.qcut(
                matched.log_count_plus_one, q=10, duplicates="drop"
            )
            differences = []
            for _, indices in matched.groupby(strata, observed=True).groups.items():
                destroyed = matched.loc[indices]
                destroyed = destroyed.loc[
                    destroyed.outcome.eq("destroyed"), "log10_exposure"
                ]
                survived = matched.loc[indices]
                survived = survived.loc[
                    survived.outcome.eq("no_damage"), "log10_exposure"
                ]
                if len(destroyed) >= 20 and len(survived) >= 20:
                    differences.append(destroyed.median() - survived.median())
            gap_rows.append({
                "fire": fire,
                "radius_ft": radius,
                "matched_strata": len(differences),
                "destroyed_to_surviving_F_star_ratio": (
                    float(10 ** np.mean(differences)) if differences else np.nan
                ),
            })
    binned = pd.concat(binned_rows, ignore_index=True)
    smooth = pd.DataFrame(smooth_rows)
    gaps = pd.DataFrame(gap_rows)
    return binned, smooth, gaps

