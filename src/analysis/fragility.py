"""Primary exposure–response and distance-comparison analysis for Figure 1."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit
from scipy.spatial import cKDTree
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.proportion import proportion_confint


FIRES = ("EATON", "PALISADES")
M_TO_FT = 3.28084


def logistic_5pl(x, p_min, p_max, k, hill, asymmetry):
    """Five-parameter logistic response evaluated at positive exposure x."""
    x = np.maximum(np.asarray(x, float), 1e-300)
    with np.errstate(over="ignore", invalid="ignore"):
        return p_min + (p_max - p_min) / (
            1 + (k / x) ** hill
        ) ** asymmetry


def fit_5pl(x, y) -> dict[str, object]:
    """Fit a 5PL and numerically invert its asymptotic midpoint, F50."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]
    base = y[x < np.quantile(x, .1)].mean()
    top = y[x > np.quantile(x, .9)].mean()
    popt, _ = curve_fit(
        logistic_5pl,
        x,
        y,
        p0=[np.clip(base, 1e-3, .4), np.clip(top, .1, .99),
            np.median(x), 1., 1.],
        bounds=([0, 0, 1e-12, .1, .1],
                [.5, 1., x.max() * 5, 20., 10.]),
        maxfev=50_000,
    )
    target = (popt[0] + popt[1]) / 2
    f50 = brentq(
        lambda value: logistic_5pl(value, *popt) - target,
        x.min() / 10,
        x.max() * 10,
    )
    return {
        "p_min": float(popt[0]),
        "p_max": float(popt[1]),
        "k": float(popt[2]),
        "hill": float(popt[3]),
        "asymmetry": float(popt[4]),
        "f50": float(f50),
        "parameters": popt,
    }


def fit_curves_by_fire(data: pd.DataFrame, outcome_col: str,
                       n_boot: int = 200, seed: int = 0
                       ) -> tuple[dict[str, dict[str, object]], pd.DataFrame,
                                  pd.DataFrame]:
    """Fit fire-specific curves and spatial-block F50 confidence intervals."""
    exposed = data[data.exposed.eq(1)].copy()
    fits: dict[str, dict[str, object]] = {}
    for fire in FIRES:
        subset = exposed[exposed.fire.eq(fire)]
        fits[fire] = fit_5pl(subset.F_destroyed_wmean, subset[outcome_col])

    rng = np.random.default_rng(seed)
    groups = {
        fire: [group for _, group in exposed[exposed.fire.eq(fire)].groupby(
            "grid_id", observed=True)]
        for fire in FIRES
    }
    draw_rows: list[dict[str, object]] = []
    for draw in range(n_boot):
        estimates: dict[str, float] = {}
        try:
            for fire in FIRES:
                fire_groups = groups[fire]
                selected = rng.integers(0, len(fire_groups), len(fire_groups))
                sample = pd.concat(
                    [fire_groups[index] for index in selected], ignore_index=True
                )
                estimates[fire] = fit_5pl(
                    sample.F_destroyed_wmean, sample[outcome_col]
                )["f50"]
            draw_rows.append({
                "draw": draw,
                "outcome": outcome_col,
                "EATON_f50": estimates["EATON"],
                "PALISADES_f50": estimates["PALISADES"],
                "ratio_PAL_to_EAT": estimates["PALISADES"] / estimates["EATON"],
            })
        except (RuntimeError, ValueError):
            continue
    draws = pd.DataFrame(draw_rows)
    if len(draws) < max(20, int(.8 * n_boot)):
        raise RuntimeError(
            f"Only {len(draws)} of {n_boot} spatial bootstrap draws converged"
        )

    parameter_rows: list[dict[str, object]] = []
    for fire in FIRES:
        fit = fits[fire]
        lo, hi = np.percentile(draws[f"{fire}_f50"], [2.5, 97.5])
        fit["f50_lo"] = float(lo)
        fit["f50_hi"] = float(hi)
        subset = exposed[exposed.fire.eq(fire)]
        parameter_rows.append({
            "fire": fire,
            "outcome": outcome_col,
            "n": len(subset),
            **{name: fit[name] for name in
               ["p_min", "p_max", "k", "hill", "asymmetry", "f50",
                "f50_lo", "f50_hi"]},
        })
    parameters = pd.DataFrame(parameter_rows)
    return fits, parameters, draws


def binned_proportions(x, y, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-bin a binary outcome and calculate Wilson intervals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    index = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    rows = []
    for bin_id in range(len(edges) - 1):
        mask = index == bin_id
        if mask.sum() < 20:
            continue
        successes, total = int(y[mask].sum()), int(mask.sum())
        lo, hi = proportion_confint(successes, total, method="wilson")
        rows.append({
            "x": float(np.exp(np.log(x[mask]).mean())),
            "probability": successes / total,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n": total,
        })
    return pd.DataFrame(rows)


def partial_damage_summary(destroyed_fits: dict, any_damage_fits: dict,
                           x_min: float = 1e-6,
                           x_max: float = 2.0) -> pd.DataFrame:
    """Locate the maximum fitted partial-damage probability for each fire."""
    grid = np.geomspace(x_min, x_max, 20_000)
    rows = []
    for fire in FIRES:
        partial = (
            logistic_5pl(grid, *any_damage_fits[fire]["parameters"])
            - logistic_5pl(grid, *destroyed_fits[fire]["parameters"])
        )
        position = int(np.nanargmax(partial))
        rows.append({
            "fire": fire,
            "peak_partial_probability": float(partial[position]),
            "peak_F_star": float(grid[position]),
        })
    return pd.DataFrame(rows)


def prepare_distance_comparison(data_dir: Path,
                                exposed: pd.DataFrame) -> pd.DataFrame:
    """Attach CCD and visible-surface SSD to the exposed analysis population."""
    data_dir = Path(data_dir)
    distance = pd.read_parquet(
        data_dir / "radex.parquet",
        columns=["BLD_ID", "dist_destroyed_min"],
    )
    distance["BLD_ID"] = distance.BLD_ID.astype(str)
    result = exposed.copy()
    result["BLD_ID"] = result.BLD_ID.astype(str)
    result = result.merge(distance, on="BLD_ID", how="left")

    ccd_rows = []
    for fire in FIRES:
        buildings = gpd.read_parquet(
            data_dir / "nx" / f"{fire}_buildings.parquet",
            columns=["BLD_ID", "geometry"],
        )
        buildings["BLD_ID"] = buildings.BLD_ID.astype(str)
        centroids = buildings.geometry.centroid
        buildings["x"] = centroids.x.to_numpy()
        buildings["y"] = centroids.y.to_numpy()
        outcomes = result[result.fire.eq(fire)][["BLD_ID", "is_destroyed"]]
        buildings = (buildings.merge(outcomes, on="BLD_ID", how="inner")
                     .drop_duplicates("BLD_ID"))
        destroyed_xy = buildings.loc[
            buildings.is_destroyed.eq(1), ["x", "y"]
        ].to_numpy()
        tree = cKDTree(destroyed_xy)
        xy = buildings[["x", "y"]].to_numpy()
        nearest, _ = tree.query(xy, k=1)
        nearest_two, _ = tree.query(xy, k=2)
        nearest = np.where(
            buildings.is_destroyed.to_numpy() == 1,
            nearest_two[:, 1],
            nearest,
        )
        ccd_rows.append(pd.DataFrame({
            "BLD_ID": buildings.BLD_ID,
            "fire": fire,
            "ccd_ft": nearest * M_TO_FT,
        }))
    result = result.merge(
        pd.concat(ccd_rows, ignore_index=True), on=["BLD_ID", "fire"], how="left"
    )
    result["ssd_ft"] = result.dist_destroyed_min * M_TO_FT
    return result


def distance_outcome_summary(data: pd.DataFrame, distance_col: str,
                             n_bins: int = 18,
                             gap_bins: int = 12,
                             lowess_fraction: float = .42
                             ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Return binned exposure, LOWESS curves and the distance-matched gap."""
    subset = data.dropna(
        subset=[distance_col, "F_destroyed_wmean", "is_destroyed"]
    ).copy()
    subset = subset[
        subset[distance_col].gt(0) & subset.F_destroyed_wmean.gt(0)
    ].copy()
    subset["log_distance"] = np.log(subset[distance_col])
    subset["log_exposure"] = np.log(subset.F_destroyed_wmean)

    quantiles = pd.qcut(subset.log_distance, n_bins, duplicates="drop")
    rows = []
    for (destroyed, _), group in subset.groupby(
        ["is_destroyed", quantiles], observed=True
    ):
        if len(group) < 20:
            continue
        mean = group.log_exposure.mean()
        se = group.log_exposure.std(ddof=1) / np.sqrt(len(group))
        rows.append({
            "is_destroyed": int(destroyed),
            "distance_ft": float(np.exp(group.log_distance.mean())),
            "F_star": float(np.exp(mean)),
            "F_star_lo": float(np.exp(mean - 1.96 * se)),
            "F_star_hi": float(np.exp(mean + 1.96 * se)),
            "n": len(group),
            "log_distance": float(group.log_distance.mean()),
            "log_exposure": float(mean),
        })
    binned = pd.DataFrame(rows).sort_values(["is_destroyed", "distance_ft"])

    smooth_rows = []
    for destroyed in (0, 1):
        group = binned[binned.is_destroyed.eq(destroyed)]
        smooth = lowess(
            group.log_exposure, group.log_distance,
            frac=lowess_fraction, return_sorted=True,
        )
        smooth_rows.extend({
            "is_destroyed": destroyed,
            "distance_ft": float(np.exp(x_value)),
            "F_star": float(np.exp(y_value)),
        } for x_value, y_value in smooth)
    smooth = pd.DataFrame(smooth_rows)

    gap_groups = pd.qcut(np.log(subset[distance_col]), gap_bins, duplicates="drop")
    log10_exposure = np.log10(subset.F_destroyed_wmean)
    gaps = []
    for _, indices in subset.groupby(gap_groups, observed=True).groups.items():
        destroyed = log10_exposure.loc[indices][
            subset.loc[indices, "is_destroyed"].eq(1)
        ]
        survived = log10_exposure.loc[indices][
            subset.loc[indices, "is_destroyed"].eq(0)
        ]
        if len(destroyed) >= 20 and len(survived) >= 20:
            gaps.append(destroyed.median() - survived.median())
    exposure_ratio = float(10 ** np.mean(gaps))
    return binned, smooth, exposure_ratio


def fit_distance_calibration(data: pd.DataFrame) -> dict[str, float]:
    """Fit pooled log(F*) = a + b log(SSD) calibration."""
    subset = data[
        data.dist_destroyed_min.gt(0) & data.F_destroyed_wmean.gt(0)
    ]
    log_distance = np.log(subset.dist_destroyed_min.to_numpy(float))
    log_exposure = np.log(subset.F_destroyed_wmean.to_numpy(float))
    slope, intercept = np.polyfit(log_distance, log_exposure, 1)
    residual = log_exposure - (intercept + slope * log_distance)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared": float(1 - residual.var() / log_exposure.var()),
        "residual_sd": float(residual.std(ddof=2)),
        "n": int(len(subset)),
    }


def exposure_to_equivalent_distance(exposure, calibration,
                                    feet: bool = True):
    """Invert the pooled exposure-distance calibration."""
    distance_m = np.exp(
        (np.log(np.asarray(exposure, float)) - calibration["intercept"])
        / calibration["slope"]
    )
    return distance_m * (M_TO_FT if feet else 1.)


def distance_equivalence_table(parameters: pd.DataFrame,
                               calibration: dict[str, float]) -> pd.DataFrame:
    """Translate each fitted F50 and its interval with one pooled calibration."""
    rows = []
    for row in parameters.itertuples(index=False):
        point = exposure_to_equivalent_distance(row.f50, calibration)
        # Exposure and distance are inversely related, so interval bounds swap.
        distance_lo = exposure_to_equivalent_distance(row.f50_hi, calibration)
        distance_hi = exposure_to_equivalent_distance(row.f50_lo, calibration)
        rows.append({
            "fire": row.fire,
            "outcome": row.outcome,
            "f50": row.f50,
            "f50_lo": row.f50_lo,
            "f50_hi": row.f50_hi,
            "equivalent_ssd_ft": float(point),
            "equivalent_ssd_lo_ft": float(distance_lo),
            "equivalent_ssd_hi_ft": float(distance_hi),
            "equivalent_ssd_m": float(point / M_TO_FT),
        })
    return pd.DataFrame(rows)

