"""Exposure-dependent construction-attribute sensitivity analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.special import expit

from src.analysis.construction import ROOF_TERM, fit_clustered_logit


INTERACTION_FORMULA = (
    f"is_destroyed ~ log_F_star * ({ROOF_TERM} + ndvi_iqr + year_decade) "
    "+ C(fire)"
)
ADDITIVE_FORMULA = (
    f"is_destroyed ~ log_F_star + {ROOF_TERM} + ndvi_iqr + year_decade "
    "+ C(fire)"
)


def fit_exposure_interactions(data: pd.DataFrame):
    """Fit clustered interaction model and an MLE likelihood-ratio test.

    Cluster-robust covariance is used for uncertainty in standardized margins.
    The likelihood-ratio test compares ordinary maximum-likelihood fits because
    robust pseudo-likelihoods are not directly comparable.
    """
    clustered = fit_clustered_logit(data, INTERACTION_FORMULA)
    full_mle = smf.logit(INTERACTION_FORMULA, data=data).fit(
        disp=0, maxiter=500)
    additive_mle = smf.logit(ADDITIVE_FORMULA, data=data).fit(
        disp=0, maxiter=500)
    statistic = 2 * (full_mle.llf - additive_mle.llf)
    degrees_freedom = int(full_mle.df_model - additive_mle.df_model)
    test = {
        "analysis_n": int(clustered.nobs),
        "lr_chi2": float(statistic),
        "degrees_freedom": degrees_freedom,
        "p_value": float(stats.chi2.sf(statistic, degrees_freedom)),
    }
    return clustered, test


def _prediction_difference(model, frame: pd.DataFrame, grid: np.ndarray,
                           vulnerable: dict, protective: dict,
                           parameter_draws: np.ndarray | None = None):
    """Standardized vulnerable-minus-protective probability difference."""
    from patsy import build_design_matrices

    design_info = model.model.data.design_info
    point = []
    draw_values = [] if parameter_draws is not None else None
    for exposure in grid:
        common = {"log_F_star": np.log(exposure)}
        vulnerable_frame = frame.assign(**common, **vulnerable)
        protective_frame = frame.assign(**common, **protective)
        xv = np.asarray(build_design_matrices(
            [design_info], vulnerable_frame, return_type="dataframe")[0])
        xp = np.asarray(build_design_matrices(
            [design_info], protective_frame, return_type="dataframe")[0])
        beta = np.asarray(model.params)
        pv = expit(xv @ beta)
        pp = expit(xp @ beta)
        point.append(100 * np.mean(pv - pp))
        if parameter_draws is not None:
            # Chunk coefficient simulations to keep peak memory modest on
            # reviewer laptops (the complete sample contains >21,000 rows).
            simulated = []
            for start in range(0, len(parameter_draws), 25):
                draws = parameter_draws[start:start + 25]
                vulnerable_mean = expit(xv @ draws.T).mean(axis=0)
                protective_mean = expit(xp @ draws.T).mean(axis=0)
                simulated.extend(100 * (vulnerable_mean - protective_mean))
            draw_values.append(np.asarray(simulated))
    return np.asarray(point), (
        None if draw_values is None else np.asarray(draw_values)
    )


def exposure_dependent_margins(data: pd.DataFrame, model, *,
                               n_grid: int = 60, n_draws: int = 400,
                               seed: int = 20250808):
    """Estimate standardized protective margins across realized coupling.

    Vegetation and vintage contrasts compare their observed 75th and 25th
    percentiles. Confidence bands use simulations from the cluster-robust
    coefficient covariance matrix and therefore describe model uncertainty,
    not uncertainty in the selected contrast values.
    """
    grid = np.geomspace(
        data.F_destroyed_wmean.quantile(.02),
        data.F_destroyed_wmean.quantile(.98),
        n_grid,
    )
    ndvi_low, ndvi_high = data.ndvi_iqr.quantile([.25, .75])
    year_old, year_new = data.year_decade.quantile([.25, .75])
    contrasts = {
        "Tile roofing": (
            {"roof_class": "Asphalt"}, {"roof_class": "Tile"},
            "Tile rather than asphalt",
        ),
        "Lower vegetation": (
            {"ndvi_iqr": float(ndvi_high)},
            {"ndvi_iqr": float(ndvi_low)},
            "25th rather than 75th percentile NDVI",
        ),
        "Newer construction": (
            {"year_decade": float(year_old)},
            {"year_decade": float(year_new)},
            "75th rather than 25th percentile construction year",
        ),
    }
    rng = np.random.default_rng(seed)
    parameter_draws = rng.multivariate_normal(
        np.asarray(model.params), np.asarray(model.cov_params()), size=n_draws,
        check_valid="ignore",
    )
    curve_rows, summary_rows = [], []
    for attribute, (vulnerable, protective, contrast) in contrasts.items():
        margin, draws = _prediction_difference(
            model, data, grid, vulnerable, protective, parameter_draws)
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=1)
        peak_index = int(np.nanargmax(margin))
        after_peak = np.flatnonzero(
            (np.arange(len(grid)) > peak_index) & (margin < 5)
        )
        below_five = int(after_peak[0]) if len(after_peak) else None
        for i, exposure in enumerate(grid):
            curve_rows.append({
                "attribute": attribute,
                "contrast": contrast,
                "F_star": float(exposure),
                "margin_pp": float(margin[i]),
                "ci_lo": float(lo[i]),
                "ci_hi": float(hi[i]),
                "exposure_percentile": float(
                    100 * (data.F_destroyed_wmean <= exposure).mean()
                ),
            })
        summary_rows.append({
            "attribute": attribute,
            "contrast": contrast,
            "analysis_n": int(model.nobs),
            "peak_margin_pp": float(margin[peak_index]),
            "peak_F_star": float(grid[peak_index]),
            "peak_exposure_percentile": float(
                100 * (data.F_destroyed_wmean <= grid[peak_index]).mean()
            ),
            "margin_at_p98_pp": float(margin[-1]),
            "F_star_below_5pp_after_peak": (
                np.nan if below_five is None else float(grid[below_five])
            ),
            "percentile_below_5pp_after_peak": (
                np.nan if below_five is None else float(
                    100 * (data.F_destroyed_wmean <= grid[below_five]).mean()
                )
            ),
        })
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)
