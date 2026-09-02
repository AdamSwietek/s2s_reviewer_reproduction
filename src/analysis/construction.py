"""Construction-attribute models and cross-fire standardization tables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import brentq, curve_fit

from src.analysis.fragility import FIRES, logistic_5pl


ROOF_TERM = 'C(roof_class, Treatment("Asphalt"))'
TILE_TERM = f"{ROOF_TERM}[T.Tile]"


@dataclass
class ConstructionSamples:
    exposed: pd.DataFrame
    undefended: pd.DataFrame
    known_roof: pd.DataFrame
    complete: pd.DataFrame
    ndvi_iqr: float
    year_reference: float


def attribute_unknownness_by_outcome(
    analysis: pd.DataFrame,
    attributes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Audit post-fire attribute recording by destruction outcome.

    The audit uses every assessed structure because it evaluates DINS data
    availability rather than an exposure-response estimand. ``Survived``
    includes undamaged and partially damaged structures.
    """
    if attributes is None:
        attributes = {
            "Eaves": "eaves",
            "Vent screening": "vent_screen",
            "Window panes": "window_pane",
            "Exterior siding": "exterior_siding",
        }
    rows = []
    for label, column in attributes.items():
        if column not in analysis:
            raise KeyError(f"Miseng construction attribute: {column}")
        values = analysis[column]
        unknown = values.isna() | values.astype(str).str.strip().str.lower().isin(
            {"", "unknown", "nan", "none"}
        )
        result = {"attribute": label, "source_column": column}
        for destroyed, group_label in [(0, "survived"), (1, "destroyed")]:
            selected = analysis.is_destroyed.eq(destroyed)
            n = int(selected.sum())
            unknown_n = int(unknown[selected].sum())
            result[f"{group_label}_n"] = n
            result[f"{group_label}_unknown_n"] = unknown_n
            result[f"{group_label}_unknown_percent"] = 100 * unknown_n / n
        result["difference_percentage_points"] = (
            result["destroyed_unknown_percent"]
            - result["survived_unknown_percent"]
        )
        rows.append(result)
    return pd.DataFrame(rows)


def prepare_construction_samples(analysis: pd.DataFrame) -> ConstructionSamples:
    """Construct the prespecified exposed, undefended analysis populations."""
    exposed = analysis[analysis.exposed.eq(1)].copy()
    exposed["log_F_star"] = np.log(exposed.F_destroyed_wmean)
    for column in ["roof_class", "fire"]:
        exposed[column] = exposed[column].astype(str)
    exposed["any_damage"] = exposed.any_damage.astype(int)
    exposed["is_destroyed"] = exposed.is_destroyed.astype(int)
    ndvi_iqr = float(exposed.ndvi_mean.quantile(.75) - exposed.ndvi_mean.quantile(.25))
    if not np.isfinite(ndvi_iqr) or ndvi_iqr <= 0:
        raise AssertionError("NDVI interquartile range is not positive")
    exposed["ndvi_iqr"] = exposed.ndvi_mean / ndvi_iqr
    year_reference = float(exposed.year_built.median())
    exposed["year_decade"] = (exposed.year_built - year_reference) / 10
    exposed["age_cohort"] = pd.cut(
        exposed.year_built,
        bins=[-np.inf, 1992, 2009, np.inf],
        right=False,
        labels=["Pre-1992", "1992-2008", "Post-2008"],
        ordered=True,
    )
    undefended = exposed[~exposed.defended].copy()
    known_roof = undefended[undefended.roof_class.ne("Unknown")].copy()
    complete = known_roof.dropna(subset=["ndvi_iqr", "year_decade"]).copy()
    return ConstructionSamples(
        exposed=exposed,
        undefended=undefended,
        known_roof=known_roof,
        complete=complete,
        ndvi_iqr=ndvi_iqr,
        year_reference=year_reference,
    )


def fit_clustered_logit(data: pd.DataFrame, formula: str):
    """Fit logistic regression with approximately 250-m cluster inference."""
    return smf.logit(formula, data=data).fit(
        disp=0,
        maxiter=500,
        cov_type="cluster",
        cov_kwds={"groups": data.grid_id},
    )


def tolerance_ratio(model, term: str,
                    exposure_term: str = "log_F_star") -> dict[str, float]:
    """Convert a logit coefficient to exp(-gamma/beta), with delta-method CI."""
    gamma = float(model.params[term])
    beta = float(model.params[exposure_term])
    covariance = model.cov_params().loc[
        [term, exposure_term], [term, exposure_term]
    ].to_numpy()
    gradient = np.array([-1 / beta, gamma / beta ** 2])
    standard_error_log_ratio = float(np.sqrt(gradient @ covariance @ gradient))
    log_ratio = -gamma / beta
    return {
        "coefficient": gamma,
        "coefficient_se": float(model.bse[term]),
        "tolerance_ratio": float(np.exp(log_ratio)),
        "ci_lo": float(np.exp(log_ratio - 1.96 * standard_error_log_ratio)),
        "ci_hi": float(np.exp(log_ratio + 1.96 * standard_error_log_ratio)),
        "p_value": float(model.pvalues[term]),
    }


def _selected_coefficients(model, model_name: str,
                           terms: dict[str, str]) -> pd.DataFrame:
    rows = []
    for label, term in terms.items():
        rows.append({
            "model": model_name,
            "attribute": label,
            "term": term,
            "coefficient": float(model.params[term]),
            "standard_error": float(model.bse[term]),
            "p_value": float(model.pvalues[term]),
            "n": int(model.nobs),
        })
    return pd.DataFrame(rows)


def fit_overall_attributes(samples: ConstructionSamples):
    """Fit the overall-destruction models used in Table 1, panel A."""
    base_formula = (
        f"is_destroyed ~ log_F_star + {ROOF_TERM} + ndvi_iqr + C(fire)"
    )
    roof_vegetation = fit_clustered_logit(samples.known_roof, base_formula)
    vintage = fit_clustered_logit(
        samples.complete, base_formula + " + year_decade"
    )
    specifications = [
        ("Tile roof", "Tile versus asphalt", roof_vegetation, TILE_TERM),
        ("Near-structure vegetation", "Per IQR increase in NDVI",
         roof_vegetation, "ndvi_iqr"),
        ("Building vintage", "Per decade newer", vintage, "year_decade"),
    ]
    rows = []
    for attribute, contrast, model, term in specifications:
        rows.append({
            "attribute": attribute,
            "contrast": contrast,
            "analysis_n": int(model.nobs),
            **tolerance_ratio(model, term),
        })
    table = pd.DataFrame(rows)
    raw = pd.concat([
        _selected_coefficients(
            roof_vegetation, "roof_and_vegetation",
            {"Tile roof": TILE_TERM,
             "Near-structure vegetation": "ndvi_iqr",
             "Log realized coupling": "log_F_star"},
        ),
        _selected_coefficients(
            vintage, "roof_vegetation_and_vintage",
            {"Building vintage": "year_decade",
             "Log realized coupling": "log_F_star"},
        ),
    ], ignore_index=True)
    return table, raw, {"roof_vegetation": roof_vegetation, "vintage": vintage}


def fit_damage_stages(samples: ConstructionSamples):
    """Fit onset and escalation models and their stacked stage interactions."""
    complete = samples.complete
    formula = (
        f"{{outcome}} ~ log_F_star + {ROOF_TERM} + ndvi_iqr "
        "+ year_decade + C(fire)"
    )
    onset = fit_clustered_logit(complete, formula.format(outcome="any_damage"))
    damaged = complete[complete.any_damage.eq(1)].copy()
    escalation = fit_clustered_logit(
        damaged, formula.format(outcome="is_destroyed")
    )
    stacked = pd.concat([
        complete.assign(stage="onset", stage_outcome=complete.any_damage),
        damaged.assign(stage="escalation", stage_outcome=damaged.is_destroyed),
    ], ignore_index=True)
    stage_term = 'C(stage, Treatment("escalation"))'
    stacked_model = fit_clustered_logit(
        stacked,
        f"stage_outcome ~ {stage_term} * (log_F_star + {ROOF_TERM} + "
        "ndvi_iqr + year_decade) + "
        f"{stage_term}:C(fire) + C(fire)",
    )
    terms = [
        ("Tile roof", "Tile versus asphalt", TILE_TERM, "T.Tile"),
        ("Near-structure vegetation", "Per IQR increase in NDVI",
         "ndvi_iqr", "ndvi_iqr"),
        ("Building vintage", "Per decade newer", "year_decade", "year_decade"),
    ]
    rows = []
    for attribute, contrast, term, interaction_key in terms:
        onset_result = tolerance_ratio(onset, term)
        escalation_result = tolerance_ratio(escalation, term)
        interaction_terms = [
            name for name in stacked_model.params.index
            if "T.onset" in name and interaction_key in name
        ]
        if len(interaction_terms) != 1:
            raise RuntimeError(
                f"Could not identify unique stage interaction for {attribute}: "
                f"{interaction_terms}"
            )
        rows.append({
            "attribute": attribute,
            "contrast": contrast,
            "onset_n": int(onset.nobs),
            "onset_tolerance_ratio": onset_result["tolerance_ratio"],
            "onset_ci_lo": onset_result["ci_lo"],
            "onset_ci_hi": onset_result["ci_hi"],
            "onset_p_value": onset_result["p_value"],
            "escalation_n": int(escalation.nobs),
            "escalation_tolerance_ratio": escalation_result["tolerance_ratio"],
            "escalation_ci_lo": escalation_result["ci_lo"],
            "escalation_ci_hi": escalation_result["ci_hi"],
            "escalation_p_value": escalation_result["p_value"],
            "stage_difference_p_value": float(
                stacked_model.pvalues[interaction_terms[0]]
            ),
        })
    table = pd.DataFrame(rows)
    raw = pd.concat([
        _selected_coefficients(
            onset, "damage_onset",
            {"Tile roof": TILE_TERM,
             "Near-structure vegetation": "ndvi_iqr",
             "Building vintage": "year_decade",
             "Log realized coupling": "log_F_star"},
        ),
        _selected_coefficients(
            escalation, "destruction_given_damage",
            {"Tile roof": TILE_TERM,
             "Near-structure vegetation": "ndvi_iqr",
             "Building vintage": "year_decade",
             "Log realized coupling": "log_F_star"},
        ),
    ], ignore_index=True)
    return table, raw, {
        "onset": onset,
        "escalation": escalation,
        "stacked": stacked_model,
    }


def _weighted_f50(x, y, weights=None) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    base = y[x < np.quantile(x, .1)].mean()
    top = y[x > np.quantile(x, .9)].mean()
    sigma = None if weights is None else 1 / np.sqrt(
        np.maximum(np.asarray(weights, float), 1e-9)
    )
    parameters, _ = curve_fit(
        logistic_5pl, x, y, sigma=sigma,
        p0=[np.clip(base, 1e-3, .4), np.clip(top, .1, .99),
            np.median(x), 1., 1.],
        bounds=([0, 0, 1e-12, .1, .1],
                [.5, 1., x.max() * 5, 20., 10.]),
        maxfev=50_000,
    )
    target = (parameters[0] + parameters[1]) / 2
    return float(brentq(
        lambda value: logistic_5pl(value, *parameters) - target,
        x.min() / 10, x.max() * 10,
    ))


def _f50_pair(data: pd.DataFrame) -> tuple[float, float, float]:
    values = {}
    for fire in FIRES:
        subset = data[data.fire.eq(fire)]
        values[fire] = _weighted_f50(
            subset.F_destroyed_wmean, subset.is_destroyed
        )
    return values["EATON"], values["PALISADES"], (
        values["PALISADES"] / values["EATON"]
    )


def _composition_gaps(data: pd.DataFrame, weight_clip: float = 10.) -> dict[str, float]:
    """Return Shapley-averaged DFL composition factors and residual gap."""
    subsets = {fire: data[data.fire.eq(fire)] for fire in FIRES}
    cells = {
        "materials": lambda frame: (
            frame.roof_class.astype(str) + "|" + frame.age_cohort.astype(str)
        ),
        "vegetation": lambda frame: frame.ndvi_tercile.astype(str),
        "full_stock": lambda frame: (
            frame.roof_class.astype(str) + "|" + frame.age_cohort.astype(str)
            + "|" + frame.ndvi_tercile.astype(str)
        ),
    }

    def swap_factor(cell_function):
        cell = {fire: cell_function(subsets[fire]) for fire in FIRES}
        shares = {
            fire: cell[fire].value_counts(normalize=True) for fire in FIRES
        }
        estimates = {}
        for fire_index, fire in enumerate(FIRES):
            other = FIRES[1 - fire_index]
            subset = subsets[fire]
            weights = (
                shares[other].reindex(cell[fire]) / shares[fire].reindex(cell[fire])
            ).to_numpy()
            weights = np.clip(
                np.nan_to_num(weights, nan=0., posinf=weight_clip,
                              neginf=0.),
                0., weight_clip,
            )
            estimates[(fire, "own")] = _weighted_f50(
                subset.F_destroyed_wmean, subset.is_destroyed
            )
            estimates[(fire, "swap")] = _weighted_f50(
                subset.F_destroyed_wmean, subset.is_destroyed, weights
            )
        factor = np.sqrt(
            estimates[("EATON", "swap")] / estimates[("EATON", "own")]
            * estimates[("PALISADES", "own")] / estimates[("PALISADES", "swap")]
        )
        raw = (
            estimates[("PALISADES", "own")] / estimates[("EATON", "own")]
        )
        return float(raw), float(factor)

    output = {}
    raw = np.nan
    for name, function in cells.items():
        raw, output[name] = swap_factor(function)
    output["raw"] = float(raw)
    output["residual"] = float(raw / output["full_stock"])
    return output


def cross_fire_standardization(samples: ConstructionSamples,
                               n_boot: int = 200,
                               seed: int = 20250729):
    """Recalculate the sample ladder and DFL cross-fire standardization."""
    ladder_samples = {
        "All exposed": samples.exposed,
        "Undefended": samples.undefended,
        "Undefended with known roof": samples.known_roof,
        "Complete roof, vegetation and vintage": samples.complete,
    }
    ladder_rows = []
    for name, data in ladder_samples.items():
        eaton, palisades, ratio = _f50_pair(data)
        ladder_rows.append({
            "sample": name,
            "n": len(data),
            "EATON_f50": eaton,
            "PALISADES_f50": palisades,
            "ratio_PAL_to_EAT": ratio,
        })
    ladder = pd.DataFrame(ladder_rows)

    complete = samples.complete.copy()
    complete["ndvi_tercile"] = pd.qcut(
        complete.ndvi_mean, 3, labels=False, duplicates="drop"
    ).astype(str)
    point = _composition_gaps(complete)

    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    for draw in range(n_boot):
        sample = pd.concat([
            complete[complete.fire.eq(fire)].sample(
                frac=1, replace=True,
                random_state=int(rng.integers(0, 2 ** 32 - 1)),
            )
            for fire in FIRES
        ], ignore_index=True)
        try:
            bootstrap_rows.append({"draw": draw, **_composition_gaps(sample)})
        except (RuntimeError, ValueError):
            continue
    draws = pd.DataFrame(bootstrap_rows)
    if len(draws) < max(20, int(.8 * n_boot)):
        raise RuntimeError(
            f"Only {len(draws)} of {n_boot} standardization draws converged"
        )
    rows = []
    for component in ["raw", "materials", "vegetation", "full_stock", "residual"]:
        lo, hi = np.percentile(draws[component], [2.5, 97.5])
        share = (
            np.nan if component in {"raw", "residual"}
            else 100 * np.log(point[component]) / np.log(point["raw"])
        )
        rows.append({
            "component": component,
            "factor": point[component],
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "share_of_log_gap_percent": float(share),
            "successful_draws": len(draws),
        })
    return ladder, pd.DataFrame(rows), draws


def format_p_value(value: float) -> str:
    if value < .001:
        return "<0.001"
    return f"{value:.3f}"


def publication_table_markdown(overall: pd.DataFrame,
                               stages: pd.DataFrame) -> str:
    """Return a manuscript-ready Markdown rendering of Table 1."""
    lines = [
        "# Table 1 | Construction attributes and exposure tolerance",
        "",
        "## a. Overall destruction",
        "",
        "| Attribute | Contrast | n | Tolerance ratio (95% CI) | P value |",
        "|---|---|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.attribute} | {row.contrast} | {row.analysis_n:,} | "
            f"{row.tolerance_ratio:.2f} ({row.ci_lo:.2f}–{row.ci_hi:.2f}) | "
            f"{format_p_value(row.p_value)} |"
        )
    lines.extend([
        "",
        "## b. Damage onset and escalation",
        "",
        "| Attribute | Onset tolerance ratio (95% CI) | P value | "
        "Escalation tolerance ratio (95% CI) | P value | Stage-difference P |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in stages.itertuples(index=False):
        lines.append(
            f"| {row.attribute} | {row.onset_tolerance_ratio:.2f} "
            f"({row.onset_ci_lo:.2f}–{row.onset_ci_hi:.2f}) | "
            f"{format_p_value(row.onset_p_value)} | "
            f"{row.escalation_tolerance_ratio:.2f} "
            f"({row.escalation_ci_lo:.2f}–{row.escalation_ci_hi:.2f}) | "
            f"{format_p_value(row.escalation_p_value)} | "
            f"{format_p_value(row.stage_difference_p_value)} |"
        )
    lines.extend([
        "",
        "Tolerance ratios are exp(-gamma/beta), where gamma is the attribute "
        "coefficient and beta is the log-exposure coefficient. Values above "
        "one indicate greater realized coupling at the same modeled outcome "
        "probability. Models adjust for fire and use standard errors clustered "
        "by approximately 250-m grid cell. Onset is any damage among exposed "
        "structures; escalation is destruction conditional on recorded damage.",
        "",
    ])
    return "\n".join(lines)
