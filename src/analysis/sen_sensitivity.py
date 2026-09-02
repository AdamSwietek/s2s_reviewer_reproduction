"""Sensitivity analyses for fire-specific and regional SEN results."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.fragility import FIRES, fit_5pl
from src.analysis.regional_sen import build_regional_sen
from src.analysis.sen import coupling_at_probability, component_state


def fire_specific_probability_thresholds(analysis: pd.DataFrame,
                                         pooled_fit: dict,
                                         probability: float = .50
                                         ) -> pd.DataFrame:
    """Return pooled and fire-specific absolute-probability cutoffs."""
    rows = [{
        "calibration": "Pooled",
        "fire": "POOLED",
        "probability_equivalent": probability,
        "F_ij_threshold": coupling_at_probability(pooled_fit, probability),
        "fragility_midpoint_F50": pooled_fit["f50"],
        "n": int(analysis.exposed.eq(1).sum()),
    }]
    for fire in FIRES:
        subset = analysis[analysis.exposed.eq(1) & analysis.fire.eq(fire)]
        fit = fit_5pl(subset.F_destroyed_wmean, subset.is_destroyed)
        rows.append({
            "calibration": fire.title(),
            "fire": fire,
            "probability_equivalent": probability,
            "F_ij_threshold": coupling_at_probability(fit, probability),
            "fragility_midpoint_F50": fit["f50"],
            "n": len(subset),
        })
    return pd.DataFrame(rows)


def _destruction_profile(nodes: pd.DataFrame) -> dict:
    assessed = nodes[nodes.assessed].copy()
    isolated = assessed[assessed.component_size.eq(1)]
    connected = assessed[assessed.component_size.gt(1)]
    large = assessed[assessed.component_size.ge(100)]

    def share(frame):
        return float(frame.is_destroyed.mean()) if len(frame) else np.nan

    return {
        "assessed_isolated": len(isolated),
        "destroyed_share_isolated": share(isolated),
        "assessed_connected": len(connected),
        "destroyed_share_connected": share(connected),
        "assessed_in_SENs_ge_100": len(large),
        "destroyed_share_SENs_ge_100": share(large),
        "large_minus_isolated_pp": 100 * (share(large) - share(isolated)),
    }


def fire_threshold_sweep(networks: dict, analysis: pd.DataFrame,
                         pooled_fit: dict, probabilities) -> pd.DataFrame:
    """Rebuild each fire network across response-equivalent thresholds."""
    rows = []
    for probability in probabilities:
        threshold = coupling_at_probability(pooled_fit, float(probability))
        for fire in FIRES:
            state = component_state(networks[fire], threshold, analysis)
            rows.append({
                "fire": fire,
                "probability_equivalent": float(probability),
                **state["summary"],
                **_destruction_profile(state["nodes"]),
            })
    return pd.DataFrame(rows)


def fire_calibration_comparison(networks: dict, analysis: pd.DataFrame,
                                threshold_table: pd.DataFrame) -> pd.DataFrame:
    """Compare the common pooled cutoff with each fire's own cutoff."""
    pooled_threshold = float(
        threshold_table.loc[threshold_table.fire.eq("POOLED"),
                            "F_ij_threshold"].iloc[0]
    )
    rows = []
    for fire in FIRES:
        own_threshold = float(
            threshold_table.loc[threshold_table.fire.eq(fire),
                                "F_ij_threshold"].iloc[0]
        )
        for calibration, threshold in [
            ("Common pooled", pooled_threshold),
            ("Fire-specific", own_threshold),
        ]:
            state = component_state(networks[fire], threshold, analysis)
            rows.append({
                "fire": fire, "calibration": calibration,
                "F_ij_threshold": threshold,
                **state["summary"],
                **_destruction_profile(state["nodes"]),
            })
    return pd.DataFrame(rows)


def regional_threshold_sweep(project_root, pooled_fit: dict,
                             probabilities) -> pd.DataFrame:
    """Rebuild the production regional corridor across coupling thresholds."""
    rows = []
    for probability in probabilities:
        threshold = coupling_at_probability(pooled_fit, float(probability))
        result = build_regional_sen(project_root, threshold)
        row = result["summary"].iloc[0].to_dict()
        row["probability_equivalent"] = float(probability)
        rows.append(row)
    return pd.DataFrame(rows)


def regional_domain_sensitivity(project_root, threshold: float,
                                buffers=(500, 750, 1000)) -> pd.DataFrame:
    """Induce regional networks within alternative distances from the spine."""
    rows = []
    for distance in buffers:
        result = build_regional_sen(
            project_root, threshold, max_spine_distance_m=float(distance)
        )
        row = result["summary"].iloc[0].to_dict()
        row["domain"] = f"Within {int(distance):,} m of Interface spine"
        rows.append(row)
    production = build_regional_sen(project_root, threshold)
    row = production["summary"].iloc[0].to_dict()
    row["domain"] = "Production corridor"
    rows.append(row)
    return pd.DataFrame(rows)
