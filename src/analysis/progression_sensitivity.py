"""Defense sensitivity to alternative Eaton fire-progression surfaces."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .defense import (
    build_focal_table,
    build_view_decomposition,
    covariate_balance,
    direct_effect_table,
    match_defended,
    mechanism_table,
    prepare_spillover_match,
    spillover_table,
)


METHODS = {
    "b": "Linear interpolation",
    "c": "Kriging update (primary)",
    "d": "Direct regression kriging",
}


def _scenario_inputs(project_root: Path, method: str):
    """Combine the alternate Eaton clock with the fixed Palisades clock."""
    if method not in METHODS:
        raise ValueError(f"Unknown progression method: {method}")
    data = Path(project_root) / "data"
    sensitivity = pd.read_parquet(
        data / "arrival" / "eaton_progression_sensitivity.parquet"
    )
    sensitivity["BLD_ID"] = sensitivity.BLD_ID.astype(str)
    buildings = pd.read_parquet(
        data / "buildings_enriched.parquet",
        columns=["BLD_ID", "fire", "T_arrival_hrs"],
    )
    buildings["BLD_ID"] = buildings.BLD_ID.astype(str)
    radex = pd.read_parquet(
        data / "radex.parquet",
        columns=["BLD_ID", "fire", "F_upfire_wmean",
                 "F_upfire_interp_wmean"],
    )
    radex["BLD_ID"] = radex.BLD_ID.astype(str)

    arrival_column = {
        "b": "T_arrival_interp_hrs",
        "c": "T_arrival_hrs",
        "d": "T_arrival_direct_hrs",
    }[method]
    eaton_arrival = sensitivity[["BLD_ID", arrival_column]].rename(
        columns={arrival_column: "arrival"}
    ).assign(fire="EATON")
    palisades_arrival = (
        buildings[buildings.fire.eq("PALISADES")]
        [["BLD_ID", "fire", "T_arrival_hrs"]]
        .rename(columns={"T_arrival_hrs": "arrival"})
    )
    arrival = pd.concat([eaton_arrival, palisades_arrival], ignore_index=True)
    arrival = arrival[arrival.BLD_ID.ne("None")].drop_duplicates(
        ["BLD_ID", "fire"]
    )

    standard = radex[["BLD_ID", "fire", "F_upfire_wmean"]].copy()
    if method == "b":
        eaton_upfire = (
            radex[radex.fire.eq("EATON")]
            [["BLD_ID", "fire", "F_upfire_interp_wmean"]]
            .rename(columns={"F_upfire_interp_wmean": "F_upfire_wmean"})
        )
    elif method == "d":
        eaton_upfire = sensitivity[
            ["BLD_ID", "F_upfire_direct_wmean"]
        ].rename(columns={"F_upfire_direct_wmean": "F_upfire_wmean"})
        eaton_upfire["fire"] = "EATON"
    else:
        eaton_upfire = standard[standard.fire.eq("EATON")]
    upfire = pd.concat([
        eaton_upfire,
        standard[standard.fire.eq("PALISADES")],
    ], ignore_index=True)
    upfire = upfire[upfire.BLD_ID.ne("None")].drop_duplicates(
        ["BLD_ID", "fire"]
    )
    return arrival, upfire


def run_progression_sensitivity(project_root: Path, n_boot: int = 2000,
                                seed: int = 20250809) -> dict[str, pd.DataFrame]:
    """Repeat the primary defense analyses for progression methods b, c and d.

    Only the Eaton arrival surface, arrival ordering and corresponding
    area-weighted up-fire exposure are varied. Palisades and every other
    design, matching and model choice remain fixed.
    """
    outputs = {name: [] for name in [
        "direct", "outcomes", "spillover", "mechanism", "audit", "balance"
    ]}
    for offset, (code, label) in enumerate(METHODS.items()):
        arrival, upfire = _scenario_inputs(project_root, code)
        focal = build_focal_table(
            project_root, arrival_override=arrival, upfire_override=upfire
        )
        matched, _ = match_defended(focal)
        direct, outcomes = direct_effect_table(
            matched, n_boot=n_boot, seed=seed + 100 * offset
        )
        eligible, iv_matched, _ = prepare_spillover_match(focal, threshold=.75)
        spillover, _, _ = spillover_table(
            iv_matched, n_boot=n_boot, seed=seed + 100 * offset
        )
        view = build_view_decomposition(
            project_root, focal, arrival_override=arrival
        )
        mechanism = mechanism_table(view)
        balance = covariate_balance(eligible, iv_matched)

        for frame in [direct, outcomes, spillover, mechanism, balance]:
            frame.insert(0, "progression_method", label)
            frame.insert(0, "method", code)
        outputs["direct"].append(direct)
        outputs["outcomes"].append(outcomes)
        outputs["spillover"].append(spillover)
        outputs["mechanism"].append(mechanism)
        outputs["balance"].append(balance)
        outputs["audit"].append(pd.DataFrame([{
            "method": code,
            "progression_method": label,
            "eligible_focal_n": len(focal),
            "eligible_defended_n": int(focal.defended.sum()),
            "direct_matched_pairs": int(matched.pair_id.nunique()),
            "spillover_eligible_n": len(eligible),
            "spillover_matched_pairs": int(iv_matched.pair_id.nunique()),
            "max_abs_postmatch_smd": float(balance.smd_after.abs().max()),
        }]))
    return {key: pd.concat(value, ignore_index=True)
            for key, value in outputs.items()}


def progression_range_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compact ranges across the three progression reconstructions."""
    direct = results["direct"].query("sample == 'Pooled'")
    spill = results["spillover"].query("sample == 'Pooled'")
    mechanism = results["mechanism"]
    measures = [
        ("Matched focal survival difference", direct,
         "survival_difference_pp", 1.0),
        ("Direction-adjusted neighbor contrast", spill,
         "directional_contrast", 100.0),
        ("Local IV ratio", spill, "local_iv", 100.0),
    ]
    rows = []
    for label, frame, column, multiplier in measures:
        values = multiplier * frame[column].to_numpy(float)
        rows.append({
            "estimand": label,
            "minimum_pp": float(np.nanmin(values)),
            "maximum_pp": float(np.nanmax(values)),
            "range_width_pp": float(np.nanmax(values) - np.nanmin(values)),
        })
    for model, term, label in [
        ("Potential coupling only", "Defended component of up-fire view",
         "Earlier-arrival defended-view AME"),
        ("Potential coupling only", "Defended down-fire view (placebo)",
         "Later-arrival defended-view control AME"),
        ("+ destroyed share", "Defended component of up-fire view",
         "Earlier-arrival AME after destroyed-share adjustment"),
    ]:
        frame = mechanism[
            mechanism.model.eq(model) & mechanism.term.eq(term)
        ]
        values = frame.ame_pp_per_sd.to_numpy(float)
        rows.append({
            "estimand": label,
            "minimum_pp": float(np.nanmin(values)),
            "maximum_pp": float(np.nanmax(values)),
            "range_width_pp": float(np.nanmax(values) - np.nanmin(values)),
        })
    return pd.DataFrame(rows)
