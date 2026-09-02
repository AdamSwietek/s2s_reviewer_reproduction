"""Sensitivity analyses for the directional defense-spillover design."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from src.analysis.defense import (
    FIRES, build_focal_table, covariate_balance, iv_estimate,
    match_defended, prepare_spillover_match, spillover_table,
)


def add_pair_spatial_context(project_root: Path, focal: pd.DataFrame,
                             matched: pd.DataFrame,
                             block_m: float = 250,
                             isolation_m: float = 150) -> pd.DataFrame:
    """Attach treated-focal spatial blocks and defense-isolation distances."""
    coordinates = pd.read_parquet(
        Path(project_root) / "data" / "radex.parquet",
        columns=["BLD_ID", "lon_wgs84", "lat_wgs84"],
    ).drop_duplicates("BLD_ID")
    coordinates["BLD_ID"] = coordinates.BLD_ID.astype(str)
    reference = focal.merge(coordinates, on="BLD_ID", how="left")
    transformer = Transformer.from_crs(4326, 26911, always_xy=True)
    x, y = transformer.transform(
        reference.lon_wgs84.to_numpy(), reference.lat_wgs84.to_numpy()
    )
    reference["x_utm"], reference["y_utm"] = x, y

    treated_context = []
    for fire in FIRES:
        fire_reference = reference[reference.fire.eq(fire)].copy()
        defended = fire_reference[fire_reference.defended.eq(1)].copy()
        tree = cKDTree(defended[["x_utm", "y_utm"]].to_numpy())
        treated = matched[
            matched.fire.eq(fire) & matched.defended.eq(1)
        ][["BLD_ID", "pair_id"]].merge(
            fire_reference[["BLD_ID", "x_utm", "y_utm"]], on="BLD_ID"
        )
        distances, _ = tree.query(
            treated[["x_utm", "y_utm"]].to_numpy(), k=2
        )
        treated["distance_other_defended_m"] = distances[:, 1]
        treated["spatial_block"] = (
            fire + ":"
            + np.floor(treated.x_utm / block_m).astype(int).astype(str)
            + ":"
            + np.floor(treated.y_utm / block_m).astype(int).astype(str)
        )
        treated["isolated_defense"] = (
            treated.distance_other_defended_m.gt(isolation_m)
        )
        treated_context.append(treated[[
            "pair_id", "distance_other_defended_m", "spatial_block",
            "isolated_defense",
        ]])
    context = pd.concat(treated_context, ignore_index=True)
    return matched.merge(context, on="pair_id", how="left", validate="many_to_one")


def _pair_differences(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair_id, pair in frame.groupby("pair_id"):
        if len(pair) != 2 or pair.defended.nunique() != 2:
            continue
        treated = pair[pair.defended.eq(1)].iloc[0]
        control = pair[pair.defended.eq(0)].iloc[0]
        rows.append({
            "pair_id": pair_id,
            "fire": treated.fire,
            "spatial_block": treated.get("spatial_block", str(pair_id)),
            "first_stage": treated.survived - control.survived,
            "downfire_contrast": treated.down100 - control.down100,
            "upfire_placebo": treated.up100 - control.up100,
        })
    return pd.DataFrame(rows)


def spatial_block_interval(frame: pd.DataFrame, n_boot: int = 2000,
                           seed: int = 0) -> dict[str, np.ndarray]:
    """Resample treated-focal 250-m blocks while retaining complete pairs."""
    pairs = _pair_differences(frame)
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in [
        "first_stage", "downfire_contrast", "upfire_placebo",
        "directional_contrast", "local_iv",
    ]}
    fire_blocks = {
        fire: [group for _, group in fire_frame.groupby("spatial_block")]
        for fire, fire_frame in pairs.groupby("fire")
    }
    for _ in range(n_boot):
        fire_estimates, weights = [], []
        for fire in FIRES:
            blocks = fire_blocks.get(fire, [])
            if not blocks:
                continue
            selected = rng.integers(0, len(blocks), len(blocks))
            sample = pd.concat([blocks[index] for index in selected])
            first = sample.first_stage.mean()
            down = sample.downfire_contrast.mean()
            placebo = sample.upfire_placebo.mean()
            fire_estimates.append([first, down, placebo])
            weights.append(len(sample))
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        first, down, placebo = np.average(
            np.asarray(fire_estimates), axis=0, weights=weights
        )
        directional = down - placebo
        values = [first, down, placebo, directional, directional / first]
        for key, value in zip(draws, values):
            draws[key].append(value)
    return {
        key: np.nanpercentile(values, [2.5, 97.5])
        for key, values in draws.items()
    }


def spatial_and_isolation_table(context: pd.DataFrame, n_boot: int = 2000,
                                seed: int = 0) -> pd.DataFrame:
    """Compare primary pair and block intervals and the isolation split."""
    rows = []
    primary, _, _ = spillover_table(context, n_boot=n_boot, seed=seed)
    pooled = primary[primary["sample"].eq("Pooled")].iloc[0]
    rows.append({
        "analysis": "Primary: matched-pair bootstrap",
        "n": len(context), "pairs": context.pair_id.nunique(),
        "spatial_blocks": context.spatial_block.nunique(),
        "defended_n": int(context.defended.sum()),
        "first_stage": pooled.first_stage,
        "directional_contrast": pooled.directional_contrast,
        "directional_lo": pooled.directional_contrast_lo,
        "directional_hi": pooled.directional_contrast_hi,
        "local_iv": pooled.local_iv,
        "local_iv_lo": pooled.local_iv_lo,
        "local_iv_hi": pooled.local_iv_hi,
    })
    samples = [
        ("Primary: 250-m block bootstrap", context),
        ("Isolated defense (>150 m)", context[context.isolated_defense]),
        ("Clustered defense (<=150 m)", context[~context.isolated_defense]),
    ]
    for offset, (label, sample) in enumerate(samples):
        estimate = iv_estimate(sample)
        interval = spatial_block_interval(
            sample, n_boot=n_boot, seed=seed + offset + 1
        )
        rows.append({
            "analysis": label, "n": len(sample),
            "pairs": sample.pair_id.nunique(),
            "spatial_blocks": sample.spatial_block.nunique(),
            "defended_n": int(sample.defended.sum()),
            "first_stage": estimate["first_stage"],
            "directional_contrast": estimate["directional_contrast"],
            "directional_lo": interval["directional_contrast"][0],
            "directional_hi": interval["directional_contrast"][1],
            "local_iv": estimate["local_iv"],
            "local_iv_lo": interval["local_iv"][0],
            "local_iv_hi": interval["local_iv"][1],
        })
    return pd.DataFrame(rows)


def threshold_sensitivity(focal: pd.DataFrame, thresholds=(.60, .70, .75, .80, .90),
                          n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Rematch and re-estimate the IV analysis at each upstream threshold."""
    rows = []
    for offset, threshold in enumerate(thresholds):
        eligible, matched, _ = prepare_spillover_match(
            focal, threshold=threshold
        )
        result, _, _ = spillover_table(
            matched, n_boot=n_boot, seed=seed + offset
        )
        pooled = result[result["sample"].eq("Pooled")].iloc[0]
        balance = covariate_balance(eligible, matched)
        rows.append({
            "upstream_destroyed_threshold": threshold,
            "eligible_n": len(eligible), "pairs": matched.pair_id.nunique(),
            "max_abs_smd": balance.smd_after.abs().max(),
            "first_stage": pooled.first_stage,
            "directional_contrast": pooled.directional_contrast,
            "directional_lo": pooled.directional_contrast_lo,
            "directional_hi": pooled.directional_contrast_hi,
            "local_iv": pooled.local_iv,
            "local_iv_lo": pooled.local_iv_lo,
            "local_iv_hi": pooled.local_iv_hi,
        })
    return pd.DataFrame(rows)


def upfire_exposure_quantile_sensitivity(
        focal: pd.DataFrame, quantiles=(.75,), n_boot: int = 1000,
        seed: int = 0) -> pd.DataFrame:
    """Rematch within fire after restricting realized up-fire coupling.

    Quantile cutoffs are estimated separately within each fire from the full
    directional focal population.  This is distinct from ``up_share``: the
    restriction uses the magnitude of realized up-fire geometric coupling,
    ``F_upfire_wmean``, rather than the unweighted fraction of earlier-arrival
    visible neighbors that were destroyed.
    """
    rows = []
    for offset, quantile in enumerate(quantiles):
        cutoffs = focal.groupby("fire")["F_upfire_wmean"].quantile(quantile)
        row_cutoff = focal.fire.map(cutoffs)
        eligible = focal.loc[
            focal.F_upfire_wmean.ge(row_cutoff)
            & focal.down100.notna()
            & focal.up100.notna()
        ].copy()
        matched, _ = match_defended(eligible)
        result, _, _ = spillover_table(
            matched, n_boot=n_boot, seed=seed + offset
        )
        pooled = result[result["sample"].eq("Pooled")].iloc[0]
        balance = covariate_balance(eligible, matched)
        rows.append({
            "upfire_exposure_quantile": quantile,
            "eaton_F_upfire_cutoff": cutoffs.get("EATON", np.nan),
            "palisades_F_upfire_cutoff": cutoffs.get("PALISADES", np.nan),
            "eligible_n": len(eligible),
            "pairs": matched.pair_id.nunique(),
            "max_abs_smd": balance.smd_after.abs().max(),
            "first_stage": pooled.first_stage,
            "directional_contrast": pooled.directional_contrast,
            "directional_lo": pooled.directional_contrast_lo,
            "directional_hi": pooled.directional_contrast_hi,
            "local_iv": pooled.local_iv,
            "local_iv_lo": pooled.local_iv_lo,
            "local_iv_hi": pooled.local_iv_hi,
        })
    return pd.DataFrame(rows)


def design_sensitivity(project_root: Path, primary_focal: pd.DataFrame,
                       cache_dir: Path, n_boot: int = 1000,
                       seed: int = 0) -> pd.DataFrame:
    """Vary the arrival window and neighbor radius, rematching every design."""
    specs = [
        ("Arrival window: 1.5 h", 1.5, 100),
        ("Primary: 3 h, 100 ft", 3.0, 100),
        ("Arrival window: 6 h", 6.0, 100),
        ("Neighbor radius: 50 ft", 3.0, 50),
        ("Neighbor radius: 150 ft", 3.0, 150),
    ]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset, (label, hours, radius_ft) in enumerate(specs):
        if hours == 3.0 and radius_ft == 100:
            focal = primary_focal
        else:
            tag = f"h{hours:g}_r{radius_ft:g}".replace(".", "p")
            cache = cache_dir / f"defense_focal_{tag}.parquet"
            if cache.exists():
                focal = pd.read_parquet(cache)
            else:
                focal = build_focal_table(
                    project_root, max_abs_dt_hours=hours,
                    neighbor_radius_m=radius_ft * 0.3048,
                )
                focal.to_parquet(cache, index=False)
        eligible, matched, _ = prepare_spillover_match(focal, threshold=.75)
        result, _, _ = spillover_table(
            matched, n_boot=n_boot, seed=seed + offset
        )
        pooled = result[result["sample"].eq("Pooled")].iloc[0]
        balance = covariate_balance(eligible, matched)
        rows.append({
            "design": label, "arrival_window_h": hours,
            "neighbor_radius_ft": radius_ft,
            "eligible_n": len(eligible), "pairs": matched.pair_id.nunique(),
            "max_abs_smd": balance.smd_after.abs().max(),
            "first_stage": pooled.first_stage,
            "directional_contrast": pooled.directional_contrast,
            "directional_lo": pooled.directional_contrast_lo,
            "directional_hi": pooled.directional_contrast_hi,
            "local_iv": pooled.local_iv,
            "local_iv_lo": pooled.local_iv_lo,
            "local_iv_hi": pooled.local_iv_hi,
        })
    return pd.DataFrame(rows)


def propensity_overlap_table(eligible: pd.DataFrame,
                             matched: pd.DataFrame) -> pd.DataFrame:
    """Summarize propensity-score overlap before and after IV matching."""
    treated_min = eligible.loc[eligible.defended.eq(1), "propensity"].min()
    treated_max = eligible.loc[eligible.defended.eq(1), "propensity"].max()
    rows = []
    for sample_name, frame in [("Eligible", eligible), ("Matched", matched)]:
        for defended, group in frame.groupby("defended"):
            rows.append({
                "sample": sample_name,
                "group": "Defended" if defended else "Undefended",
                "n": len(group), "propensity_min": group.propensity.min(),
                "propensity_p05": group.propensity.quantile(.05),
                "propensity_median": group.propensity.median(),
                "propensity_p95": group.propensity.quantile(.95),
                "propensity_max": group.propensity.max(),
                "within_treated_range_share": group.propensity.between(
                    treated_min, treated_max
                ).mean(),
            })
    return pd.DataFrame(rows)
