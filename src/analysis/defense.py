"""Matched direct and directional spillover analyses of documented defense."""
from __future__ import annotations

from pathlib import Path
import tempfile

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree


FIRES = ("EATON", "PALISADES")
M100 = 30.48
IV_COVARIATES = [
    "up_share", "l_F_upfire_wmean", "l_F_total_wmean",
    "n_destroyed_bldgs", "l_n_neighbors_500ft",
    "wind_speed_arrival_mph", "arrival", "elevation", "ndvi_mean",
]


def build_focal_table(project_root: Path, clock: str = "T_arrival_hrs",
                      max_abs_dt_hours: float = 3.0,
                      neighbor_radius_m: float = M100,
                      arrival_override: pd.DataFrame | None = None,
                      upfire_override: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one row per focal structure eligible for directional analysis."""
    if max_abs_dt_hours <= 0 or neighbor_radius_m <= 0:
        raise ValueError("Arrival window and neighbor radius must be positive")
    data = Path(project_root) / "data"
    dins = pd.read_parquet(
        data / "enrichment" / "dins.parquet",
        columns=["BLD_ID", "is_defended"],
    )
    dins["BLD_ID"] = dins.BLD_ID.astype(str)
    defended_ids = set(dins.loc[dins.is_defended.eq(True), "BLD_ID"])
    buildings = pd.read_parquet(
        data / "buildings_enriched.parquet",
        columns=["BLD_ID", "fire", "damage", "is_destroyed", clock],
    )
    buildings["BLD_ID"] = buildings.BLD_ID.astype(str)
    assessed_damage = {
        "No Damage", "Affected (1-9%)", "Minor (10-25%)",
        "Major (26-50%)", "Destroyed (>50%)",
    }
    buildings["is_assessed"] = buildings.damage.isin(assessed_damage)
    buildings["any_damage"] = (
        buildings.is_assessed & buildings.damage.ne("No Damage")
    ).astype(float)
    buildings.loc[~buildings.is_assessed, "any_damage"] = np.nan
    if arrival_override is not None:
        alternate = arrival_override[["BLD_ID", "fire", "arrival"]].copy()
        alternate["BLD_ID"] = alternate.BLD_ID.astype(str)
        buildings = buildings.drop(columns=[clock]).merge(
            alternate, on=["BLD_ID", "fire"], how="left", validate="many_to_one"
        ).rename(columns={"arrival": clock})

    pair_parts = []
    for fire in FIRES:
        b = buildings[
            buildings.fire.eq(fire) & buildings.BLD_ID.ne("None")
        ].drop_duplicates("BLD_ID")
        building_map = b[["BLD_ID", clock, "is_destroyed", "any_damage"]].rename(
            columns={clock: "arrival", "is_destroyed": "destroyed"}
        )
        edge_path = data / "pairs" / f"{fire}_directed_pairs.parquet"
        escaped_path = str(edge_path).replace("'", "''")
        con = duckdb.connect()
        con.execute("SET threads=2")
        con.execute("SET memory_limit='3GB'")
        temp_dir = Path(tempfile.gettempdir()) / "wildfire_defense_duckdb"
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{str(temp_dir)}'")
        con.register("building_map", building_map)
        pairs = con.execute(f"""
            SELECT CAST(p.source_BLD_ID AS VARCHAR) AS focal,
                   CAST(p.target_BLD_ID AS VARCHAR) AS neighbor,
                   ANY_VALUE(t.arrival - s.arrival) AS dt,
                   MIN(p.ssd_m) AS distance_m,
                   ANY_VALUE(t.destroyed) AS neighbor_destroyed,
                   ANY_VALUE(t.any_damage) AS neighbor_any_damage
            FROM read_parquet('{escaped_path}') AS p
            INNER JOIN building_map AS s
                ON CAST(p.source_BLD_ID AS VARCHAR) = s.BLD_ID
            INNER JOIN building_map AS t
                ON CAST(p.target_BLD_ID AS VARCHAR) = t.BLD_ID
            WHERE s.arrival IS NOT NULL AND t.arrival IS NOT NULL
              AND ABS(t.arrival - s.arrival) <= {float(max_abs_dt_hours)}
              AND t.destroyed IS NOT NULL
            GROUP BY 1, 2
        """).fetchdf()
        con.close()
        pair_parts.append(pairs.assign(fire=fire))
    edges = pd.concat(pair_parts, ignore_index=True)

    sides = edges.assign(
        up=edges.dt.lt(0), down=edges.dt.gt(0)
    ).groupby("focal")[["up", "down"]].sum()
    eligible = sides[(sides.up.gt(0)) & (sides.down.gt(0))].index
    edges = edges[edges.focal.isin(eligible)]
    up_share = (
        edges[edges.dt.lt(0)].groupby("focal").neighbor_destroyed.mean()
        .rename("up_share")
    )
    outcome_edges = edges[
        ~edges.neighbor.isin(defended_ids)
        & edges.distance_m.le(float(neighbor_radius_m))
    ]
    down = outcome_edges[outcome_edges.dt.gt(0)].groupby("focal").agg(
        down100=("neighbor_destroyed", "mean"),
        down100_any_damage=("neighbor_any_damage", "mean"),
        n_down100=("neighbor_destroyed", "size"),
        n_down100_assessed=("neighbor_any_damage", "count"),
    )
    up = outcome_edges[outcome_edges.dt.lt(0)].groupby("focal").agg(
        up100=("neighbor_destroyed", "mean"),
        up100_any_damage=("neighbor_any_damage", "mean"),
        n_up100=("neighbor_destroyed", "size"),
        n_up100_assessed=("neighbor_any_damage", "count"),
    )
    damaged_edges = outcome_edges[outcome_edges.neighbor_any_damage.eq(1)]
    down_escalation = (
        damaged_edges[damaged_edges.dt.gt(0)].groupby("focal")
        .neighbor_destroyed.agg([("down100_escalation", "mean"),
                                 ("n_down100_damaged", "size")])
    )
    up_escalation = (
        damaged_edges[damaged_edges.dt.lt(0)].groupby("focal")
        .neighbor_destroyed.agg([("up100_escalation", "mean"),
                                 ("n_up100_damaged", "size")])
    )
    focal = pd.concat(
        [up_share, down, up, down_escalation, up_escalation], axis=1
    ).reset_index()
    focal = focal.rename(columns={"focal": "BLD_ID"})

    covariates = pd.read_parquet(
        data / "radex.parquet",
        columns=[
            "BLD_ID", "fire", "damage", "F_upfire_wmean", "F_total_wmean",
            "F_destroyed_wmean", "n_destroyed_bldgs", "n_neighbors_500ft",
            "wind_speed_arrival_mph", "elevation", clock,
        ],
    )
    covariates["BLD_ID"] = covariates.BLD_ID.astype(str)
    if arrival_override is not None:
        covariates = covariates.drop(columns=[clock]).merge(
            alternate, on=["BLD_ID", "fire"], how="left", validate="many_to_one"
        ).rename(columns={"arrival": clock})
    if upfire_override is not None:
        alternate_upfire = upfire_override[
            ["BLD_ID", "fire", "F_upfire_wmean"]
        ].copy()
        alternate_upfire["BLD_ID"] = alternate_upfire.BLD_ID.astype(str)
        covariates = covariates.drop(columns=["F_upfire_wmean"]).merge(
            alternate_upfire, on=["BLD_ID", "fire"], how="left",
            validate="many_to_one",
        )
    vegetation = pd.read_parquet(
        data / "enrichment" / "vegetation.parquet",
        columns=["BLD_ID", "ndvi_mean"],
    )
    vegetation["BLD_ID"] = vegetation.BLD_ID.astype(str)
    frame = (
        focal.merge(covariates, on="BLD_ID")
        .merge(vegetation, on="BLD_ID", how="left")
        .merge(dins, on="BLD_ID", how="left")
    )
    frame["defended"] = frame.is_defended.eq(True).astype(int)
    frame["destroyed"] = frame.damage.eq("Destroyed (>50%)").astype(int)
    frame["survived"] = 1 - frame.destroyed
    frame["outcome3"] = np.where(
        frame.damage.eq("Destroyed (>50%)"), "Destroyed",
        np.where(frame.damage.eq("No Damage"), "Undamaged", "Partial"),
    )
    frame["arrival"] = frame[clock]
    for column in ["F_upfire_wmean", "F_total_wmean", "n_neighbors_500ft"]:
        frame[f"l_{column}"] = np.log1p(frame[column])
    return frame.dropna(
        subset=IV_COVARIATES + ["defended", "damage"]
    ).reset_index(drop=True)


def match_defended(focal: pd.DataFrame, caliper_sd: float = .2) -> tuple[pd.DataFrame, object]:
    """One-to-one nearest-neighbor caliper matching exactly within fire."""
    standardized = (
        (focal[IV_COVARIATES] - focal[IV_COVARIATES].mean())
        / focal[IV_COVARIATES].std()
    )
    standardized["fire_b"] = focal.fire.eq("PALISADES").astype(int)
    standardized["defended"] = focal.defended.to_numpy()
    propensity_model = smf.logit(
        "defended ~ " + " + ".join(IV_COVARIATES + ["fire_b"]),
        data=standardized,
    ).fit(disp=0, maxiter=300)
    propensity = np.clip(np.asarray(propensity_model.predict()), 1e-6, 1-1e-6)
    logit_score = np.log(propensity / (1 - propensity))
    work = focal.assign(propensity=propensity, logit_propensity=logit_score)
    caliper = caliper_sd * work.logit_propensity.std()
    rows, pair_id = [], 0
    for _, fire_frame in work.groupby("fire"):
        treated = fire_frame[fire_frame.defended.eq(1)]
        controls = fire_frame[fire_frame.defended.eq(0)]
        tree = cKDTree(controls[["logit_propensity"]].to_numpy())
        used = set()
        for treated_index in treated.index:
            distance, neighbor_index = tree.query(
                work.loc[[treated_index], ["logit_propensity"]].to_numpy(),
                k=min(30, len(controls)),
            )
            for dist, j in zip(np.ravel(distance), np.ravel(neighbor_index)):
                control_index = controls.index[int(j)]
                if dist <= caliper and control_index not in used:
                    used.add(control_index)
                    pair_id += 1
                    rows.extend([(treated_index, pair_id), (control_index, pair_id)])
                    break
    matched = work.loc[[row[0] for row in rows]].copy()
    matched["pair_id"] = [row[1] for row in rows]
    return matched, propensity_model


def prepare_spillover_match(focal: pd.DataFrame, threshold: float = .75,
                            caliper_sd: float = .2):
    """Restrict to the IV population, then rematch defended structures.

    The restriction is imposed before propensity-score matching so that the
    high-exposure spillover analysis retains complete matched pairs and
    covariate balance within its own target population.
    """
    eligible = focal.loc[
        focal.up_share.ge(threshold)
        & focal.down100.notna()
        & focal.up100.notna()
    ].copy()
    matched, propensity_model = match_defended(
        eligible, caliper_sd=caliper_sd
    )
    standardized = (
        (eligible[IV_COVARIATES] - eligible[IV_COVARIATES].mean())
        / eligible[IV_COVARIATES].std()
    )
    standardized["fire_b"] = eligible.fire.eq("PALISADES").astype(int)
    eligible["propensity"] = np.clip(
        np.asarray(propensity_model.predict(standardized)), 1e-6, 1 - 1e-6
    )
    return eligible, matched, propensity_model


def covariate_balance(focal: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Return before- and after-match standardized mean differences."""
    def smd(frame, column):
        treated = frame.loc[frame.defended.eq(1), column]
        control = frame.loc[frame.defended.eq(0), column]
        pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
        return float((treated.mean() - control.mean()) / pooled_sd)
    return pd.DataFrame([
        {"covariate": column,
         "smd_before": smd(focal, column),
         "smd_after": smd(matched, column)}
        for column in IV_COVARIATES
    ])


def _stratified_boot_difference(frame, outcome, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    point = (
        frame.loc[frame.defended.eq(1), outcome].mean()
        - frame.loc[frame.defended.eq(0), outcome].mean()
    )
    draws = []
    strata = [group.index.to_numpy() for _, group in frame.groupby("fire")]
    for _ in range(n_boot):
        sample = frame.loc[np.concatenate([
            rng.choice(indices, len(indices), replace=True) for indices in strata
        ])]
        draws.append(
            sample.loc[sample.defended.eq(1), outcome].mean()
            - sample.loc[sample.defended.eq(0), outcome].mean()
        )
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def direct_effect_table(matched: pd.DataFrame, n_boot=2000, seed=0):
    """Matched survival contrasts pooled and separately by fire."""
    rows = []
    samples = [("Pooled", matched)] + [
        (fire.title(), matched[matched.fire.eq(fire)]) for fire in FIRES
    ]
    for offset, (label, frame) in enumerate(samples):
        estimate, lo, hi = _stratified_boot_difference(
            frame, "survived", n_boot=n_boot, seed=seed + offset)
        paired = smf.ols("survived ~ defended", data=frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame.pair_id}
        )
        rows.append({
            "sample": label, "matched_pairs": int(frame.pair_id.nunique()),
            "survival_difference_pp": 100 * estimate,
            "ci_lo": 100 * lo, "ci_hi": 100 * hi,
            "p_value": float(paired.pvalues["defended"]),
        })
    outcomes = (
        matched.groupby("defended").outcome3.value_counts(normalize=True)
        .unstack(fill_value=0)[["Undamaged", "Partial", "Destroyed"]]
        .rename(index={0: "Matched control", 1: "Defended"}) * 100
    ).reset_index(names="group")
    return pd.DataFrame(rows), outcomes


def iv_estimate(frame: pd.DataFrame, down_col: str = "down100",
                up_col: str = "up100") -> dict[str, float]:
    """Fire-weighted first stage, directional contrast and local IV ratio."""
    parts, weights = [], []
    for _, fire_frame in frame.groupby("fire"):
        defended = fire_frame[fire_frame.defended.eq(1)]
        control = fire_frame[fire_frame.defended.eq(0)]
        if min(len(defended), len(control)) < 2:
            continue
        parts.append((
            defended.survived.mean() - control.survived.mean(),
            defended[down_col].mean() - control[down_col].mean(),
            defended[up_col].mean() - control[up_col].mean(),
        ))
        weights.append(len(fire_frame))
    weights = np.asarray(weights) / np.sum(weights)
    first_stage, down, placebo = (
        float(np.dot(weights, [part[i] for part in parts])) for i in range(3)
    )
    directional = down - placebo
    return {
        "first_stage": first_stage,
        "downfire_contrast": down,
        "upfire_placebo": placebo,
        "directional_contrast": directional,
        "local_iv": directional / first_stage,
    }


def _iv_bootstrap(frame, n_boot=2000, seed=0, down_col="down100",
                  up_col="up100"):
    """Bootstrap complete matched pairs separately within each fire."""
    rng = np.random.default_rng(seed)
    fire_draws, fire_weights = [], []
    for _, fire_frame in frame.groupby("fire"):
        pair_differences = []
        for _, pair in fire_frame.groupby("pair_id"):
            treated = pair[pair.defended.eq(1)].iloc[0]
            control = pair[pair.defended.eq(0)].iloc[0]
            pair_differences.append([
                treated.survived - control.survived,
                treated[down_col] - control[down_col],
                treated[up_col] - control[up_col],
            ])
        pair_differences = np.asarray(pair_differences, dtype=float)
        indices = rng.integers(
            0, len(pair_differences), size=(n_boot, len(pair_differences))
        )
        fire_draws.append(pair_differences[indices].mean(axis=1))
        fire_weights.append(len(fire_frame))
    weights = np.asarray(fire_weights, dtype=float)
    weights /= weights.sum()
    pooled = np.sum(
        np.stack(fire_draws, axis=0) * weights[:, None, None], axis=0
    )
    first_stage, down, placebo = pooled.T
    directional = down - placebo
    draws = {
        "first_stage": first_stage,
        "downfire_contrast": down,
        "upfire_placebo": placebo,
        "directional_contrast": directional,
        "local_iv": directional / first_stage,
    }
    return {key: np.nanpercentile(values, [2.5, 97.5])
            for key, values in draws.items()}


def spillover_table(iv_matched: pd.DataFrame, n_boot=2000, seed=0):
    """Estimate directional spillovers after restriction and rematching."""
    high = iv_matched.copy()
    if high[["down100", "up100"]].isna().any().any():
        raise ValueError("The IV matched sample must have complete neighbor outcomes")
    pair_sizes = high.groupby("pair_id").size()
    if not pair_sizes.eq(2).all():
        raise ValueError("The IV matched sample must contain complete matched pairs")
    rows, figure_results = [], {}
    samples = [("Pooled", high)] + [
        (fire.title(), high[high.fire.eq(fire)]) for fire in FIRES
    ]
    for offset, (label, frame) in enumerate(samples):
        estimate = iv_estimate(frame)
        interval = _iv_bootstrap(frame, n_boot=n_boot, seed=seed + offset)
        first_stage_model = smf.ols("survived ~ defended", data=frame).fit(
            cov_type="HC1"
        )
        first_stage_f = float(
            (first_stage_model.params["defended"]
             / first_stage_model.bse["defended"]) ** 2
        )
        row = {
            "sample": label, "n": len(frame),
            "defended_n": int(frame.defended.sum()),
            "first_stage": estimate["first_stage"],
            "first_stage_F": first_stage_f,
        }
        for key in ["downfire_contrast", "upfire_placebo",
                    "directional_contrast", "local_iv"]:
            row[key] = estimate[key]
            row[f"{key}_lo"] = float(interval[key][0])
            row[f"{key}_hi"] = float(interval[key][1])
        rows.append(row)
        figure_results[label.lower()] = (
            len(frame),
            {"did": estimate["directional_contrast"],
             "late": estimate["local_iv"]},
            {"did": interval["directional_contrast"],
             "late": interval["local_iv"]},
        )
    return pd.DataFrame(rows), figure_results, high


def spillover_stage_table(iv_matched: pd.DataFrame, n_boot=2000, seed=0):
    """Decompose neighbor loss into any damage and escalation after damage.

    The same rematched high-exposure population is used as the starting point.
    For each outcome, only complete matched pairs with observed earlier- and
    later-arrival neighbor outcomes for both members are retained. Conditional
    destruction is defined only where each focal has at least one damaged
    neighbor in both temporal groups.
    """
    outcomes = [
        ("Any damage", "down100_any_damage", "up100_any_damage"),
        ("Destruction", "down100", "up100"),
        ("Destruction conditional on damage",
         "down100_escalation", "up100_escalation"),
    ]
    rows = []
    for outcome_index, (outcome, down_col, up_col) in enumerate(outcomes):
        complete = iv_matched.dropna(subset=[down_col, up_col]).copy()
        valid_pairs = complete.groupby("pair_id").size()
        valid_pairs = valid_pairs[valid_pairs.eq(2)].index
        complete = complete[complete.pair_id.isin(valid_pairs)].copy()
        samples = [("Pooled", complete)] + [
            (fire.title(), complete[complete.fire.eq(fire)]) for fire in FIRES
        ]
        for sample_index, (label, frame) in enumerate(samples):
            if frame.empty:
                continue
            estimate = iv_estimate(frame, down_col=down_col, up_col=up_col)
            interval = _iv_bootstrap(
                frame, n_boot=n_boot,
                seed=seed + outcome_index * 100 + sample_index,
                down_col=down_col, up_col=up_col,
            )
            rows.append({
                "outcome": outcome,
                "sample": label,
                "n": len(frame),
                "matched_pairs": int(frame.pair_id.nunique()),
                "defended_n": int(frame.defended.sum()),
                **estimate,
                **{
                    f"{key}_{bound}": float(interval[key][index])
                    for key in ["first_stage", "downfire_contrast",
                                "upfire_placebo", "directional_contrast",
                                "local_iv"]
                    for bound, index in [("lo", 0), ("hi", 1)]
                },
            })
    return pd.DataFrame(rows)


def build_view_decomposition(project_root: Path, focal: pd.DataFrame,
                             clock="T_arrival_hrs",
                             arrival_override: pd.DataFrame | None = None,
                             max_abs_dt_hours: float = 3.0) -> pd.DataFrame:
    """Decompose each focal's up/down-fire view by defense and destruction."""
    data = Path(project_root) / "data"
    dins = pd.read_parquet(
        data / "enrichment" / "dins.parquet",
        columns=["BLD_ID", "is_defended"],
    )
    dins["BLD_ID"] = dins.BLD_ID.astype(str)
    defended_ids = set(dins.loc[dins.is_defended.eq(True), "BLD_ID"])
    buildings = pd.read_parquet(
        data / "buildings_enriched.parquet",
        columns=["BLD_ID", "fire", "is_destroyed", clock],
    )
    buildings["BLD_ID"] = buildings.BLD_ID.astype(str)
    if arrival_override is not None:
        alternate = arrival_override[["BLD_ID", "fire", "arrival"]].copy()
        alternate["BLD_ID"] = alternate.BLD_ID.astype(str)
        buildings = buildings.drop(columns=[clock]).merge(
            alternate, on=["BLD_ID", "fire"], how="left", validate="many_to_one"
        ).rename(columns={"arrival": clock})
    outputs = []
    for fire in FIRES:
        b = buildings[
            buildings.fire.eq(fire) & buildings.BLD_ID.ne("None")
        ].drop_duplicates("BLD_ID")
        building_map = b[["BLD_ID", clock, "is_destroyed"]].rename(
            columns={clock: "arrival", "is_destroyed": "destroyed"}
        )
        building_map["neighbor_defended"] = building_map.BLD_ID.isin(defended_ids)
        edge_path = data / "pairs" / f"{fire}_directed_pairs.parquet"
        escaped_path = str(edge_path).replace("'", "''")
        con = duckdb.connect()
        con.execute("SET threads=2")
        con.execute("SET memory_limit='3GB'")
        temp_dir = Path(tempfile.gettempdir()) / "wildfire_defense_duckdb"
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{str(temp_dir)}'")
        con.register("building_map", building_map)
        view = con.execute(f"""
            SELECT CAST(p.source_BLD_ID AS VARCHAR) AS BLD_ID,
                SUM(CASE WHEN t.arrival < s.arrival
                    THEN p.geom_coupling_sum ELSE 0 END)
                    AS F_up_potential,
                SUM(CASE WHEN t.arrival > s.arrival
                    THEN p.geom_coupling_sum ELSE 0 END)
                    AS F_down_potential,
                SUM(CASE WHEN t.arrival < s.arrival AND t.neighbor_defended
                    THEN p.geom_coupling_sum ELSE 0 END)
                    AS F_up_defended,
                SUM(CASE WHEN t.arrival > s.arrival AND t.neighbor_defended
                    THEN p.geom_coupling_sum ELSE 0 END)
                    AS F_down_defended,
                SUM(CASE WHEN t.arrival < s.arrival AND t.destroyed = 1
                    THEN p.geom_coupling_sum ELSE 0 END)
                    AS F_up_destroyed
            FROM read_parquet('{escaped_path}') AS p
            INNER JOIN building_map AS s
                ON CAST(p.source_BLD_ID AS VARCHAR) = s.BLD_ID
            INNER JOIN building_map AS t
                ON CAST(p.target_BLD_ID AS VARCHAR) = t.BLD_ID
            WHERE s.arrival IS NOT NULL AND t.arrival IS NOT NULL
              AND ABS(t.arrival - s.arrival) <= {float(max_abs_dt_hours)}
            GROUP BY 1
        """).fetchdf()
        con.close()
        outputs.append(view)
    return focal.merge(pd.concat(outputs, ignore_index=True), on="BLD_ID", how="left")


def _fit_mechanism_models(view_frame: pd.DataFrame):
    """Prepare and fit the two emitter-removal diagnostic specifications."""
    frame = view_frame.copy()
    columns = ["F_up_potential", "F_down_potential", "F_up_defended",
               "F_down_defended", "F_up_destroyed"]
    frame[columns] = frame[columns].fillna(0)
    for source, target in [
        ("F_up_potential", "l_up_potential"),
        ("F_down_potential", "l_down_potential"),
        ("F_up_defended", "l_up_defended"),
        ("F_down_defended", "l_down_defended"),
    ]:
        frame[target] = np.log1p(frame[source])
    frame["destroyed_up_share"] = np.where(
        frame.F_up_potential.gt(0),
        frame.F_up_destroyed / frame.F_up_potential, 0,
    )
    frame["fire_b"] = frame.fire.eq("PALISADES").astype(int)
    base = (
        "defended + l_up_defended + l_down_defended + l_up_potential + "
        "l_down_potential + l_F_total_wmean + l_n_neighbors_500ft + "
        "wind_speed_arrival_mph + arrival + elevation + ndvi_mean + fire_b"
    )
    models = {
        "Potential coupling only": smf.logit(
            "destroyed ~ " + base, data=frame
        ).fit(disp=0, maxiter=300),
        "+ destroyed share": smf.logit(
            "destroyed ~ " + base + " + destroyed_up_share", data=frame
        ).fit(disp=0, maxiter=300),
    }
    return frame, models


def mechanism_regression_table(view_frame: pd.DataFrame) -> pd.DataFrame:
    """Return complete coefficient output for the emitter-removal models."""
    _, models = _fit_mechanism_models(view_frame)
    rows = []
    for model_name, model in models.items():
        intervals = model.conf_int(alpha=.05)
        for term in model.params.index:
            rows.append({
                "model": model_name,
                "term": term,
                "coefficient": float(model.params[term]),
                "standard_error": float(model.bse[term]),
                "z": float(model.tvalues[term]),
                "p_value": float(model.pvalues[term]),
                "ci_lo": float(intervals.loc[term, 0]),
                "ci_hi": float(intervals.loc[term, 1]),
                "odds_ratio": float(np.exp(model.params[term])),
                "odds_ratio_ci_lo": float(np.exp(intervals.loc[term, 0])),
                "odds_ratio_ci_hi": float(np.exp(intervals.loc[term, 1])),
                "n": int(model.nobs),
                "events": int(np.asarray(model.model.endog).sum()),
                "parameters": int(len(model.params)),
                "log_likelihood": float(model.llf),
                "aic": float(model.aic),
                "mcfadden_pseudo_r2": float(model.prsquared),
                "converged": bool(model.mle_retvals.get("converged", False)),
            })
    return pd.DataFrame(rows)


def mechanism_table(view_frame: pd.DataFrame) -> pd.DataFrame:
    """Fit the received-defense attenuation and down-fire placebo models."""
    frame, models = _fit_mechanism_models(view_frame)
    rows = []
    terms = [
        ("l_up_defended", "Defended component of up-fire view"),
        ("l_down_defended", "Defended down-fire view (placebo)"),
        ("destroyed_up_share", "Destroyed fraction of up-fire view"),
    ]
    for model_name, model in models.items():
        for term, label in terms:
            if term not in model.params:
                continue
            shifted = frame.copy()
            shifted[term] = frame[term] + frame[term].std()
            effect = 100 * (model.predict(shifted) - model.predict(frame)).mean()
            rows.append({
                "model": model_name, "term": label,
                "ame_pp_per_sd": float(effect),
                "coefficient": float(model.params[term]),
                "p_value": float(model.pvalues[term]),
                "n": int(model.nobs),
            })
    return pd.DataFrame(rows)


def mechanism_partial_damage_table(view_frame: pd.DataFrame) -> pd.DataFrame:
    """Separate damage onset from remaining partial after damage.

    The conditional outcome equals one for a partially damaged structure and
    zero for a destroyed structure. Undamaged structures are excluded from
    that stage, avoiding a partial-versus-everything contrast that would mix
    two qualitatively different comparison outcomes.
    """
    frame = view_frame.copy()
    columns = ["F_up_potential", "F_down_potential", "F_up_defended",
               "F_down_defended", "F_up_destroyed"]
    frame[columns] = frame[columns].fillna(0)
    for source, target in [
        ("F_up_potential", "l_up_potential"),
        ("F_down_potential", "l_down_potential"),
        ("F_up_defended", "l_up_defended"),
        ("F_down_defended", "l_down_defended"),
    ]:
        frame[target] = np.log1p(frame[source])
    frame["destroyed_up_share"] = np.where(
        frame.F_up_potential.gt(0),
        frame.F_up_destroyed / frame.F_up_potential, 0,
    )
    frame["fire_b"] = frame.fire.eq("PALISADES").astype(int)
    frame["any_damage"] = frame.outcome3.ne("Undamaged").astype(int)
    frame["partial_given_damage"] = frame.outcome3.eq("Partial").astype(int)

    base = (
        "defended + l_up_defended + l_down_defended + l_up_potential + "
        "l_down_potential + l_F_total_wmean + l_n_neighbors_500ft + "
        "wind_speed_arrival_mph + arrival + elevation + ndvi_mean + fire_b"
    )
    stages = [
        ("Any damage vs undamaged", frame, "any_damage"),
        ("Partial vs destroyed, among damaged",
         frame[frame.any_damage.eq(1)].copy(), "partial_given_damage"),
    ]
    terms = [
        ("l_up_defended", "Defended component of up-fire view"),
        ("l_down_defended", "Defended down-fire view (control)"),
        ("destroyed_up_share", "Destroyed fraction of up-fire view"),
    ]
    rows = []
    for stage, stage_frame, outcome in stages:
        models = {
            "Potential coupling only": smf.logit(
                f"{outcome} ~ " + base, data=stage_frame
            ).fit(disp=0, maxiter=300),
            "+ destroyed share": smf.logit(
                f"{outcome} ~ " + base + " + destroyed_up_share",
                data=stage_frame,
            ).fit(disp=0, maxiter=300),
        }
        for model_name, model in models.items():
            for term, label in terms:
                if term not in model.params:
                    continue
                shifted = stage_frame.copy()
                shifted[term] = stage_frame[term] + stage_frame[term].std()
                effect = 100 * (
                    model.predict(shifted) - model.predict(stage_frame)
                ).mean()
                rows.append({
                    "stage": stage,
                    "model": model_name,
                    "term": label,
                    "ame_pp_per_sd": float(effect),
                    "coefficient": float(model.params[term]),
                    "p_value": float(model.pvalues[term]),
                    "n": int(model.nobs),
                })
    return pd.DataFrame(rows)


def publication_table_markdown(direct: pd.DataFrame, outcomes: pd.DataFrame,
                               spillover: pd.DataFrame,
                               mechanism: pd.DataFrame) -> str:
    """Format the primary defense estimates as a compact manuscript table."""
    lines = [
        "# Table 2 | Documented structure defense",
        "",
        "## A. Matched association with focal-structure survival",
        "",
        "| Sample | Matched pairs | Difference, pp (95% CI) | P |",
        "|---|---:|---:|---:|",
    ]
    for row in direct.itertuples(index=False):
        p = "<0.001" if row.p_value < .001 else f"{row.p_value:.3f}"
        lines.append(
            f"| {row.sample} | {row.matched_pairs:,} | "
            f"{row.survival_difference_pp:.1f} "
            f"({row.ci_lo:.1f} to {row.ci_hi:.1f}) | {p} |"
        )
    lines.extend([
        "", "## B. Outcomes in the pooled matched sample", "",
        "| Group | Undamaged, % | Partial, % | Destroyed, % |",
        "|---|---:|---:|---:|",
    ])
    for row in outcomes.itertuples(index=False):
        lines.append(
            f"| {row.group} | {row.Undamaged:.1f} | {row.Partial:.1f} | "
            f"{row.Destroyed:.1f} |"
        )
    pooled = spillover[spillover["sample"].eq("Pooled")].iloc[0]
    lines.extend([
        "", "## C. Pooled directional spillover", "",
        "| Estimand | Estimate, pp (95% CI) |",
        "|---|---:|",
        f"| First-stage focal survival | {100*pooled.first_stage:.1f} |",
        f"| Placebo-corrected directional contrast | "
        f"{100*pooled.directional_contrast:.1f} "
        f"({100*pooled.directional_contrast_lo:.1f} to "
        f"{100*pooled.directional_contrast_hi:.1f}) |",
        f"| Local IV estimate per focal structure saved | "
        f"{100*pooled.local_iv:.1f} ({100*pooled.local_iv_lo:.1f} to "
        f"{100*pooled.local_iv_hi:.1f}) |",
        "", "## D. Emitter-removal mechanism", "",
        "| Model | Term | AME, pp per s.d. | P |",
        "|---|---|---:|---:|",
    ])
    selected = mechanism[
        mechanism.term.isin([
            "Defended component of up-fire view",
            "Defended down-fire view (placebo)",
            "Destroyed fraction of up-fire view",
        ])
    ]
    for row in selected.itertuples(index=False):
        p = "<0.001" if row.p_value < .001 else f"{row.p_value:.3f}"
        lines.append(
            f"| {row.model} | {row.term} | {row.ame_pp_per_sd:.2f} | {p} |"
        )
    lines.extend([
        "",
        "Positive direct differences indicate greater survival with documented "
        "defense; negative spillover estimates indicate fewer destroyed down-fire "
        "neighbors. AME, average marginal effect; IV, instrumental variable; "
        "pp, percentage points.",
    ])
    return "\n".join(lines)
