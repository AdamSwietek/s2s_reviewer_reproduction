"""Validate distributed schemas, population counts and headline outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

REQUIRED_COLUMNS = {
    "analysis.parquet": {
        "BLD_ID", "fire", "outcome", "is_destroyed",
        "F_destroyed_wmean", "exposed",
    },
    "radex.parquet": {"BLD_ID", "fire", "damage", "F_destroyed"},
    "pairs/EATON_directed_pairs.parquet": {
        "source_BLD_ID", "target_BLD_ID", "ssd_m", "geom_coupling_sum",
        "geom_coupling_max", "geom_coupling_max_100m",
        "geom_coupling_wmean", "visible_patch_pairs",
    },
    "pairs/PALISADES_directed_pairs.parquet": {
        "source_BLD_ID", "target_BLD_ID", "ssd_m", "geom_coupling_sum",
        "geom_coupling_max", "geom_coupling_max_100m",
        "geom_coupling_wmean", "visible_patch_pairs",
    },
    "arrival/eaton_progression_sensitivity.parquet": {
        "BLD_ID", "T_arrival_interp_hrs", "T_arrival_hrs",
        "T_arrival_direct_hrs", "T_arrival_direct_sd_hrs",
        "F_upfire_direct", "F_upfire_direct_wmean",
    },
    "arrival/arrival_interp.parquet": {
        "fire", "lon_wgs84", "lat_wgs84", "T_arrival_snap_hrs",
        "T_arrival_interp_hrs", "T_arrival_hrs", "T_arrival_sd_hrs",
        "arrival_method",
    },
}

# Primary inputs the ED01 arrival reconstruction rebuilds its products from.
REQUIRED_FILES = (
    "arrival/isochrones_eaton.gpkg",
    "arrival/isochrones_palisades.gpkg",
    "arrival/eaton_fire_events_recovered.parquet",
    "arrival/eaton_arrival_posterior_10m.tif",
    "arrival/firms_viirs_eaton_palisades_jan2025.parquet",
)


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: {actual} != {expected} ± {tolerance}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()

    for relative, expected in REQUIRED_COLUMNS.items():
        path = DATA / relative
        if not path.exists():
            raise FileNotFoundError(path)
        actual = set(pq.read_schema(path).names)
        miseng = expected - actual
        if miseng:
            raise AssertionError(f"{relative} miseng columns: {sorted(miseng)}")

    for relative in REQUIRED_FILES:
        if not (DATA / relative).exists():
            raise FileNotFoundError(DATA / relative)

    analysis = pd.read_parquet(DATA / "analysis.parquet")
    if analysis.BLD_ID.duplicated().any():
        raise AssertionError("analysis.parquet contains duplicate BLD_ID values")
    assessed = analysis.outcome.notna()
    exposed = assessed & analysis.exposed.fillna(False).astype(bool)
    if int(assessed.sum()) != 28208 or int(exposed.sum()) != 25127:
        raise AssertionError(
            f"population mismatch: assessed={assessed.sum()}, exposed={exposed.sum()}")
    by_fire = analysis.loc[exposed].groupby("fire").size().to_dict()
    if by_fire != {"EATON": 14595, "PALISADES": 10532}:
        raise AssertionError(f"fire-specific exposed counts mismatch: {by_fire}")

    pair_counts = {
        fire: pq.ParquetFile(DATA / "pairs" / f"{fire}_directed_pairs.parquet")
        .metadata.num_rows for fire in ("EATON", "PALISADES")
    }
    if pair_counts != {"EATON": 1224928, "PALISADES": 569156}:
        raise AssertionError(f"pair counts mismatch: {pair_counts}")

    if args.check_results:
        f50 = pd.read_csv(RESULTS / "f50_cross_fire.csv").set_index("outcome")
        close(f50.loc["destroyed", "ratio_PAL_to_EAT"], 1.505106, 5e-5,
              "destruction F50 ratio")
        construction = pd.read_csv(
            RESULTS / "table1a_construction_overall.csv").set_index("attribute")
        close(construction.loc["Tile roof", "tolerance_ratio"], 1.293237,
              5e-5, "tile tolerance ratio")
        defense = pd.read_csv(
            RESULTS / "table2a_defense_direct_effect.csv").set_index("sample")
        close(defense.loc["Pooled", "survival_difference_pp"], 13.538989,
              5e-4, "pooled defense survival difference")
        spill = pd.read_csv(
            RESULTS / "table2c_defense_spillover.csv").set_index("sample")
        close(spill.loc["Pooled", "directional_contrast"], -0.133954,
              5e-5, "direction-adjusted spillover contrast")
        progression_direct = pd.read_csv(
            RESULTS / "ED04_defense_progression_direct.csv")
        pooled_direct = progression_direct[
            progression_direct["sample"].eq("Pooled")]
        close(pooled_direct.survival_difference_pp.min(), 11.139456,
              5e-4, "minimum progression-map focal survival difference")
        close(pooled_direct.survival_difference_pp.max(), 13.538989,
              5e-4, "maximum progression-map focal survival difference")
        progression_spill = pd.read_csv(
            RESULTS / "ED04_defense_progression_spillover.csv")
        pooled_spill = progression_spill[progression_spill["sample"].eq("Pooled")]
        close(pooled_spill.directional_contrast.min(), -0.133954,
              5e-5, "minimum progression-map directional contrast")
        close(pooled_spill.directional_contrast.max(), -0.077766,
              5e-5, "maximum progression-map directional contrast")
        network = pd.read_csv(
            RESULTS / "sen_network_summary.csv").set_index("fire")
        if network.loc["EATON", "active_bonds"] != 9540 or network.loc[
                "PALISADES", "active_bonds"] != 7879:
            raise AssertionError("SEN active-bond counts differ from frozen results")

    print("Package validation passed.")


if __name__ == "__main__":
    main()
