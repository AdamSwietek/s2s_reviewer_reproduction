"""Create a machine-readable dictionary for all tabular package inputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PAIR_DESCRIPTIONS = {
    "source_BLD_ID": "Identifier of the emitting/source building in the directed pair",
    "target_BLD_ID": "Identifier of the receiving/target building in the directed pair",
    "ssd_m": "Minimum visible patch-to-patch surface distance, metres",
    "geom_coupling_sum": "Sum of bounded patch contributions across the directed building pair",
    "geom_coupling_max": "Maximum bounded patch contribution across all distances",
    "geom_coupling_max_100m": "Maximum patch contribution among visible patches within 100 m; weakest-patch sensitivity field",
    "geom_coupling_wmean": "Whole-receiver-surface coupling contributed by one visible emitting building; directed SEN input",
    "visible_patch_pairs": "Number of unobstructed source-target patch pairs aggregated",
}


def main() -> None:
    rows = []
    manifest = pd.read_csv(DATA / "manifest.csv")
    for relative in manifest.loc[manifest.format.eq("parquet"), "path"]:
        path = ROOT / relative
        schema = pq.read_schema(path)
        for field in schema:
            rows.append({
                "table": relative,
                "column": field.name,
                "arrow_type": str(field.type),
                "nullable": field.nullable,
                "description": PAIR_DESCRIPTIONS.get(
                    field.name,
                    "Analysis field retained from the frozen post-LOS source table; see notebook using this table and docs/data_provenance.md.",
                ),
            })
    dictionary = pd.DataFrame(rows)
    dictionary.to_csv(DATA / "data_dictionary.csv", index=False)
    print(f"Wrote {len(dictionary):,} column records to data/data_dictionary.csv")


if __name__ == "__main__":
    main()
