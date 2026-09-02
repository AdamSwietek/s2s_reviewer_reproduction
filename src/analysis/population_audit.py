"""Build and validate the population tables used throughout the paper."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


FIRES = ("EATON", "PALISADES")

# These two pre-deduplication counts come from the frozen DINS-to-LARIAC
# linkage run. The distributed, deduplicated linkage table independently
# verifies the final count and the complete match-distance distribution.
FROZEN_LINKAGE_COUNTS = {
    "DINS records for Eaton and Palisades": 30_492,
    "DINS records linked within 50 m": 30_401,
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_text_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip()


def _analysis_tables(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome = pd.crosstab(analysis["outcome"], analysis["fire"])
    outcome = outcome.reindex(["no_damage", "partial", "destroyed"], fill_value=0)
    outcome["POOLED"] = outcome.sum(axis=1)
    outcome.index.name = "outcome"

    exposure = pd.crosstab(analysis["exposed"], analysis["fire"])
    exposure = exposure.reindex([0, 1], fill_value=0)
    exposure.index = pd.Index(["zero_F_star", "positive_F_star"], name="exposure")
    exposure["POOLED"] = exposure.sum(axis=1)
    return outcome, exposure


def _height_table(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fire in FIRES:
        path = data_dir / "nx" / f"{fire}_buildings.parquet"
        buildings = pd.read_parquet(
            path,
            columns=["height", "height_lariac", "height_source",
                     "height_impute_distance_m"],
        )
        if buildings["height"].isna().any() or buildings["height"].le(0).any():
            raise AssertionError(f"{fire}: final building heights are not all positive")
        counts = buildings["height_source"].value_counts()
        rows.append({
            "fire": fire,
            "buildings": len(buildings),
            "lariac_height": int(counts.get("lariac", 0)),
            "nearby_median": int(counts.get("nearby_median", 0)),
            "one_storey_fallback": int(counts.get("one_storey_fallback", 0)),
            "miseng_final_height": int(buildings["height"].isna().sum()),
        })
    height = pd.DataFrame(rows)
    height.loc[len(height)] = {
        "fire": "POOLED",
        **{column: int(height[column].sum()) for column in height.columns if column != "fire"},
    }
    return height


def _mesh_table(data_dir: Path, analysis: pd.DataFrame,
                radex: pd.DataFrame) -> pd.DataFrame:
    perimeters = (gpd.read_parquet(data_dir / "nx" / "fire_perims.parquet")
                  .set_index("FIRE_NAME"))
    rows: list[dict[str, object]] = []
    for fire in FIRES:
        buildings = gpd.read_parquet(
            data_dir / "nx" / f"{fire}_buildings.parquet",
            columns=["BLD_ID", "geometry"],
        )
        buildings["BLD_ID"] = _as_text_id(buildings["BLD_ID"])
        if buildings["BLD_ID"].duplicated().any():
            raise AssertionError(f"{fire}: duplicate BLD_ID values in mesh")

        perimeter = gpd.GeoSeries(
            [perimeters.loc[fire].geometry], crs=perimeters.crs
        ).to_crs(buildings.crs).iloc[0]
        inside = buildings.geometry.representative_point().within(perimeter).to_numpy()

        assessed_ids = set(analysis.loc[analysis.fire.eq(fire), "BLD_ID"])
        fire_radex = radex[radex.fire.eq(fire)].drop_duplicates("BLD_ID")
        radex_ids = set(fire_radex["BLD_ID"])
        positive_ids = set(
            fire_radex.loc[fire_radex.F_destroyed_wmean.fillna(0).gt(0), "BLD_ID"]
        )
        assessed = buildings.BLD_ID.isin(assessed_ids).to_numpy()
        has_radex = buildings.BLD_ID.isin(radex_ids).to_numpy()
        positive = buildings.BLD_ID.isin(positive_ids).to_numpy()
        rows.append({
            "fire": fire,
            "mesh_buildings": len(buildings),
            "inside_perimeter": int(inside.sum()),
            "outside_perimeter": int((~inside).sum()),
            "radex_records": int(has_radex.sum()),
            "assessed": int(assessed.sum()),
            "assessed_inside_perimeter": int((assessed & inside).sum()),
            "assessed_outside_perimeter": int((assessed & ~inside).sum()),
            "positive_F_star_all_buildings": int(positive.sum()),
            "positive_F_star_assessed": int((positive & assessed).sum()),
            "positive_F_star_unassessed": int((positive & ~assessed).sum()),
        })
    mesh = pd.DataFrame(rows)
    mesh.loc[len(mesh)] = {
        "fire": "POOLED",
        **{column: int(mesh[column].sum()) for column in mesh.columns if column != "fire"},
    }
    return mesh


def _linkage_tables(data_dir: Path, analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    links = pd.read_parquet(
        data_dir / "enrichment" / "dins.parquet",
        columns=["BLD_ID", "dins_dist_m"],
    )
    links["BLD_ID"] = _as_text_id(links["BLD_ID"])
    if links.BLD_ID.duplicated().any():
        raise AssertionError("The frozen DINS linkage table is not unique by BLD_ID")

    flow = [
        {
            "stage": "DINS records for Eaton and Palisades",
            "buildings_or_records": FROZEN_LINKAGE_COUNTS[
                "DINS records for Eaton and Palisades"],
            "definition": "Source DINS records before spatial linkage",
        },
        {
            "stage": "DINS records linked within 50 m",
            "buildings_or_records": FROZEN_LINKAGE_COUNTS[
                "DINS records linked within 50 m"],
            "definition": "Nearest LARIAC centroid within the prespecified cap",
        },
        {
            "stage": "Unique closest building matches",
            "buildings_or_records": len(links),
            "definition": "Closest DINS record retained when inspections shared a BLD_ID",
        },
        {
            "stage": "Eligible assessed structures",
            "buildings_or_records": len(analysis),
            "definition": "Linked to the exposure table with an accessible assessed outcome",
        },
        {
            "stage": "Primary exposed analysis population",
            "buildings_or_records": int(analysis.exposed.eq(1).sum()),
            "definition": "Eligible assessed structures with F* > 0",
        },
    ]
    flow = pd.DataFrame(flow)
    distances = pd.DataFrame([{
        "unique_matches": len(links),
        "median_m": float(links.dins_dist_m.median()),
        "p90_m": float(links.dins_dist_m.quantile(.90)),
        "p95_m": float(links.dins_dist_m.quantile(.95)),
        "p99_m": float(links.dins_dist_m.quantile(.99)),
        "maximum_m": float(links.dins_dist_m.max()),
    }])
    return flow, distances


def build_population_audit(data_dir: Path, results_dir: Path) -> dict[str, object]:
    """Validate the canonical population and write the reviewer audit files."""
    data_dir = Path(data_dir).resolve()
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = data_dir / "analysis.parquet"
    radex_path = data_dir / "radex.parquet"
    dins_path = data_dir / "enrichment" / "dins.parquet"
    analysis = pd.read_parquet(analysis_path)
    radex = pd.read_parquet(
        radex_path, columns=["BLD_ID", "fire", "F_destroyed_wmean"]
    )
    analysis["BLD_ID"] = _as_text_id(analysis["BLD_ID"])
    radex["BLD_ID"] = _as_text_id(radex["BLD_ID"])

    if analysis.BLD_ID.duplicated().any():
        raise AssertionError("analysis.parquet is not unique by BLD_ID")
    if set(analysis.fire.unique()) != set(FIRES):
        raise AssertionError("Unexpected fires in analysis.parquet")
    expected_exposed = analysis.F_destroyed_wmean.fillna(0).gt(0).astype(int)
    if not expected_exposed.equals(analysis.exposed.astype(int)):
        raise AssertionError("analysis.exposed does not equal F* > 0")

    outcome, exposure = _analysis_tables(analysis)
    height = _height_table(data_dir)
    mesh = _mesh_table(data_dir, analysis, radex)
    flow, linkage = _linkage_tables(data_dir, analysis)

    output_tables = {
        "population_flow": flow,
        "dins_linkage_distance": linkage,
        "analysis_outcomes": outcome.reset_index(),
        "analysis_exposure": exposure.reset_index(),
        "height_imputation": height,
        "mesh_population": mesh,
    }
    for name, table in output_tables.items():
        table.to_csv(results_dir / f"{name}.csv", index=False)

    input_paths = {
        "analysis": analysis_path,
        "radex": radex_path,
        "dins_linkage": dins_path,
        "fire_perimeters": data_dir / "nx" / "fire_perims.parquet",
        "eaton_buildings": data_dir / "nx" / "EATON_buildings.parquet",
        "palisades_buildings": data_dir / "nx" / "PALISADES_buildings.parquet",
    }
    manifest = {
        "analysis_population": {
            "assessed": len(analysis),
            "positive_F_star": int(analysis.exposed.eq(1).sum()),
            "by_fire": {
                fire: {
                    "assessed": int(analysis.fire.eq(fire).sum()),
                    "positive_F_star": int(
                        (analysis.fire.eq(fire) & analysis.exposed.eq(1)).sum()
                    ),
                }
                for fire in FIRES
            },
        },
        "definitions": {
            "assessed": "DINS outcome is no damage, partial damage or destroyed",
            "exposed": "F_destroyed_wmean > 0",
            "perimeter_eligibility": False,
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "geopandas": gpd.__version__,
            "numpy": np.__version__,
        },
        "inputs": {
            name: {
                "path": str(path.relative_to(data_dir.parent)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in input_paths.items()
        },
    }
    (results_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "analysis": analysis,
        "population_flow": flow,
        "dins_linkage_distance": linkage,
        "analysis_outcomes": outcome,
        "analysis_exposure": exposure,
        "height_imputation": height,
        "mesh_population": mesh,
        "manifest": manifest,
    }

