"""Validate the frozen inputs required by the seven S2S notebooks."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import (  # noqa: E402
    DATA_DIR, DERIVED_3D, OPENVIEW_3D_PATCH_RUN, OPENVIEW_3D_RUN,
    OPENVIEW_CA_2D_RUN, OPENVIEW_FIRE_2D_RUN, REFERENCE_DIR,
)


REQUIRED_PARQUET_COLUMNS = {
    DERIVED_3D / "analysis_openview3d.parquet": {
        "graph_id", "fire", "outcome", "is_destroyed", "exposed",
        "F_destroyed_wmean",
    },
    DERIVED_3D / "county_buildings.parquet": {"building_id", "geometry"},
    DERIVED_3D / "dins_openview3d.parquet": {"graph_id"},
    OPENVIEW_3D_RUN / "building_ids.parquet": {"building_id"},
    OPENVIEW_3D_RUN / "building_edges.parquet": set(),
    OPENVIEW_FIRE_2D_RUN / "building_ids.parquet": {"building_id"},
    OPENVIEW_FIRE_2D_RUN / "building_edges.parquet": set(),
    OPENVIEW_CA_2D_RUN / "building_ids.parquet": {"building_id"},
    OPENVIEW_CA_2D_RUN / "building_edges.parquet": set(),
    REFERENCE_DIR / "california_interface_intermix_influence_75m.parquet": {
        "WUI_DESC", "Shape",
    },
    DATA_DIR / "derived" / "carsen_2d_vs_3d" /
    "california_wui_all_3mi_osm_2d" / "california_fhsz_2024_2025.parquet": {
        "FHSZ_Description", "geometry",
    },
    DATA_DIR / "derived" / "carsen_2d_vs_3d" /
    "california_wui_all_3mi_osm_2d" /
    "california_building_fhsz_2024_2025.parquet": {
        "building_id", "severity",
    },
}

REQUIRED_FILES = (
    DATA_DIR / "calfire_wui_la.gpkg",
    DATA_DIR / "nx" / "fire_perims.parquet",
    DATA_DIR / "enrichment" / "vegetation.parquet",
    OPENVIEW_3D_RUN / "manifest.json",
    OPENVIEW_3D_PATCH_RUN / "manifest.json",
    OPENVIEW_3D_PATCH_RUN / "patch_exchange.parquet",
    OPENVIEW_FIRE_2D_RUN / "manifest.json",
    OPENVIEW_CA_2D_RUN / "manifest.json",
    REFERENCE_DIR / "cb_2025_us_state_500k.zip",
    REFERENCE_DIR / "la_county_mainland_boundary.parquet",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_ocean.shp",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_ocean.dbf",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_ocean.shx",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_ocean.prj",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_lakes.shp",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_lakes.dbf",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_lakes.shx",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_lakes.prj",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_rivers_lake_centerlines.shp",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_rivers_lake_centerlines.dbf",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_rivers_lake_centerlines.shx",
    REFERENCE_DIR / "natural_earth_hydro" / "ne_10m_rivers_lake_centerlines.prj",
)

REQUIRED_GLOBS = (
    (OPENVIEW_3D_PATCH_RUN / "patches", "*.parquet"),
    (OPENVIEW_CA_2D_RUN / "patches", "*.parquet"),
)


def digest(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data_manifest.csv",
        help="Optional file-level checksum manifest.",
    )
    parser.add_argument(
        "--checksums", action="store_true",
        help="Verify every checksum in the manifest; this can take several minutes.",
    )
    args = parser.parse_args()

    problems: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            problems.append(f"missing file: {path}")

    for directory, pattern in REQUIRED_GLOBS:
        if not directory.is_dir() or not next(directory.glob(pattern), None):
            problems.append(f"missing files: {directory / pattern}")

    for path, required in REQUIRED_PARQUET_COLUMNS.items():
        if not path.exists():
            problems.append(f"missing parquet: {path}")
            continue
        actual = set(pq.read_schema(path).names)
        missing = required - actual
        if missing:
            problems.append(f"{path}: missing columns {sorted(missing)}")

    if args.manifest.exists():
        manifest = pd.read_csv(args.manifest)
        for column in ("path", "bytes", "sha256"):
            if column not in manifest:
                problems.append(f"manifest missing column: {column}")
        if not problems:
            for row in manifest.itertuples(index=False):
                path = DATA_DIR / row.path
                if not path.exists():
                    problems.append(f"manifest file missing: {path}")
                    continue
                if path.stat().st_size != int(row.bytes):
                    problems.append(f"size mismatch: {path}")
                if args.checksums and digest(path) != row.sha256:
                    problems.append(f"checksum mismatch: {path}")
    else:
        print(f"Checksum manifest not yet present: {args.manifest}")

    if problems:
        print("S2S data validation failed:")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)
    print("S2S data validation passed.")


if __name__ == "__main__":
    main()
