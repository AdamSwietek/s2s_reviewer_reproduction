"""Build the post-LOS inputs distributed with the reproduction package.

The default reviewer workflow begins after patch-level line-of-sight simulation.
This script copies frozen building-level products and reduces each multi-GB LOS
table to one row per directed visible building pair.  It is a release-building
utility and is not needed by reviewers once the packaged inputs exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import duckdb
import pandas as pd
import pyarrow.parquet as pq


FIRES = ("EATON", "PALISADES")


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build_directed_pairs(source: Path, target: Path) -> None:
    """Aggregate visible patches to directed, single-neighbour coupling.

    ``geom_coupling_wmean`` is the contribution that one emitting building
    makes to the receiver's whole-surface exposure: patch contributions from
    that emitter are summed at each receiver viewpoint, weighted by receiver
    patch area, and divided by the total sampled receiver area.  Summation
    across different neighbouring buildings is deliberately deferred; the
    SEN therefore asks whether any one neighbour can cross the empirical
    fragility threshold on its own.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    source_sql = str(source).replace("'", "''")
    target_sql = str(target).replace("'", "''")
    con.execute(f"""
        COPY (
            WITH valid AS (
                SELECT
                    CAST(source_BLD_ID AS VARCHAR) AS source_BLD_ID,
                    CAST(target_BLD_ID AS VARCHAR) AS target_BLD_ID,
                    source_uuid,
                    CAST(dist AS DOUBLE) AS dist,
                    CAST(A_recv AS DOUBLE) AS A_recv,
                    LEAST(GREATEST(
                        COALESCE(CAST(geom_exposure_ij AS DOUBLE), 0.0), 0.0
                    ), 1.0) AS coupling
                FROM read_parquet('{source_sql}')
                WHERE COALESCE(CAST(vis AS BOOLEAN), FALSE)
                  AND CAST(source_BLD_ID AS VARCHAR) <> 'None'
                  AND CAST(target_BLD_ID AS VARCHAR) <> 'None'
                  AND CAST(source_BLD_ID AS VARCHAR)
                      <> CAST(target_BLD_ID AS VARCHAR)
                  AND CAST(dist AS DOUBLE) > 0
                  AND CAST(A_recv AS DOUBLE) > 0
            ), receiver_viewpoints AS (
                SELECT source_BLD_ID, source_uuid,
                       ANY_VALUE(A_recv) AS A_recv
                FROM valid
                GROUP BY 1, 2
            ), receiver_area AS (
                SELECT source_BLD_ID, SUM(A_recv) AS receiver_area
                FROM receiver_viewpoints
                GROUP BY 1
            ), directed_pair AS (
                SELECT
                    source_BLD_ID,
                    target_BLD_ID,
                    MIN(dist) AS ssd_m,
                    SUM(coupling) AS geom_coupling_sum,
                    MAX(coupling) AS geom_coupling_max,
                    MAX(CASE WHEN dist <= 100.0 THEN coupling ELSE NULL END)
                        AS geom_coupling_max_100m,
                    SUM(coupling * A_recv) AS coupling_area_numerator,
                    COUNT(*)::BIGINT AS visible_patch_pairs
                FROM valid
                GROUP BY 1, 2
            )
            SELECT p.source_BLD_ID, p.target_BLD_ID, p.ssd_m,
                   p.geom_coupling_sum, p.geom_coupling_max,
                   p.geom_coupling_max_100m,
                   p.coupling_area_numerator
                       / NULLIF(a.receiver_area, 0)
                       AS geom_coupling_wmean,
                   p.visible_patch_pairs
            FROM directed_pair p
            INNER JOIN receiver_area a USING (source_BLD_ID)
        ) TO '{target_sql}' (
            FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000
        )
    """)
    con.close()


def parquet_record(path: Path, package_root: Path, description: str,
                   generated_by: str, restrictions: str = "none") -> dict:
    metadata = pq.ParquetFile(path).metadata
    return {
        "path": str(path.relative_to(package_root)),
        "format": "parquet",
        "description": description,
        "rows": metadata.num_rows,
        "columns": metadata.num_columns,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "generated_by": generated_by,
        "restrictions": restrictions,
    }


def generic_record(path: Path, package_root: Path, description: str,
                   generated_by: str, restrictions: str = "none") -> dict:
    """Manifest record for non-Parquet immutable inputs."""
    rows = columns = None
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        rows, columns = frame.shape
    return {
        "path": str(path.relative_to(package_root)),
        "format": path.suffix.lower().lstrip("."),
        "description": description,
        "rows": rows,
        "columns": columns,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "generated_by": generated_by,
        "restrictions": restrictions,
    }


def write_manifest(package_root: Path) -> pd.DataFrame:
    """Rebuild data/manifest.csv from the immutable inputs on disk."""
    destination = package_root / "data"
    manifest_rows = []
    descriptions = {
        "analysis.parquet": "Building outcomes and analysis attributes",
        "radex.parquet": "Building-level geometric exposure and arrival covariates",
        "buildings_enriched.parquet": "Building outcomes and reconstructed arrival fields",
        "arrival/eaton_progression_sensitivity.parquet": "Eaton linear, kriging-update and direct-kriging arrival clocks with direct-map up-fire exposure",
        "arrival/firms_viirs_eaton_palisades_jan2025.parquet": "NASA FIRMS VIIRS active-fire detections for the January 2025 fires",
        "enrichment/dins.parquet": "Linked DINS attributes and documented defense",
        "enrichment/vegetation.parquet": "Pre-fire vegetation attributes",
        "pairs/EATON_directed_pairs.parquet": "Post-LOS directed Eaton building-pair coupling",
        "pairs/PALISADES_directed_pairs.parquet": "Post-LOS directed Palisades building-pair coupling",
    }
    immutable = [
        path for path in destination.rglob("*")
        if path.is_file()
        and "derived" not in path.relative_to(destination).parts
        and path.suffix.lower() in {".parquet", ".csv", ".gpkg", ".tif", ".png", ".pdf"}
        and path.name not in {"manifest.csv", "manifest_summary.json", "data_dictionary.csv"}
        and not path.name.startswith(".")
    ]
    for path in sorted(immutable):
        relative_data = str(path.relative_to(destination))
        restriction = (
            "derived from access-controlled LARIAC 6; public redistribution pending licence review"
            if any(term in relative_data.lower() for term in
                   ("building", "analysis.parquet", "radex.parquet", "pairs/"))
            else "see docs/data_provenance.md"
        )
        kwargs = (
            path, package_root,
            descriptions.get(relative_data, "Packaged spatial or regional analysis input"),
            "scripts/build_package_inputs.py", restriction,
        )
        record = (parquet_record(*kwargs) if path.suffix.lower() == ".parquet"
                  else generic_record(*kwargs))
        manifest_rows.append(record)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(destination / "manifest.csv", index=False)
    (destination / "manifest_summary.json").write_text(json.dumps({
        "files": len(manifest),
        "bytes": int(manifest.bytes.sum()),
        "rows": int(manifest.rows.fillna(0).sum()),
    }, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Development project root containing data/",
    )
    parser.add_argument(
        "--package-root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--skip-pairs", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    package_root = args.package_root.resolve()
    destination = package_root / "data"

    copies = {
        source_root / "data/analysis.parquet": destination / "analysis.parquet",
        source_root / "data/radex.parquet": destination / "radex.parquet",
        source_root / "data/buildings_enriched.parquet": destination / "buildings_enriched.parquet",
        source_root / "data/enrichment/dins.parquet": destination / "enrichment/dins.parquet",
        source_root / "data/enrichment/vegetation.parquet": destination / "enrichment/vegetation.parquet",
        source_root / "data/nx/EATON_buildings.parquet": destination / "nx/EATON_buildings.parquet",
        source_root / "data/nx/PALISADES_buildings.parquet": destination / "nx/PALISADES_buildings.parquet",
        source_root / "data/nx/fire_perims.parquet": destination / "nx/fire_perims.parquet",
        source_root / "data/calfire_wui_la.gpkg": destination / "calfire_wui_la.gpkg",
        source_root / "data/arrival/arrival_interp.parquet": destination / "arrival/arrival_interp.parquet",
        source_root / "data/arrival/eaton_arrival_posterior_10m.tif": destination / "arrival/eaton_arrival_posterior_10m.tif",
        source_root / "data/arrival/isochrones_eaton.gpkg": destination / "arrival/isochrones_eaton.gpkg",
        source_root / "data/arrival/isochrones_palisades.gpkg": destination / "arrival/isochrones_palisades.gpkg",
        source_root / "data/firms_viirs_eaton_palisades_jan2025.parquet": destination / "arrival/firms_viirs_eaton_palisades_jan2025.parquet",
        source_root / "data/arrival/eaton_fire_events_recovered.parquet": destination / "arrival/eaton_fire_events_recovered.parquet",
        source_root / "data/arrival/eaton_progression_sensitivity.parquet": destination / "arrival/eaton_progression_sensitivity.parquet",
        source_root / "data/nx/samo_silverlake/corridor_buildings.parquet": destination / "nx/samo_silverlake/corridor_buildings.parquet",
        source_root / "data/nx/samo_silverlake/corridor_geom_pairs.parquet": destination / "nx/samo_silverlake/corridor_geom_pairs.parquet",
        source_root / "data/nx/samo_silverlake/corridor.parquet": destination / "nx/samo_silverlake/corridor.parquet",
        source_root / "data/nx/samo_silverlake/interface_boundary.parquet": destination / "nx/samo_silverlake/interface_boundary.parquet",
        source_root / "data/nx/samo_silverlake/tile_manifest.csv": destination / "nx/samo_silverlake/tile_manifest.csv",
    }
    for source, target in copies.items():
        if not source.exists():
            raise FileNotFoundError(source)
        copy_file(source, target)
        print("copied", target.relative_to(package_root))

    if not args.skip_pairs:
        for fire in FIRES:
            source = source_root / "data/nx" / f"{fire}_los_features.parquet"
            target = destination / "pairs" / f"{fire}_directed_pairs.parquet"
            print("aggregating", fire, "from", source.name)
            build_directed_pairs(source, target)
            print("created", target.relative_to(package_root))

    manifest = write_manifest(package_root)
    print("wrote", (destination / "manifest.csv").relative_to(package_root))


if __name__ == "__main__":
    main()
