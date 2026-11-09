"""Assemble the minimal S2S Dropbox data bundle from local working products."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENVIEW = ROOT.parent.parent / "openview" / "results"
RUN_3D_NAME = "lariac6_wui_all_2mi_bvf_3d_nf_cone85_nosolar_pads3m"
PATCH_RUN_NAME = "fires_eaton_palisades_800m_bvf_3d_nf_cone85_nosolar_pads3m_patches"
FIRE_2D_NAME = "fires_eaton_palisades_800m_bvf_2d_lariac_seg2m"
STATE_RUN_RELATIVE = Path("california_wui_all_3mi_osm_2d") / "full_california"
DERIVED_3D = Path("derived") / "openview3d_lariac6_2mi"
DERIVED_CA = (
    Path("derived") / "carsen_2d_vs_3d" / "california_wui_all_3mi_osm_2d"
)


def transfer(source: Path, destination: Path) -> None:
    """Hard-link a file when possible, otherwise copy it with metadata."""
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def transfer_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.name != ".DS_Store":
            transfer(path, destination / path.relative_to(source))


def selected_files(source: Path, destination: Path, relative_paths) -> None:
    for relative in relative_paths:
        transfer(source / relative, destination / relative)


def build_patch_extract(source: Path, destination: Path) -> None:
    """Keep only the four observed buildings used by the Figure 2 scene."""
    selected = (2_133_192, 1_989_387, 1_989_374, 2_050_140)
    transfer(source / "manifest.json", destination / "manifest.json")
    destination.mkdir(parents=True, exist_ok=True)
    patch_output = destination / "patches" / "figure2_buildings.parquet"
    exchange_output = destination / "patch_exchange.parquet"
    if patch_output.exists() and exchange_output.exists():
        return
    patch_output.parent.mkdir(parents=True, exist_ok=True)
    id_sql = ", ".join(map(str, selected))
    patch_glob = str(source / "patches" / "*.parquet").replace("'", "''")
    exchange_source = str(source / "patch_exchange.parquet").replace("'", "''")
    patch_destination = str(patch_output).replace("'", "''")
    exchange_destination = str(exchange_output).replace("'", "''")
    with duckdb.connect() as connection:
        connection.execute(f"""
            COPY (
                SELECT * FROM read_parquet('{patch_glob}', union_by_name=true)
                WHERE building_id IN ({id_sql})
            ) TO '{patch_destination}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        connection.execute(f"""
            COPY (
                SELECT * FROM read_parquet('{exchange_source}')
                WHERE building_p IN ({id_sql}) AND building_q IN ({id_sql})
                  AND (building_p = {selected[0]} OR building_q = {selected[0]})
            ) TO '{exchange_destination}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, default=ROOT / "data_copy")
    parser.add_argument("--destination", type=Path, default=ROOT / "data")
    parser.add_argument("--openview-results", type=Path, default=DEFAULT_OPENVIEW)
    parser.add_argument(
        "--reference", type=Path,
        default=ROOT.parent / "data" / "reference",
    )
    args = parser.parse_args()
    source = args.source_data.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    openview = args.openview_results.expanduser().resolve()
    reference = args.reference.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"Destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    selected_files(source, destination, (
        "README.md", "analysis.parquet", "buildings_enriched.parquet",
        "calfire_wui_la.gpkg", "radex.parquet",
        "enrichment/dins.parquet", "enrichment/vegetation.parquet",
        "nx/EATON_buildings.parquet", "nx/PALISADES_buildings.parquet",
        "nx/fire_perims.parquet",
    ))
    transfer_tree(
        source / "derived" / "openview2d_eaton_scene",
        destination / "derived" / "openview2d_eaton_scene",
    )

    county_files = (
        "analysis_openview3d.parquet", "county_buildings.parquet",
        "dins_openview3d.parquet", "distance_comparison_openview3d.parquet",
        "defense_openview3d_directed_edges.parquet",
        "county_building_centroids.parquet",
        "county_sen_distance30ft_assignments.parquet",
        "fire_building_dose_2d_seg2m.parquet",
        "county_sen_top2_map_source.parquet",
        "county_sen_top2_map_source_components.parquet",
        "county_sen_top2_map_source_summary.parquet",
        "county_sen_eeat_map_source.parquet",
        "county_sen_eeat_map_source_components.parquet",
        "county_sen_eeat_map_source_summary.parquet",
        "county_sen_top2_alpha075_map_source.parquet",
        "county_sen_top2_alpha075_map_source_components.parquet",
        "county_sen_top2_alpha075_map_source_summary.parquet",
    )
    selected_files(source / DERIVED_3D, destination / DERIVED_3D, county_files)
    transfer_tree(
        source / DERIVED_3D / "defense_design_cache_openview3d",
        destination / DERIVED_3D / "defense_design_cache_openview3d",
    )

    state_files = (
        "state_2d_buildings_in_county_3d_extent.parquet",
        "county_3d_to_state_2d_building_crosswalk.parquet",
        "county_matched_pair_F_2d_vs_3d.parquet",
        "california_building_centroids_2d.parquet",
        "california_building_wui_class.parquet",
        "california_fhsz_2024_2025.parquet",
        "california_building_fhsz_2024_2025.parquet",
        "statewide_components/california_top2_ranked_incidence.parquet",
    )
    selected_files(source / DERIVED_CA, destination / DERIVED_CA, state_files)
    transfer(
        source / "derived" / "carsen_2d_vs_3d" / "fire_building_dose_2d.parquet",
        destination / "derived" / "carsen_2d_vs_3d" / "fire_building_dose_2d.parquet",
    )
    component_roots = (
        "absolute_p50_ctop2_alpha075",
        "threshold_comparison/local_threshold_region_equivalent",
        "threshold_comparison/midpoint_average_robustness",
    )
    component_files = (
        "california_top2_active_edges.parquet",
        "california_top2_assignments.parquet",
        "california_top2_component_size_frequency.csv",
        "california_top2_summary.csv",
    )
    for component_root in component_roots:
        selected_files(
            source / DERIVED_CA / "statewide_components" / component_root,
            destination / DERIVED_CA / "statewide_components" / component_root,
            component_files,
        )
    transfer(
        source / DERIVED_CA / "statewide_components" /
        "threshold_comparison/midpoint_average_robustness/spatial" /
        "california_top2_component_spatial_summary.parquet",
        destination / DERIVED_CA / "statewide_components" /
        "threshold_comparison/midpoint_average_robustness/spatial" /
        "california_top2_component_spatial_summary.parquet",
    )

    run_destination = destination / "openview_runs"
    selected_files(openview / RUN_3D_NAME, run_destination / RUN_3D_NAME, (
        "manifest.json", "building_ids.parquet", "building_edges.parquet",
        "building_metrics.parquet",
    ))
    selected_files(openview / FIRE_2D_NAME, run_destination / FIRE_2D_NAME, (
        "manifest.json", "building_ids.parquet", "building_edges.parquet",
    ))
    selected_files(
        openview / STATE_RUN_RELATIVE,
        run_destination / STATE_RUN_RELATIVE,
        ("manifest.json", "building_ids.parquet", "building_edges.parquet"),
    )
    build_patch_extract(
        openview / PATCH_RUN_NAME, run_destination / PATCH_RUN_NAME,
    )

    selected_files(reference, destination / "reference", (
        "cb_2025_us_state_500k.zip",
        "california_interface_intermix_influence_75m.parquet",
        "la_county_mainland_boundary.parquet",
    ))
    for stem in (
        "ne_10m_ocean", "ne_10m_lakes", "ne_10m_rivers_lake_centerlines",
    ):
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            transfer(
                reference / "natural_earth_hydro" / f"{stem}{suffix}",
                destination / "reference" / "natural_earth_hydro" /
                f"{stem}{suffix}",
            )

    total = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    print(f"Built {destination} ({total / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()
