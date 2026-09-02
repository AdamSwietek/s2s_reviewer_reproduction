# Structure-to-Structure Coupling and Urban Conflagration Risk

Post-line-of-sight reproduction package 
manuscript by Adam Swietek. The package regenerates the reported statistical
analyses, tables and figures from frozen building-level and directed
building-pair products for the 2025 Eaton and Palisades fires.

## Scope

The standard workflow starts **after** the high-compute reconstruction of
terrain/building meshes, surface patches and patch-level lines of sight. It
does not require LARIAC source files, proprietary meshes or the original
2.8-GB patch-level LOS tables. The separately distributed compact pair tables retain one
row per directed visible building pair and reproduce the defense and SEN
analyses exactly. See [`docs/full_los_reconstruction.md`](docs/full_los_reconstruction.md)
for the upstream workflow and [`docs/data_provenance.md`](docs/data_provenance.md)
for access and redistribution restrictions.

## Install

Publication analyses were developed with Python 3.10.18. The clean-room test
used Python 3.10.20 on macOS; a recent Linux system should also work.

```bash
conda env create -f environment.yml
conda activate s2s-fire-reproduction
python scripts/validate_package.py
```

## Obtain the data

The frozen post-LOS inputs are hosted separately in the
[reviewer data folder on Dropbox](https://www.dropbox.com/scl/fo/yvqj2aku2f2qhld70dyvh/AIZWOsbXFX0_8aAU-8MsElo?rlkey=g5ng7brlr1do807ti17e676q5&dl=0)
and are not stored in this GitHub repository. Download the folder contents and
place them under `data/`, preserving the supplied directory structure. A
correct installation will contain `data/analysis.parquet`,
`data/radex.parquet` and `data/pairs/`. Then run
`python scripts/validate_package.py` to verify file checksums and schemas.

The clean environment installed in approximately 16 min on the test workstation
(including downloads). The immutable packaged inputs occupy about 169 MiB. The current generated
intermediates, results and figures add about 62 MiB. Allow at least 4 GB RAM and
1 GB free working space. No GPU is required. Installation time depends on the
Conda solver and download speed.

## Run

Fast integrity check plus the population notebook:

```bash
python scripts/run_all.py --smoke
```

Main-text notebooks only:

```bash
python scripts/run_all.py --main-only
```

Complete publication workflow, including Extended Data:

```bash
python scripts/run_all.py
```

The notebooks are executed in place. Tables are written to `results/`, main
figures to `figures/main/`, Extended Data figures to
`figures/extended_data/`, and intermediate generated products to
`data/derived/`. Publication settings and fixed seeds are recorded in the
notebooks and summarized in `config.example.yaml`.

The measured clean-room smoke test took 0.1 min. Before the expanded
progression-map sensitivity, the complete 11-notebook workflow took 5.2 min on
the test workstation; the expanded ED04 notebook alone currently takes about
18 min at the publication bootstrap settings. A new complete-workflow timing is
still to be recorded. Runtime will vary with processor and storage; the LOS
simulation is not part of these timings.

## Notebook order

1. `00_population_and_data_audit.ipynb`
2. `01_geometric_coupling_and_fragility.ipynb`
3. `02_construction_attributes.ipynb`
4. `03_defensive_actions.ipynb`
5. `04_structure_exposure_networks.ipynb`
6. `05_regional_sen_extent.ipynb`
7. `extended_data/ED01_population_scene_and_arrival.ipynb`
8. `extended_data/ED02_fragility_sensitivity.ipynb`
9. `extended_data/ED03_construction_sensitivity.ipynb`
10. `extended_data/ED04_defense_sensitivity.ipynb`
11. `extended_data/ED05_sen_sensitivity.ipynb`

## Headline checks

`scripts/validate_package.py --check-results` verifies, within numerical
tolerance:

- 28,208 assessed and 25,127 exposed structures;
- 14,595 exposed Eaton and 10,532 exposed Palisades structures;
- 1.505 Palisades:Eaton destruction F50 ratio;
- 1.293 tile-roof exposure-tolerance ratio;
- 13.54-percentage-point pooled defense survival difference;
- −13.40-percentage-point direction-adjusted neighbor contrast;
- 11.14–13.54-percentage-point pooled defense survival differences and
  −13.40 to −7.78-point directional neighbor contrasts across the three Eaton
  progression reconstructions; and
- 9,540 Eaton and 7,879 Palisades active SEN bonds.

## Directory guide

```text
data/                 immutable post-LOS inputs and generated intermediates
  pairs/              compact directed building-pair coupling tables
  derived/            notebook-generated products; safe to recreate
notebooks/             main analysis notebooks in manuscript order
notebooks/extended_data/
src/                   reusable analysis and visualization functions
scripts/               runner, validators and release-building utilities
results/               machine-readable estimates and publication tables
figures/               main and Extended Data figures
docs/                  provenance, workflow and manuscript crosswalk
```

Every immutable input is checksummed in `data/manifest.csv`; table schemas are
listed in `data/data_dictionary.csv`. Generated tables and figures are
checksummed in `results/output_manifest.csv`. `scripts/build_package_inputs.py` is a
release-building utility for the data custodian and is not needed by reviewers.

## Important interpretation

`F*` is realized geometric coupling from neighboring structures that were
observed destroyed. The resulting fragility models are retrospective
exposure–response analyses, not prospective forecasts. SEN bonds use the
single-neighbour whole-surface geometric coupling at the empirically selected
threshold. Contributions from different neighbours are not accumulated, so
the SEN is a conservative weakest-link representation.

## Troubleshooting

- Run commands from the repository root so relative paths resolve correctly.
- Create the supplied Conda environment rather than mixing system geospatial
  libraries with pip wheels.
- If Matplotlib cannot write its cache, set `MPLCONFIGDIR` to a writable
  directory.
- Jupyter must be permitted to open local loopback ports for notebook kernels.
- `data/manifest.csv` identifies source-derived files whose public release
  remains subject to LARIAC/CAL FIRE terms; do not post those files publicly
  until the licence review is complete.

## Licence and citation

Code is released under the MIT License. Data retain their source terms and are
not relicensed by the code licence. Citation metadata are in `CITATION.cff`.
The archival DOI and final manuscript citation should replace the provisional
review metadata when the submission release is deposited.

Preparation and unresolved release actions are tracked in
[`PACKAGE_CHECKLIST.md`](PACKAGE_CHECKLIST.md).
