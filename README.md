# Structure-to-Structure Connectivity and Urban Wildfire Risk

This repository reproduces the analyses, tables and figures reported in the
manuscript from frozen post-ray-tracing building and coupling products for the
2025 Eaton and Palisades fires and the California statewide extension.

The reproduction begins after LiDAR processing, three-dimensional scene
construction, surface sampling and ray tracing. Those computationally intensive
upstream steps and their proprietary source data are outside the scope of this
package.

## Repository contents

- `s2s_nc/notebooks/` contains the seven canonical analysis notebooks.
- `src/analysis/` contains reusable statistical and network-analysis code.
- `src/viz/` contains the figure-building code.
- `scripts/` contains data validation and notebook-execution utilities.
- `data/` is populated from the separately distributed Dropbox bundle.
- `results/` and `figures/` are generated locally and are not tracked by Git.

## Installation

The publication analyses were developed with Python 3.10. Create the pinned
environment with:

```bash
conda env create -f environment.yml
conda activate s2s-fire-reproduction
python -m ipykernel install --user --name s2s-fire-reproduction \
  --display-name "S2S fire reproduction"
```

## Obtain the data

The frozen inputs are distributed separately through the
[S2S reviewer data folder on Dropbox](https://www.dropbox.com/scl/fo/yvqj2aku2f2qhld70dyvh/AIZWOsbXFX0_8aAU-8MsElo?rlkey=g5ng7brlr1do807ti17e676q5&dl=0).

Download and extract the bundle into `data/`, preserving its directory
structure. The download can also be initiated from the command line:

```bash
python scripts/download_data.py
python scripts/validate_code_release.py
python scripts/validate_s2s_data.py
```

The archive checksum will be added to the release record when the final data
bundle is frozen. Until then, `download_data.py` prints a warning when run
without an expected SHA-256 digest.

Large data may be stored outside the repository:

```bash
export S2S_DATA_DIR=/path/to/s2s_nc_data
python scripts/validate_s2s_data.py
```

Individual source groups can also be overridden with `S2S_REFERENCE_DIR`,
`OPENVIEW_3D_BVF_DIR`, `OPENVIEW_3D_PATCH_DIR`,
`OPENVIEW_FIRE_2D_SEG2M_DIR`, and `OPENVIEW_CA_2D_BVF_DIR`.

## Notebook order

1. `s2s_nc/notebooks/00_population_sample.ipynb`
2. `s2s_nc/notebooks/01_coupling_and_fragility.ipynb`
3. `s2s_nc/notebooks/02_construction_materials.ipynb`
4. `s2s_nc/notebooks/03_defense.ipynb`
5. `s2s_nc/notebooks/04_sen.ipynb`
6. `s2s_nc/notebooks/05_rsen.ipynb`
7. `s2s_nc/notebooks/06_carsen.ipynb`

Run all notebooks in publication order with:

```bash
python scripts/run_s2s_nc.py
```

For a faster installation check using fewer bootstrap and permutation draws:

```bash
python scripts/run_s2s_nc.py --smoke
```

Individual portions of the workflow can be selected with `--start` and
`--stop`, using notebook indices 0 through 6.

## Reproduction boundary

The distributed inputs retain the building identifiers, geometries, exposure
metrics and building-pair couplings required by the notebooks. The package does
not distribute the original LARIAC source products or full patch-to-patch line-
of-sight tables. Small figure-specific extracts are included where a publication
figure requires patch-level geometry.

All reported analyses should run without machine-specific paths or access to
the author's local OpenView directories. Environment variables are optional
overrides rather than requirements.

## Outputs

Generated tables are written under `results/`; figures are written under
`figures/`. A small number of reusable spatial caches and figure-source tables
are written under `data/derived/` beside their frozen inputs. All three output
locations are ignored by Git so that the repository contains only code,
documentation and lightweight metadata.

## Citation and license

Citation metadata are provided in `CITATION.cff`. See `LICENSE` for reuse terms.
