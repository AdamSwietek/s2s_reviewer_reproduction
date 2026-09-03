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
analyses exactly.

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


```
