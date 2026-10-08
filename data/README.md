# Data bundle

Large analysis inputs are distributed separately through Dropbox and are not
stored in Git. Download the bundle from the link in the repository README and
extract its contents into this directory, preserving the supplied paths.

The reproduction begins with frozen post-ray-tracing building and coupling
tables. It does not reproduce upstream LiDAR processing, mesh construction,
surface sampling or ray tracing.

Run the following after installing the environment:

```bash
python scripts/download_data.py
python scripts/validate_s2s_data.py
```

Set `S2S_DATA_DIR` if the bundle is stored outside the repository.

The extracted bundle has four principal sections:

- `derived/openview3d_lariac6_2mi/`: frozen county analysis and crosswalk tables;
- `derived/carsen_2d_vs_3d/`: frozen statewide classifications and calibration caches;
- `openview_runs/`: post-ray-tracing building and pair-coupling products;
- `reference/`, `enrichment/`, and `nx/`: WUI, hazard, hydrography, vegetation,
  fire-perimeter, and boundary layers.

The validator checks both file presence and required Parquet schemas. Use
`python scripts/validate_s2s_data.py --checksums` after the final
`data_manifest.csv` is published.

The maintainer can regenerate that manifest after freezing the Dropbox bundle
with `python scripts/build_data_manifest.py --data-dir /path/to/bundle`.
