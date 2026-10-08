"""Portable paths for the Structure-to-Structure reproduction package.

The repository contains the post-processing and statistical analyses. Large
post-ray-tracing inputs are distributed separately and default to ``data/``.
Set ``S2S_DATA_DIR`` when the downloaded data live elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("S2S_DATA_DIR", ROOT / "data")).expanduser().resolve()
REFERENCE_DIR = Path(
    os.environ.get("S2S_REFERENCE_DIR", DATA_DIR / "reference")
).expanduser().resolve()
RESULTS_DIR = Path(
    os.environ.get("S2S_RESULTS_DIR", ROOT / "results")
).expanduser().resolve()
FIGURES_DIR = Path(
    os.environ.get("S2S_FIGURES_DIR", ROOT / "figures")
).expanduser().resolve()

OPENVIEW_DIR = DATA_DIR / "openview_runs"
OPENVIEW_3D_RUN = Path(
    os.environ.get(
        "OPENVIEW_3D_BVF_DIR",
        OPENVIEW_DIR / "lariac6_wui_all_2mi_bvf_3d_nf_cone85_nosolar_pads3m",
    )
).expanduser().resolve()
OPENVIEW_3D_PATCH_RUN = Path(
    os.environ.get(
        "OPENVIEW_3D_PATCH_DIR",
        OPENVIEW_DIR / "fires_eaton_palisades_800m_bvf_3d_nf_cone85_nosolar_pads3m_patches",
    )
).expanduser().resolve()
OPENVIEW_FIRE_2D_RUN = Path(
    os.environ.get(
        "OPENVIEW_FIRE_2D_SEG2M_DIR",
        OPENVIEW_DIR / "fires_eaton_palisades_800m_bvf_2d_lariac_seg2m",
    )
).expanduser().resolve()
OPENVIEW_CA_2D_RUN = Path(
    os.environ.get(
        "OPENVIEW_CA_2D_BVF_DIR",
        OPENVIEW_DIR / "california_wui_all_3mi_osm_2d" / "full_california",
    )
).expanduser().resolve()

RUN_TAG_3D = "openview3d_lariac6_2mi"
STATE_RUN_TAG = "california_wui_all_3mi_osm_2d"
DERIVED_3D = DATA_DIR / "derived" / RUN_TAG_3D
DERIVED_CA = DATA_DIR / "derived" / "carsen_2d_vs_3d" / STATE_RUN_TAG


def ensure_output_directories() -> None:
    """Create generated-output directories without creating input folders."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
