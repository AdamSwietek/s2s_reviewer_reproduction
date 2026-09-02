"""Run the post-LOS reproduction workflow in manuscript order."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import nbformat


ROOT = Path(__file__).resolve().parents[1]
MAIN = [
    "notebooks/00_population_and_data_audit.ipynb",
    "notebooks/01_geometric_coupling_and_fragility.ipynb",
    "notebooks/02_construction_attributes.ipynb",
    "notebooks/03_defensive_actions.ipynb",
    "notebooks/04_structure_exposure_networks.ipynb",
    "notebooks/05_regional_sen_extent.ipynb",
]
EXTENDED = [
    "notebooks/extended_data/ED01_population_scene_and_arrival.ipynb",
    "notebooks/extended_data/ED02_fragility_sensitivity.ipynb",
    "notebooks/extended_data/ED03_construction_sensitivity.ipynb",
    "notebooks/extended_data/ED04_defense_sensitivity.ipynb",
    "notebooks/extended_data/ED05_sen_sensitivity.ipynb",
]


def sanitize_notebook(path: Path) -> None:
    """Remove machine-specific absolute paths from captured cell output."""
    notebook = nbformat.read(path, as_version=4)
    serialized = nbformat.writes(notebook)
    serialized = serialized.replace(str(ROOT), ".")
    serialized = serialized.replace(sys.prefix, "<CONDA_ENV>")
    nbformat.write(nbformat.reads(serialized, as_version=4), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run schema/headline checks and notebook 00 only.",
    )
    parser.add_argument(
        "--main-only", action="store_true",
        help="Run notebooks 00–05 but omit Extended Data notebooks.",
    )
    args = parser.parse_args()

    subprocess.run(
        [sys.executable, "scripts/validate_package.py"], cwd=ROOT, check=True)
    notebooks = [MAIN[0]] if args.smoke else MAIN + ([] if args.main_only else EXTENDED)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    kernel_name = env.get("S2S_KERNEL", "python3")
    jupyter = Path(sys.executable).with_name("jupyter")
    if not jupyter.exists():
        raise FileNotFoundError(f"Jupyter launcher not found beside Python: {jupyter}")
    started = time.perf_counter()
    for index, notebook in enumerate(notebooks, 1):
        print(f"[{index}/{len(notebooks)}] {notebook}", flush=True)
        subprocess.run([
            str(jupyter), "nbconvert",
            "--to", "notebook", "--execute", "--inplace", notebook,
            f"--ExecutePreprocessor.kernel_name={kernel_name}",
            "--ExecutePreprocessor.timeout=1800",
        ], cwd=ROOT, env=env, check=True)
        sanitize_notebook(ROOT / notebook)
    elapsed = (time.perf_counter() - started) / 60
    subprocess.run(
        [sys.executable, "scripts/validate_package.py", "--check-results"],
        cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, "scripts/build_output_manifest.py"],
        cwd=ROOT, check=True)
    print(f"Completed {len(notebooks)} notebook(s) in {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
