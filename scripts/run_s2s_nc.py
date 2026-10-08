"""Execute the seven canonical S2S notebooks in manuscript order."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "s2s_nc" / "notebooks" / name
    for name in (
        "00_population_sample.ipynb",
        "01_coupling_and_fragility.ipynb",
        "02_construction_materials.ipynb",
        "03_defense.ipynb",
        "04_sen.ipynb",
        "05_rsen.ipynb",
        "06_carsen.ipynb",
    )
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", type=int, choices=range(7), default=0,
        help="First notebook index to execute (default: 0).",
    )
    parser.add_argument(
        "--stop", type=int, choices=range(7), default=6,
        help="Last notebook index to execute (default: 6).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use reduced bootstrap and permutation counts for an installation test.",
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--kernel",
        default=os.environ.get("S2S_KERNEL", "s2s-fire-reproduction"),
        help="Jupyter kernel used to execute notebooks.",
    )
    args = parser.parse_args()
    if args.start > args.stop:
        parser.error("--start must not exceed --stop")

    for validator in ("validate_code_release.py", "validate_s2s_data.py"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / validator)],
            cwd=ROOT, check=True,
        )
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    if args.smoke:
        env.update({
            "OPENVIEW_FRAGILITY_BOOTSTRAPS": "20",
            "OPENVIEW_CONSTRUCTION_BOOTSTRAPS": "20",
            "OPENVIEW_DEFENSE_BOOTSTRAPS": "100",
            "OPENVIEW_DEFENSE_SENSITIVITY_BOOTSTRAPS": "100",
            "OPENVIEW_SEN_SIZE_SHUFFLES": "100",
            "OPENVIEW_SEN_FATE_SHUFFLES": "50",
        })

    selected = NOTEBOOKS[args.start:args.stop + 1]
    for index, notebook in enumerate(selected, args.start):
        print(f"[{index}/6] {notebook.relative_to(ROOT)}", flush=True)
        subprocess.run([
            sys.executable, "-m", "nbconvert",
            "--to", "notebook", "--execute", "--inplace", str(notebook),
            f"--ExecutePreprocessor.timeout={args.timeout}",
            f"--ExecutePreprocessor.kernel_name={args.kernel}",
        ], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
