"""Restore the frozen Figure 1 conceptual composite into the output folder.

The conceptual three-dimensional panel depends on licensed meshes and the
patch-level LOS scene, which are outside the default reviewer package. Notebook
01 regenerates all quantitative panels and calls this helper only for the
frozen graphical asset.
"""
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data" / "figure_assets"
OUTPUT = ROOT / "figures" / "main"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        source = ASSETS / f"Fig1_geometric_coupling_and_fragility{suffix}"
        target = OUTPUT / source.name
        shutil.copy2(source, target)
        print("restored", target.relative_to(ROOT))


if __name__ == "__main__":
    main()
