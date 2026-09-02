"""Rebuild and render the three-stage Eaton fire-progression figure."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.arrival_reconstruction import build_arrival_reconstruction
from src.viz.arrival_reconstruction import make_eaton_arrival_steps_figure


def main() -> None:
    results = ROOT / "results"
    figures = ROOT / "figures" / "extended_data"
    source = ROOT / "data" / "derived"
    build_arrival_reconstruction(ROOT, results)
    output = make_eaton_arrival_steps_figure(ROOT, figures, source)
    print(f"saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
