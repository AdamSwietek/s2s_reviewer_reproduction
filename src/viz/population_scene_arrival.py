"""Extended Data figure for population, scene and arrival reconstruction."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .style import FIRE_COLORS, apply_style


def make_ed01_figure(project_root: Path, results_dir: Path,
                     figure_dir: Path) -> Path:
    root, results, figures = map(Path, (project_root, results_dir, figure_dir))
    figures.mkdir(parents=True, exist_ok=True)
    apply_style()

    flow = pd.read_csv(results / "population_flow.csv")
    heights = pd.read_csv(results / "height_imputation.csv")
    heights = heights[heights.fire.ne("POOLED")].copy()
    cv = pd.read_csv(results / "ED01_eaton_arrival_cv.csv")
    pred = pd.read_csv(results / "ED01_eaton_arrival_cv_predictions.csv")
    base = pd.read_csv(results / "ED01_eaton_arrival_baseline.csv").iloc[0]
    chosen = cv[cv.selected.astype(str).str.lower().eq("true")].iloc[0]

    fig = plt.figure(figsize=(7.25, 6.6))
    grid = fig.add_gridspec(2, 2, height_ratios=(.78, 1.22),
                            width_ratios=(.92, 1.08), hspace=.34, wspace=.29)
    ax_flow = fig.add_subplot(grid[0, 0])
    ax_height = fig.add_subplot(grid[0, 1])
    ax_map = fig.add_subplot(grid[1, 0])
    ax_cv = fig.add_subplot(grid[1, 1])

    # a: linkage and analytical population
    labels = ["Source DINS", "Linked ≤50 m", "Unique buildings",
              "Assessed", "$F^*>0$"]
    counts = flow.buildings_or_records.to_numpy()
    y = np.arange(len(counts))[::-1]
    ax_flow.hlines(y, 0, counts, color="#D4D4D4", lw=2.2)
    ax_flow.scatter(counts, y, s=28, color="#333333", zorder=3)
    for yy, count in zip(y, counts):
        ax_flow.text(count + 420, yy, f"{count:,}", va="center", fontsize=7.2)
    ax_flow.set_yticks(y, labels)
    ax_flow.set_xlim(0, 34000)
    ax_flow.set_xlabel("Records or buildings")
    ax_flow.set_title("a  Population linkage and eligibility", loc="left",
                      fontweight="bold")
    ax_flow.grid(axis="x", color="#E5E5E5", lw=.45)

    # b: height provenance
    categories = ["lariac_height", "nearby_median", "one_storey_fallback"]
    colors = ["#404040", "#7EA6B8", "#D8A65C"]
    bottoms = np.zeros(len(heights))
    for category, color, label in zip(
        categories, colors,
        ["LARIAC height", "nearby median", "3-m fallback"],
    ):
        values = heights[category].to_numpy()
        ax_height.bar(heights.fire.str.title(), values, bottom=bottoms,
                      color=color, width=.62, label=label)
        bottoms += values
    for x, total in enumerate(heights.buildings):
        ax_height.text(x, total + 900, f"{total:,}", ha="center", fontsize=7.2)
    ax_height.set_ylim(0, 41500)
    ax_height.set_ylabel("Building footprints")
    ax_height.set_title("b  Scene height completion", loc="left",
                        fontweight="bold")
    ax_height.legend(loc="upper right", fontsize=6.7)
    ax_height.grid(axis="y", color="#E5E5E5", lw=.45)

    # c: frozen posterior surface and validation locations
    raster_path = root / "data/arrival/eaton_arrival_posterior_10m.tif"
    with rasterio.open(raster_path) as src:
        arrival = src.read(2, masked=True)
        extent = (src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top)
        raster_crs = src.crs
    vmax = float(np.quantile(arrival.compressed(), .98))
    im = ax_map.imshow(arrival, extent=extent, origin="upper", cmap="viridis",
                       norm=Normalize(0, vmax), rasterized=True)
    perimeter = (gpd.read_parquet(root / "data/nx/fire_perims.parquet")
                 .query("FIRE_NAME == 'EATON'").to_crs(raster_crs))
    perimeter.boundary.plot(ax=ax_map, color="white", lw=1.8, zorder=3)
    perimeter.boundary.plot(ax=ax_map, color="#222222", lw=.65, zorder=4)
    points = gpd.GeoDataFrame(
        pred, geometry=gpd.points_from_xy(pred.longitude, pred.latitude), crs=4326
    ).to_crs(raster_crs)
    ax_map.scatter(points.geometry.x, points.geometry.y, s=7, facecolor="white",
                   edgecolor="#111111", linewidth=.25, alpha=.8, zorder=5)
    ax_map.set_aspect("equal"); ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_map.set_title("c  Eaton posterior arrival surface", loc="left",
                     fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_map, orientation="horizontal", fraction=.045,
                        pad=.035, aspect=28)
    cbar.set_label("Hours since ignition", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.7)
    ax_map.text(.02, .025, f"{len(points):,} spatial-CV control locations",
                transform=ax_map.transAxes, fontsize=6.7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=2))

    # d: spatial block out-of-fold validation
    maximum = float(np.ceil(max(pred.observed_h.max(), pred.baseline_h.max(),
                                pred.predicted_h.max()) / 5) * 5)
    ax_cv.plot([0, maximum], [0, maximum], color="#222222", lw=.8, ls="--")
    ax_cv.scatter(pred.observed_h, pred.baseline_h, s=10, facecolor="none",
                  edgecolor="#9B9B9B", linewidth=.5, alpha=.62,
                  label="baseline drift")
    ax_cv.scatter(pred.observed_h, pred.predicted_h, s=11,
                  color=FIRE_COLORS["EATON"], alpha=.58, linewidth=0,
                  label="regression-kriging")
    ax_cv.set_xlim(0, maximum); ax_cv.set_ylim(-2, maximum)
    ax_cv.set_xlabel("Observed arrival (h since ignition)")
    ax_cv.set_ylabel("Out-of-fold predicted arrival (h)")
    ax_cv.set_title("d  Ten-fold spatial block validation", loc="left",
                    fontweight="bold")
    ax_cv.grid(color="#E2E2E2", lw=.45)
    ax_cv.legend(loc="upper left")
    ax_cv.text(
        .98, .04,
        f"MAE: {base.mae_h:.1f} → {chosen.mae_h:.1f} h\n"
        f"bias: {base.bias_h:+.1f} → {chosen.bias_h:+.1f} h\n"
        f"Spearman: {base.spearman:.2f} → {chosen.spearman:.2f}\n"
        f"nested 90% coverage: {100*chosen.nested_cv_90_coverage:.0f}%",
        transform=ax_cv.transAxes, ha="right", va="bottom", fontsize=7.1,
        bbox=dict(facecolor="white", edgecolor="#CFCFCF", lw=.45, alpha=.92,
                  boxstyle="round,pad=.28"),
    )

    fig.suptitle("Population, reconstructed scenes and Eaton arrival-time validation",
                 x=.07, y=.995, ha="left", fontsize=10.2, fontweight="bold")
    stem = figures / "ED_fig_population_scene_arrival"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")
