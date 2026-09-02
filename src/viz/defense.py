"""Publication graphics for documented structure defense."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.transforms import Affine2D

from src.viz.style import apply_style


NAVY = "#2E5B82"
VERMILLION = "#B95038"
INK = "#222222"
MID = "#777777"
PALE = "#F1F1EE"
PALE_RED = "#F3E4DF"


def plot_defense_locations(project_root: Path, matched, output_dir: Path):
    """Map all documented defense targets and the matched analytical subset."""
    apply_style()
    project_root = Path(project_root)
    data = project_root / "data"
    dins = pd.read_parquet(
        data / "enrichment" / "dins.parquet",
        columns=["BLD_ID", "is_defended"],
    )
    defended_ids = set(
        dins.loc[dins.is_defended.eq(True), "BLD_ID"].astype(str)
    )
    matched_ids = set(
        matched.loc[matched.defended.eq(1), "BLD_ID"].astype(str)
    )
    perimeters = gpd.read_parquet(data / "nx" / "fire_perims.parquet")
    perimeters = perimeters.set_index("FIRE_NAME")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55))
    summaries = []
    for panel, (ax, fire) in enumerate(zip(axes, ["EATON", "PALISADES"])):
        buildings = gpd.read_parquet(
            data / "nx" / f"{fire}_buildings.parquet",
            columns=["BLD_ID", "geometry"],
        )
        buildings["BLD_ID"] = buildings.BLD_ID.astype(str)
        is_defended = buildings.BLD_ID.isin(defended_ids)
        is_matched = buildings.BLD_ID.isin(matched_ids)
        background = buildings[~is_defended]
        unmatched_targets = buildings[is_defended & ~is_matched]
        matched_targets = buildings[is_matched]

        background.plot(ax=ax, color="#DDDDDA", edgecolor="none",
                        rasterized=True, zorder=1)
        if len(unmatched_targets):
            centers = unmatched_targets.geometry.centroid
            ax.scatter(centers.x, centers.y, s=5.0, marker="o",
                       facecolors="none", edgecolors="#D8846B",
                       linewidths=.45, alpha=.95, rasterized=True, zorder=2)
        if len(matched_targets):
            centers = matched_targets.geometry.centroid
            ax.scatter(centers.x, centers.y, s=5.0, marker="o",
                       color=VERMILLION, edgecolors="white", linewidths=.15,
                       alpha=.95, rasterized=True, zorder=3)
        perimeter = perimeters.loc[fire].geometry.boundary
        for part in getattr(perimeter, "geoms", [perimeter]):
            x, y = part.xy
            ax.plot(x, y, color=INK, lw=.8, zorder=4)

        xmin, ymin, xmax, ymax = buildings.total_bounds
        pad_x, pad_y = .015 * (xmax - xmin), .015 * (ymax - ymin)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(
            f"{chr(97 + panel)}  {fire.title()}\n"
            f"{int(is_defended.sum()):,} documented targets; "
            f"{int(is_matched.sum()):,} matched",
            loc="left", fontsize=8.3, fontweight="bold", pad=4,
        )
        summaries.append({
            "fire": fire,
            "documented_defense_targets": int(is_defended.sum()),
            "matched_defense_targets": int(is_matched.sum()),
            "not_retained_after_matching": int(
                (is_defended & ~is_matched).sum()
            ),
        })

    fig.legend(handles=[
        Line2D([0], [0], marker="o", ls="", markerfacecolor=VERMILLION,
               markeredgecolor="white", markersize=4.5,
               label="target retained in matched analysis"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor="none",
               markeredgecolor="#D8846B", markersize=4.5,
               label="other documented defense target"),
        Patch(facecolor="#DDDDDA", edgecolor="none",
              label="other structure"),
        Line2D([0], [0], color=INK, lw=.8, label="fire perimeter"),
    ], frameon=False, loc="lower center", ncol=4,
       bbox_to_anchor=(.5, .015), fontsize=6.2)
    fig.subplots_adjust(left=.015, right=.995, top=.91, bottom=.14, wspace=.035)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "ED_defense_target_locations"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight",
                facecolor="white")
    return fig, pd.DataFrame(summaries)


def _building(ax, x, y, scale=1, angle=0, face=PALE, edge=MID):
    footprint = np.array([
        [-.50, -.34], [.20, -.34], [.20, -.17], [.50, -.17],
        [.50, .34], [-.12, .34], [-.12, .18], [-.50, .18],
    ]) * scale
    transform = Affine2D().rotate_deg(angle).translate(x, y) + ax.transData
    ax.add_patch(Polygon(
        footprint, closed=True, facecolor=face, edgecolor=edge,
        linewidth=.65, transform=transform, zorder=3,
    ))


def plot_defense_spillover(results: dict, output_dir: Path):
    """Draw the directional design and fire-specific spillover estimates."""
    apply_style()
    fig = plt.figure(figsize=(7.2, 3.25))
    grid = fig.add_gridspec(1, 2, width_ratios=[.95, 1.28], wspace=.27)
    design = fig.add_subplot(grid[0, 0])
    forest = fig.add_subplot(grid[0, 1])

    design.set_xlim(-3.35, 3.35); design.set_ylim(-2.25, 2.25)
    design.set_aspect("equal"); design.axis("off")
    design.text(-3.28, 2.18, "a", fontsize=9, fontweight="bold", va="top")
    design.text(-2.90, 2.18, "Directional comparison", fontsize=8.3,
                fontweight="bold", va="top")
    design.annotate("", xy=(2.88, 1.48), xytext=(-2.82, 1.48),
                    arrowprops=dict(arrowstyle="-|>", lw=.8,
                                    color=VERMILLION, mutation_scale=8))
    design.text(0, 1.61, "Fire progression", ha="center", va="bottom",
                fontsize=6.4, color=VERMILLION)
    for spec in [(-2.58,.70,.68,10),(-2.32,-.46,.68,-8),
                 (-1.72,1.03,.68,-15),(-1.66,-1.24,.68,14)]:
        _building(design, *spec, edge="#A5A5A2")
    for spec in [(1.56,.98,.68,13),(2.43,.43,.68,-7),
                 (1.66,-.82,.68,-13),(2.53,-1.14,.68,9)]:
        _building(design, *spec, face=PALE_RED, edge=VERMILLION)
    _building(design, 0, 0, .80, 0, face=NAVY, edge=NAVY)
    design.plot([.22,.98],[-.18,-.95],color="#A9A9A6",lw=.55,zorder=1)
    design.text(.60,-.63,"100 ft",fontsize=5.6,color="#888888",rotation=-45)
    design.text(0,.47,"defended focal",ha="center",fontsize=6.5,color=NAVY)
    design.text(-2.25,-1.72,"Earlier arrival\nUp-fire placebo",ha="center",
                va="top",fontsize=6.4,color=MID)
    design.text(2.20,-1.72,"Later arrival\nDown-fire neighbors",ha="center",
                va="top",fontsize=6.4,color=VERMILLION)

    labels = [key for key in ["pooled", "eaton", "palisades"] if key in results]
    interval_lows = [
        100 * float(results[label][2][key][0])
        for label in labels for key in ["did", "late"]
    ]
    x_min = 20 * np.floor((min(interval_lows) - 4) / 20)
    x_min = min(-100, x_min)
    y = np.arange(len(labels))[::-1]
    for i, label in enumerate(labels):
        n, estimate, interval = results[label]
        for key, dy, color, marker in [
            ("did", .16, MID, "o"), ("late", -.16, VERMILLION, "s")
        ]:
            point = 100 * estimate[key]
            lo, hi = 100 * np.asarray(interval[key])
            forest.errorbar(
                point, y[i]+dy, xerr=[[point-lo],[hi-point]], fmt=marker,
                color=color, markerfacecolor="white" if key=="did" else color,
                markersize=4.2 if key=="did" else 5.2, capsize=2.4,
                elinewidth=.9 if key=="did" else 1.3, zorder=3,
            )
        forest.text(x_min + 1.5, y[i], f"{label.title()} (n={n:,})",
                    ha="left", va="center", fontsize=6.8)
    forest.axvline(0,color="#707070",lw=.9)
    forest.grid(axis="x",color="#ECECE9",lw=.45)
    forest.set_yticks([]); forest.set_xlim(x_min, 5)
    forest.set_xticks(np.arange(x_min + 20, 1, 20))
    forest.set_xlabel("Effect on down-fire neighbor destruction\n(percentage points)")
    forest.set_title("")
    forest.text(0, 1.115, "b", transform=forest.transAxes, fontsize=9,
                fontweight="bold", va="bottom")
    forest.text(.055, 1.115, "Directional spillover estimates",
                transform=forest.transAxes, fontsize=8.3,
                fontweight="bold", va="bottom")
    forest.text(0, 1.045, "Negative values indicate fewer destroyed neighbors",
                transform=forest.transAxes, fontsize=6.2, color=MID,
                va="bottom")
    forest.legend(handles=[
        Line2D([0],[0],marker="s",ls="",color=VERMILLION,
               label="Local IV estimate"),
        Line2D([0],[0],marker="o",ls="",markerfacecolor="white",color=MID,
               label="Placebo-corrected directional contrast"),
    ],loc="lower left",frameon=False,fontsize=6.2)
    forest.spines[["top","right","left"]].set_visible(False)
    forest.set_ylim(-.72,len(labels)-.28)
    fig.subplots_adjust(left=.025,right=.99,top=.87,bottom=.16)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    stem = output_dir / "Fig3_defense_spillover"
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    fig.savefig(stem.with_suffix(".png"),dpi=600,bbox_inches="tight",facecolor="white")
    return fig


def plot_defense_spillover_stages(table, output_dir: Path):
    """Compare directional neighbor effects for damage and escalation."""
    apply_style()
    outcome_order = [
        "Any damage", "Destruction", "Destruction conditional on damage",
    ]
    outcome_labels = [
        "Any damage", "Total destruction", "Destruction among damaged",
    ]
    samples = [
        ("Pooled", INK, "o", 0.00, 5.0),
        ("Eaton", VERMILLION, "o", 0.13, 3.8),
        ("Palisades", NAVY, "s", -0.13, 3.8),
    ]
    panels = [
        ("directional_contrast", "Direction-adjusted neighbor contrast"),
        ("local_iv", "Survival-scaled local estimate"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharey=True)
    y_base = np.arange(len(outcome_order))[::-1]
    for panel_index, (ax, (metric, title)) in enumerate(zip(axes, panels)):
        for sample, color, marker, offset, marker_size in samples:
            frame = table[table["sample"].eq(sample)].set_index("outcome")
            points = 100 * frame.loc[outcome_order, metric].to_numpy()
            lows = 100 * frame.loc[outcome_order, f"{metric}_lo"].to_numpy()
            highs = 100 * frame.loc[outcome_order, f"{metric}_hi"].to_numpy()
            ax.errorbar(
                points, y_base + offset,
                xerr=np.vstack((points - lows, highs - points)),
                fmt=marker, color=color,
                markerfacecolor=color if sample == "Pooled" else "white",
                markeredgewidth=.8, markersize=marker_size,
                elinewidth=1.15 if sample == "Pooled" else .75,
                capsize=2.0, zorder=4 if sample == "Pooled" else 3,
                label=sample,
            )
        ax.axvline(0, color="#777777", lw=.8, zorder=1)
        ax.grid(axis="x", color="#E8E8E5", lw=.5, zorder=0)
        ax.set_title(f"{chr(97 + panel_index)}  {title}", loc="left",
                     fontsize=8.3, fontweight="bold", pad=7)
        ax.set_xlabel("Change in neighbor outcome\n(percentage points)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y_base, outcome_labels)
    axes[0].set_xlim(-25, 8)
    axes[1].set_xlim(-100, 25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5,
               loc="lower center", ncol=3, bbox_to_anchor=(.57, .005))
    fig.subplots_adjust(left=.24, right=.99, bottom=.28, top=.89, wspace=.16)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "ED_defense_spillover_stages"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight",
                facecolor="white")
    return fig
