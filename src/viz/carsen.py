"""Independent statewide and regional panels for California 2D SENs."""
from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import (
    BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm, to_rgba,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from pyproj import Transformer
from shapely import intersects_xy


INK = "#242424"
OTHER = "#D9D9D6"
FIRE_PERIMETER_COLOR = "#555555"
CATEGORY_COLORS = {
    "Influence": "#E3B341",
    "Intermix": "#6F9B8A",
    "Interface": "#3B78A3",
    "Interconnectivity": "#B24C63",
}
METHOD_LABELS = {
    "top1": "Cumulative Top-1",
    "top2": "Cumulative Top-2 · α=.25",
    "eeat": "EEAT",
    "eeat_top2_fmax": "EEAT Top-2 · Fmax-calibrated",
}


def load_california_context(
    boundary_path: Path, wui_classes_path: Path,
) -> tuple[gpd.GeoDataFrame, dict[str, gpd.GeoDataFrame]]:
    """Load California boundary and the three official mapped-WUI classes."""
    boundary = gpd.read_file(f"zip://{Path(boundary_path)}")
    boundary = boundary.loc[boundary.STUSPS.eq("CA")].to_crs(3310)
    wui = gpd.read_parquet(wui_classes_path).to_crs(3310)
    labels = wui.WUI_DESC.astype(str).str.casefold()
    classes = {
        "Influence": wui.loc[labels.eq("influence zone")].copy(),
        "Intermix": wui.loc[labels.eq("intermix")].copy(),
        "Interface": wui.loc[labels.eq("interface")].copy(),
    }
    if len(boundary) != 1 or any(frame.empty for frame in classes.values()):
        raise RuntimeError("California boundary and all three WUI classes are required")
    return boundary, classes


def build_component_spatial_summary(
    assignments_path: Path,
    centroids_path: Path,
    wui_classes: dict[str, gpd.GeoDataFrame],
    output_path: Path,
    *,
    force: bool = False,
) -> Path:
    """Cache component centers, Interface contact, and beyond-WUI share."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path
    con = duckdb.connect()
    try:
        points = con.execute(
            """
            SELECT a.component_id, a.component_size, c.x, c.y
            FROM read_parquet(?) a
            JOIN read_parquet(?) c USING (building_id)
            WHERE a.component_size > 1
            """,
            [str(assignments_path), str(centroids_path)],
        ).fetchdf()
    finally:
        con.close()
    interface_geometry = wui_classes["Interface"].geometry.union_all()
    mapped_geometry = pd.concat(
        [frame.geometry for frame in wui_classes.values()], ignore_index=True,
    ).union_all()
    points["inside_interface"] = intersects_xy(
        interface_geometry, points.x.to_numpy(), points.y.to_numpy(),
    )
    points["outside_mapped_wui"] = ~intersects_xy(
        mapped_geometry, points.x.to_numpy(), points.y.to_numpy(),
    )
    summary = (
        points.groupby("component_id", as_index=False)
        .agg(
            component_size=("component_size", "first"),
            x=("x", "mean"), y=("y", "mean"),
            xmin=("x", "min"), xmax=("x", "max"),
            ymin=("y", "min"), ymax=("y", "max"),
            interface_buildings=("inside_interface", "sum"),
            outside_mapped_wui_buildings=("outside_mapped_wui", "sum"),
        )
    )
    summary["touches_interface"] = summary.interface_buildings.gt(0)
    summary["outside_mapped_wui_share"] = (
        summary.outside_mapped_wui_buildings / summary.component_size
    )
    summary.to_parquet(output_path, index=False, compression="zstd")
    return output_path


def _density_background(ax, centroids: pd.DataFrame, bounds, bins=700):
    xmin, ymin, xmax, ymax = bounds
    width, height = xmax - xmin, ymax - ymin
    nx = bins
    ny = max(1, int(bins * height / width))
    density, _, _ = np.histogram2d(
        centroids.y, centroids.x,
        bins=(ny, nx), range=((ymin, ymax), (xmin, xmax)),
    )
    image = np.ma.masked_equal(np.log1p(density), 0)
    nonzero = image.compressed()
    upper = float(np.percentile(nonzero, 99.7)) if nonzero.size else 1.0
    ax.imshow(
        image, extent=(xmin, xmax, ymin, ymax), origin="lower",
        cmap="Greys", vmin=0, vmax=max(1.0, upper),
        alpha=.28, interpolation="nearest", zorder=0,
    )


def plot_statewide_panel(
    boundary: gpd.GeoDataFrame,
    wui_classes: dict[str, gpd.GeoDataFrame],
    component_summary_paths: dict[str, Path] | Path,
    output_stem: Path | dict[str, Path],
    legacy_output_stem: Path | None = None,
    *,
    minimum_component_size: int = 10,
    hexbin_gridsize: int = 145,
    method_labels: dict[str, str] | None = None,
):
    """Draw statewide component-size hexbins with mapped-WUI boundaries.

    ``legacy_output_stem`` preserves the former call shape that included an
    unused centroid-path argument between ``wui_classes`` and the component
    summaries. New calls should omit that argument.
    """
    if legacy_output_stem is not None:
        component_summary_paths, output_stem = output_stem, legacy_output_stem
    if not isinstance(component_summary_paths, dict):
        raise TypeError("component_summary_paths must be a mapping keyed by method")
    xmin, ymin, xmax, ymax = boundary.total_bounds
    methods = tuple(component_summary_paths)
    if not methods:
        raise ValueError("component_summary_paths cannot be empty")
    shown_by_method = {}
    for method in methods:
        metrics = pd.read_parquet(component_summary_paths[method])
        shown_by_method[method] = metrics.loc[
            metrics.touches_interface
            & metrics.component_size.ge(minimum_component_size)
        ].copy()
    displayed_sizes = np.concatenate([
        frame.component_size.to_numpy(dtype=float)
        for frame in shown_by_method.values() if len(frame)
    ])
    color_min = max(1, minimum_component_size)
    color_max = float(displayed_sizes.max()) if displayed_sizes.size else color_min
    norm = LogNorm(vmin=color_min, vmax=max(color_min * 1.01, color_max))
    cubehelix = LinearSegmentedColormap.from_list(
        "cluster_cubehelix",
        plt.get_cmap("cubehelix_r")(np.linspace(.22, .96, 256)),
    )
    fig_width = 6.2 if len(methods) == 1 else 5.4 * len(methods)
    fig, axes = plt.subplots(
        1, len(methods), figsize=(fig_width, 7.0), sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes)
    panel_rows = []
    mappable = None
    for ax, method in zip(axes, methods):
        shown = shown_by_method[method]
        if len(shown):
            mappable = ax.hexbin(
                shown.x, shown.y, C=shown.component_size,
                reduce_C_function=np.max, gridsize=hexbin_gridsize,
                extent=(xmin, xmax, ymin, ymax), mincnt=1,
                cmap=cubehelix, norm=norm, linewidths=0,
                alpha=.9, rasterized=True, zorder=2,
            )
        for class_name in ("Influence", "Intermix", "Interface"):
            wui_classes[class_name].boundary.plot(
                ax=ax, color=CATEGORY_COLORS[class_name],
                linewidth=.22, alpha=.72, zorder=3,
            )
        boundary.boundary.plot(ax=ax, color=INK, linewidth=.7, zorder=4)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
        ax.axis("off")
        total_buildings = int(shown.component_size.sum())
        outside = int(shown.outside_mapped_wui_buildings.sum())
        outside_share = outside / total_buildings if total_buildings else np.nan
        largest = int(shown.component_size.max()) if len(shown) else 0
        display_label = (method_labels or {}).get(
            method, METHOD_LABELS.get(method, method),
        )
        ax.set_title(
            f"{display_label}\n"
            f"{len(shown):,} Interface-connected components ≥{minimum_component_size}\n"
            f"{total_buildings:,} buildings · {outside_share:.0%} beyond WUI · "
            f"largest {largest:,}",
            loc="left", fontsize=7.6, fontweight="bold", pad=5,
        )
        panel_rows.append({
            "method": method,
            "minimum_component_size": minimum_component_size,
            "displayed_interface_connected_components": len(shown),
            "displayed_buildings": total_buildings,
            "outside_mapped_wui_buildings": outside,
            "outside_mapped_wui_share": outside_share,
            "largest_displayed_component": largest,
        })
    handles = [
        Line2D([0], [0], color=CATEGORY_COLORS[class_name], lw=1.2,
               label=f"{class_name} boundary")
        for class_name in ("Influence", "Intermix", "Interface")
    ]
    fig.legend(
        handles=handles, title="Official mapped-WUI classes",
        loc="lower center", ncol=3, frameon=False, fontsize=6.8,
        title_fontsize=6.8, bbox_to_anchor=(.45, .02),
    )
    if mappable is not None:
        colorbar = fig.colorbar(
            mappable, ax=axes, orientation="vertical",
            fraction=.025, pad=.018, aspect=34,
        )
        colorbar.set_label(
            "Largest component centered in hexagon (buildings; log scale)",
            fontsize=7,
        )
        colorbar.ax.tick_params(labelsize=6.5)
    right = .87 if len(methods) == 1 else .91
    fig.subplots_adjust(left=.01, right=right, top=.93, bottom=.09, wspace=.04)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".png"), dpi=600,
        bbox_inches="tight", facecolor="white",
    )
    return fig, pd.DataFrame(panel_rows)


def _class_masks(points: pd.DataFrame, classes):
    interface = intersects_xy(
        classes["Interface"].geometry.union_all(), points.x, points.y,
    )
    intermix = intersects_xy(
        classes["Intermix"].geometry.union_all(), points.x, points.y,
    ) & ~interface
    influence = intersects_xy(
        classes["Influence"].geometry.union_all(), points.x, points.y,
    ) & ~interface & ~intermix
    return {"Influence": influence, "Intermix": intermix, "Interface": interface}


def _regional_points(assignments_path, centroids_path, bounds):
    xmin, ymin, xmax, ymax = bounds
    con = duckdb.connect()
    try:
        return con.execute(
            """
            SELECT c.building_id, c.x, c.y, a.component_id, a.component_size
            FROM read_parquet(?) c
            JOIN read_parquet(?) a USING (building_id)
            WHERE c.x BETWEEN ? AND ? AND c.y BETWEEN ? AND ?
            """,
            [str(centroids_path), str(assignments_path), xmin, xmax, ymin, ymax],
        ).fetchdf()
    finally:
        con.close()


def _scale_bar(ax, bounds):
    xmin, ymin, xmax, ymax = bounds
    width, height = xmax - xmin, ymax - ymin
    length = 10_000 if width >= 30_000 else 5_000
    x0, y0 = xmin + .06 * width, ymin + .055 * height
    ax.plot([x0, x0 + length], [y0, y0], color=INK, lw=1.0, zorder=10)
    ax.plot([x0, x0], [y0 - .008 * height, y0 + .008 * height], color=INK, lw=.8)
    ax.plot([x0 + length, x0 + length],
            [y0 - .008 * height, y0 + .008 * height], color=INK, lw=.8)
    ax.text(x0 + length / 2, y0 + .015 * height, f"{length // 1000} km",
            ha="center", va="bottom", fontsize=6)
    ax.text(.96, .96, "N\n↑", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.5)


def _place_labels(ax, places, bounds, crs=3310, halo=False):
    xmin, ymin, xmax, ymax = bounds
    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    effects = (
        [patheffects.withStroke(linewidth=1.8, foreground="white")]
        if halo else None
    )
    for name, (lon, lat) in places.items():
        x, y = transformer.transform(lon, lat)
        if xmin <= x <= xmax and ymin <= y <= ymax:
            ax.scatter([x], [y], s=4, facecolor="white", edgecolor=INK,
                       linewidth=.5, zorder=12)
            ax.annotate(name, (x, y), xytext=(3, 3), textcoords="offset points",
                        fontsize=5.8, color=INK, zorder=12,
                        path_effects=effects)


def plot_regional_panel(
    region_name: str,
    bounds: tuple[float, float, float, float],
    assignments_paths: dict[str, Path],
    component_summary_paths: dict[str, Path],
    centroids_path: Path,
    boundary: gpd.GeoDataFrame,
    wui_classes: dict[str, gpd.GeoDataFrame],
    output_stem: Path,
    *,
    minimum_component_size: int = 10,
    places: dict[str, tuple[float, float]] | None = None,
):
    """Draw one fixed regional panel with Top-2 and EEAT side by side."""
    xmin, ymin, xmax, ymax = bounds
    local_classes = {
        name: frame.cx[xmin:xmax, ymin:ymax]
        for name, frame in wui_classes.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharex=True, sharey=True)
    rows = []
    for ax, method in zip(axes, ("top2", "eeat")):
        points = _regional_points(assignments_paths[method], centroids_path, bounds)
        masks = _class_masks(points, local_classes)
        outside = ~np.logical_or.reduce(list(masks.values()))
        metrics = pd.read_parquet(component_summary_paths[method])
        connected = metrics.loc[
            metrics.touches_interface
            & metrics.component_size.ge(minimum_component_size)
        ]
        connected_ids = set(connected.component_id)
        interface_plus = outside & points.component_id.isin(connected_ids).to_numpy()
        ax.scatter(points.x, points.y, s=.22, color=OTHER, alpha=.34,
                   linewidths=0, rasterized=True, zorder=1)
        for zorder, name in enumerate(("Influence", "Intermix", "Interface"), 2):
            selected = masks[name]
            ax.scatter(points.loc[selected, "x"], points.loc[selected, "y"],
                       s=.34, color=CATEGORY_COLORS[name], alpha=.76,
                       linewidths=0, rasterized=True, zorder=zorder)
        ax.scatter(points.loc[interface_plus, "x"], points.loc[interface_plus, "y"],
                   s=.42, color=CATEGORY_COLORS["Interconnectivity"], alpha=.88,
                   linewidths=0, rasterized=True, zorder=6)
        for name, frame in local_classes.items():
            if len(frame):
                frame.boundary.plot(ax=ax, color=CATEGORY_COLORS[name],
                                    linewidth=.42, alpha=.62, zorder=7)
        local_boundary = boundary.cx[xmin:xmax, ymin:ymax]
        if len(local_boundary):
            local_boundary.boundary.plot(ax=ax, color=INK, linewidth=.8, zorder=8)
        if places:
            _place_labels(ax, places, bounds)
        _scale_bar(ax, bounds)
        represented = connected.loc[
            connected.component_id.isin(points.component_id.unique())
        ]
        if len(represented):
            featured = represented.sort_values(
                ["component_size", "component_id"], ascending=[False, True]
            ).iloc[0]
            largest = int(featured.component_size)
            beyond = float(featured.outside_mapped_wui_share)
        else:
            largest, beyond = 0, np.nan
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
        ax.axis("off")
        ax.set_title(
            f"{METHOD_LABELS[method]}\n"
            f"largest Interface-connected component: {largest:,} buildings · "
            f"{beyond:.0%} beyond WUI" if largest else
            f"{METHOD_LABELS[method]}\nno Interface-connected component ≥{minimum_component_size}",
            loc="left", fontsize=7.8, fontweight="bold", pad=4,
        )
        rows.append({
            "region": region_name, "method": method,
            "buildings_in_view": len(points),
            "interface_plus_buildings_in_view": int(interface_plus.sum()),
            "largest_interface_connected_component": largest,
            "largest_component_outside_wui_share": beyond,
        })
    handles = [Patch(facecolor=OTHER, edgecolor="none", label="Other buildings")]
    handles += [
        Patch(facecolor=CATEGORY_COLORS[name], edgecolor="none", label=name)
        for name in ("Influence", "Intermix", "Interface", "Interconnectivity")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=6.4, bbox_to_anchor=(.5, .01))
    fig.suptitle(region_name, x=.015, ha="left", y=.99,
                 fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=.015, right=.99, top=.89, bottom=.12, wspace=.035)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600,
                bbox_inches="tight", facecolor="white")
    return fig, pd.DataFrame(rows)


def plot_eeat_interface_sensitivity_maps(
    region_name: str,
    bounds: tuple[float, float, float, float],
    assignments_path: Path,
    component_sensitivity: pd.DataFrame,
    centroids_path: Path,
    boundary: gpd.GeoDataFrame,
    wui_classes: dict[str, gpd.GeoDataFrame],
    output_stem: Path,
    *,
    buffer_miles: tuple[float, ...] = (0, .5, 1, 1.5, 2),
    minimum_component_size: int = 10,
    places: dict[str, tuple[float, float]] | None = None,
):
    """Map EEAT Interface-expansion scenarios in the regional-panel style.

    Official WUI colors describe each building's mapped class. ``Interconnectivity``
    identifies buildings outside mapped WUI whose fixed EEAT component reaches
    the Interface after the stated expansion. Counts and shares in panel titles
    refer to component members represented inside ``bounds``.
    """
    xmin, ymin, xmax, ymax = bounds
    local_classes = {
        name: frame.cx[xmin:xmax, ymin:ymax]
        for name, frame in wui_classes.items()
    }
    points = _regional_points(assignments_path, centroids_path, bounds)
    masks = _class_masks(points, local_classes)
    outside = ~np.logical_or.reduce(list(masks.values()))
    local_boundary = boundary.cx[xmin:xmax, ymin:ymax]

    ncols = 3
    nrows = int(np.ceil(len(buffer_miles) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15.6, 6.3), sharex=True, sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)
    rows = []
    for ax, miles in zip(axes, buffer_miles):
        interacting = component_sensitivity.loc[
            component_sensitivity.interface_expansion_miles.eq(miles)
            & component_sensitivity.global_component_size.ge(minimum_component_size)
        ].sort_values(
            ["global_component_size", "component_id"], ascending=[False, True],
        )
        interacting_ids = set(interacting.component_id)
        represented = points.component_id.isin(interacting_ids).to_numpy()
        interface_plus = outside & represented

        ax.scatter(
            points.x, points.y, s=.22, color=OTHER, alpha=.34,
            linewidths=0, rasterized=True, zorder=1,
        )
        for zorder, name in enumerate(("Influence", "Intermix", "Interface"), 2):
            selected = masks[name]
            ax.scatter(
                points.loc[selected, "x"], points.loc[selected, "y"],
                s=.34, color=CATEGORY_COLORS[name], alpha=.76,
                linewidths=0, rasterized=True, zorder=zorder,
            )
        ax.scatter(
            points.loc[interface_plus, "x"], points.loc[interface_plus, "y"],
            s=.42, color=CATEGORY_COLORS["Interconnectivity"], alpha=.88,
            linewidths=0, rasterized=True, zorder=6,
        )
        for name, frame in local_classes.items():
            if len(frame):
                frame.boundary.plot(
                    ax=ax, color=CATEGORY_COLORS[name],
                    linewidth=.42, alpha=.62, zorder=7,
                )
        if len(local_boundary):
            local_boundary.boundary.plot(ax=ax, color=INK, linewidth=.8, zorder=8)
        if places:
            _place_labels(ax, places, bounds)
        _scale_bar(ax, bounds)

        represented_components = interacting.loc[
            interacting.component_id.isin(points.component_id.unique())
        ]
        if len(represented_components):
            featured = represented_components.iloc[0]
            featured_members = points.component_id.eq(featured.component_id).to_numpy()
            featured_beyond = (
                int((featured_members & outside).sum()) / int(featured_members.sum())
            )
            largest = int(featured.global_component_size)
        else:
            largest, featured_beyond = 0, np.nan
        n_components = int(points.loc[represented, "component_id"].nunique())
        ax.set_title(
            f"EEAT · Interface +{miles:g} mi\n"
            f"largest connected component: {largest:,} buildings · "
            f"{featured_beyond:.0%} beyond WUI in view" if largest else
            f"EEAT · Interface +{miles:g} mi\n"
            f"no connected component ≥{minimum_component_size} in view",
            loc="left", fontsize=7.8, fontweight="bold", pad=4,
        )
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
        ax.axis("off")
        rows.append({
            "region": region_name,
            "interface_expansion_miles": miles,
            "buildings_in_view": len(points),
            "represented_interacting_components": n_components,
            "interface_plus_buildings_in_view": int(interface_plus.sum()),
            "largest_interacting_global_component": largest,
            "largest_component_beyond_wui_share_in_view": featured_beyond,
        })

    for ax in axes[len(buffer_miles):]:
        ax.axis("off")
    handles = [Patch(facecolor=OTHER, edgecolor="none", label="Other buildings")]
    handles += [
        Patch(facecolor=CATEGORY_COLORS[name], edgecolor="none", label=name)
        for name in ("Influence", "Intermix", "Interface", "Interconnectivity")
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False,
        fontsize=6.4, bbox_to_anchor=(.5, .012),
    )
    fig.suptitle(
        f"{region_name} — EEAT Interface-expansion sensitivity",
        x=.012, ha="left", y=.995, fontsize=10, fontweight="bold",
    )
    fig.subplots_adjust(
        left=.012, right=.995, top=.90, bottom=.11, wspace=.035, hspace=.16,
    )
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".png"), dpi=600,
        bbox_inches="tight", facecolor="white",
    )
    return fig, pd.DataFrame(rows)


def plot_eeat_interface_expansion_persistence_map(
    region_name: str,
    bounds: tuple[float, float, float, float],
    assignments_path: Path,
    component_sensitivity: pd.DataFrame,
    centroids_path: Path,
    boundary: gpd.GeoDataFrame,
    wui_classes: dict[str, gpd.GeoDataFrame],
    output_stem: Path,
    *,
    buffer_miles: tuple[float, ...] = (0, .5, 1, 1.5),
    minimum_component_size: int = 10,
    places: dict[str, tuple[float, float]] | None = None,
    CMAP = 'Reds',
):
    """Map the first Interface expansion that captures each EEAT component.

    Official WUI classes retain their categorical colors. Red is reserved for
    component members outside all mapped WUI classes; darker red denotes
    contact under the smaller, more conservative Interface expansion.
    """
    distances = tuple(float(value) for value in buffer_miles)
    xmin, ymin, xmax, ymax = bounds
    local_classes = {
        name: frame.cx[xmin:xmax, ymin:ymax]
        for name, frame in wui_classes.items()
    }
    points = _regional_points(assignments_path, centroids_path, bounds)
    masks = _class_masks(points, local_classes)
    outside = ~np.logical_or.reduce(list(masks.values()))
    local_boundary = boundary.cx[xmin:xmax, ymin:ymax]

    eligible = component_sensitivity.loc[
        component_sensitivity.interface_expansion_miles.isin(distances)
        & component_sensitivity.global_component_size.ge(minimum_component_size)
    ]
    first_contact = (
        eligible.groupby("component_id").interface_expansion_miles.min()
    )
    points["first_interface_expansion_miles"] = points.component_id.map(
        first_contact
    )
    red_values = plt.get_cmap(CMAP)(np.linspace(.82, .42, len(distances)))
    distance_colors = dict(zip(distances, red_values))

    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.scatter(
        points.x, points.y, s=.22, color=OTHER, alpha=.34,
        linewidths=0, rasterized=True, zorder=1,
    )
    # Draw the broadest/lightest expansion first and the closest/darkest last.
    for zorder, miles in enumerate(reversed(distances), 2):
        selected = (
            outside
            & points.first_interface_expansion_miles.eq(miles).to_numpy()
        )
        ax.scatter(
            points.loc[selected, "x"], points.loc[selected, "y"],
            s=.42, color=distance_colors[miles], alpha=.90,
            linewidths=0, rasterized=True, zorder=zorder,
        )
    # Official WUI building classes and boundaries remain visually dominant.
    for zorder, name in enumerate(("Influence", "Intermix", "Interface"), 6):
        selected = masks[name]
        ax.scatter(
            points.loc[selected, "x"], points.loc[selected, "y"],
            s=.34, color=CATEGORY_COLORS[name], alpha=.78,
            linewidths=0, rasterized=True, zorder=zorder,
        )
        frame = local_classes[name]
        if len(frame):
            frame.boundary.plot(
                ax=ax, color=CATEGORY_COLORS[name], linewidth=.42,
                alpha=.68, zorder=10,
            )
    if len(local_boundary):
        local_boundary.boundary.plot(ax=ax, color=INK, linewidth=.7, zorder=11)
    if places:
        _place_labels(ax, places, bounds)
    _scale_bar(ax, bounds)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
    ax.axis("off")

    handles = [Patch(facecolor=OTHER, edgecolor="none", label="Other buildings")]
    handles += [
        Patch(facecolor=CATEGORY_COLORS[name], edgecolor="none", label=name)
        for name in ("Influence", "Intermix", "Interface")
    ]
    handles += [
        Patch(
            facecolor=distance_colors[miles], edgecolor="none",
            label=(
                "Interconnectivity connected to existing Interface"
                if np.isclose(miles, 0)
                else f"Interconnectivity first captured at {miles:g} mi"
            ),
        )
        for miles in distances
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=7, frameon=False,
        fontsize=6.2, bbox_to_anchor=(.5, .015),
    )
    fig.suptitle(
        f"{region_name} — EEAT Interface-expansion sensitivity",
        x=.015, ha="left", y=.99, fontsize=10, fontweight="bold",
    )
    ax.set_title(
        f"First expansion distance connecting each outside-WUI building "
        f"(components ≥{minimum_component_size} buildings)",
        loc="left", fontsize=7.8, fontweight="bold", pad=4,
    )
    fig.subplots_adjust(left=.015, right=.99, top=.87, bottom=.14)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".png"), dpi=600,
        bbox_inches="tight", facecolor="white",
    )
    summary = pd.DataFrame({
        "interface_expansion_miles": distances,
        "outside_wui_buildings_first_captured": [
            int((outside & points.first_interface_expansion_miles.eq(miles)).sum())
            for miles in distances
        ],
    })
    return fig, summary


def plot_eeat_interface_expansion_building_counts(
    assignments_path: Path,
    component_sensitivity: pd.DataFrame,
    centroids_path: Path,
    boundary: gpd.GeoDataFrame,
    buffer_geometries: gpd.GeoDataFrame,
    output_stem: Path,
    *,
    buffer_miles: tuple[float, ...] = (0, .5, 1, 1.5),
    minimum_component_size: int = 10,
):
    """Stack buildings inside expanded Interface zones and network reach beyond."""
    distances = tuple(float(value) for value in buffer_miles)
    county = boundary.to_crs(buffer_geometries.crs)
    county_geometry = county.geometry.union_all()
    points = _regional_points(
        assignments_path, centroids_path, tuple(county.total_bounds),
    )
    in_county = intersects_xy(
        county_geometry, points.x.to_numpy(), points.y.to_numpy(),
    )
    points = points.loc[in_county].reset_index(drop=True)

    rows = []
    for miles in distances:
        interacting_ids = set(component_sensitivity.loc[
            component_sensitivity.interface_expansion_miles.eq(miles)
            & component_sensitivity.global_component_size.ge(minimum_component_size),
            "component_id",
        ])
        associated = points.component_id.isin(interacting_ids).to_numpy()
        zone = buffer_geometries.loc[
            buffer_geometries.interface_expansion_miles.eq(miles), "geometry"
        ].iloc[0]
        inside_zone = intersects_xy(
            zone, points.x.to_numpy(), points.y.to_numpy(),
        )
        rows.append({
            "interface_expansion_miles": miles,
            "associated_buildings_inside_expanded_interface": int(
                (associated & inside_zone).sum()
            ),
            "associated_buildings_beyond_expanded_interface": int(
                (associated & ~inside_zone).sum()
            ),
        })
    counts = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = np.arange(len(counts))
    inside = counts.associated_buildings_inside_expanded_interface.to_numpy()
    beyond = counts.associated_buildings_beyond_expanded_interface.to_numpy()
    ax.bar(
        x, inside, width=.62, color=CATEGORY_COLORS["Interface"],
        label="Associated buildings inside expanded Interface zone",
    )
    ax.bar(
        x, beyond, width=.62, bottom=inside,
        color=CATEGORY_COLORS["Interconnectivity"],
        label="Additional component members beyond expanded zone",
    )
    for position, lower, upper in zip(x, inside, beyond):
        ax.text(
            position, lower / 2, f"{lower:,}", ha="center", va="center",
            fontsize=7, color="white", fontweight="bold",
        )
        ax.text(
            position, lower + upper / 2, f"{upper:,}",
            ha="center", va="center", fontsize=7, color="white",
            fontweight="bold",
        )
        ax.text(
            position, lower + upper, f"{lower + upper:,}",
            ha="center", va="bottom", fontsize=6.8, color=INK,
        )
    ax.set_xticks(x, [
        "Existing\nInterface" if np.isclose(miles, 0) else f"+{miles:g} mi"
        for miles in distances
    ])
    ax.set(
        xlabel="Outward Interface expansion",
        ylabel="Buildings in interacting EEAT components",
        title=f"EEAT P50 network reach (components ≥{minimum_component_size} buildings)",
    )
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=.18, linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".png"), dpi=600,
        bbox_inches="tight", facecolor="white",
    )
    return fig, counts


def plot_existing_wui_connectivity(
    method_buildings: dict[str, gpd.GeoDataFrame],
    wui: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    output_stem: Path,
    cmap: 'Reds',
    INFLUENCE_CMAP = "#F3E8C8"
):
    """Map connectivity within the combined existing mapped-WUI extent."""
    if not method_buildings:
        raise ValueError("At least one method is required")
    mapped_classes = ("Influence Zone", "Intermix", "Interface")
    xmin, ymin, xmax, ymax = bounds
    prepared = {}
    rows = []
    maximum = 1
    for method, source in method_buildings.items():
        frame = source.copy()
        mapped = frame.wui_class.isin(mapped_classes)
        mapped_frame = frame.loc[mapped]
        component_wui_size = mapped_frame.groupby("component_id").size()
        frame["mapped_wui_component_size"] = (
            frame.component_id.map(component_wui_size).fillna(0).astype(np.int64)
        )
        points = frame.geometry.representative_point()
        visible = (
            points.x.between(xmin, xmax) & points.y.between(ymin, ymax)
        )
        local = frame.loc[visible].copy()
        local["x"] = points.loc[visible].x.to_numpy()
        local["y"] = points.loc[visible].y.to_numpy()
        if len(local):
            maximum = max(maximum, int(local.mapped_wui_component_size.max()))
        prepared[method] = local

    scale_max = int(2 ** np.ceil(np.log2(maximum)))
    norm = LogNorm(vmin=1, vmax=max(2, scale_max))
    cmap = LinearSegmentedColormap.from_list(
        "mapped_wui_connectivity_reds",
        plt.get_cmap(cmap)(np.linspace(.32, .92, 256)),
    )
    fig, axes = plt.subplots(
        1, len(prepared), figsize=(12.8, 4.8),
        sharex=True, sharey=True, squeeze=False,
    )
    local_wui = wui.to_crs(next(iter(method_buildings.values())).crs).cx[
        xmin:xmax, ymin:ymax
    ]
    combined_wui = gpd.GeoDataFrame(
        {"zone": ["Existing mapped WUI"]},
        geometry=[local_wui.geometry.union_all()], crs=local_wui.crs,
    )
    influence_zone = local_wui.loc[
        local_wui.WUI_DESC.eq("Influence Zone")
    ]
    influence_sage = INFLUENCE_CMAP# "#A8B89A" #"#6F9B8A",
    for ax, (method, local) in zip(axes.ravel(), prepared.items()):
        if len(influence_zone):
            influence_zone.plot(
                ax=ax, facecolor=to_rgba(influence_sage, .3),
                edgecolor="#6F9B8A", zorder=.5,
            )
        ax.scatter(
            local.x, local.y, s=.16, color=OTHER, alpha=.24,
            linewidths=0, rasterized=True, zorder=1,
        )
        selected = local.wui_class.isin(mapped_classes)
        shown = local.loc[selected]
        ax.scatter(
            shown.x, shown.y,
            c=shown.mapped_wui_component_size.clip(lower=1),
            cmap=cmap, norm=norm, s=.38, alpha=.90,
            linewidths=0, rasterized=True, zorder=3,
        )
        if len(combined_wui):
            combined_wui.boundary.plot(
                ax=ax, color=CATEGORY_COLORS["Interface"],
                linewidth=.46, alpha=.70, zorder=5,
            )
        connected = shown.mapped_wui_component_size.gt(1)
        largest = int(
            shown.mapped_wui_component_size.max() if len(shown) else 0
        )
        connected_share = float(connected.mean()) if len(shown) else np.nan
        ax.set_title(
            f"{method}\n"
            f"largest mapped-WUI component: {largest:,} buildings · "
            f"{connected_share:.0%} connected",
            loc="left", fontsize=7.8, fontweight="bold", pad=4,
        )
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
        ax.axis("off")
        rows.append({
            "method": method,
            "buildings_in_view": len(local),
            "mapped_wui_buildings_in_view": len(shown),
            "largest_mapped_wui_component": largest,
            "connected_mapped_wui_buildings_in_view": int(connected.sum()),
            "connected_mapped_wui_share_in_view": connected_share,
        })

    cbar_ax = fig.add_axes([.27, .045, .46, .018])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax, orientation="horizontal",
    )
    ticks = 2 ** np.arange(0, int(np.log2(max(2, scale_max))) + 1, 2)
    if ticks[-1] != scale_max:
        ticks = np.append(ticks, scale_max)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{value:,}" for value in ticks])
    cbar.set_label(
        "Buildings from mapped WUI classes in full-county component",
        fontsize=6.8,
    )
    cbar.ax.tick_params(labelsize=6, length=2)
    fig.legend(
        handles=[Patch(
            facecolor=to_rgba(influence_sage, .22), edgecolor="none",
            label="Influence Zone context",
        )],
        loc="lower left", bbox_to_anchor=(.025, .035),
        frameon=False, fontsize=6.5,
    )
    fig.suptitle(
        "Santa Monica Mountains — connectivity within existing mapped WUI",
        x=.012, ha="left", y=.995, fontsize=10, fontweight="bold",
    )
    fig.subplots_adjust(
        left=.015, right=.995, top=.87, bottom=.14, wspace=.025,
    )
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".png"), dpi=600,
        bbox_inches="tight", facecolor="white",
    )
    return fig, pd.DataFrame(rows)


MAPPED_WUI_CLASSES = ("Influence Zone", "Intermix", "Interface")
WUI_BOUNDARY_COLOR = "#5B6B73"


def inscribed_bounds(polygon) -> tuple[float, float, float, float]:
    """Axis-aligned rectangle inside a slightly rotated projected window.

    A lon/lat box projected to UTM becomes a near-rectangular quadrilateral.
    Cropping to this rectangle avoids empty wedges at the map edges when data
    were selected with the quadrilateral itself.
    """
    corners = np.asarray(polygon.exterior.coords)[:-1]
    if len(corners) != 4:
        raise ValueError("Expected a projected four-corner window")
    xs, ys = np.sort(corners[:, 0]), np.sort(corners[:, 1])
    return float(xs[1]), float(ys[1]), float(xs[2]), float(ys[2])


def _window_points(assignments_path, centroids_path, bounds) -> pd.DataFrame:
    """Return window buildings with full-county component and mapped-WUI sizes."""
    xmin, ymin, xmax, ymax = bounds
    con = duckdb.connect()
    try:
        return con.execute(
            f"""
            WITH assignments AS (
                SELECT building_id, component_id, component_size, wui_class
                FROM read_parquet(?)
            ), mapped_sizes AS (
                SELECT component_id, COUNT(*) AS mapped_wui_component_size
                FROM assignments
                WHERE wui_class IN {MAPPED_WUI_CLASSES}
                GROUP BY component_id
            )
            SELECT c.x, c.y, a.component_id, a.component_size, a.wui_class,
                   COALESCE(m.mapped_wui_component_size, 0)
                       AS mapped_wui_component_size
            FROM assignments a
            JOIN read_parquet(?) c USING (building_id)
            LEFT JOIN mapped_sizes m USING (component_id)
            WHERE c.x BETWEEN ? AND ? AND c.y BETWEEN ? AND ?
            """,
            [str(assignments_path), str(centroids_path),
             xmin, xmax, ymin, ymax],
        ).fetchdf()
    finally:
        con.close()


def _county_outline(boundary, crs, bounds) -> gpd.GeoSeries:
    """Outer county line only; interior rings are small unincorporated holes."""
    xmin, ymin, xmax, ymax = bounds
    return gpd.GeoSeries(
        [part.exterior for part in boundary.to_crs(crs).geometry.explode()],
        crs=crs,
    ).cx[xmin:xmax, ymin:ymax]


def _draw_key(ax, rows, *, line_height=.135):
    """Draw a manual legend in an inch-scaled axes, top to bottom.

    ``rows`` holds ``(kind, color, label)`` tuples where ``kind`` is
    ``"header"``, ``"patch"``, ``"outlined_patch"``, ``"line"`` or ``"gap"``.
    Returns the height (in the axes' inch units) where the key ends.
    """
    _, height = ax.get_ylim()
    y = height
    for kind, color, label in rows:
        if kind == "gap":
            y -= line_height * .55
            continue
        if kind == "header":
            ax.text(0, y, label, ha="left", va="top", fontsize=6.5,
                    fontweight="bold", color=INK, linespacing=1.15)
            y -= .09 * (label.count("\n") + 1) + .065
            continue
        center = y - line_height / 2
        if kind in ("patch", "outlined_patch"):
            ax.add_patch(plt.Rectangle(
                (0, center - .035), .11, .07, facecolor=color,
                edgecolor="#AEB5B1" if kind == "outlined_patch" else "none",
                linewidth=.4,
            ))
        else:
            ax.plot([0, .11], [center, center], color=color, lw=.9,
                    solid_capstyle="butt")
        ax.text(.17, center, label, ha="left", va="center", fontsize=6.3,
                color=INK)
        y -= line_height
    return y


def plot_wui_connectivity_figure(
    assignments_path: Path,
    centroids_path: Path,
    persistence_map: pd.DataFrame,
    persistence_method: str,
    wui: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    output_stem: Path,
    *,
    probabilities: tuple[float, ...] = (.2, .3, .4, .5, .6, .7, .8),
    analysis_probability: float = .5,
    minimum_component_size: int = 10,
    method_label: str | None = None,
    places: dict[str, tuple[float, float]] | None = None,
    fire_perimeters: gpd.GeoDataFrame | None = None,
    size_scale_max: int | None = None,
    cmap: str = "YlOrRd",
    persistence_cmap: str = "YlOrRd",
    influence_color: str = "#E8ECE9",
    within_only: bool = False,
):
    """Paper figure of SEN connectivity within and beyond mapped WUI.

    Panel a colors mapped-WUI buildings by the number of mapped-WUI buildings
    in their full-county component at the analysis threshold. Panel b renders
    all non-highlighted structures in neutral gray and colors buildings outside
    mapped WUI by the highest probability-equivalent threshold at which they
    remain in an Interface-connected component, read from
    ``persistence_map[f"{persistence_method}_max_probability"]``.
    ``wui`` must share the CRS of the centroid table and ``persistence_map``;
    components are never rebuilt within ``bounds``. When supplied,
    ``fire_perimeters`` is filtered to the Eaton and Palisades incidents and
    drawn as a shared geographic reference in both panels.
    ``within_only=True`` renders panel a alone as a single-panel figure.
    """
    probabilities = tuple(float(value) for value in probabilities)
    persistence_column = f"{persistence_method}_max_probability"
    persistence_size_column = (
        f"{persistence_method}_component_size_at_max_probability"
    )
    xmin, ymin, xmax, ymax = bounds
    points = _window_points(assignments_path, centroids_path, bounds)

    # Panel a: component size counted over mapped-WUI members county-wide.
    mapped = points.wui_class.isin(MAPPED_WUI_CLASSES).to_numpy()
    outside = ~mapped
    mapped_sizes = points.loc[mapped, "mapped_wui_component_size"]
    largest = int(mapped_sizes.max()) if len(mapped_sizes) else 0
    connected_share = (
        float(mapped_sizes.gt(1).mean()) if len(mapped_sizes) else np.nan
    )
    if size_scale_max is None:
        size_scale_max = int(2 ** np.ceil(np.log2(max(2, largest))))
    size_norm = LogNorm(vmin=1, vmax=size_scale_max)
    size_cmap = LinearSegmentedColormap.from_list(
        "mapped_wui_connectivity",
        plt.get_cmap(cmap)(np.linspace(.32, .92, 256)),
    )

    # Panel b: highest threshold retaining outside-WUI Interconnectivity.
    # The single-panel form does not read or validate persistence data.
    persistence = pd.DataFrame()
    persistence_outside = np.zeros(0, dtype=bool)
    level = np.zeros(0, dtype=int)
    interconnectivity = np.zeros(0, dtype=bool)
    level_colors = plt.get_cmap(persistence_cmap)(
        np.linspace(.32, .92, len(probabilities))
    )
    spanning_counts = {}
    if not within_only:
        missing = {
            persistence_column, persistence_size_column,
        }.difference(persistence_map.columns)
        if missing:
            raise ValueError(f"persistence_map is missing {sorted(missing)}")
        persistence = persistence_map.loc[
            persistence_map.x.between(xmin, xmax)
            & persistence_map.y.between(ymin, ymax),
            ["x", "y", "wui_class", persistence_column,
             persistence_size_column],
        ].reset_index(drop=True)
        retained = persistence[persistence_column].to_numpy(float)
        has_value = ~np.isnan(retained)
        level = np.full(len(persistence), -1)
        for index, probability in enumerate(probabilities):
            level[has_value & np.isclose(retained, probability)] = index
        if (has_value & (level < 0)).any():
            raise ValueError(
                "Persistence values fall outside the stated probabilities"
            )
        persistence_outside = persistence.wui_class.eq(
            "Outside mapped WUI"
        ).to_numpy()
        large_enough = persistence[persistence_size_column].fillna(0).ge(
            minimum_component_size,
        ).to_numpy()
        interconnectivity = persistence_outside & large_enough & (level >= 0)
        spanning_counts = {
            probability: int((interconnectivity & (level >= index)).sum())
            for index, probability in enumerate(probabilities)
        }

    crs = wui.crs
    local_wui = wui.cx[xmin:xmax, ymin:ymax]
    wui_outline = gpd.GeoSeries(
        [local_wui.geometry.union_all()], crs=crs,
    ).boundary
    influence_zone = local_wui.loc[local_wui.WUI_DESC.eq("Influence Zone")]
    county_outline = _county_outline(boundary, crs, bounds)
    local_fire_perimeters = None
    if fire_perimeters is not None:
        if "FIRE_NAME" not in fire_perimeters.columns:
            raise ValueError("fire_perimeters must contain a FIRE_NAME column")
        local_fire_perimeters = fire_perimeters.to_crs(crs)
        fire_names = local_fire_perimeters.FIRE_NAME.astype(str).str.upper()
        local_fire_perimeters = local_fire_perimeters.loc[
            fire_names.isin(("EATON", "PALISADES"))
        ].cx[xmin:xmax, ymin:ymax]

    # Fixed physical layout (inches) for a 180-mm double-column figure.
    width_in, map_w, left = 7.2, 5.35, .04
    map_h = map_w * (ymax - ymin) / (xmax - xmin)
    head, gap, pad = (.27 if within_only else .44), .1, .04
    height_in = (pad + map_h + head if within_only
                 else pad + 2 * (map_h + head) + gap)
    key_x, key_w = left + map_w + .16, width_in - (left + map_w + .16) - .02
    fig = plt.figure(figsize=(width_in, height_in))

    def inches(x, y, w, h):
        return [x / width_in, y / height_in, w / width_in, h / height_in]

    y_panel = ({"a": pad} if within_only else
               {"a": pad + map_h + head + gap, "b": pad})
    axes = {
        panel: fig.add_axes(inches(left, y, map_w, map_h))
        for panel, y in y_panel.items()
    }

    def context(ax, *, draw_mapped_outline=True):
        if len(county_outline):
            county_outline.plot(ax=ax, color=INK, linewidth=.55, zorder=11)
        if draw_mapped_outline:
            wui_outline.plot(ax=ax, color=WUI_BOUNDARY_COLOR, linewidth=.35,
                             alpha=.8, zorder=10)
        if local_fire_perimeters is not None and len(local_fire_perimeters):
            # A narrow white casing keeps the perimeter legible over both the
            # gray building field and the red connectivity scale.
            local_fire_perimeters.boundary.plot(
                ax=ax, color="white", linewidth=1.15, zorder=15,
            )
            local_fire_perimeters.boundary.plot(
                ax=ax, color=FIRE_PERIMETER_COLOR, linewidth=.62, zorder=16,
            )
        if places:
            _place_labels(ax, places, bounds, crs=crs, halo=True)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
        ax.axis("off")

    ax = axes["a"]
    if len(influence_zone):
        influence_zone.plot(ax=ax, facecolor=to_rgba(influence_color, .55),
                            edgecolor="none", zorder=.5)
    ax.scatter(points.x[outside], points.y[outside], s=.16, color=OTHER,
               alpha=.28, linewidths=0, rasterized=True, zorder=1)
    shown = points.loc[mapped]
    ax.scatter(shown.x, shown.y,
               c=shown.mapped_wui_component_size.clip(lower=1),
               cmap=size_cmap, norm=size_norm, s=.38, alpha=.92,
               linewidths=0, rasterized=True, zorder=3)
    context(ax)

    if not within_only:
        ax = axes["b"]
        ax.scatter(
            persistence.x, persistence.y,
            s=.18, color=OTHER, alpha=.34, linewidths=0,
            rasterized=True, zorder=1,
        )
        # Least persistent first so robust (darkest) members sit on top.
        for index, color in enumerate(level_colors):
            mask = interconnectivity & (level == index)
            ax.scatter(persistence.x[mask], persistence.y[mask], s=.4,
                       color=color, alpha=.92, linewidths=0, rasterized=True,
                       zorder=2 + .1 * index)
        context(ax, draw_mapped_outline=False)
        for zorder, (source_name, display_name) in enumerate(
            {
                "Influence Zone": "Influence",
                "Intermix": "Intermix",
                "Interface": "Interface",
            }.items(), 12,
        ):
            frame = local_wui.loc[local_wui.WUI_DESC.eq(source_name)]
            if len(frame):
                frame.boundary.plot(
                    ax=ax, color=CATEGORY_COLORS[display_name],
                    linewidth=.52, zorder=zorder,
                )
        _scale_bar(ax, bounds)

    lowest, highest = probabilities[0], probabilities[-1]
    count_parts = [
        f"{spanning_counts[probability]:,} at P{100 * probability:.0f}"
        for probability in dict.fromkeys((lowest, analysis_probability, highest))
        if probability in spanning_counts
    ]
    headings = {
        "a": (
            "Connectivity within the mapped WUI",
            f"Largest SEN holds {largest:,} mapped-WUI buildings · "
            f"{connected_share:.0%} of mapped-WUI buildings in view share an "
            "SEN with another mapped-WUI building",
        ),
        "b": (
            "Interconnectivity extending beyond the mapped WUI",
            f"Interconnectivity buildings in SENs ≥{minimum_component_size} "
            "buildings: "
            + " · ".join(count_parts),
        ),
    }
    shown_headings = {"a": headings["a"]} if within_only else headings
    for panel, (title, subtitle) in shown_headings.items():
        top = y_panel[panel] + map_h + head
        title_x = left if within_only else left + .19
        if not within_only:
            fig.text(left / width_in, (top - .06) / height_in, panel,
                     ha="left", va="top", fontsize=9, fontweight="bold")
        fig.text(title_x / width_in, (top - .07) / height_in, title,
                 ha="left", va="top", fontsize=8, fontweight="bold")
        if not within_only:
            fig.text(title_x / width_in, (top - .26) / height_in, subtitle,
                     ha="left", va="top", fontsize=6.3, color="#555555")
    if method_label:
        fig.text((width_in - .02) / width_in, (height_in - .07) / height_in,
                 method_label, ha="right", va="top", fontsize=6.5,
                 color="#555555")

    # Panel a key: vertical size colorbar above a context legend.
    y_a = y_panel["a"]
    cbar_h = .46 * map_h
    cbar_ax = fig.add_axes(inches(key_x, y_a + .40 * map_h, .09, cbar_h))
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=size_norm, cmap=size_cmap),
                        cax=cbar_ax, orientation="vertical")
    ticks = 2 ** np.arange(0, int(np.log2(size_scale_max)) + 1, 2)
    if ticks[-1] != size_scale_max:
        ticks = np.append(ticks, size_scale_max)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{value:,}" for value in ticks])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=6, length=2, pad=1.5)
    cbar.outline.set_linewidth(.4)
    fig.text(key_x / width_in, (y_a + .40 * map_h + cbar_h + .06) / height_in,
             "Mapped-WUI buildings\nin the same SEN", ha="left", va="bottom",
             fontsize=6.5, fontweight="bold", color=INK, linespacing=1.15)
    key_h = .34 * map_h
    key_ax = fig.add_axes(inches(key_x, y_a + .02 * map_h, key_w, key_h))
    key_ax.set(xlim=(0, key_w), ylim=(0, key_h))
    key_ax.axis("off")
    _draw_key(key_ax, [
        ("outlined_patch", to_rgba(influence_color, .9), "Influence Zone area"),
        ("patch", OTHER, "Outside mapped WUI"),
        ("line", WUI_BOUNDARY_COLOR, "Mapped WUI boundary"),
        ("line", FIRE_PERIMETER_COLOR, "2025 Eaton–Palisades perimeters"),
        ("line", INK, "County boundary"),
    ])

    if not within_only:
        # Panel b key: neutral building context, class boundaries, and a
        # stepped red scale for outside-WUI Interconnectivity persistence.
        y_b = y_panel["b"]
        key_ax = fig.add_axes(inches(key_x, y_b, key_w, map_h))
        key_ax.set(xlim=(0, key_w), ylim=(0, map_h))
        key_ax.axis("off")
        rows = [
            ("header", None, "Building context"),
            ("patch", OTHER, "Other buildings"),
            ("line", CATEGORY_COLORS["Influence"], "Influence boundary"),
            ("line", CATEGORY_COLORS["Intermix"], "Intermix boundary"),
            ("line", CATEGORY_COLORS["Interface"], "Interface boundary"),
        ]
        rows += [("gap", None, None),
                 ("header", None,
                  "Outside mapped WUI:\nhighest threshold keeping\n"
                  "Interconnectivity")]
        key_end = _draw_key(key_ax, rows)
        step_h = .115
        steps_h = step_h * len(probabilities)
        steps_top = y_b + key_end - .03
        steps_ax = fig.add_axes(inches(key_x, steps_top - steps_h, .09, steps_h))
        percent = 100 * np.asarray(probabilities)
        half_step = np.diff(percent).min() / 2 if len(percent) > 1 else 5
        edges = np.r_[percent - half_step, percent[-1] + half_step]
        steps = fig.colorbar(
            plt.cm.ScalarMappable(
                norm=BoundaryNorm(edges, len(probabilities)),
                cmap=ListedColormap(level_colors),
            ),
            cax=steps_ax, orientation="vertical",
        )
        steps.set_ticks(percent)
        steps.set_ticklabels([f"P{value:.0f}" for value in percent])
        steps.ax.minorticks_off()
        steps.ax.tick_params(labelsize=6, length=2, pad=1.5)
        steps.outline.set_linewidth(.4)
        for label, value in zip(steps.ax.get_yticklabels(), percent):
            if np.isclose(value, 100 * analysis_probability):
                label.set_fontweight("bold")
        tail_top = steps_top - steps_h - .05
        tail_ax = fig.add_axes(inches(key_x, y_b, key_w, tail_top - y_b))
        tail_ax.set(xlim=(0, key_w), ylim=(0, tail_top - y_b))
        tail_ax.axis("off")
        _draw_key(tail_ax, [
            ("line", FIRE_PERIMETER_COLOR, "2025 Eaton–Palisades perimeters"),
            ("line", INK, "County boundary"),
        ])

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, facecolor="white")
    summary = {
        "method": method_label,
        "persistence_method": persistence_method,
        "buildings_in_view": len(points),
        "mapped_wui_buildings_in_view": int(mapped.sum()),
        "largest_mapped_wui_component": largest,
        "connected_mapped_wui_share_in_view": connected_share,
        "outside_wui_buildings_in_view": int(persistence_outside.sum()),
        "minimum_component_size": minimum_component_size,
        "fire_perimeters_shown": (
            0 if local_fire_perimeters is None else len(local_fire_perimeters)
        ),
    }
    summary.update({
        f"outside_wui_interface_spanning_at_p{100 * probability:.0f}": count
        for probability, count in spanning_counts.items()
    })
    return fig, pd.DataFrame([summary])


def plot_wui_overlap_clusters(
    method_assignments: dict[str, Path],
    centroids_path: Path,
    wui: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    output_stem: Path,
    *,
    places: dict[str, tuple[float, float]] | None = None,
    size_scale_max: int | None = None,
    wui_cmap: str = "YlOrRd",
    outside_cmap: str = "YlGnBu",
):
    """Map cluster size, split by whether each cluster touches mapped WUI.

    One stacked panel per entry in ``method_assignments`` (label -> county
    map cache). A cluster touches the mapped WUI when at least one member's
    centroid lies in any mapped WUI class; its members use ``wui_cmap``.
    Clusters entirely outside the mapped WUI use ``outside_cmap``. Both
    palettes share one log scale of full-county cluster size; isolated
    buildings are gray. ``wui`` must share the CRS of the centroid table.
    """
    xmin, ymin, xmax, ymax = bounds
    panels = {
        label: _window_points(path, centroids_path, bounds)
        for label, path in method_assignments.items()
    }
    if size_scale_max is None:
        largest = max(int(points.component_size.max()) for points in panels.values())
        size_scale_max = int(2 ** np.ceil(np.log2(max(4, largest))))
    norm = LogNorm(vmin=2, vmax=size_scale_max)
    # Trim the shared pale-yellow start so small clusters stay distinguishable.
    palettes = {
        touches: LinearSegmentedColormap.from_list(
            f"cluster_{name}", plt.get_cmap(name)(np.linspace(.3, .95, 256)),
        )
        for touches, name in ((True, wui_cmap), (False, outside_cmap))
    }

    crs = wui.crs
    local_wui = wui.cx[xmin:xmax, ymin:ymax]
    wui_outline = gpd.GeoSeries(
        [local_wui.geometry.union_all()], crs=crs,
    ).boundary
    county_outline = _county_outline(boundary, crs, bounds)

    width_in, map_w, left = 7.2, 5.35, .04
    map_h = map_w * (ymax - ymin) / (xmax - xmin)
    head, gap, pad = .44, .1, .04
    n_panels = len(panels)
    height_in = pad + n_panels * (map_h + head) + (n_panels - 1) * gap
    key_x, key_w = left + map_w + .16, width_in - (left + map_w + .16) - .02
    fig = plt.figure(figsize=(width_in, height_in))

    def inches(x, y, w, h):
        return [x / width_in, y / height_in, w / width_in, h / height_in]

    rows = []
    for index, (label, points) in enumerate(panels.items()):
        y_map = pad + (n_panels - 1 - index) * (map_h + head + gap)
        ax = fig.add_axes(inches(left, y_map, map_w, map_h))
        connected = points.component_size.ge(2).to_numpy()
        touches = points.mapped_wui_component_size.gt(0).to_numpy()
        ax.scatter(points.x[~connected], points.y[~connected], s=.16,
                   color=OTHER, alpha=.4, linewidths=0, rasterized=True,
                   zorder=1)
        # Outside clusters first; within each group, largest clusters on top.
        for zorder, touch in ((2, False), (3, True)):
            group = points.loc[connected & (touches == touch)].sort_values(
                "component_size",
            )
            ax.scatter(group.x, group.y, c=group.component_size,
                       cmap=palettes[touch], norm=norm, s=.36, alpha=.92,
                       linewidths=0, rasterized=True, zorder=zorder)
        if len(county_outline):
            county_outline.plot(ax=ax, color=INK, linewidth=.55, zorder=11)
        wui_outline.plot(ax=ax, color=WUI_BOUNDARY_COLOR, linewidth=.4,
                         alpha=.85, zorder=10)
        if places:
            _place_labels(ax, places, bounds, crs=crs, halo=True)
        if index == n_panels - 1:
            _scale_bar(ax, bounds)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
        ax.axis("off")

        inside_members = points.loc[connected & touches]
        outside_members = points.loc[connected & ~touches]
        largest_outside = (
            int(outside_members.component_size.max())
            if len(outside_members) else 0
        )
        top = y_map + map_h + head
        fig.text(left / width_in, (top - .06) / height_in,
                 "abcdefgh"[index], ha="left", va="top", fontsize=9,
                 fontweight="bold")
        fig.text((left + .19) / width_in, (top - .07) / height_in, label,
                 ha="left", va="top", fontsize=8, fontweight="bold")
        fig.text(
            (left + .19) / width_in, (top - .26) / height_in,
            f"{len(inside_members):,} buildings in clusters touching the "
            f"mapped WUI · {len(outside_members):,} in "
            f"{outside_members.component_id.nunique():,} clusters entirely "
            f"outside it (largest {largest_outside:,})",
            ha="left", va="top", fontsize=6.3, color="#555555",
        )
        rows.append({
            "method": label,
            "buildings_in_view": len(points),
            "isolated_buildings": int((~connected).sum()),
            "buildings_in_wui_touching_clusters": len(inside_members),
            "wui_touching_clusters": inside_members.component_id.nunique(),
            "buildings_in_outside_wui_clusters": len(outside_members),
            "outside_wui_clusters": outside_members.component_id.nunique(),
            "largest_outside_wui_cluster": largest_outside,
            "largest_wui_touching_cluster": (
                int(inside_members.component_size.max())
                if len(inside_members) else 0
            ),
        })

    # One key for all panels, centered in the right-hand column.
    bar_h = min(1.45, .55 * height_in)
    key_top = height_in / 2 + (bar_h + .87) / 2
    fig.text(key_x / width_in, key_top / height_in, "Buildings in cluster",
             ha="left", va="top", fontsize=6.5, fontweight="bold", color=INK)
    bar_top = key_top - .42
    ticks = 2 ** np.arange(1, int(np.log2(size_scale_max)) + 1, 2)
    if ticks[-1] != size_scale_max:
        ticks = np.append(ticks, size_scale_max)
    for offset, touch, caption in (
        (0, True, "touches\nmapped WUI"),
        (.72, False, "outside\nmapped WUI"),
    ):
        fig.text((key_x + offset) / width_in, (key_top - .16) / height_in,
                 caption, ha="left", va="top", fontsize=6.1, color=INK,
                 linespacing=1.1)
        cax = fig.add_axes(inches(key_x + offset, bar_top - bar_h, .09, bar_h))
        bar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=palettes[touch]),
            cax=cax, orientation="vertical",
        )
        bar.set_ticks(ticks)
        bar.set_ticklabels([f"{value:,}" for value in ticks])
        bar.ax.minorticks_off()
        bar.ax.tick_params(labelsize=5.8, length=2, pad=1.5)
        bar.outline.set_linewidth(.4)
    legend_h = .45
    legend_ax = fig.add_axes(inches(
        key_x, bar_top - bar_h - .12 - legend_h, key_w, legend_h,
    ))
    legend_ax.set(xlim=(0, key_w), ylim=(0, legend_h))
    legend_ax.axis("off")
    _draw_key(legend_ax, [
        ("patch", OTHER, "Isolated building"),
        ("line", WUI_BOUNDARY_COLOR, "Mapped WUI boundary"),
        ("line", INK, "County boundary"),
    ])

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, facecolor="white")
    return fig, pd.DataFrame(rows)


def plot_threshold_persistence_panel(
    persistence_map: gpd.GeoDataFrame,
    sensitivity_summary: pd.DataFrame,
    wui: gpd.GeoDataFrame,
    output_stem: Path,
    *,
    places: dict[str, tuple[float, float]] | None = None,
):
    """Map the highest P-threshold retaining Interface-spanning membership."""
    bounds = persistence_map.total_bounds
    xmin, ymin, xmax, ymax = bounds
    width, height = xmax - xmin, ymax - ymin
    red_cmap = LinearSegmentedColormap.from_list(
        "interface_persistence_reds",
        plt.get_cmap("Reds")(np.linspace(.32, .92, 256)),
    )
    probabilities = sensitivity_summary.probability.to_numpy(float)
    norm = plt.Normalize(100 * probabilities.min(), 100 * probabilities.max())
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), sharex=True, sharey=True)
    class_lookup = {
        "Influence Zone": "Influence", "Intermix": "Intermix",
        "Interface": "Interface",
    }
    local_wui = wui.to_crs(persistence_map.crs).cx[xmin:xmax, ymin:ymax]
    place_transformer = Transformer.from_crs(
        4326, persistence_map.crs, always_xy=True,
    )
    for ax, method in zip(axes, ("top2", "eeat", "eeat_top2_fmax")):
        ax.scatter(
            persistence_map.x, persistence_map.y, s=.18,
            color=OTHER, alpha=.30, linewidths=0, rasterized=True, zorder=1,
        )
        # Use the WUI interiors as a lightly tinted background above the gray
        # building field but below the persistence layer.
        for index, (source_name, display_name) in enumerate(class_lookup.items()):
            frame = local_wui.loc[local_wui.WUI_DESC.eq(source_name)]
            if len(frame):
                frame.plot(
                    ax=ax,
                    facecolor=to_rgba(CATEGORY_COLORS[display_name], .10),
                    edgecolor="none", zorder=2 + .1 * index,
                )
        # Structures within the mapped WUI retain their categorical zone
        # colors. Persistence red is reserved strictly for component members
        # beyond the existing WUI boundary.
        for source_name, display_name in class_lookup.items():
            in_zone = persistence_map.wui_class.eq(source_name)
            ax.scatter(
                persistence_map.loc[in_zone, "x"],
                persistence_map.loc[in_zone, "y"],
                s=.34, color=CATEGORY_COLORS[display_name], alpha=.76,
                linewidths=0, rasterized=True, zorder=2.5,
            )
        values = persistence_map[f"{method}_max_probability"]
        selected = (
            values.notna()
            & persistence_map.wui_class.eq("Outside mapped WUI")
        )
        ax.scatter(
            persistence_map.loc[selected, "x"],
            persistence_map.loc[selected, "y"],
            c=100 * values.loc[selected], cmap=red_cmap, norm=norm,
            s=.42, alpha=.90, linewidths=0, rasterized=True, zorder=3,
        )
        # Keep the official WUI boundaries above the persistence layer.
        for zorder, (source_name, display_name) in enumerate(class_lookup.items(), 5):
            frame = local_wui.loc[local_wui.WUI_DESC.eq(source_name)]
            if len(frame):
                frame.boundary.plot(
                    ax=ax,
                    color=CATEGORY_COLORS[display_name],
                    linewidth=.38, zorder=zorder,
                )
        if places:
            for name, (lon, lat) in places.items():
                x, y = place_transformer.transform(lon, lat)
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    ax.scatter(
                        [x], [y], s=4, facecolor="white", edgecolor=INK,
                        linewidth=.5, zorder=9,
                    )
                    ax.annotate(
                        name, (x, y), xytext=(3, 3), textcoords="offset points",
                        fontsize=5.8, color=INK, zorder=9,
                    )
        crs_units = persistence_map.crs.axis_info[0].unit_name.casefold()
        scale_length = 10_000 / .3048 if "foot" in crs_units else 10_000
        x0, y0 = xmin + .06 * width, ymin + .055 * height
        ax.plot([x0, x0 + scale_length], [y0, y0], color=INK, lw=1, zorder=10)
        ax.plot(
            [x0, x0], [y0 - .008 * height, y0 + .008 * height],
            color=INK, lw=.8, zorder=10,
        )
        ax.plot(
            [x0 + scale_length, x0 + scale_length],
            [y0 - .008 * height, y0 + .008 * height],
            color=INK, lw=.8, zorder=10,
        )
        ax.text(
            x0 + scale_length / 2, y0 + .015 * height, "10 km",
            ha="center", va="bottom", fontsize=6,
        )
        ax.text(
            .96, .96, "N\n↑", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.5,
        )
        p50 = sensitivity_summary.loc[
            sensitivity_summary.method.eq(method)
            & np.isclose(sensitivity_summary.probability, .5)
        ].iloc[0]
        method_label = METHOD_LABELS[method]
        if method == "top2" and pd.notna(p50.get("edge_floor_fraction")):
            method_label = (
                f"Cumulative Top-2 · α="
                f"{float(p50.edge_floor_fraction):.2f}"
            )
        ax.set_title(
            f"{method_label}\n"
            f"P50: {int(p50.largest_interface_spanning_component):,}-building "
            f"largest component · {p50.share_in_interface_spanning_components:.1%} "
            "of county buildings",
            loc="left", fontsize=7.8, fontweight="bold", pad=4,
        )
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
        ax.axis("off")

    context_handles = [Patch(facecolor=OTHER, edgecolor="none", label="Other buildings")]
    context_handles += [
        Patch(facecolor=CATEGORY_COLORS[name], edgecolor="none", label=name)
        for name in ("Influence", "Intermix", "Interface")
    ]
    fig.legend(
        handles=context_handles, loc="lower center", ncol=4, frameon=False,
        fontsize=6.3, bbox_to_anchor=(.34, .012),
    )
    cbar_ax = fig.add_axes([.67, .052, .27, .018])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=red_cmap),
        cax=cbar_ax, orientation="horizontal",
    )
    ticks = 100 * np.sort(np.unique(probabilities))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"P{int(value)}" for value in ticks])
    cbar.set_label(
        "Highest destruction-probability threshold retaining "
        "Interface-spanning membership outside mapped WUI",
        fontsize=6.3,
    )
    cbar.ax.tick_params(labelsize=5.8, length=2)
    fig.suptitle(
        "Los Angeles County — Santa Monica Mountains threshold persistence",
        x=.015, ha="left", y=.99, fontsize=10, fontweight="bold",
    )
    fig.subplots_adjust(left=.015, right=.99, top=.87, bottom=.14, wspace=.035)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".png"), dpi=600,
        bbox_inches="tight", facecolor="white",
    )
    return fig


def plot_eeat_interconnectivity_persistence_map(
    persistence_map: gpd.GeoDataFrame,
    wui: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    output_stem: Path,
    *,
    probabilities: tuple[float, ...] = (.25, .50, .75),
    minimum_component_size: int = 10,
    places: dict[str, tuple[float, float]] | None = None,
):
    """Map only outside-WUI EEAT Interconnectivity across P thresholds.

    Structures inside any mapped WUI class are neutral gray. Buildings beyond
    mapped WUI are shown only when their countywide EEAT component touches the
    Interface and meets ``minimum_component_size``. The three official WUI
    class boundaries remain separate, colored reference lines.
    """
    probabilities = tuple(float(value) for value in probabilities)
    if tuple(sorted(set(probabilities))) != probabilities:
        raise ValueError("probabilities must be unique and increasing")
    probability_column = "eeat_max_probability"
    size_column = "eeat_component_size_at_max_probability"
    required = {"x", "y", "wui_class", probability_column, size_column}
    missing = required.difference(persistence_map.columns)
    if missing:
        raise ValueError(f"persistence_map is missing {sorted(missing)}")

    xmin, ymin, xmax, ymax = bounds
    shown = persistence_map.loc[
        persistence_map.x.between(xmin, xmax)
        & persistence_map.y.between(ymin, ymax)
    ].copy()
    mapped = shown.wui_class.ne("Outside mapped WUI")
    outside = ~mapped
    retained = shown[probability_column].to_numpy(float)
    component_size = shown[size_column].fillna(0).to_numpy(float)
    level = np.full(len(shown), -1, dtype=int)
    for index, probability in enumerate(probabilities):
        level[np.isclose(retained, probability, equal_nan=False)] = index
    unexpected = np.isfinite(retained) & (level < 0)
    if unexpected.any():
        values = np.unique(retained[unexpected])
        raise ValueError(f"Persistence values fall outside {probabilities}: {values}")
    interconnectivity = (
        outside.to_numpy() & (level >= 0)
        & (component_size >= minimum_component_size)
    )

    red_colors = plt.get_cmap("Reds")(np.linspace(.38, .90, len(probabilities)))
    class_lookup = {
        "Influence Zone": "Influence",
        "Intermix": "Intermix",
        "Interface": "Interface",
    }
    local_wui = wui.to_crs(persistence_map.crs).cx[xmin:xmax, ymin:ymax]
    county_outline = _county_outline(boundary, persistence_map.crs, bounds)

    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.scatter(
        shown.loc[mapped, "x"], shown.loc[mapped, "y"],
        s=.26, color="#C9CBC8", alpha=.62, linewidths=0,
        rasterized=True, zorder=1,
    )
    # Least persistent points first; the darkest, most robust level sits above.
    for index, (probability, color) in enumerate(zip(probabilities, red_colors)):
        mask = interconnectivity & (level == index)
        ax.scatter(
            shown.loc[mask, "x"], shown.loc[mask, "y"],
            s=.42, color=color, alpha=.92, linewidths=0,
            rasterized=True, zorder=2 + .1 * index,
        )
    for zorder, (source_name, display_name) in enumerate(class_lookup.items(), 5):
        frame = local_wui.loc[local_wui.WUI_DESC.eq(source_name)]
        if len(frame):
            frame.boundary.plot(
                ax=ax, color=CATEGORY_COLORS[display_name],
                linewidth=.70, zorder=zorder,
            )
    if len(county_outline):
        county_outline.plot(ax=ax, color=INK, linewidth=.65, zorder=9)
    if places:
        _place_labels(ax, places, bounds, crs=persistence_map.crs, halo=True)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
    ax.axis("off")
    _scale_bar(ax, bounds)

    counts = {
        probability: int((interconnectivity & (level >= index)).sum())
        for index, probability in enumerate(probabilities)
    }
    ax.set_title(
        "Outside-WUI Interconnectivity retained across P25, P50, and P75\n"
        f"EEAT components ≥{minimum_component_size} buildings · "
        + " · ".join(
            f"{counts[probability]:,} at P{100 * probability:.0f}"
            for probability in probabilities
        ),
        loc="left", fontsize=8.2, fontweight="bold", pad=5,
    )
    fig.suptitle(
        "Los Angeles County — Santa Monica Mountains — EEAT threshold persistence",
        x=.015, ha="left", y=.985, fontsize=10, fontweight="bold",
    )
    handles = [
        Patch(facecolor="#C9CBC8", edgecolor="none", label="Buildings inside mapped WUI"),
        *[
            Line2D([], [], color=CATEGORY_COLORS[display_name], lw=1.2,
                   label=f"{display_name} boundary")
            for display_name in ("Influence", "Intermix", "Interface")
        ],
        *[
            Patch(facecolor=color, edgecolor="none",
                  label=f"Interconnectivity: highest retained P{100 * probability:.0f}")
            for probability, color in zip(probabilities, red_colors)
        ],
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(.5, .015),
        ncol=4, frameon=False, fontsize=6.2,
    )
    fig.subplots_adjust(left=.015, right=.985, top=.88, bottom=.15)

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".png"), dpi=600,
        bbox_inches="tight", facecolor="white",
    )
    summary = {
        "method": "EEAT",
        "minimum_component_size": minimum_component_size,
        "buildings_inside_mapped_wui_in_view": int(mapped.sum()),
        "outside_wui_buildings_hidden": int((outside.to_numpy() & ~interconnectivity).sum()),
    }
    summary.update({
        f"interconnectivity_retained_at_p{100 * probability:.0f}": count
        for probability, count in counts.items()
    })
    return fig, pd.DataFrame([summary])


def plot_threshold_sensitivity_curves(
    sensitivity_summary: pd.DataFrame, output_stem: Path,
):
    """Plot Interface-spanning responses across calibrated P thresholds."""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    colors = {
        "top2": "#B24C63",
        "eeat": "#3B78A3",
        "eeat_top2_fmax": "#7A5195",
    }
    labels = {method: METHOD_LABELS[method] for method in colors}
    top2_floors = sensitivity_summary.loc[
        sensitivity_summary.method.eq("top2"), "edge_floor_fraction"
    ].dropna().unique()
    if len(top2_floors) == 1:
        labels["top2"] = f"Cumulative Top-2 · α={float(top2_floors[0]):.2f}"
    specifications = (
        ("share_in_interface_spanning_components", "Share of county buildings", False),
        ("largest_interface_spanning_component", "Largest component (buildings)", True),
        ("interface_spanning_components", "Interface-spanning components", True),
    )
    for ax, (column, ylabel, log_scale) in zip(axes, specifications):
        for method in ("top2", "eeat", "eeat_top2_fmax"):
            shown = sensitivity_summary.loc[
                sensitivity_summary.method.eq(method)
            ].sort_values("probability")
            ax.plot(
                100 * shown.probability, shown[column], marker="o", ms=3.4,
                lw=1.3, color=colors[method], label=labels[method],
            )
        if log_scale:
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set(xlabel="Destruction-probability equivalent", ylabel=ylabel)
        ax.set_xticks(100 * np.sort(sensitivity_summary.probability.unique()))
        ax.set_xticklabels([
            f"P{int(value)}"
            for value in 100 * np.sort(sensitivity_summary.probability.unique())
        ])
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(alpha=.18, linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".png"), dpi=600,
        bbox_inches="tight", facecolor="white",
    )
    return fig
