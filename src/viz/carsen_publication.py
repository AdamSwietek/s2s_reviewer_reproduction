"""Publication composites for the statewide California SEN analysis."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def save_png_as_pdf(png_path, pdf_path, dpi=600):
    """Embed the verified PNG as one PDF image without mixed-mode transforms."""
    image = plt.imread(png_path)
    height, width = image.shape[:2]
    figure = plt.figure(
        figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False,
    )
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(image, interpolation="none")
    axis.set_axis_off()
    figure.savefig(pdf_path, format="pdf", dpi=dpi, facecolor="white")
    plt.close(figure)


def _compact_count_axis(
    ax, summary, class_column, class_order, colors, display_labels,
    network_bin_order,
):
    """Draw compact grouped counts over common SEN-size bins."""
    x = np.arange(len(network_bin_order))
    group_width = 0.84
    bar_width = group_width / len(class_order)
    for index, category in enumerate(class_order):
        values = (
            summary.loc[summary[class_column].eq(category)]
            .set_index("network_size_bin")
            .reindex(network_bin_order)
            .building_count.fillna(0).to_numpy()
        )
        positions = x - group_width / 2 + bar_width * (index + 0.5)
        ax.bar(
            positions, np.where(values > 0, values, np.nan),
            width=bar_width * 0.92, color=colors[category],
            label=display_labels.get(category, category), linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_ylim(10, 5_000_000)
    ax.set_xticks(x, network_bin_order)
    ax.set_xlabel("SEN size (buildings)", labelpad=2)
    ax.set_ylabel("Structures", labelpad=2)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: (
            f"{value / 1_000_000:g}M" if value >= 1_000_000
            else f"{value / 1_000:g}k" if value >= 1_000
            else f"{value:g}"
        )
    ))
    ax.grid(axis="y", which="major", color="#D7D7D3", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=5.5, pad=2)
    ax.tick_params(axis="y", labelsize=5.5)


def plot_california_connectivity_context(
    *, clusters, state_boundary, state_ocean, state_lakes,
    water_color, outline_color, network_cmap, network_norm, network_bounds,
    gridsize_from_extent, severity_counts, severity_order, severity_colors,
    zone_counts, zone_order, zone_colors, outside_zone, network_bin_order,
    threshold, alpha,
):
    """Combine the statewide SEN map with hazard- and WUI-stratified counts."""
    fig = plt.figure(figsize=(7.4, 5.15))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.08, 1.55), height_ratios=(1, 1),
        left=0.035, right=0.985, top=0.855, bottom=0.105,
        wspace=0.22, hspace=0.53,
    )
    ax_map = fig.add_subplot(grid[:, 0])
    ax_hazard = fig.add_subplot(grid[0, 1])
    ax_wui = fig.add_subplot(grid[1, 1])

    xmin, ymin, xmax, ymax = state_boundary.total_bounds
    for water in (state_ocean, state_lakes):
        if not water.empty:
            water.plot(
                ax=ax_map, color=water_color, edgecolor="none",
                linewidth=0, zorder=0,
            )
    map_hex = ax_map.hexbin(
        clusters.x.to_numpy(), clusters.y.to_numpy(),
        C=clusters.component_size.to_numpy(), reduce_C_function=np.max,
        gridsize=gridsize_from_extent(xmin, xmax), mincnt=1,
        extent=(xmin, xmax, ymin, ymax), cmap=network_cmap,
        norm=network_norm, linewidths=0, edgecolors="none",
        rasterized=True, zorder=2,
    )
    state_boundary.boundary.plot(
        ax=ax_map, color=outline_color, linewidth=0.42, zorder=4,
    )
    ax_map.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
    ax_map.set_anchor("N")
    ax_map.set_axis_off()
    ax_map.set_title(
        "a   California-wide connectivity", loc="left",
        fontsize=8, fontweight="bold", pad=5,
    )
    cax = inset_axes(
        ax_map, width="94%", height="2.7%", loc="lower center",
        bbox_to_anchor=(0, -0.085, 1, 1),
        bbox_transform=ax_map.transAxes, borderpad=0,
    )
    cbar = fig.colorbar(
        map_hex, cax=cax, orientation="horizontal",
        boundaries=network_bounds, ticks=network_bounds,
        spacing="uniform", drawedges=False,
    )
    cbar.set_ticklabels([f"{int(value):,}" for value in network_bounds])
    cbar.ax.tick_params(length=1.8, width=0.45, pad=1.2, labelsize=5.3)
    cbar.outline.set_linewidth(0.45)
    cbar.set_label(
        "Maximum SEN size per hexagon (buildings)", fontsize=5.8, labelpad=2,
    )

    _compact_count_axis(
        ax_hazard, severity_counts, "severity", severity_order,
        severity_colors, {}, network_bin_order,
    )
    ax_hazard.set_title(
        "b   Connectivity by fire-hazard severity", loc="left",
        fontsize=8, fontweight="bold", pad=23,
    )
    ax_hazard.legend(
        loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False,
        ncol=3, fontsize=5.35, handlelength=1.0,
        columnspacing=0.85, borderaxespad=0,
    )

    wui_labels = {
        "Influence": "Influence", "Intermix": "Intermix",
        "Interface": "Interface", outside_zone: "Outside mapped WUI",
        "Interconnectivity": "Outside WUI linked to Interface",
    }
    _compact_count_axis(
        ax_wui, zone_counts, "analysis_zone", zone_order,
        zone_colors, wui_labels, network_bin_order,
    )
    ax_wui.set_title(
        "c   Connectivity by WUI context", loc="left",
        fontsize=8, fontweight="bold", pad=23,
    )
    ax_wui.legend(
        loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False,
        ncol=3, fontsize=5.35, handlelength=1.0,
        columnspacing=0.85, borderaxespad=0,
    )

    fig.suptitle(
        "California building connectivity across fire-hazard and WUI contexts",
        x=0.035, y=0.975, ha="left", fontsize=9.2, fontweight="bold",
    )
    fig.text(
        0.035, 0.935,
        (f"2D cumulative Top-2 · midpoint-average $F^*={threshold:.3f}$ "
         f"· $\\alpha={alpha:g}$"),
        ha="left", va="top", fontsize=6.5, color="#555555",
    )
    return fig
