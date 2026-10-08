"""Main Figure 1 composite: geometric coupling and structural fragility.

The figure is organised as a compact conceptual sequence followed by two
labelled empirical bands:

    a-d   GEOMETRIC EXPOSURE DEFINITION    (schematic, quiet)
    e-f   DISTANCE-EXPOSURE RELATIONSHIP   (empirical decay)
    g-h   EXPOSURE-DAMAGE RELATIONSHIP     (response, the result)

Every panel is rendered from frozen packaged source data; no model is refit
here. The conceptual row is drawn analytically rather than restored from the
rendered scene asset, so the composite no longer depends on the licensed mesh
or the patch-level LOS table.

Colour language (one meaning per hue):
    black / grey   outcome contrast, destroyed vs surviving (panels a-f)
    orange / blue  fire identity, Eaton vs Palisades (panels g-h)

The a-d row is deliberately sparse: the same visual vocabulary progresses
from distance definition to pairwise and cumulative exposure without a large
section heading competing with the panel titles.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.tri import Triangulation
from matplotlib.ticker import FixedFormatter, FixedLocator, LogFormatterMathtext
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from src.viz.style import FIRE_COLORS


# ---------------------------------------------------------------- palette ---
INK = "#1A1A1A"        # destroyed structure / emitting source
GREY = "#6B7075"       # surviving structure / receiving target
FAINT = "#ECEDEE"      # structures irrelevant to the illustrated measurement
FAINT_EDGE = "#DCDDDF"
# The conceptual row is explanatory, not evidentiary, so its fills sit one
# step softer than the analytic rows to keep the reader's eye on e-h.
SCHEMA_INK = "#33383D"
SCHEMA_GREY = "#8C9298"
RULE = "#C9CBCD"       # band hairlines
LABEL = "#5A5E63"      # band label text
GRID = "#E7E8EA"

MM = 1 / 25.4
FIG_W = 180 * MM       # Nature two-column width
FIG_H = 155 * MM

# Type sizes are set here rather than inherited so the composite is not
# sensitive to whichever notebook last called apply_style().
FIG_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    # Without a custom mathtext set, every $...$ label renders in DejaVu Sans
    # while the body text is Arial, which ships a two-typeface figure.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "it",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "legend.fontsize": 7.0,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

TITLE_SIZE = 8.5

F_STAR = r"$F^{*}$"
COUPLING_LABEL = "Realized geometric coupling, " + F_STAR


def _fx(inches: float) -> float:
    return inches / FIG_W


def _fy(inches: float) -> float:
    return inches / FIG_H


def _panel_title(ax, letter, text, *, size=TITLE_SIZE, rise=0.055,
                 gap_pt=8.5):
    """Bold panel letter, then a regular-weight descriptor beside it.

    Both are offset in inches rather than axes fractions so the spacing is
    identical in narrow schematic panels and wide analytic ones.
    """
    trans = ax.transAxes + ScaledTranslation(
        0.0, rise, ax.figure.dpi_scale_trans
    )
    add_text = ax.text2D if hasattr(ax, "text2D") else ax.text
    add_text(0.0, 1.0, letter, transform=trans, ha="left", va="baseline",
             fontsize=size, fontweight="bold", color=INK, clip_on=False)
    shifted = ax.transAxes + ScaledTranslation(
        gap_pt / 72, rise, ax.figure.dpi_scale_trans
    )
    add_text(0.0, 1.0, text, transform=shifted, ha="left", va="baseline",
             fontsize=size, color=INK, clip_on=False)


# ------------------------------------------------------- schematic helpers ---
def _poly(cx, cy, w, h, rot=0.0, cut=0.0):
    """Axis-aligned footprint, optionally with one clipped corner, rotated."""
    x, y = w / 2, h / 2
    if cut > 0:
        pts = np.array([
            [-x, -y], [x, -y], [x, y - cut * h],
            [x - cut * w, y - cut * h], [x - cut * w, y], [-x, y],
        ])
    else:
        pts = np.array([[-x, -y], [x, -y], [x, y], [-x, y]])
    theta = np.deg2rad(rot)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return pts @ rotation.T + np.array([cx, cy])


def _draw(ax, pts, fc, ec="none", lw=0.0, z=2, alpha=1.0):
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec,
                         lw=lw, zorder=z, alpha=alpha, joinstyle="round"))


def _centroid(pts):
    return pts.mean(axis=0)


def _facade(pts, toward, n=5, inset=0.15):
    """n points spread along the polygon edge that faces ``toward``."""
    toward = np.asarray(toward, float)
    edges = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    start, end = min(
        edges, key=lambda e: np.linalg.norm((e[0] + e[1]) / 2 - toward)
    )
    t = np.linspace(inset, 1 - inset, n)
    return start + (end - start) * t[:, None]


def _densify(pts, n=160):
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n)
    return np.column_stack([np.interp(s, cum, closed[:, 0]),
                            np.interp(s, cum, closed[:, 1])])


def _nearest_points(a, b):
    """Closest point pair between two footprints, by dense boundary search."""
    pa, pb = _densify(a), _densify(b)
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return pa[i], pb[j]


def _dimension(ax, p0, p1, text, offset=(0, 0), ha="center", va="bottom",
               fontsize=6.4, z=8, end_ticks=0.0):
    """Black measurement line with arrowheads at both ends, plus its label.

    A white casing is drawn as a separate underlay rather than as a stroke
    path effect, which would hollow out the small arrowheads. ``end_ticks``
    adds perpendicular witness marks, the drawing convention that makes a
    surface-to-surface span read as a measurement rather than a connector.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=2.3,
            solid_capstyle="round", zorder=z - 0.5)
    ax.annotate(
        "", xy=tuple(p1), xytext=tuple(p0),
        arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.8,
                        mutation_scale=5.5, shrinkA=0, shrinkB=0),
        zorder=z,
    )
    if end_ticks > 0:
        span = p1 - p0
        normal = np.array([-span[1], span[0]])
        normal = normal / max(np.linalg.norm(normal), 1e-9) * end_ticks
        for point in (p0, p1):
            ax.plot(*np.column_stack([point - normal, point + normal]),
                    color=INK, lw=0.8, solid_capstyle="butt", zorder=z)
    mid = (p0 + p1) / 2 + np.asarray(offset)
    ax.text(*mid, text, ha=ha, va=va, fontsize=fontsize, color=INK,
            zorder=z + 1,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


def _rays(ax, source, target, n_src=5, n_tgt=3, alpha=0.22, lw=0.32):
    """Thin semi-transparent exchange field between two facing facades."""
    s_pts = _facade(source, _centroid(target), n=n_src)
    t_pts = _facade(target, _centroid(source), n=n_tgt)
    for s in s_pts:
        for t in t_pts:
            ax.plot([s[0], t[0]], [s[1], t[1]], color=INK, lw=lw,
                    alpha=alpha, zorder=3, solid_capstyle="round")
    return s_pts, t_pts


def _point_to_segment(point, start, end):
    start, end = np.asarray(start, float), np.asarray(end, float)
    span = end - start
    t = np.clip(np.dot(np.asarray(point, float) - start, span)
                / max(np.dot(span, span), 1e-9), 0, 1)
    return float(np.linalg.norm(np.asarray(point, float) - (start + t * span)))


def _context(ax, rng, n, xlim, ylim, avoid=(), corridors=(), pad=15,
             corridor_pad=13, self_pad=17):
    """Very faint surrounding fabric.

    Candidates are rejected if they crowd an illustrated structure, sit in the
    corridor whose geometry is being explained, or pile onto each other; the
    row should read as context, not as clutter.
    """
    placed: list[np.ndarray] = []
    guard = 0
    while len(placed) < n and guard < 1200:
        guard += 1
        centre = np.array([rng.uniform(xlim[0] + 8, xlim[1] - 8),
                           rng.uniform(ylim[0] + 7, ylim[1] - 7)])
        if any(np.linalg.norm(_centroid(a) - centre) < pad for a in avoid):
            continue
        if any(_point_to_segment(centre, *segment) < corridor_pad
               for segment in corridors):
            continue
        if any(np.linalg.norm(other - centre) < self_pad for other in placed):
            continue
        pts = _poly(*centre, rng.uniform(12, 17), rng.uniform(10, 14),
                    rot=rng.uniform(-14, 14), cut=rng.choice([0.0, 0.32]))
        _draw(ax, pts, FAINT, FAINT_EDGE, 0.4, z=1)
        placed.append(centre)


def _schematic_axes(ax, x_units, y_units):
    ax.set_xlim(0, x_units)
    ax.set_ylim(0, y_units)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------- conceptual panels ---
# The pair geometry is shared by a-c so the reader watches the same two
# structures gain information: centroid separation, then surface separation,
# then the resolved exchange that separation stands in for.
SOURCE = _poly(28, 30, 23, 18, rot=-7, cut=0.30)
TARGET = _poly(72, 22, 22, 17, rot=6)
PAIR_CORRIDOR = ((_centroid(SOURCE), _centroid(TARGET)),)


def _plan_pair_axes(ax):
    ax.set_xlim(7, 93)
    ax.set_ylim(7, 43)
    ax.set_aspect("equal")
    ax.axis("off")


def _distance_segment(ax, p0, p1, text, *, linestyle="-", offset=(0, 0),
                      end_ticks=0.0):
    """Non-directional measurement line with optional witness marks."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax.plot(
        [p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=2.8,
        solid_capstyle="round", zorder=8,
    )
    ax.plot(
        [p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=1.15,
        ls=linestyle, solid_capstyle="round", zorder=9,
    )
    if end_ticks > 0:
        span = p1 - p0
        normal = np.array([-span[1], span[0]])
        normal = normal / max(np.linalg.norm(normal), 1e-9) * end_ticks
        for point in (p0, p1):
            ax.plot(*np.column_stack([point - normal, point + normal]),
                    color=INK, lw=1.0, solid_capstyle="butt", zorder=10)
    mid = (p0 + p1) / 2 + np.asarray(offset)
    ax.text(
        *mid, text, ha="center", va="bottom", fontsize=8.2, color=INK,
        zorder=11,
        path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
    )


def panel_a_centroid(ax, rng, y_units):
    _plan_pair_axes(ax)
    _context(ax, rng, 2, (7, 93), (7, 43), avoid=(SOURCE, TARGET),
             corridors=PAIR_CORRIDOR, pad=18, corridor_pad=14)
    _draw(ax, SOURCE, SCHEMA_INK, z=2)
    _draw(ax, TARGET, SCHEMA_GREY, z=2)
    cs, ct = _centroid(SOURCE), _centroid(TARGET)
    for point in (cs, ct):
        ax.plot(*point, "o", ms=4.0, mfc="white", mec=INK, mew=.75,
                zorder=12)
    _distance_segment(
        ax, cs, ct, r"$d_{CC}$", linestyle=(0, (3.0, 2.0)),
        offset=(0, 1.8),
    )


def panel_b_surface(ax, rng, y_units):
    _plan_pair_axes(ax)
    _context(ax, rng, 2, (7, 93), (7, 43), avoid=(SOURCE, TARGET),
             corridors=PAIR_CORRIDOR, pad=18, corridor_pad=14)
    _draw(ax, SOURCE, SCHEMA_INK, z=2)
    _draw(ax, TARGET, SCHEMA_GREY, z=2)
    ps, pt = _nearest_points(SOURCE, TARGET)
    source_face = _facade(SOURCE, _centroid(TARGET), n=2, inset=0)
    target_face = _facade(TARGET, _centroid(SOURCE), n=2, inset=0)
    for face in (source_face, target_face):
        ax.plot(face[:, 0], face[:, 1], color=EXCHANGE, lw=2.7,
                solid_capstyle="round", zorder=7)
    _distance_segment(
        ax, ps, pt, r"$d_{SS}$", offset=(0, 1.8), end_ticks=2.6,
    )


EXCHANGE = "#675481"


def _prism_faces(footprint, height):
    """Return roof and wall faces for a simple extruded footprint."""
    bottom = [(x, y, 0.0) for x, y in footprint]
    top = [(x, y, height) for x, y in footprint]
    walls = [
        [bottom[index], bottom[(index + 1) % len(bottom)],
         top[(index + 1) % len(top)], top[index]]
        for index in range(len(footprint))
    ]
    return top, walls


def _draw_prism3d(ax, footprint, height, facecolor, *, edgecolor="#45494D",
                  alpha=1.0, linewidth=.45, zorder=3):
    roof, walls = _prism_faces(footprint, height)
    wall_colour = tuple(np.asarray(to_rgb(facecolor)) * .82)
    collection = Poly3DCollection(
        walls + [roof],
        facecolors=[wall_colour] * len(walls) + [facecolor],
        edgecolors=edgecolor, linewidths=linewidth, alpha=alpha,
        zorder=zorder,
    )
    ax.add_collection3d(collection)


def _setup_3d(ax):
    ax.set_xlim(5, 95)
    ax.set_ylim(2, 47)
    ax.set_zlim(-2.2, 32)
    ax.set_box_aspect((1.9, 1.0, .62), zoom=1.42)
    ax.set_proj_type("ortho")
    ax.view_init(elev=27, azim=-58)
    ax.set_axis_off()
    ax.patch.set_alpha(0)


def _terrain_3d(ax):
    """Quiet triangulated ground surface for spatial and slope context."""
    x_grid, y_grid = np.meshgrid(
        np.linspace(6, 94, 10), np.linspace(3, 46, 7),
    )
    z_grid = (
        -1.25
        + .34 * np.sin(x_grid / 17.0)
        + .22 * np.cos(y_grid / 9.0)
        + .10 * np.sin((x_grid + y_grid) / 12.0)
    )
    x_flat, y_flat, z_flat = (
        x_grid.ravel(), y_grid.ravel(), z_grid.ravel()
    )
    triangulation = Triangulation(x_flat, y_flat)
    edges = set()
    for triangle in triangulation.triangles:
        for start, end in zip(triangle, np.roll(triangle, -1)):
            edges.add(tuple(sorted((int(start), int(end)))))
    segments = [
        [(x_flat[start], y_flat[start], z_flat[start]),
         (x_flat[end], y_flat[end], z_flat[end])]
        for start, end in edges
    ]
    ax.add_collection3d(Line3DCollection(
        segments, colors="#BFC7BC", linewidths=.32, alpha=.72, zorder=0,
    ))


def _context_3d(ax):
    context = (
        (_poly(11, 42, 14, 10, rot=-8), 10),
        (_poly(89, 39, 14, 10, rot=-5), 9),
    )
    for footprint, height in context:
        _draw_prism3d(
            ax, footprint, height, "#F2F3F3", edgecolor="#C8CCCE",
            alpha=.34, linewidth=.35, zorder=1,
        )


def _surface_exchange_3d(ax, source, source_height, target, target_height,
                         *, density=3, alpha=.82):
    """Draw exchange paths between sampled points on opposing 3D facades."""
    source_edge = _facade(source, _centroid(target), n=density, inset=.12)
    target_edge = _facade(target, _centroid(source), n=density, inset=.12)
    source_z = np.linspace(.28, .72, 2) * source_height
    target_z = np.linspace(.28, .72, 2) * target_height
    for point in source_edge:
        ax.plot(
            [point[0], point[0]], [point[1], point[1]], [0, source_height],
            color=EXCHANGE, lw=.72, alpha=.90, zorder=6,
        )
    for point in target_edge:
        ax.plot(
            [point[0], point[0]], [point[1], point[1]], [0, target_height],
            color=EXCHANGE, lw=.72, alpha=.90, zorder=6,
        )
    for level, (z_source, z_target) in enumerate(zip(source_z, target_z)):
        for index, source_point in enumerate(source_edge):
            target_point = target_edge[(index + level) % len(target_edge)]
            ax.plot(
                [source_point[0], target_point[0]],
                [source_point[1], target_point[1]],
                [z_source, z_target], color=EXCHANGE, lw=.82,
                alpha=alpha, zorder=5,
            )


def panel_c_pairwise(ax, rng, y_units):
    _setup_3d(ax)
    _terrain_3d(ax)
    _context_3d(ax)
    _draw_prism3d(ax, SOURCE, 25, SCHEMA_INK, zorder=4)
    _draw_prism3d(ax, TARGET, 18, SCHEMA_GREY, zorder=4)
    _surface_exchange_3d(ax, SOURCE, 25, TARGET, 18, density=3, alpha=.86)


# Cumulative panel: several burned emitters resolved onto one receiver.
D_TARGET = _poly(70, 20, 21, 16, rot=4)
D_SOURCES = (
    _poly(17, 30, 18, 13, rot=-9, cut=0.28),
    _poly(43, 33, 17, 12, rot=8),
    _poly(18, 10, 18, 13, rot=5),
)
D_CORRIDORS = tuple((_centroid(source), _centroid(D_TARGET))
                    for source in D_SOURCES)


def panel_d_cumulative(ax, rng, y_units):
    _setup_3d(ax)
    _terrain_3d(ax)
    _context_3d(ax)
    source_heights = (15, 25, 18)
    for source, height in zip(D_SOURCES, source_heights):
        _draw_prism3d(ax, source, height, SCHEMA_INK, zorder=4)
        _surface_exchange_3d(
            ax, source, height, D_TARGET, 19, density=2, alpha=.72,
        )
    _draw_prism3d(ax, D_TARGET, 19, SCHEMA_GREY, zorder=5)


# -------------------------------------------------------- empirical panels ---
DISTANCE_SETTINGS = {
    "ccd_ft": {
        "title": "Centroid-to-centroid distance",
        "xlabel": r"Nearest destroyed structure, $d_{CC}$ (ft)",
        "ticks": [30, 50, 100, 300, 500, 800],
        "limits": (26, 820),
    },
    "ssd_ft": {
        "title": "Surface-to-surface distance",
        "xlabel": r"Nearest visible destroyed surface, $d_{SS}$ (ft)",
        "ticks": [5, 10, 25, 50, 100, 250, 500, 800],
        "limits": (5, 820),
    },
}

OUTCOME_STYLE = {
    0: {"label": "Survived", "colour": GREY, "line": (0, (3.4, 2.0)),
        "fill": "white"},
    1: {"label": "Destroyed", "colour": INK, "line": "-", "fill": INK},
}


# One marker specification for every empirical bin in e-h, so the points and
# whiskers read the same way in the distance band and the damage band.
BIN_MARKERS = dict(fmt="o", ms=2.7, mew=0.6, elinewidth=0.4, capsize=0, lw=0,
                   alpha=0.72)


def _light_grid(ax, axis="both"):
    ax.grid(True, which="major", axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def plot_distance_panel(ax, source: pd.DataFrame, measure: str, gap: float,
                        *, show_ylabel: bool, letter: str):
    config = DISTANCE_SETTINGS[measure]
    subset = source[source.distance_measure.eq(measure)]
    for destroyed in (0, 1):
        style = OUTCOME_STYLE[destroyed]
        rows = subset[subset.is_destroyed.eq(destroyed)]
        curve = rows[rows.series.eq("lowess")].sort_values("distance_ft")
        points = rows[rows.series.eq("empirical_bin")]
        ax.errorbar(
            points.distance_ft, points.F_star,
            yerr=[points.F_star - points.F_star_lo,
                  points.F_star_hi - points.F_star],
            color=style["colour"], ecolor=style["colour"],
            mfc=style["fill"], zorder=3, **BIN_MARKERS,
        )
        ax.plot(curve.distance_ft, curve.F_star, color=style["colour"],
                ls=style["line"], lw=1.9, zorder=4,
                solid_capstyle="round", dash_capstyle="round")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*config["limits"])
    ax.set_ylim(1e-4, 0.6)
    ax.set_xticks(config["ticks"])
    ax.xaxis.set_major_formatter(
        FixedFormatter([str(value) for value in config["ticks"]])
    )
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    _light_grid(ax)
    ax.set_xlabel(config["xlabel"], labelpad=2.0)
    if show_ylabel:
        ax.set_ylabel(COUPLING_LABEL, labelpad=2.0)
    else:
        ax.set_yticklabels([])
    _panel_title(ax, letter, config["title"])

    ax.text(0.975, 0.975, f"{gap:.2f}×", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.0, fontweight="bold", color=INK,
            zorder=9,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax.text(0.975, 0.865, "destroyed-to-surviving\nexposure gap",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.9,
            color=LABEL, linespacing=1.25, zorder=9,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])


def plot_destruction_panel(ax, source: pd.DataFrame, params: pd.DataFrame,
                           cross_fire: pd.Series, *, letter: str,
                           x_max: float):
    handles = []
    for fire in ("EATON", "PALISADES"):
        colour = FIRE_COLORS[fire]
        rows = source[source.fire.eq(fire)]
        fit = params[params.fire.eq(fire)].iloc[0]
        ax.axvspan(fit.f50_lo, fit.f50_hi, color=colour, alpha=0.11, lw=0,
                   zorder=1)
        ax.axvline(fit.f50, color=colour, ls=(0, (2.4, 1.6)), lw=1.0, zorder=2)
        curve = rows[rows.series.eq("fitted_curve")].sort_values("F_star")
        points = rows[rows.series.eq("empirical_bin")]
        ax.errorbar(
            points.F_star, points.probability,
            yerr=[points.probability - points.ci_lo,
                  points.ci_hi - points.probability],
            mfc="white", mec=colour, color=colour, ecolor=colour,
            zorder=4, **BIN_MARKERS,
        )
        ax.plot(curve.F_star, curve.probability, color=colour, lw=2.2,
                zorder=5, solid_capstyle="round")
        handles.append(Line2D(
            [0], [0], color=colour, lw=2.0,
            label=f"{fire.title()}  ($F_{{50}}$ = {fit.f50:.3f})",
        ))

    eaton = params[params.fire.eq("EATON")].iloc[0]
    palisades = params[params.fire.eq("PALISADES")].iloc[0]

    ax.set_xscale("log")
    ax.set_xlim(1e-4, x_max * 1.1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(
        FixedFormatter(["0", "0.25", "0.50", "0.75", "1.00"])
    )
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    _light_grid(ax, axis="y")
    ax.set_xlabel(COUPLING_LABEL, labelpad=2.0)
    ax.set_ylabel("Probability of complete destruction", labelpad=2.5)
    _panel_title(ax, letter, "Complete destruction")

    ax.legend(handles=handles, loc="upper left", handlelength=1.4,
              borderpad=0.15, labelspacing=0.32, handletextpad=0.5)

    # The threshold separation is the result, so a bracket ties it to the two
    # lines; the ratio is set in ink above the bracket, with the per-fire
    # values carried by the legend rather than by coloured labels.
    ax.annotate(
        "", xy=(palisades.f50, 0.885), xytext=(eaton.f50, 0.885),
        arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.8,
                        mutation_scale=5, shrinkA=0, shrinkB=0),
        zorder=8,
    )
    ax.text(np.sqrt(eaton.f50 * palisades.f50), 0.905,
            f"{cross_fire.ratio_PAL_to_EAT:.2f}×", ha="center", va="bottom",
            fontsize=8.4, fontweight="bold", color=INK, zorder=9,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])


def plot_partial_panel(ax, source: pd.DataFrame, peaks: pd.DataFrame, *,
                       letter: str, x_max: float):
    handles = []
    for fire in ("EATON", "PALISADES"):
        colour = FIRE_COLORS[fire]
        rows = source[source.fire.eq(fire)]
        peak = peaks[peaks.fire.eq(fire)].iloc[0]
        ax.axvline(peak.peak_F_star, color=colour, ls=(0, (2.4, 1.6)), lw=1.0,
                   zorder=2)
        curve = rows[rows.series.eq("implied_curve")].sort_values("F_star")
        points = rows[rows.series.eq("empirical_bin")]
        ax.errorbar(
            points.F_star, points.probability,
            yerr=[points.probability - points.ci_lo,
                  points.ci_hi - points.probability],
            mfc="white", mec=colour, color=colour, ecolor=colour,
            zorder=4, **BIN_MARKERS,
        )
        ax.plot(curve.F_star, curve.probability, color=colour, lw=2.2,
                zorder=5, solid_capstyle="round")
        ax.plot(peak.peak_F_star, peak.peak_partial_probability, "o", ms=3.4,
                mfc=colour, mec="white", mew=0.7, zorder=7)
        handles.append(Line2D(
            [0], [0], color=colour, lw=2.0,
            label=(f"{fire.title()}  (peak "
                   f"{100 * peak.peak_partial_probability:.1f}%)"),
        ))

    ax.set_xscale("log")
    ax.set_xlim(1e-4, x_max * 1.1)
    ax.set_ylim(0, 0.30)
    ax.set_yticks([0, 0.10, 0.20, 0.30])
    ax.yaxis.set_major_formatter(
        FixedFormatter(["0", "0.10", "0.20", "0.30"])
    )
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    _light_grid(ax, axis="y")
    ax.set_xlabel(COUPLING_LABEL, labelpad=2.0)
    ax.set_ylabel("Probability of partial damage", labelpad=2.5)
    _panel_title(ax, letter, "Partial damage, short of destruction")
    ax.legend(handles=handles, loc="upper left", handlelength=1.4,
              borderpad=0.15, labelspacing=0.32, handletextpad=0.5)


# -------------------------------------------------------------- assembly ---
def _band(fig, span: str, text: str, y_label: float, y_rule: float | None,
          x_left: float = 0.30, x_right: float = 6.97):
    """Tracked band label, optionally preceded by a separating hairline."""
    if y_rule is not None:
        fig.add_artist(Line2D([_fx(x_left), _fx(x_right)],
                              [_fy(y_rule), _fy(y_rule)],
                              color=RULE, lw=0.6, transform=fig.transFigure))
    fig.text(_fx(x_left), _fy(y_label), span, ha="left", va="baseline",
             fontsize=8.0, fontweight="bold", color=INK)
    fig.text(_fx(x_left + 0.40), _fy(y_label), text,
             ha="left", va="baseline", fontsize=7.4,
             fontweight="bold", color=LABEL)


def build_figure1(source_dir: Path, results_dir: Path, output_dir: Path, *,
                  stem: str = "Fig1_geometric_coupling_and_fragility",
                  seed: int = 11, data_variant: str = "legacy",
                  openview_run: Path | None = None,
                  building_cache: Path | None = None,
                  analysis_path: Path | None = None,
                  receiver_id: int | str | None = None,
                  primary_emitter_id: int | str | None = None,
                  scene_2d_run: Path | None = None,
                  openview_patch_run: Path | None = None):
    """Render the three-band Figure 1 composite from frozen packaged data."""
    source_dir, results_dir = Path(source_dir), Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_variant == "openview3d":
        distance = pd.read_csv(
            source_dir / "ccd_ssd_vs_F_openview3d_source.csv"
        )
        fragility = pd.read_csv(
            source_dir / "fragility_openview3d_source.csv"
        )
        partial = pd.read_csv(
            source_dir / "partial_damage_openview3d_source.csv"
        )
        parameters = pd.read_csv(
            results_dir / "fragility_parameters_openview3d.csv"
        )
        gaps = pd.read_csv(
            results_dir / "distance_outcome_gap_openview3d.csv"
        )
        peaks = pd.read_csv(
            results_dir / "partial_damage_summary_openview3d.csv"
        )
        destroyed = parameters[parameters.outcome.eq("destroyed")]
        f50 = destroyed.set_index("fire").f50
        cross_fire = pd.DataFrame([{
            "outcome": "destroyed",
            "ratio_PAL_to_EAT": f50["PALISADES"] / f50["EATON"],
        }])
    elif data_variant == "legacy":
        distance = pd.read_csv(
            source_dir / "fig1b_distance_comparison_source.csv"
        )
        fragility = pd.read_csv(source_dir / "fig1c_fragility_source.csv")
        partial = pd.read_csv(source_dir / "fig1d_partial_damage_source.csv")
        parameters = pd.read_csv(results_dir / "fragility_parameters.csv")
        gaps = pd.read_csv(results_dir / "distance_outcome_gap.csv")
        peaks = pd.read_csv(results_dir / "partial_damage_summary.csv")
        cross_fire = pd.read_csv(results_dir / "f50_cross_fire.csv")
    else:
        raise ValueError(f"Unknown Figure 1 data variant: {data_variant}")

    destroyed_params = parameters[parameters.outcome.eq("destroyed")]
    destruction_ratio = cross_fire[cross_fire.outcome.eq("destroyed")].iloc[0]
    gap = gaps.set_index("distance_measure").destroyed_to_surviving_F_star_ratio
    x_max = fragility.F_star.max()

    # Panels a-d share one Eaton scene, drawn by src.viz.figure1_measures.
    CONCEPT_HEIGHT, CONCEPT_GAP, CONCEPT_LEFT = 1.22, .10, .20
    concept_widths = np.array([1., 1., 1., 1.35])
    concept_widths *= (6.82 - 3 * CONCEPT_GAP) / concept_widths.sum()
    concept_lefts = CONCEPT_LEFT + np.concatenate(
        [[0], np.cumsum(concept_widths[:-1] + CONCEPT_GAP)])

    measure_scene = None
    if all(value is not None for value in (
        openview_run, building_cache, analysis_path, receiver_id,
    )):
        from src.viz.figure1_measures import (
            CONTEXT as MEASURE_CONTEXT, CONTEXT_EDGE as MEASURE_CONTEXT_EDGE,
            CORRIDOR as MEASURE_CORRIDOR, HIGH as MEASURE_HIGH,
            LOW as MEASURE_LOW, MEDIUM as MEASURE_MEDIUM,
            PANEL_TITLES as MEASURE_TITLES, RECEIVER as MEASURE_RECEIVER,
            SOURCE as MEASURE_SOURCE, draw_measure_row, load_measure_scene,
        )
        scene_run = Path(
            scene_2d_run if scene_2d_run is not None
            else source_dir.parent / "openview2d_eaton_scene"
        )
        measure_scene = load_measure_scene(
            openview_run, building_cache, analysis_path, scene_run,
            receiver_id=receiver_id, source_id=primary_emitter_id,
            plan_aspect=concept_widths[0] / CONCEPT_HEIGHT,
            patch_run=openview_patch_run,
        )

    with plt.rc_context(FIG_RC):
        fig = plt.figure(figsize=(FIG_W, FIG_H))

        # --- panels a-d: one scene, four measures -----------------------
        # Distance-based definitions in a-b, surface-based exposure in c-d.
        concept_bottom = 4.70
        titles = (MEASURE_TITLES if measure_scene is not None else [
            "Centroid separation", "Surface separation", "Pairwise exposure",
            "Accumulated exposure"])
        axes_ad = []
        for index, (left, width, letter, title) in enumerate(zip(
            concept_lefts, concept_widths, "abcd", titles,
        )):
            axes_kwargs = (
                {"projection": "3d", "computed_zorder": False}
                if index == 3 else {}
            )
            ax = fig.add_axes(
                [_fx(left), _fy(concept_bottom), _fx(width),
                 _fy(CONCEPT_HEIGHT)],
                **axes_kwargs,
            )
            axes_ad.append(ax)
            _panel_title(ax, letter, title, size=7.3, rise=0.025, gap_pt=7.2)

        if measure_scene is None:
            schematic_panels = [
                panel_a_centroid, panel_b_surface,
                panel_c_pairwise, panel_d_cumulative,
            ]
            for index, (ax, draw_panel, width) in enumerate(zip(
                axes_ad, schematic_panels, concept_widths,
            )):
                panel_rng = np.random.default_rng(
                    seed if index < 3 else seed + 1,
                )
                draw_panel(ax, panel_rng, 100.0 * CONCEPT_HEIGHT / width)
            legend_handles = [
                Patch(facecolor=SCHEMA_INK, edgecolor="none",
                      label=r"Destroyed emitter $j$"),
                Patch(facecolor=SCHEMA_GREY, edgecolor="none",
                      label=r"Receiver $i$"),
                Patch(facecolor=FAINT, edgecolor=FAINT_EDGE, lw=.4,
                      label="Neighbor"),
                Line2D([0], [0], color=EXCHANGE, lw=1.1,
                       label="Visible surface exchange"),
            ]
        else:
            draw_measure_row(
                axes_ad, measure_scene,
                frame_aspect=concept_widths[3] / CONCEPT_HEIGHT, zoom=1.9,
                links=True,
            )
            legend_handles = [
                Patch(facecolor=MEASURE_SOURCE, edgecolor="none",
                      label=r"Destroyed emitter $j$"),
                Patch(facecolor=MEASURE_RECEIVER, edgecolor="none",
                      label=r"Receiver $i$"),
                Patch(facecolor=MEASURE_CONTEXT, edgecolor=MEASURE_CONTEXT_EDGE,
                      lw=.4, label="Neighbor"),
                Patch(facecolor=MEASURE_CORRIDOR, edgecolor="none", alpha=.25,
                      label="Surface exchange"),
            ]
        fig.legend(
            handles=legend_handles, loc="lower left",
            bbox_to_anchor=(_fx(.30), _fy(4.50)), ncol=len(legend_handles),
            frameon=False, fontsize=5.9, handlelength=.95,
            handleheight=.70, handletextpad=.35, columnspacing=.8,
            labelspacing=.20, borderpad=0,
        )

        # --- band e-f: distance-exposure decay ---------------------------
        _band(fig, "e-f", "Distance-exposure relationship", 4.35, 4.47)
        e_w, e_gap = 3.035, 0.30
        ax_e = fig.add_axes([_fx(0.62), _fy(2.91), _fx(e_w), _fy(1.20)])
        ax_f = fig.add_axes([_fx(0.62 + e_w + e_gap), _fy(2.91), _fx(e_w),
                             _fy(1.20)])
        plot_distance_panel(ax_e, distance, "ccd_ft", gap["CCD"],
                            show_ylabel=True, letter="e")
        plot_distance_panel(ax_f, distance, "ssd_ft", gap["SSD"],
                            show_ylabel=False, letter="f")
        outcome_handles = [
            Line2D([0], [0], color=OUTCOME_STYLE[value]["colour"],
                   ls=OUTCOME_STYLE[value]["line"], lw=1.7, marker="o",
                   markersize=2.8,
                   markerfacecolor=OUTCOME_STYLE[value]["fill"],
                   markeredgecolor=OUTCOME_STYLE[value]["colour"],
                   markeredgewidth=0.7,
                   label=OUTCOME_STYLE[value]["label"])
            for value in (1, 0)
        ]
        ax_e.legend(handles=outcome_handles, loc="lower left",
                    handlelength=2.1, labelspacing=0.32, borderpad=0.2,
                    handletextpad=0.6)

        # --- band g-h: exposure-damage response, the result ---------------
        _band(fig, "g-h", "Exposure-damage relationship", 2.51, 2.63)
        g_w, g_gap = 2.875, 0.62
        ax_g = fig.add_axes([_fx(0.62), _fy(0.42), _fx(g_w), _fy(1.83)])
        ax_h = fig.add_axes([_fx(0.62 + g_w + g_gap), _fy(0.42), _fx(g_w),
                             _fy(1.83)])
        plot_destruction_panel(ax_g, fragility, destroyed_params,
                               destruction_ratio, letter="g", x_max=x_max)
        plot_partial_panel(ax_h, partial, peaks, letter="h", x_max=x_max)

        fig.savefig(output_dir / f"{stem}.pdf", facecolor="white")
        fig.savefig(output_dir / f"{stem}.png", dpi=400, facecolor="white")
    return fig
