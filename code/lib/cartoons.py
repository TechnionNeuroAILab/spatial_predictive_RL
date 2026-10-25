"""Vector glyphs for the schematic panels (model architecture, experiment
protocols), drawn in data coordinates on an equal-aspect axes.

One module so the three schematics share one visual language: the same mouse,
the same track, the same arena, the same arrows and line weights.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch,
                                 PathPatch, Polygon, Wedge)
from matplotlib.path import Path

INK = "#2B2B2B"          # outlines and text
MUTED = "#6E6E6E"
LW = 0.7                  # default outline weight (pt)


def mouse(ax, x, y, length=0.30, color="#4A4A4A", facing=1, z=6):
    """Side-view mouse, nose at the right (facing=1) or left (facing=-1).

    (x, y) is the point under the hind feet; `length` is nose-to-rump.
    """
    s = length
    def P(px, py):
        return (x + facing * px * s, y + py * s)

    body = Path(
        [P(0.02, 0.06),
         P(0.20, -0.01), P(0.45, -0.01), P(0.60, 0.02),
         P(0.75, 0.04), P(0.90, 0.08), P(1.00, 0.13),
         P(0.93, 0.19), P(0.83, 0.25), P(0.72, 0.28),
         P(0.63, 0.30), P(0.57, 0.31), P(0.50, 0.35),
         P(0.34, 0.46), P(0.08, 0.43), P(0.02, 0.28),
         P(-0.01, 0.18), P(0.00, 0.10), P(0.02, 0.06)],
        [Path.MOVETO] + [Path.CURVE4] * 18)
    ax.add_patch(PathPatch(body, facecolor=color, edgecolor="none", zorder=z))
    # ear, with a pale inner disc
    ax.add_patch(Ellipse(P(0.63, 0.35), 0.17 * s, 0.20 * s, facecolor=color,
                         edgecolor="white", lw=0.5, zorder=z + 0.1))
    ax.add_patch(Ellipse(P(0.63, 0.35), 0.09 * s, 0.11 * s, facecolor="#D8B4B4",
                         edgecolor="none", zorder=z + 0.2))
    # eye and nose tip
    ax.add_patch(Circle(P(0.83, 0.19), 0.022 * s, facecolor="white", edgecolor="none",
                        zorder=z + 0.3))
    ax.add_patch(Circle(P(1.00, 0.13), 0.020 * s, facecolor="#D8B4B4", edgecolor="none",
                        zorder=z + 0.3))
    # feet, just below the belly line
    for fx in (0.20, 0.60):
        ax.add_patch(Ellipse(P(fx, -0.012), 0.11 * s, 0.04 * s, facecolor=color,
                             edgecolor="none", zorder=z))
    # tail
    tail = Path([P(0.01, 0.12), P(-0.22, 0.04), P(-0.36, 0.22), P(-0.60, 0.14)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(tail, facecolor="none", edgecolor=color, lw=0.8,
                           capstyle="round", zorder=z - 0.1))


def star(ax, x, y, color, size=60, crossed=False, z=7):
    ax.scatter([x], [y], marker="*", s=size, color=color, edgecolor=INK,
               linewidths=0.4, zorder=z)
    if crossed:
        r = 0.035 * np.sqrt(size / 60)
        for sx in (-1, 1):
            ax.plot([x - r, x + r], [y - sx * r, y + sx * r], color="#C0392B",
                    lw=1.0, zorder=z + 1, solid_capstyle="round")


def track(ax, x0, x1, y, h=0.05, face="#E9E9E9"):
    """A linear VR track seen from the side: a thin rounded bar."""
    ax.add_patch(FancyBboxPatch((x0, y - h), x1 - x0, h,
                                boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                facecolor=face, edgecolor=INK, lw=LW, zorder=3))


def zone(ax, x0, x1, y, color, h=0.05):
    ax.add_patch(FancyBboxPatch((x0, y - h), x1 - x0, h, boxstyle="square,pad=0",
                                facecolor=color, edgecolor="none", alpha=0.75, zorder=3.5))


def arena(ax, x0, y0, w, face="#F3F6FA"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, w, boxstyle=f"round,pad=0,rounding_size={0.04 * w}",
                                facecolor=face, edgecolor=INK, lw=LW, zorder=3))


def path_in_arena(ax, x0, y0, w, seed=3, color=MUTED, z=4, with_agent=True):
    """A fixed, smooth exploratory trajectory inside an arena (dashed), with
    the agent at its start; the same seed draws the same path everywhere."""
    rng = np.random.default_rng(seed)
    p = np.cumsum(rng.normal(0, 1.0, size=(900, 2)), axis=0)
    k = np.ones(18) / 18
    p = np.column_stack([np.convolve(p[:, d], k, mode="valid") for d in (0, 1)])[::3]
    p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-9)
    p[:, 0] = 1.0 - p[:, 0]                  # start in the lower right, clear of rewards A and B
    p = 0.14 + 0.72 * p
    xs, ys = x0 + p[:, 0] * w, y0 + p[:, 1] * w
    ax.plot(xs, ys, color=color, lw=0.55, ls=(0, (2.2, 1.6)), zorder=z,
            solid_capstyle="round")
    if with_agent:
        ax.add_patch(Circle((xs[0], ys[0]), 0.035 * w, facecolor=INK, edgecolor="white",
                            lw=0.4, zorder=z + 1))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.8, z=5, style="-|>", ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def box(ax, x0, y0, w, h, face="white", edge=INK, lw=LW, r=0.04, z=4, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=face, edgecolor=edge, lw=lw, zorder=z, linestyle=ls))


def fov_wedge(ax, x, y, heading_deg, radius, half_angle=30, color="#7FA7D1", z=4.5):
    ax.add_patch(Wedge((x, y), radius, heading_deg - half_angle, heading_deg + half_angle,
                       facecolor=color, edgecolor="none", alpha=0.45, zorder=z))


def agent(ax, x, y, heading_deg, r=0.035, z=6):
    """Agent seen from above: a disc with a heading tick."""
    ax.add_patch(Circle((x, y), r, facecolor=INK, edgecolor="none", zorder=z))
    th = np.deg2rad(heading_deg)
    tip = (x + 1.9 * r * np.cos(th), y + 1.9 * r * np.sin(th))
    ax.add_patch(Polygon([tip, (x + r * np.cos(th + 2.2), y + r * np.sin(th + 2.2)),
                          (x + r * np.cos(th - 2.2), y + r * np.sin(th - 2.2))],
                         closed=True, facecolor=INK, edgecolor="none", zorder=z))
