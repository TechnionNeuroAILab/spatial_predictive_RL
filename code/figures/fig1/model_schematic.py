#!/usr/bin/env python3
"""Figure 1: the two-pathway model, a single-column vector schematic.

Replaces the externally authored raster (paper_figures/v2/fig_model_v3.png)
with the same, deliberately simple content on one row: the environment and
the agent's egocentric input, the recurrent network trained on sensory
prediction (Phase A), and the goal-conditioned actor-critic that reads its
state (Phase B). Everything sits on one horizontal axis through the network,
so boxes and arrows align.

    python code/figures/fig1/model_schematic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))

import cartoons as ct  # noqa: E402
from pubstyle import (  # noqa: E402
    C, COL_SINGLE, FS_PANEL, FS_SMALL, FS_TICK, OI, apply_nature_style, save_figure,
)

apply_nature_style()

W = COL_SINGLE / 72.0       # single column, inches
H = 1.62
YC = 0.70                   # the horizontal axis everything is aligned on
INK = ct.INK
A_FACE, A_EDGE = "#EDF4FB", C["pre"]
B_FACE, B_EDGE = "#FCF0EA", C["policy"]


def _text(ax, x, y, s, size=FS_SMALL, **kw):
    kw = {"ha": "center", "va": "center", "color": INK, "zorder": 10, **kw}
    return ax.text(x, y, s, fontsize=size, **kw)


def _phase_box(ax, x0, x1, face, edge, title, subtitle):
    ax.add_patch(FancyBboxPatch((x0, 0.06), x1 - x0, H - 0.12,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=face, edgecolor=edge, lw=0.8, zorder=1))
    _text(ax, x0 + 0.07, H - 0.17, title, FS_TICK, ha="left", fontweight="bold", color=edge)
    _text(ax, x0 + 0.07, H - 0.31, subtitle, FS_SMALL, ha="left")


def environment(ax, x0, w):
    y0 = YC - w / 2
    _text(ax, x0 + w / 2, y0 + w + 0.12, "Environment", FS_TICK, fontweight="bold")
    ct.arena(ax, x0, y0, w, face="#FAFAFA")
    for k in range(1, 6):
        ax.plot([x0 + k * w / 6] * 2, [y0, y0 + w], color="#E6E6E6", lw=0.3, zorder=3.2)
        ax.plot([x0, x0 + w], [y0 + k * w / 6] * 2, color="#E6E6E6", lw=0.3, zorder=3.2)
    cols = (OI["blue"], OI["orange"], OI["green"])
    for i, (px, py) in enumerate(((0.25, 1), (0.6, 1), (1, 0.85), (1, 0.18), (0.7, 0),
                                  (0.3, 0), (0, 0.45), (0, 0.8))):
        ax.add_patch(Circle((x0 + px * w, y0 + py * w), 0.016, facecolor="white",
                            edgecolor=cols[i % 3], lw=0.6, zorder=4))
    ct.star(ax, x0 + 0.3 * w, y0 + 0.3 * w, C["reward"], size=34)
    agx, agy = x0 + 0.62 * w, y0 + 0.58 * w
    ct.fov_wedge(ax, agx, agy, 40, 0.30 * w, half_angle=28)
    ct.agent(ax, agx, agy, 40, r=0.022)
    _text(ax, x0 + w / 2, y0 - 0.10, "egocentric view", FS_SMALL, color=ct.MUTED)


def rnn(ax, cx, cy, r):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=C["pre"], lw=0.8, zorder=3))
    n = 6
    pts = [(cx + 0.62 * r * np.cos(2 * np.pi * k / n + np.pi / 6),
            cy + 0.62 * r * np.sin(2 * np.pi * k / n + np.pi / 6)) for k in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color="#B9CFE6",
                    lw=0.35, zorder=3.5)
    for p in pts:
        ax.add_patch(Circle(p, 0.034, facecolor=C["pre"], edgecolor="white", lw=0.5, zorder=4))


def actor_bars(ax, x0, y0, w, h):
    vals = np.array([0.35, 0.55, 0.40, 0.95, 0.50, 0.70, 0.30, 0.60])
    bw = w / len(vals)
    for k, v in enumerate(vals):
        ax.add_patch(Rectangle((x0 + k * bw + 0.1 * bw, y0), 0.8 * bw, v * h,
                               facecolor=C["policy"], edgecolor="none", zorder=4))
    ax.plot([x0, x0 + w], [y0, y0], color=INK, lw=0.5, zorder=4)


def critic_gauge(ax, cx, cy, r):
    ax.add_patch(Wedge((cx, cy), r, 0, 180, width=0.022, facecolor=C["policy"],
                       edgecolor="none", zorder=4))
    th = np.deg2rad(55)
    ax.plot([cx, cx + 0.85 * r * np.cos(th)], [cy, cy + 0.85 * r * np.sin(th)],
            color=INK, lw=0.8, zorder=5, solid_capstyle="round")
    ax.add_patch(Circle((cx, cy), 0.018, facecolor=INK, edgecolor="none", zorder=5))


def main(out_stem: Path | None = None):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    # ---- environment and input ------------------------------------------------
    ex0, ew = 0.03, 0.56
    environment(ax, ex0, ew)

    # ---- Phase A: recurrent network trained on sensory prediction ------------
    a0, a1 = 0.88, 1.70
    _phase_box(ax, a0, a1, A_FACE, A_EDGE, "Phase A", "Sensory prediction")
    cx, r = (a0 + a1) / 2, 0.27
    rnn(ax, cx, YC, r)
    ct.arrow(ax, ex0 + ew + 0.03, YC, cx - r, YC)
    mid = (ex0 + ew + a0) / 2 + 0.02
    _text(ax, mid, YC + 0.10, "$\\mathit{o}_t$", FS_PANEL)
    _text(ax, mid, YC - 0.11, "$\\mathit{a}_{t-1}$", FS_PANEL)
    # prediction of the next observation
    py = YC + r + 0.17
    ct.arrow(ax, cx, YC + r, cx, py - 0.07)
    _text(ax, cx, py, "$\\hat{\\mathit{o}}_{t+1}$", FS_PANEL)

    # ---- Phase B: goal-conditioned actor-critic -------------------------------
    b0, b1 = 1.82, W - 0.03
    _phase_box(ax, b0, b1, B_FACE, B_EDGE, "Phase B", "Actor–critic; prediction continues")
    kx0, kw, kh = 2.10, 0.50, 0.30
    ct.box(ax, kx0, YC - kh / 2, kw, kh, edge="#555555")
    _text(ax, kx0 + kw / 2, YC, "Controller", FS_SMALL)
    ct.arrow(ax, cx + r, YC, kx0, YC)
    _text(ax, (b0 + kx0) / 2 + 0.01, YC + 0.10, "$\\mathit{h}_t$", FS_PANEL)
    gy = 0.22
    ct.arrow(ax, kx0 + kw / 2, gy + 0.10, kx0 + kw / 2, YC - kh / 2)
    _text(ax, kx0 + kw / 2, gy, "goal $\\mathit{g}_t$", FS_PANEL)
    # outputs
    sx = kx0 + kw + 0.08
    hx = sx + 0.10
    ay, vy = YC + 0.30, YC - 0.30
    ax.plot([kx0 + kw, sx], [YC, YC], color=INK, lw=0.8, zorder=3)
    ax.plot([sx, sx], [vy, ay], color=INK, lw=0.8, zorder=3)
    ct.arrow(ax, sx, ay, hx, ay)
    ct.arrow(ax, sx, vy, hx, vy)
    bw = b1 - 0.05 - hx - 0.02
    actor_bars(ax, hx + 0.02, ay - 0.09, bw, 0.22)
    _text(ax, hx + 0.02 + bw / 2, ay - 0.17, "$\\mathit{\\pi}$", FS_PANEL, color=C["policy"])
    critic_gauge(ax, hx + 0.02 + bw / 2, vy - 0.06, 0.13)
    _text(ax, hx + 0.02 + bw / 2, vy - 0.17, "$\\mathit{V}$", FS_PANEL, color=C["policy"])

    out_stem = out_stem or HERE / "figures" / "figure_model_raw"
    return save_figure(fig, out_stem, exact_size=True)


if __name__ == "__main__":
    main()
