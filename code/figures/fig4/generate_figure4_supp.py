#!/usr/bin/env python3
"""
Supplementary Figure 4 — relocation, network by network, for the 52 retained
relocation networks (analysis_helpers.campaign_passers("reloc")).

One stored random-walk trajectory is replayed through the baseline, after-A
and after-B networks. Place cells: Skaggs >= 0.15 in both periods of a
comparison. Shifts: delta = d_after - d_before to the active reward
(negative = toward). Every panel shows the field peak and the centre of mass;
error bars are s.e.m. across networks.

Panels:
  a  % of each network's cells moving toward A (baseline -> after A)
  b  % moving toward B (after A -> after B)
  c  per-cell shift toward the active reward, pooled over networks (violins),
     with each network's mean

Usage:
    python code/figures/fig4/generate_figure4_supp.py --no_show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[3]
FIG4_DIR = REPO / "code" / "figures" / "fig4"
sys.path.insert(0, str(FIG4_DIR))
from utils import (  # noqa: E402
    COLOR_A, COLOR_B, PERIOD_POST_A, PERIOD_POST_B, PERIOD_PRE, REWARD_A,
    REWARD_B, SKAGGS_THRESHOLD, C, FS_SMALL, FS_TICK, apply_nature_style,
    ensure_dir, fmt_p, save_figure, title,
)
import figlayout  # noqa: E402
from figure_cache import add_plot_only, load_figure_data, require_keys, update_figure_data  # noqa: E402

sys.path.insert(0, str(FIG4_DIR.parents[1] / "lib"))
from analysis_helpers import campaign_passers, load_period_centers_cached  # noqa: E402
from pubstyle import METRIC_ALPHA, METRIC_LABEL, network_strip, sem, style_violin, zero_line  # noqa: E402

apply_nature_style()

SUPP_FIGURES_DIR = FIG4_DIR / "figures_supp"

#   a  fraction toward A        b  fraction toward B
#   c  pooled distributions
LAYOUT = figlayout.Layout(
    [("a", 0, 0, 3), ("b", 0, 3, 3),
     ("c", 1, 0, 6)],
    ncols=6, row_h_in=[1.95, 1.95], caption_lines=6,
    name="figS_relocation",
)


def load_seed(seed: int, multi_seed_dir: Path) -> dict | None:
    """Per-cell shifts toward A and toward B, peak and centre of mass."""
    data = multi_seed_dir / f"reloc_seed{seed}" / "data"
    try:
        P = {p: load_period_centers_cached(str(data), p)
             for p in (PERIOD_PRE, PERIOD_POST_A, PERIOD_POST_B)}
    except FileNotFoundError as exc:
        print(f"  seed {seed}: skip ({exc})")
        return None
    row = {"seed": seed}
    for step, (p0, p1, R) in (("A", (PERIOD_PRE, PERIOD_POST_A, REWARD_A)),
                              ("B", (PERIOD_POST_A, PERIOD_POST_B, REWARD_B))):
        gate = (P[p0][0] >= SKAGGS_THRESHOLD) & (P[p1][0] >= SKAGGS_THRESHOLD)
        for metric, idx in (("peak", 2), ("com", 1)):
            d = (np.linalg.norm(P[p1][idx][gate] - R, axis=1)
                 - np.linalg.norm(P[p0][idx][gate] - R, axis=1))
            row[f"delta_{step}_{metric}"] = d
            row[f"pct_{step}_{metric}"] = 100.0 * float((d < 0).mean())
    return row


def fig_pct_toward(rows, step, stem, size_in):
    """% of each network's cells moving toward `step` (A or B), peak and COM."""
    col = COLOR_A if step == "A" else COLOR_B
    fig, ax = plt.subplots(figsize=size_in)
    vals = [np.array([r[f"pct_{step}_{m}"] for r in rows]) for m in ("peak", "com")]
    for x, m, v in zip((0, 1), ("peak", "com"), vals):
        network_strip(ax, [v], [x], [col], pair=False,
                      alpha=0.6 * METRIC_ALPHA[m] + 0.1, seed=x)
    ax.axhline(50, color=C["zero"], lw=0.5, ls=(0, (2.5, 2)), zorder=0.5)
    ax.text(1.5, 50, "chance", fontsize=FS_SMALL, color=C["muted"], ha="right", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([METRIC_LABEL["peak"], METRIC_LABEL["com"]])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(f"Cells moving toward {step} (%)")
    for x, v in zip((0, 1), vals):
        ax.text(x, 104, f"{v.mean():.1f} ± {sem(v):.1f}%\n{fmt_p(wilcoxon(v - 50).pvalue)}",
                ha="center", va="bottom", fontsize=FS_SMALL, color=C["muted"], linespacing=1.3)
    title(ax, ("Baseline → after A" if step == "A" else "After A → after B")
          + f" (n = {len(rows)})")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def fig_pooled(rows, stem, size_in):
    """Per-cell shift toward the active reward, pooled, with network means."""
    fig, ax = plt.subplots(figsize=size_in)
    rng = np.random.default_rng(2)
    allv = np.concatenate([r[f"delta_{s}_{m}"] for r in rows for s in "AB" for m in ("peak", "com")])
    lo, hi = np.percentile(allv, [1, 99])
    for metric, x0 in (("peak", 0.0), ("com", 2.4)):
        for x, step, col in ((x0, "A", COLOR_A), (x0 + 1, "B", COLOR_B)):
            v = np.concatenate([r[f"delta_{step}_{metric}"] for r in rows])
            v = v[(v >= lo) & (v <= hi)]
            style_violin(ax.violinplot([v], positions=[x], widths=0.8, showmeans=False,
                                       showmedians=False, showextrema=False),
                         col, alpha=0.3 + 0.35 * METRIC_ALPHA[metric])
        ma = np.array([r[f"delta_A_{metric}"].mean() for r in rows])
        mb = np.array([r[f"delta_B_{metric}"].mean() for r in rows])
        j = rng.uniform(-0.07, 0.07, len(rows))
        for i in range(len(rows)):
            ax.plot([x0 + j[i], x0 + 1 + j[i]], [ma[i], mb[i]], color="#8C8C8C", lw=0.3, alpha=0.6)
        ax.scatter(x0 + j, ma, s=4, color="#1A1A1A", lw=0, zorder=4)
        ax.scatter(x0 + 1 + j, mb, s=4, color="#1A1A1A", lw=0, zorder=4)
        ax.text(x0 + 0.5, -0.15, METRIC_LABEL[metric], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FS_TICK)
    zero_line(ax)
    ax.set_xticks([0, 1, 2.4, 3.4])
    ax.set_xticklabels(["to A", "to B", "to A", "to B"])
    ax.set_xlim(-0.6, 4.0)
    ax.set_ylabel(r"$\Delta$ distance to active reward")
    title(ax, "Pooled cells and network means")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--multi_seed_dir", default=str(REPO / "results" / "multi_seed_reloc"))
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    add_plot_only(p)
    args = p.parse_args()
    args.panels = {x.strip() for x in args.panels.split(",") if x.strip()}
    return args


def main():
    args = parse_args()
    out = Path(args.output_dir)
    panels_dir = ensure_dir(out / "panels")
    cache_path = out / "figure_data.npz"
    if args.plot_only:
        rows = require_keys(load_figure_data(cache_path), "rows", path=cache_path)["rows"]
    else:
        seeds = sorted(campaign_passers("reloc"))
        rows = [r for r in (load_seed(s, Path(args.multi_seed_dir)) for s in seeds) if r]
        update_figure_data(cache_path, {"rows": rows})
    print(f"  n = {len(rows)} retained networks")
    S = LAYOUT.slot_size_in
    makers = {
        "a": lambda: fig_pct_toward(rows, "A", panels_dir / "panel_a", S("a")),
        "b": lambda: fig_pct_toward(rows, "B", panels_dir / "panel_b", S("b")),
        "c": lambda: fig_pooled(rows, panels_dir / "panel_c", S("c")),
    }
    panel_pdfs = {k: makers[k]()[1] for k in LAYOUT.keys() if k in args.panels}
    if not args.no_assemble and panel_pdfs:
        figlayout.compose(LAYOUT, panel_pdfs, out / "figure4_supp_raw")
    print(f"\nDone. Outputs: {out}")


if __name__ == "__main__":
    main()
