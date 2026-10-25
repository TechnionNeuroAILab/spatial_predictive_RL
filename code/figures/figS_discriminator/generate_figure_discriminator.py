#!/usr/bin/env python3
"""
Supplementary/main figure — the occupancy-matched estimator, validated in
silico, and the path-dependence limit of marginal matching.

Reads results/aggregate/occupancy_matched_estimator.json (written by
code/aggregate/run_estimator_validation.py; 79 gate-passing networks x 2 post
periods, K=20 subsample reps per estimate).

Story the panels tell (honest version -- the pre-registered ordering-recovery
criterion failed and is reported as such):
  a  The estimator ladder: naive rate-map contrast overestimates the
     weight-driven shift ~6x; position matching helps; joint position x
     heading matching brings the population magnitude into range.
  b  Specificity: on a same-weights contrast (true shift exactly 0) the naive
     read-out reports a large false shift; the matched estimator reports ~0.
  c  The limit: per-seed agreement with frozen-replay ground truth is weak
     (r ~ 0.1-0.3) and the head-vs-no-head ordering is not recovered --
     because the recurrent state is path-dependent, matching instantaneous
     (position, heading) marginals cannot equate state distributions. Only
     trajectory-level control identifies the weight-driven component.
  d  The pre-specified ordering test, matched estimate vs fixed-trajectory replay.

Usage:
    python code/figures/figS_discriminator/generate_figure_discriminator.py --no_show
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from utils import (  # noqa: E402
    C, COLOR_HEAD, COLOR_NOHEAD, FS_SMALL, FS_TICK, MM_TO_PT, PAGE_W,
    plain_log_ticks, stat_annot, title, apply_nature_style, ensure_dir, save_figure,
    fmt_p,
)

import figlayout  # noqa: E402
from figure_cache import add_plot_only  # noqa: E402

from pubstyle import sem  # noqa: E402
apply_nature_style()

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
# The agreement scatter and the ordering test used to share one grid cell as a
# 1x2 pair; they are panels of their own now, so their labels sit on the type
# scale rather than being shrunk twice.
#
#   a  estimator ladder        b  same-weights specificity control
#   c  per-network agreement   d  paired ordering test
LAYOUT = figlayout.Layout(
    [("a", 0, 0, 3), ("b", 0, 3, 3),
     ("c", 1, 0, 3), ("d", 1, 3, 3)],
    ncols=6, row_h_in=[2.40, 2.20], caption_lines=6,
    name="figS_discriminator",
)
CACHE = REPO / "results" / "aggregate" / "occupancy_matched_estimator.json"
FIGURES_DIR = HERE / "figures_supp"
POST = "cycle_1_goal"          # weight-aligned variant (post weights == GT post)


def _fmt_p(p):
    """Format a p-value for a panel annotation; scientific notation below 0.001."""
    """Panel p-value, in the same format as every other figure in the paper.

    Routed to the shared formatter so no panel invents its own notation, and
    none of them use a mathtext superscript (which prints at 0.7x and falls
    under the 5 pt floor).
    """
    return fmt_p(p, italic=False)


def load():
    """Read the estimator JSON and keep the paired-family passers used in the text."""
    sys.path.insert(0, str(REPO / "code" / "lib"))
    from analysis_helpers import campaign_passers
    with open(CACHE) as f:
        d = json.load(f)
    keep = campaign_passers("paired")
    rows = d["per_seed"][POST]
    nh = [r for r in rows["nohead"] if int(r["seed"]) in keep]
    hd = [r for r in rows["head"] if int(r["seed"]) in keep]
    return d, nh, hd


def _arr(rows, key):
    """Pull one key out of a list of per-seed rows as a float array (missing -> NaN)."""
    return np.array([r.get(key) for r in rows], dtype=float)


def fig_ladder(nh, hd, stem, size_in=None):
    """Panel a -- the estimator ladder: naive vs position- vs joint-matched vs reweighted Delta-d, per condition."""
    fig, ax = plt.subplots(figsize=size_in or (3.4, 2.6))
    rng = np.random.default_rng(0)
    labels = ["Naive", "Position\nmatched", "Position \u00d7\nheading", "Fixed-traj.\nreplay"]
    keys = ["est_naive", "est_pos", "est_joint", "gt_mean_dd"]
    for k, (rows, c, off) in enumerate([(nh, COLOR_NOHEAD, -0.17), (hd, COLOR_HEAD, +0.17)]):
        for i, key in enumerate(keys):
            v = _arr(rows, key)
            v = v[np.isfinite(v)]
            ax.scatter(np.full(len(v), i + off) + rng.uniform(-0.06, 0.06, len(v)),
                       v, s=5, color=c, alpha=0.35, linewidths=0, zorder=3)
            ax.errorbar([i + off], [v.mean()], yerr=[sem(v)], fmt="D",
                        color="black", markersize=3.5, capsize=2, lw=0.9, zorder=5)
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=FS_SMALL)
    ax.set_ylabel("Δ distance to reward\n(negative = toward reward)")
    title(ax, "Estimator ladder")
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="none", color=COLOR_NOHEAD,
                              markersize=3, label="Without teaching"),
                       Line2D([0], [0], marker="o", ls="none", color=COLOR_HEAD,
                              markersize=3, label="With teaching")],
              fontsize=FS_SMALL, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_specificity(nh, hd, stem, size_in=None):
    """Panel b -- specificity on a same-weights contrast, where the true shift is exactly zero."""
    fig, ax = plt.subplots(figsize=size_in or (2.6, 2.6))
    rng = np.random.default_rng(1)
    rows_all = nh + hd
    for i, key in enumerate(["null_naive", "null_joint"]):
        v = _arr(rows_all, key)
        v = v[np.isfinite(v)]
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.1, 0.1, len(v)), v,
                   s=6, color="0.5", alpha=0.5, linewidths=0, zorder=3)
        ax.errorbar([i], [v.mean()], yerr=[sem(v)], fmt="D", color="black",
                    markersize=4, capsize=3, lw=1.0, zorder=5)
    ax.axhline(0, color=C["policy"], lw=0.9, ls="--", label="True shift = 0")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Naive", "Position \u00d7 heading\nmatched"], fontsize=FS_SMALL)
    ax.set_ylabel("Δ distance to reward")
    title(ax, "Same-weights control")
    ax.legend(fontsize=FS_SMALL, frameon=False)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_agreement(nh, hd, stem, size_in=None):
    """Panel c -- matched estimator against ground truth, network by network.

    The honest-failure panel: matching fixes the population mean but not the
    per-network estimate, because the two periods still traverse different
    paths through the state space.
    """
    fig, ax = plt.subplots(figsize=size_in or (2.3, 2.2))
    for rows, c, lab in [(nh, COLOR_NOHEAD, "Without teaching"), (hd, COLOR_HEAD, "With teaching")]:
        g, e = _arr(rows, "gt_mean_dd"), _arr(rows, "est_joint")
        ok = np.isfinite(g) & np.isfinite(e)
        ax.scatter(g[ok], e[ok], s=8, color=c, alpha=0.65, linewidths=0, label=lab)
    g_all = np.concatenate([_arr(nh, "gt_mean_dd"), _arr(hd, "gt_mean_dd")])
    e_all = np.concatenate([_arr(nh, "est_joint"), _arr(hd, "est_joint")])
    ok = np.isfinite(g_all) & np.isfinite(e_all)
    lim = [min(g_all[ok].min(), e_all[ok].min()) - 0.02,
           max(g_all[ok].max(), e_all[ok].max()) + 0.02]
    ax.plot(lim, lim, color="0.6", lw=0.7, ls="--")
    ax.set_xlabel("Fixed-trajectory replay")
    ax.set_ylabel("Matched estimate")
    title(ax, "Per-network agreement")
    bits = []
    for rows, lab in ((nh, "without"), (hd, "with")):
        g, e = _arr(rows, "gt_mean_dd"), _arr(rows, "est_joint")
        m = np.isfinite(g) & np.isfinite(e)
        bits.append(f"{lab} r={np.corrcoef(e[m], g[m])[0, 1]:.2f}")
    stat_annot(ax, "\n".join(bits), loc="lower right")
    ax.legend(fontsize=FS_SMALL, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_ordering(nh, hd, stem, size_in=None):
    """Panel d -- the pre-specified ordering test, paired by seed.

    Only the fixed-trajectory control recovers the head vs no-head ordering;
    the matched estimator reverses it.
    """
    fig, ax = plt.subplots(figsize=size_in or (2.3, 2.2))
    nhd = {r_["seed"]: r_ for r_ in nh}
    hdd = {r_["seed"]: r_ for r_ in hd}
    common = sorted(set(nhd) & set(hdd))
    for i, (key, lab) in enumerate([("est_joint", "matched\nestimator"),
                                    ("gt_mean_dd", "frozen replay\n(truth)")]):
        a = np.array([nhd[s][key] for s in common], dtype=float)
        b = np.array([hdd[s][key] for s in common], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        # pairing variable: seed (same pre-learning network in both conditions)
        p = wilcoxon(a[ok], b[ok]).pvalue
        ax.errorbar([i - 0.12], [a[ok].mean()], yerr=[sem(a[ok])], fmt="o",
                    color=COLOR_NOHEAD, markersize=4, capsize=2, lw=0.9)
        ax.errorbar([i + 0.12], [b[ok].mean()], yerr=[sem(b[ok])], fmt="o",
                    color=COLOR_HEAD, markersize=4, capsize=2, lw=0.9)
        ax.annotate(f"paired {_fmt_p(p)}",
                    (i, max(a[ok].mean(), b[ok].mean()) + 0.012),
                    ha="center", fontsize=FS_SMALL,
                    color="#333333")
    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Matched\nestimate", "Fixed-trajectory\nreplay"], fontsize=FS_TICK)
    ax.set_ylabel("Δ distance to reward")
    title(ax, "Teaching-condition ordering")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def _pdf_to_png(pdf, png, dpi=300):
    """Render a one-page PDF to PNG at `dpi`, via Poppler's pdftoppm."""
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(pre)], check=True)
        copy2(pre.with_suffix(".png"), png)


def assemble(pdfs, out_stem, dpi=300):
    """Compose onto `LAYOUT`, with the panel letters the caption promises."""
    return figlayout.compose(LAYOUT, pdfs, out_stem, dpi=dpi)


def main():
    """Build every panel from the cached JSON and assemble the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--assemble_only", action="store_true")
    add_plot_only(ap)
    args = ap.parse_args()
    panels = ensure_dir(FIGURES_DIR / "panels")
    if args.assemble_only:
        print("Assembly-only mode: reusing existing panel PDFs...")
        assemble(figlayout.existing_panels(panels, LAYOUT.keys()),
                 FIGURES_DIR / "figure_discriminator_supp_raw")
        print("Done.")
        return
    _, nh, hd = load()
    if args.plot_only:
        print("Plot-only: drawing from the existing JSON cache.")
    print(f"loaded {len(nh)} no-head + {len(hd)} head rows (post={POST})")

    pdfs = {}
    print("Panel a (estimator ladder)…")
    _, pdfs["a"] = fig_ladder(nh, hd, panels / "panel_a", LAYOUT.slot_size_in("a"))
    print("Panel b (specificity control)…")
    _, pdfs["b"] = fig_specificity(nh, hd, panels / "panel_b", LAYOUT.slot_size_in("b"))
    print("Panel c (per-network agreement)…")
    _, pdfs["c"] = fig_agreement(nh, hd, panels / "panel_c", LAYOUT.slot_size_in("c"))
    print("Panel d (ordering test)…")
    _, pdfs["d"] = fig_ordering(nh, hd, panels / "panel_d", LAYOUT.slot_size_in("d"))
    print("Assembling…"); assemble(pdfs, FIGURES_DIR / "figure_discriminator_supp_raw")
    print("Done.")


if __name__ == "__main__":
    main()
