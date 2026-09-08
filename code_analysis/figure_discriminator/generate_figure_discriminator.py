#!/usr/bin/env python3
"""
Supplementary/main figure — the occupancy-matched estimator, validated in
silico, and the path-dependence limit of marginal matching.

Reads Repro/aggregate/occupancy_matched_estimator.json (written by
New_Code/run_estimator_validation.py; 79 gate-passing networks x 2 post
periods, K=20 subsample reps per estimate).

Story the panels tell (honest version -- the pre-registered ordering-recovery
criterion FAILED and is reported as such):
  a  The estimator ladder: naive rate-map contrast overestimates the
     weight-driven shift ~6x; position matching helps; joint position x
     heading matching brings the population magnitude into range.
  b  Specificity: on a same-weights contrast (true shift exactly 0) the naive
     read-out reports a large false shift; the matched estimator reports ~0.
  c  The limit: per-seed agreement with frozen-replay ground truth is weak
     (r ~ 0.1-0.3) and the head-vs-no-head ordering is NOT recovered --
     because the recurrent state is PATH-dependent, matching instantaneous
     (position, heading) marginals cannot equate state distributions. Only
     trajectory-level control identifies the weight-driven component.
  d  Sample-size dependence of the matched estimator (practical guidance).

Usage:
    python code_analysis/figure_discriminator/generate_figure_discriminator.py --no_show
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
import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from utils import (  # noqa: E402
    COLOR_HEAD, COLOR_NOHEAD, MM_TO_PT, PAGE_W,
    apply_nature_style, ensure_dir, save_figure,
)

apply_nature_style()
CACHE = REPO / "Repro" / "aggregate" / "occupancy_matched_estimator.json"
FIGURES_DIR = HERE / "figures_supp"
POST = "cycle_1_goal"          # weight-aligned variant (post weights == GT post)


def _fmt_p(p):
    return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"


def load():
    with open(CACHE) as f:
        d = json.load(f)
    rows = d["per_seed"][POST]
    return d, rows["nohead"], rows["head"]


def _arr(rows, key):
    return np.array([r.get(key) for r in rows], dtype=float)


def fig_ladder(nh, hd, stem):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    rng = np.random.default_rng(0)
    labels = ["naive", "position\nmatched", "pos x heading\nmatched", "frozen replay\n(ground truth)"]
    keys = ["est_naive", "est_pos", "est_joint", "gt_mean_dd"]
    for k, (rows, c, off) in enumerate([(nh, COLOR_NOHEAD, -0.17), (hd, COLOR_HEAD, +0.17)]):
        for i, key in enumerate(keys):
            v = _arr(rows, key)
            v = v[np.isfinite(v)]
            ax.scatter(np.full(len(v), i + off) + rng.uniform(-0.06, 0.06, len(v)),
                       v, s=5, color=c, alpha=0.35, linewidths=0, zorder=3)
            ax.errorbar([i + off], [v.mean()], yerr=[v.std(ddof=1)], fmt="D",
                        color="black", markersize=3.5, capsize=2, lw=0.9, zorder=5)
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("Δ distance to reward\n(negative = toward reward)")
    ax.set_title("Matching removes the sampling bias\n(blue no-head, brown head-ON)", fontsize=6)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_specificity(nh, hd, stem):
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    rng = np.random.default_rng(1)
    rows_all = nh + hd
    for i, key in enumerate(["null_naive", "null_joint"]):
        v = _arr(rows_all, key)
        v = v[np.isfinite(v)]
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.1, 0.1, len(v)), v,
                   s=6, color="0.5", alpha=0.5, linewidths=0, zorder=3)
        ax.errorbar([i], [v.mean()], yerr=[v.std(ddof=1)], fmt="D", color="black",
                    markersize=4, capsize=3, lw=1.0, zorder=5)
    ax.axhline(0, color="crimson", lw=0.9, ls="--", label="true shift = 0")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["naive", "pos x heading\nmatched"], fontsize=5.5)
    ax.set_ylabel("Δ distance to reward")
    ax.set_title("Same-weights contrast:\nfalse shift eliminated", fontsize=6)
    ax.legend(fontsize=5, frameon=False)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_limit(nh, hd, stem):
    """Scatter vs GT + the ordering result: the honest-failure panel."""
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4))
    ax = axes[0]
    for rows, c, lab in [(nh, COLOR_NOHEAD, "no-head"), (hd, COLOR_HEAD, "head-ON")]:
        g, e = _arr(rows, "gt_mean_dd"), _arr(rows, "est_joint")
        ok = np.isfinite(g) & np.isfinite(e)
        ax.scatter(g[ok], e[ok], s=8, color=c, alpha=0.65, linewidths=0, label=lab)
    g_all = np.concatenate([_arr(nh, "gt_mean_dd"), _arr(hd, "gt_mean_dd")])
    e_all = np.concatenate([_arr(nh, "est_joint"), _arr(hd, "est_joint")])
    ok = np.isfinite(g_all) & np.isfinite(e_all)
    r = np.corrcoef(g_all[ok], e_all[ok])[0, 1]
    lim = [min(g_all[ok].min(), e_all[ok].min()) - 0.02,
           max(g_all[ok].max(), e_all[ok].max()) + 0.02]
    ax.plot(lim, lim, color="0.6", lw=0.7, ls="--")
    ax.set_xlabel("ground truth (frozen replay)")
    ax.set_ylabel("matched estimator")
    ax.set_title(f"Per-network agreement is weak\nr = {r:.2f}: the path-dependence limit", fontsize=6)
    ax.legend(fontsize=5, frameon=False)

    ax = axes[1]
    # ordering: paired seeds
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
        ax.errorbar([i - 0.12], [a[ok].mean()], yerr=[a[ok].std(ddof=1)], fmt="o",
                    color=COLOR_NOHEAD, markersize=4, capsize=2, lw=0.9)
        ax.errorbar([i + 0.12], [b[ok].mean()], yerr=[b[ok].std(ddof=1)], fmt="o",
                    color=COLOR_HEAD, markersize=4, capsize=2, lw=0.9)
        correct = b[ok].mean() < a[ok].mean()
        ax.annotate(f"{_fmt_p(p)}\n{'correct' if correct else 'WRONG'} order",
                    (i, max(a[ok].mean(), b[ok].mean()) + 0.012),
                    ha="center", fontsize=5, color="green" if correct else "crimson")
    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["matched\nestimator", "frozen replay\n(truth)"], fontsize=5.5)
    ax.set_ylabel("Δ distance to reward")
    ax.set_title("Only trajectory control recovers\nthe head vs no-head ordering", fontsize=6)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    return save_figure(fig, stem)


def fig_samplesize(nh, hd, stem):
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    for rows, c, lab in [(nh, COLOR_NOHEAD, "no-head"), (hd, COLOR_HEAD, "head-ON")]:
        fracs, means, sds, ns = [], [], [], []
        for fr in ("0.125", "0.25", "0.5", "1.0"):
            vals = [r["size_curve"][fr]["mean_dd"] for r in rows
                    if r.get("size_curve", {}).get(fr)]
            nm = [r["size_curve"][fr]["n_matched"] for r in rows
                  if r.get("size_curve", {}).get(fr)]
            if vals:
                fracs.append(float(fr)); means.append(np.mean(vals))
                sds.append(np.std(vals, ddof=1)); ns.append(int(np.median(nm)))
        ax.errorbar(ns, means, yerr=sds, marker="o", markersize=3, lw=1.0,
                    capsize=2, color=c, label=lab)
    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("matched timesteps (per period)")
    ax.set_ylabel("matched-estimator Δd")
    ax.set_title("Estimate vs matched sample size", fontsize=6)
    ax.legend(fontsize=5, frameon=False)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def _pdf_to_png(pdf, png, dpi=300):
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(pre)], check=True)
        copy2(pre.with_suffix(".png"), png)


def assemble(pdfs, out_stem, dpi=300):
    out_stem = Path(out_stem).with_suffix("")
    M_, G = 10 * MM_TO_PT, 5 * MM_TO_PT
    rows = [[k for k in ("a", "b") if k in pdfs], [k for k in ("c", "d") if k in pdfs]]
    rows = [r for r in rows if r]
    sizes = {k: (lambda b: (float(b.width), float(b.height)))(PdfReader(str(pdfs[k])).pages[0].mediabox)
             for r in rows for k in r}
    heights = []
    for r in rows:
        cw = (PAGE_W - 2 * M_ - (len(r) - 1) * G) / len(r)
        heights.append(max(sizes[k][1] * cw / sizes[k][0] for k in r))
    page_h = 2 * M_ + sum(heights) + (len(rows) - 1) * G
    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W, height=page_h)
    y = M_
    for r, rh in zip(reversed(rows), reversed(heights)):
        cw = (PAGE_W - 2 * M_ - (len(r) - 1) * G) / len(r)
        for i, k in enumerate(r):
            iw, ih = sizes[k]
            sc = min(cw / iw, rh / ih)
            left = M_ + i * (cw + G) + (cw - iw * sc) / 2
            page.merge_transformed_page(PdfReader(str(pdfs[k])).pages[0],
                                        Transformation().scale(sc).translate(left, y + (rh - ih * sc) / 2),
                                        over=True)
        y += rh + G
    pdf_path = out_stem.with_suffix(".pdf")
    with open(pdf_path, "wb") as f:
        writer.write(f)
    _pdf_to_png(pdf_path, out_stem.with_suffix(".png"), dpi)
    print(f"  Saved: {pdf_path}\n  Saved: {out_stem.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()
    panels = ensure_dir(FIGURES_DIR / "panels")
    _, nh, hd = load()
    print(f"loaded {len(nh)} no-head + {len(hd)} head rows (post={POST})")

    pdfs = {}
    print("Panel a (ladder)…");      _, pdfs["a"] = fig_ladder(nh, hd, panels / "panel_a")
    print("Panel b (specificity)…"); _, pdfs["b"] = fig_specificity(nh, hd, panels / "panel_b")
    print("Panel c (limit)…");       _, pdfs["c"] = fig_limit(nh, hd, panels / "panel_c")
    print("Panel d (sample size)…"); _, pdfs["d"] = fig_samplesize(nh, hd, panels / "panel_d")
    print("Assembling…"); assemble(pdfs, FIGURES_DIR / "figure_discriminator_supp_raw")
    print("Done.")


if __name__ == "__main__":
    main()
