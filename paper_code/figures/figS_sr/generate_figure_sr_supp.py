#!/usr/bin/env python3
"""
Supplementary Figure — the successor-representation comparison, done correctly.

Replaces the retired Figure 5 panel f. Panels:
  a  Why the old metric was wrong: observed projection vs its shuffled-cell null
  b  The correct metric (change in distance to reward), per condition
  c  Nulls: reward-location null for the mean; shuffle null for frac-toward
  d  Spatial specificity: Delta-d toward the true reward vs other locations

Usage:
    python code_analysis/figure_sr/generate_figure_sr_supp.py --no_show
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
sys.path.insert(0, str(HERE))

from analysis import MIRROR_LOCATIONS, load_condition  # noqa: E402
from utils import (  # noqa: E402
    HEAD_DIR, MM_TO_PT, NOHEAD_DIR, PAGE_W, RIGID_DIR,
    COLOR_HEAD, COLOR_NOHEAD, apply_nature_style, ensure_dir, save_figure,
)

apply_nature_style()

COLOR_RIGID = "#7a7a7a"
FIGURES_DIR = HERE / "figures_supp"


def _fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"


def _arr(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


# ---------------------------------------------------------------------------
# Panel a — the diagnosis
# ---------------------------------------------------------------------------

def fig_diagnosis(rows_nh, rows_hd, stem):
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    groups = [
        ("no-head\nobserved", _arr(rows_nh, "proj_mean"), COLOR_NOHEAD, False),
        ("no-head\nshuffled", _arr(rows_nh, "proj_null_mean"), COLOR_NOHEAD, True),
        ("head-ON\nobserved", _arr(rows_hd, "proj_mean"), COLOR_HEAD, False),
        ("head-ON\nshuffled", _arr(rows_hd, "proj_null_mean"), COLOR_HEAD, True),
    ]
    rng = np.random.default_rng(0)
    for i, (lab, v, c, hatched) in enumerate(groups):
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=8,
                   color=c, alpha=0.35 if hatched else 0.7, linewidths=0, zorder=3)
        ax.errorbar([i], [v.mean()], yerr=[v.std(ddof=1)], fmt="D", color="black",
                    markersize=4, capsize=3, lw=1.0, zorder=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups], fontsize=5)
    ax.set_ylabel("Retired metric: shift projected\nonto toward-reward axis")
    ax.axhline(0, color="0.5", lw=0.6, ls="--")
    ax.set_title("The retired metric fails its own null\n"
                 "(shuffled > observed in both conditions)", fontsize=6)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel b — the correct metric
# ---------------------------------------------------------------------------

def fig_correct_metric(conds, stem):
    """conds: list of (label, rows, colour)."""
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    rng = np.random.default_rng(1)
    stats_txt = []
    for i, (lab, rows, c) in enumerate(conds):
        if not rows:
            continue
        v = _arr(rows, "dd_mean")
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=10, color=c,
                   alpha=0.7, linewidths=0, zorder=3)
        ax.errorbar([i], [v.mean()], yerr=[v.std(ddof=1)], fmt="D", color="black",
                    markersize=4.5, capsize=3, lw=1.1, zorder=5)
        try:
            p = wilcoxon(v).pvalue
        except Exception:
            p = float("nan")
        stats_txt.append(f"{lab.split(chr(10))[0]} {_fmt_p(p)}")
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], fontsize=5.5)
    ax.set_ylabel("Δ distance to reward, frozen replay\n(negative = toward reward)")
    ax.set_title("Correct metric\n" + "  ·  ".join(stats_txt), fontsize=5.5)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel c — the two nulls
# ---------------------------------------------------------------------------

def fig_nulls(rows_nh, rows_hd, stem):
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.3))

    ax = axes[0]
    for i, (lab, rows, c) in enumerate([("no-head", rows_nh, COLOR_NOHEAD),
                                         ("head-ON", rows_hd, COLOR_HEAD)]):
        obs = _arr(rows, "dd_mean")
        nul = _arr(rows, "dd_null_mean")
        ax.scatter(np.full(len(obs), i - 0.15), obs, s=9, color=c, alpha=0.7, linewidths=0)
        ax.scatter(np.full(len(nul), i + 0.15), nul, s=9, facecolors="none",
                   edgecolors=c, alpha=0.7, linewidths=0.4)
        for a, b in zip(obs, nul):
            ax.plot([i - 0.15, i + 0.15], [a, b], color="0.75", lw=0.3, zorder=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no-head", "head-ON"], fontsize=6)
    ax.set_ylabel("Mean Δd")
    ax.set_title("Cell-shuffle null is ANALYTICALLY\nINERT for the mean (filled=obs, open=null)",
                 fontsize=5.2)

    ax = axes[1]
    for i, (lab, rows, c) in enumerate([("no-head", rows_nh, COLOR_NOHEAD),
                                         ("head-ON", rows_hd, COLOR_HEAD)]):
        obs = _arr(rows, "dd_frac_toward") * 100
        nul = _arr(rows, "dd_frac_toward_null") * 100
        ax.scatter(np.full(len(obs), i - 0.15), obs, s=9, color=c, alpha=0.7, linewidths=0)
        ax.scatter(np.full(len(nul), i + 0.15), nul, s=9, facecolors="none",
                   edgecolors=c, alpha=0.7, linewidths=0.4)
        for a, b in zip(obs, nul):
            ax.plot([i - 0.15, i + 0.15], [a, b], color="0.75", lw=0.3, zorder=1)
    ax.axhline(50, color="0.4", lw=0.7, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no-head", "head-ON"], fontsize=6)
    ax.set_ylabel("% cells toward reward")
    ax.set_title("…but it IS a valid null for the\npaired statistic", fontsize=5.2)

    fig.suptitle("Choosing the right null", fontsize=6.5, y=1.03)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel d — spatial specificity
# ---------------------------------------------------------------------------

def fig_specificity(rows_nh, rows_hd, stem):
    names = list(MIRROR_LOCATIONS)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    w = 0.36
    xs = np.arange(len(names))
    for k, (lab, rows, c) in enumerate([("no-head", rows_nh, COLOR_NOHEAD),
                                         ("head-ON", rows_hd, COLOR_HEAD)]):
        means = [_arr(rows, f"dd_at::{n}").mean() for n in names]
        sds = [_arr(rows, f"dd_at::{n}").std(ddof=1) for n in names]
        ax.bar(xs + (k - 0.5) * w, means, yerr=sds, width=w, color=c, alpha=0.85,
               capsize=2, error_kw={"lw": 0.7}, label=lab)
    ax.axhline(0, color="0.4", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=5, rotation=20, ha="right")
    ax.set_ylabel("Mean Δd toward location\n(negative = toward)")
    ax.set_title("Reward-location null: the shift is specific\nto the true reward A", fontsize=6)
    ax.legend(fontsize=5, frameon=False)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300):
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "page"
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                        str(pdf_path), str(pre)], check=True)
        copy2(pre.with_suffix(".png"), png_path)


def assemble(panel_pdfs: dict[str, Path], out_stem: Path, dpi: int = 300):
    out_stem = Path(out_stem).with_suffix("")
    pdf_path, png_path = out_stem.with_suffix(".pdf"), out_stem.with_suffix(".png")
    MARGIN, GAP = 10 * MM_TO_PT, 5 * MM_TO_PT

    def size(p):
        b = PdfReader(str(p)).pages[0].mediabox
        return float(b.width), float(b.height)

    sizes = {k: size(v) for k, v in panel_pdfs.items()}
    rows = [[k for k in r if k in panel_pdfs] for r in (("a", "b"), ("c", "d"))]
    rows = [r for r in rows if r]

    def row_h(keys):
        cw = (PAGE_W - 2 * MARGIN - (len(keys) - 1) * GAP) / len(keys)
        return max(sizes[k][1] * (cw / sizes[k][0]) for k in keys)

    hs = [row_h(r) for r in rows]
    page_h = 2 * MARGIN + sum(hs) + (len(hs) - 1) * GAP

    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W, height=page_h)
    y = MARGIN
    for keys, rh in zip(reversed(rows), reversed(hs)):
        cw = (PAGE_W - 2 * MARGIN - (len(keys) - 1) * GAP) / len(keys)
        for i, k in enumerate(keys):
            iw, ih = sizes[k]
            sc = min(cw / iw, rh / ih)
            left = MARGIN + i * (cw + GAP) + (cw - iw * sc) / 2
            bottom = y + (rh - ih * sc) / 2
            src = PdfReader(str(panel_pdfs[k])).pages[0]
            page.merge_transformed_page(src, Transformation().scale(sc).translate(left, bottom), over=True)
        y += rh + GAP

    with open(pdf_path, "wb") as f:
        writer.write(f)
    _pdf_to_png(pdf_path, png_path, dpi)
    print(f"  Saved: {pdf_path}\n  Saved: {png_path}")
    return png_path, pdf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    panels_dir = ensure_dir(FIGURES_DIR / "panels")

    print("Loading no-head…")
    rows_nh = load_condition(NOHEAD_DIR, "nohead_", args.seeds, n_perm=args.n_perm)
    print(f"  {len(rows_nh)} clean seeds")
    print("Loading head-ON…")
    rows_hd = load_condition(HEAD_DIR, "head_", args.seeds, n_perm=args.n_perm)
    print(f"  {len(rows_hd)} clean seeds")
    print("Loading rigid (if present)…")
    rows_rg = load_condition(RIGID_DIR, "rigid_", args.seeds, n_perm=args.n_perm)
    print(f"  {len(rows_rg)} clean seeds" + ("  [not yet run]" if not rows_rg else ""))

    with open(panels_dir / "sr_summary.json", "w") as f:
        json.dump({"nohead": rows_nh, "head": rows_hd, "rigid": rows_rg}, f, indent=2)
    print(f"  Saved: {panels_dir / 'sr_summary.json'}")

    pdfs = {}
    print("Panel a (diagnosis)…");    _, pdfs["a"] = fig_diagnosis(rows_nh, rows_hd, panels_dir / "panel_a")

    conds = []
    if rows_rg:
        conds.append(("rigid\n(2e-6)", rows_rg, COLOR_RIGID))
    conds += [("no-head\n(5e-5)", rows_nh, COLOR_NOHEAD), ("head-ON\n(5e-5)", rows_hd, COLOR_HEAD)]
    print("Panel b (correct metric)…"); _, pdfs["b"] = fig_correct_metric(conds, panels_dir / "panel_b")
    print("Panel c (nulls)…");          _, pdfs["c"] = fig_nulls(rows_nh, rows_hd, panels_dir / "panel_c")
    print("Panel d (specificity)…");    _, pdfs["d"] = fig_specificity(rows_nh, rows_hd, panels_dir / "panel_d")

    print("\nAssembling…")
    assemble(pdfs, FIGURES_DIR / "figure_sr_supp_raw")
    print(f"\nDone. {FIGURES_DIR}")


if __name__ == "__main__":
    main()
