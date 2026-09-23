#!/usr/bin/env python3
"""
Supplementary Figure — the plasticity gate (rigid vs plastic recurrent weights).

Panels:
  a  Per-network frozen-replay shift: rigid (2e-6) / no-head (5e-5) / head-ON (5e-5)
  b  Per-cell displacement magnitude for the same three conditions. This is the
     panel that makes the claim precise: a near-zero net shift is ambiguous
     between "the fields did not move" and "the fields moved in directions that
     cancel", and only the former supports a plasticity gate.
  c  Matching verification: phase-1 final loss per seed across the three
     conditions. Because Phase 1 is re-seeded immediately before data
     generation, matched runs are bit-identical here — visual proof that the
     conditions start from the same pre-learning network rather than merely
     sharing a seed number.

Usage:
    python paper_code/figures/figS_sr/generate_figure_rigid_supp.py --no_show
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
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from analysis import load_condition  # noqa: E402
from utils import (  # noqa: E402
    HEAD_DIR, MM_TO_PT, NOHEAD_DIR, PAGE_W, RIGID_DIR,
    COLOR_HEAD, COLOR_NOHEAD, apply_nature_style, ensure_dir, save_figure,
)

apply_nature_style()
COLOR_RIGID = "#7a7a7a"
FIGURES_DIR = HERE / "figures_supp"


def _fmt_p(p):
    """Format a p-value for a panel annotation; 'n/a' when it is not finite."""
    if not np.isfinite(p):
        return "n/a"
    return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"


def _arr(rows, key):
    """Pull one key out of a list of per-seed rows as a float array."""
    return np.array([r[key] for r in rows], dtype=float)


def _strip(ax, i, vals, colour, rng):
    """Draw one condition as a jittered strip of per-seed points plus a mean +- SD marker."""
    ax.scatter(i + rng.uniform(-0.14, 0.14, len(vals)), vals, s=11, color=colour,
               alpha=0.7, linewidths=0, zorder=3)
    ax.errorbar([i], [vals.mean()], yerr=[vals.std(ddof=1) if len(vals) > 1 else 0],
                fmt="D", color="black", markersize=4.5, capsize=3, lw=1.1, zorder=5)


def fig_shift(conds, stem):
    """Panel a -- frozen-replay Delta-d for rigid, no-head and head-ON networks."""
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    rng = np.random.default_rng(0)
    notes = []
    for i, (lab, rows, c) in enumerate(conds):
        v = _arr(rows, "dd_mean")
        _strip(ax, i, v, c, rng)
        try:
            notes.append(f"{lab.split(chr(10))[0]} {_fmt_p(wilcoxon(v).pvalue)}")
        except Exception:
            pass
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], fontsize=5.5)
    ax.set_ylabel("Δ distance to reward, frozen replay\n(negative = toward reward)")
    ax.set_title("Frozen-replay shift by plasticity\n" + "  ·  ".join(notes), fontsize=5.5)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_displacement(conds, stem):
    """Panel b -- mean per-cell displacement magnitude, separating stationary fields from cancelling ones."""
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    rng = np.random.default_rng(1)
    for i, (lab, rows, c) in enumerate(conds):
        _strip(ax, i, _arr(rows, "disp_mean"), c, rng)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], fontsize=5.5)
    ax.set_ylabel("Mean per-cell displacement\n$\\|c^{post}-c^{pre}\\|$ (arena units)")
    ax.set_title("The fields are stationary, not\nmoving-and-cancelling", fontsize=6)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_match(stem, seeds):
    """phase-1 final loss across conditions — bit-identical when matched."""
    def p1(d, pre, s):
        try:
            return json.load(open(Path(d) / f"{pre}seed{s}" / "metrics.json")).get("phase1_final_loss")
        except Exception:
            return None

    rows = []
    for s in seeds:
        a, b, c = p1(RIGID_DIR, "rigid_", s), p1(NOHEAD_DIR, "nohead_", s), p1(HEAD_DIR, "head_", s)
        if a is not None and b is not None and c is not None:
            rows.append((s, a, b, c))

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    if rows:
        ss = [r[0] for r in rows]
        rg = np.array([r[1] for r in rows])
        nh = np.array([r[2] for r in rows])
        hd = np.array([r[3] for r in rows])
        ax.plot(ss, rg, "o", ms=7, mfc="none", mec=COLOR_RIGID, mew=1.2, label="rigid")
        ax.plot(ss, nh, "x", ms=5, color=COLOR_NOHEAD, label="no-head")
        ax.plot(ss, hd, "+", ms=6, color=COLOR_HEAD, label="head-ON")
        n_ident = int(np.sum((rg == nh) & (rg == hd)))
        ax.set_title(f"Phase-1 final loss is bit-identical\nacross conditions in {n_ident}/{len(rows)} seeds",
                     fontsize=6)
        ax.legend(fontsize=5, frameon=False)
    else:
        ax.text(0.5, 0.5, "no matched metrics.json yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("seed"); ax.set_ylabel("phase-1 final loss")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def _pdf_to_png(pdf, png, dpi=300):
    """Render a one-page PDF to PNG at `dpi`, via Poppler's pdftoppm."""
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(pre)], check=True)
        copy2(pre.with_suffix(".png"), png)


def assemble(pdfs, out_stem, dpi=300):
    """Lay the panel PDFs out in one row and write the composite as PDF and PNG."""
    out_stem = Path(out_stem).with_suffix("")
    pdf_path, png_path = out_stem.with_suffix(".pdf"), out_stem.with_suffix(".png")
    M, G = 10 * MM_TO_PT, 5 * MM_TO_PT
    keys = [k for k in ("a", "b", "c") if k in pdfs]
    sizes = {k: (lambda b: (float(b.width), float(b.height)))(PdfReader(str(pdfs[k])).pages[0].mediabox)
             for k in keys}
    cw = (PAGE_W - 2 * M - (len(keys) - 1) * G) / len(keys)
    rh = max(sizes[k][1] * (cw / sizes[k][0]) for k in keys)
    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W, height=2 * M + rh)
    for i, k in enumerate(keys):
        iw, ih = sizes[k]
        sc = min(cw / iw, rh / ih)
        left = M + i * (cw + G) + (cw - iw * sc) / 2
        page.merge_transformed_page(PdfReader(str(pdfs[k])).pages[0],
                                    Transformation().scale(sc).translate(left, M + (rh - ih * sc) / 2), over=True)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    _pdf_to_png(pdf_path, png_path, dpi)
    print(f"  Saved: {pdf_path}\n  Saved: {png_path}")


def main():
    """Load the three conditions, build every panel, and assemble the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=50)
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()
    panels = ensure_dir(FIGURES_DIR / "panels")

    print("Loading rigid…");   rg = load_condition(RIGID_DIR, "rigid_", None, n_perm=args.n_perm)
    print(f"  {len(rg)} clean")
    if not rg:
        print("No gate-passing rigid seeds yet — run paper_code/runs/run_rigid_seeds.py first.")
        return
    print("Loading no-head…"); nh = load_condition(NOHEAD_DIR, "nohead_", None, n_perm=args.n_perm)
    print(f"  {len(nh)} clean")
    print("Loading head-ON…"); hd = load_condition(HEAD_DIR, "head_", None, n_perm=args.n_perm)
    print(f"  {len(hd)} clean")

    conds = [("rigid\n$2\\times10^{-6}$", rg, COLOR_RIGID),
             ("no-head\n$5\\times10^{-5}$", nh, COLOR_NOHEAD),
             ("head-ON\n$5\\times10^{-5}$", hd, COLOR_HEAD)]

    with open(panels / "rigid_summary.json", "w") as f:
        json.dump({"rigid": rg, "nohead": nh, "head": hd}, f, indent=2)

    pdfs = {}
    print("Panel a…"); _, pdfs["a"] = fig_shift(conds, panels / "rigid_panel_a")
    print("Panel b…"); _, pdfs["b"] = fig_displacement(conds, panels / "rigid_panel_b")
    print("Panel c…"); _, pdfs["c"] = fig_match(panels / "rigid_panel_c", sorted(r["seed"] for r in rg))
    print("Assembling…"); assemble(pdfs, FIGURES_DIR / "figure_rigid_supp_raw")

    d = {k: _arr(v, "dd_mean") for k, v, _ in conds}
    p = {k: _arr(v, "disp_mean") for k, v, _ in conds}
    print("\n=== summary ===")
    for (lab, rows, _) in conds:
        k = lab
        print(f"  {lab.splitlines()[0]:>8}: dd={d[k].mean():+.4f}±{d[k].std(ddof=1):.4f}  "
              f"disp={p[k].mean():.4f}±{p[k].std(ddof=1):.4f}  n={len(rows)}")
    print(f"\nDone. {FIGURES_DIR}")


if __name__ == "__main__":
    main()
