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
    python code/figures/figS_sr/generate_figure_rigid_supp.py --no_show
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
    FS_BASE, FS_SMALL, FS_TICK, title, stat_annot,
    fmt_p,
)
import figlayout  # noqa: E402
from figure_cache import add_plot_only, load_figure_data, require_keys, update_figure_data  # noqa: E402

apply_nature_style()
COLOR_RIGID = "#7a7a7a"
FIGURES_DIR = HERE / "figures_supp"


def _fmt_p(p):
    """Format a p-value for a panel annotation; 'n/a' when it is not finite."""
    if not np.isfinite(p):
        return "n/a"
    """Panel p-value, in the same format as every other figure in the paper.

    Routed to the shared formatter so no panel invents its own notation, and
    none of them use a mathtext superscript (which prints at 0.7x and falls
    under the 5 pt floor).
    """
    return fmt_p(p, italic=False)


def _arr(rows, key):
    """Pull one key out of a list of per-seed rows as a float array."""
    return np.array([r[key] for r in rows], dtype=float)


def _strip(ax, i, vals, colour, rng):
    """Draw one condition as a jittered strip of per-seed points plus a mean +- SD marker."""
    ax.scatter(i + rng.uniform(-0.14, 0.14, len(vals)), vals, s=11, color=colour,
               alpha=0.7, linewidths=0, zorder=3)
    ax.errorbar([i], [vals.mean()], yerr=[vals.std(ddof=1) if len(vals) > 1 else 0],
                fmt="D", color="black", markersize=4.5, capsize=3, lw=1.1, zorder=5)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
# One row of three. Every glyph in this figure used to print below 5 pt: the
# panels were drawn at ~3.1 in, fitted into a third of a 183 mm page, and then
# the whole plate was included at 0.95\linewidth.
#
#   a  fixed-trajectory shift by condition   b  per-cell displacement
#   c  bit-identical Phase-1 losses verify matched initial states
LAYOUT = figlayout.Layout(
    [("a", 0, 0, 2), ("b", 0, 2, 2), ("c", 0, 4, 2)],
    ncols=6, row_h_in=[2.40], caption_lines=3,
    name="figS_rigid",
)


def fig_shift(conds, stem, size_in=None):
    """Panel a -- frozen-replay Delta-d for rigid, no-head and head-ON networks."""
    fig, ax = plt.subplots(figsize=size_in or (3.1, 2.5))
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
    ax.set_xticklabels([c[0] for c in conds], fontsize=FS_SMALL)
    ax.set_ylabel("Δ distance to reward, frozen replay\n(negative = toward reward)")
    title(ax, "Frozen-replay shift by plasticity")
    # One p-value per line inside the axes. Joined onto the title with " · "
    # they ran 25 pt past the right edge of this panel's slot.
    stat_annot(ax, "\n".join(notes), loc="lower left")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_displacement(conds, stem, size_in=None):
    """Panel b -- mean per-cell displacement magnitude, separating stationary fields from cancelling ones."""
    fig, ax = plt.subplots(figsize=size_in or (3.1, 2.5))
    rng = np.random.default_rng(1)
    for i, (lab, rows, c) in enumerate(conds):
        _strip(ax, i, _arr(rows, "disp_mean"), c, rng)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], fontsize=FS_SMALL)
    # Was $\|c^{post}-c^{pre}\|$: mathtext sets those superscripts at 0.7x,
    # i.e. 4.9 pt, under the 5 pt floor. Said in words instead.
    ax.set_ylabel("Mean per-cell displacement\nbetween pre and post (arena units)")
    ax.set_title("The fields are stationary, not\nmoving-and-cancelling", fontsize=FS_TICK)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_match(stem, seeds, size_in=None):
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

    fig, ax = plt.subplots(figsize=size_in or (3.4, 2.5))
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
                     fontsize=FS_TICK)
        ax.legend(fontsize=FS_SMALL, frameon=False)
    else:
        ax.text(0.5, 0.5, "no matched metrics.json yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS_BASE)
    ax.set_xlabel("seed"); ax.set_ylabel("phase-1 final loss")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def _pdf_to_png(pdf, png, dpi=300):
    """Render a one-page PDF to PNG at `dpi`, via Poppler's pdftoppm."""
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(pre)], check=True)
        copy2(pre.with_suffix(".png"), png)


def assemble(panel_pdfs: dict[str, Path], out_stem: Path, dpi: int = 300):
    """Compose onto `LAYOUT`, with the panel letters the caption promises."""
    return figlayout.compose(LAYOUT, panel_pdfs, out_stem, dpi=dpi)


def main():
    """Load the three conditions, build every panel, and assemble the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=50)
    ap.add_argument("--no_show", action="store_true")
    add_plot_only(ap)
    args = ap.parse_args()
    panels = ensure_dir(FIGURES_DIR / "panels")
    cache_path = FIGURES_DIR / "figure_data.npz"
    if args.plot_only:
        saved = require_keys(load_figure_data(cache_path), "rg", "nh", "hd", path=cache_path)
        rg, nh, hd = saved["rg"], saved["nh"], saved["hd"]
    else:
        print("Loading rigid…");   rg = load_condition(RIGID_DIR, "rigid_", None, n_perm=args.n_perm)
        print("Loading no-head…"); nh = load_condition(NOHEAD_DIR, "nohead_", None, n_perm=args.n_perm)
        print("Loading head-ON…"); hd = load_condition(HEAD_DIR, "head_", None, n_perm=args.n_perm)
        update_figure_data(cache_path, {"rg": rg, "nh": nh, "hd": hd})
    print(f"  {len(rg)} rigid, {len(nh)} no-head, {len(hd)} head-ON")
    if not rg:
        print("No gate-passing rigid seeds yet — run code/runs/run_rigid_seeds.py first.")
        return

    # The learning rates used to be set here as mathtext superscripts, which
    # print at 0.7x the tick size (3.85 pt). They are in the caption instead.
    conds = [("rigid", rg, COLOR_RIGID),
             ("no-head", nh, COLOR_NOHEAD),
             ("head-ON", hd, COLOR_HEAD)]

    with open(panels / "rigid_summary.json", "w") as f:
        json.dump({"rigid": rg, "nohead": nh, "head": hd}, f, indent=2)

    pdfs = {}
    print("Panel a…"); _, pdfs["a"] = fig_shift(conds, panels / "rigid_panel_a", LAYOUT.slot_size_in("a"))
    print("Panel b…"); _, pdfs["b"] = fig_displacement(conds, panels / "rigid_panel_b", LAYOUT.slot_size_in("b"))
    print("Panel c…"); _, pdfs["c"] = fig_match(panels / "rigid_panel_c", sorted(r["seed"] for r in rg),
                                       size_in=LAYOUT.slot_size_in("c"))
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
