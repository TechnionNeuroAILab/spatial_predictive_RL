#!/usr/bin/env python3
"""
Supplementary Figure 3 — aggregate statistics across all seeds (for Figure 3).

Panels:
  a  Per-seed mean COM Δd: A→B (should all be negative with head ON)
  b  Per-seed fraction toward reward: A→B (should all be > 50%)
  c  Pooled violin: Δd A→B across all seeds, p << 0.001
  d  Head vs no-head paired comparison: Δd(A→B) per seed — a "quick look"
     double dissociation on the RL-policy contrast (occupancy-confounded by
     design). This is complementary to, not a replacement for, Figure 5
     panel b, which is the entropy-gated frozen-replay diagnostic (isolates
     weight-only change) and is the paper's definitive H1-vs-H2 test.

Usage:
    python code/figures/fig3/generate_figure3_supp.py --no_show
    python code/figures/fig3/generate_figure3_supp.py \\
        --multi_seed_dir results/multi_seed_head \\
        --nohead_dir results/multi_seed \\
        --seeds 0 1 2 ... 49 --no_show
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon as _wilcoxon
from pypdf import PdfReader, PdfWriter, Transformation

REPO = Path(__file__).resolve().parents[3]
FIG3_DIR = REPO / "code" / "figures" / "fig3"
FIG2_DIR = REPO / "code" / "figures" / "fig2"
FIG1_DIR = REPO / "code" / "figures" / "fig1"

# fig2 first: `analysis` is fig2's shared implementation and must execute with
# fig2's utils, which alone defines PHASE_A/PHASE_B/PHASE_C_*. fig1 stays on the
# path because fig2/analysis.py imports fig1's rate_maps.
sys.path.insert(0, str(FIG1_DIR))
sys.path.insert(0, str(FIG2_DIR))

from analysis import load_phases, shared_place_cell_ids, paired_distance_stats  # noqa: E402

# fig3's utils, loaded by path on purpose. FIG3_DIR was never placed on sys.path,
# so a bare `from utils import ...` here resolved to fig2's copy -- and would keep
# doing so even if it were, because importing `analysis` above already bound the
# name "utils" to fig2's module in sys.modules.
_spec = importlib.util.spec_from_file_location("fig3_utils", FIG3_DIR / "utils.py")
_fig3_utils = importlib.util.module_from_spec(_spec)
sys.modules["fig3_utils"] = _fig3_utils
_spec.loader.exec_module(_fig3_utils)

DEFAULT_REWARD = _fig3_utils.DEFAULT_REWARD
apply_nature_style = _fig3_utils.apply_nature_style
ensure_dir = _fig3_utils.ensure_dir
save_figure = _fig3_utils.save_figure
PAGE_W = _fig3_utils.PAGE_W
MM_TO_PT = _fig3_utils.MM_TO_PT

apply_nature_style()
from figure_cache import add_plot_only, load_figure_data, require_keys, update_figure_data  # noqa: E402

SUPP_FIGURES_DIR = FIG3_DIR / "figures_supp"
REWARD_POS = np.asarray(DEFAULT_REWARD, dtype=float)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_seed(seed: int, multi_seed_dir: Path, prefix: str, skaggs_threshold: float = 0.05):
    """Per-seed summary row for the head-ON contrasts; None if the run is missing or fails."""
    run_dir = multi_seed_dir / f"{prefix}seed{seed}"
    data_dir = run_dir / "data"
    if not data_dir.exists():
        return None
    try:
        phases = load_phases(data_dir)
        ids = shared_place_cell_ids(phases, skaggs_threshold=skaggs_threshold)
        stats = paired_distance_stats(phases, REWARD_POS, ids)
        return {
            "seed": seed,
            "n_cells": int(len(ids)),
            "com_mean_delta_AB": float(stats["com_mean_delta_AB"]),
            "com_frac_toward_AB": float(stats["com_frac_toward_AB"]),
            "delta_ab": np.asarray(stats["delta_com_AB"], dtype=float),
        }
    except Exception as exc:
        print(f"  [seed {seed}] error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def _wilcoxon_p(vals, alternative="less"):
    """Paired two-sided Wilcoxon p-value; NaN when there are too few pairs or no differences."""
    try:
        _, p = _wilcoxon(vals, alternative=alternative)
        return float(p)
    except Exception:
        return float("nan")


def _fmt_p(p):
    """Format a p-value for a panel annotation; scientific notation below 0.001."""
    if not np.isfinite(p):
        return ""
    """Panel p-value, in the same format as every other figure in the paper.

    Routed to the shared formatter so no panel invents its own notation, and
    none of them use a mathtext superscript (which prints at 0.7x and falls
    under the 5 pt floor).
    """
    return fmt_p(p, italic=False)


def fig_supp_delta_strips(rows, stem):
    """Panel a — per-seed mean Δd A→B."""
    vals = np.array([r["com_mean_delta_AB"] for r in rows])
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(2.5, 2.2))
    jitter = rng.uniform(-0.08, 0.08, len(vals))
    ax.scatter(jitter, vals, s=14, color="#8B4513", alpha=0.75, zorder=3, linewidths=0)
    mean, sd = np.mean(vals), np.std(vals, ddof=1)
    p = _wilcoxon_p(vals, alternative="less")
    ax.errorbar([0], [mean], yerr=[sd], fmt="D", color="black",
                markersize=5, capsize=4, linewidth=1.2, zorder=5,
                label=f"{mean:+.3f}±{sd:.3f}")
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks([0])
    ax.set_xticklabels(["A→B\n(head ON)"])
    ax.set_title(f"{_fmt_p(p)}", fontsize=6, pad=2)
    ax.set_ylabel(r"Mean $\Delta$ COM dist to reward")
    ax.legend(fontsize=5, loc="lower right")
    fig.suptitle(f"COM shift per seed (n={len(rows)} seeds)", fontsize=6.5, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4)
    return save_figure(fig, stem)


def fig_supp_frac_strips(rows, stem):
    """Panel b — per-seed fraction toward reward A→B."""
    vals = np.array([r["com_frac_toward_AB"] * 100 for r in rows])
    rng = np.random.default_rng(1)

    fig, ax = plt.subplots(figsize=(2.5, 2.2))
    jitter = rng.uniform(-0.08, 0.08, len(vals))
    ax.scatter(jitter, vals, s=14, color="#CC6677", alpha=0.75, zorder=3, linewidths=0)
    mean, sd = np.mean(vals), np.std(vals, ddof=1)
    ax.errorbar([0], [mean], yerr=[sd], fmt="D", color="black",
                markersize=5, capsize=4, linewidth=1.2, zorder=5,
                label=f"{mean:.1f}±{sd:.1f}%")
    ax.axhline(50, color="0.4", lw=0.8, ls="--", label="chance")
    ax.set_xticks([0])
    ax.set_xticklabels(["A→B\n(head ON)"])
    ax.set_ylabel("% cells toward reward")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=5, loc="lower right")
    fig.suptitle(f"Fraction toward reward per seed (n={len(rows)} seeds)", fontsize=6.5, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4)
    return save_figure(fig, stem)


def fig_supp_pooled_violin(rows, stem):
    """Panel c — pooled violin of Δd A→B with per-seed means."""
    all_ab = np.concatenate([r["delta_ab"] for r in rows])
    per_ab = np.array([r["com_mean_delta_AB"] for r in rows])
    p_ab = _wilcoxon_p(per_ab, alternative="less")

    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    parts = ax.violinplot([all_ab], positions=[1], showmeans=False, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#8B4513")
        body.set_alpha(0.55)
        body.set_edgecolor("0.2")
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("0.25")
            parts[key].set_linewidth(0.9)

    jitter = np.random.default_rng(2).uniform(-0.07, 0.07, len(rows))
    ax.scatter(np.ones(len(rows)) + jitter, per_ab, s=14, color="#5C2900",
               zorder=5, linewidths=0, alpha=0.9, label="seed mean")
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks([1])
    ax.set_xticklabels([f"A→B\n{_fmt_p(p_ab)}"])
    ax.set_ylabel(r"$\Delta$ COM dist to reward")
    ax.set_title(f"Pooled ({len(rows)} seeds, n={len(all_ab)} pairs)", fontsize=5.5, pad=2)
    ax.legend(fontsize=5)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_supp_head_vs_nohead(head_rows, nohead_rows, stem):
    """Panel d — paired head vs no-head per-seed Δd(A→B): the double dissociation.

    Each seed is shown as a line connecting its no-head and head Δd values.
    """
    # Match by seed
    head_dict = {r["seed"]: r["com_mean_delta_AB"] for r in head_rows}
    nohead_dict = {r["seed"]: r["com_mean_delta_AB"] for r in nohead_rows}
    common_seeds = sorted(set(head_dict) & set(nohead_dict))

    if not common_seeds:
        print("  WARNING: no common seeds between head and no-head; skipping panel d")
        return None, None

    vals_head = np.array([head_dict[s] for s in common_seeds])
    vals_nohead = np.array([nohead_dict[s] for s in common_seeds])

    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    rng = np.random.default_rng(3)

    # Draw lines connecting paired seeds
    for vn, vh in zip(vals_nohead, vals_head):
        ax.plot([1, 2], [vn, vh], color="0.6", lw=0.8, alpha=0.6)

    jitter = rng.uniform(-0.06, 0.06, len(common_seeds))
    ax.scatter(np.ones(len(common_seeds)) + jitter, vals_nohead,
               s=16, color="#4C72B0", zorder=5, linewidths=0, alpha=0.85, label="no head")
    ax.scatter(np.ones(len(common_seeds)) * 2 + jitter, vals_head,
               s=16, color="#8B4513", zorder=5, linewidths=0, alpha=0.85, label="head ON")

    # Grand means
    ax.errorbar([1], [vals_nohead.mean()], yerr=[vals_nohead.std(ddof=1)],
                fmt="D", color="#1A3A6E", markersize=6, capsize=4, linewidth=1.2, zorder=6)
    ax.errorbar([2], [vals_head.mean()], yerr=[vals_head.std(ddof=1)],
                fmt="D", color="#5C2900", markersize=6, capsize=4, linewidth=1.2, zorder=6)

    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["No head", "Head ON"])
    ax.set_ylabel(r"Mean $\Delta$ COM dist to reward (A→B)")
    ax.set_title(
        f"Quick-look double dissociation (n={len(common_seeds)} seeds, A→B, occupancy-confounded)\n"
        f"See Fig. 5b for the entropy-gated frozen-replay diagnostic (definitive H1/H2 test)",
        fontsize=5.3, pad=2,
    )
    ax.legend(fontsize=5, loc="upper left")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------

def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300):
    """Render a one-page PDF to PNG at `dpi`, via Poppler's pdftoppm."""
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), str(out_prefix)],
            check=True,
        )
        copy2(out_prefix.with_suffix(".png"), png_path)


def assemble_supp(panel_pdfs: dict[str, Path], out_stem: Path, dpi: int = 300):
    """Place each panel PDF in its slot and write the composite page as PDF and PNG."""
    out_stem = Path(out_stem).with_suffix("")
    pdf_path = out_stem.with_suffix(".pdf")
    png_path = out_stem.with_suffix(".png")

    MARGIN = 10 * MM_TO_PT
    GAP = 5 * MM_TO_PT

    def get_size(pdf):
        box = PdfReader(str(pdf)).pages[0].mediabox
        return float(box.width), float(box.height)

    sizes = {k: get_size(panel_pdfs[k]) for k in panel_pdfs}

    def row_height(keys):
        ncols = len(keys)
        cw = (PAGE_W - 2 * MARGIN - (ncols - 1) * GAP) / ncols
        return max((sizes[k][1] * (cw / max(sizes[k][0], 1e-6)) for k in keys), default=0.0)

    rows_def = [("a", "b"), ("c", "d")]
    row_keys = [[k for k in row if k in panel_pdfs] for row in rows_def]
    row_keys = [r for r in row_keys if r]
    rhs = [row_height(r) for r in row_keys]
    page_h = 2 * MARGIN + sum(rhs) + (len(rhs) - 1) * GAP

    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W, height=page_h)

    y = MARGIN
    for keys, rh in zip(reversed(row_keys), reversed(rhs)):
        ncols = len(keys)
        cw = (PAGE_W - 2 * MARGIN - (ncols - 1) * GAP) / ncols
        for i, k in enumerate(keys):
            iw, ih = sizes[k]
            scale = min(cw / iw, rh / ih)
            sw, sh = iw * scale, ih * scale
            left = MARGIN + i * (cw + GAP) + (cw - sw) / 2
            bottom = y + (rh - sh) / 2
            src = PdfReader(str(panel_pdfs[k])).pages[0]
            page.merge_transformed_page(src, Transformation().scale(scale).translate(left, bottom), over=True)
        y += rh + GAP

    with open(pdf_path, "wb") as f:
        writer.write(f)
    _pdf_to_png(pdf_path, png_path, dpi=dpi)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    return png_path, pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    """Command-line arguments: multi-seed directories, seed list, output directory."""
    p = argparse.ArgumentParser()
    p.add_argument("--multi_seed_dir", default=str(REPO / "results" / "multi_seed_head"),
                   help="Root dir for head-condition seeds (head_seed{N}/)")
    p.add_argument("--nohead_dir", default=str(REPO / "results" / "multi_seed"),
                   help="Root dir for no-head seeds (nohead_seed{N}/) for panel d")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(50)))
    p.add_argument("--skaggs_threshold", type=float, default=0.05)
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c,d")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    add_plot_only(p)
    args = p.parse_args()
    args.panels = {x.strip() for x in args.panels.split(",") if x.strip()}
    return args


def main():
    """Load every seed, build the panels, and assemble the supplementary figure."""
    args = parse_args()
    multi_seed_dir = Path(args.multi_seed_dir)
    nohead_dir = Path(args.nohead_dir)
    panels_dir = ensure_dir(Path(args.output_dir) / "panels")
    ensure_dir(Path(args.output_dir))

    cache_path = Path(args.output_dir) / "figure_data.npz"
    if args.plot_only:
        saved = require_keys(
            load_figure_data(cache_path), "head_rows", "nohead_rows", path=cache_path)
        head_rows = saved["head_rows"]
        nohead_rows = saved["nohead_rows"]
        print(f"Plot-only: {len(head_rows)} head, {len(nohead_rows)} no-head")
    else:
        print(f"Loading {len(args.seeds)} head-condition seeds from {multi_seed_dir} ...")
        head_rows = []
        for seed in args.seeds:
            r = load_seed(seed, multi_seed_dir, prefix="head_", skaggs_threshold=args.skaggs_threshold)
            if r is None:
                print(f"  seed {seed}: skip (missing/failed)")
                continue
            print(f"  seed {seed}: n={r['n_cells']}  Δd(A→B)={r['com_mean_delta_AB']:+.3f} ({100*r['com_frac_toward_AB']:.0f}% toward)")
            head_rows.append(r)

        nohead_rows = []
        print(f"\nLoading no-head seeds from {nohead_dir} (for panel d) ...")
        for seed in args.seeds:
            r = load_seed(seed, nohead_dir, prefix="nohead_", skaggs_threshold=args.skaggs_threshold)
            if r is not None:
                nohead_rows.append(r)
        if not nohead_rows:
            print("  No no-head seeds found — panel d will be skipped.")
        update_figure_data(cache_path, {"head_rows": head_rows, "nohead_rows": nohead_rows})

    if not head_rows:
        print("ERROR: no head seeds loaded.")
        sys.exit(1)

    panel_pdfs: dict[str, Path] = {}

    if "a" in args.panels:
        print("\nSupp panel a (Δd strip chart A→B, head ON)...")
        _, pdf = fig_supp_delta_strips(head_rows, panels_dir / "panel_a")
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Supp panel b (fraction toward reward, head ON)...")
        _, pdf = fig_supp_frac_strips(head_rows, panels_dir / "panel_b")
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Supp panel c (pooled violin A→B)...")
        _, pdf = fig_supp_pooled_violin(head_rows, panels_dir / "panel_c")
        panel_pdfs["c"] = pdf

    if "d" in args.panels:
        if nohead_rows:
            print("Supp panel d (head vs no-head paired comparison)...")
            result = fig_supp_head_vs_nohead(head_rows, nohead_rows, panels_dir / "panel_d")
            if result[1] is not None:
                panel_pdfs["d"] = result[1]
        else:
            print("Supp panel d: skipped (no no-head seeds available)")

    if not args.no_assemble and panel_pdfs:
        print("\nAssembling supplementary figure PDF...")
        assemble_supp(panel_pdfs, Path(args.output_dir) / "figure3_supp_raw")

    print(f"\nDone. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
