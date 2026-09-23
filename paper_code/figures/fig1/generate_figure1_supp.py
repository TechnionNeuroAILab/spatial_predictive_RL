#!/usr/bin/env python3
"""
Supplementary Figure 1 — aggregate statistics across all seeds (for Figure 1).

Panels:
  a  # shared place cells (A∩B, Skaggs ≥ threshold) per seed + mean ± SD
  b  Mean Skaggs info (Phase A) per seed + mean ± SD
  c  Mean COM Δd (A→B) per seed + grand mean ± SD — the RL remapping signal
  d  Fraction of cells shifting toward reward (A→B) per seed + mean ± SD
  e  Pooled violin: COM distance-to-reward (Phase A vs B) across all seeds
  f  Skaggs-threshold sensitivity: place-cell count and mean Δd vs threshold
     (justifies the inconsistent 0.05/0.15/0.3 thresholds used across figures)

Usage (run from repo root):
    python paper_code/figures/fig1/generate_figure1_supp.py --no_show
    python paper_code/figures/fig1/generate_figure1_supp.py \
        --multi_seed_dir Repro/multi_seed --seeds 0 1 2 ... 49 --no_show
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
FIG2_DIR = REPO / "paper_code" / "figures" / "fig2"
FIG1_DIR = REPO / "paper_code" / "figures" / "fig1"

# fig2 first: `analysis` is fig2's (fig1 has none) and must execute with fig2's
# utils, which alone defines PHASE_A/PHASE_B/PHASE_C_*. fig1 stays on the path
# because fig2/analysis.py imports fig1's rate_maps.
sys.path.insert(0, str(FIG1_DIR))
sys.path.insert(0, str(FIG2_DIR))

from analysis import load_phases, shared_place_cell_ids, paired_distance_stats, dist_to_reward  # noqa: E402

# fig1's utils, loaded by path on purpose. A bare `from utils import ...` here
# returns fig2's copy instead: importing `analysis` above already bound the name
# "utils" to it in sys.modules, so neither this file's sys.path order nor the
# import system would ever reach fig1/utils.py.
_spec = importlib.util.spec_from_file_location("fig1_utils", FIG1_DIR / "utils.py")
_fig1_utils = importlib.util.module_from_spec(_spec)
sys.modules["fig1_utils"] = _fig1_utils
_spec.loader.exec_module(_fig1_utils)

DEFAULT_REWARD = _fig1_utils.DEFAULT_REWARD
apply_nature_style = _fig1_utils.apply_nature_style
ensure_dir = _fig1_utils.ensure_dir
save_figure = _fig1_utils.save_figure
PAGE_W = _fig1_utils.PAGE_W
PAGE_MARGIN = _fig1_utils.PAGE_MARGIN
MM_TO_PT = _fig1_utils.MM_TO_PT

apply_nature_style()

SUPP_FIGURES_DIR = FIG1_DIR / "figures_supp"
REWARD_POS = np.asarray(DEFAULT_REWARD, dtype=float)

PHASE_A = "cycle_1"
PHASE_B = "cycle_1_goal"
PHASE_C_POST = "replay_c1_postrl"
PHASE_C_PRE = "replay_c1_pred"


# ---------------------------------------------------------------------------
# Per-seed data loading
# ---------------------------------------------------------------------------

SKAGGS_SWEEP_THRESHOLDS = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3]


def _threshold_sensitivity(phases: dict, thresholds: list[float]) -> dict:
    """Cheap re-gating at multiple Skaggs thresholds, reusing already-loaded
    phases (COM/Skaggs arrays) -- no rate-map recomputation."""
    com_a_full = phases["A"]["com"]
    com_b_full = phases["B"]["com"]
    sk_a = phases["A"]["sk"]
    sk_b = phases["B"]["sk"]
    out = {}
    for thr in thresholds:
        mask = (sk_a >= thr) & (sk_b >= thr)
        n = int(mask.sum())
        if n == 0:
            out[thr] = {"n_cells": 0, "mean_delta_ab": float("nan")}
            continue
        d_a = dist_to_reward(com_a_full[mask], REWARD_POS)
        d_b = dist_to_reward(com_b_full[mask], REWARD_POS)
        out[thr] = {"n_cells": n, "mean_delta_ab": float(np.mean(d_b - d_a))}
    return out


def load_seed(seed: int, multi_seed_dir: Path, skaggs_threshold: float = 0.05,
              sweep_thresholds: list[float] | None = None):
    """Per-seed summary row (cell counts, Skaggs, A->B distances); None if the run is missing or fails."""
    run_dir = multi_seed_dir / f"nohead_seed{seed}"
    data_dir = run_dir / "data"
    if not data_dir.exists():
        return None
    try:
        phases = load_phases(data_dir)
        ids = shared_place_cell_ids(phases, skaggs_threshold=skaggs_threshold)
        stats = paired_distance_stats(phases, REWARD_POS, ids)

        # Mean Skaggs in phase A
        sk_a = phases["A"]["sk"]
        mean_skaggs = float(np.nanmean(sk_a[sk_a >= skaggs_threshold])) if (sk_a >= skaggs_threshold).any() else float("nan")

        # COM distances A and B for all shared cells
        com_dist_a = np.asarray(stats["com"]["A"], dtype=float)
        com_dist_b = np.asarray(stats["com"]["B"], dtype=float)

        row = {
            "seed": seed,
            "n_cells": int(len(ids)),
            "mean_skaggs_a": mean_skaggs,
            "com_mean_delta_AB": float(stats["com_mean_delta_AB"]),
            "com_frac_toward_AB": float(stats["com_frac_toward_AB"]),
            "com_dist_a": com_dist_a,
            "com_dist_b": com_dist_b,
            "delta_ab": np.asarray(stats["delta_com_AB"], dtype=float),
        }
        if sweep_thresholds:
            row["threshold_sensitivity"] = _threshold_sensitivity(phases, sweep_thresholds)
        return row
    except Exception as exc:
        print(f"  [seed {seed}] error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def _strip_panel(ax, values, seeds, ylabel, title, hline=None, hline_label="chance", ymin=None, ymax=None):
    """Shared strip-chart + mean±SD layout."""
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, len(values))
    ax.scatter(jitter, values, s=14, color="#2E5FA3", alpha=0.75, zorder=3, linewidths=0)
    mean, sd = np.mean(values), np.std(values, ddof=1)
    ax.errorbar([0], [mean], yerr=[sd], fmt="D", color="black",
                markersize=5, capsize=4, linewidth=1.2, zorder=5,
                label=f"mean±SD\n{mean:.3g}±{sd:.3g}")
    if hline is not None:
        ax.axhline(hline, color="0.45", lw=0.8, ls="--", label=hline_label)
    ax.set_xticks([0])
    ax.set_xticklabels([f"n={len(seeds)} seeds"])
    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_title(title, fontsize=6, pad=2)
    ax.legend(fontsize=5, loc="best")
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)


def fig_supp_n_cells(rows, stem):
    """Panel a -- number of shared place cells per seed."""
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    vals = np.array([r["n_cells"] for r in rows], dtype=float)
    seeds = [r["seed"] for r in rows]
    _strip_panel(ax, vals, seeds, "# shared place cells\n(A∩B, Skaggs ≥ 0.05)", "Place cell count")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_supp_skaggs(rows, stem):
    """Panel b -- mean phase-A Skaggs information per seed."""
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    vals = np.array([r["mean_skaggs_a"] for r in rows], dtype=float)
    seeds = [r["seed"] for r in rows]
    _strip_panel(ax, vals, seeds, "Mean Skaggs info (Phase A)\n[bits/spike]", "Spatial information")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_supp_delta_com(rows, stem):
    """Panel c -- mean COM Delta-d for the A->B contrast per seed, with a Wilcoxon against zero."""
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    vals = np.array([r["com_mean_delta_AB"] for r in rows], dtype=float)
    seeds = [r["seed"] for r in rows]

    # Wilcoxon vs 0
    try:
        _, wp = _wilcoxon(vals, alternative="less")
        p_str = f"p={wp:.3f}" if wp >= 0.001 else f"p={wp:.1e}"
    except Exception:
        p_str = ""

    _strip_panel(ax, vals, seeds, r"Mean $\Delta$ COM dist to reward (A→B)", f"COM shift\n{p_str}", hline=0, hline_label="no shift")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_supp_frac_toward(rows, stem):
    """Panel d -- fraction of cells moving toward the reward in A->B, per seed."""
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    vals = np.array([r["com_frac_toward_AB"] * 100 for r in rows], dtype=float)
    seeds = [r["seed"] for r in rows]
    _strip_panel(ax, vals, seeds, "% cells toward reward (A→B)", "Fraction toward", hline=50, hline_label="chance (50%)", ymin=0, ymax=100)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_supp_pooled_violin(rows, stem):
    """Panel e — pooled COM distance-to-reward (A vs B) as paired violin."""
    all_a = np.concatenate([r["com_dist_a"] for r in rows])
    all_b = np.concatenate([r["com_dist_b"] for r in rows])
    per_seed_mean_a = np.array([r["com_dist_a"].mean() for r in rows])
    per_seed_mean_b = np.array([r["com_dist_b"].mean() for r in rows])

    # Wilcoxon on pooled paired differences
    delta = all_b - all_a
    try:
        _, wp = _wilcoxon(delta, alternative="less")
        p_str = f"p={wp:.1e}" if wp < 0.001 else f"p={wp:.3f}"
    except Exception:
        p_str = ""

    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    parts = ax.violinplot([all_a, all_b], positions=[1, 2],
                          showmeans=True, showmedians=False, showextrema=True)
    colors = ["#56B4E9", "#E69F00"]
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.6)
        body.set_edgecolor("0.2")
    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        if key in parts:
            parts[key].set_color("0.25")
            parts[key].set_linewidth(0.9)

    # Per-seed means as dots with lines connecting A and B
    rng = np.random.default_rng(1)
    jitter = rng.uniform(-0.06, 0.06, len(rows))
    for i, (ma, mb) in enumerate(zip(per_seed_mean_a, per_seed_mean_b)):
        ax.plot([1 + jitter[i], 2 + jitter[i]], [ma, mb], color="0.5", lw=0.5, alpha=0.6)
    ax.scatter(np.ones(len(rows)) + jitter, per_seed_mean_a,
               s=12, color="#0072B2", zorder=5, linewidths=0, alpha=0.9)
    ax.scatter(np.ones(len(rows)) * 2 + jitter, per_seed_mean_b,
               s=12, color="#D55E00", zorder=5, linewidths=0, alpha=0.9)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["A\nprediction", "B\nRL policy"])
    ax.set_ylabel("COM distance to reward")
    n_total = len(all_a)
    ax.set_title(f"Pooled: {len(rows)} seeds, {n_total} cell-pairs\n{p_str} (Wilcoxon A vs B)", fontsize=5.5, pad=2)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_supp_skaggs_sensitivity(rows, thresholds, stem):
    """Panel f — Skaggs-threshold sensitivity: place-cell count and mean Δd
    (A→B) as a function of the gating threshold, aggregated across all seeds.
    Justifies why thresholds differ across figures (Figs 1-3 use 0.05,
    Fig 4 uses 0.15, Fig 5 uses 0.3): shows whether the headline effects are
    robust to this choice."""
    n_means, n_sds = [], []
    d_means, d_sds = [], []
    for thr in thresholds:
        n_vals = np.array([r["threshold_sensitivity"][thr]["n_cells"] for r in rows], dtype=float)
        d_vals = np.array([r["threshold_sensitivity"][thr]["mean_delta_ab"] for r in rows], dtype=float)
        n_means.append(np.nanmean(n_vals))
        n_sds.append(np.nanstd(n_vals, ddof=1))
        d_means.append(np.nanmean(d_vals))
        d_sds.append(np.nanstd(d_vals, ddof=1))

    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.2))

    axes[0].errorbar(thresholds, n_means, yerr=n_sds, fmt="o-", color="#2E5FA3",
                      capsize=3, lw=1.0, markersize=4)
    axes[0].set_xlabel("Skaggs threshold")
    axes[0].set_ylabel("# place cells\n(mean ± SD)")
    axes[0].set_title("Place-cell count\nvs threshold", fontsize=6, pad=2)

    axes[1].errorbar(thresholds, d_means, yerr=d_sds, fmt="o-", color="#D55E00",
                      capsize=3, lw=1.0, markersize=4)
    axes[1].axhline(0, color="0.4", lw=0.7, ls="--")
    axes[1].set_xlabel("Skaggs threshold")
    axes[1].set_ylabel(r"Mean $\Delta$ COM (A→B)" + "\n(mean ± SD)")
    axes[1].set_title("Remapping effect\nvs threshold", fontsize=6, pad=2)

    n = len(rows)
    fig.suptitle(f"Skaggs-threshold sensitivity (n={n} seeds)", fontsize=6.5, y=1.03)
    fig.tight_layout(pad=0.4, w_pad=0.6)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# PDF assembly (simple row layout)
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
    """Two-row layout: [a b c] / [d e f]."""
    out_stem = Path(out_stem).with_suffix("")
    pdf_path = out_stem.with_suffix(".pdf")
    png_path = out_stem.with_suffix(".png")

    MARGIN = 10 * MM_TO_PT
    GAP = 5 * MM_TO_PT
    keys_row0 = [k for k in ("a", "b", "c") if k in panel_pdfs]
    keys_row1 = [k for k in ("d", "e", "f") if k in panel_pdfs]

    def get_size(pdf):
        box = PdfReader(str(pdf)).pages[0].mediabox
        return float(box.width), float(box.height)

    sizes = {k: get_size(panel_pdfs[k]) for k in panel_pdfs}
    col_w = (PAGE_W - 2 * MARGIN - (3 - 1) * GAP) / 3

    def row_height(keys):
        hs = []
        ncols = len(keys)
        cw = (PAGE_W - 2 * MARGIN - (ncols - 1) * GAP) / ncols
        for k in keys:
            iw, ih = sizes[k]
            hs.append(ih * (cw / max(iw, 1e-6)))
        return max(hs) if hs else 0.0

    rh0 = row_height(keys_row0)
    rh1 = row_height(keys_row1)
    page_h = 2 * MARGIN + rh0 + (GAP if keys_row1 else 0) + rh1

    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W, height=page_h)

    def place_row(keys, row_y, row_h):
        ncols = len(keys)
        if ncols == 0:
            return
        cw = (PAGE_W - 2 * MARGIN - (ncols - 1) * GAP) / ncols
        for i, k in enumerate(keys):
            iw, ih = sizes[k]
            scale = min(cw / iw, row_h / ih)
            sw, sh = iw * scale, ih * scale
            left = MARGIN + i * (cw + GAP) + (cw - sw) / 2
            bottom = row_y + (row_h - sh) / 2
            src = PdfReader(str(panel_pdfs[k])).pages[0]
            page.merge_transformed_page(
                src,
                Transformation().scale(scale).translate(left, bottom),
                over=True,
            )

    # Row 1 at top (row0), row 2 below
    place_row(keys_row0, MARGIN + (GAP if keys_row1 else 0) + rh1, rh0)
    place_row(keys_row1, MARGIN, rh1)

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
    """Command-line arguments: multi-seed directory, seed list, output directory."""
    p = argparse.ArgumentParser()
    p.add_argument("--multi_seed_dir", default=str(REPO / "Repro" / "multi_seed"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(50)))
    p.add_argument("--skaggs_threshold", type=float, default=0.05)
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c,d,e,f")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    args = p.parse_args()
    args.panels = {x.strip() for x in args.panels.split(",") if x.strip()}
    return args


def main():
    """Load every seed, build the panels, and assemble the supplementary figure."""
    args = parse_args()
    multi_seed_dir = Path(args.multi_seed_dir)
    panels_dir = ensure_dir(Path(args.output_dir) / "panels")
    ensure_dir(Path(args.output_dir))

    need_sweep = "f" in args.panels
    sweep_thresholds = SKAGGS_SWEEP_THRESHOLDS if need_sweep else None

    print(f"Loading {len(args.seeds)} seeds from {multi_seed_dir} ...")
    rows = []
    for seed in args.seeds:
        r = load_seed(seed, multi_seed_dir, skaggs_threshold=args.skaggs_threshold,
                      sweep_thresholds=sweep_thresholds)
        if r is None:
            print(f"  seed {seed}: skip (missing/failed)")
            continue
        print(f"  seed {seed}: n={r['n_cells']}  skaggs={r['mean_skaggs_a']:.3f}  "
              f"Δd={r['com_mean_delta_AB']:+.3f}  frac_toward={100*r['com_frac_toward_AB']:.0f}%")
        rows.append(r)

    if not rows:
        print("ERROR: no seeds loaded.")
        sys.exit(1)

    panel_pdfs: dict[str, Path] = {}

    if "a" in args.panels:
        print("Supp panel a (# place cells)...")
        _, pdf = fig_supp_n_cells(rows, panels_dir / "panel_a")
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Supp panel b (mean Skaggs)...")
        _, pdf = fig_supp_skaggs(rows, panels_dir / "panel_b")
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Supp panel c (mean COM Δd A→B)...")
        _, pdf = fig_supp_delta_com(rows, panels_dir / "panel_c")
        panel_pdfs["c"] = pdf

    if "d" in args.panels:
        print("Supp panel d (fraction toward reward)...")
        _, pdf = fig_supp_frac_toward(rows, panels_dir / "panel_d")
        panel_pdfs["d"] = pdf

    if "e" in args.panels:
        print("Supp panel e (pooled COM distance violin)...")
        _, pdf = fig_supp_pooled_violin(rows, panels_dir / "panel_e")
        panel_pdfs["e"] = pdf

    if "f" in args.panels:
        print("Supp panel f (Skaggs-threshold sensitivity)...")
        _, pdf = fig_supp_skaggs_sensitivity(rows, SKAGGS_SWEEP_THRESHOLDS, panels_dir / "panel_f")
        panel_pdfs["f"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("Assembling supplementary figure PDF...")
        assemble_supp(panel_pdfs, Path(args.output_dir) / "figure1_supp_raw")

    print(f"\nDone. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
