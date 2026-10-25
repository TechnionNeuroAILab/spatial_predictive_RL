#!/usr/bin/env python3
"""
Supplementary Figure 2 — aggregate statistics across all seeds (for Figure 2).

Figure 2 asks: does remapping come from exploration (A→B) or just weight updates
on the same frozen trajectory (C:pre→post)? This supplement shows that contrast
generalises across all seeds.

Panels:
  a  Per-seed mean COM Δd: A→B and C:pre→post side-by-side strip charts
  b  Per-seed fraction toward reward: A→B vs C:pre→post strip charts
  c  Pooled violin: Δd for both contrasts (with per-seed medians)
  d  Per-seed scatter: Δd(A→B) vs Δd(C:pre→post) — shows they are decorrelated
  e  Peak vs COM convergent-validity scatter for the corrected C:pre→post contrast

Usage (run from repo root):
    python code/figures/fig2/generate_figure2_supp.py --no_show
    python code/figures/fig2/generate_figure2_supp.py \
        --multi_seed_dir results/multi_seed --seeds 0 1 2 ... 49 --no_show
"""
from __future__ import annotations

import argparse
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
FIG2_DIR = REPO / "code" / "figures" / "fig2"
FIG1_DIR = REPO / "code" / "figures" / "fig1"

sys.path.insert(0, str(FIG1_DIR))
sys.path.insert(0, str(FIG2_DIR))

from analysis import load_phases, shared_place_cell_ids, paired_distance_stats  # noqa: E402
from utils import (  # noqa: E402
    DEFAULT_REWARD, FIGURES_DIR, apply_nature_style, ensure_dir, save_figure,
    PAGE_W, MM_TO_PT, FS_BASE, FS_SMALL, FS_TICK, title, stat_annot,
    fmt_p,
    C,
)
import figlayout  # noqa: E402
from figure_cache import add_plot_only, load_figure_data, require_keys, update_figure_data  # noqa: E402

apply_nature_style()

SUPP_FIGURES_DIR = FIG2_DIR / "figures_supp"
REWARD_POS = np.asarray(DEFAULT_REWARD, dtype=float)


# ---------------------------------------------------------------------------
# Per-seed data loading
# ---------------------------------------------------------------------------

def load_seed(seed: int, multi_seed_dir: Path, skaggs_threshold: float = 0.05):
    """Per-seed summary row for the A->B and frozen-replay contrasts; None if the run is missing or fails."""
    run_dir = multi_seed_dir / f"nohead_seed{seed}"
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
            "n_cells_ac": int(stats["n_ac"]),
            "com_mean_delta_AB": float(stats["com_mean_delta_AB"]),
            "com_mean_delta_AC": float(stats["com_mean_delta_AC"]),
            "com_frac_toward_AB": float(stats["com_frac_toward_AB"]),
            "com_frac_toward_AC": float(stats["com_frac_toward_AC"]),
            "peak_mean_delta_AC": float(stats["peak_mean_delta_AC"]),
            "peak_frac_toward_AC": float(stats["peak_frac_toward_AC"]),
            "delta_ab": np.asarray(stats["delta_com_AB"], dtype=float),
            "delta_ac": np.asarray(stats["delta_com_AC"], dtype=float),
        }
    except Exception as exc:
        print(f"  [seed {seed}] error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def _wilcoxon_p(vals, alternative="less"):
    """One-sided Wilcoxon p (default alternative="less": shift toward reward); NaN if it cannot be computed."""
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


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
# Each panel is authored at its slot size, so the assembler has nothing left to
# scale. Previously the panels were drawn at hand-picked inch sizes and
# contain-fitted, which is what put this figure's type below the 5 pt floor.
#
#   a  per-network COM shift, both contrasts   b  fraction toward reward
#   c  pooled distributions   d  cross-contrast scatter   e  peak vs centre of mass
LAYOUT = figlayout.Layout(
    [("a", 0, 0, 3), ("b", 0, 3, 3),
     ("c", 1, 0, 2), ("d", 1, 2, 2), ("e", 1, 4, 2)],
    ncols=6, row_h_in=[2.10, 2.20], caption_lines=4,
    name="figS_frozenreplay",
)


def fig_supp_delta_strips(rows, stem, size_in=None):
    """Panel a — per-seed mean Δd for A→B and C:pre→post, side by side."""
    vals_ab = np.array([r["com_mean_delta_AB"] for r in rows])
    vals_ac = np.array([r["com_mean_delta_AC"] for r in rows])
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=size_in or (4.2, 2.2), sharey=True)

    for ax, vals, color, label, alt in [
        (axes[0], vals_ab, C["policy"], "A→B\n(RL policy)", "less"),
        (axes[1], vals_ac, C["frozen_post"], "C:pre→post\n(frozen replay)", "two-sided"),
    ]:
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(jitter, vals, s=14, color=color, alpha=0.75, zorder=3, linewidths=0)
        mean, sd = np.mean(vals), np.std(vals, ddof=1)
        ax.errorbar([0], [mean], yerr=[sd], fmt="D", color="black",
                    markersize=5, capsize=4, linewidth=1.2, zorder=5,
                    label=f"{mean:+.3f}±{sd:.3f}")
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        p = _wilcoxon_p(vals, alternative=alt)
        ax.set_xticks([0])
        ax.set_xticklabels([label])
        ax.set_title(f"{_fmt_p(p)}", fontsize=FS_TICK, pad=2)
        ax.legend(fontsize=FS_SMALL, loc="best")

    axes[0].set_ylabel(r"Mean $\Delta$ COM dist to reward")
    n = len(rows)
    fig.suptitle(f"COM shift per contrast (n={n} seeds)", fontsize=FS_BASE, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4, w_pad=0.5)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_supp_frac_strips(rows, stem, size_in=None):
    """Panel b — per-seed fraction toward reward for A→B and C:pre→post."""
    vals_ab = np.array([r["com_frac_toward_AB"] * 100 for r in rows])
    vals_ac = np.array([r["com_frac_toward_AC"] * 100 for r in rows])
    rng = np.random.default_rng(1)

    fig, axes = plt.subplots(1, 2, figsize=size_in or (4.2, 2.2), sharey=True)

    for ax, vals, color, label in [
        (axes[0], vals_ab, C["policy"], "A→B\n(RL policy)"),
        (axes[1], vals_ac, C["frozen_post"], "C:pre→post\n(frozen replay)"),
    ]:
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(jitter, vals, s=14, color=color, alpha=0.75, zorder=3, linewidths=0)
        mean, sd = np.mean(vals), np.std(vals, ddof=1)
        ax.errorbar([0], [mean], yerr=[sd], fmt="D", color="black",
                    markersize=5, capsize=4, linewidth=1.2, zorder=5,
                    label=f"{mean:.1f}±{sd:.1f}%")
        ax.axhline(50, color="0.4", lw=0.8, ls="--", label="chance")
        ax.set_xticks([0])
        ax.set_xticklabels([label])
        ax.legend(fontsize=FS_SMALL, loc="best")

    axes[0].set_ylabel("% cells toward reward")
    axes[0].set_ylim(0, 100)
    n = len(rows)
    fig.suptitle(f"Fraction toward reward per contrast (n={n} seeds)", fontsize=FS_BASE, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4, w_pad=0.5)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_supp_pooled_violin(rows, stem, size_in=None):
    """Panel c — pooled violin of Δd for A→B and C:pre→post with per-seed means."""
    all_ab = np.concatenate([r["delta_ab"] for r in rows])
    all_ac = np.concatenate([r["delta_ac"] for r in rows])
    per_ab = np.array([r["com_mean_delta_AB"] for r in rows])
    per_ac = np.array([r["com_mean_delta_AC"] for r in rows])

    p_ab = _wilcoxon_p(per_ab, alternative="less")
    p_ac = _wilcoxon_p(per_ac, alternative="two-sided")

    fig, ax = plt.subplots(figsize=size_in or (3.0, 2.4))
    parts = ax.violinplot([all_ab, all_ac], positions=[1, 2],
                          showmeans=False, showmedians=True, showextrema=True)
    colors = [C["policy"], C["frozen_post"]]
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.55)
        body.set_edgecolor("0.2")
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("0.25")
            parts[key].set_linewidth(0.9)

    rng = np.random.default_rng(2)
    jitter = rng.uniform(-0.07, 0.07, len(rows))
    # Connect per-seed means with a line
    for ma, mc, j in zip(per_ab, per_ac, jitter):
        ax.plot([1 + j, 2 + j], [ma, mc], color="0.5", lw=0.5, alpha=0.5)
    ax.scatter(np.ones(len(rows)) + jitter, per_ab, s=14, color=C["policy"],
               zorder=5, linewidths=0, alpha=0.9, label="seed mean")
    ax.scatter(np.ones(len(rows)) * 2 + jitter, per_ac, s=14, color=C["frozen_post"],
               zorder=5, linewidths=0, alpha=0.9)

    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"A→B\n{_fmt_p(p_ab)}", f"C:pre→post\n{_fmt_p(p_ac)}"])
    ax.set_ylabel(r"$\Delta$ COM dist to reward")
    n_ab, n_ac = len(all_ab), len(all_ac)
    ax.set_title(f"Pooled (A→B: {n_ab} pairs; C:pre→post: {n_ac} pairs)", fontsize=FS_SMALL, pad=2)
    ax.legend(fontsize=FS_SMALL)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_supp_scatter_ab_vs_ac(rows, stem, size_in=None):
    """Panel d — scatter of per-seed Δd(A→B) vs Δd(C:pre→post)."""
    vals_ab = np.array([r["com_mean_delta_AB"] for r in rows])
    vals_ac = np.array([r["com_mean_delta_AC"] for r in rows])
    seeds = [r["seed"] for r in rows]

    fig, ax = plt.subplots(figsize=size_in or (2.4, 2.4))
    ax.scatter(vals_ac, vals_ab, s=22, color=C["pre"], alpha=0.8, zorder=3, linewidths=0.5, edgecolors="0.3")
    for v_ac, v_ab, s in zip(vals_ac, vals_ab, seeds):
        ax.text(v_ac, v_ab, str(s), fontsize=FS_SMALL, ha="left", va="bottom", color="0.35")
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.axvline(0, color="0.4", lw=0.7, ls="--")
    lim = max(abs(np.concatenate([vals_ab, vals_ac]))) * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\Delta$ COM (C:pre→post, frozen replay)", fontsize=FS_TICK)
    ax.set_ylabel(r"$\Delta$ COM (A→B, RL policy)", fontsize=FS_TICK)
    ax.set_title("Per-seed shifts:\nA→B (exploration) vs C:pre→post (weight update)", fontsize=FS_SMALL, pad=2)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_supp_peak_vs_com(rows, stem, size_in=None):
    """Panel e — peak vs COM convergent-validity scatter for the corrected
    C:pre→post frozen-replay contrast. Both metrics are computed from the
    same underlying rate maps by paired_distance_stats(); this shows the
    corrected finding is not an artifact of the COM metric specifically."""
    vals_com = np.array([r["com_mean_delta_AC"] for r in rows])
    vals_peak = np.array([r["peak_mean_delta_AC"] for r in rows])
    seeds = [r["seed"] for r in rows]

    from scipy.stats import spearmanr
    r_s, p_s = spearmanr(vals_com, vals_peak)

    fig, ax = plt.subplots(figsize=size_in or (2.4, 2.4))
    lim = max(abs(np.concatenate([vals_com, vals_peak]))) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.6", lw=0.8, zorder=1, label="y = x")
    ax.scatter(vals_com, vals_peak, s=22, color=C["muted"], alpha=0.85, zorder=3,
               linewidths=0.5, edgecolors="0.3")
    for v_com, v_peak, s in zip(vals_com, vals_peak, seeds):
        ax.text(v_com, v_peak, str(s), fontsize=FS_SMALL, ha="left", va="bottom", color="0.35")
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.axvline(0, color="0.4", lw=0.7, ls="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\Delta$ COM (C:pre→post)", fontsize=FS_TICK)
    ax.set_ylabel(r"$\Delta$ peak (C:pre→post)", fontsize=FS_TICK)
    ax.set_title(f"Peak vs COM convergent validity\nr={r_s:+.2f}, p={p_s:.3f}", fontsize=FS_SMALL, pad=2)
    ax.legend(fontsize=FS_SMALL, loc="best")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


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
    """Compose onto `LAYOUT`, with the panel letters the caption promises."""
    return figlayout.compose(LAYOUT, panel_pdfs, out_stem, dpi=dpi)



def _discover_seeds(multi_seed_dir: Path) -> list[int]:
    """Every seed directory on disk that has the data this figure needs.

    `load_seed` already tolerates an incomplete run by returning None, so this
    only has to enumerate; the existence check keeps the "skip" lines in the
    log down to runs that really are broken rather than merely absent.
    """
    multi_seed_dir = Path(multi_seed_dir)
    if not multi_seed_dir.exists():
        return []
    seeds = []
    for p in sorted(multi_seed_dir.glob("nohead_seed*")):
        try:
            seed = int(p.name.replace("nohead_seed", ""))
        except ValueError:
            continue
        if (p / "data").is_dir():
            seeds.append(seed)
    return sorted(seeds)


def parse_args():
    """Command-line arguments: multi-seed directory, seed list, output directory."""
    p = argparse.ArgumentParser()
    p.add_argument("--multi_seed_dir", default=str(REPO / "results" / "multi_seed"))
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seed indices (default: discover all nohead_seed* dirs)")
    p.add_argument("--skaggs_threshold", type=float, default=0.05)
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c,d,e")
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
    panels_dir = ensure_dir(Path(args.output_dir) / "panels")
    ensure_dir(Path(args.output_dir))

    cache_path = Path(args.output_dir) / "figure_data.npz"
    if args.plot_only:
        rows = require_keys(load_figure_data(cache_path), "rows", path=cache_path)["rows"]
        print(f"Plot-only: {len(rows)} seeds")
    else:
        seeds = args.seeds if args.seeds is not None else _discover_seeds(multi_seed_dir)
        print(f"Loading {len(seeds)} seeds from {multi_seed_dir} ...")
        rows = []
        for seed in seeds:
            r = load_seed(seed, multi_seed_dir, skaggs_threshold=args.skaggs_threshold)
            if r is None:
                print(f"  seed {seed}: skip (missing/failed)")
                continue
            print(f"  seed {seed}: n(AB)={r['n_cells']} n(C:pre→post)={r['n_cells_ac']}  "
                  f"Δd(A→B)={r['com_mean_delta_AB']:+.3f} ({100*r['com_frac_toward_AB']:.0f}% toward)  "
                  f"Δd(C:pre→post)={r['com_mean_delta_AC']:+.3f} ({100*r['com_frac_toward_AC']:.0f}% toward)")
            rows.append(r)
        update_figure_data(cache_path, {"rows": rows})

    if not rows:
        print("ERROR: no seeds loaded.")
        sys.exit(1)

    panel_pdfs: dict[str, Path] = {}

    if "a" in args.panels:
        print("Supp panel a (Δd strip charts A→B vs A→C)...")
        _, pdf = fig_supp_delta_strips(rows, panels_dir / "panel_a", LAYOUT.slot_size_in("a"))
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Supp panel b (fraction toward reward strips)...")
        _, pdf = fig_supp_frac_strips(rows, panels_dir / "panel_b", LAYOUT.slot_size_in("b"))
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Supp panel c (pooled violin A→B vs A→C)...")
        _, pdf = fig_supp_pooled_violin(rows, panels_dir / "panel_c", LAYOUT.slot_size_in("c"))
        panel_pdfs["c"] = pdf

    if "d" in args.panels:
        print("Supp panel d (per-seed scatter A→B vs C:pre→post)...")
        _, pdf = fig_supp_scatter_ab_vs_ac(rows, panels_dir / "panel_d", LAYOUT.slot_size_in("d"))
        panel_pdfs["d"] = pdf

    if "e" in args.panels:
        print("Supp panel e (peak vs COM convergent validity, C:pre→post)...")
        _, pdf = fig_supp_peak_vs_com(rows, panels_dir / "panel_e", LAYOUT.slot_size_in("e"))
        panel_pdfs["e"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("Assembling supplementary figure PDF...")
        assemble_supp(panel_pdfs, Path(args.output_dir) / "figure2_supp_raw")

    print(f"\nDone. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
