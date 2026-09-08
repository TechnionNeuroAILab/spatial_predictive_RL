#!/usr/bin/env python3
"""
Figure 5 — Dissociating "policy-type" (occupancy-driven) remapping from
"real" (reward-prediction-error-driven) representational remapping.

Panels
  a  Schematic: two hypotheses and the frozen-replay diagnostic
  b  Double dissociation: mean Δd under frozen replay, head OFF vs head ON
  c  Behavior/remapping decoupling: cycle-2 test success vs % cells toward B
  e  Predictions table for real hippocampal experiments

Panels d (proactive interference) and f (SR comparison) were REMOVED; see the
comments at their former call sites for why. Panel keys are left unrenumbered
so that cached panel_*.pdf files and downstream references stay unambiguous.

Usage:
    python "code_analysis/figure 5/generate_figure5.py" --no_show
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import subprocess
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
FIG5_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG5_DIR))

from analysis import (  # noqa: E402
    load_double_dissociation, double_dissociation_stats,
    load_relocation_summary,
)
from utils import (  # noqa: E402
    NOHEAD_DIR, HEAD_DIR, RELOC_DIR,
    FIGURES_DIR, PANELS_DIR, MM_TO_PT, PAGE_W,
    COLOR_NOHEAD, COLOR_HEAD, COLOR_A, COLOR_B,
    apply_nature_style, ensure_dir, save_figure,
)

sys.path.insert(0, str(REPO / "New_Code"))
# NOTE: the former panel f (SR comparison, via New_Code/sr_vs_model_panel.py)
# has been REMOVED. Its metric projected each cell's shift onto a toward-reward
# direction taken from that cell's PRE position, which scores generic drift
# toward the arena centre as "toward reward"; a shuffled-cell null returned MORE
# than the observed value, and the panel consequently reported head-ON and
# no-head as indistinguishable (p = 0.83). The corrected SR comparison now lives
# in code_analysis/figure_sr/ as its own supplementary figure. Panel b of THIS
# figure uses the correct change-in-distance metric and is unaffected.
from analysis_helpers import seed_is_clean  # noqa: E402

apply_nature_style()

_RELOC_PERIODS = ["replay_reloc_pred", "replay_reloc_postA", "replay_reloc_postB"]


def _discover_reloc_clean_seeds(reloc_dir: Path) -> list[int]:
    """Scan RELOC_DIR for all complete reloc_seed{N} runs and return those
    passing the entropy-collapse gate (same diagnostic used everywhere else
    in this figure). Computed dynamically so this stays correct as more
    relocation seeds are added, instead of a hardcoded list going stale."""
    reloc_dir = Path(reloc_dir)
    if not reloc_dir.exists():
        return []
    seeds = []
    for p in sorted(reloc_dir.glob("reloc_seed*")):
        try:
            seed = int(p.name.replace("reloc_seed", ""))
        except ValueError:
            continue
        data_dir = p / "data"
        if (data_dir / "df_activations_replay_reloc_postB.pkl").exists():
            seeds.append(seed)
    return [s for s in seeds if seed_is_clean(reloc_dir / f"reloc_seed{s}" / "data", _RELOC_PERIODS)]


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), str(out_prefix)],
            check=True,
        )
        copy2(out_prefix.with_suffix(".png"), png_path)


# ---------------------------------------------------------------------------
# Panel a — schematic: two hypotheses + frozen-replay diagnostic
# ---------------------------------------------------------------------------

def fig_schematic(stem) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="0.25", fs=6.3, weight="normal"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                            fc=fc, ec=ec, lw=0.8, zorder=3)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight, zorder=4)
        return b

    def arrow(x0, y0, x1, y1, color="0.25", style="-|>", lw=1.0):
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8,
                             color=color, lw=lw, zorder=2)
        ax.add_patch(a)

    # Top: shared origin
    box(3.5, 3.55, 3.0, 0.55, "Reward learning occurs\n(behavior + representation both change)",
        fc="#eeeeee", fs=6.3, weight="bold")

    # Left branch — Policy-type / occupancy-driven
    box(0.3, 2.35, 3.4, 0.65,
        "H1: Policy-type remapping\nBehavior changes → animal revisits\nreward-adjacent bins more often",
        fc="#dbe7f5", fs=6.0)
    arrow(3.5, 3.55, 2.0, 3.0)

    box(0.3, 1.15, 3.4, 0.65,
        "Naive rate maps look reorganized,\nbut single-cell tuning is unchanged",
        fc="#dbe7f5", fs=6.0)
    arrow(2.0, 2.35, 2.0, 1.8)

    # Right branch — Real / RPE-driven
    box(6.3, 2.35, 3.4, 0.65,
        "H2: Real remapping\nReward-prediction-error gradient\nreshapes world-model weights",
        fc="#f5e2d6", fs=6.0)
    arrow(6.5, 3.55, 8.0, 3.0)

    box(6.3, 1.15, 3.4, 0.65,
        "Single-cell tuning itself shifts\ntoward the reward location",
        fc="#f5e2d6", fs=6.0)
    arrow(8.0, 2.35, 8.0, 1.8)

    arrow(2.0, 1.15, 4.4, 0.55, color="0.35")
    arrow(8.0, 1.15, 5.6, 0.55, color="0.35")

    # Diagnostic diamond
    diamond = mpatches.RegularPolygon((5.0, 0.35), numVertices=4, radius=0.85,
                                       orientation=np.pi / 4, fc="#fff6cc", ec="0.25", lw=0.9, zorder=3)
    ax.add_patch(diamond)
    ax.text(5.0, 0.35, "Frozen-replay\ndiagnostic\n(fixed trajectory,\npre vs post)",
            ha="center", va="center", fontsize=5.6, zorder=4)

    ax.text(1.6, 0.5, "predicts:\nshift VANISHES", ha="center", va="center",
            fontsize=6.0, color="#2255aa", style="italic")
    ax.text(8.4, 0.5, "predicts:\nshift PERSISTS", ha="center", va="center",
            fontsize=6.0, color="#a0522d", style="italic")

    fig.tight_layout(pad=0.3)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel b — double dissociation (head OFF vs head ON, frozen replay)
# ---------------------------------------------------------------------------

def fig_double_dissociation(rows: list[dict], stem) -> Path:
    stats = double_dissociation_stats(rows)
    # analysis.py returns positive = toward reward, but Figures 2-4 and the
    # manuscript all use negative = toward reward. Negate here, at the plotting
    # layer only, so a reader moving between figures never sees the sign flip.
    # The statistics are sign-invariant (paired Wilcoxon), so only the axis and
    # the plotted points change.
    d_nh = -np.array([r["delta_nohead"] for r in rows])
    d_hd = -np.array([r["delta_head"] for r in rows])
    n = len(rows)

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.06, 0.06, n)

    ax.scatter(np.zeros(n) + jitter, d_nh, s=20, color=COLOR_NOHEAD, zorder=4, linewidths=0)
    ax.scatter(np.ones(n) + jitter, d_hd, s=20, color=COLOR_HEAD, zorder=4, linewidths=0)
    for j, a, b in zip(jitter, d_nh, d_hd):
        ax.plot([0 + j, 1 + j], [a, b], color="0.6", lw=0.5, alpha=0.6, zorder=2)

    ax.errorbar([0], [-stats["mean_nohead"]], yerr=[stats["sd_nohead"]], fmt="D",
                color="black", markersize=5, capsize=3, lw=1.1, zorder=5)
    ax.errorbar([1], [-stats["mean_head"]], yerr=[stats["sd_head"]], fmt="D",
                color="black", markersize=5, capsize=3, lw=1.1, zorder=5)

    ax.axhline(0, color="0.4", lw=0.7, ls="--", zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No head\n(Fig 2)", "Head ON\n(Fig 3)"])
    ax.set_ylabel(r"Mean $\Delta d$ frozen replay (pre→post weights)" + "\n(negative = toward reward; SR predicts 0)")
    p = stats["wilcoxon_p_paired"]
    p_str = f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"
    # also show individual vs-zero tests
    from scipy.stats import wilcoxon as _wc
    _, p_nh = _wc(d_nh)
    _, p_hd = _wc(d_hd)
    p_nh_str = f"{p_nh:.3f}" if p_nh >= 0.001 else f"{p_nh:.1e}"
    p_hd_str = f"{p_hd:.3f}" if p_hd >= 0.001 else f"{p_hd:.1e}"
    ax.set_title(
        f"Frozen-replay: weight-driven shift\n"
        f"nh p={p_nh_str}  hd p={p_hd_str}  paired {p_str}",
        fontsize=6.0, pad=3
    )
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel c — behavior / remapping decoupling
# ---------------------------------------------------------------------------

def fig_behavior_decoupling(rows: list[dict], stem) -> Path:
    xs = np.array([r["succ_c2"] for r in rows])
    ys = np.array([r["frac_B"] for r in rows]) * 100
    seeds = [r["seed"] for r in rows]
    r_s, p_s = spearmanr(xs, ys)

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    m, b = np.polyfit(xs, ys, 1)
    xf = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 100)
    ax.plot(xf, m * xf + b, color="0.5", lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.scatter(xs, ys, s=32, color=COLOR_B, zorder=4, edgecolors="0.3", linewidths=0.4)
    for x, y, s in zip(xs, ys, seeds):
        ax.text(x + 0.012, y + 1.0, str(s), fontsize=5.6, color="0.35")
    ax.axhline(50, color="0.5", lw=0.6, ls=":", alpha=0.7)
    ax.set_xlabel("Cycle-2 test success\n(reward B, behavior)")
    ax.set_ylabel("% cells toward B\n(postA→postB)")
    ax.set_ylim(20, 75)
    ax.set_title(f"Behavior does not predict\nrepresentational shift\nr={r_s:+.2f}, p={p_s:.3f}",
                 fontsize=6.3, pad=3)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel d — proactive interference
# ---------------------------------------------------------------------------

def fig_proactive_interference(rows: list[dict], stem) -> Path:
    xs = np.array([r["delta_A"] for r in rows])
    ys = np.array([r["delta_B"] for r in rows])
    seeds = [r["seed"] for r in rows]
    r_s, p_s = spearmanr(xs, ys)

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    m, b = np.polyfit(xs, ys, 1)
    xf = np.linspace(xs.min() - 0.005, xs.max() + 0.005, 100)
    ax.plot(xf, m * xf + b, color="0.5", lw=1.0, ls="--", alpha=0.7, zorder=1)
    colors = [COLOR_A if x < np.median(xs) else COLOR_B for x in xs]
    ax.scatter(xs, ys, s=32, color=colors, zorder=4, edgecolors="0.3", linewidths=0.4)
    for x, y, s in zip(xs, ys, seeds):
        ax.text(x + 0.001, y + 0.001, str(s), fontsize=5.6, color="0.35")
    ax.axhline(0, color="0.5", lw=0.6, ls=":", alpha=0.7)
    ax.axvline(0, color="0.5", lw=0.6, ls=":", alpha=0.7)
    ax.set_xlabel(r"Mean $\Delta d$ toward A (cycle 1)")
    ax.set_ylabel(r"Mean $\Delta d$ toward B (cycle 2)")
    ax.set_title(f"Strong A-encoding predicts\nresistance to B-remapping\nr={r_s:+.2f}, p={p_s:.3f}",
                 fontsize=6.3, pad=3)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel e — predictions table for real hippocampal experiments
# ---------------------------------------------------------------------------

PREDICTIONS = [
    ("1. Occupancy-matched\ntrajectories",
     "VR/treadmill with repeated\nfixed trajectory pre/post learning",
     "reward-ward shift\nvanishes",
     "reward-ward shift\npersists"),
    ("2. Dopamine / RPE\nsilencing",
     "Optogenetic VTA inhibition\nduring reward learning",
     "no effect on\n(absent) shift",
     "shift is\nabolished"),
    ("3. Behavior vs\nremapping correlation",
     "Correlate per-animal learning\nspeed with remapping magnitude",
     "strong positive\ncorrelation",
     "weak / no\ncorrelation"),
    # Row 4 (proactive interference) REMOVED together with panel d: the model
    # data show r=+0.26, p=0.152, i.e. the OPPOSITE sign to the "negative
    # correlation" this row predicted under H2. Publishing a prediction the
    # model itself contradicts would be indefensible.
    ("4. Goal-vector vs\nfield density",
     "Move a learned goal; track bearing\ntuning and field density together",
     "both follow the\ngoal equally",
     "vector code retargets\nfast, density lags"),
]


def fig_predictions_table(stem) -> Path:
    n_rows = len(PREDICTIONS)
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_rows + 1)

    col_x = [0.1, 2.6, 5.6, 7.6, 9.6]
    headers = ["Prediction", "Manipulation / analysis", "If H1 (policy-type)", "If H2 (real remapping)"]
    header_colors = ["#eeeeee", "#eeeeee", "#dbe7f5", "#f5e2d6"]

    y_top = n_rows + 0.55
    for i, (h, cx0, cx1, col) in enumerate(zip(headers, col_x[:-1], col_x[1:], header_colors)):
        ax.add_patch(plt.Rectangle((cx0, y_top - 0.5), cx1 - cx0 - 0.05, 0.5,
                                    fc=col, ec="0.4", lw=0.6, zorder=2))
        ax.text((cx0 + cx1) / 2 - 0.025, y_top - 0.25, h, ha="center", va="center",
                fontsize=6.2, weight="bold", zorder=3)

    for r, (pred, manip, h1, h2) in enumerate(PREDICTIONS):
        y = n_rows - r - 0.5
        row_cells = [pred, manip, h1, h2]
        row_colors = ["white", "white", "#eef4fb", "#fbf1ea"]
        for (cx0, cx1), text, fc in zip(zip(col_x[:-1], col_x[1:]), row_cells, row_colors):
            ax.add_patch(plt.Rectangle((cx0, y - 0.42), cx1 - cx0 - 0.05, 0.84,
                                        fc=fc, ec="0.75", lw=0.5, zorder=2))
            ax.text((cx0 + cx1) / 2 - 0.025, y, text, ha="center", va="center",
                    fontsize=5.6, zorder=3)

    fig.tight_layout(pad=0.3)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------

def assemble_figure(panel_pdfs: dict[str, Path], stem, save_dpi: int = 300):
    stem = Path(stem).with_suffix("")
    ensure_dir(stem.parent)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")

    MARGIN = 5 * MM_TO_PT
    GAP = 3 * MM_TO_PT
    ROW_GAP = 4 * MM_TO_PT

    def get_size(pdf):
        box = PdfReader(str(pdf)).pages[0].mediabox
        return float(box.width), float(box.height)

    sizes = {k: get_size(v) for k, v in panel_pdfs.items() if v is not None}
    available = {k for k in panel_pdfs if k in sizes}

    rows_def = [("a",), ("b", "c"), ("e",)]
    row_keys = [[k for k in row if k in available] for row in rows_def]
    row_keys = [r for r in row_keys if r]

    def row_height(keys):
        ncols = len(keys)
        cw = (PAGE_W - 2 * MARGIN - (ncols - 1) * GAP) / ncols
        return max((sizes[k][1] * (cw / max(sizes[k][0], 1e-6)) for k in keys), default=0.0)

    rhs = [row_height(r) for r in row_keys]
    page_h = 2 * MARGIN + sum(rhs) + (len(rhs) - 1) * ROW_GAP

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
        y += rh + ROW_GAP

    with open(pdf_path, "wb") as f:
        writer.write(f)
    _pdf_to_png(pdf_path, png_path, dpi=save_dpi)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    return png_path, pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Panel f — SR vs RNN model field-shift comparison
# ---------------------------------------------------------------------------

# fig_sr_comparison() removed — see the note at the top of this file.
# The corrected version is code_analysis/figure_sr/generate_figure_sr_supp.py.


def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 5 (dissociating the two hypotheses).")
    p.add_argument("--output_dir", default=str(FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c,e")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    args = p.parse_args()
    args.panels = {x.strip().lower() for x in args.panels.split(",") if x.strip()}
    return args


def main():
    args = parse_args()
    panels_dir = ensure_dir(Path(args.output_dir) / "panels")
    figures_dir = ensure_dir(Path(args.output_dir))

    panel_pdfs: dict[str, Path] = {}

    if "a" in args.panels:
        print("Panel a (schematic: two hypotheses)...")
        _, pdf = fig_schematic(panels_dir / "panel_a")
        panel_pdfs["a"] = pdf

    if {"b"} & args.panels:
        print("Loading double-dissociation data (head OFF vs head ON, frozen replay)...")
        dd_rows = load_double_dissociation(NOHEAD_DIR, HEAD_DIR, seeds=range(50))
        stats = double_dissociation_stats(dd_rows)
        print(f"  n={stats['n_seeds']} paired seeds  "
              f"no-head={stats['mean_nohead']:+.4f}±{stats['sd_nohead']:.4f}  "
              f"head={stats['mean_head']:+.4f}±{stats['sd_head']:.4f}  "
              f"p={stats['wilcoxon_p_paired']:.3g}")
        print("Panel b (double dissociation)...")
        _, pdf = fig_double_dissociation(dd_rows, panels_dir / "panel_b")
        panel_pdfs["b"] = pdf

    if {"c", "d"} & args.panels:
        reloc_clean_seeds = _discover_reloc_clean_seeds(RELOC_DIR)
        print(f"Loading relocation summary (behavior + tracking, {len(reloc_clean_seeds)} clean seeds)...")
        reloc_rows = load_relocation_summary(RELOC_DIR, reloc_clean_seeds)
        for r in reloc_rows:
            print(f"  seed {r['seed']}: succ_c2={r['succ_c2']:.2f}  "
                  f"→A={100*r['frac_A']:.0f}%  →B={100*r['frac_B']:.0f}%")

        if "c" in args.panels:
            print("Panel c (behavior/remapping decoupling)...")
            _, pdf = fig_behavior_decoupling(reloc_rows, panels_dir / "panel_c")
            panel_pdfs["c"] = pdf

        # Panel d (proactive interference) REMOVED. It reported
        # Spearman r=+0.26, p=0.152 under a title asserting "strong A-encoding
        # predicts resistance to B-remapping": not significant, and the sign is
        # opposite to the claim (with negative = toward reward, a positive
        # correlation means seeds that remap strongly to A also remap strongly
        # to B). It also contradicted row 4 of the paper's own predictions
        # table, which has been removed with it.

    if "e" in args.panels:
        print("Panel e (predictions table)...")
        _, pdf = fig_predictions_table(panels_dir / "panel_e")
        panel_pdfs["e"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("Assembling vector composite PDF...")
        assemble_figure(panel_pdfs, figures_dir / "figure5_raw")

    print("\nDone.")
    print(f"  Figures: {figures_dir}")


if __name__ == "__main__":
    main()
