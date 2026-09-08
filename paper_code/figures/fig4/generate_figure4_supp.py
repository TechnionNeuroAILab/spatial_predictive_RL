#!/usr/bin/env python3
"""
Supplementary Figure 4 — aggregate statistics across all seeds (for Figure 4).

Panels:
  a  Per-seed fraction toward A (pre→postA) — should all be > 50%
     (gate-clean seeds filled, collapsed seeds open markers)
  b  Per-seed fraction toward B (postA→postB) — should all be > 50%
     (gate-clean seeds filled, collapsed seeds open markers)
  c  Pooled violin: Δd toward A and Δd toward B across all seeds
  d  Scatter: fraction-toward-A vs fraction-toward-B per seed (tracking quality)
     (gate-clean seeds filled, collapsed seeds open markers)
  e  Sorted per-seed entropy strip with the MIN_ENTROPY cutoff — shows
     exactly which seeds are gate-clean vs collapsed

Usage:
    python "code_analysis/figure 4/generate_figure4_supp.py" --no_show
    python "code_analysis/figure 4/generate_figure4_supp.py" \\
        --multi_seed_dir Repro/multi_seed_reloc \\
        --seeds 0 1 2 ... 44 --no_show
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
from pypdf import PdfReader, PdfWriter, Transformation
from scipy.stats import wilcoxon as _wilcoxon

REPO = Path(__file__).resolve().parents[3]
FIG4_DIR = REPO / "paper_code" / "figures" / "fig4"

sys.path.insert(0, str(FIG4_DIR))
from analysis import load_reloc_states, paired_tracking_stats, seed_entropy_status  # noqa: E402
from utils import (  # noqa: E402
    apply_nature_style, ensure_dir, save_figure,
    PAGE_W, MM_TO_PT, COLOR_A, COLOR_B, SKAGGS_THRESHOLD, MIN_ENTROPY,
    PERIOD_PRE, PERIOD_POST_A, PERIOD_POST_B,
)

_NEW_CODE = FIG4_DIR.parents[1] / "lib"
sys.path.insert(0, str(_NEW_CODE))
from analysis_helpers import peak_entropy  # noqa: E402

apply_nature_style()

SUPP_FIGURES_DIR = FIG4_DIR / "figures_supp"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_seed(seed: int, multi_seed_dir: Path, skaggs_threshold: float = SKAGGS_THRESHOLD):
    run_dir = multi_seed_dir / f"reloc_seed{seed}"
    data_dir = run_dir / "data"
    if not data_dir.exists():
        return None
    try:
        states = load_reloc_states(data_dir, skaggs_threshold=skaggs_threshold)
        ts = paired_tracking_stats(states, skaggs_threshold=skaggs_threshold)
        periods = (PERIOD_PRE, PERIOD_POST_A, PERIOD_POST_B)
        entropies = {p: peak_entropy(data_dir, p) for p in periods}
        is_clean = seed_entropy_status(data_dir, periods)
        return {
            "seed": seed,
            "frac_toward_A": ts["shift_A"]["frac_toward"],
            "frac_toward_B": ts["shift_B"]["frac_toward"],
            "mean_delta_A": ts["shift_A"]["mean_delta"],
            "mean_delta_B": ts["shift_B"]["mean_delta"],
            "delta_A": ts["shift_A"]["delta"],
            "delta_B": ts["shift_B"]["delta"],
            "wilcoxon_p_A": ts["shift_A"]["wilcoxon_p"],
            "wilcoxon_p_B": ts["shift_B"]["wilcoxon_p"],
            "is_clean": is_clean,
            "min_entropy": min(v for v in entropies.values() if v is not None) if any(v is not None for v in entropies.values()) else float("nan"),
        }
    except Exception as exc:
        print(f"  [seed {seed}] error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def _fmt_p(p):
    if not np.isfinite(p):
        return ""
    return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"


def _wil_p(vals, alternative="greater"):
    try:
        _, p = _wilcoxon(np.array(vals) - 0.5, alternative=alternative)
        return float(p)
    except Exception:
        return float("nan")


def fig_supp_frac_toward_a(rows, stem):
    """Panel a — per-seed fraction toward A (pre→postA). Gate-clean seeds
    are filled; entropy-collapsed seeds are shown as open markers."""
    vals = np.array([r["frac_toward_A"] * 100 for r in rows])
    clean = np.array([bool(r["is_clean"]) for r in rows])
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(2.5, 2.2))
    jitter = rng.uniform(-0.08, 0.08, len(vals))
    ax.scatter(jitter[clean], vals[clean], s=14, color=COLOR_A, alpha=0.8, zorder=3, linewidths=0, label="gate-clean")
    if (~clean).any():
        ax.scatter(jitter[~clean], vals[~clean], s=18, facecolors="none", edgecolors=COLOR_A,
                   alpha=0.9, zorder=3, linewidths=1.0, label="collapsed")
    mean, sd = np.mean(vals), np.std(vals, ddof=1)
    p = _wil_p([r["frac_toward_A"] for r in rows])
    ax.errorbar([0], [mean], yerr=[sd], fmt="D", color="black",
                markersize=5, capsize=4, linewidth=1.2, zorder=5,
                label=f"{mean:.1f}±{sd:.1f}%")
    ax.axhline(50, color="0.4", lw=0.8, ls="--", label="chance")
    ax.set_xticks([0])
    ax.set_xticklabels(["pre→postA\n(toward A)"])
    ax.set_ylabel("% cells toward A")
    ax.set_ylim(0, 100)
    ax.set_title(f"{_fmt_p(p)}", fontsize=6, pad=2)
    ax.legend(fontsize=4.5)
    fig.suptitle(f"Fraction toward A per seed (n={len(rows)}, {clean.sum()} clean)", fontsize=6.5, y=1.02)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_supp_frac_toward_b(rows, stem):
    """Panel b — per-seed fraction toward B (postA→postB). Gate-clean seeds
    are filled; entropy-collapsed seeds are shown as open markers."""
    vals = np.array([r["frac_toward_B"] * 100 for r in rows])
    clean = np.array([bool(r["is_clean"]) for r in rows])
    rng = np.random.default_rng(1)

    fig, ax = plt.subplots(figsize=(2.5, 2.2))
    jitter = rng.uniform(-0.08, 0.08, len(vals))
    ax.scatter(jitter[clean], vals[clean], s=14, color=COLOR_B, alpha=0.8, zorder=3, linewidths=0, label="gate-clean")
    if (~clean).any():
        ax.scatter(jitter[~clean], vals[~clean], s=18, facecolors="none", edgecolors=COLOR_B,
                   alpha=0.9, zorder=3, linewidths=1.0, label="collapsed")
    mean, sd = np.mean(vals), np.std(vals, ddof=1)
    p = _wil_p([r["frac_toward_B"] for r in rows])
    ax.errorbar([0], [mean], yerr=[sd], fmt="D", color="black",
                markersize=5, capsize=4, linewidth=1.2, zorder=5,
                label=f"{mean:.1f}±{sd:.1f}%")
    ax.axhline(50, color="0.4", lw=0.8, ls="--", label="chance")
    ax.set_xticks([0])
    ax.set_xticklabels(["postA→postB\n(toward B)"])
    ax.set_ylabel("% cells toward B")
    ax.set_ylim(0, 100)
    ax.set_title(f"{_fmt_p(p)}", fontsize=6, pad=2)
    ax.legend(fontsize=4.5)
    fig.suptitle(f"Fraction toward B per seed (n={len(rows)}, {clean.sum()} clean)", fontsize=6.5, y=1.02)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_supp_pooled_violin(rows, stem):
    """Panel c — pooled violin of Δd toward A and Δd toward B with per-seed means."""
    all_dA = np.concatenate([r["delta_A"] for r in rows])
    all_dB = np.concatenate([r["delta_B"] for r in rows])
    per_dA = np.array([r["mean_delta_A"] for r in rows])
    per_dB = np.array([r["mean_delta_B"] for r in rows])

    p_A_per = _wil_p(per_dA, alternative="less")  # toward = negative delta
    p_B_per = _wil_p(per_dB, alternative="less")

    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    parts = ax.violinplot([all_dA, all_dB], positions=[1, 2],
                          showmeans=False, showmedians=True, showextrema=True)
    for body, c in zip(parts["bodies"], [COLOR_A, COLOR_B]):
        body.set_facecolor(c)
        body.set_alpha(0.55)
        body.set_edgecolor("0.2")
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("0.25")
            parts[key].set_linewidth(0.9)

    rng = np.random.default_rng(2)
    jitter = rng.uniform(-0.07, 0.07, len(rows))
    for dA, dB, j in zip(per_dA, per_dB, jitter):
        ax.plot([1 + j, 2 + j], [dA, dB], color="0.5", lw=0.5, alpha=0.5)
    ax.scatter(np.ones(len(rows)) + jitter, per_dA, s=14, color="#1a6b63",
               zorder=5, linewidths=0, alpha=0.9, label="seed mean")
    ax.scatter(np.ones(len(rows)) * 2 + jitter, per_dB, s=14, color="#9e3b1f",
               zorder=5, linewidths=0, alpha=0.9)

    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"→A\n{_fmt_p(p_A_per)}", f"→B\n{_fmt_p(p_B_per)}"])
    ax.set_ylabel(r"$\Delta$ field dist to reward")
    ax.set_title(f"Pooled ({len(rows)} seeds)", fontsize=5.5, pad=2)
    ax.legend(fontsize=5)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_supp_tracking_scatter(rows, stem):
    """Panel d — scatter: fraction-toward-A vs fraction-toward-B per seed.
    Gate-clean seeds are filled; entropy-collapsed seeds are open markers
    (their high apparent tracking score is often a field-collapse artifact)."""
    fa = np.array([r["frac_toward_A"] * 100 for r in rows])
    fb = np.array([r["frac_toward_B"] * 100 for r in rows])
    seeds = [r["seed"] for r in rows]
    clean = np.array([bool(r["is_clean"]) for r in rows])

    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.scatter(fa[clean], fb[clean], s=22, color="#555555", alpha=0.8, zorder=3,
               linewidths=0.5, edgecolors="0.3", label="gate-clean")
    if (~clean).any():
        ax.scatter(fa[~clean], fb[~clean], s=26, facecolors="none", edgecolors="#B23A48",
                   alpha=0.9, zorder=4, linewidths=1.1, label="collapsed")
    for x, y, s in zip(fa, fb, seeds):
        ax.text(x, y, str(s), fontsize=4.5, ha="left", va="bottom", color="0.35")
    ax.axhline(50, color="0.4", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(50, color="0.4", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("% cells toward A (pre→postA)")
    ax.set_ylabel("% cells toward B (postA→postB)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_title(f"Tracking quality per seed ({clean.sum()}/{len(rows)} clean)", fontsize=5.5, pad=2)
    ax.legend(fontsize=4.5, loc="lower right")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_supp_entropy_strip(rows, stem):
    """Panel e — sorted per-seed entropy strip with the MIN_ENTROPY cutoff
    line. Shows explicitly which seeds are gate-clean vs collapsed."""
    seeds = np.array([r["seed"] for r in rows])
    ents = np.array([r["min_entropy"] for r in rows])
    order = np.argsort(ents)
    seeds_s, ents_s = seeds[order], ents[order]
    clean_s = ents_s >= MIN_ENTROPY

    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    x = np.arange(len(seeds_s))
    colors = ["#2a9d8f" if c else "#B23A48" for c in clean_s]
    ax.bar(x, ents_s, color=colors, alpha=0.85, width=0.7)
    ax.axhline(MIN_ENTROPY, color="0.25", lw=1.0, ls="--", label=f"MIN_ENTROPY={MIN_ENTROPY}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds_s], fontsize=5)
    ax.set_xlabel("Seed (sorted by min entropy)")
    ax.set_ylabel("Min peak-entropy across periods\n(bits)")
    n_clean = int(clean_s.sum())
    ax.set_title(f"Entropy-gate status ({n_clean}/{len(rows)} clean)", fontsize=6, pad=2)
    ax.legend(fontsize=5)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------

def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300):
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), str(out_prefix)],
            check=True,
        )
        copy2(out_prefix.with_suffix(".png"), png_path)


def assemble_supp(panel_pdfs: dict[str, Path], out_stem: Path, dpi: int = 300):
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

    rows_def = [("a", "b"), ("c", "d"), ("e",)]
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
    p = argparse.ArgumentParser()
    p.add_argument("--multi_seed_dir", default=str(REPO / "Repro" / "multi_seed_reloc"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(45)))
    p.add_argument("--skaggs_threshold", type=float, default=SKAGGS_THRESHOLD)
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c,d,e")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    args = p.parse_args()
    args.panels = {x.strip() for x in args.panels.split(",") if x.strip()}
    return args


def main():
    args = parse_args()
    multi_seed_dir = Path(args.multi_seed_dir)
    panels_dir = ensure_dir(Path(args.output_dir) / "panels")
    ensure_dir(Path(args.output_dir))

    print(f"Loading {len(args.seeds)} relocation seeds from {multi_seed_dir} ...")
    rows = []
    for seed in args.seeds:
        r = load_seed(seed, multi_seed_dir, skaggs_threshold=args.skaggs_threshold)
        if r is None:
            print(f"  seed {seed}: skip (missing/failed)")
            continue
        gate = "clean" if r["is_clean"] else "COLLAPSED"
        print(f"  seed {seed}: →A={100*r['frac_toward_A']:.0f}%  →B={100*r['frac_toward_B']:.0f}%  "
              f"pA={r['wilcoxon_p_A']:.1e}  pB={r['wilcoxon_p_B']:.1e}  "
              f"min_entropy={r['min_entropy']:.2f} [{gate}]")
        rows.append(r)

    if not rows:
        print("ERROR: no seeds loaded.")
        sys.exit(1)

    panel_pdfs: dict[str, Path] = {}

    if "a" in args.panels:
        print("\nSupp panel a (fraction toward A per seed)...")
        _, pdf = fig_supp_frac_toward_a(rows, panels_dir / "panel_a")
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Supp panel b (fraction toward B per seed)...")
        _, pdf = fig_supp_frac_toward_b(rows, panels_dir / "panel_b")
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Supp panel c (pooled violin)...")
        _, pdf = fig_supp_pooled_violin(rows, panels_dir / "panel_c")
        panel_pdfs["c"] = pdf

    if "d" in args.panels:
        print("Supp panel d (tracking scatter)...")
        _, pdf = fig_supp_tracking_scatter(rows, panels_dir / "panel_d")
        panel_pdfs["d"] = pdf

    if "e" in args.panels:
        print("Supp panel e (entropy-gate status strip)...")
        _, pdf = fig_supp_entropy_strip(rows, panels_dir / "panel_e")
        panel_pdfs["e"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("\nAssembling supplementary figure PDF...")
        assemble_supp(panel_pdfs, Path(args.output_dir) / "figure4_supp_raw")

    print(f"\nDone. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
