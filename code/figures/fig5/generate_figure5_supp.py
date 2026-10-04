#!/usr/bin/env python3
"""
Supplementary Figure 5 — robustness checks for the H1-vs-H2 dissociation
(entropy/Skaggs threshold sensitivity, SR-direction test, and relocation
correlation-vs-sample-size stability).

Panels:
  a  Entropy-gate (MIN_ENTROPY) sensitivity sweep: double-dissociation stat
     (panel b of the main figure) and relocation clean-seed count vs threshold
  b  Skaggs-threshold sensitivity for the same double-dissociation stat
  c  SR-predicted shift direction test (A→B contrast): pooled angle histogram
     (Rayleigh test) + residual-vs-distance-to-reward correlation
  d  Relocation correlation-vs-n-seeds robustness: how panels c/d of the main
     figure (Spearman rho) stabilize as the clean-seed pool grows

Usage (run from repo root):
    python code/figures/fig5/generate_figure5_supp.py --no_show
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
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
FIG5_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG5_DIR))

from analysis import (  # noqa: E402
    load_relocation_summary, seed_is_clean,
)
from utils import (  # noqa: E402
    NOHEAD_DIR, HEAD_DIR, RELOC_DIR,
    MM_TO_PT, PAGE_W,
    COLOR_NOHEAD, COLOR_HEAD,
    MIN_ENTROPY, SKAGGS_THRESHOLD,
    apply_nature_style, ensure_dir, save_figure,
)
from sr_direction_analysis import load_all_seeds_direction  # noqa: E402

_NEW_CODE = REPO / "code" / "lib"
if str(_NEW_CODE) not in sys.path:
    sys.path.insert(0, str(_NEW_CODE))
from analysis_helpers import load_period_centers, peak_entropy  # noqa: E402

apply_nature_style()

SUPP_FIGURES_DIR = FIG5_DIR / "figures_supp"

_RELOC_PERIODS = ["replay_reloc_pred", "replay_reloc_postA", "replay_reloc_postB"]

ENTROPY_SWEEP = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
SKAGGS_SWEEP = [0.05, 0.15, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _discover_reloc_seeds(reloc_dir: Path) -> list[int]:
    """All complete relocation seeds present on disk (regardless of gate)."""
    reloc_dir = Path(reloc_dir)
    if not reloc_dir.exists():
        return []
    seeds = []
    for p in sorted(reloc_dir.glob("reloc_seed*")):
        try:
            seed = int(p.name.replace("reloc_seed", ""))
        except ValueError:
            continue
        if (p / "data" / "df_activations_replay_reloc_postB.pkl").exists():
            seeds.append(seed)
    return seeds


def _fmt_p(p):
    """Format a p-value for a panel annotation; scientific notation below 0.001."""
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# Shared expensive precomputation (loaded once, reused across every threshold
# in the entropy/Skaggs sweeps below -- avoids reloading multi-hundred-MB
# rate-map pickles once per sweep point).
# ---------------------------------------------------------------------------

def _M_GRID_entropy(peak_xy, place_mask, m_grid=32):
    """Shannon entropy (bits) of peak locations restricted to place_mask."""
    if place_mask.sum() == 0:
        return None
    pk = peak_xy[place_mask]
    binned = (np.clip(pk, 0, 0.9999) * m_grid).astype(int)
    counts = np.zeros((m_grid, m_grid), dtype=int)
    for bx, by in binned:
        counts[bx, by] += 1
    nz = counts.flatten()
    nz = nz[nz > 0]
    p = nz / nz.sum()
    return float(-np.sum(p * np.log2(p)))


def _precompute_frozen_replay_raw(cond_dir, prefix, seeds):
    """Load (sk_pre, pk_pre, sk_post, pk_post) once per seed for the cycle-1
    frozen-replay periods. Reused by both the entropy sweep (panel a) and
    the Skaggs sweep (panel b) so rate maps are only loaded from disk once."""
    out = {}
    for seed in seeds:
        data_dir = Path(cond_dir) / f"{prefix}seed{seed}" / "data"
        try:
            sk_pre, _, pk_pre = load_period_centers(str(data_dir), "replay_c1_pred")
            sk_post, _, pk_post = load_period_centers(str(data_dir), "replay_c1_postrl")
        except Exception:
            continue
        out[seed] = {"sk_pre": sk_pre, "pk_pre": pk_pre, "sk_post": sk_post, "pk_post": pk_post}
    return out


def _entropy_at_threshold(raw, skaggs_thr=SKAGGS_THRESHOLD):
    """Min entropy across the two frozen-replay periods, for a fixed Skaggs
    gate used to select place cells (this is independent of MIN_ENTROPY --
    MIN_ENTROPY only thresholds the resulting value, computed once here)."""
    e_pre = _M_GRID_entropy(raw["pk_pre"], raw["sk_pre"] >= skaggs_thr)
    e_post = _M_GRID_entropy(raw["pk_post"], raw["sk_post"] >= skaggs_thr)
    if e_pre is None or e_post is None:
        return None
    return min(e_pre, e_post)


def _delta_at_skaggs_threshold(raw, reward, skaggs_thr, min_cells=10):
    """Per-cell frozen-replay Δd at a given Skaggs gate, from cached arrays
    (no disk I/O -- cheap, reused across the whole Skaggs sweep)."""
    gate = (raw["sk_pre"] >= skaggs_thr) & (raw["sk_post"] >= skaggs_thr)
    n = int(gate.sum())
    if n < min_cells:
        return None
    d_pre = np.linalg.norm(raw["pk_pre"][gate] - reward, axis=1)
    d_post = np.linalg.norm(raw["pk_post"][gate] - reward, axis=1)
    delta = d_pre - d_post   # positive = toward reward
    return {"n": n, "mean_delta": float(delta.mean())}


# ---------------------------------------------------------------------------
# Panel a — entropy-gate sensitivity sweep
# ---------------------------------------------------------------------------

def fig_supp_entropy_sensitivity(raw_nh, raw_hd, reloc_dir, stem):
    """Panel a -- how the double-dissociation statistic and the clean-seed count move with the entropy gate."""
    from scipy.stats import wilcoxon as _wilcoxon
    common_seeds = sorted(set(raw_nh) & set(raw_hd))
    REWARD_A = np.array([0.3, 0.3])

    # entropy + delta computed once per seed (independent of the swept threshold)
    ent_nh = {s: _entropy_at_threshold(raw_nh[s]) for s in common_seeds}
    ent_hd = {s: _entropy_at_threshold(raw_hd[s]) for s in common_seeds}
    delta_nh = {s: _delta_at_skaggs_threshold(raw_nh[s], REWARD_A, SKAGGS_THRESHOLD) for s in common_seeds}
    delta_hd = {s: _delta_at_skaggs_threshold(raw_hd[s], REWARD_A, SKAGGS_THRESHOLD) for s in common_seeds}

    dd_p, dd_n = [], []
    for thr in ENTROPY_SWEEP:
        pass_seeds = [
            s for s in common_seeds
            if ent_nh[s] is not None and ent_hd[s] is not None
            and ent_nh[s] >= thr and ent_hd[s] >= thr
            and delta_nh[s] is not None and delta_hd[s] is not None
        ]
        if len(pass_seeds) < 3:
            dd_p.append(np.nan); dd_n.append(len(pass_seeds))
            continue
        d_nh = np.array([delta_nh[s]["mean_delta"] for s in pass_seeds])
        d_hd = np.array([delta_hd[s]["mean_delta"] for s in pass_seeds])
        try:
            _, p = _wilcoxon(d_nh, d_hd)
        except Exception:
            p = float("nan")
        dd_p.append(p)
        dd_n.append(len(pass_seeds))

    print("  Precomputing relocation entropy sensitivity...")
    all_reloc_seeds = _discover_reloc_seeds(reloc_dir)
    # entropy value per seed computed once (independent of min_entropy --
    # min_entropy only thresholds the already-computed value below)
    reloc_min_ent = {}
    for s in all_reloc_seeds:
        data_dir = Path(reloc_dir) / f"reloc_seed{s}" / "data"
        ents = [peak_entropy(data_dir, p) for p in _RELOC_PERIODS]
        reloc_min_ent[s] = min(ents) if all(e is not None for e in ents) else None
    reloc_n_clean = [
        sum(1 for e in reloc_min_ent.values() if e is not None and e >= thr)
        for thr in ENTROPY_SWEEP
    ]

    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2))

    ax = axes[0]
    ax.plot(ENTROPY_SWEEP, dd_p, "o-", color="0.25", lw=1.0, markersize=4)
    ax.axhline(0.05, color="0.5", lw=0.7, ls="--", label="p=0.05")
    ax.axvline(MIN_ENTROPY, color="#8B4513", lw=0.9, ls=":", label=f"used ({MIN_ENTROPY})")
    ax.set_yscale("log")
    ax.set_xlabel("MIN_ENTROPY threshold")
    ax.set_ylabel("Double-dissociation p\n(head vs no-head, paired)")
    ax.set_title("Fig. 5b stat vs\nentropy threshold", fontsize=6, pad=2)
    ax.legend(fontsize=4.5)

    ax2 = axes[1]
    ax2.plot(ENTROPY_SWEEP, reloc_n_clean, "s-", color="#2a9d8f", lw=1.0, markersize=4)
    ax2.axvline(MIN_ENTROPY, color="#8B4513", lw=0.9, ls=":", label=f"used ({MIN_ENTROPY})")
    ax2.set_xlabel("MIN_ENTROPY threshold")
    ax2.set_ylabel("# clean relocation seeds")
    ax2.set_title(f"Relocation clean count\nvs threshold (of {len(all_reloc_seeds)} total)", fontsize=6, pad=2)
    ax2.legend(fontsize=4.5)

    fig.suptitle("Entropy-gate sensitivity", fontsize=6.5, y=1.04)
    fig.tight_layout(pad=0.4, w_pad=0.6)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel b — Skaggs-threshold sensitivity
# ---------------------------------------------------------------------------

def fig_supp_skaggs_sensitivity(raw_nh, raw_hd, stem):
    """Panel b -- the same double-dissociation statistic swept over the Skaggs threshold."""
    from scipy.stats import wilcoxon as _wilcoxon
    common_seeds = sorted(set(raw_nh) & set(raw_hd))
    # seed selection (entropy gate) fixed at MIN_ENTROPY / default SKAGGS_THRESHOLD,
    # matching the main figure -- only the AC-contrast Skaggs gate is swept below
    clean_seeds = [
        s for s in common_seeds
        if (e_nh := _entropy_at_threshold(raw_nh[s])) is not None and e_nh >= MIN_ENTROPY
        and (e_hd := _entropy_at_threshold(raw_hd[s])) is not None and e_hd >= MIN_ENTROPY
    ]
    REWARD_A = np.array([0.3, 0.3])

    means_nh, means_hd, ps, ns = [], [], [], []
    for thr in SKAGGS_SWEEP:
        d_nh_list, d_hd_list = [], []
        for s in clean_seeds:
            dn = _delta_at_skaggs_threshold(raw_nh[s], REWARD_A, thr)
            dh = _delta_at_skaggs_threshold(raw_hd[s], REWARD_A, thr)
            if dn is not None and dh is not None:
                d_nh_list.append(dn["mean_delta"])
                d_hd_list.append(dh["mean_delta"])
        if len(d_nh_list) < 3:
            means_nh.append(np.nan); means_hd.append(np.nan); ps.append(np.nan); ns.append(len(d_nh_list))
            continue
        d_nh, d_hd = np.array(d_nh_list), np.array(d_hd_list)
        means_nh.append(float(d_nh.mean()))
        means_hd.append(float(d_hd.mean()))
        try:
            _, p = _wilcoxon(d_nh, d_hd)
        except Exception:
            p = float("nan")
        ps.append(p)
        ns.append(len(d_nh_list))

    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.plot(SKAGGS_SWEEP, means_nh, "o-", color=COLOR_NOHEAD, lw=1.0, markersize=4, label="no-head")
    ax.plot(SKAGGS_SWEEP, means_hd, "D-", color=COLOR_HEAD, lw=1.0, markersize=4, label="head ON")
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.axvline(SKAGGS_THRESHOLD, color="0.3", lw=0.9, ls=":", label=f"used ({SKAGGS_THRESHOLD})")
    ax.set_xlabel("Skaggs threshold")
    ax.set_ylabel("Mean Δd frozen replay\n(positive = toward reward)")
    p_strs = "  ".join(_fmt_p(p) for p in ps)
    ax.set_title(f"Skaggs-threshold sensitivity\np: {p_strs}", fontsize=5.3, pad=2)
    ax.legend(fontsize=5)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel c — SR-predicted shift direction test
# ---------------------------------------------------------------------------

def fig_supp_sr_direction(nohead_dir, head_dir, seeds, stem):
    """Panel c -- SR-predicted against actual shift direction: pooled angle histogram and residual against distance to reward."""
    rows_nh = load_all_seeds_direction(nohead_dir, "nohead_", seeds)
    rows_hd = load_all_seeds_direction(head_dir, "head_", seeds)

    all_angles = np.concatenate(
        [r["angles"] for r in rows_nh + rows_hd if len(r["angles"])]
    )
    all_angles = all_angles[np.isfinite(all_angles)]

    from scipy.stats import wilcoxon as _wilcoxon
    all_residual = np.concatenate([r["residual_mag"] for r in rows_nh + rows_hd])
    all_dist = np.concatenate([r["dist_to_reward"] for r in rows_nh + rows_hd])
    r_pooled, _ = spearmanr(all_residual, all_dist)          # pooled r for display only
    # tests are across networks: cells within a network are not independent
    per_seed_r = np.array([r["residual_vs_dist_r"] for r in rows_nh + rows_hd], dtype=float)
    per_seed_r = per_seed_r[np.isfinite(per_seed_r)]
    p_resid = float(_wilcoxon(per_seed_r)[1]) if len(per_seed_r) >= 5 else float("nan")

    # "R" holds the V statistic of rayleigh_from_angles: mean cos(angle), 0 = no relationship
    per_seed_V = np.array([r["R"] for r in rows_nh + rows_hd], dtype=float)
    per_seed_V = per_seed_V[np.isfinite(per_seed_V)]
    mean_V = np.mean(per_seed_V) if len(per_seed_V) else float("nan")
    p_V = float(_wilcoxon(per_seed_V, alternative="greater")[1]) if len(per_seed_V) >= 5 else float("nan")

    fig = plt.figure(figsize=(4.6, 2.3))

    # Left: polar histogram of pooled angles (0 = perfectly aligned with SR prediction)
    ax0 = fig.add_subplot(1, 2, 1, projection="polar")
    bins = np.linspace(0, np.pi, 19)
    counts, edges = np.histogram(all_angles, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    ax0.bar(centers, counts, width=width, color="#4C72B0", alpha=0.75, edgecolor="0.3", linewidth=0.4)
    ax0.set_thetamin(0)
    ax0.set_thetamax(180)
    ax0.set_title(f"Actual vs SR-predicted shift angle\n(mean V={mean_V:+.2f}, one-sided p={_fmt_p(p_V)} over "
                  f"{len(per_seed_V)} networks; n={len(all_angles)} cells)", fontsize=5.8, pad=8)

    # Right: residual magnitude vs distance to reward
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.scatter(all_dist, all_residual, s=3, alpha=0.25, color="#B23A48", linewidths=0)
    m, b = np.polyfit(all_dist, all_residual, 1)
    xf = np.linspace(all_dist.min(), all_dist.max(), 50)
    ax1.plot(xf, m * xf + b, "--", color="0.2", lw=1.0)
    ax1.set_xlabel("Distance to reward (Phase A)")
    ax1.set_ylabel("Residual shift magnitude\n(orthogonal to SR prediction)")
    ax1.set_title(f"Residual vs distance\npooled r={r_pooled:+.2f}; per-network r vs 0: p={_fmt_p(p_resid)}",
                  fontsize=5.8, pad=2)

    fig.suptitle(f"SR direction test (A→B contrast, {len(rows_nh)+len(rows_hd)} seed-conditions)",
                 fontsize=6.3, y=1.03)
    fig.tight_layout(pad=0.5, w_pad=1.2)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel d — relocation correlation-vs-n-seeds robustness
# ---------------------------------------------------------------------------

def fig_supp_reloc_robustness(reloc_dir, clean_seeds, stem, n_resamples=200):
    """Panel d -- how the relocation Spearman correlations stabilise as the clean-seed pool grows."""
    n_clean = len(clean_seeds)
    if n_clean < 4:
        print(f"  WARNING: only {n_clean} clean relocation seeds -- robustness curve will be short.")

    all_rows = load_relocation_summary(reloc_dir, clean_seeds)
    if len(all_rows) < 4:
        print("  Too few relocation rows for panel d; skipping.")
        return None, None

    n_grid = sorted(set(list(range(4, len(all_rows))) + [len(all_rows)]))
    rng = np.random.default_rng(0)

    rho_c_med, rho_c_lo, rho_c_hi = [], [], []
    rho_d_med, rho_d_lo, rho_d_hi = [], [], []

    for n in n_grid:
        rhos_c, rhos_d = [], []
        for _ in range(n_resamples):
            idx = rng.choice(len(all_rows), size=n, replace=False)
            sub = [all_rows[i] for i in idx]
            xs_c = np.array([r["succ_c2"] for r in sub])
            ys_c = np.array([r["frac_B"] for r in sub])
            xs_d = np.array([r["delta_A"] for r in sub])
            ys_d = np.array([r["delta_B"] for r in sub])
            if len(set(xs_c)) > 1 and len(set(ys_c)) > 1:
                rc, _ = spearmanr(xs_c, ys_c)
                rhos_c.append(rc)
            if len(set(xs_d)) > 1 and len(set(ys_d)) > 1:
                rd, _ = spearmanr(xs_d, ys_d)
                rhos_d.append(rd)
        rhos_c = np.array(rhos_c) if rhos_c else np.array([np.nan])
        rhos_d = np.array(rhos_d) if rhos_d else np.array([np.nan])
        rho_c_med.append(np.nanmedian(rhos_c))
        rho_c_lo.append(np.nanpercentile(rhos_c, 25))
        rho_c_hi.append(np.nanpercentile(rhos_c, 75))
        rho_d_med.append(np.nanmedian(rhos_d))
        rho_d_lo.append(np.nanpercentile(rhos_d, 25))
        rho_d_hi.append(np.nanpercentile(rhos_d, 75))

    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    ax.plot(n_grid, rho_c_med, color="#e76f51", lw=1.2, label="Fig 5c (behavior vs %B)")
    ax.fill_between(n_grid, rho_c_lo, rho_c_hi, color="#e76f51", alpha=0.2, linewidth=0)
    ax.plot(n_grid, rho_d_med, color="#2a9d8f", lw=1.2, label="Fig 5d (ΔA vs ΔB)")
    ax.fill_between(n_grid, rho_d_lo, rho_d_hi, color="#2a9d8f", alpha=0.2, linewidth=0)
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.set_xlabel("# relocation seeds (n, resampled)")
    ax.set_ylabel("Spearman ρ\n(median, IQR band)")
    ax.set_title(f"Correlation stability vs n\n(full pool: {len(all_rows)} clean seeds)", fontsize=6, pad=2)
    ax.legend(fontsize=4.8, loc="best")
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
    """Two-row layout: [a b] / [c d]."""
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
    """Command-line arguments: condition directories, seed list, output directory, panel subset."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nohead_dir", default=str(NOHEAD_DIR))
    p.add_argument("--head_dir", default=str(HEAD_DIR))
    p.add_argument("--reloc_dir", default=str(RELOC_DIR))
    p.add_argument("--dd_seeds", type=int, nargs="+", default=list(range(50)),
                   help="Seed range for the double-dissociation sensitivity sweeps (a, b)")
    p.add_argument("--direction_seeds", type=int, nargs="+", default=list(range(20)),
                   help="Seed range for the SR-direction test (c) -- capped by default "
                        "since each seed needs an SR matrix solve")
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c,d")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    args = p.parse_args()
    args.panels = {x.strip() for x in args.panels.split(",") if x.strip()}
    return args


def main():
    """Precompute the shared frozen-replay maps, build the requested panels, and assemble the figure."""
    args = parse_args()
    panels_dir = ensure_dir(Path(args.output_dir) / "panels")
    ensure_dir(Path(args.output_dir))

    nohead_dir, head_dir, reloc_dir = Path(args.nohead_dir), Path(args.head_dir), Path(args.reloc_dir)
    panel_pdfs: dict[str, Path] = {}

    # Panels a and b both sweep thresholds over the same underlying
    # frozen-replay rate maps -- precompute once here and reuse, rather than
    # each panel independently reloading from disk.
    if {"a", "b"} & args.panels:
        print(f"Precomputing frozen-replay data once for {len(args.dd_seeds)} seeds (nohead + head)...")
        raw_nh = _precompute_frozen_replay_raw(nohead_dir, "nohead_", args.dd_seeds)
        raw_hd = _precompute_frozen_replay_raw(head_dir, "head_", args.dd_seeds)
        print(f"  Loaded {len(raw_nh)} nohead / {len(raw_hd)} head seeds.")

    if "a" in args.panels:
        print("Panel a (entropy-gate sensitivity sweep)...")
        _, pdf = fig_supp_entropy_sensitivity(raw_nh, raw_hd, reloc_dir, panels_dir / "panel_a")
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Panel b (Skaggs-threshold sensitivity)...")
        _, pdf = fig_supp_skaggs_sensitivity(raw_nh, raw_hd, panels_dir / "panel_b")
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Panel c (SR-predicted shift direction test)...")
        _, pdf = fig_supp_sr_direction(nohead_dir, head_dir, args.direction_seeds, panels_dir / "panel_c")
        panel_pdfs["c"] = pdf

    if "d" in args.panels:
        print("Panel d (relocation correlation-vs-n-seeds robustness)...")
        all_reloc_seeds = _discover_reloc_seeds(reloc_dir)
        clean_seeds = [s for s in all_reloc_seeds
                       if seed_is_clean(reloc_dir / f"reloc_seed{s}" / "data", _RELOC_PERIODS)]
        print(f"  {len(clean_seeds)}/{len(all_reloc_seeds)} relocation seeds pass the gate")
        result = fig_supp_reloc_robustness(reloc_dir, clean_seeds, panels_dir / "panel_d")
        if result and result[1] is not None:
            panel_pdfs["d"] = result[1]

    if not args.no_assemble and panel_pdfs:
        print("\nAssembling supplementary figure PDF...")
        assemble_supp(panel_pdfs, Path(args.output_dir) / "figure5_supp_raw")

    print(f"\nDone. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
