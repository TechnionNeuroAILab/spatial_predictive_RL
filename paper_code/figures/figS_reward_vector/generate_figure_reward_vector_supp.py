#!/usr/bin/env python3
"""
Supplementary Figure — Reward-distance and goal-vectorial representation.

Converts two hippocampal-physiology citations that were previously only
narrative analogies in the paper into real quantitative tests:
  - Gauthier & Tank 2018 (reward-distance-modulated activity / "distance cells")
  - Sarel et al. 2017 (egocentric goal-vector tuning / "goal-direction cells"),
    with an allocentric head-direction control and a relocation (A->B)
    rotation test establishing goal-specificity.

Panels:
  a  Gauthier distance-tuning summary (no-head vs head-ON)
  b  Sarel goal-vector summary (no-head vs head-ON)
  c  Relocation rotation test, behavioral version (cycle_1_goal vs cycle_2_goal)
  d  Relocation rotation test, occupancy-controlled version (frozen-replay triple)
  e  Example tuning curves (one distance cell, one goal-vector cell, one HD-only cell)
  f  Wall-buffer / border-effect sensitivity sweep
  g  Frozen-replay negative control (validity check, reproduces legacy null result)

Usage (run from repo root):
    python "code_analysis/figure_reward_vector/generate_figure_reward_vector_supp.py" --no_show
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analysis import (  # noqa: E402
    discover_seeds, clean_seeds_for_reward_vector, load_condition_population_summary,
    load_reloc_rotation_summary, save_summary_cache,
    GOAL_DIRECTED_PERIODS, RELOC_GATE_PERIODS,
)
from utils import (  # noqa: E402
    NOHEAD_DIR, HEAD_DIR, RELOC_DIR, MM_TO_PT, PAGE_W,
    COLOR_NOHEAD, COLOR_HEAD, COLOR_A, COLOR_B, COLOR_DISTANCE, COLOR_GOALDIR, COLOR_HD,
    apply_nature_style, ensure_dir, save_figure,
)

_NEW_CODE = REPO / "paper_code" / "lib"
sys.path.insert(0, str(_NEW_CODE))
from reward_vector_analysis import (  # noqa: E402
    analyze_period, classify_cells, wall_buffer_mask, EDGE_MARGIN_CELLS,
)

apply_nature_style()

SUPP_FIGURES_DIR = HERE / "figures_supp"


def _fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# Panel a — Gauthier distance-tuning summary
# ---------------------------------------------------------------------------

def fig_gauthier_summary(rows_nh, rows_hd, stem):
    """Bar: baseline-subtracted fraction of genuine distance_cells
    (goal_conditioned minus before), no-head vs head-ON."""
    delta_nh = np.array([r["delta_frac_distance_cell"] for r in rows_nh]) * 100
    delta_hd = np.array([r["delta_frac_distance_cell"] for r in rows_hd]) * 100
    raw_nh = np.array([r["frac_distance_cell"] for r in rows_nh]) * 100
    raw_hd = np.array([r["frac_distance_cell"] for r in rows_hd]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.3))

    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, (vals, color, label) in enumerate([(raw_nh, COLOR_NOHEAD, "no-head"), (raw_hd, COLOR_HEAD, "head-ON")]):
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, s=10, color=color, alpha=0.6, zorder=3, linewidths=0)
        ax.errorbar([i], [vals.mean()], yerr=[vals.std(ddof=1)], fmt="D", color="black",
                    markersize=5, capsize=3, lw=1.0, zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no-head", "head-ON"])
    ax.set_ylabel("% genuine distance cells\n(goal_conditioned, raw)")
    ax.set_title(f"Raw fraction\n(n_nh={len(raw_nh)}, n_hd={len(raw_hd)})", fontsize=6)

    ax = axes[1]
    for i, (vals, color, label) in enumerate([(delta_nh, COLOR_NOHEAD, "no-head"), (delta_hd, COLOR_HEAD, "head-ON")]):
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, s=10, color=color, alpha=0.6, zorder=3, linewidths=0)
        ax.errorbar([i], [vals.mean()], yerr=[vals.std(ddof=1)], fmt="D", color="black",
                    markersize=5, capsize=3, lw=1.0, zorder=5)
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no-head", "head-ON"])
    ax.set_ylabel("Δ % distance cells\n(goal_conditioned − before)")
    try:
        _, p_nh = _wilcoxon(delta_nh)
        _, p_hd = _wilcoxon(delta_hd)
    except Exception:
        p_nh = p_hd = float("nan")
    ax.set_title(f"Reward-specific component\nnh {_fmt_p(p_nh)}  hd {_fmt_p(p_hd)}", fontsize=6)

    fig.suptitle("Gauthier & Tank 2018: reward-distance-modulated activity", fontsize=6.5, y=1.03)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel b — Sarel goal-vector summary
# ---------------------------------------------------------------------------

def fig_sarel_summary(rows_nh, rows_hd, stem):
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.3))

    ax = axes[0]
    mrl_ego_nh = np.array([r["mean_mrl_ego"] for r in rows_nh])
    mrl_hd_nh = np.array([r["mean_mrl_hd"] for r in rows_nh])
    mrl_ego_hd = np.array([r["mean_mrl_ego"] for r in rows_hd])
    mrl_hd_hd = np.array([r["mean_mrl_hd"] for r in rows_hd])
    rng = np.random.default_rng(1)
    positions = [0, 1, 2, 3]
    data = [mrl_ego_nh, mrl_hd_nh, mrl_ego_hd, mrl_hd_hd]
    colors = [COLOR_GOALDIR, COLOR_HD, COLOR_GOALDIR, COLOR_HD]
    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, s=8, color=color, alpha=0.55, zorder=3, linewidths=0)
        ax.errorbar([i], [vals.mean()], yerr=[vals.std(ddof=1)], fmt="D", color="black",
                    markersize=4, capsize=3, lw=1.0, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(["ego\n(nh)", "HD\n(nh)", "ego\n(hd)", "HD\n(hd)"], fontsize=5.5)
    ax.set_ylabel("Mean MRL")
    ax.set_title("Ego-bearing vs head-direction\ntuning strength", fontsize=6)

    ax = axes[1]
    frac_goaldir_nh = np.array([r["frac_goal_direction_cell"] for r in rows_nh]) * 100
    frac_hd_nh = np.array([r["frac_head_direction_cell"] for r in rows_nh]) * 100
    frac_specific_nh = np.array([r["frac_goal_specific_cell"] for r in rows_nh]) * 100
    frac_vector_nh = np.array([r["frac_goal_vector_cell"] for r in rows_nh]) * 100
    labels = ["goal-dir", "HD", "goal-\nspecific", "goal-\nvector"]
    means = [frac_goaldir_nh.mean(), frac_hd_nh.mean(), frac_specific_nh.mean(), frac_vector_nh.mean()]
    sds = [frac_goaldir_nh.std(ddof=1), frac_hd_nh.std(ddof=1), frac_specific_nh.std(ddof=1), frac_vector_nh.std(ddof=1)]
    xs = np.arange(4)
    ax.bar(xs, means, yerr=sds, color=[COLOR_GOALDIR, COLOR_HD, "#e69f00", COLOR_DISTANCE],
           alpha=0.8, capsize=3, width=0.6, error_kw={"lw": 1.0})
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("% cells (no-head)")
    ax.set_title("Cell-type fractions\n(goal_conditioned)", fontsize=6)

    fig.suptitle("Sarel et al. 2017: egocentric goal-direction tuning", fontsize=6.5, y=1.03)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panels c/d — relocation rotation test
# ---------------------------------------------------------------------------

def fig_rotation_test(rot_rows, title, stem):
    """Rose histogram + circular-mean-vs-CI comparison for the relocation
    A->B rotation test. Works for either the behavioral or occupancy-
    controlled period pair (caller supplies pre-computed rot_rows).

    The headline statistic is the POOLED cell-level circular mean +- bootstrap
    CI (large n across all stable goal-vector cells pooled over seeds) -- this
    is far better powered than eyeballing the noisy per-seed circular-mean
    scatter (each seed's own mean is based on only ~300 cells and single-
    trajectory sampling, so per-seed magnitude vs Delta_geom is a noisier,
    seed-level comparison). Both numbers are reported: the pooled mean is the
    rigorous test of "is the population-level shift ~0", while the per-seed
    paired magnitude comparison (Wilcoxon, |obs| vs |geom|) tests whether
    individual seeds' rotations track the geometric artifact less than
    expected under an HD-driven account."""
    all_dphi_goal = np.concatenate([np.asarray(r["dphi_goal_arr"]) for r in rot_rows if len(r["dphi_goal_arr"])])
    delta_geoms = np.array([r["delta_geom"] for r in rot_rows])
    circ_means = np.array([r["circ_mean_dphi_goal"] for r in rot_rows])
    n_stable = np.array([r["n_stable_goal"] for r in rot_rows])

    # Pooled cell-level circular mean +- bootstrap CI (headline statistic).
    rng = np.random.default_rng(0)
    n = len(all_dphi_goal)
    if n > 0:
        boot = np.array([np.angle(np.mean(np.exp(1j * all_dphi_goal[rng.integers(0, n, n)])))
                          for _ in range(2000)])
        pooled_mean = np.angle(np.mean(np.exp(1j * all_dphi_goal)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
    else:
        pooled_mean, lo, hi = float("nan"), float("nan"), float("nan")

    # Per-seed paired magnitude test: does the observed rotation track the
    # geometric artifact less than expected (|obs| < |geom|)?
    try:
        _, p_mag = _wilcoxon(np.abs(circ_means), np.abs(delta_geoms))
    except Exception:
        p_mag = float("nan")
    n_favorable = int((np.abs(circ_means) < np.abs(delta_geoms)).sum())

    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4))
    axes[0].remove()
    axp = fig.add_subplot(1, 2, 1, projection="polar")
    bins = np.linspace(-np.pi, np.pi, 25)
    counts, edges = np.histogram(all_dphi_goal, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    axp.bar(centers, counts, width=width, color=COLOR_GOALDIR, alpha=0.75, edgecolor="0.3", linewidth=0.3)
    axp.set_theta_zero_location("N")
    axp.set_title(f"Δφ_ego (goal-direction cells)\n(n={len(all_dphi_goal)} cells, {len(rot_rows)} seeds)", fontsize=6)

    ax = axes[1]
    rng2 = np.random.default_rng(2)
    jitter = rng2.uniform(-0.08, 0.08, len(circ_means))
    ax.scatter(jitter, np.degrees(circ_means), s=14, color=COLOR_GOALDIR, alpha=0.7, zorder=3, linewidths=0, label="Δφ_ego (per-seed)")
    ax.scatter(1 + jitter, np.degrees(delta_geoms), s=14, color="0.5", alpha=0.7, zorder=3, linewidths=0, marker="s", label="Δ_geom (per-seed)")
    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.errorbar([0], [np.degrees(pooled_mean)], yerr=[[np.degrees(pooled_mean - lo)], [np.degrees(hi - pooled_mean)]],
                fmt="D", color="black", markersize=5, capsize=3, lw=1.2, zorder=6,
                label="pooled mean ± 95% CI")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Δφ_ego\n(observed)", "Δ_geom\n(HD-artifact\nprediction)"], fontsize=5.5)
    ax.set_ylabel("Bearing shift A→B (deg)")
    ax.set_title(
        f"Pooled Δφ_ego: {np.degrees(pooled_mean):+.1f}° [{np.degrees(lo):+.1f},{np.degrees(hi):+.1f}]\n"
        f"|obs|<|geom| in {n_favorable}/{len(circ_means)} seeds (Wilcoxon {_fmt_p(p_mag)})",
        fontsize=5.5)
    ax.legend(fontsize=5, frameon=False, loc="best")

    fig.suptitle(title, fontsize=6.5, y=1.03)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel e — example tuning curves
# ---------------------------------------------------------------------------

def fig_example_curves(nohead_seed_dir, reloc_seed_dir, reloc_period_a, reloc_period_b, stem):
    res_nh = analyze_period(nohead_seed_dir, "goal_conditioned", n_null=50, seed=0)
    labels_nh = classify_cells(res_nh["ramp"], res_nh["ramp_p"], res_nh["mrl_ego"], res_nh["mrl_ego_p"],
                                res_nh["mrl_hd"], res_nh["mrl_hd_p"], ring_mrl=res_nh["ring_mrl"])

    res_a = analyze_period(reloc_seed_dir, reloc_period_a, n_null=50, seed=0)
    res_b = analyze_period(reloc_seed_dir, reloc_period_b, n_null=50, seed=0)
    labels_a = classify_cells(res_a["ramp"], res_a["ramp_p"], res_a["mrl_ego"], res_a["mrl_ego_p"],
                               res_a["mrl_hd"], res_a["mrl_hd_p"], ring_mrl=res_a["ring_mrl"])
    labels_b = classify_cells(res_b["ramp"], res_b["ramp_p"], res_b["mrl_ego"], res_b["mrl_ego_p"],
                               res_b["mrl_hd"], res_b["mrl_hd_p"], ring_mrl=res_b["ring_mrl"])

    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.0))

    # (1) example distance cell
    ax = axes[0]
    dist_ids = np.where(labels_nh["distance_cell"])[0]
    if len(dist_ids):
        n = dist_ids[np.argmax(np.abs(res_nh["ramp"][dist_ids]))]
        curve = res_nh["dist_curve"][n]
        ax.plot(res_nh["dbins"], curve / (curve.max() + 1e-12), "-o", color=COLOR_DISTANCE, ms=3)
        ax.set_title(f"Example distance cell\n(ramp={res_nh['ramp'][n]:+.2f})", fontsize=6)
    else:
        ax.text(0.5, 0.5, "no distance cell\nfound", ha="center", va="center", transform=ax.transAxes, fontsize=6)
        ax.set_title("Example distance cell", fontsize=6)
    ax.set_xlabel("distance to reward"); ax.set_ylabel("norm. activity")

    # (2) rotation-confirmed goal-vector cell (ego-bearing curve A vs B overlaid)
    ax = axes[1]
    stable_goal = labels_a["goal_direction_cell"] & labels_b["goal_direction_cell"]
    stable_goal_only = stable_goal & ~labels_a["head_direction_cell"] & ~labels_b["head_direction_cell"]
    candidates = np.where(stable_goal_only)[0] if stable_goal_only.sum() else np.where(stable_goal)[0]
    if len(candidates):
        n = candidates[np.argmax(res_a["mrl_ego"][candidates] + res_b["mrl_ego"][candidates])]
        ca, cb = res_a["curve_ego"][n], res_b["curve_ego"][n]
        ax.plot(np.degrees(res_a["phi_ego"]), ca / (ca.max() + 1e-12), "-o", color=COLOR_A, ms=3, label="phase A")
        ax.plot(np.degrees(res_b["phi_ego"]), cb / (cb.max() + 1e-12), "-o", color=COLOR_B, ms=3, label="phase B")
        ax.set_title("Goal-direction cell\n(overlapping = re-anchors)", fontsize=6)
        ax.legend(fontsize=5, frameon=False)
    else:
        ax.text(0.5, 0.5, "no stable goal-direction\ncell found", ha="center", va="center", transform=ax.transAxes, fontsize=6)
        ax.set_title("Goal-direction cell", fontsize=6)
    ax.set_xlabel("ego-bearing to goal (deg)"); ax.set_ylabel("norm. activity")

    # (3) HD-only cell (should visibly rotate in ego-bearing terms)
    ax = axes[2]
    stable_hd_only = (labels_a["head_direction_cell"] & labels_b["head_direction_cell"]
                       & ~labels_a["goal_direction_cell"] & ~labels_b["goal_direction_cell"])
    candidates_hd = np.where(stable_hd_only)[0]
    if len(candidates_hd):
        n = candidates_hd[np.argmax(res_a["mrl_hd"][candidates_hd])]
        ca, cb = res_a["curve_ego"][n], res_b["curve_ego"][n]
        ax.plot(np.degrees(res_a["phi_ego"]), ca / (ca.max() + 1e-12), "-o", color=COLOR_A, ms=3, label="phase A")
        ax.plot(np.degrees(res_b["phi_ego"]), cb / (cb.max() + 1e-12), "-o", color=COLOR_B, ms=3, label="phase B")
        ax.set_title("HD-only cell\n(shifted = artifact)", fontsize=6)
        ax.legend(fontsize=5, frameon=False)
    else:
        ax.text(0.5, 0.5, "no stable HD-only\ncell found", ha="center", va="center", transform=ax.transAxes, fontsize=6)
        ax.set_title("HD-only cell", fontsize=6)
    ax.set_xlabel("ego-bearing to goal (deg)"); ax.set_ylabel("norm. activity")

    fig.suptitle("Example cells", fontsize=6.5, y=1.04)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel f — wall-buffer sensitivity sweep
# ---------------------------------------------------------------------------

def fig_wallbuffer_sweep(cond_dirs_prefixes_seeds, stem, margins=(0, 1, 2, 3)):
    """cond_dirs_prefixes_seeds: list of (cond_dir, prefix, seeds, color, label)."""
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.2))

    for cond_dir, prefix, seeds, color, label in cond_dirs_prefixes_seeds:
        frac_means, mrl_means = [], []
        for margin in margins:
            fracs, mrls = [], []
            for s in seeds:
                run_dir = Path(cond_dir) / f"{prefix}seed{s}"
                res = analyze_period(run_dir, "goal_conditioned", margin_cells=margin, n_null=50, seed=s)
                if res is None:
                    continue
                labels = classify_cells(res["ramp"], res["ramp_p"], res["mrl_ego"], res["mrl_ego_p"],
                                         res["mrl_hd"], res["mrl_hd_p"], ring_mrl=res["ring_mrl"])
                fracs.append(labels["distance_cell"].mean())
                mrls.append(res["mrl_ego"].mean())
            frac_means.append(np.mean(fracs) * 100 if fracs else np.nan)
            mrl_means.append(np.mean(mrls) if mrls else np.nan)
        axes[0].plot(margins, frac_means, "-o", color=color, label=label, ms=4)
        axes[1].plot(margins, mrl_means, "-o", color=color, label=label, ms=4)

    axes[0].set_xlabel("wall-buffer margin (cells)")
    axes[0].set_ylabel("% distance cells")
    axes[0].set_title("Distance-cell fraction\nvs border margin", fontsize=6)
    axes[0].legend(fontsize=5, frameon=False)

    axes[1].set_xlabel("wall-buffer margin (cells)")
    axes[1].set_ylabel("mean ego-bearing MRL")
    axes[1].set_title("Goal-direction MRL\nvs border margin", fontsize=6)
    axes[1].legend(fontsize=5, frameon=False)

    fig.suptitle("Wall-buffer / border-effect sensitivity", fontsize=6.5, y=1.03)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel g — frozen-replay negative control
# ---------------------------------------------------------------------------

def fig_frozen_replay_control(rows_nh_replay, rows_hd_replay, rows_nh_goal, rows_hd_goal, stem):
    """Reproduces the legacy README's null result: ramping/MRL on the single
    frozen-replay trajectory (replay_c1_pred/postrl) are near-zero/non-
    significant, contrasting with real tuning on goal_conditioned --
    justifying why tuning curves are built from goal-directed periods."""
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.2))

    ax = axes[0]
    labels = ["replay\n(no-head)", "goal_cond.\n(no-head)", "replay\n(head)", "goal_cond.\n(head)"]
    data = [
        np.array([r["frac_distance_cell"] for r in rows_nh_replay]) * 100,
        np.array([r["frac_distance_cell"] for r in rows_nh_goal]) * 100,
        np.array([r["frac_distance_cell"] for r in rows_hd_replay]) * 100,
        np.array([r["frac_distance_cell"] for r in rows_hd_goal]) * 100,
    ]
    colors = [COLOR_NOHEAD, COLOR_NOHEAD, COLOR_HEAD, COLOR_HEAD]
    xs = np.arange(4)
    means = [d.mean() if len(d) else 0 for d in data]
    sds = [d.std(ddof=1) if len(d) > 1 else 0 for d in data]
    ax.bar(xs, means, yerr=sds, color=colors, capsize=3, width=0.6)
    for i, c in enumerate(colors):
        ax.patches[i].set_alpha(0.4 if i % 2 == 0 else 0.9)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("% distance cells")
    ax.set_title("Distance-cell fraction:\nfrozen replay vs goal-directed", fontsize=6)

    ax = axes[1]
    data2 = [
        np.array([r["mean_mrl_ego"] for r in rows_nh_replay]),
        np.array([r["mean_mrl_ego"] for r in rows_nh_goal]),
        np.array([r["mean_mrl_ego"] for r in rows_hd_replay]),
        np.array([r["mean_mrl_ego"] for r in rows_hd_goal]),
    ]
    means2 = [d.mean() if len(d) else 0 for d in data2]
    sds2 = [d.std(ddof=1) if len(d) > 1 else 0 for d in data2]
    ax.bar(xs, means2, yerr=sds2, color=colors, capsize=3, width=0.6)
    for i in range(4):
        ax.patches[i].set_alpha(0.4 if i % 2 == 0 else 0.9)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("mean ego-bearing MRL")
    ax.set_title("Goal-direction MRL:\nfrozen replay vs goal-directed", fontsize=6)

    fig.suptitle("Frozen-replay negative control (validity check)", fontsize=6.5, y=1.03)
    fig.tight_layout(pad=0.4, w_pad=0.8)
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
    """Layout: [a b] / [c d] / [e] / [f g]."""
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

    rows_def = [("a", "b"), ("c", "d"), ("e",), ("f", "g")]
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nohead_seeds", type=int, nargs="+", default=None)
    p.add_argument("--head_seeds", type=int, nargs="+", default=None)
    p.add_argument("--reloc_seeds", type=int, nargs="+", default=None)
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c,d,e,f,g")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    args = p.parse_args()
    args.panels = {x.strip() for x in args.panels.split(",") if x.strip()}
    return args


def main():
    args = parse_args()
    panels_dir = ensure_dir(Path(args.output_dir) / "panels")
    ensure_dir(Path(args.output_dir))
    panel_pdfs: dict[str, Path] = {}

    nohead_seeds = args.nohead_seeds if args.nohead_seeds is not None else discover_seeds(NOHEAD_DIR, "nohead_")
    head_seeds = args.head_seeds if args.head_seeds is not None else discover_seeds(HEAD_DIR, "head_")
    reloc_seeds = args.reloc_seeds if args.reloc_seeds is not None else discover_seeds(RELOC_DIR, "reloc_")

    print(f"Gating {len(nohead_seeds)} no-head, {len(head_seeds)} head-ON, {len(reloc_seeds)} reloc seeds...")
    nh_clean = clean_seeds_for_reward_vector(NOHEAD_DIR, "nohead_", nohead_seeds, GOAL_DIRECTED_PERIODS)
    hd_clean = clean_seeds_for_reward_vector(HEAD_DIR, "head_", head_seeds, GOAL_DIRECTED_PERIODS)
    reloc_clean = clean_seeds_for_reward_vector(RELOC_DIR, "reloc_", reloc_seeds, RELOC_GATE_PERIODS)
    print(f"  clean: {len(nh_clean)} no-head, {len(hd_clean)} head-ON, {len(reloc_clean)} reloc")

    rows_nh = rows_hd = None
    if {"a", "b", "g"} & args.panels:
        print("Loading no-head population summaries (goal_conditioned vs before)...")
        rows_nh = load_condition_population_summary(NOHEAD_DIR, "nohead_", nh_clean)
        save_summary_cache(NOHEAD_DIR, rows_nh)
        print("Loading head-ON population summaries...")
        rows_hd = load_condition_population_summary(HEAD_DIR, "head_", hd_clean)
        save_summary_cache(HEAD_DIR, rows_hd)

    if "a" in args.panels:
        print("Panel a (Gauthier distance-tuning summary)...")
        _, pdf = fig_gauthier_summary(rows_nh, rows_hd, panels_dir / "panel_a")
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Panel b (Sarel goal-vector summary)...")
        _, pdf = fig_sarel_summary(rows_nh, rows_hd, panels_dir / "panel_b")
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Panel c (relocation rotation test, behavioral version)...")
        rot_c = load_reloc_rotation_summary(RELOC_DIR, reloc_clean, "cycle_1_goal", "cycle_2_goal")
        save_summary_cache(RELOC_DIR, rot_c, name="reward_vector_rotation_behavioral.json")
        _, pdf = fig_rotation_test(rot_c, "Relocation rotation test (behavioral: cycle_1_goal vs cycle_2_goal)",
                                    panels_dir / "panel_c")
        panel_pdfs["c"] = pdf

    if "d" in args.panels:
        print("Panel d (relocation rotation test, occupancy-controlled version)...")
        rot_d = load_reloc_rotation_summary(RELOC_DIR, reloc_clean, "replay_reloc_postA", "replay_reloc_postB")
        save_summary_cache(RELOC_DIR, rot_d, name="reward_vector_rotation_occupancy_controlled.json")
        _, pdf = fig_rotation_test(rot_d, "Relocation rotation test (occupancy-controlled: frozen-replay triple)",
                                    panels_dir / "panel_d")
        panel_pdfs["d"] = pdf

    if "e" in args.panels:
        print("Panel e (example tuning curves)...")
        nh_example = Path(NOHEAD_DIR) / f"nohead_seed{nh_clean[0]}"
        reloc_example = Path(RELOC_DIR) / f"reloc_seed{reloc_clean[0]}"
        _, pdf = fig_example_curves(nh_example, reloc_example, "replay_reloc_postA", "replay_reloc_postB",
                                     panels_dir / "panel_e")
        panel_pdfs["e"] = pdf

    if "f" in args.panels:
        print("Panel f (wall-buffer sensitivity sweep)...")
        subset_nh = nh_clean[:10]
        subset_hd = hd_clean[:10]
        _, pdf = fig_wallbuffer_sweep(
            [(NOHEAD_DIR, "nohead_", subset_nh, COLOR_NOHEAD, "no-head"),
             (HEAD_DIR, "head_", subset_hd, COLOR_HEAD, "head-ON")],
            panels_dir / "panel_f",
        )
        panel_pdfs["f"] = pdf

    if "g" in args.panels:
        print("Panel g (frozen-replay negative control)...")
        rows_nh_replay = load_condition_population_summary(NOHEAD_DIR, "nohead_", nh_clean,
                                                            period="replay_c1_postrl", baseline_period="replay_c1_pred")
        rows_hd_replay = load_condition_population_summary(HEAD_DIR, "head_", hd_clean,
                                                            period="replay_c1_postrl", baseline_period="replay_c1_pred")
        _, pdf = fig_frozen_replay_control(rows_nh_replay, rows_hd_replay, rows_nh, rows_hd, panels_dir / "panel_g")
        panel_pdfs["g"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("\nAssembling supplementary figure PDF...")
        assemble_supp(panel_pdfs, Path(args.output_dir) / "figure_reward_vector_supp_raw")

    print(f"\nDone. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
