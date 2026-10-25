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
  a  reward-distance and goal-bearing measures on replayed versus goal-directed
     trajectories
  b  relocation rotation test on goal-directed trajectories
  c  relocation rotation test on the replayed trajectory

Usage (run from repo root):
    python code/figures/figS_reward_vector/generate_figure_reward_vector_supp.py --no_show
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
    COLOR_NOHEAD, COLOR_HEAD, COLOR_DISTANCE, COLOR_GOALDIR, COLOR_HD,
    apply_nature_style, ensure_dir, save_figure,
    FS_BASE, FS_SMALL, FS_TICK, title, stat_annot,
    fmt_p,
)
import figlayout  # noqa: E402
sys.path.insert(0, str(REPO / "code" / "lib"))
from analysis_helpers import campaign_passers  # noqa: E402
from figure_cache import (  # noqa: E402
    add_plot_only, load_figure_data, require_keys, update_figure_data,
)

_NEW_CODE = REPO / "code" / "lib"
sys.path.insert(0, str(_NEW_CODE))

from pubstyle import sem  # noqa: E402
apply_nature_style()

SUPP_FIGURES_DIR = HERE / "figures_supp"


def _fmt_p(p):
    """Format a p-value for a panel annotation; scientific notation below 0.001."""
    if not np.isfinite(p):
        return "n/a"
    """Panel p-value, in the same format as every other figure in the paper.

    Routed to the shared formatter so no panel invents its own notation, and
    none of them use a mathtext superscript (which prints at 0.7x and falls
    under the 5 pt floor).
    """
    return fmt_p(p, italic=False).replace("P ", "")


# ---------------------------------------------------------------------------
# Panel a — Gauthier distance-tuning summary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
#   a  replay versus goal-directed trajectories
#   b  relocation, goal-directed         c  relocation, replayed trajectory
LAYOUT = figlayout.Layout(
    [("a", 0, 0, 6),
     ("b", 1, 0, 3), ("c", 1, 3, 3)],
    ncols=6, row_h_in=[1.35, 1.75], caption_lines=6,
    name="figS_rewardvector",
)


def fig_gauthier_summary(rows_nh, rows_hd, stem, size_in=None):
    """Bar: baseline-subtracted fraction of genuine distance_cells
    (goal_conditioned minus before), no-head vs head-ON."""
    delta_nh = np.array([r["delta_frac_distance_cell"] for r in rows_nh]) * 100
    delta_hd = np.array([r["delta_frac_distance_cell"] for r in rows_hd]) * 100
    raw_nh = np.array([r["frac_distance_cell"] for r in rows_nh]) * 100
    raw_hd = np.array([r["frac_distance_cell"] for r in rows_hd]) * 100

    fig, axes = plt.subplots(1, 2, figsize=size_in or (4.4, 2.3))

    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, (vals, color, label) in enumerate([(raw_nh, COLOR_NOHEAD, "no-head"), (raw_hd, COLOR_HEAD, "head-ON")]):
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, s=10, color=color, alpha=0.6, zorder=3, linewidths=0)
        ax.errorbar([i], [vals.mean()], yerr=[sem(vals)], fmt="D", color="black",
                    markersize=5, capsize=3, lw=1.0, zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Without\nteaching", "With\nteaching"])
    ax.set_ylabel("Distance cells (%)")
    ax.set_title(f"Goal-directed (n = {len(raw_nh)})", fontsize=FS_TICK)

    ax = axes[1]
    for i, (vals, color, label) in enumerate([(delta_nh, COLOR_NOHEAD, "no-head"), (delta_hd, COLOR_HEAD, "head-ON")]):
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, s=10, color=color, alpha=0.6, zorder=3, linewidths=0)
        ax.errorbar([i], [vals.mean()], yerr=[sem(vals)], fmt="D", color="black",
                    markersize=5, capsize=3, lw=1.0, zorder=5)
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Without\nteaching", "With\nteaching"])
    ax.set_ylabel("Δ distance cells (%)\nvs before learning")
    try:
        _, p_nh = _wilcoxon(delta_nh)
        _, p_hd = _wilcoxon(delta_hd)
    except Exception:
        p_nh = p_hd = float("nan")
    ax.set_title(f"After − before\nP {_fmt_p(p_nh)}; P {_fmt_p(p_hd)}", fontsize=FS_TICK)

    fig.suptitle("Reward-distance cells", fontsize=FS_BASE, y=0.995, va="top")
    fig.tight_layout(rect=(0.012, 0, 1, 0.93), pad=0.4, w_pad=0.8)
    return save_figure(fig, stem, exact_size=size_in is not None)


# ---------------------------------------------------------------------------
# Panel b — Sarel goal-vector summary
# ---------------------------------------------------------------------------

def fig_sarel_summary(rows_nh, rows_hd, stem, size_in=None):
    """Panel b -- ego-bearing against head-direction tuning strength (mean MRL), no-head vs head-ON."""
    fig, axes = plt.subplots(1, 2, figsize=size_in or (4.4, 2.3))

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
        ax.errorbar([i], [vals.mean()], yerr=[sem(vals)], fmt="D", color="black",
                    markersize=4, capsize=3, lw=1.0, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Goal\nw/o", "HD\nw/o", "Goal\nwith", "HD\nwith"], fontsize=FS_SMALL)
    ax.set_ylabel("Mean resultant length")
    ax.set_title("Tuning strength", fontsize=FS_TICK)

    ax = axes[1]
    frac_goaldir_nh = np.array([r["frac_goal_direction_cell"] for r in rows_nh]) * 100
    frac_hd_nh = np.array([r["frac_head_direction_cell"] for r in rows_nh]) * 100
    frac_specific_nh = np.array([r["frac_goal_specific_cell"] for r in rows_nh]) * 100
    frac_vector_nh = np.array([r["frac_goal_vector_cell"] for r in rows_nh]) * 100
    labels = ["Goal\ndir.", "Head\ndir.", "Goal\nonly", "Goal\nvector"]
    means = [frac_goaldir_nh.mean(), frac_hd_nh.mean(), frac_specific_nh.mean(), frac_vector_nh.mean()]
    sds = [sem(frac_goaldir_nh), sem(frac_hd_nh), sem(frac_specific_nh), sem(frac_vector_nh)]
    xs = np.arange(4)
    ax.bar(xs, means, yerr=sds, color=[COLOR_GOALDIR, COLOR_HD, "#e69f00", COLOR_DISTANCE],
           alpha=0.8, capsize=3, width=0.6, error_kw={"lw": 1.0})
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=FS_SMALL)
    ax.set_ylabel("Cells (%)")
    ax.set_title("Cell types, without teaching", fontsize=FS_TICK)

    fig.suptitle("Goal-bearing tuning (goal-directed)", fontsize=FS_BASE, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4, w_pad=0.8)
    return save_figure(fig, stem, exact_size=size_in is not None)


# ---------------------------------------------------------------------------
# Panels d/e — relocation rotation test
# ---------------------------------------------------------------------------

def fig_rotation_test(rot_rows, suptitle_text, stem, size_in=None):
    """Rose histogram + circular-mean-vs-CI comparison for the relocation
    A->B rotation test. Works for either the behavioral or occupancy-
    controlled period pair (caller supplies pre-computed rot_rows).

    Headline: pooled circular mean of all stable goal-direction cells with a network-level
    bootstrap CI. Also a per-network paired Wilcoxon of |observed| vs |Delta_geom|; in the
    fixed-trajectory pair Delta_geom is near 0 by symmetry, so that comparison is not diagnostic there."""
    all_dphi_goal = np.concatenate([np.asarray(r["dphi_goal_arr"]) for r in rot_rows if len(r["dphi_goal_arr"])])
    delta_geoms = np.array([r["delta_geom"] for r in rot_rows])
    circ_means = np.array([r["circ_mean_dphi_goal"] for r in rot_rows])
    n_stable = np.array([r["n_stable_goal"] for r in rot_rows])

    # Pooled circular mean; 95% CI from resampling NETWORKS (cells within a network are not independent).
    rng = np.random.default_rng(0)
    per_net = [np.asarray(r["dphi_goal_arr"]) for r in rot_rows if len(r["dphi_goal_arr"])]
    n = len(all_dphi_goal)
    if n > 0:
        boot = np.array([np.angle(np.mean(np.exp(1j * np.concatenate(
                             [per_net[k] for k in rng.integers(0, len(per_net), len(per_net))]))))
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

    fig, axes = plt.subplots(1, 2, figsize=size_in or (4.6, 2.4))
    axes[0].remove()
    axp = fig.add_subplot(1, 2, 1, projection="polar")
    bins = np.linspace(-np.pi, np.pi, 25)
    counts, edges = np.histogram(all_dphi_goal, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    axp.bar(centers, counts, width=width, color=COLOR_GOALDIR, alpha=0.75, edgecolor="0.3", linewidth=0.3)
    axp.set_theta_zero_location("N")
    # Two radial ticks, off the 0-degree spoke: the default set stacked its
    # counts down the vertical axis, across the rose itself.
    _top = float(counts.max()) if len(counts) else 1.0
    axp.set_rgrids([_top / 2, _top], labels=[f"{_top/2:.0f}", f"{_top:.0f}"],
                   angle=145, fontsize=FS_SMALL)
    axp.set_title(f"Bearing change, pooled\n({len(all_dphi_goal)} cells, {len(rot_rows)} networks)", fontsize=FS_TICK)

    ax = axes[1]
    rng2 = np.random.default_rng(2)
    jitter = rng2.uniform(-0.08, 0.08, len(circ_means))
    ax.scatter(jitter, np.degrees(circ_means), s=14, color=COLOR_GOALDIR, alpha=0.7, zorder=3, linewidths=0, label="Δφ_ego (per-seed)")
    ax.scatter(1 + jitter, np.degrees(delta_geoms), s=14, color="0.5", alpha=0.7, zorder=3, linewidths=0, marker="s", label="Δ_geom (per-seed)")
    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.errorbar([0], [np.degrees(pooled_mean)], yerr=[[np.degrees(pooled_mean - lo)], [np.degrees(hi - pooled_mean)]],
                fmt="D", color="black", markersize=5, capsize=3, lw=1.2, zorder=6,
                label="pooled mean ± 95% CI")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Observed", "Allocentric\nprediction"], fontsize=FS_SMALL)
    ax.set_ylabel("Bearing shift A→B (deg)")
    title(ax, "Per network")
    # Headroom first, then the statistics in the strip it opens up. Set as a
    # title this block ran 30 pt off the right edge; dropped straight into the
    # axes it landed on the Δ_geom column.
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y + 0.55 * (hi_y - lo_y))
    # Three short lines: the subplot is ~1.6 in, so anything past ~26
    # characters runs off it. The black pooled-mean marker is named in the
    # caption rather than here.
    stat_annot(ax,
               f"pooled {np.degrees(pooled_mean):+.1f}° "
               f"[{np.degrees(lo):+.1f}, {np.degrees(hi):+.1f}]\n"
               f"|obs.| < |pred.| in {n_favorable}/{len(circ_means)}\n"
               f"Wilcoxon P {_fmt_p(p_mag)}",
               loc="upper left")
    # No legend: the x tick labels already name the two columns, and at three
    # entries it sat on the observed-shift cloud.

    fig.suptitle(suptitle_text, fontsize=FS_BASE, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4, w_pad=1.0)
    return save_figure(fig, stem, exact_size=size_in is not None)


# ---------------------------------------------------------------------------
# Panel c — replay versus goal-directed negative control
# ---------------------------------------------------------------------------

def fig_frozen_replay_control(rows_nh_replay, rows_hd_replay, rows_nh_goal, rows_hd_goal, stem, size_in=None):
    """Reproduces the legacy README's null result: ramping/MRL on the single
    frozen-replay trajectory (replay_c1_pred/postrl) are near-zero/non-
    significant, contrasting with real tuning on goal_conditioned --
    justifying why tuning curves are built from goal-directed periods."""
    fig, axes = plt.subplots(1, 2, figsize=size_in or (4.4, 2.2))

    ax = axes[0]
    labels = ["Replay", "Goal-\ndirected", "Replay", "Goal-\ndirected"]
    data = [
        np.array([r["frac_distance_cell"] for r in rows_nh_replay]) * 100,
        np.array([r["frac_distance_cell"] for r in rows_nh_goal]) * 100,
        np.array([r["frac_distance_cell"] for r in rows_hd_replay]) * 100,
        np.array([r["frac_distance_cell"] for r in rows_hd_goal]) * 100,
    ]
    colors = [COLOR_NOHEAD, COLOR_NOHEAD, COLOR_HEAD, COLOR_HEAD]
    xs = np.arange(4)
    means = [d.mean() if len(d) else 0 for d in data]
    sds = [sem(d) if len(d) > 1 else 0 for d in data]
    ax.bar(xs, means, yerr=sds, color=colors, capsize=3, width=0.6)
    for i, c in enumerate(colors):
        ax.patches[i].set_alpha(0.4 if i % 2 == 0 else 0.9)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=FS_SMALL)
    ax.set_ylabel("Distance cells (%)")
    ax.set_title("Distance cells", fontsize=FS_TICK)

    ax = axes[1]
    data2 = [
        np.array([r["mean_mrl_ego"] for r in rows_nh_replay]),
        np.array([r["mean_mrl_ego"] for r in rows_nh_goal]),
        np.array([r["mean_mrl_ego"] for r in rows_hd_replay]),
        np.array([r["mean_mrl_ego"] for r in rows_hd_goal]),
    ]
    means2 = [d.mean() if len(d) else 0 for d in data2]
    sds2 = [sem(d) if len(d) > 1 else 0 for d in data2]
    ax.bar(xs, means2, yerr=sds2, color=colors, capsize=3, width=0.6)
    for i in range(4):
        ax.patches[i].set_alpha(0.4 if i % 2 == 0 else 0.9)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=FS_SMALL)
    ax.set_ylabel("Mean resultant length")
    ax.set_title("Goal-bearing tuning", fontsize=FS_TICK)

    fig.suptitle("Replayed random walk versus goal-directed trajectories", fontsize=FS_BASE, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4, w_pad=0.8)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    """Command-line arguments: condition directories, seed list, output directory, panel subset."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nohead_seeds", type=int, nargs="+", default=None)
    p.add_argument("--head_seeds", type=int, nargs="+", default=None)
    p.add_argument("--reloc_seeds", type=int, nargs="+", default=None)
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    p.add_argument("--assemble_only", action="store_true",
                   help="recompose from the existing panel PDFs; a full rebuild "
                        "of this figure reads every cohort's pickles and takes ~2 h")
    p.add_argument("--rotation-json", action="store_true",
                   help="draw panels b and c from the saved rotation summaries, "
                        "restricted to the relocation campaign passers")
    add_plot_only(p)
    args = p.parse_args()
    args.panels = {x.strip() for x in args.panels.split(",") if x.strip()}
    return args


def main():
    """Load or rebuild the per-condition summaries, build the panels, and assemble the figure."""
    args = parse_args()

    if args.assemble_only:
        print("Assembly-only mode: reusing existing panel PDFs...")
        panels_dir = Path(args.output_dir) / "panels"
        assemble_supp(figlayout.existing_panels(panels_dir, LAYOUT.keys()),
                      Path(args.output_dir) / "figure_reward_vector_supp_raw")
        return
    panels_dir = ensure_dir(Path(args.output_dir) / "panels")
    ensure_dir(Path(args.output_dir))
    panel_pdfs: dict[str, Path] = {}

    cache_path = Path(args.output_dir) / "figure_data.npz"
    fresh = {}
    saved = {}
    if args.plot_only and ({"a", "b", "c"} & args.panels):
        saved = load_figure_data(cache_path)

    def _take(*keys):
        return require_keys(saved, *keys, path=cache_path)

    if args.rotation_json and (args.panels - {"b", "c"}):
        raise SystemExit("--rotation-json only draws panels b and c")

    if not args.plot_only and not args.rotation_json:
        nohead_seeds = args.nohead_seeds if args.nohead_seeds is not None else discover_seeds(NOHEAD_DIR, "nohead_")
        head_seeds = args.head_seeds if args.head_seeds is not None else discover_seeds(HEAD_DIR, "head_")
        reloc_seeds = args.reloc_seeds if args.reloc_seeds is not None else discover_seeds(RELOC_DIR, "reloc_")
        print(f"Gating {len(nohead_seeds)} no-head, {len(head_seeds)} head-ON, {len(reloc_seeds)} reloc seeds...")
        # the same retained cohorts as every other analysis in the paper
        paired = campaign_passers("paired")
        nh_clean = sorted(s for s in nohead_seeds if s in paired)
        hd_clean = sorted(s for s in head_seeds if s in paired)
        reloc_clean = sorted(s for s in reloc_seeds if s in campaign_passers("reloc"))
        print(f"  clean: {len(nh_clean)} no-head, {len(hd_clean)} head-ON, {len(reloc_clean)} reloc")

    rows_nh = rows_hd = None
    if "a" in args.panels:
        if args.plot_only:
            got = _take("rows_nh", "rows_hd")
            rows_nh, rows_hd = got["rows_nh"], got["rows_hd"]
            print(f"Plot-only: {len(rows_nh)} no-head, {len(rows_hd)} head-ON")
        else:
            print("Loading no-head population summaries (goal_conditioned vs before)...")
            rows_nh = load_condition_population_summary(NOHEAD_DIR, "nohead_", nh_clean)
            save_summary_cache(NOHEAD_DIR, rows_nh)
            print("Loading head-ON population summaries...")
            rows_hd = load_condition_population_summary(HEAD_DIR, "head_", hd_clean)
            save_summary_cache(HEAD_DIR, rows_hd)
            fresh["rows_nh"] = rows_nh
            fresh["rows_hd"] = rows_hd

    def _keep_reloc(rows):
        keep = campaign_passers("reloc")
        return [r for r in rows if int(r.get("seed", -1)) in keep]

    if "b" in args.panels:
        if args.rotation_json:
            rot_c = json.loads((REPO / "results/multi_seed_reloc/aggregate/reward_vector_rotation_behavioral.json").read_text())
        elif args.plot_only:
            rot_c = _take("rot_c")["rot_c"]
        else:
            print("Panel b data (relocation rotation, behavioral)...")
            rot_c = load_reloc_rotation_summary(RELOC_DIR, reloc_clean, "cycle_1_goal", "cycle_2_goal")
            save_summary_cache(RELOC_DIR, rot_c, name="reward_vector_rotation_behavioral.json")
            fresh["rot_c"] = rot_c
        rot_c = _keep_reloc(rot_c)

    if "c" in args.panels:
        if args.rotation_json:
            rot_d = json.loads((REPO / "results/multi_seed_reloc/aggregate/reward_vector_rotation_occupancy_controlled.json").read_text())
        elif args.plot_only:
            rot_d = _take("rot_d")["rot_d"]
        else:
            print("Panel c data (relocation rotation, occupancy-controlled)...")
            rot_d = load_reloc_rotation_summary(RELOC_DIR, reloc_clean, "replay_reloc_postA", "replay_reloc_postB")
            save_summary_cache(RELOC_DIR, rot_d, name="reward_vector_rotation_occupancy_controlled.json")
            fresh["rot_d"] = rot_d
        rot_d = _keep_reloc(rot_d)

    if "a" in args.panels:
        if args.plot_only:
            got = _take("rows_nh_replay", "rows_hd_replay")
            rows_nh_replay, rows_hd_replay = got["rows_nh_replay"], got["rows_hd_replay"]
        else:
            print("Panel a data (frozen-replay negative control)...")
            rows_nh_replay = load_condition_population_summary(
                NOHEAD_DIR, "nohead_", nh_clean,
                period="replay_c1_postrl", baseline_period="replay_c1_pred")
            rows_hd_replay = load_condition_population_summary(
                HEAD_DIR, "head_", hd_clean,
                period="replay_c1_postrl", baseline_period="replay_c1_pred")
            fresh["rows_nh_replay"] = rows_nh_replay
            fresh["rows_hd_replay"] = rows_hd_replay

    if fresh:
        update_figure_data(cache_path, fresh)

    if "a" in args.panels:
        print("Panel a (replay versus goal-directed trajectories)...")
        _, pdf = fig_frozen_replay_control(
            rows_nh_replay, rows_hd_replay, rows_nh, rows_hd,
            panels_dir / "panel_a", LAYOUT.slot_size_in("a"))
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Panel b (relocation rotation test, behavioral version)...")
        _, pdf = fig_rotation_test(rot_c, "Relocation, goal-directed trajectories",
                                    panels_dir / "panel_b",
                                    LAYOUT.slot_size_in("b"))
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Panel c (relocation rotation test, occupancy-controlled version)...")
        _, pdf = fig_rotation_test(rot_d, "Relocation, replayed trajectory",
                                    panels_dir / "panel_c",
                                    LAYOUT.slot_size_in("c"))
        panel_pdfs["c"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("\nAssembling supplementary figure PDF...")
        assemble_supp(panel_pdfs, Path(args.output_dir) / "figure_reward_vector_supp_raw")

    print(f"\nDone. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
