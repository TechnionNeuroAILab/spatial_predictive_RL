"""
DEPRECATED — DO NOT USE. Retained only for provenance; not imported anywhere.

This module's headline metric is BIASED and its published output was wrong.
`_seed_metrics` projected each cell's displacement onto a toward-reward unit
vector computed from that cell's PRE position:

    proj_i = (post_i - pre_i) . normalise(REWARD - pre_i)

With an off-centre reward, generic drift toward the middle of the arena scores
as "toward reward" for every cell on the far side, so the statistic measures
regression to the centre rather than remapping. A shuffled-cell null returns
+0.42 against an observed +0.22/+0.25 — i.e. the null EXCEEDS the observation.
Because the bias is condition-independent, the panel reported head-ON and
no-head as indistinguishable (0.2554 vs 0.2556, p = 0.83), erasing the effect
that the correct metric shows at p = 0.0001.

Replacement: code_analysis/figure_sr/ (uses change-in-distance-to-reward, with
both a reward-location null for the mean and a cell-shuffle null for the paired
statistics). Shared constants moved to New_Code/sr_common.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "New_Code"))
sys.path.insert(0, str(REPO / "code_analysis" / "figure 5"))

from analysis_helpers import load_period_centers          # noqa: E402
from successor_representation import (                    # noqa: E402
    build_T_random,
    transition_from_behavior_positions,
    compute_sr_matrix,
    sr_predicted_shift_vectors,
    state_centers,
    nearest_state_for_points,
)

# ── constants ─────────────────────────────────────────────────────────────────
M, WB      = 32, 1           # grid size, wall buffer (matches simulator)
GAMMA      = 0.99
SKAGGS_THR = 0.3
REWARD     = np.array([0.3, 0.3])
GOAL_RADIUS = 0.15           # radius for ORI computation

COLOR_SR     = "#888888"
COLOR_NOHEAD = "#4c72b0"
COLOR_HEAD   = "#8B4513"


def _apply_nature_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.75,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ── SR computation (shared random matrix) ─────────────────────────────────────

_T_RANDOM  = None
_SR_RANDOM = None
_STATE_XY  = None


def _get_shared_random():
    global _T_RANDOM, _SR_RANDOM, _STATE_XY
    if _T_RANDOM is None:
        _T_RANDOM = build_T_random(M, WB)
        _SR_RANDOM, _, _ = compute_sr_matrix(_T_RANDOM, GAMMA, "auto", 500, 1e10)
        _STATE_XY = state_centers(M, WB)
    return _T_RANDOM, _SR_RANDOM, _STATE_XY


# ── ORI helper ────────────────────────────────────────────────────────────────

def _compute_ori(peaks: np.ndarray, reward: np.ndarray = REWARD,
                 r: float = GOAL_RADIUS) -> float:
    """Over-Representation Index: density in goal zone / density in rest of arena."""
    dists  = np.linalg.norm(peaks - reward, axis=1)
    in_z   = (dists <= r).sum()
    out_z  = (dists  > r).sum()
    n      = len(peaks)
    if out_z == 0 or n == 0:
        return np.nan
    area_goal = np.pi * r ** 2
    area_rest = 1.0 - area_goal          # unit arena
    # density ratio (cells per unit area)
    d_in  = in_z  / area_goal
    d_out = out_z / area_rest
    return d_in / d_out if d_out > 0 else np.nan


# ── per-seed analysis ─────────────────────────────────────────────────────────

def _seed_metrics(data_dir: Path) -> dict | None:
    """
    Return per-seed scalar metrics for the frozen-replay diagnostic, or None on failure.

    Comparison: replay_c1_pred (pre-RL weights, fixed traj) vs
                replay_c1_postrl (post-RL weights, same fixed traj)

    SR predicts ZERO shift for this comparison (trajectory is fixed → occupancy
    unchanged → no SR-predicted shift).  Any measured shift is purely weight-driven.

    Convention: positive = field moved TOWARD reward.
    """
    _, SR_ran, state_xy = _get_shared_random()

    # --- field centers: A = replay_c1_pred (pre-RL weights, frozen traj)
    #                    B = replay_c1_postrl (post-RL weights, SAME frozen traj)
    # Trajectory is identical → any shift is purely weight-driven (H2 signal).
    # SR predicts zero shift here (occupancy fixed), so model > 0 ⇒ model > SR.
    try:
        sk_A, _, pk_A = load_period_centers(str(data_dir), "replay_c1_pred")
        sk_B, _, pk_B = load_period_centers(str(data_dir), "replay_c1_postrl")
    except Exception:
        return None

    gate = (sk_A >= SKAGGS_THR) & (sk_B >= SKAGGS_THR)
    if gate.sum() < 10:
        return None

    pk_A_g = pk_A[gate]
    pk_B_g = pk_B[gate]
    actual = pk_B_g - pk_A_g                         # (N, 2); shift vector

    # SR prediction for frozen replay = 0 (trajectory fixed, no occupancy change).
    # We set sr_pred to zero and sr_tr to 0 throughout.
    sr_pred = np.zeros_like(actual)                   # (N, 2); SR baseline

    # unit vectors pointing FROM each cell's starting position TOWARD reward
    rew_dir = REWARD - pk_A_g                         # (N, 2)
    rew_hat = rew_dir / np.maximum(
        np.linalg.norm(rew_dir, axis=1, keepdims=True), 1e-10
    )

    # projections onto toward-reward axis
    # positive  = the shift aligns with the toward-reward direction
    # negative  = the shift is away from reward
    actual_tr = np.sum(actual   * rew_hat, axis=1)    # (N,)
    sr_tr     = np.zeros(len(actual_tr))               # SR = 0 for frozen replay

    # ORI: field-center density ratio (post vs pre in goal zone)
    ori_pre  = _compute_ori(pk_A_g)
    ori_post = _compute_ori(pk_B_g)
    ori_delta = ori_post - ori_pre

    return {
        "n":            int(gate.sum()),
        "actual_mean":  float(actual_tr.mean()),     # positive = toward reward
        "sr_mean":      float(sr_tr.mean()),          # positive = toward reward
        "excess":       float(actual_tr.mean() - sr_tr.mean()),
        "ori_pre":      float(ori_pre),
        "ori_post":     float(ori_post),
        "ori_delta":    float(ori_delta),
        "actual_cells": actual_tr,
        "sr_cells":     sr_tr,
    }


def load_all_seeds(nohead_dir: Path, head_dir: Path, seeds: list[int]) -> list[dict]:
    """Return list of per-seed records with head + nohead metrics."""
    records = []
    for seed in seeds:
        nh = _seed_metrics(nohead_dir / f"nohead_seed{seed}" / "data")
        hd = _seed_metrics(head_dir   / f"head_seed{seed}"   / "data")
        if nh is None:
            print(f"  seed {seed}: nohead data missing — skip")
            continue
        if hd is None:
            print(f"  seed {seed}: head data missing — skip")
            continue
        records.append({"seed": seed, "nohead": nh, "head": hd})
        print(
            f"  seed {seed:2d}  "
            f"SR={nh['sr_mean']:+.3f}  "
            f"nohead={nh['actual_mean']:+.3f}  "
            f"head={hd['actual_mean']:+.3f}  "
            f"excess_head={hd['excess']:+.3f}  "
            f"ORI_nh Δ={nh['ori_delta']:+.2f}  ORI_hd Δ={hd['ori_delta']:+.2f}"
        )
    return records


# ── plotting helpers ──────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "p=n/a"
    if p < 0.001:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _sig_bracket(ax, x0, x1, y, text, dy=0.005):
    ax.plot([x0, x0, x1, x1], [y, y + dy, y + dy, y], lw=0.8, color="0.3")
    ax.text((x0 + x1) / 2, y + dy + 0.001, text, ha="center", va="bottom",
            fontsize=5.5, color="0.3")


# ── main figure ───────────────────────────────────────────────────────────────

def fig_panel(records: list[dict], out_stem: Path) -> tuple[Path, Path]:
    """
    3-subplot panel:
      a) Bar chart: mean toward-reward shift for SR / nohead / head
         (positive = toward reward)
      b) Per-seed paired scatter: SR predicted vs actual (nohead & head)
      c) Excess beyond SR per condition (strip + mean ± SD)
    """
    n = len(records)

    # --- per-seed arrays (positive = toward reward) ---
    # SR = 0 for frozen replay (trajectory fixed, no occupancy change)
    sr_means  = np.zeros(n)
    nh_means  = np.array([r["nohead"]["actual_mean"] for r in records])
    hd_means  = np.array([r["head"]["actual_mean"]   for r in records])
    # excess = actual - SR = actual - 0 = actual
    nh_excess = nh_means
    hd_excess = hd_means

    # --- stats ---
    _, p_sr_nh     = wilcoxon(sr_means,  nh_means)
    _, p_sr_hd     = wilcoxon(sr_means,  hd_means)
    _, p_nh_hd     = wilcoxon(nh_means,  hd_means)
    _, p_exc_nh    = wilcoxon(nh_excess, np.zeros(n))
    _, p_exc_hd    = wilcoxon(hd_excess, np.zeros(n))
    _, p_exc_nh_hd = wilcoxon(nh_excess, hd_excess)

    _apply_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))

    # ── subplot a: bar chart ──
    ax = axes[0]
    labels = ["SR\n(= 0, fixed traj)", "No-head\nmodel", "Head-ON\nmodel"]
    means  = [sr_means.mean(), nh_means.mean(), hd_means.mean()]
    sds    = [sr_means.std(ddof=1), nh_means.std(ddof=1), hd_means.std(ddof=1)]
    colors = [COLOR_SR, COLOR_NOHEAD, COLOR_HEAD]
    xs     = np.arange(3)

    rng = np.random.default_rng(42)
    per_seed_vals = [sr_means, nh_means, hd_means]
    for xi, (m, s, col, vals) in enumerate(zip(means, sds, colors, per_seed_vals)):
        ax.bar(xi, m, yerr=s, color=col, alpha=0.80, capsize=4, width=0.6,
               error_kw={"lw": 1.0, "ecolor": "0.3"},
               edgecolor="0.25", linewidth=0.5)
        jitter = rng.uniform(-0.22, 0.22, len(vals))
        ax.scatter(xi + jitter, vals, s=8, color=col, alpha=0.55, zorder=4, linewidths=0)

    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Mean Δd frozen replay (pre→post weights)\n(positive = toward reward; SR = 0)")
    ax.set_title(f"Head-ON exceeds SR (=0) in frozen replay\n(n={n} seeds)", fontsize=6.5)

    # significance brackets above the bars
    y_ceil = max(m + s for m, s in zip(means, sds)) + 0.01
    _sig_bracket(ax, 0, 1, y_ceil,        _fmt_p(p_sr_nh), dy=0.006)
    _sig_bracket(ax, 0, 2, y_ceil + 0.04, _fmt_p(p_sr_hd), dy=0.006)

    # ── subplot b: per-seed paired strip for both conditions ──
    # SR = 0 for frozen replay (trajectory fixed); positive = toward reward
    ax = axes[1]
    rng_b = np.random.default_rng(99)
    for nh_v, hd_v in zip(nh_means, hd_means):
        ax.plot([0, 1], [nh_v, hd_v], color="0.80", lw=0.6, zorder=1)
    ax.scatter(rng_b.uniform(-0.15, 0.15, n), nh_means, s=22, color=COLOR_NOHEAD,
               zorder=4, edgecolors="white", linewidths=0.4, label="No-head")
    ax.scatter(1 + rng_b.uniform(-0.15, 0.15, n), hd_means, s=22, color=COLOR_HEAD,
               zorder=4, edgecolors="white", linewidths=0.4, label="Head ON", marker="D")
    ax.errorbar([0], [nh_means.mean()], yerr=[nh_means.std(ddof=1)],
                fmt="D", color=COLOR_NOHEAD, capsize=4, ms=6, lw=1.2, zorder=5)
    ax.errorbar([1], [hd_means.mean()], yerr=[hd_means.std(ddof=1)],
                fmt="D", color=COLOR_HEAD, capsize=4, ms=6, lw=1.2, zorder=5)
    ax.axhline(0, color="0.4", lw=0.9, ls="--", label="SR = 0", zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No-head", "Head ON"])
    ax.set_ylabel("Frozen-replay shift (positive = toward reward)")
    ax.set_title("Per-seed frozen-replay shift\n(dashed = SR baseline = 0)", fontsize=6.5)
    ax.legend(fontsize=5.5, frameon=False)

    # ── subplot c: excess beyond SR ──
    ax = axes[2]
    rng2 = np.random.default_rng(7)
    for ne, he in zip(nh_excess, hd_excess):
        ax.plot([1, 2], [ne, he], color="0.80", lw=0.6, zorder=1)
    ax.scatter(rng2.uniform(0.82, 1.18, n), nh_excess,
               s=16, color=COLOR_NOHEAD, zorder=4, linewidths=0, alpha=0.8)
    ax.scatter(rng2.uniform(1.82, 2.18, n), hd_excess,
               s=16, color=COLOR_HEAD,   zorder=4, linewidths=0, alpha=0.8, marker="D")
    ax.errorbar([1], [nh_excess.mean()], yerr=[nh_excess.std(ddof=1)],
                fmt="D", color=COLOR_NOHEAD, capsize=4, ms=6, lw=1.2, zorder=5)
    ax.errorbar([2], [hd_excess.mean()], yerr=[hd_excess.std(ddof=1)],
                fmt="D", color=COLOR_HEAD,   capsize=4, ms=6, lw=1.2, zorder=5)
    ax.axhline(0, color="0.4", lw=0.7, ls="--", label="at SR level")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["No-head", "Head ON"])
    ax.set_ylabel("Excess shift toward reward\nbeyond SR prediction")
    ax.set_title(
        f"Head-ON > SR (=0); No-head ≈ SR\n"
        f"nh {_fmt_p(p_exc_nh)}  hd {_fmt_p(p_exc_hd)}  "
        f"nh vs hd {_fmt_p(p_exc_nh_hd)}",
        fontsize=5.8,
    )

    fig.suptitle(
        f"Frozen-replay diagnostic: Head-ON shows H2 signal that SR cannot explain  "
        f"({n} seeds, Skaggs≥{SKAGGS_THR}; SR = 0 for fixed trajectory)",
        fontsize=6.5, y=1.01,
    )
    fig.tight_layout(pad=0.4, w_pad=1.2)

    stem = Path(out_stem).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {png}")
    print(f"  Saved: {pdf}")

    # ── print and save JSON summary ──
    summary = {
        "n_seeds": n,
        "gamma": GAMMA,
        "skaggs_threshold": SKAGGS_THR,
        "goal_radius_ori": GOAL_RADIUS,
        "comparison": "frozen-replay (pre-RL weights → post-RL weights, fixed trajectory)",
        "sr_prediction": "0 (occupancy unchanged → SR predicts no shift)",
        "field_center_pre": "replay_c1_pred",
        "field_center_post": "replay_c1_postrl",
        "sign_convention": "positive = toward reward",
        "sr": {
            "mean": float(sr_means.mean()),
            "sd":   float(sr_means.std(ddof=1)),
            "per_seed": sr_means.tolist(),
        },
        "nohead": {
            "mean":   float(nh_means.mean()),
            "sd":     float(nh_means.std(ddof=1)),
            "p_vs_sr": float(p_sr_nh),
            "excess_mean": float(nh_excess.mean()),
            "excess_sd":   float(nh_excess.std(ddof=1)),
            "p_excess_vs_0": float(p_exc_nh),
            "per_seed": nh_means.tolist(),
            "ori_delta_per_seed": [r["nohead"]["ori_delta"] for r in records],
        },
        "head": {
            "mean":   float(hd_means.mean()),
            "sd":     float(hd_means.std(ddof=1)),
            "p_vs_sr": float(p_sr_hd),
            "excess_mean": float(hd_excess.mean()),
            "excess_sd":   float(hd_excess.std(ddof=1)),
            "p_excess_vs_0": float(p_exc_hd),
            "per_seed": hd_means.tolist(),
            "ori_delta_per_seed": [r["head"]["ori_delta"] for r in records],
        },
        "head_vs_nohead": {"p": float(p_nh_hd)},
        "excess_head_vs_nohead": {"p": float(p_exc_nh_hd)},
    }
    json_path = png.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {json_path}")

    print(f"\n  SR mean:     {sr_means.mean():+.4f} ± {sr_means.std(ddof=1):.4f}")
    print(f"  Nohead mean: {nh_means.mean():+.4f} ± {nh_means.std(ddof=1):.4f}  vs SR: {_fmt_p(p_sr_nh)}")
    print(f"  Head mean:   {hd_means.mean():+.4f} ± {hd_means.std(ddof=1):.4f}  vs SR: {_fmt_p(p_sr_hd)}")
    print(f"  Head vs nohead: {_fmt_p(p_nh_hd)}")
    print(f"  Excess nohead: {nh_excess.mean():+.4f}  head: {hd_excess.mean():+.4f}  diff: {_fmt_p(p_exc_nh_hd)}")

    return png, pdf


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nohead-dir", default=str(REPO / "Repro" / "multi_seed"))
    p.add_argument("--head-dir",   default=str(REPO / "Repro" / "multi_seed_head"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--output-dir", default=str(REPO / "Analysis_Plots" / "sr_vs_model"))
    return p.parse_args()


def main():
    _apply_nature_style()
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(args.seeds)} seeds …")
    records = load_all_seeds(
        Path(args.nohead_dir), Path(args.head_dir), args.seeds
    )

    if not records:
        print("No complete seed pairs found.")
        return

    print(f"\n{len(records)} seed pairs loaded. Generating figure …")
    fig_panel(records, out_dir / "sr_vs_model_panel")
    print("\nDone.")


if __name__ == "__main__":
    main()
