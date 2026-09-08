"""
Figure 4 analysis helpers — reward relocation experiment.

Loads the three frozen-replay periods (baseline, postA, postB), computes
field-center distances to A and B, and returns tracking statistics.

All heavy lifting is done by New_Code/analysis_helpers.py (load_period_centers)
and scipy. No duplication of rate-map code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_NEW_CODE = Path(__file__).parents[2] / "New_Code"
sys.path.insert(0, str(_NEW_CODE))

from analysis_helpers import load_period_centers, has_period, seed_is_clean  # noqa: E402
from plot_goal_directed_maps import build_rate_maps, compute_skaggs, smooth_and_norm  # noqa: E402
from plot_replay_v11 import field_peak  # noqa: E402

from utils import (  # noqa: E402
    PERIOD_PRE, PERIOD_POST_A, PERIOD_POST_B, PERIODS,
    REWARD_A, REWARD_B, SKAGGS_THRESHOLD, MIN_ENTROPY,
    COLOR_A, COLOR_B, PERIOD_COLORS, M, HIDDEN_DIM,
    save_figure,
)


def seed_entropy_status(data_dir, periods=(PERIOD_PRE, PERIOD_POST_A, PERIOD_POST_B),
                         min_entropy: float = MIN_ENTROPY) -> bool:
    """Whether a relocation seed passes the entropy-collapse gate (same
    diagnostic used by Figure 5) across the given periods. A seed whose
    post-relocation field distribution has collapsed into a tiny region
    gives an artificially high tracking score without genuine tracking."""
    return seed_is_clean(data_dir, list(periods), min_entropy)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reloc_states(data_dir, skaggs_threshold: float = SKAGGS_THRESHOLD) -> dict:
    """Load all three periods and return per-period dicts with sk, peak, com."""
    data_dir = str(data_dir)
    states = {}
    for period, _ in PERIODS:
        if not has_period(data_dir, period):
            raise FileNotFoundError(
                f"Missing relocation period '{period}' in {data_dir}. "
                "Did the run use --relocate_at_cycle?"
            )
        sk, com, peak = load_period_centers(data_dir, period)
        states[period] = {"sk": sk, "peak": peak, "com": com}
    return states


def _wilcoxon_p(a, b):
    if len(a) < 5 or np.all(np.abs(a - b) < 1e-12):
        return float("nan")
    try:
        return float(wilcoxon(a, b)[1])
    except Exception:
        return float("nan")


def paired_tracking_stats(states: dict, skaggs_threshold: float = SKAGGS_THRESHOLD) -> dict:
    """
    Compute tracking statistics:
      - mean distance to A and B in each period
      - shift toward A (pre→postA) and toward B (postA→postB)
    """
    pre, pA, pB = PERIOD_PRE, PERIOD_POST_A, PERIOD_POST_B

    def paired_shift(p_from, p_to, reward):
        sk_f = states[p_from]["sk"]
        sk_t = states[p_to]["sk"]
        active = np.where(np.maximum(sk_f, sk_t) >= skaggs_threshold)[0]
        d_f = np.linalg.norm(states[p_from]["peak"][active] - reward[None, :], axis=1)
        d_t = np.linalg.norm(states[p_to]["peak"][active] - reward[None, :], axis=1)
        d = d_t - d_f
        n = int(len(active))
        return {
            "delta": d,
            "n": n,
            "mean_delta": float(np.mean(d)) if n > 0 else float("nan"),
            "frac_toward": float(np.mean(d < 0)) if n > 0 else float("nan"),
            "wilcoxon_p": _wilcoxon_p(d_t, d_f),
        }

    # Mean distances per period
    dists_A, dists_B = [], []
    for period, _ in PERIODS:
        sk = states[period]["sk"]
        pk = states[period]["peak"]
        place = np.where(sk >= skaggs_threshold)[0]
        dA = np.linalg.norm(pk[place] - REWARD_A[None, :], axis=1)
        dB = np.linalg.norm(pk[place] - REWARD_B[None, :], axis=1)
        dists_A.append(dA)
        dists_B.append(dB)

    shift_A = paired_shift(pre, pA, REWARD_A)
    shift_B = paired_shift(pA, pB, REWARD_B)

    return {
        "dists_A": dists_A,   # list of 3 arrays (one per period)
        "dists_B": dists_B,
        "shift_A": shift_A,   # pre → postA toward A
        "shift_B": shift_B,   # postA → postB toward B
        "n_per_period": [len(d) for d in dists_A],
    }


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def fig_tracking_curves(ts: dict, stem) -> tuple:
    """Panel c — mean distance to A and B across three periods (crossing curves)."""
    labels = [lab.replace("\n", " ") for _, lab in PERIODS]
    x = np.arange(len(PERIODS))
    mA = [d.mean() for d in ts["dists_A"]]
    sA = [d.std(ddof=1) / np.sqrt(len(d)) for d in ts["dists_A"]]
    mB = [d.mean() for d in ts["dists_B"]]
    sB = [d.std(ddof=1) / np.sqrt(len(d)) for d in ts["dists_B"]]

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.errorbar(x, mA, yerr=sA, fmt="-o", lw=1.8, color=COLOR_A, capsize=3,
                label=f"dist to A ({REWARD_A[0]},{REWARD_A[1]})")
    ax.errorbar(x, mB, yerr=sB, fmt="-s", lw=1.8, color=COLOR_B, capsize=3,
                label=f"dist to B ({REWARD_B[0]},{REWARD_B[1]})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean field-centre distance (±SEM)")
    ax.set_title("Fields leave A and arrive at B\n(frozen trajectory; occupancy fixed)", fontsize=6, pad=2)
    ax.legend(fontsize=5)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_shift_violins(ts: dict, stem) -> tuple:
    """Panel d — Δd violin: toward A and toward B."""
    from utils import apply_nature_style
    apply_nature_style()

    sA = ts["shift_A"]
    sB = ts["shift_B"]

    def _fmt_p(p):
        if not np.isfinite(p):
            return ""
        return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"

    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    data = [sA["delta"], sB["delta"]]
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=False, showextrema=True)
    colors = [COLOR_A, COLOR_B]
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.6)
        body.set_edgecolor("0.2")
    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        if key in parts:
            parts[key].set_color("0.25")
            parts[key].set_linewidth(0.8)

    ax.axhline(0, color="0.35", lw=0.7, ls="--")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"pre→postA\n(toward A)\n{_fmt_p(sA['wilcoxon_p'])}",
        f"postA→postB\n(toward B)\n{_fmt_p(sB['wilcoxon_p'])}",
    ])
    ax.set_ylabel(r"$\Delta$ field-centre dist to reward")
    ax.set_title(
        f"Shift toward A: {sA['mean_delta']:+.3f} ({100*sA['frac_toward']:.0f}%)\n"
        f"Shift toward B: {sB['mean_delta']:+.3f} ({100*sB['frac_toward']:.0f}%)",
        fontsize=5.5, pad=2,
    )
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_com_scatter_three(states: dict, stem, reward_a=REWARD_A, reward_b=REWARD_B,
                          skaggs_threshold: float = SKAGGS_THRESHOLD) -> tuple:
    """Panel b — COM field-centre clouds for three periods, coloured by period.

    Same graphic as Figure 3 panel b (fig_centers_abc): centre-of-mass dots,
    s=3, alpha=0.35, no edge, rasterized — not grid-snapped field peaks.
    """
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.85))
    for ax, (period, label) in zip(axes, PERIODS):
        sk = states[period]["sk"]
        com = states[period]["com"]
        place = np.where(sk >= skaggs_threshold)[0]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        ax.scatter(com[place, 0], com[place, 1],
                   s=3, alpha=0.35, c=PERIOD_COLORS[period], linewidths=0, rasterized=True)
        ax.plot(*reward_a, "*", color=COLOR_A, markersize=7, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)
        ax.plot(*reward_b, "*", color=COLOR_B, markersize=7, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)
        ax.set_title(f"{label.replace(chr(10), ' ')}  (n={len(place)})", fontsize=6, pad=2)
    fig.suptitle("Field centres (COM) across phases", fontsize=6.5, y=1.02)
    fig.tight_layout(pad=0.35, w_pad=0.35)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel a — example tracking cells (local implementation)
#
# Replaces the New_Code/reward_relocation_maps.py exemplar panel, whose
# quality score REWARDED field sharpness (quality = minsk + 0.7*sharp - ...),
# so needle-thin "speckle" fields won even when they did not track the reward.
# Here exemplars must (i) pass the Skaggs gate in ALL three periods,
# (ii) have a minimum smoothed-field area (anti-speckle), and (iii) genuinely
# track: move toward A in phase A and toward B in phase B, ending near each.
# ---------------------------------------------------------------------------

def _load_rate_maps(data_dir, period, m: int = M, hidden_dim: int = HIDDEN_DIM):
    beh = pd.read_pickle(os.path.join(str(data_dir), f"df_behavior_{period}.pkl"))
    act = pd.read_pickle(os.path.join(str(data_dir), f"df_activations_{period}.pkl"))
    rm, occ = build_rate_maps(beh, act, m, hidden_dim)
    sk = compute_skaggs(rm, occ)
    return rm, sk


def select_example_cells(sm: dict, sks: dict, n_examples: int,
                         skaggs_threshold: float = SKAGGS_THRESHOLD,
                         reward_a=REWARD_A, reward_b=REWARD_B,
                         min_field_area: float = 0.01) -> np.ndarray:
    """Pick exemplar tracking cells from smoothed maps `sm[period]` (N,m,m).

    Gates: Skaggs >= threshold in every period; smoothed field area
    (fraction of bins >= 0.5) >= min_field_area in every period (rejects
    single-bin "speckle" fields); field moves toward A in phase A and toward
    B in phase B and ends within `radius` of each. The radius is relaxed in
    0.05 steps until n_examples candidates exist; candidates are ranked by
    how close they end to A (after A) plus to B (after B).
    """
    pre, pA, pB = [p for p, _ in PERIODS]
    peaks = {p: np.array([field_peak(s) for s in sm[p]]) for p in sm}

    def d(p, R):
        return np.linalg.norm(peaks[p] - np.asarray(R)[None, :], axis=1)

    place = np.logical_and.reduce([sks[p] >= skaggs_threshold for p in (pre, pA, pB)])
    area_ok = np.logical_and.reduce(
        [(sm[p] >= 0.5).mean(axis=(1, 2)) >= min_field_area for p in (pre, pA, pB)])
    toward_A = d(pre, reward_a) - d(pA, reward_a)
    toward_B = d(pA, reward_b) - d(pB, reward_b)
    dA_mid = d(pA, reward_a)
    dB_end = d(pB, reward_b)

    base = place & area_ok & (toward_A > 0) & (toward_B > 0)
    for radius in np.arange(0.25, 0.45 + 1e-9, 0.05):
        good = base & (dA_mid < radius) & (dB_end < radius)
        if int(good.sum()) >= n_examples:
            break
    ids = np.where(good)[0]
    order = np.argsort(dA_mid[ids] + dB_end[ids])
    sel = ids[order[:n_examples]]
    print(f"  [panel a] {len(ids)} tracking exemplar candidates "
          f"(radius {radius:.2f}); showing {len(sel)}: {sel.tolist()}")
    return sel


def fig_example_cells_grid(run_dir, stem, n_examples: int = 6,
                           skaggs_threshold: float = SKAGGS_THRESHOLD,
                           reward_a=REWARD_A, reward_b=REWARD_B, m: int = M) -> tuple:
    """Panel a — 3×N grid of example tracking-cell rate maps."""
    data_dir = Path(run_dir) / "data"
    sm, sks = {}, {}
    for period, _ in PERIODS:
        rm, sk = _load_rate_maps(data_dir, period)
        sm[period] = np.array([smooth_and_norm(rm[n]) for n in range(rm.shape[0])])
        sks[period] = sk
        del rm
    sel = select_example_cells(sm, sks, n_examples, skaggs_threshold,
                               reward_a, reward_b)

    ax_a = (reward_a[0] * m, reward_a[1] * m)
    ax_b = (reward_b[0] * m, reward_b[1] * m)
    n = len(sel)
    fig, axes = plt.subplots(3, n, figsize=(0.95 * n + 0.35, 3.05), squeeze=False)
    for col, nid in enumerate(sel):
        for row, (period, label) in enumerate(PERIODS):
            ax = axes[row][col]
            ax.imshow(sm[period][int(nid)], origin="lower", cmap="jet", vmin=0, vmax=1)
            ax.plot(*ax_a, "*", color=COLOR_A, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(*ax_b, "*", color=COLOR_B, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label.replace("\n", " "), fontsize=6)
            if row == 0:
                ax.set_title(f"#{int(nid)}", fontsize=6, pad=2)
    fig.suptitle("Example cells: place fields track the reward from A (teal) to B (orange)",
                 fontsize=7, y=1.0)
    fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.25)
    return save_figure(fig, stem)
