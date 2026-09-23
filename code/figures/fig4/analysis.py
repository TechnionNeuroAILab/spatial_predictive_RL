"""
Figure 4 analysis helpers — reward relocation experiment.

Loads the three frozen-replay periods (baseline, postA, postB), computes
field-center distances to A and B, and returns tracking statistics.

All heavy lifting is done by code/lib/analysis_helpers.py (load_period_centers)
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

_NEW_CODE = Path(__file__).parents[2] / "lib"
sys.path.insert(0, str(_NEW_CODE))

from analysis_helpers import load_period_centers, has_period, seed_is_clean  # noqa: E402
from ratemaps import build_rate_maps, compute_skaggs, smooth_and_norm, field_peak  # noqa: E402

from utils import (
    C, FS_BASE, FS_SMALL, fmt_p, stat_annot, style_violin, title, zero_line,
    RATEMAP,  # noqa: E402
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
    """Paired two-sided Wilcoxon p; NaN for n < 5 or when the two arrays are identical."""
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
    ax.errorbar(x, mA, yerr=sA, fmt="-o", lw=1.1, ms=3.2, color=COLOR_A, capsize=2,
                capthick=0.6, elinewidth=0.7, label="Reward A")
    ax.errorbar(x, mB, yerr=sB, fmt="-s", lw=1.1, ms=3.0, color=COLOR_B, capsize=2,
                capthick=0.6, elinewidth=0.7, label="Reward B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Field-centre distance (arena units)")
    title(ax, "Distance to each reward")
    ax.legend(loc="best")
    # SEM is stated once, here, rather than folded into the axis label.
    stat_annot(ax, "mean ± s.e.m.", loc="lower left")
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

    fig, ax = plt.subplots(figsize=(2.9, 2.4))
    spec = ((1, sA, COLOR_A), (2, sB, COLOR_B))
    for pos, s, colour in spec:
        style_violin(ax.violinplot([s["delta"]], positions=[pos], widths=0.72,
                                   showmeans=True, showmedians=False,
                                   showextrema=True), colour)
    zero_line(ax)
    ax.set_xticks([1, 2])
    # Tick labels name the contrast only. Statistics under a tick label is not
    # a tick label, and it pushed the axis furniture down by three lines.
    ax.set_xticklabels(["pre → post A", "post A → post B"])
    ax.set_xlim(0.45, 2.55)
    ax.set_ylabel(r"$\Delta$ distance to reward")
    title(ax, "Shift toward the active reward")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.20 * (hi - lo))
    for pos, s, _c in spec:
        top = float(np.nanmax(s["delta"])) if len(s["delta"]) else 0.0
        ax.text(pos, top + 0.05 * (hi - lo),
                f"{s['mean_delta']:+.3f}\n{100*s['frac_toward']:.0f}% toward\n"
                f"{fmt_p(s['wilcoxon_p'] if np.isfinite(s['wilcoxon_p']) else None)}",
                ha="center", va="bottom", fontsize=FS_SMALL, color=C["muted"],
                linespacing=1.3)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_com_scatter_three(states: dict, stem, reward_a=REWARD_A, reward_b=REWARD_B,
                          skaggs_threshold: float = SKAGGS_THRESHOLD) -> tuple:
    """Panel b — COM field-centre clouds for three periods, coloured by period.

    Same graphic as Figure 3 panel b (fig_centers_abc): centre-of-mass dots,
    s=3, alpha=0.35, no edge, rasterized — not grid-snapped field peaks.
    """
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.95))
    counts = []
    for i, (ax, (period, label)) in enumerate(zip(axes, PERIODS)):
        sk = states[period]["sk"]
        com = states[period]["com"]
        place = np.where(sk >= skaggs_threshold)[0]
        counts.append(len(place))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        # Closed box + arena coordinates on the first panel: bare L-shaped
        # spines with no ticks gave the reader no sense of the arena extent.
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.5); spine.set_color("#B0B0B0")
        if i == 0:
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=FS_SMALL, pad=1.5)
            ax.set_xlabel("Arena x", labelpad=1.5); ax.set_ylabel("Arena y", labelpad=1.5)
        else:
            ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(com[place, 0], com[place, 1],
                   s=3.2, alpha=0.42, c=PERIOD_COLORS[period], linewidths=0, rasterized=True)
        for (rx, ry), col in ((reward_a, COLOR_A), (reward_b, COLOR_B)):
            ax.plot([rx], [ry], marker="*", ms=8.6, mfc="none", mec="white",
                    mew=1.3, ls="none", zorder=5, clip_on=False)
            ax.plot([rx], [ry], marker="*", ms=7.0, mfc=col, mec="#333333",
                    mew=0.4, ls="none", zorder=5.1, clip_on=False)
        title(ax, label.replace(chr(10), " "))
    # n once, outside the boxes, instead of three times in three titles.
    axes[2].text(1.0, -0.06, "n = " + " / ".join(str(c) for c in counts) + " cells",
                 transform=axes[2].transAxes, ha="right", va="top",
                 fontsize=FS_SMALL, color=C["muted"])
    fig.tight_layout(pad=0.35, w_pad=0.35)
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Panel a — example tracking cells (local implementation)
#
# Replaces an earlier exemplar panel whose quality score rewarded field sharpness
# (quality = minsk + 0.7*sharp - ...), so needle-thin "speckle" fields won even
# when they did not track the reward.
# Here exemplars must (i) pass the Skaggs gate in all three periods,
# (ii) have a minimum smoothed-field area (anti-speckle), and (iii) genuinely
# track: move toward A in phase A and toward B in phase B, ending near each.
# ---------------------------------------------------------------------------

def _load_rate_maps(data_dir, period, m: int = M, hidden_dim: int = HIDDEN_DIM):
    """Rate maps and occupancy for one period's pickles."""
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
            ax.imshow(sm[period][int(nid)], origin="lower", cmap=RATEMAP, vmin=0, vmax=1)
            ax.plot(*ax_a, "*", color=COLOR_A, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(*ax_b, "*", color=COLOR_B, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label.replace("\n", " "), fontsize=6)
            if row == 0:
                ax.set_title(f"#{int(nid)}", fontsize=6, pad=2)
    # Title must not name colours: it silently goes wrong the moment the
    # palette changes. The A/B stars are identified by the panel legend.
    fig.suptitle("",
                 fontsize=7, y=1.0)
    fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.25)
    return save_figure(fig, stem)
