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
    C, FS_BASE, FS_SMALL, FS_TICK, arena_axes, fmt_p, stat_annot,
    style_violin, title, zero_line,
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


def _center_key(center: str) -> str:
    key = str(center).strip().lower()
    if key not in ("com", "peak"):
        raise ValueError(f"center must be 'com' or 'peak', not {center!r}")
    return key


def paired_tracking_stats(states: dict, skaggs_threshold: float = SKAGGS_THRESHOLD,
                          center: str = "com") -> dict:
    """
    Compute tracking statistics:
      - mean distance to A and B in each period
      - shift toward A (pre→postA) and toward B (postA→postB)

    `center` is 'com' (centre of mass; manuscript default) or 'peak'.
    """
    key = _center_key(center)
    pre, pA, pB = PERIOD_PRE, PERIOD_POST_A, PERIOD_POST_B

    def paired_shift(p_from, p_to, reward):
        sk_f = states[p_from]["sk"]
        sk_t = states[p_to]["sk"]
        # place cell in BOTH periods: the rule the text uses (analysis_helpers.relocation_asymmetry)
        active = np.where(np.minimum(sk_f, sk_t) >= skaggs_threshold)[0]
        d_f = np.linalg.norm(states[p_from][key][active] - reward[None, :], axis=1)
        d_t = np.linalg.norm(states[p_to][key][active] - reward[None, :], axis=1)
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
        xy = states[period][key]
        place = np.where(sk >= skaggs_threshold)[0]
        dA = np.linalg.norm(xy[place] - REWARD_A[None, :], axis=1)
        dB = np.linalg.norm(xy[place] - REWARD_B[None, :], axis=1)
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

def fig_tracking_curves(ts: dict, stem,
                        size_in: tuple[float, float] | None = None,
                        center: str = "com") -> tuple:
    """Panel c — mean distance to A and B across three periods (crossing curves)."""
    # Keep the two-line period names. Flattened to one line they overlapped
    # each other in the narrower bottom-row slot ("baseline (pre-RL)After A
    # After B (relocated)" ran together on the page).
    labels = [lab for _, lab in PERIODS]
    x = np.arange(len(PERIODS))
    mA = [d.mean() for d in ts["dists_A"]]
    sA = [d.std(ddof=1) / np.sqrt(len(d)) for d in ts["dists_A"]]
    mB = [d.mean() for d in ts["dists_B"]]
    sB = [d.std(ddof=1) / np.sqrt(len(d)) for d in ts["dists_B"]]

    fig, ax = plt.subplots(figsize=size_in or (3.2, 2.4))
    ax.errorbar(x, mA, yerr=sA, fmt="-o", lw=1.1, ms=3.2, color=COLOR_A, capsize=2,
                capthick=0.6, elinewidth=0.7, label="Reward A")
    ax.errorbar(x, mB, yerr=sB, fmt="-s", lw=1.1, ms=3.0, color=COLOR_B, capsize=2,
                capthick=0.6, elinewidth=0.7, label="Reward B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(("Field-peak" if _center_key(center) == "peak"
                   else "Field-centre") + " distance (arena units)")
    title(ax, "Distance to each reward")
    # Headroom first, then a single legend row across the top: the data span
    # only ~0.04 arena units, so without it the legend sat on the curves.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.38 * (hi - lo))
    ax.legend(loc="upper center", ncol=2, columnspacing=0.8)
    # SEM is stated once, here, rather than folded into the axis label.
    stat_annot(ax, "mean ± s.e.m.", loc="lower left")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_shift_violins(ts: dict, stem,
                      size_in: tuple[float, float] | None = None) -> tuple:
    """Panel d — Δd violin: toward A and toward B."""
    from utils import apply_nature_style
    apply_nature_style()

    sA = ts["shift_A"]
    sB = ts["shift_B"]

    def _fmt_p(p):
        if not np.isfinite(p):
            return ""
        return fmt_p(p, italic=False)

    fig, ax = plt.subplots(figsize=size_in or (2.9, 2.4))
    spec = ((1, sA, COLOR_A), (2, sB, COLOR_B))
    for pos, s, colour in spec:
        style_violin(ax.violinplot([s["delta"]], positions=[pos], widths=0.72,
                                   showmeans=True, showmedians=False,
                                   showextrema=True), colour)
    zero_line(ax)
    ax.set_xticks([1, 2])
    # Tick labels name the contrast only. Statistics under a tick label is not
    # a tick label, and it pushed the axis furniture down by three lines.
    # Two lines: side by side on one line the two contrast names ran
    # together ("pre → post Apost A → post B") in the narrow slot.
    ax.set_xticklabels(["pre →\npost A", "post A →\npost B"])
    ax.set_xlim(0.45, 2.55)
    ax.set_ylabel(r"$\Delta$ distance to reward")
    title(ax, "Shift toward the\nactive reward")
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
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_com_scatter_three(states: dict, stem, reward_a=REWARD_A, reward_b=REWARD_B,
                          skaggs_threshold: float = SKAGGS_THRESHOLD,
                          size_in: tuple[float, float] | None = None,
                          center: str = "com") -> tuple:
    """Panel b — field-centre clouds for three periods, coloured by period.

    `center='com'` plots centres of mass (manuscript default). `center='peak'`
    plots the argmax of each smoothed map. Dot style matches Figure 3 panel b.
    """
    key = _center_key(center)
    fig, axes = plt.subplots(1, 3, figsize=size_in or (5.4, 1.95))
    counts = []
    for i, (ax, (period, label)) in enumerate(zip(axes, PERIODS)):
        sk = states[period]["sk"]
        xy = states[period][key]
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
        ax.scatter(xy[place, 0], xy[place, 1],
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
    return save_figure(fig, stem, exact_size=size_in is not None)


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
    beh = pd.read_parquet(os.path.join(str(data_dir), f"df_behavior_{period}.parquet"))
    act = pd.read_parquet(os.path.join(str(data_dir), f"df_activations_{period}.parquet"))
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


def _ab_marker(ax, x, y, color):
    """Reward A/B star, drawn the same way as everywhere else in the paper.

    This panel used to carry its own `ax.plot("*", markersize=6/7/8.6)`
    implementations, so the same landmark had three sizes across one figure.
    """
    ax.plot([x], [y], marker="*", ms=7.0, mfc=color, mec="white", mew=0.6,
            ls="none", zorder=6, clip_on=False)


def load_example_cell_maps(run_dir, n_examples: int = 8,
                           skaggs_threshold: float = SKAGGS_THRESHOLD,
                           reward_a=REWARD_A, reward_b=REWARD_B):
    """Smoothed rate maps and the example-cell ids panel a draws."""
    data_dir = Path(run_dir) / "data"
    sm, sks = {}, {}
    for period, _ in PERIODS:
        rm, sk = _load_rate_maps(data_dir, period)
        sm[period] = np.array([smooth_and_norm(rm[n]) for n in range(rm.shape[0])])
        sks[period] = sk
        del rm
    sel = select_example_cells(sm, sks, n_examples, skaggs_threshold,
                               reward_a, reward_b)
    return {"sm": sm, "sel": np.asarray(sel, dtype=int)}


def fig_example_cells_grid(run_dir, stem, n_examples: int = 8,
                           size_in: tuple[float, float] | None = None,
                           skaggs_threshold: float = SKAGGS_THRESHOLD,
                           reward_a=REWARD_A, reward_b=REWARD_B, m: int = M,
                           maps=None) -> tuple:
    """Panel a — 3×N grid of example tracking-cell rate maps."""
    if maps is None:
        maps = load_example_cell_maps(
            run_dir, n_examples=n_examples, skaggs_threshold=skaggs_threshold,
            reward_a=reward_a, reward_b=reward_b,
        )
    sm = maps["sm"]
    sel = maps["sel"]

    ax_a = (reward_a[0] * m, reward_a[1] * m)
    ax_b = (reward_b[0] * m, reward_b[1] * m)
    n = len(sel)
    fig, axes = plt.subplots(3, n, figsize=size_in or (0.95 * n + 0.35, 3.05),
                             squeeze=False)
    for col, nid in enumerate(sel):
        for row, (period, label) in enumerate(PERIODS):
            ax = axes[row][col]
            ax.imshow(sm[period][int(nid)], origin="lower", cmap=RATEMAP, vmin=0, vmax=1)
            _ab_marker(ax, *ax_a, COLOR_A)
            _ab_marker(ax, *ax_b, COLOR_B)
            arena_axes(ax)
            if col == 0:
                ax.set_ylabel(label.replace("\n", " "), fontsize=FS_TICK)
            if row == 0:
                ax.set_title(f"#{int(nid)}", fontsize=FS_SMALL, pad=2,
                             color=C["muted"])
    # The empty suptitle that used to sit here still reserved vertical space.
    fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.25)
    return save_figure(fig, stem, exact_size=size_in is not None)


def _bracket(ax, x0, x1, y, text):
    """Thin bracket between two groups with the paired-test label above it."""
    h = 0.012 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color="#555555", lw=0.5,
            clip_on=False)
    ax.text((x0 + x1) / 2, y + h, text, ha="center", va="bottom",
            fontsize=FS_SMALL, color="#333333")


def fig_network_shifts_reloc(stem, size_in: tuple[float, float] | None = None):
    """Panel c: per-network mean shift toward A (baseline -> after A) and
    toward B (after A -> after B), field peak and centre of mass, in the 50
    retained relocation networks. Grey lines pair a network's two steps;
    diamonds are mean +/- s.e.m.; brackets carry the paired Wilcoxon P of
    the A - B difference."""
    from pubstyle import METRIC_ALPHA, load_headline_stats, network_strip

    hs = load_headline_stats()["reloc"]
    seeds = set(hs["cohort"]["seeds"])
    rows = [r for r in hs["rows"] if r["seed"] in seeds]
    fig, ax = plt.subplots(figsize=size_in or (2.35, 2.3))
    groups = [("peak", 1.0), ("com", 3.0)]
    tops = []
    for metric, x0 in groups:
        a = np.array([r[f"A_{metric}"]["mean"] for r in rows])
        b = np.array([r[f"B_{metric}"]["mean"] for r in rows])
        network_strip(ax, [a, b], [x0, x0 + 1.0], [COLOR_A, COLOR_B],
                      alpha=0.55 * METRIC_ALPHA[metric] + 0.1)
        p = hs["cohort"][metric]["A_minus_B"]["p_two_sided"]
        tops.append((x0, float(max(a.max(), b.max())), p))
    zero_line(ax)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["to A", "to B", "to A", "to B"])
    ax.set_xlim(0.4, 4.6)
    ax.set_ylabel(r"$\Delta$ distance to active reward")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))
    for x0, top, p in tops:
        _bracket(ax, x0, x0 + 1.0, top + 0.05 * (hi - lo), fmt_p(p))
    for metric, x0 in groups:
        ax.text(x0 + 0.5, -0.13, "Peak" if metric == "peak" else "Centre of mass",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=FS_TICK)
    title(ax, f"Across networks (n = {len(rows)})")
    fig.tight_layout(pad=0.3)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_shift_violins_both(ts_peak: dict, ts_com: dict, stem,
                           size_in: tuple[float, float] | None = None) -> tuple:
    """Panel d: per-cell shift toward the active reward in the example network,
    baseline -> after A (toward A) and after A -> after B (toward B), for the
    field peak (left) and centre of mass (right), with the per-cell paired
    Wilcoxon P."""
    # Field peaks can jump between sub-fields, so a few cells shift by most of
    # the arena. Violins show the central 98% of cells (1st-99th percentile of
    # all four distributions, stated in the caption); means, fractions and P
    # use every cell.
    allv = np.concatenate([t[k]["delta"] for t in (ts_peak, ts_com)
                           for k in ("shift_A", "shift_B")])
    lo, hi = np.percentile(allv, [1, 99])
    lo, hi = min(lo, -0.05), max(hi, 0.05)

    fig, axes = plt.subplots(1, 2, figsize=size_in or (3.2, 2.3), sharey=True)
    for ax, ts, ttl in ((axes[0], ts_peak, "Peak"), (axes[1], ts_com, "Centre of mass")):
        spec = ((1, ts["shift_A"], COLOR_A), (2, ts["shift_B"], COLOR_B))
        for pos, sh, colour in spec:
            shown = sh["delta"][(sh["delta"] >= lo) & (sh["delta"] <= hi)]
            parts = ax.violinplot([shown], positions=[pos], widths=0.72,
                                  showmeans=False, showmedians=False,
                                  showextrema=False)
            style_violin(parts, colour)
            ax.hlines(sh["mean_delta"], pos - 0.22, pos + 0.22, color="#222222",
                      lw=0.8, zorder=3)
        zero_line(ax)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["to A", "to B"])
        ax.set_xlim(0.4, 2.6)
        title(ax, ttl)
    axes[0].set_ylim(lo, hi + 0.45 * (hi - lo))
    for ax, ts in ((axes[0], ts_peak), (axes[1], ts_com)):
        for pos, sh in ((1, ts["shift_A"]), (2, ts["shift_B"])):
            p = sh["wilcoxon_p"]
            ax.text(pos, hi + 0.06 * (hi - lo),
                    f"{sh['mean_delta']:+.3f}\n{100 * sh['frac_toward']:.0f}% toward\n"
                    f"{fmt_p(p if np.isfinite(p) else None)}",
                    ha="center", va="bottom", fontsize=FS_SMALL, color=C["muted"],
                    linespacing=1.3)
    axes[0].set_ylabel(r"$\Delta$ distance to active reward")
    fig.tight_layout(pad=0.3, w_pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)
