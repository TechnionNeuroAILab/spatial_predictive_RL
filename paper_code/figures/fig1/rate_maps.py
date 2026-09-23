"""Figure-1 rate-map plotting helpers.

The rate-map primitives themselves come from lib/ratemaps.py -- this module
used to carry its own verbatim copy of them.
"""
from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon as _wilcoxon

from utils import FS_SMALL, FS_TICK, arena_axes, reward_marker, thin_colorbar, C, fmt_p, hist_outline, stat_annot, RATEMAP_CMAP, apply_nature_style, save_figure

import sys as _sys
from pathlib import Path as _Path
_LIB = _Path(__file__).resolve().parents[2] / "lib"
if str(_LIB) not in _sys.path:
    _sys.path.insert(0, str(_LIB))
from pubstyle import RATEMAP as _RATEMAP, DIVERGING as _DIVERGING, SEQUENTIAL as _SEQUENTIAL  # noqa: E402
from ratemaps import (  # noqa: E402
    build_rate_maps, compute_skaggs, smooth_and_norm,
    field_center_of_mass, field_peak,
)


apply_nature_style()


def dist_to_reward(centers: np.ndarray, reward) -> np.ndarray:
    """Euclidean distance from each field centre to `reward`."""
    reward = np.asarray(reward, dtype=float)
    return np.linalg.norm(np.asarray(centers, dtype=float) - reward[None, :], axis=1)


def compute_field_centers(rate_maps, sigma: float = 1.5):
    """Peak and COM for every unit. Returns (peak[N,2], com[N,2])."""
    n = rate_maps.shape[0]
    peak = np.zeros((n, 2), dtype=float)
    com = np.zeros((n, 2), dtype=float)
    for i in range(n):
        sm = smooth_and_norm(rate_maps[i], sigma=sigma)
        peak[i] = field_peak(sm)
        com[i] = field_center_of_mass(sm)
    return peak, com


def place_cell_mask(sk_before, sk_after, threshold: float = 0.05) -> np.ndarray:
    """Units that pass the Skaggs place-cell gate in both periods."""
    return (np.asarray(sk_before) >= threshold) & (np.asarray(sk_after) >= threshold)


def analyse_peak_com_distances(
    rm_before,
    sk_before,
    rm_after,
    sk_after,
    reward_pos,
    skaggs_threshold: float = 0.05,
):
    """Distance-to-reward for peak and COM before vs after RL (place cells only)."""
    mask = place_cell_mask(sk_before, sk_after, skaggs_threshold)
    ids = np.where(mask)[0]
    peak_b, com_b = compute_field_centers(rm_before)
    peak_a, com_a = compute_field_centers(rm_after)
    R = np.asarray(reward_pos, dtype=float)
    out = {
        "neuron_ids": ids,
        "n": int(len(ids)),
        "skaggs_threshold": float(skaggs_threshold),
        "peak_before": dist_to_reward(peak_b[ids], R),
        "peak_after": dist_to_reward(peak_a[ids], R),
        "com_before": dist_to_reward(com_b[ids], R),
        "com_after": dist_to_reward(com_a[ids], R),
    }
    for key in ("peak", "com"):
        db, da = out[f"{key}_before"], out[f"{key}_after"]
        delta = da - db  # < 0 = toward reward
        out[f"{key}_mean_before"] = float(np.mean(db)) if len(db) else float("nan")
        out[f"{key}_mean_after"] = float(np.mean(da)) if len(da) else float("nan")
        out[f"{key}_mean_delta"] = float(np.mean(delta)) if len(delta) else float("nan")
        out[f"{key}_frac_toward"] = float(np.mean(delta < 0)) if len(delta) else float("nan")
        # Paired Wilcoxon signed-rank test: H0 = no shift (delta == 0)
        if len(da) >= 5 and not np.all(da == db):
            try:
                _, p = _wilcoxon(da, db)
                out[f"{key}_wilcoxon_p"] = float(p)
            except Exception:
                out[f"{key}_wilcoxon_p"] = float("nan")
        else:
            out[f"{key}_wilcoxon_p"] = float("nan")
    return out


def _fmt_p(p: float) -> str:
    """Format a p-value for figure annotation.

    An unavailable p-value prints nothing. The previous version rendered the
    literal "p = n/a" onto the panel, which both looks unfinished and, because
    it was drawn in the same corner as the legend, overlapped the legend text.
    """
    return fmt_p(p if np.isfinite(p) else None)


def fig_dist_histogram(
    d_before,
    d_after,
    stem,
    title: str,
    xlabel: str = "Distance to reward",
    bins: int = 20,
    color_before: str = C["pre"],
    color_after: str = C["policy"],
    label_before: str = "Prediction",
    label_after: str = "Goal policy",
    wilcoxon_p: float = float("nan"),
    width_in: float | None = None,
):
    """Overlay before/after histograms of distance-to-reward for one metric.

    Parameters
    ----------
    wilcoxon_p : float
        Pre-computed paired Wilcoxon p-value to annotate on the figure.
        Pass from ``analyse_peak_com_distances`` output.
    """
    d_before = np.asarray(d_before, dtype=float)
    d_after = np.asarray(d_after, dtype=float)
    if len(d_before) == 0 and len(d_after) == 0:
        fig, ax = plt.subplots(figsize=(2.4, 1.8))
        ax.text(0.5, 0.5, "no place cells", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return save_figure(fig, stem)

    lo = 0.0
    hi = float(np.nanmax([d_before.max() if len(d_before) else 0, d_after.max() if len(d_after) else 0]))
    hi = max(hi * 1.05, 0.15)
    edges = np.linspace(lo, hi, bins + 1)

    w = width_in if width_in else 2.6
    fig, ax = plt.subplots(figsize=(w, w * 0.73))
    hist_outline(ax, d_before, edges, color_before, label=label_before)
    hist_outline(ax, d_after, edges, color_after, label=label_after)
    if len(d_before):
        ax.axvline(float(np.mean(d_before)), color=color_before,
                   ls=(0, (2.5, 2)), lw=0.7, zorder=1)
    if len(d_after):
        ax.axvline(float(np.mean(d_after)), color=color_after,
                   ls=(0, (2.5, 2)), lw=0.7, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cells")
    # NB: the `title` parameter shadows pubstyle's title() helper here, so set
    # it directly with the same left-aligned convention.
    ax.set_title(title, loc="left", pad=3)
    ax.legend(loc="upper right")
    # p-value bottom-right, so it cannot collide with the upper-right legend.
    stat_annot(ax, _fmt_p(wilcoxon_p), loc="lower right")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem)


def cycle_num(label: str) -> int:
    """Cycle number parsed out of a period label; -1 when there is none."""
    mobj = re.search(r"cycle_(\d+)", label)
    return int(mobj.group(1)) if mobj else -1


def load_cycle_rate_maps(data_dir, m: int = 32, hidden_dim: int = 512, probe_type: str = "goal"):
    """Rate maps and Skaggs scores for every cycle of one probe type ('goal' or random-walk)."""
    if probe_type == "goal":
        cycle_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("df_activations_cycle_") and f.endswith("_goal.pkl")
        )
    else:
        cycle_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("df_activations_cycle_") and f.endswith(".pkl") and "goal" not in f
        )
    cycle_data = {}
    for fname in cycle_files:
        label = fname.replace("df_activations_", "").replace(".pkl", "")
        df_a = pd.read_pickle(os.path.join(data_dir, fname))
        beh_fname = fname.replace("df_activations_", "df_behavior_")
        df_b = pd.read_pickle(os.path.join(data_dir, beh_fname))
        rm, occ = build_rate_maps(df_b, df_a, m, hidden_dim)
        sk = compute_skaggs(rm, occ)
        cycle_data[label] = (rm, occ, sk)
    return cycle_data


def fig_100cells(rate_maps, skaggs, top_idx, reward_bin, title: str = "Cognitive Map", stem=None):
    """10x10 grid of the top units' smoothed rate maps, with the reward marked."""
    n_show = min(100, len(top_idx))
    rx, ry = reward_bin
    fig = plt.figure(figsize=(15, 15), dpi=150)
    for i in range(n_show):
        nid = top_idx[i]
        ax = fig.add_subplot(10, 10, i + 1)
        rm = smooth_and_norm(rate_maps[nid])
        ax.imshow(rm, origin="lower", cmap=RATEMAP_CMAP, interpolation="nearest", vmin=0, vmax=1)
        ax.plot(rx, ry, "w*", markersize=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(title, fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if stem is not None:
        return save_figure(fig, stem)
    plt.close(fig)
    return None, None


def fig_pred_vs_rl_across_cycles(rw_cycles, goal_cycles, neuron_id, reward_bin, stem=None):
    """One unit across cycles: prediction probe on the top row, RL probe below."""
    rw_labels = sorted(rw_cycles.keys(), key=cycle_num)
    goal_labels = sorted(goal_cycles.keys(), key=cycle_num)
    n = max(len(rw_labels), len(goal_labels))
    if n == 0:
        return None, None
    rx, ry = reward_bin
    fig, axes = plt.subplots(2, n, figsize=(n * 1.9, 2 * 2.1), squeeze=False)
    fig.patch.set_facecolor("#1a1a1a")
    for col in range(n):
        for row, cycles, labels in ((0, rw_cycles, rw_labels), (1, goal_cycles, goal_labels)):
            ax = axes[row][col]
            if col < len(labels):
                rm = cycles[labels[col]][0]
                ax.imshow(smooth_and_norm(rm[neuron_id]), origin="lower", cmap=_RATEMAP, vmin=0, vmax=1)
                ax.plot(rx, ry, "w*", markersize=5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("black")
            for spine in ax.spines.values():
                spine.set_visible(False)
    axes[0][0].set_ylabel("Prediction", fontsize=10, color="white")
    axes[1][0].set_ylabel("RL", fontsize=10, color="white")
    for col in range(n):
        lab = goal_labels[col] if col < len(goal_labels) else rw_labels[col]
        axes[1][col].set_xlabel(f"C{cycle_num(lab)}", fontsize=9, color="white")
    plt.tight_layout()
    if stem is not None:
        return save_figure(fig, stem)
    plt.close(fig)
    return None, None


def fig_pred_vs_rl_cycle1_many(
    rw_cycles,
    goal_cycles,
    neuron_ids,
    reward_bin,
    stem=None,
    ylabel_top: str = "Before RL",
    ylabel_bottom: str = "After RL",
    cmap=_RATEMAP,
    width_in: float | None = None,
):
    """Many neurons: row 0 = cycle_1 RW probe, row 1 = cycle_1 goal probe."""
    rw_key = next((k for k in rw_cycles if cycle_num(k) == 1 and "goal" not in k), None)
    goal_key = next((k for k in goal_cycles if cycle_num(k) == 1), None)
    if rw_key is None or goal_key is None:
        raise RuntimeError("Cycle 1 probe pickles not found in run data.")

    n = len(neuron_ids)
    rx, ry = reward_bin
    # Rendered at final on-page width so the cell IDs and row labels keep their
    # type size through assembly.
    w = width_in if width_in else n * 1.05
    # Maps are square: n across means each is w/n wide, so two rows need
    # 2*w/n of height, plus a strip for the cell IDs and row labels.
    fig, axes = plt.subplots(2, n, figsize=(w, 2.0 * w / n + 0.30), squeeze=False)
    im = None
    for col, nid in enumerate(neuron_ids):
        for row, cycles, key in ((0, rw_cycles, rw_key), (1, goal_cycles, goal_key)):
            ax = axes[row][col]
            rm = cycles[key][0]
            im = ax.imshow(smooth_and_norm(rm[int(nid)]), origin="lower",
                           cmap=cmap, vmin=0, vmax=1)
            arena_axes(ax)
            reward_marker(ax, rx, ry, size=3.4)
            if row == 0:
                ax.set_title(str(nid), fontsize=FS_SMALL, pad=1.5,
                             color=C["muted"], loc="center")
    axes[0][0].set_ylabel(ylabel_top, fontsize=FS_TICK, labelpad=3)
    axes[1][0].set_ylabel(ylabel_bottom, fontsize=FS_TICK, labelpad=3)
    plt.tight_layout(pad=0.25, w_pad=0.12, h_pad=0.18)
    if im is not None:
        thin_colorbar(fig, im, axes[0][n - 1], label="Rate",
                      ticklabels=("0", "max"), shrink=0.85, pad=0.016, width=0.008)
    if stem is not None:
        return save_figure(fig, stem)
    plt.close(fig)
    return None, None


def select_random_neurons(
    rm_before,
    rm_after,
    n: int = 8,
    seed: int = 42,
    min_peak: float = 0.15,
    min_field_bins: int = 50,
    field_thr: float = 0.25,
) -> list[int]:
    """Random unit ids with visible activity in both periods (not Skaggs-gated).

    A cell is kept if each period has peak rate ≥ ``min_peak`` and at least
    ``min_field_bins`` bins with smoothed normalized rate ≥ ``field_thr``.
    """
    rng = np.random.default_rng(seed)
    n_neurons = int(rm_before.shape[0])

    def _ok(rm) -> bool:
        if float(np.max(rm)) < min_peak:
            return False
        s = smooth_and_norm(rm)
        return int(np.sum(s >= field_thr)) >= min_field_bins

    pool = np.array([i for i in range(n_neurons) if _ok(rm_before[i]) and _ok(rm_after[i])], dtype=int)
    if len(pool) < n:
        # Relax field area, keep peak gate.
        def _ok_peak(rm) -> bool:
            return float(np.max(rm)) >= min_peak
        pool = np.array(
            [i for i in range(n_neurons) if _ok_peak(rm_before[i]) and _ok_peak(rm_after[i])],
            dtype=int,
        )
    if len(pool) == 0:
        raise RuntimeError("No active neurons found for random panel-d selection.")
    chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
    return [int(i) for i in chosen]


def select_toward_reward_neurons(
    rm_before,
    rm_after,
    sk_before,
    sk_after,
    reward_pos,
    n: int = 8,
    min_skaggs: float = 1.0,
    min_delta: float = 0.12,
    min_corr: float = 0.25,
    max_sparsity: float = 0.30,
    max_post_dist: float = 0.45,
):
    """Pick localized cells whose field peak moves toward the reward after RL."""
    from scipy.ndimage import gaussian_filter

    R = np.asarray(reward_pos, dtype=float)
    m = rm_before.shape[1]

    def peak_xy(rm):
        s = gaussian_filter(rm.astype(float), 1.5)
        x, y = divmod(int(np.argmax(s)), m)
        return np.array([x / m, y / m], dtype=float)

    cands = []
    for nid in range(rm_before.shape[0]):
        if sk_before[nid] < min_skaggs or sk_after[nid] < min_skaggs:
            continue
        sp_b = float((smooth_and_norm(rm_before[nid]) >= 0.3).mean())
        sp_a = float((smooth_and_norm(rm_after[nid]) >= 0.3).mean())
        if sp_b > max_sparsity or sp_a > max_sparsity:
            continue
        pb, pa = peak_xy(rm_before[nid]), peak_xy(rm_after[nid])
        db, da = float(np.linalg.norm(pb - R)), float(np.linalg.norm(pa - R))
        delta = db - da
        if delta < min_delta or da > max_post_dist:
            continue
        vb = smooth_and_norm(rm_before[nid]).ravel()
        va = smooth_and_norm(rm_after[nid]).ravel()
        if vb.std() < 1e-8 or va.std() < 1e-8:
            continue
        corr = float(np.corrcoef(vb, va)[0, 1])
        if corr < min_corr:
            continue
        score = delta + 0.35 * corr
        cands.append((score, nid, delta, corr, da))
    cands.sort(reverse=True)
    return [int(c[1]) for c in cands[:n]], cands[:n]
