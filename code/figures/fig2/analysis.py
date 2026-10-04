"""Figure 2 data and panels: phases A (prediction), B (goal policy) and the frozen-replay pair C.

Field centres come from lib/ratemaps.py via figures/fig1/rate_maps.py, as (x, y).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    C, FS_BASE, FS_SMALL, FS_TICK, arena_axes, fmt_p, hist_outline, reward_marker, stat_annot, style_violin, thin_colorbar, title, zero_line,
    RATEMAP,
    PHASE_A,
    PHASE_B,
    PHASE_C_POST,
    PHASE_C_PRE,
    PHASE_COLORS,
    FIG1_DIR,
    HIDDEN_DIM,
    M,
    apply_nature_style,
    save_figure,
)

sys.path.insert(0, str(FIG1_DIR))
from rate_maps import (  # noqa: E402
    build_rate_maps,
    compute_field_centers,
    compute_skaggs,
    dist_to_reward,
    select_random_neurons,
    smooth_and_norm,
)

_NEW_CODE_DIR = FIG1_DIR.parents[1] / "lib"
if str(_NEW_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_NEW_CODE_DIR))
from analysis_helpers import frozen_replay_delta_from_centers  # noqa: E402

apply_nature_style()

# Fig. 1d example cells (seed 42, n = 8); used only if the random selection fails
FIG1_PANEL_D_NEURONS = [510, 364, 442, 334, 227, 229, 47, 400]


def load_period(data_dir: Path, period: str, m: int = M, hidden_dim: int = HIDDEN_DIM):
    """Rate maps, occupancy and Skaggs scores for one period's pickles."""
    beh = pd.read_pickle(data_dir / f"df_behavior_{period}.pkl")
    act = pd.read_pickle(data_dir / f"df_activations_{period}.pkl")
    rm, occ = build_rate_maps(beh, act, m, hidden_dim)
    sk = compute_skaggs(rm, occ)
    return rm, occ, sk


def load_phases(data_dir: Path):
    """Load Phase A, B, and frozen-replay C (pred / postrl)."""
    periods = {
        "A": PHASE_A,
        "B": PHASE_B,
        "C_pre": PHASE_C_PRE,
        "C_post": PHASE_C_POST,
    }
    out = {}
    for key, period in periods.items():
        rm, occ, sk = load_period(data_dir, period)
        peak, com = compute_field_centers(rm)
        out[key] = {
            "period": period,
            "rm": rm,
            "occ": occ,
            "sk": sk,
            "peak": peak,
            "com": com,
        }
    return out


def fig1_panel_d_neuron_ids(phases: dict, n: int = 8, seed: int = 42) -> list[int]:
    """Same selection rule as Figure 1 panel d (active on A and B)."""
    try:
        return select_random_neurons(phases["A"]["rm"], phases["B"]["rm"], n=n, seed=seed)
    except Exception as e:
        print(f"  [warn] example-cell selection failed ({type(e).__name__}: {e}); using the fixed Fig. 1d IDs",
              file=sys.stderr)
        return list(FIG1_PANEL_D_NEURONS[:n])


def shared_place_cell_ids(phases: dict, skaggs_threshold: float = 0.05) -> np.ndarray:
    """Units passing the Skaggs gate in A, B and C_post: the cell set of the A -> B contrast."""
    mask = (
        (phases["A"]["sk"] >= skaggs_threshold)
        & (phases["B"]["sk"] >= skaggs_threshold)
        & (phases["C_post"]["sk"] >= skaggs_threshold)
    )
    return np.where(mask)[0]


def paired_distance_stats(phases: dict, reward_pos, ids: np.ndarray,
                          ac_skaggs_threshold: float = 0.05, ac_min_cells: int = 10):
    """COM / peak distances and shifts (negative = toward reward) for two contrasts.

    A -> B: prediction walk vs goal policy (occupancy differs by design), on `ids`.
    "AC": frozen replay C_pre -> C_post (same trajectory, only weights differ), with its own gate.
    """
    R = np.asarray(reward_pos, dtype=float)
    com_a = phases["A"]["com"][ids]
    com_b = phases["B"]["com"][ids]
    peak_a = phases["A"]["peak"][ids]
    peak_b = phases["B"]["peak"][ids]

    d_com = {"A": dist_to_reward(com_a, R), "B": dist_to_reward(com_b, R)}
    d_peak = {"A": dist_to_reward(peak_a, R), "B": dist_to_reward(peak_b, R)}

    # frozen-replay contrast, gated on C_pre and C_post only
    com_ac = frozen_replay_delta_from_centers(
        phases["C_pre"]["sk"], phases["C_pre"]["com"],
        phases["C_post"]["sk"], phases["C_post"]["com"],
        R, skaggs_threshold=ac_skaggs_threshold, min_cells=ac_min_cells,
    )
    peak_ac = frozen_replay_delta_from_centers(
        phases["C_pre"]["sk"], phases["C_pre"]["peak"],
        phases["C_post"]["sk"], phases["C_post"]["peak"],
        R, skaggs_threshold=ac_skaggs_threshold, min_cells=ac_min_cells,
    )
    if com_ac is None or peak_ac is None:
        raise RuntimeError(
            "Too few place cells pass the frozen-replay (C_pre/C_post) Skaggs "
            f"gate (>= {ac_skaggs_threshold}, need >= {ac_min_cells})."
        )

    stats = {
        "n": int(len(ids)),                  # A/B population size (unchanged)
        "n_ac": com_ac["n"],                 # C_pre/C_post population size (independent gate)
        "neuron_ids": ids,
        "com": {"A": d_com["A"], "B": d_com["B"],
                "C_pre": com_ac["d_pre"], "C": com_ac["d_post"]},
        "peak": {"A": d_peak["A"], "B": d_peak["B"],
                 "C_pre": peak_ac["d_pre"], "C": peak_ac["d_post"]},
        "delta_com_AB": d_com["B"] - d_com["A"],
        "delta_com_AC": com_ac["d_post"] - com_ac["d_pre"],   # negative = toward reward (matches AB convention)
        "delta_peak_AB": d_peak["B"] - d_peak["A"],
        "delta_peak_AC": peak_ac["d_post"] - peak_ac["d_pre"],
    }
    for tag, delta in (
        ("AB", stats["delta_com_AB"]),
        ("AC", stats["delta_com_AC"]),
    ):
        stats[f"com_mean_delta_{tag}"] = float(np.mean(delta)) if len(delta) else float("nan")
        stats[f"com_frac_toward_{tag}"] = float(np.mean(delta < 0)) if len(delta) else float("nan")
    for tag, delta in (
        ("AB", stats["delta_peak_AB"]),
        ("AC", stats["delta_peak_AC"]),
    ):
        stats[f"peak_mean_delta_{tag}"] = float(np.mean(delta)) if len(delta) else float("nan")
        stats[f"peak_frac_toward_{tag}"] = float(np.mean(delta < 0)) if len(delta) else float("nan")
    return stats


def fig_replay_remap_grid(
    rm_pre,
    rm_post,
    neuron_ids,
    reward_bin,
    stem,
    ylabel_top: str = "Pre-RL",
    ylabel_bottom: str = "Post-RL",
    cmap=RATEMAP,
    title: str | None = None,
):
    """Panel a: pre / post rate maps of the example cells."""
    n = len(neuron_ids)
    rx, ry = reward_bin
    fig, axes = plt.subplots(2, n, figsize=(n * 1.05, 2.05), squeeze=False)
    im = None
    for col, nid in enumerate(neuron_ids):
        for row, rm in ((0, rm_pre), (1, rm_post)):
            ax = axes[row][col]
            im = ax.imshow(smooth_and_norm(rm[int(nid)]), origin="lower",
                           cmap=cmap, vmin=0, vmax=1)
            arena_axes(ax)
            reward_marker(ax, rx, ry, size=3.6)
            if row == 0:
                ax.set_title(str(nid), fontsize=FS_SMALL, pad=1.5,
                             color=C["muted"], loc="center")
    # Row identity, plain language, horizontal so it is readable at final size.
    for ax, lbl in ((axes[0][0], ylabel_top), (axes[1][0], ylabel_bottom)):
        ax.set_ylabel(lbl, fontsize=FS_TICK, rotation=90, va="center", labelpad=3)
    if title:
        fig.suptitle(title, fontsize=FS_BASE, y=1.015, x=0.005, ha="left")
    plt.tight_layout(pad=0.25, w_pad=0.12, h_pad=0.18)
    if im is not None:                      # one shared scale bar for the grid
        thin_colorbar(fig, im, axes[0][n - 1], label="Rate",
                      ticklabels=("0", "max"), shrink=0.85, pad=0.016,
                      width=0.008)
    return save_figure(fig, stem)


def _draw_arena(ax, reward_pos, title_text: str, show_axes: bool = False):
    """Boxed unit arena with the reward marker; ticks only if `show_axes`."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    for side, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#B0B0B0")
    if show_axes:
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=FS_SMALL, pad=1.5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    reward_marker(ax, reward_pos[0], reward_pos[1], size=7.0)
    title(ax, title_text)


def fig_centers_abc(phases: dict, reward_pos, ids: np.ndarray, stem):
    """Panel b: COM clouds for A, B and C_post; centres are (x, y), so column 0 is plotted on the x axis."""
    R = np.asarray(reward_pos, dtype=float)
    com_a = phases["A"]["com"][ids]
    com_b = phases["B"]["com"][ids]
    com_c = phases["C_post"]["com"][ids]

    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.55))
    specs = [
        (axes[0], com_a, PHASE_COLORS["A"], "Prediction"),
        (axes[1], com_b, PHASE_COLORS["B"], "RL policy"),
        (axes[2], com_c, PHASE_COLORS["C_post"], "Frozen post-RL"),
    ]
    for i, (ax, com, colour, ttl) in enumerate(specs):
        _draw_arena(ax, R, ttl, show_axes=(i == 0))
        ax.scatter(com[:, 0], com[:, 1], s=3.2, alpha=0.42, c=colour,
                   linewidths=0, rasterized=True)
    axes[0].set_xlabel("Arena x", labelpad=1.5)
    axes[0].set_ylabel("Arena y", labelpad=1.5)
    axes[2].text(1.0, -0.06, f"n = {len(ids)} cells", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=FS_SMALL, color=C["muted"])
    fig.tight_layout(pad=0.3, w_pad=0.3)
    return save_figure(fig, stem)


def fig_delta_violins(stats: dict, stem):
    """Panel c: violins of COM shift for A -> B and frozen replay (negative = toward reward)."""
    delta_ab = np.asarray(stats["delta_com_AB"], dtype=float)
    delta_ac = np.asarray(stats["delta_com_AC"], dtype=float)

    fig, ax = plt.subplots(figsize=(2.35, 1.8))
    spec = ((1, delta_ab, "policy", stats["com_mean_delta_AB"], stats["com_frac_toward_AB"]),
            (2, delta_ac, "frozen_post", stats["com_mean_delta_AC"], stats["com_frac_toward_AC"]))
    for pos, vals, role, mean_d, frac in spec:
        style_violin(ax.violinplot([vals], positions=[pos], widths=0.72,
                                   showmeans=True, showmedians=False,
                                   showextrema=True), C[role])
    zero_line(ax)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Policy\nA→B", "Frozen replay\npre→post"])
    ax.set_xlim(0.45, 2.55)
    ax.set_ylabel(r"$\Delta$ distance to reward")

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.34 * (hi - lo))
    for pos, vals, role, mean_d, frac in spec:
        top = float(np.nanmax(vals)) if len(vals) else 0.0
        ax.text(pos, top + 0.06 * (hi - lo),
                f"{mean_d:+.3f}\n{100 * frac:.0f}% toward",
                ha="center", va="bottom", fontsize=FS_SMALL, color=C["muted"],
                linespacing=1.3)
    title(ax, "Field-centre shift")
    fig.tight_layout(pad=0.3)
    return save_figure(fig, stem)


def fig_distance_histograms(stats: dict, stem):
    """Panel d: COM distance histograms, frozen replay (left) and A -> B (right), with paired two-sided Wilcoxon p."""
    from scipy.stats import wilcoxon as _wil

    def _wilcoxon_p(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if len(a) < 5 or np.all(a == b):
            return float("nan")
        try:
            return float(_wil(a, b)[1])
        except Exception:
            return float("nan")

    def _fmt_p(p):
        if not np.isfinite(p):
            return ""
        return f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"

    # COM distances for both contrasts
    d_com_a = np.asarray(stats["com"]["A"], dtype=float)
    d_com_b = np.asarray(stats["com"]["B"], dtype=float)
    d_com_c_pre = np.asarray(stats["com"]["C_pre"], dtype=float)
    d_com_c_post = np.asarray(stats["com"]["C"], dtype=float)

    p_ac = _wilcoxon_p(d_com_c_post, d_com_c_pre)   # paired, same cells, valid
    p_ab = _wilcoxon_p(d_com_b, d_com_a)

    hi = max(
        float(np.nanmax(d_com_a)) if len(d_com_a) else 0,
        float(np.nanmax(d_com_b)) if len(d_com_b) else 0,
        float(np.nanmax(d_com_c_pre)) if len(d_com_c_pre) else 0,
        float(np.nanmax(d_com_c_post)) if len(d_com_c_post) else 0,
        0.15,
    ) * 1.05
    edges = np.linspace(0.0, hi, 21)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 1.5))

    panels = [
        (axes[0], "Frozen replay (weights only)",
         [(d_com_c_pre, "C_pre", "Pre-RL"), (d_com_c_post, "C_post", "Post-RL")],
         stats["com_mean_delta_AC"], stats["com_frac_toward_AC"], p_ac),
        (axes[1], "RL policy (exploration)",
         [(d_com_a, "A", "Prediction"), (d_com_b, "B", "RL policy")],
         stats["com_mean_delta_AB"], stats["com_frac_toward_AB"], p_ab),
    ]
    for ax, ttl, series, mean_d, frac, pval in panels:
        for vals, role, lbl in series:
            hist_outline(ax, vals, edges, PHASE_COLORS[role], label=lbl)
            if len(vals):
                ax.axvline(float(np.mean(vals)), color=PHASE_COLORS[role],
                           ls=(0, (2.5, 2)), lw=0.7, zorder=1)
        ax.set_xlabel("Field-centre distance to reward")
        ax.set_ylabel("Cells")
        title(ax, ttl)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
        stat_annot(ax, f"$\\Delta$d = {mean_d:+.3f}   {100 * frac:.0f}% toward\n"
                       f"{fmt_p(pval if np.isfinite(pval) else None)}",
                   loc="lower right")

    fig.tight_layout(pad=0.3, w_pad=0.4)
    return save_figure(fig, stem)
