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


def load_period(data_dir: Path, period: str, m: int = M, hidden_dim: int = HIDDEN_DIM,
                trim_resets: bool = False):
    """Rate maps, occupancy and Skaggs scores for one period's pickles."""
    from analysis_helpers import load_period_frames
    beh, act = load_period_frames(str(data_dir), period, hidden_dim, trim_resets=trim_resets)
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
        # the goal-directed probe resets after every goal arrival
        rm, occ, sk = load_period(data_dir, period, trim_resets=(key == "B"))
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
    ylabel_top: str = "Before",
    ylabel_bottom: str = "After",
    cmap=RATEMAP,
    title: str | None = None,
    size_in: tuple[float, float] | None = None,
):
    """Panel a: pre / post rate maps of the example cells."""
    n = len(neuron_ids)
    rx, ry = reward_bin
    fig, axes = plt.subplots(2, n, figsize=size_in or (n * 1.05, 2.05), squeeze=False)
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
    # Reserve the right-hand strip thin_colorbar draws into: under an
    # exact-size save it would otherwise be clipped off the page edge.
    fig.tight_layout(rect=(0, 0, 0.915, 1), pad=0.25, w_pad=0.12, h_pad=0.18)
    if im is not None:                      # one shared scale bar for the grid
        thin_colorbar(fig, im, axes[0][n - 1], label="Rate",
                      ticklabels=("0", "max"), shrink=0.85, pad=0.016,
                      width=0.008)
    return save_figure(fig, stem, exact_size=size_in is not None)


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


def _center_key(center: str) -> str:
    """'com' (centre of mass) or 'peak' (argmax of the smoothed map)."""
    key = str(center).strip().lower()
    if key not in ("com", "peak"):
        raise ValueError(f"center must be 'com' or 'peak', not {center!r}")
    return key


def _centre_word(center: str) -> str:
    """Axis wording. The COM label stays the hyphenated string already in print."""
    return "Field-centre" if _center_key(center) == "com" else "Field-peak"


def fig_centers_abc(phases: dict, reward_pos, ids: np.ndarray, stem,
                    size_in: tuple[float, float] | None = None,
                    center: str = "com"):
    """Panel b — field-centre clouds for Phases A, B, C (shared place cells).

    `center` selects centre of mass (`com`, the manuscript default) or the
    field peak (`peak`).
    """
    key = _center_key(center)
    R = np.asarray(reward_pos, dtype=float)
    xy_a = phases["A"][key][ids]
    xy_b = phases["B"][key][ids]
    xy_c = phases["C_post"][key][ids]

    fig, axes = plt.subplots(1, 3, figsize=size_in or (5.4, 1.55))
    specs = [
        (axes[0], xy_a, PHASE_COLORS["A"], "Exploration, before"),
        (axes[1], xy_b, PHASE_COLORS["B"], "Goal-directed, after"),
        (axes[2], xy_c, PHASE_COLORS["C_post"], "Replay, after learning"),
    ]
    for i, (ax, xy, colour, ttl) in enumerate(specs):
        _draw_arena(ax, R, ttl, show_axes=(i == 0))
        ax.scatter(xy[:, 0], xy[:, 1], s=3.2, alpha=0.42, c=colour,
                   linewidths=0, rasterized=True)
    axes[0].set_xlabel("Arena x", labelpad=1.5)
    axes[0].set_ylabel("Arena y", labelpad=1.5)
    axes[2].text(1.0, -0.06, f"n = {len(ids)} cells", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=FS_SMALL, color=C["muted"])
    fig.tight_layout(pad=0.3, w_pad=0.3)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_delta_violins(stats: dict, stem,
                      size_in: tuple[float, float] | None = None,
                      center: str = "com"):
    """Panel c — violin plots of Δd for A→B (policy) and C:pre→post
    (frozen replay, weight-only) (< 0 = toward reward).

    `center` selects the centre-of-mass shift (default) or the field-peak shift.
    """
    key = _center_key(center)
    delta_ab = np.asarray(stats[f"delta_{key}_AB"], dtype=float)
    delta_ac = np.asarray(stats[f"delta_{key}_AC"], dtype=float)

    fig, ax = plt.subplots(figsize=size_in or (2.35, 1.8))
    spec = ((1, delta_ab, "policy", stats[f"{key}_mean_delta_AB"], stats[f"{key}_frac_toward_AB"]),
            (2, delta_ac, "frozen_post", stats[f"{key}_mean_delta_AC"], stats[f"{key}_frac_toward_AC"]))
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
    title(ax, f"{_centre_word(center)} shift")
    fig.tight_layout(pad=0.3)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_distance_histograms(stats: dict, stem,
                            size_in: tuple[float, float] | None = None,
                            center: str = "com"):
    """Panel d — distance histograms for both contrasts (C:pre→post frozen
    replay, A→B policy). `center` selects centre of mass or field peak.

    The C:pre→post contrast (left) uses the same frozen trajectory before and
    after RL, so it tests whether weight changes alone shift fields (occupancy
    genuinely fixed). The A–B contrast (right) tests whether active RL
    exploration shifts them (occupancy-confounded by design).
    """
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
        # Route to the shared formatter so every panel in the paper writes a
        # p-value the same way (and none of them use a mathtext superscript,
        # which prints at 0.7x and falls under the 5 pt floor).
        return fmt_p(p if np.isfinite(p) else None)

    key = _center_key(center)
    block = stats[key]
    d_a = np.asarray(block["A"], dtype=float)
    d_b = np.asarray(block["B"], dtype=float)
    d_c_pre = np.asarray(block["C_pre"], dtype=float)
    d_c_post = np.asarray(block["C"], dtype=float)

    p_ac = _wilcoxon_p(d_c_post, d_c_pre)   # paired, same cells, valid
    p_ab = _wilcoxon_p(d_b, d_a)

    hi = max(
        float(np.nanmax(d_a)) if len(d_a) else 0,
        float(np.nanmax(d_b)) if len(d_b) else 0,
        float(np.nanmax(d_c_pre)) if len(d_c_pre) else 0,
        float(np.nanmax(d_c_post)) if len(d_c_post) else 0,
        0.15,
    ) * 1.05
    edges = np.linspace(0.0, hi, 21)

    fig, axes = plt.subplots(1, 2, figsize=size_in or (6.2, 1.5))

    panels = [
        (axes[0], "Frozen replay (weights only)",
         [(d_c_pre, "C_pre", "Pre-RL"), (d_c_post, "C_post", "Post-RL")],
         stats[f"{key}_mean_delta_AC"], stats[f"{key}_frac_toward_AC"], p_ac),
        (axes[1], "RL policy (exploration)",
         [(d_a, "A", "Prediction"), (d_b, "B", "RL policy")],
         stats[f"{key}_mean_delta_AB"], stats[f"{key}_frac_toward_AB"], p_ab),
    ]
    for ax, ttl, series, mean_d, frac, pval in panels:
        for vals, role, lbl in series:
            hist_outline(ax, vals, edges, PHASE_COLORS[role], label=lbl)
            if len(vals):
                ax.axvline(float(np.mean(vals)), color=PHASE_COLORS[role],
                           ls=(0, (2.5, 2)), lw=0.7, zorder=1)
        ax.set_xlabel(f"{_centre_word(center)} distance to reward")
        ax.set_ylabel("Cells")
        title(ax, ttl)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
        stat_annot(ax, f"$\\Delta$d = {mean_d:+.3f}   {100 * frac:.0f}% toward\n"
                       f"{fmt_p(pval if np.isfinite(pval) else None)}",
                   loc="lower right")

    fig.tight_layout(pad=0.3, w_pad=0.4)
    return save_figure(fig, stem, exact_size=size_in is not None)


def _paired_bracket(ax, x0, x1, y, text):
    """Thin bracket between two groups with the paired-test label above it."""
    h = 0.012 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color="#555555", lw=0.5,
            clip_on=False)
    ax.text((x0 + x1) / 2, y + h, text, ha="center", va="bottom",
            fontsize=FS_SMALL, color="#333333")


def fig_network_shifts(cond: str, stem, size_in: tuple[float, float] | None = None):
    """Panel c: per-network mean shift for the goal-directed contrast and for
    fixed-trajectory replay, field peak and centre of mass, in the 50 retained
    networks of one condition (`cond` = "nh" without, "hd" with proximity
    teaching). Grey lines pair a network's two readouts; diamonds are
    mean +/- s.e.m.; brackets carry the paired Wilcoxon P.
    """
    from pubstyle import METRIC_ALPHA, load_headline_stats, network_strip
    from scipy.stats import wilcoxon as _wil

    hs = load_headline_stats()["paired"]
    seeds = set(hs["cohort"]["seeds"])
    rows = [r for r in hs["rows"] if r["seed"] in seeds]
    fig, ax = plt.subplots(figsize=size_in or (2.35, 1.8))
    groups = [("peak", 1.0), ("com", 3.0)]
    tops = []
    for metric, x0 in groups:
        beh = np.array([r[f"{cond}_beh_{metric}"]["mean"] for r in rows])
        frz = np.array([r[f"{cond}_frz_{metric}"]["mean"] for r in rows])
        a = METRIC_ALPHA[metric]
        network_strip(ax, [beh, frz], [x0, x0 + 1.0],
                      [C["policy"], C["frozen_post"]], alpha=0.55 * a + 0.1)
        tops.append((x0, float(max(beh.max(), frz.max())), _wil(beh, frz).pvalue,
                     frz.mean() / beh.mean()))
    zero_line(ax)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Goal-\ndirected", "Replay", "Goal-\ndirected", "Replay"])
    ax.set_xlim(0.4, 4.6)
    ax.set_ylabel(r"$\Delta$ distance to reward")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.32 * (hi - lo))
    for x0, top, p, kept in tops:
        _paired_bracket(ax, x0, x0 + 1.0, top + 0.06 * (hi - lo),
                        f"{fmt_p(p)}\nreplay keeps {100 * kept:.0f}%")
    for metric, x0 in groups:
        ax.text(x0 + 0.5, -0.30, "Peak" if metric == "peak" else "Centre of mass",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=FS_TICK)
    title(ax, f"Across networks (n = {len(rows)})")
    fig.tight_layout(pad=0.3)
    return save_figure(fig, stem, exact_size=size_in is not None)


def fig_replay_histograms(stats: dict, stem, size_in: tuple[float, float] | None = None):
    """Panel d: distance-to-reward distributions of the example network's cells
    on the replayed trajectory before and after learning, for the field peak
    (left) and centre of mass (right), with the per-cell paired Wilcoxon P."""
    from scipy.stats import wilcoxon as _wil

    fig, axes = plt.subplots(1, 2, figsize=size_in or (3.2, 1.8))
    for ax, key, ttl in ((axes[0], "peak", "Peak"), (axes[1], "com", "Centre of mass")):
        pre = np.asarray(stats[key]["C_pre"], dtype=float)
        post = np.asarray(stats[key]["C"], dtype=float)
        hi = max(float(pre.max()), float(post.max()), 0.15) * 1.05
        edges = np.linspace(0.0, hi, 21)
        headroom = 1.8
        for vals, role, lbl in ((pre, "C_pre", "Before"), (post, "C_post", "After")):
            hist_outline(ax, vals, edges, PHASE_COLORS[role], label=lbl)
            ax.axvline(float(vals.mean()), ymax=1.0 / headroom, color=PHASE_COLORS[role],
                       ls=(0, (2.5, 2)), lw=0.7, zorder=1)
        ax.set_ylim(0, ax.get_ylim()[1] * headroom)
        delta = post - pre
        p = _wil(post, pre).pvalue if len(pre) >= 5 and not np.all(delta == 0) else None
        ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.1)
        ax.text(0.02, 0.76, f"$\\Delta$d = {delta.mean():+.3f}, {100 * (delta < 0).mean():.0f}% toward\n"
                            f"{fmt_p(p)}, n = {len(delta)}",
                transform=ax.transAxes, fontsize=FS_SMALL, color=C["muted"],
                ha="left", va="top", linespacing=1.35)
        ax.set_xlabel("Distance to reward")
        title(ax, ttl)
    axes[0].set_ylabel("Cells")
    fig.tight_layout(pad=0.3, w_pad=0.5)
    return save_figure(fig, stem, exact_size=size_in is not None)
