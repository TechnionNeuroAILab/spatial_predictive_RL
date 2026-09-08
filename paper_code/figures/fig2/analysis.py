"""Rate-map and field-center helpers for Figure 2 (frozen replay + A/B/C)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
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

# Figure 1 panel d uses these IDs (cycle_1 vs cycle_1_goal, seed=42, n=8).
# Kept as fallback if recomputation is unavailable.
FIG1_PANEL_D_NEURONS = [510, 364, 442, 334, 227, 229, 47, 400]


def load_period(data_dir: Path, period: str, m: int = M, hidden_dim: int = HIDDEN_DIM):
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
    except Exception:
        return list(FIG1_PANEL_D_NEURONS[:n])


def shared_place_cell_ids(phases: dict, skaggs_threshold: float = 0.05) -> np.ndarray:
    """Place cells gated in A, B, and C (postrl) for the A→B contrast (kept
    name/gate for backward compatibility). The "AC" frozen-replay contrast
    in paired_distance_stats() uses its OWN independent C_pre/C_post gate,
    not this one."""
    mask = (
        (phases["A"]["sk"] >= skaggs_threshold)
        & (phases["B"]["sk"] >= skaggs_threshold)
        & (phases["C_post"]["sk"] >= skaggs_threshold)
    )
    return np.where(mask)[0]


def paired_distance_stats(phases: dict, reward_pos, ids: np.ndarray,
                          ac_skaggs_threshold: float = 0.05, ac_min_cells: int = 10):
    """COM / peak distances and Δd for two DIFFERENT contrasts:

    - A→B: RL policy vs prediction-only exploration. Occupancy is NOT fixed
      here (this is intentional -- it answers "does active RL exploration
      reshape place fields?", confounded with occupancy by design).

    - "AC" (historically buggy; fixed here): now means C_pre -> C_post, i.e.
      replay_c1_pred (frozen trajectory, pre-RL weights) vs replay_c1_postrl
      (the SAME frozen trajectory, post-RL weights). Occupancy is genuinely
      fixed -- only the weights differ -- so this isolates weight-driven
      (H2 / real-remapping) shifts. This replaces the old (bug) definition
      that used A (a real exploratory trajectory) as the "occupancy fixed"
      baseline, which it was not. Key names (delta_com_AC, com_mean_delta_AC,
      ...) are kept for backward compatibility with downstream callers;
      only their contents and the cell population they're computed over
      have changed. See analysis_helpers.frozen_replay_delta_from_centers.
    """
    R = np.asarray(reward_pos, dtype=float)
    com_a = phases["A"]["com"][ids]
    com_b = phases["B"]["com"][ids]
    peak_a = phases["A"]["peak"][ids]
    peak_b = phases["B"]["peak"][ids]

    d_com = {"A": dist_to_reward(com_a, R), "B": dist_to_reward(com_b, R)}
    d_peak = {"A": dist_to_reward(peak_a, R), "B": dist_to_reward(peak_b, R)}

    # --- corrected frozen-replay contrast: C_pre -> C_post (weights only) ---
    # Independently gated on C_pre/C_post Skaggs (NOT on the A/B/ids gate above) --
    # matches figure 5's own frozen-replay diagnostic exactly.
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
    ylabel_top: str = "C: pred\n(frozen traj)",
    ylabel_bottom: str = "C: postrl\n(same traj)",
    cmap: str = "jet",
    title: str | None = None,
):
    """Panel a — remapping grid (Fig. 1d / 2a / 3a style)."""
    n = len(neuron_ids)
    rx, ry = reward_bin
    fig, axes = plt.subplots(2, n, figsize=(n * 1.05, 2.2), squeeze=False)
    for col, nid in enumerate(neuron_ids):
        for row, rm in ((0, rm_pre), (1, rm_post)):
            ax = axes[row][col]
            ax.imshow(smooth_and_norm(rm[int(nid)]), origin="lower", cmap=cmap, vmin=0, vmax=1)
            ax.plot(rx, ry, "w*", markersize=3, markeredgewidth=0.3, markeredgecolor="0.2")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if row == 0:
                ax.set_title(str(nid), fontsize=6, pad=1)
    axes[0][0].set_ylabel(ylabel_top, fontsize=6)
    axes[1][0].set_ylabel(ylabel_bottom, fontsize=6)
    if title is None:
        title = "Frozen replay (Phase C): same cells as Fig. 1d — identical trajectory, pre- vs post-RL weights"
    fig.suptitle(title, fontsize=6, y=1.02)
    plt.tight_layout(pad=0.3, w_pad=0.15, h_pad=0.25)
    return save_figure(fig, stem)


def _draw_arena(ax, reward_pos, title: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.plot(reward_pos[0], reward_pos[1], "*", color="crimson", markersize=7, zorder=5)
    ax.set_title(title, fontsize=6, pad=2)


def fig_centers_abc(phases: dict, reward_pos, ids: np.ndarray, stem):
    """Panel b — COM clouds for Phases A, B, C (shared place cells)."""
    R = np.asarray(reward_pos, dtype=float)
    com_a = phases["A"]["com"][ids]
    com_b = phases["B"]["com"][ids]
    com_c = phases["C_post"]["com"][ids]

    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.45))
    specs = [
        (axes[0], com_a, PHASE_COLORS["A"], f"A  prediction\n(n={len(ids)})"),
        (axes[1], com_b, PHASE_COLORS["B"], f"B  RL policy\n(n={len(ids)})"),
        (axes[2], com_c, PHASE_COLORS["C_post"], f"C  frozen postrl\n(n={len(ids)})"),
    ]
    for ax, com, color, title in specs:
        _draw_arena(ax, R, title)
        ax.scatter(com[:, 0], com[:, 1], s=3, alpha=0.35, c=color, linewidths=0, rasterized=True)
    fig.suptitle("Field centres (COM) across phases", fontsize=6.5, y=1.02)
    fig.tight_layout(pad=0.35, w_pad=0.35)
    return save_figure(fig, stem)


def fig_delta_violins(stats: dict, stem):
    """Panel c — violin plots of COM Δd for A→B (policy) and C:pre→post
    (frozen replay, weight-only) (< 0 = toward reward)."""
    delta_ab = np.asarray(stats["delta_com_AB"], dtype=float)
    delta_ac = np.asarray(stats["delta_com_AC"], dtype=float)

    fig, ax = plt.subplots(figsize=(2.1, 1.6))
    data = [delta_ab, delta_ac]
    parts = ax.violinplot(
        data, positions=[1, 2], showmeans=True, showmedians=False, showextrema=True,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(["#E69F00", "#009E73"][i])
        body.set_alpha(0.65)
        body.set_edgecolor("0.2")
    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        if key in parts:
            parts[key].set_color("0.25")
            parts[key].set_linewidth(0.8)

    ax.axhline(0, color="0.35", lw=0.7, ls="--")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["A→B\n(policy)", "C:pre→post\n(frozen replay)"])
    ax.set_ylabel(r"$\Delta$ COM distance to reward")
    ax.set_title(
        f"Distance change\n"
        f"A→B {stats['com_mean_delta_AB']:+.3f} ({100 * stats['com_frac_toward_AB']:.0f}% toward)\n"
        f"C:pre→post {stats['com_mean_delta_AC']:+.3f} ({100 * stats['com_frac_toward_AC']:.0f}% toward)  [weight-only]",
        fontsize=5.5, pad=2,
    )
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem)


def fig_distance_histograms(stats: dict, stem):
    """Panel d — COM distance histograms for both contrasts (C:pre→post frozen
    replay, A→B policy).

    The C:pre→post contrast (left) uses the SAME frozen trajectory before and
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

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 1.35))

    # Left: C:pre->post COM (frozen replay -- weight change only, no occupancy change).
    axes[0].hist(d_com_c_pre, bins=edges, color=PHASE_COLORS["C_pre"], alpha=0.55, label="C pre-RL (frozen)")
    axes[0].hist(d_com_c_post, bins=edges, color=PHASE_COLORS["C_post"], alpha=0.55, label="C post-RL (frozen, same traj)")
    if len(d_com_c_pre):
        axes[0].axvline(float(np.mean(d_com_c_pre)), color=PHASE_COLORS["C_pre"], ls="--", lw=0.9)
    if len(d_com_c_post):
        axes[0].axvline(float(np.mean(d_com_c_post)), color=PHASE_COLORS["C_post"], ls="--", lw=0.9)
    axes[0].set_xlabel("COM distance to reward")
    axes[0].set_ylabel("Neurons")
    axes[0].set_title(
        f"C:pre→post COM (frozen replay)\n"
        f"mean Δd = {stats['com_mean_delta_AC']:+.3f}, "
        f"{100 * stats['com_frac_toward_AC']:.0f}% toward  {_fmt_p(p_ac)}",
        fontsize=6, pad=2,
    )
    axes[0].legend(frameon=False, loc="upper right")

    # Right: A–B COM (policy / exploration).
    axes[1].hist(d_com_a, bins=edges, color=PHASE_COLORS["A"], alpha=0.55, label="A prediction")
    axes[1].hist(d_com_b, bins=edges, color=PHASE_COLORS["B"], alpha=0.55, label="B RL policy")
    if len(d_com_a):
        axes[1].axvline(float(np.mean(d_com_a)), color=PHASE_COLORS["A"], ls="--", lw=0.9)
    if len(d_com_b):
        axes[1].axvline(float(np.mean(d_com_b)), color=PHASE_COLORS["B"], ls="--", lw=0.9)
    axes[1].set_xlabel("COM distance to reward")
    axes[1].set_ylabel("Neurons")
    axes[1].set_title(
        f"A–B COM (policy / exploration)\n"
        f"mean Δd = {stats['com_mean_delta_AB']:+.3f}, "
        f"{100 * stats['com_frac_toward_AB']:.0f}% toward  {_fmt_p(p_ab)}",
        fontsize=6, pad=2,
    )
    axes[1].legend(frameon=False, loc="upper right")

    fig.suptitle(
        "COM distance to reward — C:pre→post frozen replay (left) vs A–B RL policy (right)",
        fontsize=6, y=1.02,
    )
    fig.tight_layout(pad=0.35, w_pad=0.4)
    return save_figure(fig, stem)
