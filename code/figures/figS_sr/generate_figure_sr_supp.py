#!/usr/bin/env python3
"""
Supplementary Figure — what a successor representation (SR) does and does not
account for, in the 55 matched networks of the main text
(analysis_helpers.campaign_passers("paired")).

The SR of a policy is M = (I - gamma T)^-1 with T the empirical transition
matrix of the trajectory (sr_common: 32 x 32 grid, gamma = 0.99). The
exploration SR is the random-walk SR shared by every network; the
goal-directed SR is built from each network's own goal-directed trajectory.
A successor field is a column of M; its centre is the M-weighted mean state.

Panels:
  a  Behavioural concentration: mean distance to reward of successor-field
     centres (exploration vs goal-directed SR) beside the recurrent units'
     field centres (exploration vs goal-directed probe)
  b  Fixed-trajectory replay: the trajectory-derived SR is identical before
     and after learning (predicted shift 0), whereas the recurrent fields
     shift toward the reward and not toward control locations

Usage:
    python code/figures/figS_sr/generate_figure_sr_supp.py --no_show
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "code" / "lib"))

from utils import HEAD_DIR, NOHEAD_DIR, C, FS_SMALL, FS_TICK, apply_nature_style, ensure_dir, fmt_p, save_figure, stat_annot, title  # noqa: E402
import figlayout  # noqa: E402
from analysis_helpers import campaign_passers, load_period_centers_cached  # noqa: E402
from figure_cache import add_plot_only, load_figure_data, require_keys, update_figure_data  # noqa: E402
from pubstyle import mean_sem_marker, network_strip, sem, zero_line  # noqa: E402
from sr_common import GAMMA, M, WB, get_shared_random  # noqa: E402
from successor_representation import (  # noqa: E402
    angle_between_vectors, compute_sr_matrix, nearest_state_for_points,
    sr_predicted_shift_vectors, transition_from_behavior_positions,
)

apply_nature_style()

FIGURES_DIR = HERE / "figures_supp"
REWARD = np.array([0.3, 0.3])
SKAGGS = 0.3
# exploration before learning -> goal-directed after (reset transients dropped)
EXPLORE, GOAL = "replay_c1_pred", "cycle_1_goal"
FRZ = ("replay_c1_pred", "replay_c1_postrl")
LOCATIONS = {"Reward": np.array([0.3, 0.3]), "(0.7, 0.3)": np.array([0.7, 0.3]),
             "(0.3, 0.7)": np.array([0.3, 0.7]), "(0.7, 0.7)": np.array([0.7, 0.7])}
CONDS = {"nh": (NOHEAD_DIR, "nohead_", C["nohead"], "Without teaching"),
         "hd": (HEAD_DIR, "head_", C["head"], "With teaching")}

LAYOUT = figlayout.Layout(
    [("a", 0, 0, 3), ("b", 0, 3, 4)],
    ncols=7, row_h_in=[1.95], caption_lines=5,
    name="figS_sr",
)


def _field_centres(Msr, state_xy):
    """Centre of each successor field (column of M: where unit j is active)."""
    w = np.clip(Msr, 0.0, None)
    return (w.T @ state_xy) / w.sum(axis=0)[:, None]


def _occupancy_centres(Msr, state_xy):
    """Centre of each state's expected future occupancy (row of M)."""
    w = np.clip(Msr, 0.0, None)
    return (w @ state_xy) / w.sum(axis=1)[:, None]


def sr_row(job) -> dict | None:
    """SR and recurrent-field statistics for one network of one condition."""
    cond, seed = job
    root, prefix, _, _ = CONDS[cond]
    data = Path(root) / f"{prefix}seed{seed}" / "data"
    _, SR_rand, state_xy = get_shared_random()
    pos = pd.read_parquet(data / f"df_behavior_{GOAL}.parquet")[["x", "y"]].to_numpy(float)
    # The teleport after each goal arrival is kept: it is how the episodic
    # policy restarts, so the SR is that of the policy with restarts. Dropping
    # it leaves the goal with no recorded exit, which the transition builder
    # then fills with random-walk moves -- an artefact, unlike the activity
    # transients after a reset, which are excluded from the rate maps.
    T_pol, _ = transition_from_behavior_positions(pos, M, WB)
    SR_pol, _, _ = compute_sr_matrix(T_pol, GAMMA, "auto", 500, 1e10)
    row = {"cond": cond, "seed": seed}
    row["sr_dist_explore"] = float(np.linalg.norm(_field_centres(SR_rand, state_xy) - REWARD, axis=1).mean())
    row["sr_dist_goal"] = float(np.linalg.norm(_field_centres(SR_pol, state_xy) - REWARD, axis=1).mean())
    row["occ_dist_explore"] = float(np.linalg.norm(_occupancy_centres(SR_rand, state_xy) - REWARD, axis=1).mean())
    row["occ_dist_goal"] = float(np.linalg.norm(_occupancy_centres(SR_pol, state_xy) - REWARD, axis=1).mean())

    sk_e, com_e, _ = load_period_centers_cached(str(data), EXPLORE)
    sk_g, com_g, _ = load_period_centers_cached(str(data), GOAL, trim_resets=True)
    gate = (sk_e >= SKAGGS) & (sk_g >= SKAGGS)
    row["rnn_dist_explore"] = float(np.linalg.norm(com_e[gate] - REWARD, axis=1).mean())
    row["rnn_dist_goal"] = float(np.linalg.norm(com_g[gate] - REWARD, axis=1).mean())

    pred = sr_predicted_shift_vectors(SR_rand, SR_pol, state_xy)
    pred_cell = pred[nearest_state_for_points(com_e[gate], state_xy)]
    ang = angle_between_vectors(com_g[gate] - com_e[gate], pred_cell)
    ang = ang[np.isfinite(ang)]
    row["angles_deg"] = np.degrees(ang)
    row["V"] = float(np.mean(np.cos(ang)))

    sk0, com0, pk0 = load_period_centers_cached(str(data), FRZ[0])
    sk1, com1, pk1 = load_period_centers_cached(str(data), FRZ[1])
    g = (sk0 >= SKAGGS) & (sk1 >= SKAGGS)
    for name, L in LOCATIONS.items():
        row[f"replay_{name}"] = float((np.linalg.norm(pk1[g] - L, axis=1)
                                       - np.linalg.norm(pk0[g] - L, axis=1)).mean())
    return row


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def fig_concentration(rows, stem, size_in):
    """Exploration vs goal-directed: SR future occupancy, SR successor fields, recurrent fields."""
    fig, ax = plt.subplots(figsize=size_in)
    groups = (("occ", 0.0, "SR future\noccupancy"), ("sr", 2.6, "SR successor\nfields"),
              ("rnn", 5.2, "Recurrent\nunit fields"))
    for k, (cond, off) in enumerate((("nh", -0.17), ("hd", 0.17))):
        rr = [r for r in rows if r["cond"] == cond]
        col = CONDS[cond][2]
        for g, x0, _ in groups:
            vals = [np.array([r[f"{g}_dist_{p}"] for r in rr]) for p in ("explore", "goal")]
            network_strip(ax, vals, [x0 + off, x0 + 1 + off], [col, col], jitter=0.04,
                          size=2.5, seed=k)
    ticks = [x for _, x0, _ in groups for x in (x0, x0 + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Expl.", "Goal"] * 3)
    for _, x0, lbl in groups:
        ax.text(x0 + 0.5, -0.17, lbl, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=FS_SMALL, linespacing=1.1)
    ax.set_xlim(-0.6, 6.8)
    ax.set_ylabel("Centre distance to reward")
    title(ax, "Concentration toward reward")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def fig_angles(rows, stem, size_in):
    fig, ax = plt.subplots(figsize=size_in)
    edges = np.linspace(0, 180, 19)
    for cond in ("nh", "hd"):
        a = np.concatenate([r["angles_deg"] for r in rows if r["cond"] == cond])
        h, _ = np.histogram(a, edges)
        dens = h / h.sum() / np.diff(edges)
        ax.stairs(dens, edges, color=CONDS[cond][2], lw=0.9,
                  label=f"{CONDS[cond][3]} ({len(a)} cells)")
    ax.axhline(1 / 180, color=C["muted"], lw=0.5, ls=(0, (2.5, 2)))
    ax.text(178, 1 / 180, "no relationship", fontsize=FS_SMALL, color=C["muted"],
            ha="right", va="bottom")
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Angle between observed and SR-predicted shift (deg)")
    ax.set_ylabel("Density (per deg)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(loc="upper right")
    title(ax, "Direction of behavioural shifts")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def fig_alignment(rows, stem, size_in):
    fig, ax = plt.subplots(figsize=size_in)
    vals = [np.array([r["V"] for r in rows if r["cond"] == c]) for c in ("nh", "hd")]
    network_strip(ax, vals, [0, 1], [CONDS["nh"][2], CONDS["hd"][2]])
    zero_line(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Without\nteaching", "With\nteaching"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Alignment V (mean cos \u03b8)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -0.05), hi + 0.3 * (hi - lo))
    for x, v in zip((0, 1), vals):
        ax.text(x, hi + 0.05 * (hi - lo), f"{v.mean():.2f} ± {sem(v):.2f}\n{fmt_p(wilcoxon(v).pvalue)}",
                ha="center", va="bottom", fontsize=FS_SMALL, color=C["muted"], linespacing=1.3)
    title(ax, "Alignment per network")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def fig_replay_vs_sr(rows, stem, size_in):
    fig, ax = plt.subplots(figsize=size_in)
    names = list(LOCATIONS)
    for k, (cond, off) in enumerate((("nh", -0.18), ("hd", 0.18))):
        rr = [r for r in rows if r["cond"] == cond]
        vals = [np.array([r[f"replay_{n}"] for r in rr]) for n in names]
        xs = np.arange(len(names)) + off
        network_strip(ax, vals, list(xs), [CONDS[cond][2]] * len(names), pair=False,
                      jitter=0.05, size=3, seed=k)
        ax.plot([], [], "o", color=CONDS[cond][2], ms=3, label=CONDS[cond][3])
    ax.axhline(0, color="#1A1A1A", lw=0.8, zorder=1, label="SR prediction (same trajectory)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["Reward\n(0.3, 0.3)", "(0.7, 0.3)", "(0.3, 0.7)", "(0.7, 0.7)"])
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel(r"$\Delta$ peak distance (replay)")
    ax.set_xlabel("Location the distance is measured to")
    ax.legend(loc="lower right", ncol=3)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.12 * (hi - lo), hi)
    title(ax, "Replay shift vs SR prediction")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=str(FIGURES_DIR))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    add_plot_only(p)
    return p.parse_args()


def summarize(rows) -> dict:
    out = {}
    for cond in ("nh", "hd"):
        rr = [r for r in rows if r["cond"] == cond]
        s = {}
        for key in ("occ_dist_explore", "occ_dist_goal", "sr_dist_explore", "sr_dist_goal",
                    "rnn_dist_explore", "rnn_dist_goal", "V",
                    *[f"replay_{n}" for n in LOCATIONS]):
            v = np.array([r[key] for r in rr])
            s[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "sem": sem(v),
                      "p_vs_0": float(wilcoxon(v).pvalue) if np.ptp(v) > 0 else None}
        a = np.concatenate([r["angles_deg"] for r in rr])
        s["n_cells_angles"] = int(len(a))
        s["pct_angle_below_90"] = float(100 * (a < 90).mean())
        s["p_goal_vs_explore_sr"] = float(wilcoxon([r["sr_dist_goal"] - r["sr_dist_explore"] for r in rr]).pvalue)
        s["n"] = len(rr)
        out[cond] = s
    return out


def main():
    args = parse_args()
    out = Path(args.output_dir)
    panels_dir = ensure_dir(out / "panels")
    cache_path = out / "figure_data.npz"
    if args.plot_only:
        rows = require_keys(load_figure_data(cache_path), "rows", path=cache_path)["rows"]
    else:
        seeds = sorted(campaign_passers("paired"))
        jobs = [(c, s) for c in ("nh", "hd") for s in seeds]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = [r for r in pool.map(sr_row, jobs) if r]
        update_figure_data(cache_path, {"rows": rows})
    import json
    summ = summarize(rows)
    (panels_dir / "sr_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

    S = LAYOUT.slot_size_in
    panel_pdfs = {
        "a": fig_concentration(rows, panels_dir / "panel_a", S("a"))[1],
        "b": fig_replay_vs_sr(rows, panels_dir / "panel_b", S("b"))[1],
    }
    if not args.no_assemble:
        figlayout.compose(LAYOUT, panel_pdfs, out / "figure_sr_supp_raw")
    print(f"\nDone. Outputs: {out}")


if __name__ == "__main__":
    main()
