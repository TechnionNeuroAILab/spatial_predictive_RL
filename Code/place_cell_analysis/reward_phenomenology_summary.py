"""
Cross-condition dissociation summary for the three reward phenomena.

Reads the reward_phenomenology.py stats already written into
Positive_Results/data/statistics_summary.json and
Negative_Results/data/statistics_summary.json, and renders a single
publication-style figure contrasting the GENUINE (frozen-replay, occupancy-
controlled) pre->post change in the reward-head model (Positive) against the
task-RL control (Negative), for:

    1. concentration toward reward   (field-peak distance-to-reward shift)
    3. reward-distance ramping       (ramping-index shift)
    4. goal-direction tuning         (goal-direction MRL shift, with the
                                      allocentric head-direction shift as a
                                      specificity control)

The metric plotted is the pre->post DELTA, which -- unlike the raw indices --
is comparable across the two conditions despite their different reward
locations ((0.3,0.3) off-centre vs (0.5,0.5) centre).
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from pc_common import save_all


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-condition dissociation summary figure for the three reward "
                    "phenomena (reads reward_phenomenology.py stats from two run dirs)."
    )
    p.add_argument("--positive_dir", default="Positive_Results",
                   help="Run dir for the reward-head (genuine remapping) condition.")
    p.add_argument("--negative_dir", default="Negative_Results",
                   help="Run dir for the task-RL (control) condition.")
    p.add_argument("--positive_label", default="Reward-head\n(genuine)",
                   help="Legend/axis label for the positive condition.")
    p.add_argument("--negative_label", default="Task-RL\n(control)",
                   help="Legend/axis label for the negative condition.")
    p.add_argument("--out_dir", default="summary_figures",
                   help="Directory to write the summary figure triple into.")
    return p.parse_args()


def load_records(run_dir):
    recs = json.load(open(os.path.join(run_dir, "data", "statistics_summary.json")))
    out = {}
    for r in recs:
        if r.get("script") == "reward_phenomenology.py":
            out[r["test"]] = r
    return out


def _stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def main():
    args = parse_args()
    CONDS = [(args.positive_dir, args.positive_label, "crimson"),
             (args.negative_dir, args.negative_label, "steelblue")]
    data = {d: load_records(d) for d, _, _ in CONDS}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: concentration -- field-peak distance-to-reward shift (post-pre)
    ax = axes[0]
    for i, (d, lab, col) in enumerate(CONDS):
        r = data[d]["over_representation_near_reward"]
        delta = r["mean_field_dist_post"] - r["mean_field_dist_pre"]
        ax.bar(i, delta, color=col)
        ax.text(i, delta, f"{delta:+.3f}\n{_stars(r.get('field_dist_mannwhitney_p'))}",
                ha="center", va="top" if delta < 0 else "bottom", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([c[1] for c in CONDS])
    ax.set_ylabel("Δ mean field distance to reward")
    ax.set_title("1. Concentration toward reward\n(negative = fields move to reward)")

    # Panel 2: ramping-index shift
    ax = axes[1]
    for i, (d, lab, col) in enumerate(CONDS):
        r = data[d]["reward_distance_ramping"]
        delta = r["mean_ramping_post"] - r["mean_ramping_pre"]
        ax.bar(i, delta, color=col)
        ax.text(i, delta, f"{delta:+.4f}\n{_stars(r.get('wilcoxon_p'))}",
                ha="center", va="top" if delta < 0 else "bottom", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([c[1] for c in CONDS])
    ax.set_ylabel("Δ ramping index")
    ax.set_title("3. Reward-distance ramping\n(frozen replay: null = behavioural overlay)")

    # Panel 3: goal-direction vs head-direction MRL shift (grouped)
    ax = axes[2]
    width = 0.35
    for i, (d, lab, col) in enumerate(CONDS):
        r = data[d]["goal_direction_tuning"]
        dg = r["mean_mrl_post"] - r["mean_mrl_pre"]
        dh = r["mean_headdir_mrl_post"] - r["mean_headdir_mrl_pre"]
        ax.bar(i - width / 2, dg, width, color=col, label="goal-dir" if i == 0 else None)
        ax.bar(i + width / 2, dh, width, color=col, alpha=0.45, hatch="//",
               label="head-dir (control)" if i == 0 else None)
        ax.text(i - width / 2, dg, f"{dg:+.3f}\n{_stars(r.get('wilcoxon_p'))}",
                ha="center", va="bottom" if dg >= 0 else "top", fontsize=8)
        ax.text(i + width / 2, dh, f"{dh:+.3f}", ha="center",
                va="bottom" if dh >= 0 else "top", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([c[1] for c in CONDS])
    ax.set_ylabel("Δ tuning strength (MRL)")
    ax.set_title("4. Goal-direction tuning\n(head-dir co-rises: fixed goal can't fully dissociate)")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Reward phenomenology: genuine (frozen-replay) change, reward-head model vs task-RL control",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    save_all(os.path.join(out_dir, "reward_phenomenology_dissociation"))
    plt.close(fig)
    print(f"Wrote {out_dir}/reward_phenomenology_dissociation.(png|jpg|pdf)")

    # Console dissociation table
    print("\n=== Dissociation summary (pre->post delta, frozen replay) ===")
    for d, lab, _ in CONDS:
        o = data[d]["over_representation_near_reward"]
        rr = data[d]["reward_distance_ramping"]
        g = data[d]["goal_direction_tuning"]
        print(f"\n{d}:")
        print(f"  field-dist to reward: {o['mean_field_dist_pre']:.3f} -> {o['mean_field_dist_post']:.3f}  "
              f"(Δ {o['mean_field_dist_post']-o['mean_field_dist_pre']:+.3f}, MW p={o.get('field_dist_mannwhitney_p'):.2g})")
        print(f"  ramping index:        {rr['mean_ramping_pre']:+.3f} -> {rr['mean_ramping_post']:+.3f}  "
              f"(Δ {rr['mean_ramping_post']-rr['mean_ramping_pre']:+.4f}, p={rr.get('wilcoxon_p'):.2g})")
        print(f"  goal-dir MRL:         {g['mean_mrl_pre']:.3f} -> {g['mean_mrl_post']:.3f}  "
              f"(Δ {g['mean_mrl_post']-g['mean_mrl_pre']:+.3f}, p={g.get('wilcoxon_p'):.2g})  "
              f"| head-dir Δ {g['mean_headdir_mrl_post']-g['mean_headdir_mrl_pre']:+.3f}")


if __name__ == "__main__":
    main()
