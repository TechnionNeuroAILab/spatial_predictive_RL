"""
reward_relocation_analysis.py
Does the map TRACK the reward when it moves A -> B?

The relocation run replays ONE frozen trajectory through three weight snapshots:
  replay_reloc_pred    baseline (pre-RL weights)          -> F0
  replay_reloc_postA   after training at reward A         -> F_A
  replay_reloc_postB   after relocating and training at B -> F_B
Because the trajectory is identical in all three, occupancy is fixed and any
change in the fields is representational.

Tracking signature: mean field-centre distance to A should drop from F0 to F_A
and rise again by F_B, while distance to B should stay high through F_A and drop
at F_B -- i.e. the fields leave A and arrive at B. We use the de-confounded PEAK
centre.

Usage:
  python New_Code/reward_relocation_analysis.py \
      --run-dir New_Results_v2/reward_relocation \
      --output-dir Analysis_Plots/reward_relocation \
      --reward-a-x 0.3 --reward-a-y 0.3 --reward-b-x 0.7 --reward-b-y 0.7
"""
from __future__ import annotations
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_helpers import load_period_centers, dist_to_reward, has_period  # noqa: E402
from common import ensure_dir, save_json  # noqa: E402

PERIODS = [("replay_reloc_pred", "baseline\n(pre-RL)"),
           ("replay_reloc_postA", "after A"),
           ("replay_reloc_postB", "after B\n(relocated)")]


def _wilcoxon_p(a, b):
    if len(a) < 5 or np.all(a == b):
        return float("nan")
    try:
        return float(wilcoxon(a, b)[1])
    except Exception:
        return float("nan")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", "--run_dir", dest="run_dir", required=True)
    ap.add_argument("--output-dir", "--output_dir", dest="output_dir", default="Analysis_Plots/reward_relocation")
    ap.add_argument("--reward-a-x", type=float, default=0.3)
    ap.add_argument("--reward-a-y", type=float, default=0.3)
    ap.add_argument("--reward-b-x", type=float, default=0.7)
    ap.add_argument("--reward-b-y", type=float, default=0.7)
    ap.add_argument("--skaggs-threshold", type=float, default=0.15)
    ap.add_argument("--metric", choices=["peak", "com"], default="peak")
    return ap.parse_args()


def main():
    args = parse_args()
    A = np.array([args.reward_a_x, args.reward_a_y], dtype=float)
    B = np.array([args.reward_b_x, args.reward_b_y], dtype=float)
    thr = args.skaggs_threshold
    data_dir = os.path.join(args.run_dir, "data")
    out_dir = ensure_dir(args.output_dir)
    states = {}
    for period, _ in PERIODS:
        if not has_period(data_dir, period):
            raise FileNotFoundError(f"Missing relocation period '{period}' in {data_dir}. "
                                    f"Did the run use --relocate_at_cycle?")
        sk, com, peak = load_period_centers(data_dir, period)
        centers = peak if args.metric == "peak" else com
        states[period] = {"skaggs": sk, "centers": centers}
    print(f"[reward_relocation] metric={args.metric}  A={A}  B={B}")
    summary = {}
    dA_by_state, dB_by_state = [], []
    for period, label in PERIODS:
        sk = states[period]["skaggs"]
        c = states[period]["centers"]
        place = np.where(sk >= thr)[0]
        dA = dist_to_reward(c[place], A)
        dB = dist_to_reward(c[place], B)
        dA_by_state.append(dA)
        dB_by_state.append(dB)
        summary[period] = {
            "n_place": int(len(place)),
            "mean_dist_to_A": float(dA.mean()),
            "mean_dist_to_B": float(dB.mean()),
        }
        print(f"  {period:20s} n={len(place):3d}  <dist to A>={dA.mean():.4f}  <dist to B>={dB.mean():.4f}")
    def paired_shift(p_from, p_to, reward):
        sk_f, sk_t = states[p_from]["skaggs"], states[p_to]["skaggs"]
        active = np.where(np.maximum(sk_f, sk_t) >= thr)[0]
        d_f = dist_to_reward(states[p_from]["centers"][active], reward)
        d_t = dist_to_reward(states[p_to]["centers"][active], reward)
        d = d_t - d_f
        return float(d.mean()), float((d < 0).mean()), _wilcoxon_p(d_t, d_f), int(len(active))
    # toward A during phase A; toward B during phase B
    shiftA = paired_shift("replay_reloc_pred", "replay_reloc_postA", A)
    shiftB = paired_shift("replay_reloc_postA", "replay_reloc_postB", B)
    print(f"  toward A (pre->postA):  meanΔ={shiftA[0]:+.4f}  {100*shiftA[1]:.0f}% toward  p={shiftA[2]:.1e}")
    print(f"  toward B (postA->postB): meanΔ={shiftB[0]:+.4f}  {100*shiftB[1]:.0f}% toward  p={shiftB[2]:.1e}")
    labels = [lab for _, lab in PERIODS]
    x = np.arange(len(PERIODS))
    mA = [d.mean() for d in dA_by_state]
    sA = [d.std(ddof=1) / np.sqrt(len(d)) for d in dA_by_state]
    mB = [d.mean() for d in dB_by_state]
    sB = [d.std(ddof=1) / np.sqrt(len(d)) for d in dB_by_state]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.errorbar(x, mA, yerr=sA, fmt="-o", lw=2.2, color="#2a9d8f", capsize=4, label="distance to A (0.3, 0.3)")
    ax.errorbar(x, mB, yerr=sB, fmt="-s", lw=2.2, color="#e76f51", capsize=4, label="distance to B (0.7, 0.7)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean field-centre distance")
    ax.set_title("a) Fields leave A and arrive at B\n(frozen trajectory; occupancy fixed)")
    ax.legend()

    ax = axes[1]
    colors = {"replay_reloc_pred": "#999999", "replay_reloc_postA": "#2a9d8f", "replay_reloc_postB": "#e76f51"}
    for period, label in PERIODS:
        sk = states[period]["skaggs"]
        c = states[period]["centers"]
        place = np.where(sk >= thr)[0]
        ax.scatter(c[place, 0], c[place, 1], s=8, alpha=0.35, color=colors[period],
                   edgecolors="none", label=label.replace("\n", " "))
    ax.plot(*A, "*", color="#2a9d8f", markersize=22, markeredgecolor="k", label="reward A")
    ax.plot(*B, "*", color="#e76f51", markersize=22, markeredgecolor="k", label="reward B")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("b) Field centres across states")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Reward relocation: the map tracks the reward from A to B ({args.metric} centre)", y=1.02)
    fig.tight_layout()
    out = os.path.join(out_dir, "reward_relocation_tracking.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    save_json(os.path.join(out_dir, "reward_relocation_summary.json"), {
        "script": "reward_relocation_analysis.py",
        "run_dir": args.run_dir,
        "metric": args.metric,
        "reward_A": A.tolist(),
        "reward_B": B.tolist(),
        "per_state": summary,
        "shift_toward_A_pre_to_postA": {"mean_delta": shiftA[0], "frac_toward": shiftA[1], "wilcoxon_p": shiftA[2], "n": shiftA[3]},
        "shift_toward_B_postA_to_postB": {"mean_delta": shiftB[0], "frac_toward": shiftB[1], "wilcoxon_p": shiftB[2], "n": shiftB[3]},
    })
    print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    main()
