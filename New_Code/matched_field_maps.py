from __future__ import annotations
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_goal_directed_maps import build_rate_maps, compute_skaggs, smooth_and_norm  # noqa: E402
from plot_replay_v11 import field_peak  # noqa: E402
from common import ensure_dir  # noqa: E402

PRE, POST = "replay_c1_pred", "replay_c1_postrl"


def load_rm(data_dir, period, m=32, hidden_dim=512):
    beh = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{period}.pkl"))
    act = pd.read_pickle(os.path.join(data_dir, f"df_activations_{period}.pkl"))
    rm, occ = build_rate_maps(beh, act, m, hidden_dim)
    return rm, compute_skaggs(rm, occ)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head-dir", required=True)
    ap.add_argument("--nohead-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reward-x", type=float, default=0.3)
    ap.add_argument("--reward-y", type=float, default=0.3)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--skaggs-threshold", type=float, default=1.0,help="cell must be a place cell at this level in all three states")
    ap.add_argument("--m", type=int, default=32)
    return ap.parse_args()


def main():
    args = parse_args()
    R = np.array([args.reward_x, args.reward_y]); m = args.m
    out_dir = ensure_dir(args.output_dir)
    rm_pre, sk_pre = load_rm(os.path.join(args.head_dir, "data"), PRE, m)
    rm_h, sk_h = load_rm(os.path.join(args.head_dir, "data"), POST, m)
    rm_n, sk_n = load_rm(os.path.join(args.nohead_dir, "data"), POST, m)
    # sanity: the two runs must share the identical pre-RL map (the RNG fix)
    rm_pre_n, _ = load_rm(os.path.join(args.nohead_dir, "data"), PRE, m)
    shared = np.allclose(rm_pre, rm_pre_n)
    print(f"[matched_field_maps] pre-RL maps identical across the pair: {shared}")
    if not shared:
        print("  WARNING: pre-RL maps differ -- the pair is NOT matched; figure would mislead.")

    peaks = {k: np.array([field_peak(smooth_and_norm(v[n])) for n in range(v.shape[0])])
             for k, v in (("pre", rm_pre), ("head", rm_h), ("nohead", rm_n))}
    d = {k: np.linalg.norm(p - R[None, :], axis=1) for k, p in peaks.items()}

    place = (sk_pre >= args.skaggs_threshold) & (sk_h >= args.skaggs_threshold) & (sk_n >= args.skaggs_threshold)
    head_toward = d["pre"] - d["head"]
    minsk = np.minimum.reduce([sk_pre, sk_h, sk_n])
    # rank example cells by head-driven movement among solid place cells
    score = np.where(place & (head_toward > 0), head_toward + 0.1 * minsk, -np.inf)
    sel = np.argsort(score)[::-1][:args.n_examples]
    n_ok = int(np.isfinite(score).sum())
    print(f"[matched_field_maps] {n_ok} place cells moved toward reward under the head; "
          f"showing {len(sel)} examples")

    Rx, Ry = R[0] * m, R[1] * m
    cols = [("PRE (shared)", rm_pre), ("POST  +head", rm_h), ("POST  no head", rm_n)]
    n = len(sel)
    fig, axes = plt.subplots(n, 3, figsize=(6.6, 2.15 * n), squeeze=False)
    for row, nid in enumerate(sel):
        for col, (label, rm) in enumerate(cols):
            ax = axes[row][col]
            ax.imshow(smooth_and_norm(rm[nid]), origin="lower", cmap="jet", vmin=0, vmax=1)
            ax.plot(Rx, Ry, "*", color="w", markersize=13, markeredgecolor="k", markeredgewidth=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"#{int(nid)}", fontsize=8)
    fig.suptitle("Same cell, matched runs: the field moves toward the reward (white star)\n"
                 "with the reward head, but not without it", y=1.005, fontsize=11)
    fig.tight_layout()
    out1 = os.path.join(out_dir, "matched_examples.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    fig.savefig(out1.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (label, key) in zip(axes, [("PRE (shared)", "pre"), ("POST  +head", "head"),
                                       ("POST  no head", "nohead")]):
        pl = np.where(place)[0]
        H, _, _ = np.histogram2d(peaks[key][pl, 0] * m, peaks[key][pl, 1] * m,
                                 bins=16, range=[[0, m], [0, m]])
        ax.imshow(smooth_and_norm(H).T, origin="lower", cmap="magma",
                  extent=[0, m, 0, m], aspect="equal")
        ax.plot(Rx, Ry, "*", color="w", markersize=20, markeredgecolor="k", markeredgewidth=0.8)
        ax.set_title(f"{label}   (n={len(pl)})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Population field-peak density, same place cells throughout "
                 "(unbiased -- no cell selection)", y=1.02)
    fig.tight_layout()
    out2 = os.path.join(out_dir, "matched_density.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    fig.savefig(out2.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"  saved -> {out1}\n  saved -> {out2}")


if __name__ == "__main__":
    main()
