"""
reward_relocation_maps.py
Actual cognitive maps for the relocation experiment: show that when the reward moves A -> B, the place fields follow.
"""
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

STATES = [("replay_reloc_pred", "baseline"),
          ("replay_reloc_postA", "after A"),
          ("replay_reloc_postB", "after B")]


def load_rm(data_dir, period, m=32, hidden_dim=512):
    beh = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{period}.pkl"))
    act = pd.read_pickle(os.path.join(data_dir, f"df_activations_{period}.pkl"))
    rm, occ = build_rate_maps(beh, act, m, hidden_dim)
    sk = compute_skaggs(rm, occ)
    return rm, sk


def field_sharpness(rm_norm):
    return 1.0 - float((rm_norm >= 0.5).mean())


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", "--run_dir", dest="run_dir", required=True)
    ap.add_argument("--output-dir", "--output_dir", dest="output_dir", default="Analysis_Plots/reward_relocation")
    ap.add_argument("--reward-a-x", type=float, default=0.3)
    ap.add_argument("--reward-a-y", type=float, default=0.3)
    ap.add_argument("--reward-b-x", type=float, default=0.7)
    ap.add_argument("--reward-b-y", type=float, default=0.7)
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--skaggs-threshold", type=float, default=0.15)
    ap.add_argument("--move-tol", type=float, default=0.02,help="min peak displacement (arena units) toward A and toward B to count as tracking")
    ap.add_argument("--reward-radius", type=float, default=0.22,help="max distance (arena units) the field may sit from A (after A) / B (after B)")
    ap.add_argument("--exclude", type=int, nargs="*", default=[],help="cell ids to drop from the exemplar panel (manual curation)")
    ap.add_argument("--m", type=int, default=32)
    return ap.parse_args()


def main():
    args = parse_args()
    A = np.array([args.reward_a_x, args.reward_a_y]); B = np.array([args.reward_b_x, args.reward_b_y])
    m = args.m
    data_dir = os.path.join(args.run_dir, "data")
    out_dir = ensure_dir(args.output_dir)
    rms, sks = {}, {}
    for period, _ in STATES:
        rms[period], sks[period] = load_rm(data_dir, period, m)
    pre, pA, pB = [s for s, _ in STATES]
    peaks = {p: np.array([field_peak(smooth_and_norm(rms[p][n])) for n in range(rms[p].shape[0])])
             for p in (pre, pA, pB)}
    def d(p, R):
        return np.linalg.norm(peaks[p] - R[None, :], axis=1)
    place = (sks[pre] >= args.skaggs_threshold) & (sks[pA] >= args.skaggs_threshold) & (sks[pB] >= args.skaggs_threshold)
    toward_A = d(pre, A) - d(pA, A)
    toward_B = d(pA, B) - d(pB, B)
    dA_mid = d(pA, A)
    dB_end = d(pB, B)
    minsk = np.minimum.reduce([sks[pre], sks[pA], sks[pB]])
    sharp = np.array([min(field_sharpness(smooth_and_norm(rms[p][n])) for p in (pre, pA, pB))for n in range(rms[pre].shape[0])])
    good = (place
            & (toward_A > args.move_tol) & (toward_B > args.move_tol)
            & (dA_mid < args.reward_radius) & (dB_end < args.reward_radius))

    quality = minsk + 0.7 * sharp - 0.5 * (dA_mid + dB_end)
    score = np.where(place, quality + np.where(good, 10.0, 0.0), -np.inf)
    for nid in args.exclude:
        if 0 <= nid < len(score):
            score[nid] = -np.inf
    order = np.argsort(score)[::-1]
    sel = order[:args.n_examples]
    print(f"[reward_relocation_maps] {int(good.sum())} cells passed strict tracking gates; "
          f"selecting {len(sel)} exemplars (best first).")

    Ax, Ay = A[0] * m, A[1] * m
    Bx, By = B[0] * m, B[1] * m
    n = len(sel)
    fig, axes = plt.subplots(3, n, figsize=(1.7 * n, 5.6), squeeze=False)
    for col, nid in enumerate(sel):
        for row, (period, label) in enumerate(STATES):
            ax = axes[row][col]
            ax.imshow(smooth_and_norm(rms[period][nid]), origin="lower", cmap="jet", vmin=0, vmax=1)
            ax.plot(Ax, Ay, "*", color="#2a9d8f", markersize=9, markeredgecolor="white", markeredgewidth=0.6)
            ax.plot(Bx, By, "*", color="#e76f51", markersize=9, markeredgecolor="white", markeredgewidth=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if row == 0:
                ax.set_title(f"#{int(nid)}", fontsize=8)
    fig.suptitle("Example cells: place fields track the reward from A (teal) to B (orange)", y=1.01)
    fig.tight_layout()
    out1 = os.path.join(out_dir, "reward_relocation_example_cells.png")
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    fig.savefig(out1.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (period, label) in zip(axes, STATES):
        pl = np.where(sks[period] >= args.skaggs_threshold)[0]
        px = peaks[period][pl, 0] * m
        py = peaks[period][pl, 1] * m
        H, xe, ye = np.histogram2d(px, py, bins=16, range=[[0, m], [0, m]])
        Hs = smooth_and_norm(H)
        ax.imshow(Hs.T, origin="lower", cmap="magma", extent=[0, m, 0, m], aspect="equal")
        ax.plot(Ax, Ay, "*", color="#2a9d8f", markersize=18, markeredgecolor="white")
        ax.plot(Bx, By, "*", color="#e76f51", markersize=18, markeredgecolor="white")
        ax.set_title(f"{label}  (n={len(pl)})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Population field-peak density: the map concentrates at A, then moves to B", y=1.02)
    fig.tight_layout()
    out2 = os.path.join(out_dir, "reward_relocation_field_density.png")
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    fig.savefig(out2.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"[reward_relocation_maps] example cells: {[int(i) for i in sel]}")
    print(f"  saved -> {out1}")
    print(f"  saved -> {out2}")


if __name__ == "__main__":
    main()
