"""
v11 experience-replay analysis: place fields before vs after cycle-1 RL.

The same cells are compared along the same frozen trajectory, recorded once
under pre-RL weights (replay_c{N}_pred) and once under post-RL weights
(replay_c{N}_postrl). Occupancy and recurrent input history are identical
across the pair, so any field change is representational rather than
trajectory skew. This module therefore asks only whether the weights moved the
fields toward the reward, and computes no random-walk comparison.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ratemaps import (
    build_rate_maps, compute_skaggs, smooth_and_norm, field_center_of_mass,
    field_peak,  # re-exported: `from plot_replay_v11 import field_peak` still works
)
from plot_goal_directed_maps import save_figure

# Publication colormaps (see lib/pubstyle.py). This module puts its own
# directory on sys.path above, so a plain import resolves either way.
try:
    from pubstyle import RATEMAP as _RATEMAP, DIVERGING as _DIVERGING, SEQUENTIAL as _SEQUENTIAL
except ImportError:  # pragma: no cover - stand-alone use
    _RATEMAP, _DIVERGING, _SEQUENTIAL = "magma", "coolwarm", "magma"



def load_maps(data_dir, period, m, hidden_dim):
    """Rate maps, occupancy and Skaggs scores for one replay period."""
    beh_path = os.path.join(data_dir, f"df_behavior_{period}.parquet")
    act_path = os.path.join(data_dir, f"df_activations_{period}.parquet")
    if not (os.path.exists(beh_path) and os.path.exists(act_path)):
        raise FileNotFoundError(
            f"Missing replay probe '{period}'.\n  expected: {beh_path}\n {act_path}\n"
            f"Did the v11 run finish the replay cycle? (--replay_cycle)")
    beh = pd.read_parquet(beh_path)
    act = pd.read_parquet(act_path)
    rm, occ = build_rate_maps(beh, act, m, hidden_dim)
    sk = compute_skaggs(rm, occ)
    return rm, occ, sk

def centres_for(rm, ids):
    """{neuron id: field centre of mass}, in [0, 1] arena coordinates."""
    # measured on the smoothed map the figures draw, so centres and panels agree
    return {int(nid): np.array(field_center_of_mass(smooth_and_norm(rm[nid]))) for nid in ids}


def peaks_for(rm, ids):
    """{neuron id: field peak}, in [0, 1] arena coordinates."""
    return {int(nid): field_peak(smooth_and_norm(rm[nid])) for nid in ids}


def fig_replay_grid(rm_pre, rm_post, rank_score, reward_bin, n_cells=20,save_path=None, no_show=False):
    """Pre-over-post rate-map grid for the top `n_cells` cells by rank_score."""
    top = np.argsort(rank_score)[::-1][:n_cells]
    ncols = min(10, n_cells)
    nrows_per = int(np.ceil(n_cells / ncols))
    rx, ry = reward_bin
    fig, axes = plt.subplots(nrows_per * 2, ncols,figsize=(ncols * 1.9, nrows_per * 2 * 2.1), squeeze=False)
    fig.patch.set_facecolor("#1a1a1a")
    # a cell's pre and post panels sit one block apart in the same column
    for i, nid in enumerate(top):
        row_pre = i // ncols
        row_post = row_pre + nrows_per
        col = i % ncols
        for ax, rm in [(axes[row_pre, col], rm_pre), (axes[row_post, col], rm_post)]:
            ax.imshow(smooth_and_norm(rm[nid]), origin="lower", cmap=_RATEMAP, vmin=0, vmax=1)
            ax.plot(rx, ry, "w*", markersize=5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("black")
            for sp in ax.spines.values():
                sp.set_visible(False)
    for r in range(nrows_per):
        axes[r, 0].set_ylabel("Pre-RL", fontsize=8, color="white")
    for r in range(nrows_per, nrows_per * 2):
        axes[r, 0].set_ylabel("Post-RL\n(replay)", fontsize=8, color="white")
    for i in range(n_cells, ncols * nrows_per):
        axes[i // ncols, i % ncols].axis("off")
        axes[i // ncols + nrows_per, i % ncols].axis("off")
    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def _draw_map_grid(rm, ids, reward_bin, title, save_path, no_show):
    """One 10-wide page of rate maps; shared by the pre and post map figures."""
    rx, ry = reward_bin
    n = len(ids)
    ncols = 10
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(ncols * 1.4, nrows * 1.4), dpi=150)
    fig.suptitle(title, fontsize=14, y=0.995)
    for i, nid in enumerate(ids):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(smooth_and_norm(rm[int(nid)]), origin="lower", cmap=_RATEMAP,
                  interpolation="gaussian", vmin=0, vmax=1)
        ax.plot(rx, ry, "w*", markersize=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, save_path, no_show, despine=False)


def fig_replay_maps(rm_pre, rm_post, rank_score, reward_bin, n_cells=100,save_dir=None, period_tag="c1", no_show=False):
    """Pre and post cognitive maps as two PNGs. Returns the cell ids drawn."""
    ids = np.argsort(rank_score)[::-1][:n_cells]
    pre_path = os.path.join(save_dir, f"replay_{period_tag}_map_PRE.png")
    post_path = os.path.join(save_dir, f"replay_{period_tag}_map_POST.png")
    _draw_map_grid(rm_pre, ids, reward_bin,
                   "Pre-RL cognitive map (same cells, frozen trajectory)",
                   pre_path, no_show)
    _draw_map_grid(rm_post, ids, reward_bin,
                   "Post-RL replay cognitive map (same cells, frozen trajectory)",
                   post_path, no_show)
    return ids


def select_example_cells(active, sk_pre, sk_post, d_pre, d_post, reward_pos,thr, n=12, max_post_dist=0.22):
    """Cells that pass Skaggs in both periods and moved toward the reward,
    ranked by distance gained and closeness of the post centre. The
    end-near-reward gate is dropped if fewer than `n` cells qualify."""
    sk_pre_a = sk_pre[active]
    sk_post_a = sk_post[active]
    delta = d_post - d_pre # < 0 => moved toward reward
    good = (sk_pre_a >= thr) & (sk_post_a >= thr) & (delta < 0) & (d_post <= max_post_dist)
    idx = np.where(good)[0]
    if len(idx) < n:# relax the end-near-reward gate if too few
        good = (sk_pre_a >= thr) & (sk_post_a >= thr) & (delta < 0)
        idx = np.where(good)[0]
    # favours cells that moved far and also ended close to the reward
    score = (d_pre[idx] - d_post[idx]) - d_post[idx]
    order = idx[np.argsort(score)[::-1]]
    return [int(active[i]) for i in order[:n]]


def fig_replay_examples(rm_pre, rm_post, c_pre, c_post, ids, reward_pos, reward_bin,save_path=None, no_show=False):
    """Example cells: pre on top, post below with an arrow from old to new centre."""
    n = len(ids)
    if n == 0:
        print("  (no qualifying example cells to plot)")
        return
    rx, ry = reward_bin
    m = rm_pre[int(ids[0])].shape[0]
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.2), squeeze=False)
    fig.patch.set_facecolor("white")
    for col, nid in enumerate(ids):
        # pre
        ax = axes[0, col]
        ax.imshow(smooth_and_norm(rm_pre[nid]), origin="lower", cmap=_RATEMAP,interpolation="gaussian", vmin=0, vmax=1)
        ax.plot(rx, ry, "w*", markersize=6)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(f"#{nid}", fontsize=7)
        # post + movement arrow (centres are in [0,1]; scale to bins)
        ax = axes[1, col]
        ax.imshow(smooth_and_norm(rm_post[nid]), origin="lower", cmap=_RATEMAP,
                  interpolation="gaussian", vmin=0, vmax=1)
        ax.plot(rx, ry, "w*", markersize=6)
        p0 = np.asarray(c_pre[nid]) * m
        p1 = np.asarray(c_post[nid]) * m
        ax.annotate("", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.4))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    axes[0, 0].set_ylabel("Pre-RL", fontsize=9)
    axes[1, 0].set_ylabel("Post-RL\n(replay)", fontsize=9)
    fig.suptitle("Example cells: place fields drift toward the reward (frozen trajectory)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def fig_replay_shift(d_pre, d_post, save_path=None, no_show=False):
    """Distance-to-reward histograms (pre vs post) and the delta-distance one."""
    delta = d_post - d_pre # negative = field moved toward reward
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    bins = np.linspace(0, 0.8, 26)
    ax.hist(d_pre, bins=bins, alpha=0.55, color="steelblue",
            label=f"Pre-RL (mean {d_pre.mean():.3f})", density=True)
    ax.hist(d_post, bins=bins, alpha=0.55, color="tomato",
            label=f"Post-RL replay (mean {d_post.mean():.3f})", density=True)
    ax.axvline(d_pre.mean(), color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(d_post.mean(), color="tomato", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Field-centre distance from reward", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9)
    ax = axes[1]
    # clip at the 99th pct so a few large shifts don't flatten the histogram
    lim = np.nanpercentile(np.abs(delta), 99) if len(delta) else 0.3
    bins2 = np.linspace(-lim, lim, 31)
    ax.hist(delta, bins=bins2, color="slateblue", alpha=0.75)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.axvline(delta.mean(), color="crimson", linestyle="--", linewidth=1.8,
               label=f"mean Δ = {delta.mean():+.3f}")
    ax.set_xlabel("Δ distance  (post − pre);  < 0 = toward reward", fontsize=11)
    ax.set_ylabel("Neurons", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_figure(fig, save_path, no_show)


def fig_replay_quiver(c_pre, c_post, reward_pos, save_path=None, no_show=False):
    """Field centres before and after, joined by arrows, over the arena."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for nid in c_pre:
        p0 = c_pre[nid]; p1 = c_post[nid]
        ax.annotate("", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.55, lw=0.8))
    P0 = np.array([c_pre[n] for n in c_pre])
    P1 = np.array([c_post[n] for n in c_pre])
    ax.scatter(P0[:, 0], P0[:, 1], s=12, c="steelblue", alpha=0.7, label="Pre-RL centre")
    ax.scatter(P1[:, 0], P1[:, 1], s=12, c="tomato", alpha=0.7, label="Post-RL centre")
    ax.plot(*reward_pos, "k*", markersize=18, zorder=5)
    ax.add_patch(plt.Circle(reward_pos, 0.15, color="k", fill=False, lw=1.3, linestyle="--"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    save_figure(fig, save_path, no_show)



def main():
    ap = argparse.ArgumentParser(description="v11 experience-replay analysis (pre vs post RL, frozen trajectory).")
    ap.add_argument("--run_dir", type=str, required=True,help="v11 run directory, e.g. New_Results_v2/final_v11_fixed")
    ap.add_argument("--replay_cycle", type=int, default=1,help="Which cycle's replay to analyse (matches --replay_cycle of the run)")
    ap.add_argument("--reward_pos", type=float, nargs=2, default=[0.5, 0.5], metavar=("X", "Y"))
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--skaggs_threshold", type=float, default=0.15)
    ap.add_argument("--n_cells", type=int, default=20, help="Cells shown in the pre/post grid")
    ap.add_argument("--n_map_cells", type=int, default=100, help="Cells in the 10x10 PRE/POST cognitive maps")
    ap.add_argument("--n_examples", type=int, default=12, help="Curated example cells with clearest toward-reward drift")
    ap.add_argument("--fig_subdir", type=str, default="replay")
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()
    data_dir = os.path.join(args.run_dir, "data")
    fig_dir = os.path.join(args.run_dir, "figures", args.fig_subdir)
    os.makedirs(fig_dir, exist_ok=True)
    m, hidden_dim = args.m, args.hidden_dim
    reward_pos = np.array(args.reward_pos)
    reward_bin = (int(reward_pos[0] * m), int(reward_pos[1] * m))
    N = args.replay_cycle
    thr = args.skaggs_threshold
    print(f"Loading replay probes for cycle {N} from {data_dir}")
    rm_pre, _, sk_pre = load_maps(data_dir, f"replay_c{N}_pred", m, hidden_dim)
    rm_post, _, sk_post = load_maps(data_dir, f"replay_c{N}_postrl", m, hidden_dim)
    print(f"  place cells (Skaggs >= {thr}):  pre={int((sk_pre>=thr).sum())}  post={int((sk_post>=thr).sum())}")
    # pre OR post: a cell recruited into the place-cell population by RL
    # still belongs in the paired comparison
    active = np.where(np.maximum(sk_pre, sk_post) >= thr)[0]
    print(f"  paired active neurons (place cell in pre OR post): {len(active)}")
    if len(active) == 0:
        print("No active place cells; nothing to plot.")
        return
    c_pre = centres_for(rm_pre, active)
    c_post = centres_for(rm_post, active)
    d_pre = np.array([np.linalg.norm(c_pre[int(n)] - reward_pos) for n in active])
    d_post = np.array([np.linalg.norm(c_post[int(n)] - reward_pos) for n in active])
    delta = d_post - d_pre
    # COM is pulled toward the arena middle, so it is confounded for a centre
    # reward; the peak metric is reported alongside as the de-confounded one
    p_pre = peaks_for(rm_pre, active)
    p_post = peaks_for(rm_post, active)
    dpk_pre = np.array([np.linalg.norm(p_pre[int(n)] - reward_pos) for n in active])
    dpk_post = np.array([np.linalg.norm(p_post[int(n)] - reward_pos) for n in active])
    delta_pk = dpk_post - dpk_pre
    frac_toward = float((delta < 0).mean())
    frac_toward_pk = float((delta_pk < 0).mean())
    print("\n=== Replay shift summary (identical trajectory, weights pre vs post RL) ===")
    print(f"  reward location: {tuple(np.round(reward_pos, 3))}"
          f"   {'(CENTRE -- COM metric is confounded, trust the PEAK metric)' if np.allclose(reward_pos, 0.5, atol=0.05) else '(OFF-CENTRE -- both metrics valid)'}")
    print("  [centre-of-mass metric]")
    print(f"    mean COM distance to reward:  pre={d_pre.mean():.4f}  post={d_post.mean():.4f}")
    print(f"    mean delta (post-pre): {delta.mean():+.4f}   (negative = moved toward reward)")
    print(f"    fraction of cells moved TOWARD reward: {frac_toward*100:.1f}%")
    print("  [peak-location metric  (de-confounded)]")
    print(f"    mean PEAK distance to reward: pre={dpk_pre.mean():.4f}  post={dpk_post.mean():.4f}")
    print(f"    mean delta (post-pre): {delta_pk.mean():+.4f}   (negative = moved toward reward)")
    print(f"    fraction of cells moved TOWARD reward: {frac_toward_pk*100:.1f}%")
    try:
        # paired across cells -- the trajectory is identical in both periods
        from scipy.stats import wilcoxon
        if len(delta) >= 5 and np.any(delta != 0):
            _, p = wilcoxon(d_post, d_pre)
            print(f"  Wilcoxon (COM)  paired p = {p:.2e}")
        if len(delta_pk) >= 5 and np.any(delta_pk != 0):
            _, p_pk = wilcoxon(dpk_post, dpk_pre)
            print(f"  Wilcoxon (PEAK) paired p = {p_pk:.2e}")
    except Exception as e:
        print(f"  (scipy wilcoxon unavailable: {e})")

    print("\nFigure 1: pre vs post field grid...")
    fig_replay_grid(rm_pre, rm_post, sk_pre + sk_post, reward_bin,
                    n_cells=args.n_cells,
                    save_path=os.path.join(fig_dir, f"replay_c{N}_grid.png"),
                    no_show=args.no_show)
    print("Figure 1b: 100-cell PRE vs POST cognitive maps (same cells)...")
    fig_replay_maps(rm_pre, rm_post, sk_pre, reward_bin,
                    n_cells=args.n_map_cells, save_dir=fig_dir,
                    period_tag=f"c{N}", no_show=args.no_show)
    print("Figure 1c: curated example cells (clearest toward-reward drift)...")
    ex_ids = select_example_cells(active, sk_pre, sk_post, d_pre, d_post,
                                  reward_pos, thr, n=args.n_examples)
    print(f"  selected example cells: {ex_ids}")
    fig_replay_examples(rm_pre, rm_post, c_pre, c_post, ex_ids, reward_pos, reward_bin,
                        save_path=os.path.join(fig_dir, f"replay_c{N}_examples.png"),
                        no_show=args.no_show)
    print("Figure 2: distance-to-reward shift...")
    fig_replay_shift(d_pre, d_post,
                     save_path=os.path.join(fig_dir, f"replay_c{N}_shift.png"),
                     no_show=args.no_show)
    print("Figure 3: field-centre movement quiver...")
    fig_replay_quiver(c_pre, c_post, reward_pos,
                      save_path=os.path.join(fig_dir, f"replay_c{N}_quiver.png"),
                      no_show=args.no_show)

    print("\nDone. Figures saved to:", fig_dir)
    print("Interpretation: a leftward shift / negative mean delta / arrows pointing to the\n"
          "reward = GENUINE representational remapping (trajectory was held fixed).\n"
          "No shift = the goal-directed concentration was trajectory/goal-driven, not\n"
          "baked into the weights.")

if __name__ == "__main__":
    main()
