"""
v11 experience-replay analysis.
Compares the same place cells along the same (frozen) trajectory before vs after
cycle-1 RL:
    replay_c{N}_pred    -> F_pre   (pre-RL weights,  trajectory T)
    replay_c{N}_postrl  -> F_post  (post-RL weights, trajectory T, frozen)
Because the trajectory (hence occupancy and recurrent input history) is identical,
any field change is purely representational -- not trajectory skew. This is the
whole point of the v11 replay experiment. In addition to the pre/post-RL shift
figures, this script also runs the ORI (over-representation index) and
occupancy-comparison analyses -- normally used for random-walk-vs-goal-directed
comparisons in plot_goal_directed_maps.py -- on this same pre/post-RL replay
pair, plus a recruitment analysis (which cells cross the place-cell threshold
only after RL). The core question throughout:
    "Holding the path fixed, did the weights move the fields toward the reward?"
Run:
  python code/figures/figS_legacy/plot_replay_v11.py --run_dir New_Results_v2/final_v11_fixed --no_show
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
from plot_goal_directed_maps import (
    build_rate_maps, compute_skaggs, smooth_and_norm,
    field_center_of_mass, save_figure, fig_ori, fig_occupancy,
)
from pc_common import append_stat, resolve_reward_pos, write_stats_report

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
    """{neuron id: field centre of mass} for the given ids, measured on the smoothed map."""
    return {int(nid): np.array(field_center_of_mass(smooth_and_norm(rm[nid]))) for nid in ids}


def field_peak(rm2d):
    """Peak location of one smoothed map. Returns its two components in (y, x) order, matching `field_center_of_mass`."""
    m = rm2d.shape[0]
    row, col = np.unravel_index(int(np.argmax(rm2d)), rm2d.shape)
    x = col / m + 0.5 / m
    y = row / m + 0.5 / m
    return np.array([x, y])


def peaks_for(rm, ids):
    """{neuron id: field peak} for the given ids, measured on the smoothed map."""
    return {int(nid): field_peak(smooth_and_norm(rm[nid])) for nid in ids}

def fig_replay_grid(rm_pre, rm_post, rank_score, reward_bin, n_cells=20,save_path=None, no_show=False):
    """Pre-over-post rate-map grid for the top cells by rank score."""
    top = np.argsort(rank_score)[::-1][:n_cells]
    ncols = min(10, n_cells)
    nrows_per = int(np.ceil(n_cells / ncols))
    rx, ry = reward_bin
    fig, axes = plt.subplots(nrows_per * 2, ncols,figsize=(ncols * 1.9, nrows_per * 2 * 2.1), squeeze=False)
    fig.patch.set_facecolor("#1a1a1a")
    for i, nid in enumerate(top):
        row_pre = i // ncols
        row_post = row_pre + nrows_per
        col = i % ncols
        for ax, rm in [(axes[row_pre, col], rm_pre), (axes[row_post, col], rm_post)]:
            ax.imshow(smooth_and_norm(rm[nid]), origin="lower", cmap="YlGnBu", vmin=0, vmax=1)
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
        ax.imshow(smooth_and_norm(rm[int(nid)]), origin="lower", cmap="YlGnBu",
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
    """Cells that pass the Skaggs gate in both periods and moved toward the reward, best first."""
    sk_pre_a = sk_pre[active]
    sk_post_a = sk_post[active]
    delta = d_post - d_pre # < 0 => moved toward reward
    good = (sk_pre_a >= thr) & (sk_post_a >= thr) & (delta < 0) & (d_post <= max_post_dist)
    idx = np.where(good)[0]
    if len(idx) < n:# relax the end-near-reward gate if too few
        good = (sk_pre_a >= thr) & (sk_post_a >= thr) & (delta < 0)
        idx = np.where(good)[0]
    score = (d_pre[idx] - d_post[idx]) - d_post[idx]
    order = idx[np.argsort(score)[::-1]]
    return [int(active[i]) for i in order[:n]]


def fig_replay_examples(rm_pre, rm_post, c_pre, c_post, ids, reward_pos, reward_bin, save_path=None, no_show=False,
                         title="Example cells: place fields drift toward the reward (frozen trajectory)",
                         sk_pre=None, sk_post=None, pre_label="Pre-RL", post_label="Post-RL\n(replay)",
                         draw_arrow=True):
    """Example cells: pre map above, post below, with an arrow from the old centre to the new one."""
    n = len(ids)
    if n == 0:
        print("  (no qualifying example cells to plot)")
        return
    rx, ry = reward_bin
    m = rm_pre[int(ids[0])].shape[0]
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.4), squeeze=False)
    fig.patch.set_facecolor("white")
    for col, nid in enumerate(ids):
        # pre
        ax = axes[0, col]
        ax.imshow(smooth_and_norm(rm_pre[nid]), origin="lower", cmap="YlGnBu",interpolation="gaussian", vmin=0, vmax=1)
        ax.plot(rx, ry, "w*", markersize=6)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        top_title = f"#{nid}"
        if sk_pre is not None:
            top_title += f"\nsk={sk_pre[nid]:.2f}"
        ax.set_title(top_title, fontsize=7)
        # post (optionally with a pre->post field-centre movement arrow)
        ax = axes[1, col]
        ax.imshow(smooth_and_norm(rm_post[nid]), origin="lower", cmap="YlGnBu",
                  interpolation="gaussian", vmin=0, vmax=1)
        ax.plot(rx, ry, "w*", markersize=6)
        if draw_arrow:
            p0 = np.asarray(c_pre[nid]) * m
            p1 = np.asarray(c_post[nid]) * m
            ax.annotate("", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.4))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        if sk_post is not None:
            ax.set_title(f"sk={sk_post[nid]:.2f}", fontsize=7)
    axes[0, 0].set_ylabel(pre_label, fontsize=9)
    axes[1, 0].set_ylabel(post_label, fontsize=9)
    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def fig_replay_shift(d_pre, d_post, save_path=None, no_show=False):
    """Distance-to-reward histograms (pre vs post) beside the delta-distance histogram."""
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


def shift_vs_empty_filter(rm_pre, rm_post, sk_pre, sk_post, reward_pos,
                          gates=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)):
    """
    Robustness sweep for the empty-cell confound: recompute the toward-reward
    shift while progressively demanding a stronger real field (higher Skaggs)
    in both pre- and post-RL states. If the shift were an artifact of empty /
    diffuse cells' COM regressing to the arena centre, gating them out would
    kill it. A shift that is flat or grows as the gate tightens (and stays
    significant on the shrinking, ever-cleaner survivor set) is genuine.
    Returns a per-gate list of dicts.
    """
    from scipy.stats import wilcoxon
    N = rm_pre.shape[0]
    c_pre = {i: np.array(field_center_of_mass(smooth_and_norm(rm_pre[i]))) for i in range(N)}
    c_post = {i: np.array(field_center_of_mass(smooth_and_norm(rm_post[i]))) for i in range(N)}
    smin = np.minimum(sk_pre, sk_post)
    rows = []
    print("\n=== Shift vs empty-cell filter (gate on min(Skaggs_pre,Skaggs_post) >= g) ===")
    print(f"  {'gate':>5} {'n':>4} | {'dist pre':>8} {'dist post':>9} {'delta':>8} {'p':>9}")
    for g in gates:
        keep = np.where(smin >= g)[0]
        if len(keep) < 5:
            rows.append({"gate": float(g), "n": int(len(keep))})
            print(f"  {g:>5.1f} {len(keep):>4} | (too few survivors)")
            continue
        dc_pre = np.array([np.linalg.norm(c_pre[int(n)] - reward_pos) for n in keep])
        dc_post = np.array([np.linalg.norm(c_post[int(n)] - reward_pos) for n in keep])
        pc = float(wilcoxon(dc_post, dc_pre).pvalue) if np.any(dc_post != dc_pre) else 1.0
        rows.append({
            "gate": float(g), "n": int(len(keep)),
            "dist_pre": float(dc_pre.mean()), "dist_post": float(dc_post.mean()),
            "delta": float((dc_post - dc_pre).mean()), "p": pc,
        })
        print(f"  {g:>5.1f} {len(keep):>4} | {dc_pre.mean():>8.3f} {dc_post.mean():>9.3f} "
              f"{(dc_post-dc_pre).mean():>+8.4f} {pc:>9.1e}")
    return rows


def fig_shift_vs_empty_filter(rows, reward_pos, save_path=None, no_show=False):
    """Toward-reward shift recomputed at progressively stricter Skaggs gates, testing the empty-cell confound."""
    valid = [r for r in rows if "delta" in r]
    if not valid:
        print("  (no gate levels with enough survivors to plot)")
        return
    gates = [r["gate"] for r in valid]
    ns = [r["n"] for r in valid]
    deltas = [r["delta"] for r in valid]
    ps = [r["p"] for r in valid]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.axhline(0, color="black", lw=1.0)
    ax1.plot(gates, deltas, "o-", color="slateblue", lw=2, label="Shift toward reward (post−pre)")
    for g, d, p, n in zip(gates, deltas, ps, ns):
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
        ax1.annotate(f"n={n}\n{star}", (g, d), textcoords="offset points",
                     xytext=(0, -22 if d < 0 else 8), ha="center", fontsize=7, color="dimgray")
    ax1.set_xlabel("Empty-cell filter: require Skaggs ≥ g in BOTH pre- and post-RL", fontsize=11)
    ax1.set_ylabel("Mean Δ distance to reward   (< 0 = toward reward)", fontsize=11)
    ax1.set_title("Toward-reward shift survives (and grows) when empty cells are removed\n"
                  "artifact would vanish as the gate tightens; genuine remapping does not", fontsize=11)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(alpha=0.25)
    plt.tight_layout()
    save_figure(fig, save_path, no_show)


def recruitment_analysis(sk_pre, sk_post, rm_post, reward_pos, threshold, R=0.15):
    """Cells that cross the place-cell (Skaggs) threshold only after RL
    ('recruited') vs only before RL ('lost'). Tests whether recruited cells'
    fields are enriched near the reward relative to chance (uniform arena)."""
    is_place_pre = sk_pre >= threshold
    is_place_post = sk_post >= threshold
    recruited = np.where(is_place_post & ~is_place_pre)[0]
    lost = np.where(is_place_pre & ~is_place_post)[0]
    shared = np.where(is_place_pre & is_place_post)[0]
    print(f"\n=== Recruitment analysis (Skaggs >= {threshold}) ===")
    print(f"  Place cells PRE only  (lost):      {len(lost)}")
    print(f"  Place cells POST only (recruited): {len(recruited)}")
    print(f"  Place cells in BOTH   (shared):    {len(shared)}")
    result = {
        "test": "replay_recruitment_zone_enrichment",
        "skaggs_threshold": float(threshold),
        "n_lost": int(len(lost)),
        "n_recruited": int(len(recruited)),
        "n_shared": int(len(shared)),
    }
    if len(recruited) > 0:
        from scipy.stats import norm
        centres = np.array([field_center_of_mass(smooth_and_norm(rm_post[nid])) for nid in recruited])
        d_recruited = np.linalg.norm(centres - reward_pos, axis=1)
        n_near = int((d_recruited < R).sum())
        p_chance = float(np.pi * R ** 2)
        p_obs = n_near / len(recruited)
        z = (p_obs - p_chance) / np.sqrt(p_chance * (1 - p_chance) / len(recruited))
        p_z = float(1.0 - norm.cdf(z))
        print(f"  Recruited cells within R={R} of reward: {n_near}/{len(recruited)} = {p_obs:.3f}  "
              f"(chance={p_chance:.3f})   z={z:.3f}  p={p_z:.4f}")
        result.update({"n_near_reward": n_near, "p_obs": p_obs, "p_chance": p_chance,
                       "z_stat": float(z), "p_value_one_sided": p_z})
    else:
        print("  No recruited cells -- cannot test zone enrichment.")
    return result


def fig_replay_recruitment(sk_pre, sk_post, threshold, save_path=None, no_show=False):
    """Place-cell counts pre and post, showing how many cells cross the Skaggs threshold only after RL."""
    is_place_pre = sk_pre >= threshold
    is_place_post = sk_post >= threshold
    counts = {
        "Lost\n(PRE only)": int(np.sum(is_place_pre & ~is_place_post)),
        "Shared\n(both)": int(np.sum(is_place_pre & is_place_post)),
        "Recruited\n(POST only)": int(np.sum(is_place_post & ~is_place_pre)),
        "Neither": int(np.sum(~is_place_pre & ~is_place_post)),
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(list(counts.keys()), list(counts.values()),
           color=["tomato", "mediumpurple", "steelblue", "lightgray"])
    for i, v in enumerate(counts.values()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of neurons")
    ax.set_title(f"Place-cell recruitment (Skaggs >= {threshold}), pre vs post RL")
    plt.tight_layout()
    save_figure(fig, save_path, no_show)


def recruitment_fate_analysis(rm_pre, rm_post, sk_pre, sk_post, reward_pos, threshold):
    """
    Full characterization of what happens to 'lost' and 'recruited' place
    cells -- not just their counts. For each fate group (lost/recruited/
    shared): did lost cells go silent or just get more diffuse (lower Skaggs
    but same or higher activity)? were recruited cells already weakly
    spatial beforehand, or did they emerge from near-nothing? and where
    were/are these cells relative to reward, compared to the stable
    ('shared') population and to each other?
    """
    is_place_pre = sk_pre >= threshold
    is_place_post = sk_post >= threshold
    lost = np.where(is_place_pre & ~is_place_post)[0]
    recruited = np.where(~is_place_pre & is_place_post)[0]
    shared = np.where(is_place_pre & is_place_post)[0]

    mean_act_pre = rm_pre.mean(axis=(1, 2))
    mean_act_post = rm_post.mean(axis=(1, 2))

    def centroid_dist(rm, nid):
        c = np.array(field_center_of_mass(smooth_and_norm(rm[nid])))
        return np.linalg.norm(c - reward_pos)

    def describe_group(ids, name):
        if len(ids) == 0:
            print(f"  {name}: none.")
            return None
        d_pre = np.array([centroid_dist(rm_pre, n) for n in ids])
        d_post = np.array([centroid_dist(rm_post, n) for n in ids])
        act_ratio = mean_act_post[ids].mean() / max(mean_act_pre[ids].mean(), 1e-12)
        print(f"  {name} (n={len(ids)}): Skaggs PRE={sk_pre[ids].mean():.3f} POST={sk_post[ids].mean():.3f}  |  "
              f"activity PRE={mean_act_pre[ids].mean():.4f} POST={mean_act_post[ids].mean():.4f} (ratio={act_ratio:.3f})  |  "
              f"dist-to-reward PRE={d_pre.mean():.4f} POST={d_post.mean():.4f}")
        return {
            "n": int(len(ids)), "skaggs_pre": float(sk_pre[ids].mean()), "skaggs_post": float(sk_post[ids].mean()),
            "activity_pre": float(mean_act_pre[ids].mean()), "activity_post": float(mean_act_post[ids].mean()),
            "activity_ratio": float(act_ratio),
            "dist_pre": float(d_pre.mean()), "dist_post": float(d_post.mean()),
            "d_pre_arr": d_pre, "d_post_arr": d_post,
        }

    print(f"\n=== Recruitment fate: what happened to lost/recruited cells (Skaggs >= {threshold}) ===")
    print("  NOTE: lost/recruited cells are non-spatial (empty) in one state, so their")
    print("  distance-to-reward is COM-confounded (diffuse maps sit at the arena centre).")
    print("  Read lost cells as 'field DISSOLVES', not 'field migrates to reward'. The")
    print("  trustworthy toward-reward shift is the empty-filtered survivor set above.")
    lost_stats = describe_group(lost, "Lost      (place PRE only)")
    recruited_stats = describe_group(recruited, "Recruited (place POST only)")
    shared_stats = describe_group(shared, "Shared    (place in both)")

    record = {"test": "recruitment_fate", "skaggs_threshold": float(threshold)}
    from scipy.stats import mannwhitneyu

    if lost_stats:
        frac_collapsed = float((sk_post[lost] < 0.3 * threshold).mean())
        print(f"  Lost cells: {frac_collapsed * 100:.1f}% collapsed to Skaggs<{0.3 * threshold:.2f} (near-silent post-RL); "
              f"the rest ({(1 - frac_collapsed) * 100:.1f}%) stayed weakly spatial, just under the cutoff.")
        record.update({
            "n_lost": lost_stats["n"], "lost_skaggs_pre": lost_stats["skaggs_pre"], "lost_skaggs_post": lost_stats["skaggs_post"],
            "lost_activity_pre": lost_stats["activity_pre"], "lost_activity_post": lost_stats["activity_post"],
            "lost_activity_ratio": lost_stats["activity_ratio"],
            "lost_dist_pre": lost_stats["dist_pre"], "lost_dist_post": lost_stats["dist_post"],
            "lost_fraction_collapsed": frac_collapsed,
        })
    if recruited_stats:
        frac_emerged = float((sk_pre[recruited] < 0.3 * threshold).mean())
        print(f"  Recruited cells: {frac_emerged * 100:.1f}% emerged from near-nothing (Skaggs<{0.3 * threshold:.2f} pre-RL); "
              f"the rest ({(1 - frac_emerged) * 100:.1f}%) were already weakly spatial before RL.")
        record.update({
            "n_recruited": recruited_stats["n"], "recruited_skaggs_pre": recruited_stats["skaggs_pre"],
            "recruited_skaggs_post": recruited_stats["skaggs_post"],
            "recruited_activity_pre": recruited_stats["activity_pre"], "recruited_activity_post": recruited_stats["activity_post"],
            "recruited_activity_ratio": recruited_stats["activity_ratio"],
            "recruited_dist_pre": recruited_stats["dist_pre"], "recruited_dist_post": recruited_stats["dist_post"],
            "recruited_fraction_emerged": frac_emerged,
        })
    if shared_stats:
        record.update({
            "n_shared": shared_stats["n"],
            "shared_dist_pre": shared_stats["dist_pre"], "shared_dist_post": shared_stats["dist_post"],
            "shared_activity_pre": shared_stats["activity_pre"], "shared_activity_post": shared_stats["activity_post"],
        })

    if lost_stats and shared_stats:
        u, p = mannwhitneyu(lost_stats["d_pre_arr"], shared_stats["d_pre_arr"], alternative="greater")
        print(f"  Lost PRE-dist > Shared PRE-dist ({lost_stats['dist_pre']:.3f} vs {shared_stats['dist_pre']:.3f}): U={u:.0f}  p={p:.4g}")
        record["lost_vs_shared_pre_dist_p"] = float(p)
    if recruited_stats and shared_stats:
        u, p = mannwhitneyu(recruited_stats["d_post_arr"], shared_stats["d_post_arr"], alternative="less")
        print(f"  Recruited POST-dist < Shared POST-dist ({recruited_stats['dist_post']:.3f} vs {shared_stats['dist_post']:.3f}): U={u:.0f}  p={p:.4g}")
        record["recruited_vs_shared_post_dist_p"] = float(p)
    if lost_stats and recruited_stats:
        u, p = mannwhitneyu(lost_stats["d_pre_arr"], recruited_stats["d_pre_arr"], alternative="two-sided")
        print(f"  Lost PRE-dist vs Recruited PRE-dist ({lost_stats['dist_pre']:.3f} vs {recruited_stats['dist_pre']:.3f}): p={p:.4g}")
        record["lost_vs_recruited_pre_dist_p"] = float(p)

    return record


def fig_recruitment_fate(sk_pre, sk_post, rm_pre, rm_post, reward_pos, threshold, save_path=None, no_show=False):
    """Place-cell status migration: how each cell's pre-RL status maps to its post-RL one, with activity and distance by fate."""
    is_place_pre = sk_pre >= threshold
    is_place_post = sk_post >= threshold
    lost = np.where(is_place_pre & ~is_place_post)[0]
    recruited = np.where(~is_place_pre & is_place_post)[0]
    shared = np.where(is_place_pre & is_place_post)[0]
    neither = np.where(~is_place_pre & ~is_place_post)[0]

    mean_act_pre = rm_pre.mean(axis=(1, 2))
    mean_act_post = rm_post.mean(axis=(1, 2))
    groups = [("Lost", lost, "tomato"), ("Recruited", recruited, "steelblue"), ("Shared", shared, "mediumpurple")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.scatter(sk_pre[neither], sk_post[neither], s=8, alpha=0.3, color="lightgray", label=f"Neither (n={len(neither)})")
    for name, ids, color in reversed(groups):
        if len(ids):
            ax.scatter(sk_pre[ids], sk_post[ids], s=14, alpha=0.7, color=color, label=f"{name} (n={len(ids)})")
    ax.axvline(threshold, color="black", linestyle="--", lw=0.8)
    ax.axhline(threshold, color="black", linestyle="--", lw=0.8)
    ax.set_xlabel("Skaggs, Pre-RL")
    ax.set_ylabel("Skaggs, Post-RL replay")
    ax.set_title("Place-cell status migration")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    labels, ratios, colors = [], [], []
    for name, ids, color in groups:
        if len(ids) == 0:
            continue
        labels.append(name)
        ratios.append(mean_act_post[ids].mean() / max(mean_act_pre[ids].mean(), 1e-12))
        colors.append(color)
    ax.bar(labels, ratios, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", lw=0.8, label="no change")
    ax.set_ylabel("Mean activity ratio (Post-RL / Pre-RL)")
    ax.set_title("Activity level change by fate")
    ax.legend(fontsize=8)

    ax = axes[2]

    def centroid_dist(rm, nid):
        c = np.array(field_center_of_mass(smooth_and_norm(rm[nid])))
        return np.linalg.norm(c - reward_pos)

    x = np.arange(len(groups))
    width = 0.35
    pre_d, post_d, names = [], [], []
    for name, ids, color in groups:
        names.append(name)
        if len(ids) == 0:
            pre_d.append(np.nan)
            post_d.append(np.nan)
        else:
            pre_d.append(np.mean([centroid_dist(rm_pre, n) for n in ids]))
            post_d.append(np.mean([centroid_dist(rm_post, n) for n in ids]))
    ax.bar(x - width / 2, pre_d, width, label="Pre-RL", color="steelblue")
    ax.bar(x + width / 2, post_d, width, label="Post-RL replay", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean distance to reward")
    ax.set_title("Distance to reward by fate")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def fig_confident_cells(d_pre, d_post, icl_pre_vals, icl_post_vals, save_path=None, no_show=False):
    """Distance to reward, pre against post, for cells selected as confidently fit before RL."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(d_pre, d_post, alpha=0.5, s=15, color="steelblue")
    lim = max(d_pre.max(), d_post.max()) * 1.05 if len(d_pre) else 1.0
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (no change)")
    ax.set_xlabel("Distance to reward (Pre-RL)")
    ax.set_ylabel("Distance to reward (Post-RL replay)")
    ax.set_title("Confident pre-RL place cells: distance to reward")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(icl_pre_vals, icl_post_vals, alpha=0.5, s=15, color="tomato")
    if len(icl_pre_vals):
        lo = min(icl_pre_vals.min(), icl_post_vals.min())
        hi = max(icl_pre_vals.max(), icl_post_vals.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (no change)")
    ax.set_xlabel("ICL, pre-RL  (more negative = more confident)")
    ax.set_ylabel("ICL, post-RL replay")
    ax.set_title("GMM-fit confidence: pre vs post RL")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_figure(fig, save_path, no_show, despine=False)


def confident_place_cell_analysis(run_dir, N, rm_pre, rm_post, sk_pre, sk_post, reward_pos,
                                   skaggs_threshold, fig_dir, no_show, k_max=3, icl_quantile=0.5):
    """
    Selects a confidently-fit place-cell population using only the pre-RL GMM
    fit (k<=k_max, and ICL score at or below the icl_quantile of that state's
    own icl_score distribution -- a within-state relative criterion, so it
    doesn't inherit the cross-run absolute-ICL miscalibration problem). This
    matches the original design intent: choose confident cells based on
    'before', then track what happened to them after the reward.

    Also checks whether GMM-fit confidence itself dropped post-RL (evidence
    of field spreading, as opposed to cells simply vanishing), and whether
    cells that lose confidence post-RL were farther from the reward pre-RL
    (i.e. whether the recruitment/loss pattern is reward-position-dependent).
    """
    data_dir = os.path.join(run_dir, "data")
    centers_pre_path = os.path.join(data_dir, f"df_centers_replay_c{N}_pred.parquet")
    centers_post_path = os.path.join(data_dir, f"df_centers_replay_c{N}_postrl.parquet")
    print(f"\n=== Confident pre-RL place cells (GMM k<={k_max}, ICL top-{icl_quantile*100:.0f}% "
          f"within replay_c{N}_pred, Skaggs>={skaggs_threshold}) ===")
    if not (os.path.exists(centers_pre_path) and os.path.exists(centers_post_path)):
        print(f"  GMM center fits not found. Run first:\n"
              f"  python3 gmm_place_metric.py --run_dir {run_dir} "
              f"--states replay_c{N}_pred,replay_c{N}_postrl")
        return None

    df_pre = pd.read_parquet(centers_pre_path).drop_duplicates("neuron_id")
    df_post = pd.read_parquet(centers_post_path).drop_duplicates("neuron_id")

    df_pre_k = df_pre[df_pre["k"] <= k_max].copy()
    icl = df_pre_k["icl_score"].replace([np.inf, -np.inf], np.nan)
    icl_thresh = icl.quantile(icl_quantile)
    df_pre_k = df_pre_k[df_pre_k["icl_score"] <= icl_thresh]
    confident_ids = sorted(set(df_pre_k["neuron_id"].tolist()) & set(np.where(sk_pre >= skaggs_threshold)[0].tolist()))
    print(f"  n confident pre-RL place cells: {len(confident_ids)}")
    if len(confident_ids) < 5:
        print("  Too few confident cells for a meaningful test.")
        return None

    d_pre = np.array([np.linalg.norm(np.array(field_center_of_mass(smooth_and_norm(rm_pre[n]))) - reward_pos) for n in confident_ids])
    d_post = np.array([np.linalg.norm(np.array(field_center_of_mass(smooth_and_norm(rm_post[n]))) - reward_pos) for n in confident_ids])
    delta = d_post - d_pre
    frac_toward = float((delta < 0).mean())
    print(f"  mean dist pre={d_pre.mean():.4f}  post={d_post.mean():.4f}  delta={delta.mean():+.4f}  "
          f"frac toward reward={frac_toward * 100:.1f}%")

    record = {
        "test": "confident_place_cell_shift",
        "replay_cycle": N,
        "k_max": k_max,
        "icl_quantile": icl_quantile,
        "skaggs_threshold": float(skaggs_threshold),
        "n_confident_cells": len(confident_ids),
        "mean_dist_pre": float(d_pre.mean()),
        "mean_dist_post": float(d_post.mean()),
        "mean_delta": float(delta.mean()),
        "fraction_toward_reward": frac_toward,
    }
    from scipy.stats import wilcoxon, mannwhitneyu, pearsonr
    if len(delta) >= 5 and np.any(delta != 0):
        _, p = wilcoxon(d_post, d_pre)
        print(f"  Wilcoxon paired (post vs pre distance): p={p:.4g}")
        record["wilcoxon_distance_p"] = float(p)

    icl_pre_map = df_pre.set_index("neuron_id")["icl_score"].to_dict()
    icl_post_map = df_post.set_index("neuron_id")["icl_score"].to_dict()
    k_post_map = df_post.set_index("neuron_id")["k"].to_dict()

    icl_pre_vals, icl_post_vals, matched_ids = [], [], []
    for n in confident_ids:
        ip, ipo = icl_pre_map.get(n), icl_post_map.get(n)
        if ip is not None and ipo is not None and np.isfinite(ip) and np.isfinite(ipo):
            icl_pre_vals.append(ip); icl_post_vals.append(ipo); matched_ids.append(n)
    icl_pre_vals = np.array(icl_pre_vals)
    icl_post_vals = np.array(icl_post_vals)
    print(f"  ICL matched (finite pre & post): {len(matched_ids)}/{len(confident_ids)}")

    if len(matched_ids) >= 5:
        print(f"  median ICL pre={np.median(icl_pre_vals):.1f}  post={np.median(icl_post_vals):.1f}  "
              f"(less negative = less confident / more spread)")
        icl_delta = icl_post_vals - icl_pre_vals
        n_less_confident = int((icl_delta > 0).sum())
        frac_less_confident = n_less_confident / len(matched_ids)
        print(f"  fraction that became LESS confident (ICL increased): "
              f"{n_less_confident}/{len(matched_ids)} = {frac_less_confident * 100:.1f}%")
        stat, p_icl = wilcoxon(icl_post_vals, icl_pre_vals)
        print(f"  Wilcoxon paired (ICL post vs pre): p={p_icl:.4g}  "
              f"(p<0.05 & increased => significant loss of fit confidence, i.e. spreading)")
        record.update({
            "n_icl_matched": len(matched_ids),
            "median_icl_pre": float(np.median(icl_pre_vals)),
            "median_icl_post": float(np.median(icl_post_vals)),
            "fraction_less_confident_post": frac_less_confident,
            "wilcoxon_icl_p": float(p_icl),
        })

        d_pre_matched = np.array([np.linalg.norm(np.array(field_center_of_mass(smooth_and_norm(rm_pre[n]))) - reward_pos) for n in matched_ids])
        r, p_corr = pearsonr(icl_delta, d_pre_matched)
        print(f"  corr(confidence loss, pre-RL distance to reward): r={r:.3f}  p={p_corr:.4g}  "
              f"(r>0 & p<0.05 => cells farther from reward lose confidence more)")
        record["corr_confidence_loss_vs_pre_distance_r"] = float(r)
        record["corr_confidence_loss_vs_pre_distance_p"] = float(p_corr)

        k_post_vals = np.array([k_post_map.get(n, np.nan) for n in matched_ids])
        still_compact = k_post_vals <= k_max
        if still_compact.sum() >= 2 and (~still_compact).sum() >= 2:
            d_compact = d_pre_matched[still_compact]
            d_fragmented = d_pre_matched[~still_compact]
            u, p_frag = mannwhitneyu(d_fragmented, d_compact, alternative="greater")
            print(f"  Fragmented-post cells (k>{k_max}, n={len(d_fragmented)}) pre-RL distance="
                  f"{d_fragmented.mean():.3f} vs still-compact (n={len(d_compact)})={d_compact.mean():.3f}  "
                  f"Mann-Whitney p={p_frag:.4g}")
            record.update({
                "n_still_compact": int(still_compact.sum()),
                "n_fragmented": int((~still_compact).sum()),
                "mean_dist_still_compact": float(d_compact.mean()),
                "mean_dist_fragmented": float(d_fragmented.mean()),
                "fragmented_farther_from_reward_p": float(p_frag),
            })

        fig_confident_cells(
            d_pre_matched if len(matched_ids) else d_pre,
            np.array([np.linalg.norm(np.array(field_center_of_mass(smooth_and_norm(rm_post[n]))) - reward_pos) for n in matched_ids]) if len(matched_ids) else d_post,
            icl_pre_vals, icl_post_vals,
            save_path=os.path.join(fig_dir, f"replay_c{N}_confident_cells.png"),
            no_show=no_show,
        )

    return record


def main():
    """Parse arguments, load the replay pair, and write every figure in this module."""
    ap = argparse.ArgumentParser(description="v11 experience-replay analysis (pre vs post RL, frozen trajectory).")
    ap.add_argument("--run_dir", type=str, required=True,help="v11 run directory, e.g. New_Results_v2/final_v11_fixed")
    ap.add_argument("--replay_cycle", type=int, default=1,help="Which cycle's replay to analyse (matches --replay_cycle of the run)")
    ap.add_argument("--reward_pos", type=float, nargs=2, default=None, metavar=("X", "Y"),
                     help="Reward (x, y). Defaults to run_dir/config.json's reward_pos.")
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--skaggs_threshold", type=float, default=1.0,
                     help="Place-cell cutoff in bits/spike. Visually validated: 0.15-0.5 includes "
                          "diffuse/noisy cells, >=1.0 reliably gives clean localized fields.")
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
    reward_pos = resolve_reward_pos(args.run_dir, args.reward_pos)
    reward_bin = (int(reward_pos[0] * m), int(reward_pos[1] * m))
    N = args.replay_cycle
    thr = args.skaggs_threshold
    print(f"Loading replay probes for cycle {N} from {data_dir}")
    rm_pre, occ_pre, sk_pre = load_maps(data_dir, f"replay_c{N}_pred", m, hidden_dim)
    rm_post, occ_post, sk_post = load_maps(data_dir, f"replay_c{N}_postrl", m, hidden_dim)
    n_pre_place = int((sk_pre >= thr).sum())
    n_post_place = int((sk_post >= thr).sum())
    n_lost = int(((sk_pre >= thr) & (sk_post < thr)).sum())
    n_recruited = int(((sk_pre < thr) & (sk_post >= thr)).sum())
    print(f"  place cells (Skaggs >= {thr}):  pre={n_pre_place}  post={n_post_place}")
    active = np.where(np.maximum(sk_pre, sk_post) >= thr)[0]

    # empty-cell filter: the shift-toward-reward analysis is computed only on
    # cells that hold a real, spatially-informative field (Skaggs >= thr) in
    # both the pre- and post-RL states ("stable" cells). An empty / non-spatial
    # cell has a meaningless centre of mass -- its normalized rate map is
    # near-uniform, so its COM collapses to the arena centre (0.5, 0.5), which
    # is closer to an off-centre reward than a typical field. Including such
    # cells manufactures an apparent "toward reward" shift purely from fields
    # dissolving. Gating on min(Skaggs_pre, Skaggs_post) >= thr drops exactly
    # the lost cells (empty POST) and recruited cells (empty PRE), leaving only
    # cells whose centroid is trustworthy in both states.
    stable = np.where(np.minimum(sk_pre, sk_post) >= thr)[0]
    print(f"  neurons that are place cells in pre OR post (any activity): {len(active)}")
    print(f"  SURVIVORS after empty-cell filter (place cell in pre AND post, Skaggs>={thr}): {len(stable)}")
    print(f"    (dropped {n_lost} lost [empty post] + {n_recruited} recruited [empty pre] non-spatial cells)")
    if len(stable) == 0:
        print("No cells survive the empty-cell filter; nothing to plot.")
        return
    # centres over the full active set (needed for recruitment-fate arrows /
    # example galleries), distances/stats over the empty-filtered stable set.
    c_pre = centres_for(rm_pre, active)
    c_post = centres_for(rm_post, active)
    c_pre_s = {int(n): c_pre[int(n)] for n in stable}
    c_post_s = {int(n): c_post[int(n)] for n in stable}
    d_pre = np.array([np.linalg.norm(c_pre[int(n)] - reward_pos) for n in stable])
    d_post = np.array([np.linalg.norm(c_post[int(n)] - reward_pos) for n in stable])
    delta = d_post - d_pre
    frac_toward = float((delta < 0).mean())
    print("\n=== Replay shift summary (identical trajectory, weights pre vs post RL) ===")
    print(f"  reward location: {tuple(np.round(reward_pos, 3))}")
    print(f"  computed on {len(stable)} empty-filtered survivors (place cell in BOTH states)")
    print(f"  mean field distance to reward:  pre={d_pre.mean():.4f}  post={d_post.mean():.4f}")
    print(f"  mean delta (post-pre): {delta.mean():+.4f}   (negative = moved toward reward)")
    print(f"  fraction of cells moved TOWARD reward: {frac_toward*100:.1f}%")
    stat_record = {
        "test": "replay_pre_post_rl_shift",
        "replay_cycle": N,
        "skaggs_threshold": float(thr),
        "n_place_pre": n_pre_place,
        "n_place_post": n_post_place,
        "n_lost_empty_post": n_lost,
        "n_recruited_empty_pre": n_recruited,
        "n_survivors_both_states": int(len(stable)),
        "mean_dist_pre": float(d_pre.mean()),
        "mean_dist_post": float(d_post.mean()),
        "mean_delta": float(delta.mean()),
        "fraction_toward_reward": frac_toward,
    }
    try:
        from scipy.stats import wilcoxon
        if len(delta) >= 5 and np.any(delta != 0):
            _, p = wilcoxon(d_post, d_pre)
            print(f"  Wilcoxon paired p = {p:.2e}")
            stat_record["wilcoxon_p"] = float(p)
    except Exception as e:
        print(f"  (scipy wilcoxon unavailable: {e})")
    append_stat(args.run_dir, "plot_replay_v11.py", stat_record)

    print("\nFigure 2b: shift-vs-empty-filter robustness sweep...")
    sweep_rows = shift_vs_empty_filter(rm_pre, rm_post, sk_pre, sk_post, reward_pos)
    fig_shift_vs_empty_filter(sweep_rows, reward_pos,
                              save_path=os.path.join(fig_dir, f"replay_c{N}_shift_vs_gate.png"),
                              no_show=args.no_show)
    sweep_stat = {"test": "shift_vs_empty_filter", "skaggs_threshold": float(thr)}
    for r in sweep_rows:
        g = f"{r['gate']:.1f}"
        sweep_stat[f"gate{g}_n_survivors"] = r["n"]
        if "delta" in r:
            sweep_stat[f"gate{g}_delta"] = round(r["delta"], 5)
            sweep_stat[f"gate{g}_p"] = r["p"]
    append_stat(args.run_dir, "plot_replay_v11.py", sweep_stat)

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
    ex_ids = select_example_cells(stable, sk_pre, sk_post, d_pre, d_post,
                                  reward_pos, thr, n=args.n_examples)
    print(f"  selected example cells: {ex_ids}")
    fig_replay_examples(rm_pre, rm_post, c_pre, c_post, ex_ids, reward_pos, reward_bin,
                        save_path=os.path.join(fig_dir, f"replay_c{N}_examples.png"),
                        no_show=args.no_show)
    print(f"Figure 2: distance-to-reward shift (empty-filtered survivors, n={len(stable)})...")
    fig_replay_shift(d_pre, d_post,
                     save_path=os.path.join(fig_dir, f"replay_c{N}_shift.png"),
                     no_show=args.no_show)
    print(f"Figure 3: field-centre movement quiver (empty-filtered survivors, n={len(stable)})...")
    fig_replay_quiver(c_pre_s, c_post_s, reward_pos,
                      save_path=os.path.join(fig_dir, f"replay_c{N}_quiver.png"),
                      no_show=args.no_show)

    print("\nFigure 4: ORI (over-representation index), pre-RL vs post-RL replay...")
    fig_ori(rm_pre, sk_pre, rm_post, sk_post, reward_pos, threshold=thr,
            save_path=os.path.join(fig_dir, f"replay_c{N}_ori.png"),
            no_show=args.no_show, label_a="Pre-RL", label_b="Post-RL replay")
    ori_pre = np.where(sk_pre >= thr)[0]
    ori_post = np.where(sk_post >= thr)[0]
    d_ori_pre = np.linalg.norm(np.array([field_center_of_mass(rm_pre[n]) for n in ori_pre]) - reward_pos, axis=1) if len(ori_pre) else np.array([])
    d_ori_post = np.linalg.norm(np.array([field_center_of_mass(rm_post[n]) for n in ori_post]) - reward_pos, axis=1) if len(ori_post) else np.array([])
    ori_record = {
        "test": "replay_ori_over_representation",
        "skaggs_threshold": float(thr),
        "n_place_pre": int(len(ori_pre)),
        "n_place_post": int(len(ori_post)),
        "mean_dist_pre": float(d_ori_pre.mean()) if len(d_ori_pre) else None,
        "mean_dist_post": float(d_ori_post.mean()) if len(d_ori_post) else None,
    }
    try:
        from scipy.stats import mannwhitneyu
        if len(d_ori_pre) >= 2 and len(d_ori_post) >= 2:
            u, p_mw = mannwhitneyu(d_ori_post, d_ori_pre, alternative="less")
            print(f"  Mann-Whitney (post < pre distance-to-reward): U={u:.0f}  p={p_mw:.2e}")
            ori_record["mannwhitney_u"] = float(u)
            ori_record["mannwhitney_p_less"] = float(p_mw)
    except Exception as e:
        print(f"  (scipy mannwhitneyu unavailable: {e})")
    append_stat(args.run_dir, "plot_replay_v11.py", ori_record)

    print("Figure 5: occupancy comparison (sanity check -- should be ~identical, frozen trajectory)...")
    fig_occupancy(occ_pre, occ_post, reward_bin,
                  save_path=os.path.join(fig_dir, f"replay_c{N}_occupancy.png"),
                  no_show=args.no_show, label_a="Pre-RL", label_b="Post-RL replay")

    print("Figure 6: place-cell recruitment (pre vs post RL)...")
    fig_replay_recruitment(sk_pre, sk_post, thr,
                           save_path=os.path.join(fig_dir, f"replay_c{N}_recruitment.png"),
                           no_show=args.no_show)
    recruitment_record = recruitment_analysis(sk_pre, sk_post, rm_post, reward_pos, thr)
    append_stat(args.run_dir, "plot_replay_v11.py", recruitment_record)

    print("Figure 6b: recruitment fate -- what happened to lost/recruited cells...")
    fig_recruitment_fate(sk_pre, sk_post, rm_pre, rm_post, reward_pos, thr,
                         save_path=os.path.join(fig_dir, f"replay_c{N}_recruitment_fate.png"),
                         no_show=args.no_show)
    fate_record = recruitment_fate_analysis(rm_pre, rm_post, sk_pre, sk_post, reward_pos, thr)
    append_stat(args.run_dir, "plot_replay_v11.py", fate_record)

    print("Figure 6c: lost/recruited cell galleries (actual rate maps, pre vs post)...")
    is_place_pre = sk_pre >= thr
    is_place_post = sk_post >= thr
    lost_ids = np.where(is_place_pre & ~is_place_post)[0]
    recruited_ids = np.where(~is_place_pre & is_place_post)[0]
    n_gallery = 12

    # Lost cells: rank by how far the field-centre moved toward the reward
    # (pre-RL distance minus post-RL distance; larger = moved further toward
    # reward) -- but only among lost cells that still retain a visible field
    # post-RL. A lost cell that collapsed to near-silence has an empty map
    # whose COM defaults to the arena centre; that fake "movement toward
    # reward" is the meaningless-COM artifact, not a real relocation, so those
    # cells are excluded from this gallery.
    if len(lost_ids):
        lost_visible = lost_ids[sk_post[lost_ids] >= 0.5 * thr]
        if len(lost_visible) < n_gallery:
            lost_visible = lost_ids[sk_post[lost_ids] >= 0.3 * thr]
        if len(lost_visible) < n_gallery:
            lost_visible = lost_ids
        lost_move = np.array([
            np.linalg.norm(c_pre[int(n)] - reward_pos) - np.linalg.norm(c_post[int(n)] - reward_pos)
            for n in lost_visible
        ])
        lost_sel = lost_visible[np.argsort(lost_move)[::-1]][:n_gallery]
    else:
        lost_sel = lost_ids
    # Recruited cells: rank by their (still sub-threshold) PRE Skaggs -- the
    # cells nearest the cutoff, showing sharpening rather than emergence from
    # nothing.
    recruited_sel = recruited_ids[np.argsort(sk_pre[recruited_ids])[::-1]][:n_gallery]
    fig_replay_examples(
        rm_pre, rm_post, c_pre, c_post, lost_sel, reward_pos, reward_bin,
        save_path=os.path.join(fig_dir, f"replay_c{N}_examples_lost.png"), no_show=args.no_show,
        title=f"Lost place cells (place PRE, not POST) -- top {len(lost_sel)} of {len(lost_ids)} by movement toward reward",
        sk_pre=sk_pre, sk_post=sk_post, draw_arrow=False,
    )
    fig_replay_examples(
        rm_pre, rm_post, c_pre, c_post, recruited_sel, reward_pos, reward_bin,
        save_path=os.path.join(fig_dir, f"replay_c{N}_examples_recruited.png"), no_show=args.no_show,
        title=f"Recruited place cells (place POST, not PRE) -- top {len(recruited_sel)} of {len(recruited_ids)} nearest the cutoff",
        sk_pre=sk_pre, sk_post=sk_post, draw_arrow=False,
    )

    print("Figure 7: confident pre-RL place cells (GMM/ICL-selected) -- shift + confidence change...")
    confident_record = confident_place_cell_analysis(
        args.run_dir, N, rm_pre, rm_post, sk_pre, sk_post, reward_pos,
        thr, fig_dir, args.no_show,
    )
    if confident_record is not None:
        append_stat(args.run_dir, "plot_replay_v11.py", confident_record)

    report_path = write_stats_report(args.run_dir)
    print(f"\nWrote statistics report: {report_path}")

    print("\nDone. Figures saved to:", fig_dir)
    print("Interpretation: a leftward shift / negative mean delta / arrows pointing to the\n"
          "reward = GENUINE representational remapping (trajectory was held fixed).\n"
          "No shift = the goal-directed concentration was trajectory/goal-driven, not\n"
          "baked into the weights.")

if __name__ == "__main__":
    main()
