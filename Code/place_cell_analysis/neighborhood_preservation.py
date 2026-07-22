#!/usr/bin/env python3
"""
Neighbourhood-preservation analysis of RL-driven representational change.

Question (from the PI): does the RL update reshape the representation *locally
around the reward*?  For each timestep we take its k nearest neighbours in the
PRE-RL representation and ask how many are still among its k nearest neighbours
POST-RL.  Low overlap = the local neighbourhood was reshuffled by RL; high
overlap = it was preserved.  We then plot mean overlap against the timestep's
distance-to-reward.

Prediction:
  * WITH the reward-prediction head (Positive_Results): overlap DROPS near the
    reward (RL reshapes the cognitive map locally around the goal) and recovers
    far away -> Spearman rho(overlap, distance) > 0.
  * WITHOUT the head (Negative_Results): overlap is FLAT vs distance (no
    reward-localised reshaping) -> rho ~ 0.

Data: the two replays of the SAME fixed trajectory through the RNN, before and
after the cycle-1 RL update:
    df_activations_replay_c1_pred    (pre-RL)
    df_activations_replay_c1_postrl  (post-RL)
Both are 50,000 timesteps x 512 neurons and share an identical (x, y, theta)
trajectory, so timestep t is the same physical state in both -> "are t's
neighbours preserved?" is well-defined.

Why this metric is honest: neighbourhoods are compared by IDENTITY (which
timesteps), so overlap is invariant to any global scaling/rotation/reflection
of either representation space -- exactly the ambiguities that made the raw PCA
figures hard to compare.  Only genuine local reshuffling lowers the overlap.
Temporally-adjacent samples (|dt| <= --min_time_gap) are removed from the
candidate neighbour sets so we measure representational, not trivial temporal,
neighbourhood structure (consecutive timesteps sit at nearby positions and are
neighbours in BOTH spaces regardless of RL).

Outputs (in <run_dir>/figures and <run_dir>/data):
  neighborhood_preservation_comparison.{png,jpg,pdf}   (with-head vs without-head)
  neighborhood_preservation_<run>.{png,jpg,pdf}        (per-run detail, hexbin)
  neighborhood_preservation.json                       (per-run overlap-vs-distance + stats)

Example:
  python3 neighborhood_preservation.py \
      --run_dir ../Positive_Results --compare_dir ../Negative_Results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.neighbors import NearestNeighbors

from pc_common import resolve_reward_pos, save_all, load_run_config

PRE_STATE = "replay_c1_pred"
POST_STATE = "replay_c1_postrl"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_activation_matrix(run_dir, period):
    """(act_mat [T, N] float64, df_behavior aligned to those T rows)."""
    data_dir = os.path.join(run_dir, "data")
    beh = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{period}.pkl"))
    act = pd.read_pickle(os.path.join(data_dir, f"df_activations_{period}.pkl"))
    n_steps = len(beh)
    n_neurons = act["neuron_id"].nunique()
    if len(act) == n_steps * n_neurons:
        # dense/ordered long format -> reshape directly (fast path)
        act_mat = act["activity"].values.reshape(n_steps, n_neurons)
        beh_sorted = beh.sort_values("timestep").reset_index(drop=True)
    else:
        piv = act.pivot(index="timestep", columns="neuron_id", values="activity").fillna(0.0)
        beh_sorted = beh[beh["timestep"].isin(piv.index)].sort_values("timestep").reset_index(drop=True)
        act_mat = piv.reindex(beh_sorted["timestep"]).values
    return act_mat.astype(np.float64), beh_sorted


# --------------------------------------------------------------------------- #
# neighbourhood-preservation core
# --------------------------------------------------------------------------- #
def ordered_valid_neighbors(A, timesteps, k_want, min_time_gap):
    """For each row of A, an int array [S, k_want] of neighbour row-indices in
    increasing-distance order, with self and any neighbour whose timestep is
    within min_time_gap of the anchor removed.  -1 pads the tail if fewer than
    k_want valid neighbours were found."""
    S = A.shape[0]
    stride = max(1.0, S and (timesteps.max() - timesteps.min() + 1) / S)
    # fetch enough candidates that, after dropping ~2*gap/stride temporal
    # neighbours (and self), k_want valid ones remain.
    est_temporal = int(np.ceil(2.0 * min_time_gap / stride)) + 5
    n_cand = int(min(S, k_want + est_temporal + 15))
    nn = NearestNeighbors(n_neighbors=n_cand, algorithm="auto").fit(A)
    _, idx = nn.kneighbors(A)  # [S, n_cand]; column 0 is self
    out = np.full((S, k_want), -1, dtype=np.int64)
    n_full = 0
    for i in range(S):
        a_ts = timesteps[i]
        keep = []
        for j in idx[i]:
            if j == i:
                continue
            if abs(timesteps[j] - a_ts) <= min_time_gap:
                continue
            keep.append(j)
            if len(keep) == k_want:
                break
        out[i, : len(keep)] = keep
        n_full += len(keep) == k_want
    frac_full = n_full / S
    return out, frac_full


def mean_nn_distance(A, nb, k):
    """Per-anchor mean distance to its k kept neighbours in A's space -> a
    local-density proxy (large = sparse).  Used to check the metric isn't a
    density artifact: in a sparse region a point's neighbour SET is more
    'forced' (few nearby candidates), which could inflate overlap regardless
    of RL.  We partial this out in the controls."""
    S = A.shape[0]
    dens = np.full(S, np.nan)
    for i in range(S):
        js = nb[i, :k]
        js = js[js >= 0]
        if len(js):
            dens[i] = np.sqrt(((A[i] - A[js]) ** 2).sum(axis=1)).mean()
    return dens


def overlap_at_k(pre_nb, post_nb, k):
    """Per-anchor fraction of the k nearest pre-RL neighbours that are still
    among the k nearest post-RL neighbours."""
    S = pre_nb.shape[0]
    ov = np.empty(S, dtype=float)
    for i in range(S):
        a = set(int(v) for v in pre_nb[i, :k] if v >= 0)
        b = set(int(v) for v in post_nb[i, :k] if v >= 0)
        denom = k
        ov[i] = len(a & b) / denom
    return ov


def analyze_run(run_dir, reward, k_list, n_sample, min_time_gap, seed):
    cfg = load_run_config(run_dir)
    has_head = bool(cfg.get("reward_pred_head", False))
    label = "with reward head" if has_head else "without reward head"

    pre, beh_pre = load_activation_matrix(run_dir, PRE_STATE)
    post, beh_post = load_activation_matrix(run_dir, POST_STATE)
    if not np.allclose(beh_pre[["x", "y"]].values, beh_post[["x", "y"]].values):
        raise ValueError(f"{run_dir}: pre/post replay trajectories differ; not comparable.")
    T = pre.shape[0]

    rng = np.random.RandomState(seed)
    S_idx = np.sort(rng.choice(T, size=int(min(n_sample, T)), replace=False))
    ts = beh_pre["timestep"].values[S_idx].astype(np.int64)
    x = beh_pre["x"].values[S_idx].astype(float)
    y = beh_pre["y"].values[S_idx].astype(float)
    dist = np.sqrt((x - reward[0]) ** 2 + (y - reward[1]) ** 2)

    A_pre, A_post = pre[S_idx], post[S_idx]
    kmax = int(max(k_list))
    pre_nb, ff_pre = ordered_valid_neighbors(A_pre, ts, kmax, min_time_gap)
    post_nb, ff_post = ordered_valid_neighbors(A_post, ts, kmax, min_time_gap)

    overlap_by_k = {int(k): overlap_at_k(pre_nb, post_nb, int(k)) for k in k_list}
    local_density_pre = mean_nn_distance(A_pre, pre_nb, min(25, kmax))
    rho_by_k = {}
    for k in k_list:
        rho, p = spearmanr(overlap_by_k[int(k)], dist)
        rho_by_k[int(k)] = (float(rho), float(p))

    print(f"[{os.path.basename(run_dir)}] {label}: T={T}, sample={len(S_idx)}, "
          f"valid-neighbour fill pre/post={ff_pre:.2f}/{ff_post:.2f}")
    print("    k    Spearman rho(overlap, distance)   p")
    for k in k_list:
        rho, p = rho_by_k[int(k)]
        print(f"   {int(k):>3}    {rho:>+7.3f}                        {p:.2e}")

    return {
        "run_dir": run_dir,
        "run_name": os.path.basename(os.path.normpath(run_dir)),
        "has_head": has_head,
        "label": label,
        "reward": [float(reward[0]), float(reward[1])],
        "n_sample": int(len(S_idx)),
        "min_time_gap": int(min_time_gap),
        "k_list": [int(k) for k in k_list],
        "dist": dist,
        "x": x,
        "y": y,
        "ts": ts,
        "local_density_pre": local_density_pre,
        "overlap_by_k": overlap_by_k,
        "rho_by_k": rho_by_k,
        "beh_full_xy": (beh_pre["x"].values.astype(float), beh_pre["y"].values.astype(float)),
    }


def save_points_npz(res, path):
    """Per-anchor arrays so control analyses (spatial maps, fake-reward,
    occupancy matching) can run without recomputing the k-NN."""
    np.savez(
        path,
        x=res["x"], y=res["y"], ts=res["ts"], dist=res["dist"],
        local_density_pre=res["local_density_pre"],
        reward=np.asarray(res["reward"], dtype=float),
        beh_x=res["beh_full_xy"][0], beh_y=res["beh_full_xy"][1],
        k_list=np.asarray(res["k_list"], dtype=int),
        has_head=np.asarray(res["has_head"]),
        **{f"overlap_k{int(k)}": res["overlap_by_k"][int(k)] for k in res["k_list"]},
    )


# --------------------------------------------------------------------------- #
# stats / binning
# --------------------------------------------------------------------------- #
def binned_mean_ci(dist, ov, bins):
    idx = np.digitize(dist, bins) - 1
    centers, means, cis, counts = [], [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        n = int(m.sum())
        if n < 5:
            continue
        v = ov[m]
        centers.append(0.5 * (bins[b] + bins[b + 1]))
        means.append(float(v.mean()))
        cis.append(float(1.96 * v.std(ddof=1) / np.sqrt(n)))
        counts.append(n)
    return (np.array(centers), np.array(means), np.array(cis), np.array(counts))


def near_far_test(dist, ov, q=0.2):
    lo, hi = np.quantile(dist, q), np.quantile(dist, 1 - q)
    near, far = ov[dist <= lo], ov[dist >= hi]
    if len(near) < 5 or len(far) < 5:
        return None
    U, p = mannwhitneyu(near, far, alternative="two-sided")
    return {
        "near_median": float(np.median(near)), "far_median": float(np.median(far)),
        "near_mean": float(near.mean()), "far_mean": float(far.mean()),
        "n_near": int(len(near)), "n_far": int(len(far)),
        "U": float(U), "p": float(p),
    }


def bootstrap_rho_diff(ov_w, d_w, ov_n, d_n, n_boot, seed):
    """Bootstrap CI on Spearman rho(with head) - rho(without head)."""
    rng = np.random.RandomState(seed)
    nw, nn_ = len(ov_w), len(ov_n)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        iw = rng.randint(0, nw, nw)
        ino = rng.randint(0, nn_, nn_)
        diffs[b] = spearmanr(ov_w[iw], d_w[iw])[0] - spearmanr(ov_n[ino], d_n[ino])[0]
    return float(diffs.mean()), [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def occupancy_pct(xy, reward, bins):
    x, y = xy
    d = np.sqrt((x - reward[0]) ** 2 + (y - reward[1]) ** 2)
    cnt, _ = np.histogram(d, bins=bins)
    return 100.0 * cnt / max(cnt.sum(), 1)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def per_run_figure(res, k_main, n_bins, out_base):
    dist, ov = res["dist"], res["overlap_by_k"][k_main]
    bins = np.linspace(0, dist.max(), n_bins + 1)
    c, m, ci, _ = binned_mean_ci(dist, ov, bins)
    rho, p = res["rho_by_k"][k_main]

    fig, (axh, axc) = plt.subplots(1, 2, figsize=(13, 5))
    hb = axh.hexbin(dist, ov, gridsize=28, cmap="magma", mincnt=1)
    fig.colorbar(hb, ax=axh, label="count")
    axh.set(xlabel="distance to reward", ylabel=f"pre->post {k_main}-NN overlap",
            title=f"{res['run_name']} ({res['label']})\nper-timestep overlap (k={k_main})")
    axc.plot(c, m, "-o", color="tab:blue")
    axc.fill_between(c, m - ci, m + ci, alpha=0.25, color="tab:blue")
    axc.set(xlabel="distance to reward", ylabel=f"mean {k_main}-NN overlap",
            title=f"binned mean +/- 95% CI\nSpearman rho={rho:+.3f} (p={p:.1e})")
    axc.grid(alpha=0.3)
    fig.tight_layout()
    save_all(out_base, fig=fig)
    plt.close(fig)


def comparison_figure(res_w, res_n, k_main, n_bins, out_base, rho_diff, rho_ci):
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.26)

    # --- main panel: overlap vs distance, both runs ---
    ax = fig.add_subplot(gs[0, :])
    for res, color in ((res_w, "tab:red"), (res_n, "tab:blue")):
        dist, ov = res["dist"], res["overlap_by_k"][k_main]
        bins = np.linspace(0, dist.max(), n_bins + 1)
        c, m, ci, _ = binned_mean_ci(dist, ov, bins)
        rho, p = res["rho_by_k"][k_main]
        ax.plot(c, m, "-o", color=color,
                label=f"{res['run_name']} ({res['label']}) — rho={rho:+.3f}, p={p:.1e}")
        ax.fill_between(c, m - ci, m + ci, alpha=0.20, color=color)
    ax.set_xlabel("distance to reward")
    ax.set_ylabel(f"mean pre->post {k_main}-NN overlap")
    ax.set_title(f"Neighbourhood preservation vs distance to reward (k={k_main})\n"
                 f"prediction: overlap dips near reward WITH head, flat WITHOUT")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # --- k-sweep robustness: Spearman rho vs k ---
    axk = fig.add_subplot(gs[1, 0])
    for res, color in ((res_w, "tab:red"), (res_n, "tab:blue")):
        ks = res["k_list"]
        rhos = [res["rho_by_k"][int(k)][0] for k in ks]
        axk.plot(ks, rhos, "-o", color=color, label=res["label"])
    axk.axhline(0, color="k", lw=0.8, ls="--")
    axk.set(xlabel="k (neighbourhood size)", ylabel="Spearman rho(overlap, dist)",
            title="robustness across k")
    axk.legend(fontsize=8)
    axk.grid(alpha=0.3)

    # --- occupancy control ---
    axo = fig.add_subplot(gs[1, 1])
    for res, color in ((res_w, "tab:red"), (res_n, "tab:blue")):
        dist = res["dist"]
        bins = np.linspace(0, dist.max(), n_bins + 1)
        occ = occupancy_pct(res["beh_full_xy"], res["reward"], bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        axo.plot(centers, occ, "-s", color=color, alpha=0.8, label=res["label"])
    axo.set(xlabel="distance to reward", ylabel="occupancy (%)",
            title="occupancy control (identical pre/post)")
    axo.legend(fontsize=8)
    axo.grid(alpha=0.3)

    verdict = ("consistent with prediction" if (rho_ci[0] > 0)
               else "NOT clearly supported")
    fig.suptitle(
        f"With-head rho - without-head rho = {rho_diff:+.3f}  "
        f"(95% CI [{rho_ci[0]:+.3f}, {rho_ci[1]:+.3f}])  ->  {verdict}",
        fontsize=12, y=0.98)
    save_all(out_base, fig=fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default="../Positive_Results",
                   help="primary run (the WITH-head run, for the comparison colour/order).")
    p.add_argument("--compare_dir", default="../Negative_Results",
                   help="second run to overlay (the WITHOUT-head run). Set to '' for single-run only.")
    p.add_argument("--reward", type=float, nargs=2, default=None,
                   help="reward xy for --run_dir (default: read from config.json).")
    p.add_argument("--compare_reward", type=float, nargs=2, default=None,
                   help="reward xy for --compare_dir (default: read from its config.json).")
    p.add_argument("--k_list", type=int, nargs="+", default=[3, 10, 25, 50],
                   help="neighbourhood sizes to evaluate (the k-sweep).")
    p.add_argument("--k_main", type=int, default=25, help="k used for the main overlap-vs-distance curve.")
    p.add_argument("--n_sample", type=int, default=8000, help="anchor/candidate timesteps subsampled per run.")
    p.add_argument("--min_time_gap", type=int, default=50,
                   help="exclude neighbours within this many timesteps of the anchor.")
    p.add_argument("--n_bins", type=int, default=12, help="distance bins for the mean+CI curve.")
    p.add_argument("--n_boot", type=int, default=1000, help="bootstrap resamples for the rho-difference CI.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None, help="figure/JSON output dir (default: <run_dir>/figures).")
    return p.parse_args()


def _json_safe(res):
    """Drop bulky arrays; keep scalars + binned curve for the JSON summary."""
    dist = res["dist"]
    out = {k: res[k] for k in ("run_name", "has_head", "label", "reward", "n_sample",
                               "min_time_gap", "k_list", "rho_by_k")}
    out["near_far_by_k"] = {int(k): near_far_test(dist, res["overlap_by_k"][int(k)])
                            for k in res["k_list"]}
    out["overall_mean_overlap_by_k"] = {int(k): float(res["overlap_by_k"][int(k)].mean())
                                        for k in res["k_list"]}
    return out


def main():
    args = parse_args()
    if args.k_main not in args.k_list:
        args.k_list = sorted(set(args.k_list) | {args.k_main})

    out_dir = args.out_dir or os.path.join(args.run_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(args.run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    reward_w = resolve_reward_pos(args.run_dir, args.reward)
    res_w = analyze_run(args.run_dir, reward_w, args.k_list, args.n_sample,
                        args.min_time_gap, args.seed)
    per_run_figure(res_w, args.k_main, args.n_bins,
                   os.path.join(out_dir, f"neighborhood_preservation_{res_w['run_name']}"))
    save_points_npz(res_w, os.path.join(data_dir, f"neighborhood_preservation_points_{res_w['run_name']}.npz"))

    summary = {"primary": _json_safe(res_w), "params": vars(args)}

    if args.compare_dir:
        reward_n = resolve_reward_pos(args.compare_dir, args.compare_reward)
        res_n = analyze_run(args.compare_dir, reward_n, args.k_list, args.n_sample,
                            args.min_time_gap, args.seed)
        per_run_figure(res_n, args.k_main, args.n_bins,
                       os.path.join(out_dir, f"neighborhood_preservation_{res_n['run_name']}"))
        compare_data_dir = os.path.join(args.compare_dir, "data")
        os.makedirs(compare_data_dir, exist_ok=True)
        save_points_npz(res_n, os.path.join(compare_data_dir, f"neighborhood_preservation_points_{res_n['run_name']}.npz"))

        rho_diff, rho_ci = bootstrap_rho_diff(
            res_w["overlap_by_k"][args.k_main], res_w["dist"],
            res_n["overlap_by_k"][args.k_main], res_n["dist"],
            args.n_boot, args.seed)
        comparison_figure(res_w, res_n, args.k_main, args.n_bins,
                          os.path.join(out_dir, "neighborhood_preservation_comparison"),
                          rho_diff, rho_ci)
        summary["compare"] = _json_safe(res_n)
        summary["rho_diff_with_minus_without"] = {
            "k": args.k_main, "estimate": rho_diff, "ci95": rho_ci}
        print(f"\nWith-head rho - without-head rho (k={args.k_main}) = "
              f"{rho_diff:+.3f}  95% CI [{rho_ci[0]:+.3f}, {rho_ci[1]:+.3f}]")

    json_path = os.path.join(data_dir, "neighborhood_preservation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir}/neighborhood_preservation_comparison.(png|jpg|pdf) and {json_path}")


if __name__ == "__main__":
    main()
