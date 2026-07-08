"""
Reward-phenomenology analyses: the three canonical hippocampal reward effects
from the paper's introduction that were not yet quantified with
literature-standard, single-cell metrics.

    1. Concentration / OVER-REPRESENTATION of place fields near reward
       (Hollup 2001; Dupret 2010; Gauthier & Tank 2018). Metric: the DENSITY
       of place-field centres as a function of distance to reward, and an
       over-representation ratio = observed near-reward field density /
       density expected if fields tiled the arena uniformly.

    2. REWARD-DISTANCE-MODULATED activity / ramping (Gauthier & Tank 2018;
       Sarel 2017). Metric: per-cell tuning of activity to distance-to-reward
       -- a ramping index = corr(activity, -distance) (positive = more active
       nearer reward) -- and the population activity-vs-distance curve.

    3. GOAL-VECTOR representation (Sarel 2017; Ormond 2015). Metric: per-cell
       tuning of activity to the EGOCENTRIC bearing to the goal (0 = goal
       straight ahead), summarised by the mean resultant length (MRL) of the
       activity-weighted bearing tuning curve -- "goal-direction cells" -- plus
       goal-distance tuning; cells tuned to both are "goal-vector cells".

Design, following the paper's central logic (Fig. 2 vs Fig. 3): every metric is
computed primarily on the FROZEN-WEIGHT REPLAY pair -- an identical, unbiased
random-walk trajectory passed through the pre-RL weights (replay_c{K}_pred) and
the post-RL weights (replay_c{K}_postrl). Because occupancy, and therefore the
sampling of distance-to-reward and bearing-to-reward, is IDENTICAL in both
passes, any pre->post change in these tuning metrics is attributable to the
weight change alone, not to reward-biased behaviour. Positive_Results (reward
head on, reward off-centre at (0.3,0.3)) is expected to develop these effects
post-RL; Negative_Results (task RL only, reward at arena centre) is the null
control.

Confound guards (matching earlier de-confounding work in this project):
  * Field CENTRES use the smoothed rate-map PEAK (argmax), not the centre of
    mass, so a diffuse/near-empty map does not fake a centre at the arena
    middle (the metric the paper explicitly trusts for the off-centre reward).
  * Over-representation counts fields only for cells that are genuine place
    cells (Skaggs >= threshold) in that state.
  * Paired pre->post tuning tests are restricted to "survivors" -- cells that
    are place cells (Skaggs >= threshold) in BOTH states -- so cells that are
    empty in one pass cannot inject meaningless tuning values.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu

from pc_common import (
    resolve_reward_pos,
    save_all,
    append_stat,
    write_stats_report,
    PLACE_CELL_SKAGGS_MIN,
)
from plot_goal_directed_maps import build_rate_maps, smooth_and_norm

M = 32
HIDDEN = 512


# --------------------------------------------------------------------------- #
# Loading / geometry
# --------------------------------------------------------------------------- #
def load_state(data_dir, state, m=M, hidden_dim=HIDDEN):
    """Returns (X, df_beh, rate_maps, occupancy, skaggs) for a v11 state.

    X is a (T, hidden_dim) activity matrix, X[t, neuron]. Relies on the v11
    activation layout (neuron-minor within timestep), verified against the
    data, and identical to build_rate_maps' reshape.
    """
    df_beh = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{state}.pkl"))
    df_act = pd.read_pickle(os.path.join(data_dir, f"df_activations_{state}.pkl"))
    n_steps = len(df_beh)
    X = df_act["activity"].values.reshape(n_steps, hidden_dim).astype(np.float32)
    rate_maps, occupancy = build_rate_maps(df_beh, df_act, m=m, hidden_dim=hidden_dim)
    sk_path = os.path.join(data_dir, f"df_skaggs_{state}.pkl")
    skaggs = pd.read_pickle(sk_path).sort_values("neuron_id")["skaggs_score"].values
    return X, df_beh, rate_maps, occupancy, skaggs


def geometry(df_beh, reward_pos):
    """Per-timestep distance to reward and egocentric bearing to reward.

    ego_bearing in [-pi, pi], 0 = reward straight ahead of current heading.
    """
    x = df_beh["x"].values
    y = df_beh["y"].values
    theta = df_beh["theta"].values
    dist = np.sqrt((x - reward_pos[0]) ** 2 + (y - reward_pos[1]) ** 2)
    target_angle = np.arctan2(reward_pos[1] - y, reward_pos[0] - x)
    ego = np.arctan2(np.sin(target_angle - theta), np.cos(target_angle - theta))
    return dist, ego


def field_peaks(rate_maps, ids, m=M):
    """Smoothed-rate-map peak (argmax) location in [0,1]^2 for each id."""
    out = np.full((len(ids), 2), np.nan)
    for i, n in enumerate(ids):
        s = smooth_and_norm(rate_maps[int(n)])
        if s.max() <= 0:
            continue
        yb, xb = np.unravel_index(np.argmax(s), s.shape)
        # rate_maps are (m, m) indexed [x_bin, y_bin] (see build_rate_maps),
        # so axis 0 is x and axis 1 is y.
        out[i] = [yb / m + 0.5 / m, xb / m + 0.5 / m]  # (x, y)
    return out


# --------------------------------------------------------------------------- #
# 1. Over-representation of place-field centres near reward
# --------------------------------------------------------------------------- #
def over_representation(rate_maps, skaggs, reward_pos, thr, r_near=0.2, nbins=10):
    """Radial density of place-field peaks vs distance to reward.

    Place cells = Skaggs >= thr in this state. Returns a dict with the radial
    density profile, the uniform-expectation profile, and the near-reward
    over-representation ratio (observed field fraction within r_near divided by
    the arena-area fraction within r_near).
    """
    place_ids = np.where(skaggs >= thr)[0]
    peaks = field_peaks(rate_maps, place_ids)
    peaks = peaks[np.isfinite(peaks).all(axis=1)]
    d = np.linalg.norm(peaks - reward_pos, axis=1)

    # Max distance reachable in the unit square from the reward corner.
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    d_max = np.linalg.norm(corners - reward_pos, axis=1).max()
    edges = np.linspace(0, d_max, nbins + 1)
    counts, _ = np.histogram(d, bins=edges)

    # Expected counts if peaks were uniform over the arena: proportional to the
    # arena AREA in each distance annulus (estimated by Monte-Carlo, since the
    # square clips the annuli).
    rng = np.random.default_rng(0)
    pts = rng.random((200000, 2))
    dd = np.linalg.norm(pts - reward_pos, axis=1)
    area_frac, _ = np.histogram(dd, bins=edges)
    area_frac = area_frac / area_frac.sum()
    expected = area_frac * counts.sum()

    centres = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(expected > 0, counts / expected, np.nan)

    # Near-reward over-representation index.
    frac_fields_near = float((d < r_near).mean()) if len(d) else np.nan
    frac_area_near = float((dd < r_near).mean())
    over_index = frac_fields_near / frac_area_near if frac_area_near > 0 else np.nan

    return {
        "n_place": len(place_ids),
        "n_fields": len(d),
        "dist": d,
        "bin_centres": centres,
        "density_ratio": ratio,  # observed / uniform-expected per annulus
        "r_near": r_near,
        "frac_fields_near": frac_fields_near,
        "frac_area_near": frac_area_near,
        "over_representation_index": over_index,
        "mean_dist": float(d.mean()) if len(d) else np.nan,
    }


# --------------------------------------------------------------------------- #
# 2. Reward-distance-modulated activity (ramping)
# --------------------------------------------------------------------------- #
def _col_corr(X, v):
    """Pearson correlation of each column of X (T,N) with vector v (T,)."""
    v = v - v.mean()
    vn = np.sqrt((v * v).sum())
    Xc = X - X.mean(axis=0, keepdims=True)
    Xn = np.sqrt((Xc * Xc).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (Xc * v[:, None]).sum(axis=0) / (Xn * vn)
    return np.where(np.isfinite(r), r, 0.0)


def distance_modulation(X, dist, nbins=12):
    """Per-neuron ramping index = corr(activity, -distance) (positive = more
    active nearer reward) and per-neuron activity-vs-distance tuning curve.
    """
    ramping = _col_corr(X, -dist)  # (N,)
    edges = np.linspace(dist.min(), dist.max(), nbins + 1)
    which = np.clip(np.digitize(dist, edges) - 1, 0, nbins - 1)
    onehot = np.zeros((len(dist), nbins), dtype=np.float32)
    onehot[np.arange(len(dist)), which] = 1.0
    counts = onehot.sum(axis=0)
    curve = (onehot.T @ X) / np.maximum(counts[:, None], 1)  # (nbins, N)
    return ramping, curve.T, 0.5 * (edges[:-1] + edges[1:])  # curve (N, nbins)


# --------------------------------------------------------------------------- #
# 3. Goal-vector (egocentric goal-direction) tuning
# --------------------------------------------------------------------------- #
def circular_tuning(X, angle, nbins=16):
    """Activity-weighted circular tuning curve of each neuron to `angle`
    (radians in [-pi, pi]) and its mean resultant length (MRL). Returns
    (mrl (N,), pref (N,), curve (N,nbins), bin_centres (nbins,)).

    Used for both egocentric goal-direction (angle = bearing to reward) and,
    as a control, allocentric head direction (angle = heading theta).
    """
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    which = np.clip(np.digitize(angle, edges) - 1, 0, nbins - 1)
    onehot = np.zeros((len(angle), nbins), dtype=np.float32)
    onehot[np.arange(len(angle)), which] = 1.0
    counts = onehot.sum(axis=0)
    curve = ((onehot.T @ X) / np.maximum(counts[:, None], 1)).T  # (N, nbins)
    phi = 0.5 * (edges[:-1] + edges[1:])
    vec = (curve * np.exp(1j * phi)[None, :]).sum(axis=1)
    tot = curve.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mrl = np.where(tot > 0, np.abs(vec) / tot, 0.0)
    return mrl, np.angle(vec), curve, phi


def bearing_tuning(X, ego, nbins=16):
    return circular_tuning(X, ego, nbins=nbins)


def bearing_tuning_null(X, ego, nbins=16, n_null=100, seed=0):
    """Cyclic-shift null distribution of the bearing MRL, preserving the
    temporal autocorrelation of the random walk. Returns the per-neuron 95th
    percentile MRL threshold (N,)."""
    rng = np.random.default_rng(seed)
    T = len(ego)
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    phi = 0.5 * (edges[:-1] + edges[1:])
    null = np.empty((n_null, X.shape[1]), dtype=np.float32)
    for s in range(n_null):
        shift = int(rng.integers(1, T))
        e = np.roll(ego, shift)
        which = np.clip(np.digitize(e, edges) - 1, 0, nbins - 1)
        onehot = np.zeros((T, nbins), dtype=np.float32)
        onehot[np.arange(T), which] = 1.0
        counts = onehot.sum(axis=0)
        curve = ((onehot.T @ X) / np.maximum(counts[:, None], 1)).T
        vec = (curve * np.exp(1j * phi)[None, :]).sum(axis=1)
        tot = curve.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            null[s] = np.where(tot > 0, np.abs(vec) / tot, 0.0)
    return np.percentile(null, 95, axis=0)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_over_representation(pre, post, reward_pos, pre_lab, post_lab, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) density-ratio vs distance
    ax = axes[0]
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="uniform (no over-rep.)")
    ax.plot(pre["bin_centres"], pre["density_ratio"], "-o", color="steelblue", label=pre_lab)
    ax.plot(post["bin_centres"], post["density_ratio"], "-o", color="crimson", label=post_lab)
    ax.axvline(pre["r_near"], color="green", ls=":", lw=1)
    ax.set_xlabel("distance from reward")
    ax.set_ylabel("field-centre density / uniform")
    ax.set_title("Over-representation profile\n(>1 = fields accumulate here)")
    ax.legend(fontsize=8)

    # (b) distance-to-reward distributions of field peaks
    ax = axes[1]
    bins = np.linspace(0, max(pre["dist"].max(), post["dist"].max()), 20)
    ax.hist(pre["dist"], bins=bins, alpha=0.5, color="steelblue", density=True, label=pre_lab)
    ax.hist(post["dist"], bins=bins, alpha=0.5, color="crimson", density=True, label=post_lab)
    ax.set_xlabel("field-peak distance to reward")
    ax.set_ylabel("density")
    ax.set_title("Field-peak distance distribution")
    ax.legend(fontsize=8)

    # (c) over-representation index bar
    ax = axes[2]
    vals = [pre["over_representation_index"], post["over_representation_index"]]
    ax.bar([0, 1], vals, color=["steelblue", "crimson"])
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([pre_lab, post_lab])
    ax.set_ylabel(f"over-representation index (r<{pre['r_near']})")
    ax.set_title("Near-reward over-representation\n(>1 = denser than uniform)")

    fig.suptitle("1. Concentration / over-representation of place fields near reward", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_all(save_path)
    plt.close(fig)


def fig_distance_modulation(ramp_pre, ramp_post, curve_pre, curve_post, dbins,
                            surv, pre_lab, post_lab, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) population activity-vs-distance curve (place-cell survivors, normalised)
    ax = axes[0]
    def _norm(c):
        c = c[surv].mean(axis=0)
        return c / (c.max() + 1e-12)
    ax.plot(dbins, _norm(curve_pre), "-o", color="steelblue", label=pre_lab)
    ax.plot(dbins, _norm(curve_post), "-o", color="crimson", label=post_lab)
    ax.set_xlabel("distance to reward")
    ax.set_ylabel("mean activity (norm.)")
    ax.set_title("Population activity vs distance\n(rise near reward = ramping)")
    ax.legend(fontsize=8)

    # (b) ramping-index distributions (survivors)
    ax = axes[1]
    bins = np.linspace(-1, 1, 41)
    ax.hist(ramp_pre[surv], bins=bins, alpha=0.5, color="steelblue", label=pre_lab)
    ax.hist(ramp_post[surv], bins=bins, alpha=0.5, color="crimson", label=post_lab)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("ramping index  corr(activity, -distance)")
    ax.set_ylabel("# cells")
    ax.set_title("Ramping index (place-cell survivors)\n(>0 = more active near reward)")
    ax.legend(fontsize=8)

    # (c) paired change per cell
    ax = axes[2]
    ax.scatter(ramp_pre[surv], ramp_post[surv], s=8, alpha=0.4, color="purple")
    lim = [-1, 1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"ramping index {pre_lab}")
    ax.set_ylabel(f"ramping index {post_lab}")
    ax.set_title(f"Paired change\nmean {ramp_pre[surv].mean():+.3f} -> {ramp_post[surv].mean():+.3f}")

    fig.suptitle("3. Reward-distance-modulated activity (ramping)", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_all(save_path)
    plt.close(fig)


def fig_goal_vector(mrl_pre, mrl_post, curve_post, phi, tuned_post, surv,
                    pre_lab, post_lab, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), subplot_kw=None)
    axes[2].remove()
    axp = fig.add_subplot(1, 3, 3, projection="polar")

    # (a) goal-direction MRL distribution (survivors)
    ax = axes[0]
    bins = np.linspace(0, max(0.5, np.nanmax(mrl_post[surv]) if surv.any() else 0.5), 30)
    ax.hist(mrl_pre[surv], bins=bins, alpha=0.5, color="steelblue", label=pre_lab)
    ax.hist(mrl_post[surv], bins=bins, alpha=0.5, color="crimson", label=post_lab)
    ax.set_xlabel("goal-direction tuning strength (MRL)")
    ax.set_ylabel("# cells")
    ax.set_title(f"Goal-direction tuning\nmean {mrl_pre[surv].mean():.3f} -> {mrl_post[surv].mean():.3f}")
    ax.legend(fontsize=8)

    # (b) paired change
    ax = axes[1]
    ax.scatter(mrl_pre[surv], mrl_post[surv], s=8, alpha=0.4, color="purple")
    lim = [0, max(mrl_pre[surv].max(), mrl_post[surv].max()) * 1.05 + 1e-6]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"MRL {pre_lab}")
    ax.set_ylabel(f"MRL {post_lab}")
    ax.set_title("Paired goal-direction tuning change")

    # (c) polar tuning curves of the most goal-direction-tuned post cells.
    # Exclude degenerate cells whose MRL is high only because they fire in ~1
    # bearing bin (near-silent/sparse) -- require activity spread across at
    # least half the bins -- so the examples are genuinely broadly-tuned
    # goal-direction cells, not single-bin artifacts.
    nbins = curve_post.shape[1]
    spread = (curve_post[surv] > 0.1 * curve_post[surv].max(axis=1, keepdims=True)).sum(axis=1)
    eligible = surv[spread >= nbins // 2]
    if len(eligible) < 6:
        eligible = surv
    order = np.argsort(mrl_post[eligible])[::-1][:6]
    top = eligible[order]
    phi_closed = np.concatenate([phi, phi[:1]])
    for n in top:
        c = curve_post[n]
        c = c / (c.max() + 1e-12)
        axp.plot(phi_closed, np.concatenate([c, c[:1]]), lw=1.2, alpha=0.8)
    axp.set_theta_zero_location("N")
    axp.set_title(f"Top post-RL goal-direction cells\n(0 = reward ahead)\n{int(tuned_post)} tuned cells", fontsize=9)

    fig.suptitle("4. Goal-vector / goal-direction representation", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_all(save_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Reward-phenomenology metrics (over-representation, "
                                            "distance modulation, goal-vector) on the frozen-replay pair.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--pre", default=None, help="pre-RL state (default replay_c{K}_pred).")
    p.add_argument("--post", default=None, help="post-RL state (default replay_c{K}_postrl).")
    p.add_argument("--reward", nargs=2, type=float, default=None)
    p.add_argument("--skaggs_thr", type=float, default=PLACE_CELL_SKAGGS_MIN)
    p.add_argument("--n_null", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    data_dir = os.path.join(run_dir, "data")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    reward_pos = resolve_reward_pos(run_dir, args.reward)
    thr = args.skaggs_thr

    import json
    with open(os.path.join(run_dir, "config.json")) as f:
        K = int(json.load(f).get("replay_cycle", 1))
    pre_state = args.pre or f"replay_c{K}_pred"
    post_state = args.post or f"replay_c{K}_postrl"
    pre_lab, post_lab = "Pre-RL", "Post-RL"

    print(f"=== Reward phenomenology: {run_dir} ===")
    print(f"  reward={tuple(np.round(reward_pos,3))}  pre='{pre_state}'  post='{post_state}'  Skaggs>={thr}")

    Xpre, beh_pre, rm_pre, _, sk_pre = load_state(data_dir, pre_state)
    Xpost, beh_post, rm_post, _, sk_post = load_state(data_dir, post_state)
    dist, ego = geometry(beh_pre, reward_pos)  # frozen trajectory => identical for both

    surv = np.where((sk_pre >= thr) & (sk_post >= thr))[0]
    print(f"  place cells: pre={(sk_pre>=thr).sum()}  post={(sk_post>=thr).sum()}  survivors(both)={len(surv)}")

    # --- 1. Over-representation ---
    orp_pre = over_representation(rm_pre, sk_pre, reward_pos, thr)
    orp_post = over_representation(rm_post, sk_post, reward_pos, thr)
    mw_p = mannwhitneyu(orp_pre["dist"], orp_post["dist"]).pvalue if (len(orp_pre["dist"]) and len(orp_post["dist"])) else np.nan
    print(f"  [1] over-rep index: pre={orp_pre['over_representation_index']:.3f}  post={orp_post['over_representation_index']:.3f}  "
          f"mean field-dist {orp_pre['mean_dist']:.3f}->{orp_post['mean_dist']:.3f}  MW p={mw_p:.3g}")
    fig_over_representation(orp_pre, orp_post, reward_pos, pre_lab, post_lab,
                            os.path.join(fig_dir, f"phenom1_over_representation_{pre_state}_vs_{post_state}"))
    append_stat(run_dir, "reward_phenomenology.py", {
        "test": "over_representation_near_reward", "state_pair": f"{pre_state}_vs_{post_state}",
        "skaggs_threshold": float(thr), "r_near": orp_pre["r_near"],
        "n_place_pre": orp_pre["n_place"], "n_place_post": orp_post["n_place"],
        "over_index_pre": orp_pre["over_representation_index"], "over_index_post": orp_post["over_representation_index"],
        "mean_field_dist_pre": orp_pre["mean_dist"], "mean_field_dist_post": orp_post["mean_dist"],
        "field_dist_mannwhitney_p": float(mw_p),
    })

    # --- 2. Distance modulation (ramping) ---
    ramp_pre, curve_pre, dbins = distance_modulation(Xpre, dist)
    ramp_post, curve_post, _ = distance_modulation(Xpost, dist)
    w_ramp = wilcoxon(ramp_post[surv], ramp_pre[surv]).pvalue if len(surv) >= 5 else np.nan
    print(f"  [2] ramping index (survivors): mean {ramp_pre[surv].mean():+.3f}->{ramp_post[surv].mean():+.3f}  "
          f"Wilcoxon p={w_ramp:.3g}")
    fig_distance_modulation(ramp_pre, ramp_post, curve_pre, curve_post, dbins, surv,
                            pre_lab, post_lab,
                            os.path.join(fig_dir, f"phenom3_distance_modulation_{pre_state}_vs_{post_state}"))
    append_stat(run_dir, "reward_phenomenology.py", {
        "test": "reward_distance_ramping", "state_pair": f"{pre_state}_vs_{post_state}",
        "skaggs_threshold": float(thr), "n_survivors": int(len(surv)),
        "mean_ramping_pre": float(ramp_pre[surv].mean()), "mean_ramping_post": float(ramp_post[surv].mean()),
        "frac_positive_ramping_post": float((ramp_post[surv] > 0).mean()),
        "wilcoxon_p": float(w_ramp),
    })

    # --- 3b. APPARENT ramping in the goal-directed read-out (behavioural overlay) ---
    # The frozen-replay ramping test (above) is occupancy-controlled and null.
    # To show that distance-modulation is a *behavioural overlay*, recompute it
    # on the goal-directed read-out, whose occupancy is reward-biased.
    apparent = None
    goal_state = "goal_conditioned"
    if os.path.exists(os.path.join(data_dir, f"df_activations_{goal_state}.pkl")):
        Xg, beh_g, rm_g, _, sk_g = load_state(data_dir, goal_state)
        dist_g, ego_g = geometry(beh_g, reward_pos)
        pc_g = np.where(sk_g >= thr)[0]
        ramp_g, _, _ = distance_modulation(Xg, dist_g)
        orp_g = over_representation(rm_g, sk_g, reward_pos, thr)
        mrl_g, _, _, _ = bearing_tuning(Xg, ego_g)
        apparent = {
            "over_index": orp_g["over_representation_index"],
            "mean_ramping": float(ramp_g[pc_g].mean()),
            "mean_goaldir_mrl": float(mrl_g[pc_g].mean()),
        }
        print(f"  [3b] APPARENT (goal-directed overlay): over-rep={apparent['over_index']:.3f}  "
              f"ramping={apparent['mean_ramping']:+.3f}  goal-dir MRL={apparent['mean_goaldir_mrl']:.3f}")
        append_stat(run_dir, "reward_phenomenology.py", {
            "test": "apparent_behavioural_overlay", "period": goal_state, "skaggs_threshold": float(thr),
            "n_place": int(len(pc_g)), **{f"apparent_{k}": v for k, v in apparent.items()},
        })

    # --- 4. Goal-vector / goal-direction (with allocentric head-direction control) ---
    mrl_pre, _, _, phi = bearing_tuning(Xpre, ego)
    mrl_post, pref_post, curve_b_post, _ = bearing_tuning(Xpost, ego)
    # Head-direction control: tuning to allocentric heading. Sarel 2017 requires
    # that goal-direction tuning is not merely head-direction tuning; since the
    # goal is fixed, the two are geometrically related, so we report both and
    # compare their pre->post change.
    hd = np.arctan2(np.sin(beh_pre["theta"].values), np.cos(beh_pre["theta"].values))
    hd_pre, _, _, _ = circular_tuning(Xpre, hd)
    hd_post, _, _, _ = circular_tuning(Xpost, hd)
    thr_pre = bearing_tuning_null(Xpre, ego, n_null=args.n_null, seed=1)
    thr_post = bearing_tuning_null(Xpost, ego, n_null=args.n_null, seed=2)
    tuned_pre = int(((mrl_pre > thr_pre) & np.isin(np.arange(len(mrl_pre)), surv)).sum())
    tuned_post = int(((mrl_post > thr_post) & np.isin(np.arange(len(mrl_post)), surv)).sum())
    w_mrl = wilcoxon(mrl_post[surv], mrl_pre[surv]).pvalue if len(surv) >= 5 else np.nan
    w_hd = wilcoxon(hd_post[surv], hd_pre[surv]).pvalue if len(surv) >= 5 else np.nan
    print(f"  [4] goal-direction MRL (survivors): mean {mrl_pre[surv].mean():.3f}->{mrl_post[surv].mean():.3f}  "
          f"Wilcoxon p={w_mrl:.3g}  tuned {tuned_pre}->{tuned_post}  | head-dir MRL {hd_pre[surv].mean():.3f}->{hd_post[surv].mean():.3f} (p={w_hd:.3g})")
    fig_goal_vector(mrl_pre, mrl_post, curve_b_post, phi, tuned_post, surv,
                    pre_lab, post_lab,
                    os.path.join(fig_dir, f"phenom4_goal_vector_{pre_state}_vs_{post_state}"))
    append_stat(run_dir, "reward_phenomenology.py", {
        "test": "goal_direction_tuning", "state_pair": f"{pre_state}_vs_{post_state}",
        "skaggs_threshold": float(thr), "n_survivors": int(len(surv)),
        "mean_mrl_pre": float(mrl_pre[surv].mean()), "mean_mrl_post": float(mrl_post[surv].mean()),
        "mean_headdir_mrl_pre": float(hd_pre[surv].mean()), "mean_headdir_mrl_post": float(hd_post[surv].mean()),
        "delta_goaldir_mrl": float(mrl_post[surv].mean() - mrl_pre[surv].mean()),
        "delta_headdir_mrl": float(hd_post[surv].mean() - hd_pre[surv].mean()),
        "n_goal_direction_cells_pre": tuned_pre, "n_goal_direction_cells_post": tuned_post,
        "wilcoxon_p": float(w_mrl), "headdir_wilcoxon_p": float(w_hd),
    })

    report = write_stats_report(run_dir)
    print(f"  wrote {report}")


if __name__ == "__main__":
    main()
