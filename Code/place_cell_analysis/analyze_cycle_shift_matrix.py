"""
analyze_cycle_shift_matrix.py

Analyze fixed-neuron place-field shifts across the actual snapshots a v11 run
produces: before -> replay_c{K}_pred -> replay_c{K}_postrl -> cycle_1 -> ... ->
cycle_{n_cycles} -> after (K = config.json's replay_cycle). v11 only saves a
pre/post-RL split for a single cycle (replay_cycle), not one per cycle, so
that's the only place a clean "RL effect" can be isolated; cycle-to-cycle
comparisons (cycle_i -> cycle_{i+1}) cover the rest of training. Goal-directed
snapshots (cycle_i_goal, goal_conditioned) are compared separately against
their matching non-goal state, not mixed into the main sequence.

Outputs:
  data/cohort_neurons.npy
  data/cycle_shift_pairwise.csv
  data/cycle_shift_matrix_<metric>.csv
  figures/cycle_shift_matrix_<metric>.png
  figures/cycle_progression_effects.png
  figures/rl_effect.png
  figures/goal_directed_effects.png
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.optimize import linear_sum_assignment

from pc_common import load_run_config, resolve_reward_pos, save_all


def parse_args():
    p = argparse.ArgumentParser(description="Analyze v11 fixed-neuron cycle shift matrices.")
    p.add_argument("--run_dir", type=str, default=".")
    p.add_argument("--reward_pos", type=float, nargs=2, default=None,
                   help="Reward (x, y). Defaults to run_dir/config.json's reward_pos.")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--n_cohort", type=int, default=100)
    p.add_argument("--skaggs_threshold", type=float, default=0.05)
    p.add_argument("--min_mean_activity", type=float, default=1e-4)
    p.add_argument("--min_peak_activity", type=float, default=1e-3)
    p.add_argument("--min_active_fraction", type=float, default=0.01)
    p.add_argument("--field_threshold_frac", type=float, default=0.55)
    p.add_argument("--min_component_bins", type=int, default=4)
    p.add_argument("--max_fields", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--no_show", action="store_true", default=True)
    return p.parse_args()


def _state_files_exist(data_dir: str, state: str) -> bool:
    beh = os.path.join(data_dir, f"df_behavior_{state}.pkl")
    act = os.path.join(data_dir, f"df_activations_{state}.pkl")
    return os.path.exists(beh) and os.path.exists(act)


def discover_main_states(data_dir: str, n_cycles: int, replay_cycle: int) -> List[str]:
    """The temporal sequence of snapshots v11 actually saves, in order:
    before, replay_c{K}_pred, replay_c{K}_postrl, cycle_1..cycle_{n_cycles}, after.
    Any state missing on disk is simply skipped."""
    candidates = ["before", f"replay_c{replay_cycle}_pred", f"replay_c{replay_cycle}_postrl"]
    candidates += [f"cycle_{i}" for i in range(1, n_cycles + 1)]
    candidates.append("after")
    return [s for s in candidates if _state_files_exist(data_dir, s)]


def discover_goal_pairs(data_dir: str, n_cycles: int) -> List[Tuple[str, str]]:
    """(base_state, goal_state) pairs for the secondary goal-directed comparison:
    cycle_i vs cycle_i_goal for each cycle, plus after vs goal_conditioned."""
    pairs = []
    for i in range(1, n_cycles + 1):
        base, goal = f"cycle_{i}", f"cycle_{i}_goal"
        if _state_files_exist(data_dir, base) and _state_files_exist(data_dir, goal):
            pairs.append((base, goal))
    if _state_files_exist(data_dir, "after") and _state_files_exist(data_dir, "goal_conditioned"):
        pairs.append(("after", "goal_conditioned"))
    return pairs


def load_probe(data_dir: str, state: str):
    beh = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{state}.pkl"))
    act = pd.read_pickle(os.path.join(data_dir, f"df_activations_{state}.pkl"))
    return beh, act


def build_rate_maps(df_beh, df_act, m=32, hidden_dim=512):
    n_steps = len(df_beh)
    if len(df_act) == n_steps * hidden_dim:
        act_mat = df_act["activity"].values.reshape(n_steps, hidden_dim)
        beh_sorted = df_beh.sort_values("timestep")
    else:
        piv = df_act.pivot(index="timestep", columns="neuron_id", values="activity").fillna(0)
        beh_sorted = df_beh[df_beh["timestep"].isin(piv.index)].sort_values("timestep")
        act_mat = piv.reindex(beh_sorted["timestep"]).values
        hidden_dim = act_mat.shape[1]
        n_steps = len(beh_sorted)
    x_bin = np.clip((beh_sorted["x"].values * m).astype(int), 0, m - 1)
    y_bin = np.clip((beh_sorted["y"].values * m).astype(int), 0, m - 1)
    gc = x_bin * m + y_bin
    n_bins = m * m
    bin_sum = np.zeros((n_bins, hidden_dim), dtype=np.float64)
    np.add.at(bin_sum, gc, act_mat)
    bin_count = np.bincount(gc, minlength=n_bins).astype(float)
    rate_flat = np.divide(bin_sum, bin_count[:, None], out=np.zeros_like(bin_sum), where=bin_count[:, None] > 0)
    rate_maps = rate_flat.T.reshape(hidden_dim, m, m)
    occupancy = (bin_count / max(1, n_steps)).reshape(m, m)
    metrics = {
        "mean_activity": act_mat.mean(axis=0),
        "peak_activity": act_mat.max(axis=0),
        "active_fraction": (act_mat > 1e-8).mean(axis=0),
    }
    return rate_maps, occupancy, metrics


def compute_skaggs(rate_maps, occupancy):
    hidden_dim = rate_maps.shape[0]
    occ_flat = occupancy.reshape(-1)
    rate_flat = rate_maps.reshape(hidden_dim, -1).T
    mean_rate = rate_flat.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mean_rate[None, :] > 1e-10, rate_flat / mean_rate[None, :], 0.0)
        log_ratio = np.where(ratio > 0, np.log2(np.maximum(ratio, 1e-300)), 0.0)
    return (occ_flat[:, None] * ratio * log_ratio).sum(axis=0)


def smooth_norm(rm, sigma=1.5):
    s = gaussian_filter(rm, sigma=sigma)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    return s


def extract_field_centers(rm, threshold_frac=0.55, min_component_bins=4, max_fields=2):
    s = smooth_norm(rm)
    if not np.isfinite(s).all() or s.max() <= 0:
        return np.empty((0, 2)), []
    mask = s >= (threshold_frac * s.max())
    lab, nlab = label(mask)
    comps = []
    m = s.shape[0]
    xs = np.arange(m) / m + 0.5 / m
    ys = np.arange(m) / m + 0.5 / m
    for k in range(1, nlab + 1):
        comp = lab == k
        n_bins = int(comp.sum())
        if n_bins < min_component_bins:
            continue
        weight = s * comp
        total = weight.sum()
        if total <= 0:
            continue
        cx = float((weight * xs[None, :]).sum() / total)
        cy = float((weight * ys[:, None]).sum() / total)
        comps.append({"center": [cx, cy], "peak": float(s[comp].max()), "area_bins": n_bins, "mass": float(total)})
    comps = sorted(comps, key=lambda d: d["mass"], reverse=True)[:max_fields]
    centers = np.asarray([c["center"] for c in comps], dtype=float) if comps else np.empty((0, 2))
    return centers, comps


def map_corr(rm1, rm2):
    a = smooth_norm(rm1).reshape(-1)
    b = smooth_norm(rm2).reshape(-1)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def match_centers(c1: np.ndarray, c2: np.ndarray):
    if len(c1) == 0 or len(c2) == 0:
        return []
    dist = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(dist)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def center_shift_metrics(c1: np.ndarray, c2: np.ndarray, reward: np.ndarray) -> Dict[str, float]:
    pairs = match_centers(c1, c2)
    if not pairs:
        return {
            "n_pairs": 0,
            "mean_abs_shift": np.nan,
            "mean_reward_delta": np.nan,
            "mean_reward_projection": np.nan,
            "fraction_toward_reward": np.nan,
        }
    abs_shifts, reward_deltas, projections, toward = [], [], [], []
    for i, j in pairs:
        before = c1[i]
        after = c2[j]
        move = after - before
        reward_vec = reward - before
        norm = np.linalg.norm(reward_vec)
        unit = reward_vec / norm if norm > 1e-12 else np.zeros(2)
        proj = float(np.dot(move, unit))
        d_before = float(np.linalg.norm(before - reward))
        d_after = float(np.linalg.norm(after - reward))
        reward_delta = d_before - d_after
        abs_shifts.append(float(np.linalg.norm(move)))
        reward_deltas.append(reward_delta)
        projections.append(proj)
        toward.append(reward_delta > 0)
    return {
        "n_pairs": len(pairs),
        "mean_abs_shift": float(np.mean(abs_shifts)),
        "mean_reward_delta": float(np.mean(reward_deltas)),
        "mean_reward_projection": float(np.mean(projections)),
        "fraction_toward_reward": float(np.mean(toward)),
    }


def bootstrap_ci(values, n_boot=2000, seed=123):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        means.append(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_values(values, n_boot=2000, seed=123):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_one_sided_gt0": np.nan}
    ci_low, ci_high = bootstrap_ci(arr, n_boot=n_boot, seed=seed)
    if len(arr) > 1:
        try:
            # Wilcoxon one-sided H1: median > 0.
            p = float(stats.wilcoxon(arr, alternative="greater").pvalue)
        except Exception:
            p = np.nan
    else:
        p = np.nan
    return {"n": int(len(arr)), "mean": float(np.mean(arr)), "median": float(np.median(arr)), "ci_low": ci_low, "ci_high": ci_high, "p_one_sided_gt0": p}


def select_cohort(state_data: Dict, args, baseline_state: str) -> np.ndarray:
    baseline = state_data[baseline_state]
    sk = baseline["skaggs"]
    met = baseline["metrics"]
    active = (
        (met["mean_activity"] >= args.min_mean_activity)
        & (met["peak_activity"] >= args.min_peak_activity)
        & (met["active_fraction"] >= args.min_active_fraction)
    )
    candidates = np.where(active)[0]
    if len(candidates) == 0:
        candidates = np.arange(len(sk))
    ranked = candidates[np.argsort(sk[candidates])[::-1]]
    return ranked[: args.n_cohort].astype(int)


def compute_state_data(states: List[str], data_dir: str, args) -> Dict:
    out = {}
    for state in states:
        print(f"Loading and mapping {state}")
        beh, act = load_probe(data_dir, state)
        rm, occ, metrics = build_rate_maps(beh, act, m=args.m, hidden_dim=args.hidden_dim)
        sk = compute_skaggs(rm, occ)
        fields = []
        for nid in range(rm.shape[0]):
            centers, comps = extract_field_centers(
                rm[nid],
                threshold_frac=args.field_threshold_frac,
                min_component_bins=args.min_component_bins,
                max_fields=args.max_fields,
            )
            fields.append({"centers": centers, "components": comps, "n_fields": len(centers)})
        out[state] = {"rate_maps": rm, "occupancy": occ, "metrics": metrics, "skaggs": sk, "fields": fields}
    return out


def pairwise_state_summary(state_a, state_b, data_a, data_b, cohort, reward, args) -> Dict:
    abs_shift, reward_delta, projection, corr_vals, frac_toward_vals = [], [], [], [], []
    n_valid = 0
    n_lost = 0
    n_gained = 0
    field_count_changes = []
    per_neuron = []
    for nid in cohort:
        c1 = data_a["fields"][nid]["centers"]
        c2 = data_b["fields"][nid]["centers"]
        if len(c1) > 0 and len(c2) > 0:
            n_valid += 1
            m = center_shift_metrics(c1, c2, reward)
            if np.isfinite(m["mean_abs_shift"]):
                abs_shift.append(m["mean_abs_shift"])
                reward_delta.append(m["mean_reward_delta"])
                projection.append(m["mean_reward_projection"])
                frac_toward_vals.append(m["fraction_toward_reward"])
                per_neuron.append({
                    "neuron_id": int(nid),
                    "from_state": state_a,
                    "to_state": state_b,
                    **m,
                    "map_corr": map_corr(data_a["rate_maps"][nid], data_b["rate_maps"][nid]),
                    "n_fields_before": int(len(c1)),
                    "n_fields_after": int(len(c2)),
                })
        elif len(c1) > 0 and len(c2) == 0:
            n_lost += 1
        elif len(c1) == 0 and len(c2) > 0:
            n_gained += 1
        field_count_changes.append(int(len(c2) - len(c1)))
        corr_vals.append(map_corr(data_a["rate_maps"][nid], data_b["rate_maps"][nid]))

    reward_summary = summarize_values(reward_delta, args.bootstrap, args.seed)
    projection_summary = summarize_values(projection, args.bootstrap, args.seed)
    abs_summary = summarize_values(abs_shift, args.bootstrap, args.seed)
    corr_summary = summarize_values(corr_vals, args.bootstrap, args.seed)
    return {
        "from_state": state_a,
        "to_state": state_b,
        "n_cohort": int(len(cohort)),
        "n_valid_fields_both": int(n_valid),
        "n_lost_fields": int(n_lost),
        "n_gained_fields": int(n_gained),
        "mean_abs_shift": abs_summary["mean"],
        "abs_shift_ci_low": abs_summary["ci_low"],
        "abs_shift_ci_high": abs_summary["ci_high"],
        "mean_reward_delta": reward_summary["mean"],
        "reward_delta_ci_low": reward_summary["ci_low"],
        "reward_delta_ci_high": reward_summary["ci_high"],
        "reward_delta_p_gt0": reward_summary["p_one_sided_gt0"],
        "mean_reward_projection": projection_summary["mean"],
        "reward_projection_ci_low": projection_summary["ci_low"],
        "reward_projection_ci_high": projection_summary["ci_high"],
        "reward_projection_p_gt0": projection_summary["p_one_sided_gt0"],
        "mean_map_corr": corr_summary["mean"],
        "fraction_toward_reward": float(np.nanmean(frac_toward_vals)) if frac_toward_vals else np.nan,
        "mean_field_count_change": float(np.mean(field_count_changes)) if field_count_changes else np.nan,
        "per_neuron": per_neuron,
    }


def plot_matrix(mat: pd.DataFrame, title: str, path: str, no_show=True):
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(mat)), max(6, 0.7 * len(mat))))
    im = ax.imshow(mat.values, origin="upper", aspect="auto")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_all(path, dpi=150)
    if not no_show:
        plt.show()
    plt.close(fig)


def plot_transition_effects(df: pd.DataFrame, fig_dir: str, out_name: str, title_prefix: str, no_show=True):
    """Generic plot for a sequence of from_state->to_state pairwise summaries
    (used for cycle progression, the single RL-effect pair, and goal-directed
    pairs alike). X-axis is the transition label, not assumed to be numeric."""
    if len(df) == 0:
        return
    labels = [f"{a}\n→{b}" for a, b in zip(df["from_state"], df["to_state"])]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(max(9, 2.2 * len(labels)), 4))

    axes[0].errorbar(x, df["mean_reward_delta"],
                     yerr=[df["mean_reward_delta"] - df["reward_delta_ci_low"], df["reward_delta_ci_high"] - df["mean_reward_delta"]],
                     marker="o", linestyle="none")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title(f"{title_prefix}: distance-to-reward delta")
    axes[0].set_ylabel("positive = toward reward")

    axes[1].errorbar(x, df["mean_reward_projection"],
                     yerr=[df["mean_reward_projection"] - df["reward_projection_ci_low"], df["reward_projection_ci_high"] - df["mean_reward_projection"]],
                     marker="o", linestyle="none")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title(f"{title_prefix}: projection onto reward vector")
    axes[1].set_ylabel("positive = movement toward reward")

    axes[2].plot(x, df["mean_map_corr"], marker="o", linestyle="none")
    axes[2].set_title("Map similarity")
    axes[2].set_ylabel("mean spatial corr")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(fig_dir, out_name)
    save_all(path, dpi=150)
    if not no_show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = os.path.join(args.run_dir, "data")
    fig_dir = os.path.join(args.run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = load_run_config(args.run_dir)
    reward = resolve_reward_pos(args.run_dir, args.reward_pos)
    n_cycles = int(cfg.get("n_cycles", 1))
    replay_cycle = int(cfg.get("replay_cycle", 1))

    states = discover_main_states(data_dir, n_cycles, replay_cycle)
    if not states:
        raise FileNotFoundError(f"No v11 probe files found in {data_dir}")
    print("States (main sequence):", states)

    goal_pairs = discover_goal_pairs(data_dir, n_cycles)
    print("Goal-directed pairs:", goal_pairs)

    goal_states = [s for pair in goal_pairs for s in pair]
    all_states_needed = list(dict.fromkeys(states + goal_states))
    state_data = compute_state_data(all_states_needed, data_dir, args)

    baseline_state = "before" if "before" in state_data else states[0]
    cohort = select_cohort(state_data, args, baseline_state)
    cohort_path = os.path.join(data_dir, "cohort_neurons.npy")
    np.save(cohort_path, cohort)
    print(f"Selected fixed cohort of {len(cohort)} neurons from '{baseline_state}': {cohort_path}")

    # Per-state summary of the cohort (main sequence only).
    state_rows = []
    for s in states:
        sk = state_data[s]["skaggs"][cohort]
        n_fields = np.array([state_data[s]["fields"][int(nid)]["n_fields"] for nid in cohort])
        state_rows.append({
            "state": s,
            "mean_skaggs_cohort": float(np.nanmean(sk)),
            "n_place_by_skaggs": int(np.sum(sk >= args.skaggs_threshold)),
            "n_with_fields": int(np.sum(n_fields > 0)),
            "n_single_field": int(np.sum(n_fields == 1)),
            "n_multi_field": int(np.sum(n_fields > 1)),
        })
    pd.DataFrame(state_rows).to_csv(os.path.join(data_dir, "cohort_state_summary.csv"), index=False)

    def _pairwise(a, b):
        return pairwise_state_summary(a, b, state_data[a], state_data[b], cohort, reward, args)

    # Main sequence matrix (before -> ... -> after).
    rows = []
    per_neuron_all = []
    matrices = {
        "mean_abs_shift": pd.DataFrame(index=states, columns=states, dtype=float),
        "mean_reward_delta": pd.DataFrame(index=states, columns=states, dtype=float),
        "mean_reward_projection": pd.DataFrame(index=states, columns=states, dtype=float),
        "mean_map_corr": pd.DataFrame(index=states, columns=states, dtype=float),
    }
    for a in states:
        for b in states:
            if a == b:
                for key in matrices:
                    matrices[key].loc[a, b] = 0.0 if key != "mean_map_corr" else 1.0
                continue
            summary = _pairwise(a, b)
            per_neuron_all.extend(summary.pop("per_neuron"))
            rows.append(summary)
            for key in matrices:
                matrices[key].loc[a, b] = summary[key]

    df_pairwise = pd.DataFrame(rows)
    pairwise_path = os.path.join(data_dir, "cycle_shift_pairwise.csv")
    df_pairwise.to_csv(pairwise_path, index=False)
    pd.DataFrame(per_neuron_all).to_csv(os.path.join(data_dir, "cycle_shift_per_neuron.csv"), index=False)
    print(f"Saved pairwise summary: {pairwise_path}")

    for key, mat in matrices.items():
        csv_path = os.path.join(data_dir, f"cycle_shift_matrix_{key}.csv")
        mat.to_csv(csv_path)
        plot_matrix(mat, key, os.path.join(fig_dir, f"cycle_shift_matrix_{key}"), args.no_show)

    # RL effect: replay_c{K}_pred -> replay_c{K}_postrl (only cycle with a clean pre/post-RL split).
    rl_a, rl_b = f"replay_c{replay_cycle}_pred", f"replay_c{replay_cycle}_postrl"
    rl_rows = []
    if rl_a in states and rl_b in states:
        hit = df_pairwise[(df_pairwise["from_state"] == rl_a) & (df_pairwise["to_state"] == rl_b)]
        if len(hit):
            rl_rows.append(hit.iloc[0].to_dict())
    df_rl = pd.DataFrame(rl_rows)
    df_rl.to_csv(os.path.join(data_dir, "rl_effect_summary.csv"), index=False)
    plot_transition_effects(df_rl, fig_dir, "rl_effect", "RL effect", args.no_show)

    # Cycle-progression effect: cycle_i -> cycle_{i+1}.
    progression_rows = []
    for i in range(1, n_cycles):
        a, b = f"cycle_{i}", f"cycle_{i + 1}"
        if a in states and b in states:
            hit = df_pairwise[(df_pairwise["from_state"] == a) & (df_pairwise["to_state"] == b)]
            if len(hit):
                progression_rows.append(hit.iloc[0].to_dict())
    df_progression = pd.DataFrame(progression_rows)
    df_progression.to_csv(os.path.join(data_dir, "cycle_progression_summary.csv"), index=False)
    plot_transition_effects(df_progression, fig_dir, "cycle_progression_effects", "Cycle progression", args.no_show)

    # Overall before -> after summary (also present in the main matrix, pulled out for convenience).
    overall_rows = []
    if "before" in states and "after" in states:
        hit = df_pairwise[(df_pairwise["from_state"] == "before") & (df_pairwise["to_state"] == "after")]
        if len(hit):
            overall_rows.append(hit.iloc[0].to_dict())
    pd.DataFrame(overall_rows).to_csv(os.path.join(data_dir, "overall_before_after_summary.csv"), index=False)

    # Goal-directed effect: cycle_i vs cycle_i_goal, and after vs goal_conditioned.
    goal_rows = []
    for a, b in goal_pairs:
        goal_rows.append(_pairwise(a, b))
    df_goal = pd.DataFrame([{k: v for k, v in r.items() if k != "per_neuron"} for r in goal_rows])
    df_goal.to_csv(os.path.join(data_dir, "goal_directed_summary.csv"), index=False)
    plot_transition_effects(df_goal, fig_dir, "goal_directed_effects", "Goal-directed", args.no_show)

    meta = {
        "run_dir": args.run_dir,
        "reward_pos": reward.tolist(),
        "n_cycles": n_cycles,
        "replay_cycle": replay_cycle,
        "states": states,
        "goal_pairs": goal_pairs,
        "cohort_size": int(len(cohort)),
        "baseline_state": baseline_state,
        "cohort_neurons_path": cohort_path,
        "main_note": (
            "Main sequence is before -> ... -> after (temporal progression). "
            "rl_effect_summary.csv isolates the one cycle with a pre/post-RL split "
            "(replay_cycle). cycle_progression_summary.csv covers cycle_i -> cycle_{i+1}. "
            "goal_directed_summary.csv compares each state against its goal-directed probe."
        ),
    }
    with open(os.path.join(data_dir, "cycle_shift_analysis_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
