import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pc_common import append_stat, resolve_reward_pos, save_all, write_stats_report, PLACE_CELL_SKAGGS_MIN
from plot_goal_directed_maps import build_rate_maps, field_center_of_mass, smooth_and_norm, compute_skaggs

# --- Configuration ---
D_R = 0.05
LOOKING_THRESHOLD_RAD = np.deg2rad(5)
EPSILON = 1e-9


def load_and_merge_data(data_dir, period="after"):
    """
    Loads pickles for a specific period (before/after)
    and merges behavior with neural data.
    """
    act_file = f"df_activations_{period}.pkl"
    beh_file = f"df_behavior_{period}.pkl"

    try:
        df_act = pd.read_pickle(os.path.join(data_dir, act_file))
        df_beh = pd.read_pickle(os.path.join(data_dir, beh_file))
    except FileNotFoundError as e:
        print(f"Could not load {period} files: {e}")
        return None

    df_merged = pd.merge(df_act, df_beh, on="timestep")
    return df_merged


def calculate_spatial_metrics(df, reward_pos):
    """Adds distance to reward and determines 'Near' status."""
    df["dist_to_reward"] = np.sqrt(
        (df["x"] - reward_pos[0]) ** 2 + (df["y"] - reward_pos[1]) ** 2
    )
    df["is_near_reward"] = df["dist_to_reward"] < D_R
    return df


def calculate_angular_metrics(df, reward_pos):
    """Calculates if the agent is looking at the reward."""
    vec_y = reward_pos[1] - df["y"]
    vec_x = reward_pos[0] - df["x"]
    df["target_angle"] = np.arctan2(vec_y, vec_x)

    diff = df["theta"] - df["target_angle"]
    df["angle_error"] = np.arctan2(np.sin(diff), np.cos(diff))

    df["is_looking_at_reward"] = np.abs(df["angle_error"]) < LOOKING_THRESHOLD_RAD
    return df


def compute_selectivity(df, condition_col, neuron_col="neuron_id", act_col="activity"):
    """Computes Selectivity Index (Condition - Rest)/(Condition + Rest)."""
    grouped = df.groupby([neuron_col, condition_col])[act_col].mean().unstack()
    
    # Handle cases where a condition might be missing (fill NaN)
    if True not in grouped.columns: grouped[True] = 0
    if False not in grouped.columns: grouped[False] = 0
    
    grouped.columns = ["mean_rest", "mean_condition"] # False, True
    
    numerator = grouped["mean_condition"] - grouped["mean_rest"]
    denominator = grouped["mean_condition"] + grouped["mean_rest"] + EPSILON
    grouped["selectivity_index"] = numerator / denominator
    
    return grouped

def calculate_place_field_shift(data_dir, state_before, state_after, m=32, hidden_dim=512):
    """
    Computes the Euclidean shift of each neuron's rate-map centroid between
    state_before and state_after. Field "center" is the center of mass of the
    smoothed, normalized rate map (matching plot_goal_directed_maps.py /
    plot_replay_v11.py's methodology) -- one center per neuron, no GMM/
    field-count ambiguity.
    """
    try:
        df_beh_b = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{state_before}.pkl"))
        df_act_b = pd.read_pickle(os.path.join(data_dir, f"df_activations_{state_before}.pkl"))
        df_beh_a = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{state_after}.pkl"))
        df_act_a = pd.read_pickle(os.path.join(data_dir, f"df_activations_{state_after}.pkl"))
    except FileNotFoundError as e:
        print(f"Data files not found ({e}). Skipping shift calculation.")
        return None

    rm_b, occ_b = build_rate_maps(df_beh_b, df_act_b, m=m, hidden_dim=hidden_dim)
    rm_a, occ_a = build_rate_maps(df_beh_a, df_act_a, m=m, hidden_dim=hidden_dim)

    # Per-neuron Skaggs in each state, so downstream can drop "empty" cells
    # (non-spatial in one state) whose centre-of-mass shift is a meaningless
    # regression-to-centre artifact rather than a real field movement.
    sk_b = compute_skaggs(rm_b, occ_b)
    sk_a = compute_skaggs(rm_a, occ_a)

    shifts = [
        float(np.linalg.norm(
            np.array(field_center_of_mass(smooth_and_norm(rm_a[nid])))
            - np.array(field_center_of_mass(smooth_and_norm(rm_b[nid])))
        ))
        for nid in range(hidden_dim)
    ]
    return pd.DataFrame({
        "neuron_id": np.arange(hidden_dim),
        "pf_shift": shifts,
        "skaggs_before": sk_b,
        "skaggs_after": sk_a,
    })


def correlate_si_shift(si_df, shift_df):
    """Pearson correlation (with p-value) between selectivity index and place-field shift."""
    df = pd.merge(si_df, shift_df, on="neuron_id")
    if len(df) < 3:
        return df, None, None
    from scipy.stats import pearsonr
    r, p = pearsonr(df["selectivity_index"], df["pf_shift"])
    return df, float(r), float(p)


def plot_correlation(df_sp, df_ang, corr_sp, corr_ang, state_after, output):
    """
    Plots the correlation between SI (state_after) and place-field shift.
    corr_sp/corr_ang are (r, p) tuples (or (None, None) if not computable).
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _title(base, corr):
        r, p = corr
        if r is None:
            return f"{base}\nCorrelation: N/A"
        return f"{base}\nr = {r:.3f}, p = {p:.3g}"

    # --- Plot 1: Spatial SI vs Shift ---
    sns.regplot(
        data=df_sp, x="selectivity_index", y="pf_shift",
        ax=axes[0], scatter_kws={'s': 15, 'alpha': 0.6}, line_kws={'color': 'red'}
    )
    axes[0].set_title(_title("Spatial Selectivity vs. PF Shift", corr_sp), fontsize=12)
    axes[0].set_xlabel(f"Spatial Selectivity Index ({state_after})")
    axes[0].set_ylabel("Place Field Shift (Euclidean Dist)")
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)

    # --- Plot 2: Angular SI vs Shift ---
    sns.regplot(
        data=df_ang, x="selectivity_index", y="pf_shift",
        ax=axes[1], scatter_kws={'s': 15, 'alpha': 0.6}, line_kws={'color': 'red'}
    )
    axes[1].set_title(_title("Gaze Selectivity vs. PF Shift", corr_ang), fontsize=12)
    axes[1].set_xlabel(f"Gaze Selectivity Index ({state_after})")
    axes[1].set_ylabel("Place Field Shift (Euclidean Dist)")
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)

    plt.suptitle("Do Reward Neurons Move More?", fontsize=16)
    plt.tight_layout()
    save_all(output)


def plot_si_before_after(sp_merge, ang_merge, state_before, state_after, output):
    """Paired before/after selectivity index comparison (did neurons become
    more selective for near-reward / looking-at-reward after RL?)."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, df, label in [(axes[0], sp_merge, "Spatial (near-reward)"), (axes[1], ang_merge, "Angular (looking-at-reward)")]:
        lo = min(df["selectivity_index_before"].min(), df["selectivity_index_after"].min())
        hi = max(df["selectivity_index_before"].max(), df["selectivity_index_after"].max())
        ax.scatter(df["selectivity_index_before"], df["selectivity_index_after"], alpha=0.4, s=12, color="mediumpurple")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (no change)")
        ax.set_xlabel(f"SI ({state_before})")
        ax.set_ylabel(f"SI ({state_after})")
        ax.set_title(f"{label} selectivity index")
        ax.legend(fontsize=8)
    plt.suptitle("Selectivity index: before vs after", fontsize=14)
    plt.tight_layout()
    save_all(output)


def parse_args():
    p = argparse.ArgumentParser(description="Selectivity vs place-field-shift correlation analysis.")
    p.add_argument("--run_dir", type=str, default=".",
                    help="Run directory containing data/ and figures/ (e.g. Negative_Results).")
    p.add_argument("--reward", nargs=2, type=float, default=None,
                    help="Reward (x, y). Defaults to run_dir/config.json's reward_pos.")
    p.add_argument("--state_before", type=str, default="before",
                    help="Earlier state token, e.g. before, replay_c1_pred.")
    p.add_argument("--state_after", type=str, default="after",
                    help="Later state token (selectivity is computed on this state), e.g. after, replay_c1_postrl.")
    return p.parse_args()


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    run_dir = args.run_dir
    data_dir = os.path.join(run_dir, "data")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    reward_pos = resolve_reward_pos(run_dir, args.reward)
    print(f"Using reward point: {tuple(reward_pos)}")
    state_before, state_after = args.state_before, args.state_after
    pair_tag = f"{state_before}_vs_{state_after}"

    # 1. Process later-state data for Selectivity
    print(f"Loading '{state_after}' data for selectivity...")
    df_after = load_and_merge_data(data_dir, period=state_after)

    if df_after is not None:
        # Process metrics
        df_after = calculate_spatial_metrics(df_after, reward_pos)
        df_after = calculate_angular_metrics(df_after, reward_pos)

        # Compute SI
        print("Computing Selectivity Indices...")
        spatial_stats = compute_selectivity(df_after, condition_col="is_near_reward").reset_index()
        angular_stats = compute_selectivity(df_after, condition_col="is_looking_at_reward").reset_index()

        # 2. Calculate Shift (state_before -> state_after)
        print("Calculating Place Field Shifts...")
        df_shift = calculate_place_field_shift(data_dir, state_before, state_after)

        if df_shift is not None and not df_shift.empty:
            n_all = len(df_shift)
            print(f"Shift calculated for {n_all} neurons.")

            # EMPTY-CELL FILTER: the field-shift is a centre-of-mass movement,
            # which is meaningless for cells that are non-spatial (near-empty /
            # diffuse) in either state -- their COM collapses to the arena
            # centre, manufacturing a fake "shift". Restrict every shift-based
            # analysis (SI<->shift correlation and the high-SI group test) to
            # cells that hold a real field (Skaggs >= threshold) in BOTH states.
            thr_pc = PLACE_CELL_SKAGGS_MIN
            df_shift = df_shift[(df_shift["skaggs_before"] >= thr_pc)
                                & (df_shift["skaggs_after"] >= thr_pc)].copy()
            n_survivors = len(df_shift)
            print(f"Empty-cell filter: {n_survivors}/{n_all} neurons are place cells "
                  f"(Skaggs >= {thr_pc}) in BOTH '{state_before}' and '{state_after}' "
                  f"-- shift analyses use these survivors only.")
            if n_survivors < 5:
                print("Too few survivors for a meaningful shift analysis; skipping shift stats.")

            # 3. Correlate SI (state_after) with shift, with significance
            df_sp, r_sp, p_sp = correlate_si_shift(spatial_stats, df_shift)
            df_ang, r_ang, p_ang = correlate_si_shift(angular_stats, df_shift)
            if r_sp is not None:
                print(f"Spatial SI vs shift:  r={r_sp:.3f}  p={p_sp:.3g}")
            if r_ang is not None:
                print(f"Angular SI vs shift:  r={r_ang:.3f}  p={p_ang:.3g}")
            plot_correlation(df_sp, df_ang, (r_sp, p_sp), (r_ang, p_ang), state_after,
                              os.path.join(fig_dir, f"Correlation_Selectivity_Shift_{pair_tag}"))

            # 4. High-SI vs rest group comparison (Mann-Whitney, not just a
            # descriptive mean) -- is the shift of highly-selective neurons
            # actually different from the rest of the population?
            from scipy.stats import mannwhitneyu

            def _group_test(si_df, label):
                high = si_df[si_df["selectivity_index"] > 0.5]["neuron_id"]
                rest = si_df[si_df["selectivity_index"] <= 0.5]["neuron_id"]
                shifts_high = df_shift[df_shift["neuron_id"].isin(high)]["pf_shift"]
                shifts_rest = df_shift[df_shift["neuron_id"].isin(rest)]["pf_shift"]
                print(f"\n{label} SI: n_high={len(shifts_high)} (mean shift={shifts_high.mean():.4f})  "
                      f"n_rest={len(shifts_rest)} (mean shift={shifts_rest.mean():.4f})")
                result = {
                    "n_high": int(len(shifts_high)),
                    "n_rest": int(len(shifts_rest)),
                    "mean_shift_high": float(shifts_high.mean()) if len(shifts_high) else None,
                    "mean_shift_rest": float(shifts_rest.mean()) if len(shifts_rest) else None,
                }
                if len(shifts_high) >= 2 and len(shifts_rest) >= 2:
                    u, p = mannwhitneyu(shifts_high, shifts_rest, alternative="two-sided")
                    print(f"  Mann-Whitney (high-SI shift vs rest): U={u:.0f}  p={p:.4g}")
                    result["mannwhitney_u"] = float(u)
                    result["mannwhitney_p"] = float(p)
                return result

            spatial_group = _group_test(spatial_stats, "Spatial")
            angular_group = _group_test(angular_stats, "Angular")

            # --- Overall Stats ---
            print(f"\nAvg Shift for empty-filtered survivors ({len(df_shift)} neurons): {df_shift['pf_shift'].mean():.4f}")

            stat_record = {
                "test": "selectivity_vs_shift_summary",
                "state_pair": pair_tag,
                "skaggs_threshold": float(thr_pc),
                "n_neurons_total": int(n_all),
                "n_survivors_both_states": int(n_survivors),
                "mean_shift_survivors": float(df_shift['pf_shift'].mean()) if n_survivors else None,
                "corr_spatial_si_shift_r": r_sp,
                "corr_spatial_si_shift_p": p_sp,
                "corr_angular_si_shift_r": r_ang,
                "corr_angular_si_shift_p": p_ang,
            }
            for k, v in spatial_group.items():
                stat_record[f"spatial_{k}"] = v
            for k, v in angular_group.items():
                stat_record[f"angular_{k}"] = v
            append_stat(run_dir, "analyze_activity.py", stat_record)

            # 5. Did selectivity itself change from state_before to state_after?
            # (Only meaningful/well-defined if state_before has its own
            # behavior+activity data, which it always does.)
            print(f"\nLoading '{state_before}' data for selectivity comparison...")
            df_before = load_and_merge_data(data_dir, period=state_before)
            if df_before is not None:
                df_before = calculate_spatial_metrics(df_before, reward_pos)
                df_before = calculate_angular_metrics(df_before, reward_pos)
                spatial_stats_before = compute_selectivity(df_before, condition_col="is_near_reward").reset_index()
                angular_stats_before = compute_selectivity(df_before, condition_col="is_looking_at_reward").reset_index()

                from scipy.stats import wilcoxon

                # EMPTY-CELL FILTER: only compare the selectivity of cells that
                # are genuine place cells (Skaggs >= threshold) in BOTH states.
                # An empty / non-spatial cell fires near-uniformly, so its SI is
                # a numerically-tiny non-measurement; including such cells just
                # dilutes the selectivity-change signal with noise.
                survivor_ids = set(df_shift["neuron_id"].tolist())
                print(f"SI before/after restricted to {len(survivor_ids)} empty-filtered survivors "
                      f"(place cell in BOTH '{state_before}' and '{state_after}').")

                def _before_after_test(si_before, si_after, label):
                    merged = si_before.merge(si_after, on="neuron_id", suffixes=("_before", "_after"))
                    merged = merged[merged["neuron_id"].isin(survivor_ids)]
                    print(f"{label} SI: {state_before} mean={merged['selectivity_index_before'].mean():.4f}  "
                          f"{state_after} mean={merged['selectivity_index_after'].mean():.4f}")
                    result = {
                        "n_survivors": int(len(merged)),
                        "mean_si_before": float(merged["selectivity_index_before"].mean()),
                        "mean_si_after": float(merged["selectivity_index_after"].mean()),
                    }
                    if len(merged) >= 5:
                        stat, p = wilcoxon(merged["selectivity_index_after"], merged["selectivity_index_before"])
                        print(f"  Wilcoxon paired ({state_after} vs {state_before} SI): p={p:.4g}")
                        result["wilcoxon_p"] = float(p)
                    return merged, result

                sp_merge, sp_result = _before_after_test(spatial_stats_before, spatial_stats, "Spatial")
                ang_merge, ang_result = _before_after_test(angular_stats_before, angular_stats, "Angular")

                append_stat(run_dir, "analyze_activity.py", {
                    "test": "selectivity_index_before_vs_after",
                    "state_pair": pair_tag,
                    **{f"spatial_{k}": v for k, v in sp_result.items()},
                    **{f"angular_{k}": v for k, v in ang_result.items()},
                })

                plot_si_before_after(sp_merge, ang_merge, state_before, state_after,
                                      os.path.join(fig_dir, f"SI_before_vs_after_{pair_tag}"))

            report_path = write_stats_report(run_dir)
            print(f"Wrote statistics report: {report_path}")

        else:
            print("Could not calculate shifts (missing files or empty overlap).")