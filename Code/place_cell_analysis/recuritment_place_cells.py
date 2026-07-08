import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
import os

from pc_common import (
    append_stat,
    ensure_skaggs,
    filter_place_cells,
    resolve_reward_pos,
    save_all,
)

def create_cells_figure(df_combined, neurons, output=None, show=True):
    """
    Takes the combined dataframe of activity and behavior of all of the neurons, another dataframe that contains only unique "neuron_id"
    merges them and saves the figure in the figures folder. Can print maximum of 150 neurons.
    If there is no output, won't save. Can turn off show by setting to false.
    """
    if len(neurons) > 150:
        print("There are too many neurons. Truncating to 150 for plotting.")
        neurons = neurons[:150]

    # Use copy to avoid SettingWithCopyWarning if 'neurons' is a slice
    neurons_df = neurons.copy()

    # Check if 'neuron_id' is the index and reset if needed
    if "neuron_id" not in neurons_df.columns:
        if neurons_df.index.name == "neuron_id":
            neurons_df.reset_index(inplace=True)
        else:
            print("Error: 'neuron_id' column not found in neurons dataframe.")
            return

    df_combined = df_combined.merge(
        neurons_df[["neuron_id"]], on="neuron_id", how="inner"
    )

    # Grid
    n_plots = len(neurons_df)
    n_cols = 15
    # Adjust rows to fit all plots
    n_rows = int(np.ceil(n_plots / n_cols))

    if n_rows == 0:
        print("No neurons to plot.")
        return

    print(f"Going to show {n_plots} neurons in a {n_rows}x{n_cols} grid")

    # Compact figure size; ~1.3" per subplot side for readability
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.3 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, neuron_id in enumerate(neurons_df["neuron_id"]):
        df_neuron = df_combined[df_combined["neuron_id"] == neuron_id]

        if len(df_neuron) > 0:
            try:
                heatmap_data = df_neuron.pivot_table(
                    index="y", columns="x", values="activity", aggfunc="mean"
                )

                sns.heatmap(
                    heatmap_data,
                    ax=axes[i],
                    cmap="jet",
                    cbar=False,  # no per-axes colorbars to stay compact
                    annot=False,  # keep compact
                    square=True,
                )
                axes[i].set_title(f"{neuron_id}", fontsize=8)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")
            except Exception as e:
                print(f"Error plotting neuron {neuron_id}: {e}")
                axes[i].text(
                    0.5,
                    0.5,
                    f"Plot Error\n{neuron_id}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    transform=axes[i].transAxes,
                )

        else:
            axes[i].text(
                0.5,
                0.5,
                f"No data\n{neuron_id}",
                ha="center",
                va="center",
                fontsize=8,
                transform=axes[i].transAxes,
            )
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")

    # Hide any unused axes
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.tight_layout()
    if show:
        plt.show()
    if output:
        save_all(output)
        print(f"Figure saved to {output}")
    plt.close()



def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two (x, y) tuples."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_shifts_and_distances(df_place_cells, df_centers_before, df_centers_after, reward_point):
    """
    Calculates the shift of place fields towards reward_point.

    Returns:
        all_shifts (list): A list of all calculated shifts (Dist_Before - Dist_After).
        distance_pairs (list): A list of (Dist_Before, Dist_After) tuples
                               for the new scatter plot.
    """
    print("Calculating place field shifts and distances...")
    all_shifts = []
    distance_pairs = []  # For the new scatter plot

    # Get the set of neuron IDs we care about (from df_place_cells)
    place_cell_ids = set(df_place_cells["neuron_id"])

    # Find neurons that are in df_place_cells AND exist in BOTH before and after data
    before_ids = set(df_centers_before["neuron_id"])
    after_ids = set(df_centers_after["neuron_id"])
    common_neurons_to_track = place_cell_ids.intersection(before_ids).intersection(
        after_ids
    )

    # Pre-filter the dataframes for efficiency
    df_before_filtered = df_centers_before[
        df_centers_before["neuron_id"].isin(common_neurons_to_track)
    ]
    df_after_filtered = df_centers_after[
        df_centers_after["neuron_id"].isin(common_neurons_to_track)
    ]
    count = 0

    # Single place cell calculation
    for neuron_id in tqdm(
        common_neurons_to_track, desc="Processing neurons (Single bump neurons)"
    ):
        center_before = df_before_filtered[
            df_before_filtered["neuron_id"] == neuron_id
        ]["center"].tolist()
        center_after = df_after_filtered[df_after_filtered["neuron_id"] == neuron_id][
            "center"
        ].tolist()
        if not center_before or not center_after:
            continue
        distance_before = np.mean([euclidean_distance(reward_point, c) for c in center_before])
        distance_after = np.mean([euclidean_distance(reward_point, c) for c in center_after])
        shift = distance_before - distance_after
        all_shifts.append(shift)
        distance_pairs.append((distance_before, distance_after))

    # Multi place cell shift calculation

    # for neuron_id in tqdm(common_neurons_to_track, desc="Processing neurons (Multi-Bump)"):

    #     # Get the list of centers for this neuron
    #     centers_before = df_before_filtered[df_before_filtered['neuron_id'] == neuron_id]['center'].tolist()
    #     centers_after = df_after_filtered[df_after_filtered['neuron_id'] == neuron_id]['center'].tolist()

    #     if not centers_before or not centers_after:
    #         continue # Skip if no centers in either period

    #     # Handle case where 'after' has more fields (or equal)
    #     if len(centers_before) < len(centers_after):
    #         count +=1
    #         distances_after = np.zeros(len(centers_after))
    #         distances_before = np.zeros(len(centers_before))
    #         for i in range(len(distances_before)):
    #             distances_before[i] = euclidean_distance(REWARD_POINT, centers_before[i])
    #         for i in range(len(distances_after)):
    #             distances_after[i] = euclidean_distance(REWARD_POINT, centers_after[i])

    #         shift = distances_before.mean() - distances_after.mean()
    #         all_shifts.append(shift)
    #         # This case does not generate (x,y) pairs for the scatter plot,
    #         # as it's a comparison of means.
    #         continue

    #     # Handle main case: len(centers_before) >= len(centers_after)
    #     # For each 'before' center, find the closest 'after' center
    #     for center_b in centers_before:
    #         # Find the single closest center_a from the 'after' list
    #         closest_center_a = min(centers_after,
    #                                key=lambda center_a: euclidean_distance(center_b, center_a))

    #         # Now, calculate the shift *relative to the reward point*
    #         dist_before_reward = euclidean_distance(center_b, REWARD_POINT)
    #         dist_after_reward = euclidean_distance(closest_center_a, REWARD_POINT)

    #         # Shift = (Distance Before) - (Distance After)
    #         # A positive value means it moved closer to the reward.
    #         shift = dist_before_reward - dist_after_reward
    #         all_shifts.append(shift)

    #         # Store the pair for the new scatter plot
    #         distance_pairs.append((dist_before_reward, dist_after_reward))

    print(f"amount of place cells skipped in case of more bumps before RL {count}")
    return all_shifts, distance_pairs


def plot_shift_histogram(all_shifts, output=None, show=True):
    """Plots a histogram of the calculated shifts."""
    if not all_shifts:
        print("No shifts were calculated. Cannot plot histogram.")
        return

    print(f"Plotting histogram for {len(all_shifts)} field shifts.")
    plt.figure(figsize=(10, 6))
    sns.histplot(all_shifts, kde=True, bins=20)
    plt.title("Distribution of Place Field Shifts Towards Reward", fontsize=16)
    plt.xlabel("Shift Towards Reward (Dist_Before - Dist_After)", fontsize=12)
    plt.ylabel("Frequency (Count of Fields)", fontsize=12)

    mean_shift = np.mean(all_shifts)
    print(f"Mean Shift: {mean_shift:.4f}")

    plt.grid(True, linestyle=":", alpha=0.7)

    if output:
        save_all(output)
        print(f"Histogram saved to {output}")
    if show:
        plt.show()
    plt.close()


def plot_shift_comparison(distance_pairs, output=None, show=True):
    """
    Plots a scatter plot of distance_before_reward vs. distance_after_reward.

    Args:
        distance_pairs (list of tuples): List of (dist_before, dist_after) pairs.
        output (str, optional): File path to save the figure.
        show (bool, optional): Whether to display the plot.
    """
    if not distance_pairs:
        print("No distance pairs to plot for comparison.")
        return

    # Convert to a DataFrame for easier plotting
    df_pairs = pd.DataFrame(
        distance_pairs, columns=["distance_before", "distance_after"]
    )

    print(f"Plotting scatter plot for {len(df_pairs)} field pairs.")
    plt.figure(figsize=(8, 8))

    sns.scatterplot(
        data=df_pairs, x="distance_before", y="distance_after", alpha=0.6, s=50
    )

    # Determine plot limits to draw the y=x line
    min_val = min(df_pairs["distance_before"].min(), df_pairs["distance_after"].min())
    max_val = max(df_pairs["distance_before"].max(), df_pairs["distance_after"].max())

    plot_min = min_val * 0.95
    plot_max = max_val * 1.05

    # Add the red identity line (y=x)
    plt.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        "r--",
        linewidth=2,
        label="y = x (No change)",
    )

    # Set plot limits to be equal for a square aspect ratio
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    plt.gca().set_aspect("equal", adjustable="box")  # Enforce square aspect

    plt.title("Place Field Distance to Reward: Before vs. After", fontsize=16)
    plt.xlabel("Distance to Reward (Before)", fontsize=12)
    plt.ylabel("Distance to Reward (After)", fontsize=12)

    # Add text to explain the plot
    plt.text(
        plot_min + (plot_max - plot_min) * 0.05,
        plot_max - (plot_max - plot_min) * 0.1,
        "Points BELOW the red line (y < x)\n...moved CLOSER to reward",
        fontsize=10,
        color="blue",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5, pad=0.3),
    )

    plt.text(
        plot_max - (plot_max - plot_min) * 0.05,
        plot_min + (plot_max - plot_min) * 0.1,
        "Points ABOVE the red line (y > x)\n...moved FURTHER from reward",
        fontsize=10,
        color="red",
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.5, pad=0.3),
    )

    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    if output:
        save_all(output)
        print(f"Scatter plot saved to {output}")
    if show:
        plt.show()
    plt.close()


def perform_shift_t_test(shifts, alpha=0.05):
    """
    Performs a one-sample, one-tailed Student's t-test on the shift data.

    H0: The true mean shift is 0.
    H1: The true mean shift is greater than 0.

    Prints test statistic (t), one-sided p-value, and interpretation.
    """
    if not shifts:
        print("No shift data provided for t-test.")
        return None

    n = len(shifts)
    if n < 2:
        print("Not enough shift data (N<2) to perform a t-test.")
        return None

    # Calculate sample statistics
    mean_shift = np.mean(shifts)
    std_dev = np.std(shifts, ddof=1)  # sample standard deviation (unbiased)

    # Handle case of zero variance (all shifts identical) or NaN
    if np.isnan(std_dev) or np.isclose(std_dev, 0.0):
        print("\n--- t Test Warning ---")
        print("Sample standard deviation is zero or NaN. All shifts may be identical.")
        if mean_shift > 0:
            print("Mean shift is positive and non-zero: p-value is effectively 0.")
            p_value = 0.0
            t_stat = np.inf
        elif np.isclose(mean_shift, 0.0):
            print("Mean shift is zero. Cannot reject H0. p-value is 1.0.")
            p_value = 1.0
            t_stat = 0.0
        else:
            print(
                f"Mean shift {mean_shift} is negative. p-value is 1.0 for H1: mean > 0."
            )
            p_value = 1.0
            t_stat = -np.inf
    else:
        std_err = std_dev / np.sqrt(n)
        t_stat = (mean_shift - 0.0) / std_err
        # one-sided p-value: P(T_{df} > t_stat)
        df = n - 1
        p_value = 1.0 - stats.t.cdf(t_stat, df=df)

    print("\n--- Place Field Shift t-Test ---")
    print(f"H0: True mean shift = 0")
    print(f"H1: True mean shift > 0 (fields moved towards reward)")
    print("-----------------------------------------")
    print(f"Sample size (N): {n}")
    print(f"Sample mean:     {mean_shift:.4f}")
    print(f"Sample std dev:  {std_dev:.4f}")
    print(f"t statistic:     {t_stat:.4f}")
    print(f"p-value (one-tailed): {p_value:.4f}")
    print("-----------------------------------------")

    if p_value < alpha:
        print(f"Conclusion: p-value ({p_value:.4f}) < alpha ({alpha}).")
        print("We REJECT the null hypothesis (H0).")
        print("There is a statistically significant shift towards the reward point.")
    else:
        print(f"Conclusion: p-value ({p_value:.4f}) >= alpha ({alpha}).")
        print("We FAIL to reject the null hypothesis (H0).")
        print("There is no statistically significant shift towards the reward point.")
    print("-----------------------------------------\n")

    return {
        "test": "recruitment_shift_one_sample_t_test",
        "n": n,
        "mean_shift": float(mean_shift),
        "std_dev": float(std_dev),
        "t_stat": float(t_stat),
        "p_value_one_tailed": float(p_value),
    }


# ============================================================================
# Population-level Hollup over-representation tests + recruited-cell figure
# ============================================================================

def distances_to_reward(centers, reward_pos):
    """Euclidean distance from each (x, y) tuple in `centers` to `reward_pos`."""
    arr = np.array([list(c) for c in centers])
    if arr.size == 0:
        return np.array([])
    return np.linalg.norm(arr - np.asarray(reward_pos), axis=1)


def two_sample_distance_test(d_before, d_after):
    """Independent two-sample test: did the place-cell *population* concentrate
    near reward? This is the right form of the Hollup question — not the paired
    shift test (which asks whether *individual* fields migrated)."""
    print("\n--- Two-Sample Distance Test (Independent Populations) ---")
    print("H0: mean(d_after) = mean(d_before)")
    print("H1: mean(d_after) < mean(d_before)   (population concentrates near reward)")
    print(f"BEFORE: N={len(d_before)}  mean={d_before.mean():.4f}  median={np.median(d_before):.4f}")
    print(f"AFTER:  N={len(d_after)}   mean={d_after.mean():.4f}  median={np.median(d_after):.4f}")
    if len(d_before) < 2 or len(d_after) < 2:
        print("Not enough samples for a test.")
        return None
    t, p_t = stats.ttest_ind(d_after, d_before, alternative="less", equal_var=False)
    u, p_mw = stats.mannwhitneyu(d_after, d_before, alternative="less")
    print(f"Welch t-test:    t = {t:.3f}    p (one-sided) = {p_t:.4f}")
    print(f"Mann-Whitney:    U = {u:.0f}    p (one-sided) = {p_mw:.4f}")
    print("-----------------------------------------")
    return {
        "test": "population_distance_welch_and_mannwhitney",
        "n_before": int(len(d_before)),
        "n_after": int(len(d_after)),
        "welch_t": float(t),
        "welch_p_one_sided": float(p_t),
        "mannwhitney_u": float(u),
        "mannwhitney_p_one_sided": float(p_mw),
    }


def recruitment_zone_test(d_recruited, reward_pos, R=0.15):
    """One-proportion z-test: are recruited cells over-concentrated in a disk
    of radius R around the reward, vs the uniform-arena chance proportion?"""
    print(f"\n--- Recruitment-Zone Enrichment Test ---")
    print(f"H0: recruited centers fall in a disk of radius {R} around {tuple(reward_pos)}")
    print(f"    with chance probability = π R² = {np.pi * R**2:.3f}")
    print(f"H1: observed proportion > chance")
    n_total = len(d_recruited)
    if n_total == 0:
        print("No recruited cells — cannot test.")
        return None
    n_near = int((d_recruited < R).sum())
    p_obs = n_near / n_total
    p_chance = np.pi * R**2
    print(f"Recruited cells with center within R of reward: {n_near}/{n_total} = {p_obs:.3f}")
    print(f"Chance:                                          {p_chance:.3f}")
    z = (p_obs - p_chance) / np.sqrt(p_chance * (1 - p_chance) / n_total)
    p_z = 1.0 - stats.norm.cdf(z)
    print(f"One-proportion z-test: z = {z:.3f}    p (one-sided) = {p_z:.4f}")
    print("-----------------------------------------")
    return {
        "test": "recruitment_zone_one_proportion_z_test",
        "n_total": n_total,
        "n_near": n_near,
        "p_obs": float(p_obs),
        "p_chance": float(p_chance),
        "z_stat": float(z),
        "p_value_one_sided": float(p_z),
    }


def build_rate_map(df_behavior, df_activity, neuron_id, m=32):
    """Single-neuron occupancy-normalized rate map (m × m), Gaussian-smoothed."""
    df_a = df_activity[df_activity["neuron_id"] == neuron_id]
    df = df_behavior.merge(df_a, on="timestep", how="inner")
    if len(df) == 0:
        return np.zeros((m, m))
    H, _, _ = np.histogram2d(df["x"].values, df["y"].values,
                              bins=m, range=[[0, 1], [0, 1]],
                              weights=df["activity"].values)
    occ, _, _ = np.histogram2d(df["x"].values, df["y"].values,
                                bins=m, range=[[0, 1], [0, 1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(occ > 0, H / occ, 0.0)
    rate = gaussian_filter(rate, sigma=1.5)
    if rate.max() > 0:
        rate = (rate - rate.min()) / (rate.max() - rate.min())
    return rate


def plot_recruited_before_after(recruited_ids, df_centers_after, reward_pos,
                                  df_centers_before_full, df_centers_after_full,
                                  df_beh_b, df_act_b, df_beh_a, df_act_a,
                                  output=None, show=True, n_plot=24, m=32):
    """For each recruited cell, plot its BEFORE rate map above its AFTER rate
    map, with k and ICL annotated so you can judge how close each cell was to
    crossing the place-cell threshold under a different cutoff. Sorted by
    distance to reward (closest first)."""
    df_recruited_centers = df_centers_after[df_centers_after["neuron_id"].isin(recruited_ids)]
    if len(df_recruited_centers) == 0:
        print("No recruited cells with centers — skipping BEFORE/AFTER figure.")
        return

    df_first = df_recruited_centers.drop_duplicates("neuron_id", keep="first").copy()
    df_first["d_rew"] = distances_to_reward(df_first["center"].values, reward_pos)
    plot_ids = df_first.sort_values("d_rew")["neuron_id"].tolist()[:n_plot]
    n_plot = len(plot_ids)

    icl_b = (df_centers_before_full.drop_duplicates("neuron_id")
             .set_index("neuron_id")[["k", "icl_score"]])
    icl_a = (df_centers_after_full.drop_duplicates("neuron_id")
             .set_index("neuron_id")[["k", "icl_score"]])

    n_cols = 6
    n_pair_rows = int(np.ceil(n_plot / n_cols))
    n_rows = n_pair_rows * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.9, n_rows * 1.9))
    axes = np.atleast_2d(axes)

    def fmt_icl(v):
        return "inf" if not np.isfinite(v) else f"{v:.1f}"

    for i, nid in enumerate(plot_ids):
        rate_b = build_rate_map(df_beh_b, df_act_b, nid, m=m)
        rate_a = build_rate_map(df_beh_a, df_act_a, nid, m=m)
        kb = icl_b.loc[nid, "k"] if nid in icl_b.index else float("nan")
        ib = icl_b.loc[nid, "icl_score"] if nid in icl_b.index else float("nan")
        ka = icl_a.loc[nid, "k"] if nid in icl_a.index else float("nan")
        ia = icl_a.loc[nid, "icl_score"] if nid in icl_a.index else float("nan")
        col = i % n_cols
        row_b = (i // n_cols) * 2
        row_a = row_b + 1
        axes[row_b, col].imshow(rate_b.T, origin="lower", cmap="jet", vmin=0, vmax=1)
        axes[row_a, col].imshow(rate_a.T, origin="lower", cmap="jet", vmin=0, vmax=1)
        axes[row_b, col].set_title(f"{int(nid)} BEFORE\nk={int(kb) if np.isfinite(kb) else '–'}  ICL={fmt_icl(ib)}", fontsize=7)
        axes[row_a, col].set_title(f"AFTER\nk={int(ka) if np.isfinite(ka) else '–'}  ICL={fmt_icl(ia)}", fontsize=7)
        axes[row_b, col].axis("off")
        axes[row_a, col].axis("off")

    for i in range(n_plot, n_pair_rows * n_cols):
        col = i % n_cols
        row_b = (i // n_cols) * 2
        row_a = row_b + 1
        axes[row_b, col].axis("off")
        axes[row_a, col].axis("off")

    fig.suptitle("Recruited cells (place cell in AFTER, not BEFORE) — sorted by distance to reward",
                 fontsize=11)
    plt.tight_layout()
    if output:
        save_all(output, dpi=150)
        print(f"Recruited BEFORE/AFTER figure saved to {output}")
    if show:
        plt.show()
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Place-cell recruitment analysis.")
    p.add_argument("--run_dir", type=str, default=".",
                    help="Run directory containing data/ and figures/ (e.g. Negative_Results).")
    p.add_argument("--reward", nargs=2, type=float, default=None,
                    help="Reward (x, y). Defaults to run_dir/config.json's reward_pos.")
    p.add_argument("--state_before", type=str, default="before",
                    help="Earlier state token, e.g. before, replay_c1_pred.")
    p.add_argument("--state_after", type=str, default="after",
                    help="Later state token, e.g. after, replay_c1_postrl.")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    data_dir = os.path.join(run_dir, "data")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    reward_point = tuple(resolve_reward_pos(run_dir, args.reward))
    print(f"Using reward point: {reward_point}")
    state_before, state_after = args.state_before, args.state_after
    pair_tag = f"{state_before}_vs_{state_after}"

    # --- Load after Data ---
    print(f"Loading '{state_after}' data...")
    df_behavior_after = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{state_after}.pkl"))
    df_activity_after = pd.read_pickle(os.path.join(data_dir, f"df_activations_{state_after}.pkl"))
    df_skaggs_after = pd.read_pickle(ensure_skaggs(run_dir, state_after))
    df_centers_after = pd.read_pickle(os.path.join(data_dir, f"df_centers_{state_after}.pkl"))
    df_combined_after = df_behavior_after.merge(
        df_activity_after, on="timestep", how="inner"
    )

    print(f"Filtering place cells {state_after}...")
    df_place_cells_after = filter_place_cells(df_skaggs_after, df_centers_after)

    print(f"Loading '{state_before}' data...")
    df_behavior_before = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{state_before}.pkl"))
    df_activity_before = pd.read_pickle(os.path.join(data_dir, f"df_activations_{state_before}.pkl"))
    df_skaggs_before = pd.read_pickle(ensure_skaggs(run_dir, state_before))
    df_centers_before = pd.read_pickle(os.path.join(data_dir, f"df_centers_{state_before}.pkl"))
    df_combined_before = df_behavior_before.merge(
        df_activity_before, on="timestep", how="inner"
    )

    print(f"Filtering place cells {state_before}...")
    df_place_cells_before = filter_place_cells(df_skaggs_before, df_centers_before)

    place_ids_before = set(df_place_cells_before["neuron_id"])
    place_ids_after = set(df_place_cells_after["neuron_id"])
    recruited_ids = place_ids_after - place_ids_before
    lost_ids = place_ids_before - place_ids_after
    shared_ids = place_ids_before & place_ids_after
    df_recurited = pd.DataFrame({"neuron_id": list(recruited_ids)})

    print("\n--- Population Summary ---")
    print(f"Place cells {state_before:20s}: {len(place_ids_before)}")
    print(f"Place cells {state_after:20s}: {len(place_ids_after)}")
    print(f"Shared (place in both):           {len(shared_ids)}")
    print(f"Recruited (place in {state_after} only): {len(recruited_ids)}")
    print(f"Lost (place in {state_before} only):     {len(lost_ids)}")

    # --- Population-level Hollup tests (independent two-sample, not paired) ---
    # Restrict to k==1 cells for distance analyses — single, unambiguous center
    # per cell. Matches the pattern in shift_toward_reward.py.
    centers_before_pc = df_centers_before[
        df_centers_before["neuron_id"].isin(place_ids_before) & (df_centers_before["k"] == 1)
    ]["center"].values
    centers_after_pc = df_centers_after[
        df_centers_after["neuron_id"].isin(place_ids_after) & (df_centers_after["k"] == 1)
    ]["center"].values
    d_before = distances_to_reward(centers_before_pc, reward_point)
    d_after = distances_to_reward(centers_after_pc, reward_point)
    stat = two_sample_distance_test(d_before, d_after)
    if stat is not None:
        stat["state_pair"] = pair_tag
        append_stat(run_dir, "recuritment_place_cells.py", stat)

    centers_recruited = df_centers_after[
        df_centers_after["neuron_id"].isin(recruited_ids) & (df_centers_after["k"] == 1)
    ]["center"].values
    d_recruited = distances_to_reward(centers_recruited, reward_point)
    stat = recruitment_zone_test(d_recruited, reward_point)
    if stat is not None:
        stat["state_pair"] = pair_tag
        append_stat(run_dir, "recuritment_place_cells.py", stat)

    # --- Figures ---
    # Existing: AFTER-period heatmap grid for recruited cells
    create_cells_figure(df_combined=df_combined_after, neurons=df_recurited,
                        output=os.path.join(fig_dir, f"recruited_cells_{pair_tag}"), show=False)

    # New: BEFORE/AFTER side-by-side with k + ICL annotations (judge near-misses)
    plot_recruited_before_after(
        recruited_ids=recruited_ids,
        df_centers_after=df_centers_after,
        reward_pos=reward_point,
        df_centers_before_full=df_centers_before,
        df_centers_after_full=df_centers_after,
        df_beh_b=df_behavior_before, df_act_b=df_activity_before,
        df_beh_a=df_behavior_after, df_act_a=df_activity_after,
        output=os.path.join(fig_dir, f"recruited_cells_before_after_{pair_tag}"),
        show=False,
    )

    # --- Existing paired shift analysis (kept; answers a different question) ---
    all_shifts, distance_pairs = calculate_shifts_and_distances(
        df_recurited, df_centers_before, df_centers_after, reward_point
    )
    plot_shift_comparison(
        distance_pairs, output=os.path.join(fig_dir, f"recruitment_shift_comparison_{pair_tag}"), show=False
    )
    stat = perform_shift_t_test(all_shifts)
    if stat is not None:
        stat["state_pair"] = pair_tag
        append_stat(run_dir, "recuritment_place_cells.py", stat)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
