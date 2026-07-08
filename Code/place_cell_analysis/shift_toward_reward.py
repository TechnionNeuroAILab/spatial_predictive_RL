import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from pc_common import (
    append_stat,
    ensure_skaggs,
    filter_place_cells,
    resolve_reward_pos,
    save_all,
)


# --- [Original Functions: create_cells_figure] ---
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


# --- [Original Functions: euclidean_distance, calculate_shifts_and_distances, ...] ---
def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two (x, y) tuples."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _nearest_center(centers, reward_point):
    """Given a list of (x, y) centers, return the one nearest to reward_point."""
    return min(centers, key=lambda c: euclidean_distance(reward_point, c))


def calculate_shifts_and_distances(df_place_cells, df_centers_before, df_centers_after, reward_point):
    """
    Calculates the shift of place fields towards reward_point.
    This is for the PAIRED analysis.
    df_place_cells is expected to be a dataframe with just 'neuron_id'.

    For neurons with more than one GMM center on either side, the center
    nearest to the reward point is used (applied symmetrically to both
    'before' and 'after', so neither side is arbitrarily favored).
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
    n_multi_before = 0
    n_multi_after = 0

    for neuron_id in tqdm(
        common_neurons_to_track, desc="Processing neurons (place cells)"
    ):
        center_before = df_before_filtered[
            df_before_filtered["neuron_id"] == neuron_id
        ]["center"].tolist()
        center_after = df_after_filtered[df_after_filtered["neuron_id"] == neuron_id][
            "center"
        ].tolist()
        if not center_before or not center_after:
            continue

        if len(center_before) > 1:
            n_multi_before += 1
        if len(center_after) > 1:
            n_multi_after += 1

        cb = _nearest_center(center_before, reward_point)
        ca = _nearest_center(center_after, reward_point)
        distance_before = euclidean_distance(reward_point, cb)
        distance_after = euclidean_distance(reward_point, ca)
        shift = distance_before - distance_after
        all_shifts.append(shift)
        distance_pairs.append((distance_before, distance_after))

    print(
        f"Multi-center neurons resolved via nearest-to-reward: "
        f"{n_multi_before} in 'before', {n_multi_after} in 'after'"
    )
    return all_shifts, distance_pairs


def plot_shift_histogram(all_shifts, output=None, show=True):
    """Plots a histogram of the calculated shifts."""
    if not all_shifts:
        print("No shifts were calculated. Cannot plot histogram.")
        return

    print(f"Plotting histogram for {len(all_shifts)} field shifts.")
    plt.figure(figsize=(10, 6))
    sns.histplot(all_shifts, kde=True, bins=20)
    plt.title("Distribution of Place Field Shifts Towards Reward (Paired)", fontsize=16)
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

    plt.title("Place Field Distance to Reward: Before vs. After (Paired)", fontsize=16)
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

    print("\n--- Place Field Shift t-Test (One-Sample on Differences) ---")
    print(f"H0: True mean shift = 0")
    print(f"H1: True mean shift > 0 (fields moved towards reward)")
    print("-----------------------------------------")
    print(f"Sample size (N): {n}")
    print(f"Sample mean (Dist_Before - Dist_After): {mean_shift:.4f}")
    print(f"Sample std dev of shift: {std_dev:.4f}")
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
        "test": "paired_shift_one_sample_t_test",
        "n": n,
        "mean_shift": float(mean_shift),
        "std_dev": float(std_dev),
        "t_stat": float(t_stat),
        "p_value_one_tailed": float(p_value),
    }


# --- [Existing New Functions: plot_distance_histograms, perform_paired_distance_t_test] ---

def plot_distance_histograms(distances_before, distances_after, title, output, show=True):
    """
    Plots overlaid histograms for the 'before' and 'after' distances
    to the reward point.
    """
    if not distances_before or not distances_after:
        print("No 'before' or 'after' distance data to plot histograms.")
        return

    print(f"Plotting distance histograms for {len(distances_before)} (before) vs. {len(distances_after)} (after) cells.")

    # Create a DataFrame for easy plotting with Seaborn
    n_before = len(distances_before)
    n_after = len(distances_after)
    
    df_data = {
        'distance': distances_before + distances_after,
        'condition': ['Before RL'] * n_before + ['After RL'] * n_after
    }
    df_plot = pd.DataFrame(df_data)

    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=df_plot,
        x='distance',
        hue='condition',
        kde=True,         # Add Kernel Density Estimate plot
        element='step',   # Use steps instead of bars for better overlay
        stat='density',   # Show density to compare shapes, not counts
        common_norm=False # Normalize each histogram independently
    )
    
    # Calculate and show means
    mean_before = np.mean(distances_before)
    mean_after = np.mean(distances_after)
    
    plt.axvline(mean_before, color=sns.color_palette()[0], linestyle='--', label=f'Mean Before (N={n_before}): {mean_before:.3f}')
    plt.axvline(mean_after, color=sns.color_palette()[1], linestyle='--', label=f'Mean After (N={n_after}): {mean_after:.3f}')
    
    plt.title(title, fontsize=16)
    plt.xlabel("Distance to Reward Point", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    if output:
        save_all(output)
        print(f"Distance histograms saved to {output}")
    if show:
        plt.show()
    plt.close()


def perform_paired_distance_t_test(distances_before, distances_after, alpha=0.05):
    """
    Performs a paired, one-tailed t-test on the 'before' and 'after' distances.
    """
    if not distances_before or not distances_after or len(distances_before) < 2:
        print("Not enough paired data (N<2) to perform a paired t-test.")
        return
    
    if len(distances_before) != len(distances_after):
        print("Error: Paired t-test requires lists of equal length.")
        return

    n = len(distances_before)
    
    t_stat, p_value = stats.ttest_rel(
        distances_before, 
        distances_after, 
        alternative='greater'
    )
    
    mean_before = np.mean(distances_before)
    mean_after = np.mean(distances_after)
    
    print("\n--- Place Field Distance Paired t-Test ---")
    print(f"H0: True mean(dist_before) = True mean(dist_after)")
    print(f"H1: True mean(dist_before) > True mean(dist_after) (fields moved closer)")
    print("-----------------------------------------")
    print(f"Sample size (N pairs): {n}")
    print(f"Sample mean (Before):  {mean_before:.4f}")
    print(f"Sample mean (After):   {mean_after:.4f}")
    print(f"t statistic:           {t_stat:.4f}")
    print(f"p-value (one-tailed):  {p_value:.4f}")
    print("-----------------------------------------")

    if p_value < alpha:
        print(f"Conclusion: p-value ({p_value:.4f}) < alpha ({alpha}).")
        print("We REJECT the null hypothesis (H0).")
        print("The mean distance 'before' is significantly greater than 'after'.")
        print("This suggests a significant shift towards the reward point.")
    else:
        print(f"Conclusion: p-value ({p_value:.4f}) >= alpha ({alpha}).")
        print("We FAIL to reject the null hypothesis (H0).")
        print("There is no statistically significant difference between mean distances.")
    print("-----------------------------------------\n")


# --- [Existing New Functions: get_distances_for_population, perform_independent_distance_t_test] ---

def get_distances_for_population(df_place_cells_with_centers, reward_point):
    """
    Calculates distances to reward for a dataframe of place cells.
    Assumes df_place_cells_with_centers has a 'center' column with (x, y) tuples.
    """
    if 'center' not in df_place_cells_with_centers.columns:
        print("Error: 'center' column not found in place cell dataframe.")
        return []

    distances = [
        euclidean_distance(reward_point, center)
        for center in df_place_cells_with_centers['center']
    ]
    return distances


def perform_independent_distance_t_test(distances_before, distances_after, alpha=0.05):
    """
    Performs an independent, two-sample, one-tailed t-test (Welch's T-test)
    on the 'before' and 'after' distance populations.

    This test is specifically set up to test:
    H0: The true mean 'before' distance is less than or equal to the 'after' distance.
        (mean(dist_before) <= mean(dist_after))
    H1: The true mean 'before' distance is greater than the 'after' distance.
        (mean(dist_before) > mean(dist_after))
    """
    n_before = len(distances_before)
    n_after = len(distances_after)

    if n_before < 2 or n_after < 2:
        print("Not enough data (N<2 in one or both groups) for independent t-test.")
        return None

    mean_before = np.mean(distances_before)
    mean_after = np.mean(distances_after)

    # Use Welch's T-test (equal_var=False) by default, robust to unequal variances
    # The 'alternative='greater'' parameter correctly sets the H0 and H1
    # as defined in the docstring above.
    t_stat, p_value = stats.ttest_ind(
        distances_before,
        distances_after,
        equal_var=False,  # Perform Welch's T-test
        alternative='greater' # H1: mean(before) > mean(after)
    )

    print("\n--- Place Field Distance Independent t-Test (Welch's) ---")
    print("Comparing two independent populations: 'Before-RL PCs' vs 'After-RL PCs'")
    
    # --- THIS IS THE ONLY LINE THAT CHANGED ---
    print(f"H0: True mean(dist_before) <= True mean(dist_after)")
    
    print(f"H1: True mean(dist_before) > True mean(dist_after)")
    print("-----------------------------------------")
    print(f"Sample size (N Before): {n_before}")
    print(f"Sample size (N After):  {n_after}")
    print(f"Sample mean (Before):   {mean_before:.4f}")
    print(f"Sample mean (After):    {mean_after:.4f}")
    print(f"t statistic:            {t_stat:.4f}")
    print(f"p-value (one-tailed):   {p_value:.4f}")
    print("-----------------------------------------")

    if p_value < alpha:
        print(f"Conclusion: p-value ({p_value:.4f}) < alpha ({alpha}).")
        print("We REJECT the null hypothesis (H0).")
        print("The mean distance of the 'Before' population is significantly greater than 'After'.")
    else:
        print(f"Conclusion: p-value ({p_value:.4f}) >= alpha ({alpha}).")
        print("We FAIL to reject the null hypothesis (H0).")
        print("No significant difference found (mean_before is not significantly > mean_after).")
    print("-----------------------------------------\n")

    return {
        "test": "independent_distance_welch_t_test",
        "n_before": n_before,
        "n_after": n_after,
        "mean_before": float(mean_before),
        "mean_after": float(mean_after),
        "t_stat": float(t_stat),
        "p_value_one_tailed": float(p_value),
    }


# --- **** FIXED FUNCTION **** ---
# --- This function now works with the original filter_place_cells ---

def run_independent_population_analysis(
    df_skaggs_before, df_centers_before,
    df_skaggs_after, df_centers_after,
    reward_point, run_dir, fig_dir, pair_tag,
):
    """
    Runs the independent population analysis.
    1. Filters place cell IDs from 'before' data.
    2. Filters place cell IDs from 'after' data.
    3. Merges with centers to get distances.
    4. Plots overlaid histograms.
    5. Performs an independent t-test.
    """
    print("\n" + "="*50)
    print("--- Running Independent Population Analysis ---")
    print(" (Comparing 'Before PC Pop.' vs 'After PC Pop.')")
    print("="*50)
    
    # 1. Filter 'before' population IDs
    print("Filtering 'before' place cells...")
    df_pc_ids_before = filter_place_cells(df_skaggs_before, df_centers_before)
    
    # Get k=1 centers for *all* neurons, then merge
    # This is necessary because filter_place_cells *uses* k=1 but doesn't return the centers
    df_centers_before_k1 = df_centers_before[df_centers_before['k'] == 1]
    
    # Merge the filtered IDs with their corresponding k=1 centers
    df_pc_before_with_centers = df_pc_ids_before.merge(
        df_centers_before_k1[['neuron_id', 'center']], 
        on='neuron_id', 
        how='inner'
    )
    print(f"Found {len(df_pc_before_with_centers)} place cells in 'before' data.")
    
    # 2. Filter 'after' population IDs
    print("Filtering 'after' place cells...")
    df_pc_ids_after = filter_place_cells(df_skaggs_after, df_centers_after)
    
    # Get k=1 centers for *all* neurons, then merge
    df_centers_after_k1 = df_centers_after[df_centers_after['k'] == 1]
    
    # Merge the filtered IDs with their corresponding k=1 centers
    df_pc_after_with_centers = df_pc_ids_after.merge(
        df_centers_after_k1[['neuron_id', 'center']], 
        on='neuron_id', 
        how='inner'
    )
    print(f"Found {len(df_pc_after_with_centers)} place cells in 'after' data.")
    
    # 3. Get distances for each population
    distances_before_pop = get_distances_for_population(df_pc_before_with_centers, reward_point)
    distances_after_pop = get_distances_for_population(df_pc_after_with_centers, reward_point)

    if not distances_before_pop or not distances_after_pop:
        print("No data in one or both populations. Halting independent analysis.")
        return

    plot_distance_histograms(
        distances_before_pop,
        distances_after_pop,
        title="Distribution of Independent Place Cell Populations",
        output=os.path.join(fig_dir, f"independent_distance_histograms_{pair_tag}"),
        show=False,
    )

    # 5. Perform independent t-test
    stat = perform_independent_distance_t_test(distances_before_pop, distances_after_pop)
    if stat is not None:
        stat["state_pair"] = pair_tag
        append_stat(run_dir, "shift_toward_reward.py", stat)

    print("--- Independent Population Analysis Complete ---")


# --- **** FIXED FUNCTION **** ---
# --- The main function is also updated to work with the original filter_place_cells ---

def parse_args():
    p = argparse.ArgumentParser(description="Place-field shift toward reward analysis.")
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

    # --- Load All Data ---
    print(f"Loading '{state_before}' data...")
    df_behavior_before = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{state_before}.pkl"))
    df_activity_before = pd.read_pickle(os.path.join(data_dir, f"df_activations_{state_before}.pkl"))
    df_skaggs_before = pd.read_pickle(ensure_skaggs(run_dir, state_before))
    df_centers_before = pd.read_pickle(os.path.join(data_dir, f"df_centers_{state_before}.pkl"))
    df_combined_before = df_behavior_before.merge(
        df_activity_before, on="timestep", how="inner"
    )

    print(f"Loading '{state_after}' data...")
    df_behavior_after = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{state_after}.pkl"))
    df_activity_after = pd.read_pickle(os.path.join(data_dir, f"df_activations_{state_after}.pkl"))
    df_centers_after = pd.read_pickle(os.path.join(data_dir, f"df_centers_{state_after}.pkl"))
    df_combined_after = df_behavior_after.merge(
        df_activity_after, on="timestep", how="inner"
    )
    df_skaggs_after = pd.read_pickle(ensure_skaggs(run_dir, state_after))

    # --- ANALYSIS 1: PAIRED (Tracking cells defined in state_before) ---
    print("\n" + "="*50)
    print("--- Running Paired Analysis ---")
    print(f" (Tracking '{state_before}' PCs and seeing how they changed by '{state_after}')")
    print("="*50)

    # 1. Define place cells based on the earlier state
    print(f"Filtering place cells (based on '{state_before}' data)...")
    df_place_cells_before_ids = filter_place_cells(df_skaggs_before, df_centers_before)
    print(f"Found {len(df_place_cells_before_ids)} place cells to track.")

    # 2. Create figures for this group in both states
    create_cells_figure(
        df_combined_before,
        df_place_cells_before_ids,
        output=os.path.join(fig_dir, f"place_cells_{state_before}_{pair_tag}"),
        show=False,
    )
    create_cells_figure(
        df_combined_after,
        df_place_cells_before_ids,
        output=os.path.join(fig_dir, f"place_cells_{state_after}_{pair_tag}"),
        show=False,
    )

    # 2b. Place cells defined independently on the later state
    print(f"Filtering place cells (based on '{state_after}' data)...")
    df_place_cells_after_ids = filter_place_cells(df_skaggs_after, df_centers_after)
    print(f"Found {len(df_place_cells_after_ids)} place cells in '{state_after}' data.")
    create_cells_figure(
        df_combined_after,
        df_place_cells_after_ids,
        output=os.path.join(fig_dir, f"place_cells_{state_after}_only_{pair_tag}"),
        show=False,
    )

    # 3. Calculate shifts and distances for this group
    all_shifts, distance_pairs = calculate_shifts_and_distances(
        df_place_cells_before_ids, df_centers_before, df_centers_after, reward_point
    )

    plot_shift_histogram(all_shifts, output=os.path.join(fig_dir, f"shift_histogram_{pair_tag}"), show=False)
    plot_shift_comparison(
        distance_pairs, output=os.path.join(fig_dir, f"shift_comparison_scatter_{pair_tag}"), show=False
    )

    # 5. Perform one-sample t-test on shifts (equivalent to paired test)
    stat = perform_shift_t_test(all_shifts)
    if stat is not None:
        stat["state_pair"] = pair_tag
        append_stat(run_dir, "shift_toward_reward.py", stat)

    shifts_path = os.path.join(data_dir, f"paired_shifts_{pair_tag}.npy")
    np.save(shifts_path, np.array(all_shifts, dtype=float))
    print(f"Saved paired shifts to {shifts_path}")

    print("--- Paired Analysis Complete ---")

    # --- ANALYSIS 2: INDEPENDENT ---
    run_independent_population_analysis(
        df_skaggs_before, df_centers_before,
        df_skaggs_after, df_centers_after,
        reward_point, run_dir, fig_dir, pair_tag,
    )

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
