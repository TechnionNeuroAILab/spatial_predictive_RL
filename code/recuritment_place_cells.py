import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.patches as patches
import os

# --- CONSTANTS ---
REWARD_POINT = (0.5, 0.5)

def create_cells_figure(df_combined, neurons, output=None, show=True):
    """
    Takes the combined dataframe of activity and behavior of all of the neurons, another dataframe that contains only unique "neuron_id"
    merges them and saves the figure in the figures folder. Can print maximum of 150 neurons.
    If there is no output, won't save. Can turn off show by setting to false.
    """
    if len(neurons) > 150:
        print("There are too many neurons.")
        return None

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
        plt.savefig(output)
        print(f"Figure saved to {output}")
    plt.close()


def filter_place_cells(df_skaggs, df_centers, skaggs_prec=None):
    """
    Takes a dataframe that contains the skaggs score and icl score per neuron and returns a dataframe with columns "neuron_id"
    that will be called place cells.
    """
    df_centers = df_centers[df_centers["k"] == 1]
    df = df_skaggs.merge(df_centers, on="neuron_id", how="inner")
    if skaggs_prec:
        skaggs_quantile = df["skaggs_score"].quantile(skaggs_prec)
        df = df[df["skaggs_score"] >= skaggs_quantile]
        print(skaggs_quantile)
    else:
        df = df[(df["skaggs_score"] >= 0.2) & (df["icl_score"] < -29000)]
    return df[["neuron_id"]].drop_duplicates()

def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two (x, y) tuples."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_shifts_and_distances(df_place_cells, df_centers_before, df_centers_after):
    """
    Calculates the shift of place fields towards the reward point (0.5, 0.5).

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
        if len(center_before) > 1:  # skip if after there is more than one center
            count += 1
            continue
        distance_before = euclidean_distance(REWARD_POINT, center_before[0])
        distance_after = euclidean_distance(REWARD_POINT, center_after[0])
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
        plt.savefig(output)
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
        plt.savefig(output)
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
        return

    n = len(shifts)
    if n < 2:
        print("Not enough shift data (N<2) to perform a t-test.")
        return

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


def main():
    # --- Load after Data ---
    print("Loading 'after' data...")
    df_behavior_after = pd.read_pickle("data/df_behavior_after.pkl")
    df_activity_after = pd.read_pickle("data/df_activations_after.pkl")
    df_skaggs_after = pd.read_pickle("data/df_skaggs_after.pkl")
    df_centers_after = pd.read_pickle("data/df_centers_after.pkl")
    df_combined_after = df_behavior_after.merge(
        df_activity_after, on="timestep", how="inner"
    )

    print("Filtering place cells after...")
    df_place_cells_after = filter_place_cells(df_skaggs_after, df_centers_after)

    print("Loading 'before' data...")
    df_behavior_before = pd.read_pickle("data/df_behavior_before.pkl")
    df_activity_before = pd.read_pickle("data/df_activations_before.pkl")
    df_skaggs_before = pd.read_pickle("data/df_skaggs_before.pkl")
    df_centers_before = pd.read_pickle("data/df_centers_before.pkl")
    df_combined_before = df_behavior_before.merge(
        df_activity_before, on="timestep", how="inner"
    )

    print("Filtering place cells before...")
    df_place_cells_before = filter_place_cells(df_skaggs_before, df_centers_before)

    diff_ids = set(df_place_cells_after["neuron_id"]) - set(df_place_cells_before["neuron_id"])

    df_recurited = pd.DataFrame({"neuron_id": list(diff_ids)})
    
    create_cells_figure(df_combined=df_combined_after, neurons=df_recurited)

        # 1. Calculate shifts and distance pairs
    all_shifts, distance_pairs = calculate_shifts_and_distances(
        df_recurited, df_centers_before, df_centers_after
    )

    # 3. Plot the new comparison scatter plot
    plot_shift_comparison(
        distance_pairs, show=True
    )

    # 4. Perform the requested Wald test
    perform_shift_t_test(all_shifts)

    print("Analysis complete.")

    

if __name__ == "__main__":
    main()
