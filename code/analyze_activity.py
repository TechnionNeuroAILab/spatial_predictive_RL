import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DATA_DIR = "data"
REWARD_POS = np.array([0.5, 0.5])
D_R = 0.05
LOOKING_THRESHOLD_RAD = np.deg2rad(5)
EPSILON = 1e-9  # To prevent division by zero in SI calculation


def load_and_merge_data():
    """Loads pickles and merges behavior with neural data."""
    df_act = pd.read_pickle(os.path.join(DATA_DIR, "df_activations_after.pkl"))
    df_beh = pd.read_pickle(os.path.join(DATA_DIR, "df_behavior_after.pkl"))

    # Merge on timestep to align behavior with neural states
    # Assuming inner join to keep only timesteps present in both
    df_merged = pd.merge(df_act, df_beh, on="timestep")
    return df_merged


def calculate_spatial_metrics(df):
    """Adds distance to reward and determines 'Near' status."""
    # Calculate Euclidean distance to reward (0.5, 0.5)
    df["dist_to_reward"] = np.sqrt(
        (df["x"] - REWARD_POS[0]) ** 2 + (df["y"] - REWARD_POS[1]) ** 2
    )

    # Boolean mask: Is the agent within radius D_R?
    df["is_near_reward"] = df["dist_to_reward"] < D_R
    return df


def calculate_angular_metrics(df):
    """Calculates if the agent is looking at the reward."""
    # 1. Calculate the angle of the vector pointing TO the reward
    # arctan2(dy, dx)
    vec_y = REWARD_POS[1] - df["y"]
    vec_x = REWARD_POS[0] - df["x"]
    df["target_angle"] = np.arctan2(vec_y, vec_x)

    # 2. Calculate difference between agent's theta and target angle
    # We use arctan2(sin, cos) to handle the wrap-around (e.g. 359 deg vs 1 deg)
    # This ensures the error is always between -pi and pi
    diff = df["theta"] - df["target_angle"]
    df["angle_error"] = np.arctan2(np.sin(diff), np.cos(diff))

    # 3. Is the agent looking at the reward? (Within the cone)
    df["is_looking_at_reward"] = np.abs(df["angle_error"]) < LOOKING_THRESHOLD_RAD
    return df


def compute_selectivity(df, condition_col, neuron_col="neuron_id", act_col="activity"):
    """
    Computes Selectivity Index for each neuron based on a boolean condition.
    SI = (Mean_Condition - Mean_Rest) / (Mean_Condition + Mean_Rest)
    """
    # Group by neuron and condition
    grouped = df.groupby([neuron_col, condition_col])[act_col].mean().unstack()

    # Rename columns for clarity (False -> Rest, True -> Condition)
    grouped.columns = ["mean_rest", "mean_condition"]

    # Calculate SI
    # We add epsilon to avoid division by zero if a neuron is silent
    numerator = grouped["mean_condition"] - grouped["mean_rest"]
    denominator = grouped["mean_condition"] + grouped["mean_rest"] + EPSILON

    grouped["selectivity_index"] = numerator / denominator

    return grouped.sort_values("selectivity_index", ascending=False)


def plot_results(spatial_stats, angular_stats):
    """Visualizes the metrics across all neurons."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # --- Plot 1: Spatial Selectivity ---
    # We reset index to make neuron_id a column for plotting,
    # and we create a generic index 0-511 for the x-axis sorted by magnitude
    spatial_sorted = spatial_stats.sort_values(
        "selectivity_index", ascending=False
    ).reset_index()

    sns.scatterplot(
        data=spatial_sorted,
        x=spatial_sorted.index,
        y="selectivity_index",
        ax=axes[0],
        hue="selectivity_index",
        palette="vlag",
        legend=False,
        s=15,
    )

    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"Spatial Reward Selectivity (Radius < {D_R})", fontsize=14)
    axes[0].set_ylabel("Selectivity Index\n(Near - Far) / (Near + Far)")
    axes[0].set_xlabel("Neurons (Sorted by Selectivity)")

    # --- Plot 2: Angular Selectivity ---
    angular_sorted = angular_stats.sort_values(
        "selectivity_index", ascending=False
    ).reset_index()

    sns.scatterplot(
        data=angular_sorted,
        x=angular_sorted.index,
        y="selectivity_index",
        ax=axes[1],
        hue="selectivity_index",
        palette="vlag",
        legend=False,
        s=15,
    )

    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title(
        f"Gaze Selectivity (Looking at Reward +/- {np.rad2deg(LOOKING_THRESHOLD_RAD):.0f}°)",
        fontsize=14,
    )
    axes[1].set_ylabel("Selectivity Index\n(Looking - Not Looking) / Sum")
    axes[1].set_xlabel("Neurons (Sorted by Selectivity)")

    plt.tight_layout()
    plt.savefig("figures/Selectivity_Index_activ_ang.png")
    plt.show()

    return spatial_sorted, angular_sorted


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load
    df = load_and_merge_data()

    if df is not None:
        print(f"Data loaded. Shape: {df.shape}")

        # 2. Process Features
        df = calculate_spatial_metrics(df)
        df = calculate_angular_metrics(df)

        # 3. Compute Metrics
        print("Computing Spatial Selectivity...")
        spatial_stats = compute_selectivity(df, condition_col="is_near_reward")

        print("Computing Angular Selectivity...")
        angular_stats = compute_selectivity(df, condition_col="is_looking_at_reward")

        # 4. Visualize
        sp_sorted, ang_sorted = plot_results(spatial_stats, angular_stats)

        # 5. Output Top Neurons
        print("\nTop 5 'Reward Location' Neurons:")
        print(
            sp_sorted[
                ["neuron_id", "mean_condition", "mean_rest", "selectivity_index"]
            ].head(5)
        )

        print("\nTop 5 'Gaze at Reward' Neurons:")
        print(
            ang_sorted[
                ["neuron_id", "mean_condition", "mean_rest", "selectivity_index"]
            ].head(5)
        )
        print(f"Amount of neurons that have positive spatial SI {len(sp_sorted[sp_sorted["selectivity_index"] > 0])}")
        print(f"Amount of neurons that have positive angular SI {len(ang_sorted[ang_sorted["selectivity_index"] > 0])}")
