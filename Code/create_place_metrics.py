import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_skaggs_scores_optimized(df_combined, output, m=32):
    """
    Vectorized computation of Skaggs scores for all neurons.

    I = ∑ mean_activity_on_grid_k *
        log2(mean_activity_on_grid_k/mean_activity_overall) *
        probability_grid_k
    """

    # Precompute grid assignments for all positions
    # Convert continuous positions to grid indices
    grid_x = np.floor(df_combined["x"] * m).astype(int)
    grid_y = np.floor(df_combined["y"] * m).astype(int)

    # Create a single grid cell ID for each position
    grid_cell = grid_x * m + grid_y

    # Add grid cell info to dataframe
    df_with_grid = df_combined.copy()
    df_with_grid["grid_cell"] = grid_cell

    # Compute mean activity per neuron (for normalization)
    neuron_mean_activity = df_with_grid.groupby("neuron_id")["activity"].mean()

    # Group by neuron and grid cell, compute mean activity in each cell
    cell_activities = (
        df_with_grid.groupby(["neuron_id", "grid_cell"])["activity"]
        .agg(["mean", "count"])
        .reset_index()
    )
    cell_activities.columns = ["neuron_id", "grid_cell", "mean_activity_cell", "count"]

    # Filter out cells with no activity
    cell_activities = cell_activities[cell_activities["mean_activity_cell"] > 0]

    # Add overall mean activity for each neuron
    cell_activities = cell_activities.merge(
        neuron_mean_activity.reset_index().rename(
            columns={"activity": "mean_activity_overall"}
        ),
        on="neuron_id",
    )

    # Compute Skaggs information for each cell
    p = 1.0 / (m * m)  # probability of each grid cell

    cell_activities["log_ratio"] = np.log2(
        cell_activities["mean_activity_cell"] / cell_activities["mean_activity_overall"]
    )

    cell_activities["skaggs_contribution"] = (
        cell_activities["mean_activity_cell"] * cell_activities["log_ratio"] * p
    )

    # Sum contributions for each neuron
    skaggs_scores = (
        cell_activities.groupby("neuron_id")["skaggs_contribution"]
        .sum()
        .reindex(neuron_mean_activity.index, fill_value=0)
    )

    scores_df = pd.DataFrame(
        {"neuron_id": skaggs_scores.index, "skaggs_score": skaggs_scores.values}
    )
    scores_df.to_pickle(output)
    return skaggs_scores.values


def plot_skaggs_scores_before_after(
    skaggs_scores_before, skaggs_scores_after, save_path, sort=True
):
    skaggs_b = skaggs_scores_before.copy()
    skaggs_a = skaggs_scores_after.copy()
    if sort:
        skaggs_b.sort()
        skaggs_a.sort()
    df_plot = pd.DataFrame(
        {
            "neuron": np.concatenate(
                [
                    np.arange(len(skaggs_b)),
                    np.arange(len(skaggs_a)),
                ]
            ),
            "score": np.concatenate([skaggs_b, skaggs_a]),
            "condition": np.repeat(
                ["Before", "After"],
                [len(skaggs_b), len(skaggs_a)],
            ),
        }
    )

    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df_plot, x="neuron", y="score", hue="condition")
    plt.ylabel("Skaggs score")
    if sort:
        plt.xlabel("Neuron ID")
        plt.title("Skaggs Scores: Before vs After")
    else:
        plt.xlabel("Array Index")
        plt.title("Skaggs Scores(Sorted): Before vs After")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.cla()


def plot_bottom_center_top_k_skaggs_neurons_heatmaps(
    df_combined, skaggs_scores_before, skaggs_scores_after, save_path, k=8
):
    """
    Plot individual heatmaps for specific neurons (compact layout for up to 100).

    Parameters:
    df_combined: DataFrame with columns ['x', 'y', 'activity', 'neuron_id']
    neuron_ids: array-like of neuron IDs to plot
    max_plots: maximum number of individual plots to show (default 100)
    """
    sorted_indices = np.argsort(skaggs_scores_after)

    # Get the k bottom, k center, and k top indices
    bottom_k_indices = sorted_indices[:k]
    top_k_indices = sorted_indices[-k:]
    midpoint = len(skaggs_scores_after) // 2
    center_k_indices = sorted_indices[midpoint - (k // 2) : midpoint + (k // 2)]

    sorted_indices = np.argsort(skaggs_scores_before)

    # Get the k bottom, k center, and k top indices
    bottom_2k_indices = np.concatenate((sorted_indices[:k], bottom_k_indices))
    top_2k_indices = np.concatenate((sorted_indices[-k:], top_k_indices))

    center_2k_indices = np.concatenate(
        (sorted_indices[midpoint - (k // 2) : midpoint + (k // 2)], center_k_indices)
    )

    # Merge all three arrays into one
    merged = np.concatenate((bottom_2k_indices, center_2k_indices, top_2k_indices))

    # Grid
    n_plots = k * 2 * 3
    n_cols = k
    n_rows = 2 * 3

    # Compact figure size; ~1.3" per subplot side for readability
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.3 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, neuron_id in enumerate(merged):
        df_neuron = df_combined[df_combined["neuron_id"] == neuron_id]

        if len(df_neuron) > 0:
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
    plt.savefig(save_path)
    plt.cla()


REWARD = np.array([0.5, 0.5])  # center reward (x=0.5, y=0.5)


def dist_to_reward_mean(centers, reward=REWARD):
    """
    The average distance from reward to centers.
    """
    d = np.linalg.norm(centers - reward, axis=1)
    return float(d.mean())


def dist_to_reward_min(centers, reward=REWARD):
    """
    The minimum distance from centers to the reward.
    """
    d = np.linalg.norm(centers - reward, axis=1)
    return float(d.min())



# --- Plot helper ---
def plot_per_neuron_delta(df, out_path, title):
    # Sort by delta for readability
    plot_df = df.sort_values("delta").reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(plot_df)), plot_df["delta"])
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Neuron (sorted by shift)")
    plt.ylabel("Δ distance to reward (before - after)")
    # Optionally thin x-ticks for large N
    if len(plot_df) <= 60:
        plt.xticks(range(len(plot_df)), plot_df["neuron_id"], rotation=90, fontsize=8)
    else:
        # fewer ticks for large sets
        step = max(1, len(plot_df) // 30)
        idxs = list(range(0, len(plot_df), step))
        labels = plot_df.loc[idxs, "neuron_id"].astype(str).tolist()
        plt.xticks(idxs, labels, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    # Load
    df_behavior_before = pd.read_pickle("data/df_behavior_before.pkl")
    df_activity_before = pd.read_pickle("data/df_activations_before.pkl")
    df_combined_before = df_behavior_before.merge(
        df_activity_before, on="timestep", how="inner"
    )
    df_behavior_after = pd.read_pickle("data/df_behavior_after.pkl")
    df_activity_after = pd.read_pickle("data/df_activations_after.pkl")
    df_combined_after = df_behavior_after.merge(
        df_activity_after, on="timestep", how="inner"
    )
    import os
    if not os.path.exists("data/df_skaggs_before.pkl"):
        print("Computing skaggs scores for Phase 1...")
        compute_skaggs_scores_optimized(df_combined_before, "data/df_skaggs_before.pkl")
    if not os.path.exists("data/df_skaggs_after.pkl"):
        print("Computing skaggs scores for Phase 2...")
        compute_skaggs_scores_optimized(df_combined_after, "data/df_skaggs_after.pkl")
        
    df_skaggs_before = pd.read_pickle("data/df_skaggs_before.pkl")
    df_skaggs_after = pd.read_pickle("data/df_skaggs_after.pkl")
    
    scores_before = df_skaggs_before["skaggs_score"].to_numpy()
    scores_after = df_skaggs_after["skaggs_score"].to_numpy()
    
    after_raw = pd.read_pickle("data/df_centers_after.pkl")
    after_raw = after_raw[after_raw["k"] <= 3]
    after_si = df_skaggs_after.merge(after_raw, on="neuron_id", how="inner")
    after_si = after_si[after_si["icl_score"] < -25_000]
    
    after_si = after_si[after_si["skaggs_score"] > 0.30]
    print(after_si["neuron_id"].unique())
    df_combined = df_combined_after.merge(after_si, on="neuron_id", how="inner")
    # before_raw = pd.read_pickle("data/df_centers_before.pkl")
    # after_raw = pd.read_pickle("data/df_centers_after.pkl")
    # icl_scatterplot(before_raw,"figures/icl_scatter_before.png")
    # icl_scatterplot(after_raw,"figures/icl_scatter_after.png")
    # print("The distribution of k-bumps in before reward")
    # print(before_raw["k"].value_counts(sort=True))
    # print("The distribution of k-bumps in after reward")
    # print(after_raw["k"].value_counts(sort=True))

    # before = prep(before_raw).rename(
    #     columns={"k": "k_before", "d2reward_mean": "d_before"}
    # )
    # after = prep(after_raw).rename(columns={"k": "k_after", "d2reward_mean": "d_after"})

    # # Inner join on neurons present in both snapshots
    # merged = pd.merge(before, after, on="neuron_id", how="inner")
    # merged["delta"] = merged["d_before"] - merged["d_after"]  # before - after
    # merged["k_diff"] = merged["k_before"] - merged["k_after"]

    # 1) ALL neurons
    # plot_per_neuron_delta(
    #     merged.dropna(subset=["delta"]),
    #     "figures/reward_center_shift.png",
    #     "Reward-center shift (all neurons)"
    # )

    # 2) Place cells (k=1) — here we require k=1 both before AND after.
    #    If you want “k=1 in either condition”, replace the filter with:
    #    place = merged[(merged["k_before"] == 1) | (merged["k_after"] == 1)]
    # place = merged[(merged["k_before"] == 1) & (merged["k_after"] == 1)]

    # lmplot_dist_to_reward_single(place)

    # plot_per_neuron_delta(
    #     place.dropna(subset=["delta"]),
    #     "figures/reward_center_placecells_shift.png",
    #     "Reward-center shift (place cells, k=1 in both conditions)"
    # )

    # If you also want quick numeric summaries:
    # print(
    #     f"Number of Positive Displacements in Single Bumps = {len(place[place['delta'] > 0])}"
    # )
    # print(
    #     f"Number of Non-Positive Displacements in Single Bumps = {len(place[place['delta'] <= 0])}"
    # )
    # print(
    #     "ALL — count:",
    #     merged["delta"].notna().sum(),
    #     " mean:",
    #     merged["delta"].mean(),
    #     " median:",
    #     merged["delta"].median(),
    # )
    # print(
    #     "PLACE(k=1, both) — count:",
    #     place["delta"].notna().sum(),
    #     " mean:",
    #     place["delta"].mean(),
    # )

    plot_skaggs_scores_before_after(
        scores_before, scores_after, save_path="../Results/Skaggs_Scores.png", sort=False
    )
    plot_skaggs_scores_before_after(
        scores_before,
        scores_after,
        save_path="../Results/Skaggs_Scores(Sorted).png",
        sort=True,
    )
    # k = 8  # how many to plot from each category (bottom,center,top) and from before and after
    # plot_bottom_center_top_k_skaggs_neurons_heatmaps(
    #     df_combined_before,
    #     scores_before,
    #     scores_after,
    #     save_path="figures/sorted_skaggs_bottom_center_top_{k}_heatmap_before.png",
    #     k=k,
    # )
    # plot_bottom_center_top_k_skaggs_neurons_heatmaps(
    #     df_combined_after,
    #     scores_before,
    #     scores_after,
    #     save_path="figures/sorted_skaggs_bottom_center_top_{k}_heatmap_after.png",
    #     k=k,
    # )


if __name__ == "__main__":
    main()
