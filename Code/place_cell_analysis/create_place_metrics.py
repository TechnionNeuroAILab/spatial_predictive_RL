import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pc_common import ensure_skaggs, filter_place_cells, save_all


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
    save_all(save_path)
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
    save_all(save_path)
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



def _build_rate_map(df_neuron, m=32, smooth_sigma=1.5):
    """
    Bin a neuron's activity onto an m x m grid, smooth with a Gaussian, and
    normalise to [0, 1] for visualisation. Same recipe as visualize_place_fields
    in ratinabox_v9.py so the figures are directly comparable.
    """
    grid = np.zeros((m, m), dtype=np.float32)
    counts = np.zeros((m, m), dtype=np.float32)
    x = np.clip((df_neuron["x"].to_numpy() * m).astype(int), 0, m - 1)
    y = np.clip((df_neuron["y"].to_numpy() * m).astype(int), 0, m - 1)
    a = df_neuron["activity"].to_numpy(dtype=np.float32)
    np.add.at(grid, (y, x), a)
    np.add.at(counts, (y, x), 1)
    with np.errstate(invalid="ignore"):
        rate = np.where(counts > 0, grid / counts, 0.0)
    smoothed = gaussian_filter(rate, sigma=smooth_sigma)
    if smoothed.max() > 0:
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    return smoothed


def plot_top_skaggs_cells(
    df_combined, df_skaggs, save_path, k=25, m=32, smooth_sigma=1.5
):
    """
    Plot rate-map heatmaps for the top-k neurons ranked by Skaggs score.
    Layout is roughly square. Each subplot title shows neuron_id and Skaggs.
    """
    df_s = df_skaggs.dropna(subset=["skaggs_score"]).sort_values(
        "skaggs_score", ascending=False
    )
    top = df_s.head(k)
    top_ids = top["neuron_id"].astype(int).tolist()
    top_scores = top["skaggs_score"].astype(float).tolist()

    n_cols = int(np.ceil(np.sqrt(k)))
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, (nid, score) in enumerate(zip(top_ids, top_scores)):
        ax = axes[i]
        df_neuron = df_combined[df_combined["neuron_id"] == nid]
        if len(df_neuron) > 0:
            rmap = _build_rate_map(df_neuron, m=m, smooth_sigma=smooth_sigma)
            ax.imshow(
                rmap, origin="lower", cmap="jet",
                interpolation="nearest", vmin=0, vmax=1, aspect="equal",
            )
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
        ax.set_title(f"#{nid}  {score:.2f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(len(top_ids), len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.25)
    plt.tight_layout()
    save_all(save_path, dpi=150)
    plt.close(fig)


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
    save_all(out_path, dpi=150)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Skaggs scores and top place-cell figures.")
    p.add_argument("--run_dir", type=str, default=".",
                    help="Run directory containing data/ and figures/ (e.g. Negative_Results).")
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
    state_before, state_after = args.state_before, args.state_after
    pair_tag = f"{state_before}_vs_{state_after}"

    # Load
    df_behavior_before = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{state_before}.pkl"))
    df_activity_before = pd.read_pickle(os.path.join(data_dir, f"df_activations_{state_before}.pkl"))
    df_combined_before = df_behavior_before.merge(
        df_activity_before, on="timestep", how="inner"
    )
    df_behavior_after = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{state_after}.pkl"))
    df_activity_after = pd.read_pickle(os.path.join(data_dir, f"df_activations_{state_after}.pkl"))
    df_combined_after = df_behavior_after.merge(
        df_activity_after, on="timestep", how="inner"
    )
    skaggs_before_path = ensure_skaggs(run_dir, state_before)
    skaggs_after_path = ensure_skaggs(run_dir, state_after)

    df_skaggs_before = pd.read_pickle(skaggs_before_path)
    df_skaggs_after = pd.read_pickle(skaggs_after_path)

    scores_before = df_skaggs_before["skaggs_score"].to_numpy()
    scores_after = df_skaggs_after["skaggs_score"].to_numpy()

    # Canonical place-cell filter shared across all place-cell scripts (see
    # pc_common.py), so this script's population matches
    # recuritment_place_cells.py's and shift_toward_reward.py's.
    df_centers_after = pd.read_pickle(os.path.join(data_dir, f"df_centers_{state_after}.pkl"))
    place_cells_after = filter_place_cells(df_skaggs_after, df_centers_after)
    print(f"Place cells in '{state_after}': {len(place_cells_after)} / {len(df_skaggs_after)}")
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
        scores_before, scores_after, save_path=os.path.join(fig_dir, f"Skaggs_Scores_{pair_tag}"), sort=False
    )
    plot_skaggs_scores_before_after(
        scores_before,
        scores_after,
        save_path=os.path.join(fig_dir, f"Skaggs_Scores_Sorted_{pair_tag}"),
        sort=True,
    )

    # Visualise the highest-Skaggs cells as rate maps so we can eyeball whether
    # they actually look like place fields (single blob) or are sparse noise.
    plot_top_skaggs_cells(
        df_combined_before, df_skaggs_before,
        save_path=os.path.join(fig_dir, f"top_skaggs_cells_{state_before}"), k=25,
    )
    plot_top_skaggs_cells(
        df_combined_after, df_skaggs_after,
        save_path=os.path.join(fig_dir, f"top_skaggs_cells_{state_after}"), k=25,
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
