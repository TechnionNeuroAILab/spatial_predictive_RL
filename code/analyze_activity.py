import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.distance import cdist

# --- Configuration ---
DATA_DIR = "data"
REWARD_POS = np.array([0.5, 0.5])
D_R = 0.05
LOOKING_THRESHOLD_RAD = np.deg2rad(5)
EPSILON = 1e-9


def load_and_merge_data(period="after"):
    """
    Loads pickles for a specific period (before/after) 
    and merges behavior with neural data.
    """
    act_file = f"df_activations_{period}.pkl"
    beh_file = f"df_behavior_{period}.pkl"
    
    try:
        df_act = pd.read_pickle(os.path.join(DATA_DIR, act_file))
        df_beh = pd.read_pickle(os.path.join(DATA_DIR, beh_file))
    except FileNotFoundError as e:
        print(f"Could not load {period} files: {e}")
        return None

    df_merged = pd.merge(df_act, df_beh, on="timestep")
    return df_merged


def calculate_spatial_metrics(df):
    """Adds distance to reward and determines 'Near' status."""
    df["dist_to_reward"] = np.sqrt(
        (df["x"] - REWARD_POS[0]) ** 2 + (df["y"] - REWARD_POS[1]) ** 2
    )
    df["is_near_reward"] = df["dist_to_reward"] < D_R
    return df


def calculate_angular_metrics(df):
    """Calculates if the agent is looking at the reward."""
    vec_y = REWARD_POS[1] - df["y"]
    vec_x = REWARD_POS[0] - df["x"]
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

def calculate_place_field_shift():
    """
    Loads center data for before/after and calculates 
    the Euclidean shift for each neuron.
    """
    try:
        df_before = pd.read_pickle(os.path.join(DATA_DIR, "df_centers_before.pkl"))
        df_after = pd.read_pickle(os.path.join(DATA_DIR, "df_centers_after.pkl"))
    except FileNotFoundError:
        print("Center files not found. Skipping shift calculation.")
        return None

    # We only need relevant columns
    df_b = df_before[['neuron_id', 'k', 'center']].set_index('neuron_id')
    df_a = df_after[['neuron_id', 'k', 'center']].set_index('neuron_id')

    # Find intersection of neurons that exist in both
    common_neurons = df_b.index.intersection(df_a.index)
    
    shifts = []
    neuron_ids = []

    for nid in common_neurons:
        centers_b = df_b.loc[nid, 'center']
        centers_a = df_a.loc[nid, 'center']

        # Helper to safely convert center data to (N, 2) array
        def get_valid_points(centers):
            if centers is None: 
                return None
            
            # Handle single numpy array case (e.g. array([0.5, 0.5]))
            if isinstance(centers, np.ndarray):
                # If it's flat (2,), reshape to (1, 2)
                if centers.ndim == 1 and centers.size == 2:
                    return centers.reshape(1, 2)
                # If it's (N, 2), return as is
                if centers.ndim == 2 and centers.shape[1] == 2:
                    return centers
            
            # Handle lists
            if isinstance(centers, list):
                if len(centers) == 0: 
                    return None
                # Check if it's a flat list [0.5, 0.5]
                if len(centers) == 2 and all(isinstance(x, (int, float)) for x in centers):
                    return np.array([centers]) # Make it (1, 2)
                
            # General fallback: try to convert and check shape
            try:
                pts = np.array([c for c in centers if c is not None])
                if pts.size == 0: return None
                
                # If we ended up with (2,) -> reshape to (1, 2)
                if pts.ndim == 1 and pts.size == 2:
                    pts = pts.reshape(1, 2)
                
                # Final check: Must be (N, 2)
                if pts.ndim == 2 and pts.shape[1] == 2:
                    return pts
            except:
                pass
            
            return None

        pts_b = get_valid_points(centers_b)
        pts_a = get_valid_points(centers_a)

        # Skip if data is missing or invalid for this neuron
        if pts_b is None or pts_a is None:
            continue

        # Calculate pairwise distances
        try:
            dists = cdist(pts_b, pts_a, metric='euclidean')
            min_dist = np.min(dists)
            shifts.append(min_dist)
            neuron_ids.append(nid)
        except ValueError:
            # Skip neurons that still cause shape issues
            continue

    df_shift = pd.DataFrame({
        'neuron_id': neuron_ids,
        'pf_shift': shifts
    })
    
    return df_shift


def plot_correlation(spatial_si, angular_si, shift_df):
    """
    Plots the correlation between SI (After) and Place Field Shift.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Merge Data
    # We look at SI from the 'After' period vs the Shift that happened
    df_sp = pd.merge(spatial_si, shift_df, on="neuron_id")
    df_ang = pd.merge(angular_si, shift_df, on="neuron_id")
    
    # Helper for correlation text
    def get_corr_text(x, y):
        if len(x) < 2: return "N/A"
        r = np.corrcoef(x, y)[0, 1]
        return f"r = {r:.2f}"

    # --- Plot 1: Spatial SI vs Shift ---
    sns.regplot(
        data=df_sp, x="selectivity_index", y="pf_shift", 
        ax=axes[0], scatter_kws={'s': 15, 'alpha': 0.6}, line_kws={'color': 'red'}
    )
    
    corr_sp = get_corr_text(df_sp["selectivity_index"], df_sp["pf_shift"])
    axes[0].set_title(f"Spatial Selectivity vs. PF Shift\nCorrelation: {corr_sp}", fontsize=12)
    axes[0].set_xlabel("Spatial Selectivity Index (After)")
    axes[0].set_ylabel("Place Field Shift (Euclidean Dist)")
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)

    # --- Plot 2: Angular SI vs Shift ---
    sns.regplot(
        data=df_ang, x="selectivity_index", y="pf_shift", 
        ax=axes[1], scatter_kws={'s': 15, 'alpha': 0.6}, line_kws={'color': 'red'}
    )

    corr_ang = get_corr_text(df_ang["selectivity_index"], df_ang["pf_shift"])
    axes[1].set_title(f"Gaze Selectivity vs. PF Shift\nCorrelation: {corr_ang}", fontsize=12)
    axes[1].set_xlabel("Gaze Selectivity Index (After)")
    axes[1].set_ylabel("Place Field Shift (Euclidean Dist)")
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)

    plt.suptitle("Do Reward Neurons Move More?", fontsize=16)
    plt.tight_layout()
    plt.savefig("figures/Correlation_Selectivity_Shift.png")
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Process "After" Data for Selectivity
    print("Loading 'After' data for selectivity...")
    df_after = load_and_merge_data(period="after")

    if df_after is not None:
        # Process metrics
        df_after = calculate_spatial_metrics(df_after)
        df_after = calculate_angular_metrics(df_after)

        # Compute SI
        print("Computing Selectivity Indices...")
        spatial_stats = compute_selectivity(df_after, condition_col="is_near_reward")
        angular_stats = compute_selectivity(df_after, condition_col="is_looking_at_reward")
        
        # Make index a column for merging
        spatial_stats = spatial_stats.reset_index()
        angular_stats = angular_stats.reset_index()

        # 2. Calculate Shift (Before -> After)
        print("Calculating Place Field Shifts...")
        df_shift = calculate_place_field_shift()

        if df_shift is not None and not df_shift.empty:
            print(f"Shift calculated for {len(df_shift)} neurons.")
            
            # 3. Plot Correlations
            plot_correlation(spatial_stats, angular_stats, df_shift)
            
            # 4. Simple stats
            
            # --- Spatial SI Stats ---
            high_sp_si = spatial_stats[spatial_stats['selectivity_index'] > 0.5]['neuron_id']
            shifts_high_sp_si = df_shift[df_shift['neuron_id'].isin(high_sp_si)]['pf_shift']
            print(f"\nAvg Shift for High Spatial SI (>0.5) neurons ({len(high_sp_si)} neurons): {shifts_high_sp_si.mean():.4f}")

            # --- Angular SI Stats ---
            high_ang_si = angular_stats[angular_stats['selectivity_index'] > 0.5]['neuron_id']
            shifts_high_ang_si = df_shift[df_shift['neuron_id'].isin(high_ang_si)]['pf_shift']
            print(f"Avg Shift for High Angular SI (>0.5) neurons ({len(high_ang_si)} neurons): {shifts_high_ang_si.mean():.4f}")

            # --- Overall Stats ---
            print(f"Avg Shift for all neurons ({len(df_shift)} neurons): {df_shift['pf_shift'].mean():.4f}")

        else:
            print("Could not calculate shifts (missing files or empty overlap).")