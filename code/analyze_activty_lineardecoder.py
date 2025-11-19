import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- Configuration ---
DATA_DIR = "data"
REWARD_POS = np.array([0.5, 0.5])


def load_data():
    """Loads and aligns neural and behavioral data."""
    try:
        df_act = pd.read_pickle(os.path.join(DATA_DIR, "df_activations_after.pkl"))
        df_beh = pd.read_pickle(os.path.join(DATA_DIR, "df_behavior_after.pkl"))

        # Merge on timestep
        df = pd.merge(df_act, df_beh, on="timestep")

        # Pivot to create Matrix X (rows=timesteps, cols=neurons)
        # We rely on the fact that 'activity' is the value we want
        X_df = df.pivot(index="timestep", columns="neuron_id", values="activity")

        # We need behavior aligned to these timesteps.
        # Since pivot sorts by index (timestep), we ensure behavior is sorted too.
        # We aggregate behavior by timestep (should be unique per timestep anyway)
        y_df = df.groupby("timestep")[["x", "y", "theta"]].first()

        # Ensure indices match perfectly
        common_indices = X_df.index.intersection(y_df.index)
        X = X_df.loc[common_indices]
        y_beh = y_df.loc[common_indices]

        return X, y_beh
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None


def prepare_targets(y_beh):
    """Calculates the continuous targets for regression."""
    # 1. Distance Target
    dist = np.sqrt(
        (y_beh["x"] - REWARD_POS[0]) ** 2 + (y_beh["y"] - REWARD_POS[1]) ** 2
    )

    # 2. Gaze Alignment Target (Cosine of angle error)
    # Vector to reward
    vec_x = REWARD_POS[0] - y_beh["x"]
    vec_y = REWARD_POS[1] - y_beh["y"]
    target_angle = np.arctan2(vec_y, vec_x)

    # Angle error
    angle_diff = y_beh["theta"] - target_angle
    # Cosine Alignment: 1 = looking at reward, -1 = looking away
    alignment = np.cos(angle_diff)

    return dist, alignment


def run_decoder(X, y, label):
    """
    Fits a Linear Decoder, calculates R2, and extracts weights.
    """
    # Split data (80% train, 20% test) to test generalization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Initialize Linear Model (Ridge is standard for neural data)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predict (Project the data)
    y_pred = model.predict(X_test)

    # Evaluate
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "r2": r2,
        "y_test": y_test,
        "y_pred": y_pred,
        "weights": model.coef_,
        "label": label,
    }


def plot_decoder_results(results_dist, results_align):
    """Plots the projection performance and feature importance."""
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1)

    # --- 1. Distance Decoding Performance ---
    ax1 = fig.add_subplot(gs[0])
    # Scatter plot of Truth vs Prediction
    sns.scatterplot(
        x=results_dist["y_test"],
        y=results_dist["y_pred"],
        ax=ax1,
        alpha=0.5,
        color="teal",
        s=10,
    )

    # Perfect fit line
    min_val, max_val = min(results_dist["y_test"]), max(results_dist["y_test"])
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

    ax1.set_title(f"Decoding Distance to Reward\n$R^2 = {results_dist['r2']:.3f}$")
    ax1.set_xlabel("True Distance")
    ax1.set_ylabel("Predicted Distance (Neural Projection)")

    # --- 2. Gaze Alignment Decoding Performance ---
    ax2 = fig.add_subplot(gs[1])
    sns.scatterplot(
        x=results_align["y_test"],
        y=results_align["y_pred"],
        ax=ax2,
        alpha=0.5,
        color="coral",
        s=10,
    )

    ax2.plot([-1, 1], [-1, 1], "k--", lw=2)
    ax2.set_title(
        f"Decoding Gaze Alignment (Cosine)\n$R^2 = {results_align['r2']:.3f}$"
    )
    ax2.set_xlabel("True Alignment\n(1=Looking At, -1=Looking Away)")
    ax2.set_ylabel("Predicted Alignment")

    plt.tight_layout()
    plt.savefig("figures/linear_decoder_fig.png")
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    X, y_beh = load_data()

    if X is not None:
        print(f"Data prepared. X shape: {X.shape}")
        neuron_ids = X.columns.values

        # 1. Prepare Targets
        target_dist, target_align = prepare_targets(y_beh)

        # 2. Run Decoders
        print("Training Distance Decoder...")
        res_dist = run_decoder(X, target_dist, "Distance")

        print("Training Gaze Alignment Decoder...")
        res_align = run_decoder(X, target_align, "Gaze Alignment")

        # 3. Output Statistics
        print("-" * 30)
        print(f"Distance Decoding R2: {res_dist['r2']:.4f}")
        print(f"Gaze Alignment R2:  {res_align['r2']:.4f}")
        print("-" * 30)

        # 4. Visualize
        plot_decoder_results(res_dist, res_align)
