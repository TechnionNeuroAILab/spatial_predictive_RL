import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from pc_common import append_stat, resolve_reward_pos, save_all, write_stats_report


def load_data(data_dir, period="after"):
    """Loads and aligns neural and behavioral data."""
    try:
        df_act = pd.read_pickle(os.path.join(data_dir, f"df_activations_{period}.pkl"))
        df_beh = pd.read_pickle(os.path.join(data_dir, f"df_behavior_{period}.pkl"))

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


def prepare_targets(y_beh, reward_pos):
    """Calculates the continuous targets for regression."""
    # 1. Distance Target
    dist = np.sqrt(
        (y_beh["x"] - reward_pos[0]) ** 2 + (y_beh["y"] - reward_pos[1]) ** 2
    )

    # 2. Gaze Alignment Target (Cosine of angle error)
    # Vector to reward
    vec_x = reward_pos[0] - y_beh["x"]
    vec_y = reward_pos[1] - y_beh["y"]
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
    return run_decoder_split(X_train, y_train, X_test, y_test, label)


def run_decoder_split(X_train, y_train, X_test, y_test, label):
    """Fits a Ridge decoder on an explicit train split and evaluates on an
    explicit test split (used by run_decoder, and by cross-decoding to reuse
    the exact same split across two different activity sources)."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return {
        "model": model,
        "r2": r2,
        "y_test": y_test,
        "y_pred": y_pred,
        "weights": model.coef_,
        "label": label,
    }


def run_cross_decoding(data_dir, period_a, period_b, reward_pos, test_size=0.2):
    """
    Cross-decoding between two periods that share the same behavioral
    trajectory (e.g. replay_c{N}_pred / replay_c{N}_postrl -- a frozen
    trajectory, so x/y/theta and hence the decoding targets are identical;
    only the neural activity differs).

    Trains a Ridge decoder on period_a's activity and tests it both on
    held-out period_a activity (within-state) and on period_b's activity at
    the SAME timesteps (cross-state), and vice versa. If the population code
    genuinely remapped between A and B, cross-state R^2 should drop much more
    than within-state R^2; if it's just noisier decoding of the same
    underlying code, within and cross should degrade similarly.
    """
    X_a, y_beh_a = load_data(data_dir, period=period_a)
    X_b, y_beh_b = load_data(data_dir, period=period_b)
    if X_a is None or X_b is None:
        return None, None

    common_idx = X_a.index.intersection(X_b.index)
    X_a = X_a.loc[common_idx]
    X_b = X_b.loc[common_idx]
    y_beh = y_beh_a.loc[common_idx]

    max_beh_diff = float((y_beh_a.loc[common_idx][["x", "y"]] - y_beh_b.loc[common_idx][["x", "y"]]).abs().to_numpy().max())
    if max_beh_diff > 1e-6:
        print(f"  NOTE: behavior differs between '{period_a}' and '{period_b}' (max |dx,dy|={max_beh_diff:.4g}); "
              f"this isn't a frozen-trajectory pair, using period_a's behavior as the shared target is an approximation.")

    target_dist, target_align = prepare_targets(y_beh, reward_pos)

    n = len(common_idx)
    idx_train, idx_test = train_test_split(np.arange(n), test_size=test_size, shuffle=False)

    def _split(X, y):
        return X.iloc[idx_train], X.iloc[idx_test], y.iloc[idx_train], y.iloc[idx_test]

    results = {}
    for target_name, target in [("dist", target_dist), ("align", target_align)]:
        Xa_train, Xa_test, y_train, y_test = _split(X_a, target)
        Xb_train, Xb_test, _, _ = _split(X_b, target)

        res_a = run_decoder_split(Xa_train, y_train, Xa_test, y_test, f"{target_name}_within_{period_a}")
        res_b = run_decoder_split(Xb_train, y_train, Xb_test, y_test, f"{target_name}_within_{period_b}")
        # Cross: reuse the model trained above, apply to the OTHER period's test activity.
        r2_a_to_b = r2_score(y_test, res_a["model"].predict(Xb_test))
        r2_b_to_a = r2_score(y_test, res_b["model"].predict(Xa_test))

        results[target_name] = {
            "within_a": res_a["r2"],
            "within_b": res_b["r2"],
            "cross_a_to_b": r2_a_to_b,
            "cross_b_to_a": r2_b_to_a,
        }
        print(f"  [{target_name}] within {period_a}={res_a['r2']:.4f}  within {period_b}={res_b['r2']:.4f}  "
              f"cross {period_a}->{period_b}={r2_a_to_b:.4f}  cross {period_b}->{period_a}={r2_b_to_a:.4f}")

    return results, n


def plot_decoder_results(results_dist, results_align, output):
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
    save_all(output)


def plot_cross_decoding_results(results, period_a, period_b, output):
    """Grouped bar chart: within-state vs cross-state R^2, for both targets."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    labels = [f"Within\n{period_a}", f"Within\n{period_b}", f"{period_a}\n↓\n{period_b}", f"{period_b}\n↓\n{period_a}"]
    colors = ["steelblue", "steelblue", "tomato", "tomato"]
    titles = {"dist": "Distance-to-reward decoding", "align": "Gaze-alignment decoding"}
    for ax, target_name in zip(axes, ["dist", "align"]):
        r = results[target_name]
        vals = [r["within_a"], r["within_b"], r["cross_a_to_b"], r["cross_b_to_a"]]
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("$R^2$")
        ax.set_title(titles[target_name])
    fig.text(0.5, 0.01, "cross bars: trained on top period, tested on bottom period (↓ = train→test)", ha="center", fontsize=8, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_all(output)


def parse_args():
    p = argparse.ArgumentParser(description="Linear decoder analysis for place-cell activity.")
    p.add_argument("--run_dir", type=str, default=".",
                    help="Run directory containing data/ and figures/ (e.g. Negative_Results).")
    p.add_argument("--reward", nargs=2, type=float, default=None,
                    help="Reward (x, y). Defaults to run_dir/config.json's reward_pos.")
    p.add_argument("--period", type=str, default="after",
                    help="v11 state token to decode from, e.g. after, replay_c1_pred, replay_c1_postrl.")
    p.add_argument("--period_b", type=str, default=None,
                    help="If given, run cross-decoding between --period (A) and --period_b (B) instead of "
                         "the single-period analysis: train on A/test on B and vice versa, compared against "
                         "within-state R^2. Intended for a frozen-trajectory pair like replay_c1_pred/replay_c1_postrl.")
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

    if args.period_b is not None:
        print(f"Cross-decoding between '{args.period}' and '{args.period_b}'...")
        results, n_common = run_cross_decoding(data_dir, args.period, args.period_b, reward_pos)
        if results is not None:
            pair_tag = f"{args.period}_vs_{args.period_b}"
            record = {"test": "cross_decoding_r2", "period_a": args.period, "period_b": args.period_b,
                      "n_common_timesteps": n_common}
            for target_name, r in results.items():
                for k, v in r.items():
                    record[f"{target_name}_{k}"] = float(v)
            append_stat(run_dir, "analyze_activty_lineardecoder.py", record)
            report_path = write_stats_report(run_dir)
            print(f"Wrote statistics report: {report_path}")
            plot_cross_decoding_results(results, args.period, args.period_b,
                                         os.path.join(fig_dir, f"linear_decoder_cross_{pair_tag}"))
        sys.exit(0)

    print(f"Decoding from period: {args.period}")

    X, y_beh = load_data(data_dir, period=args.period)

    if X is not None:
        print(f"Data prepared. X shape: {X.shape}")
        neuron_ids = X.columns.values

        # 1. Prepare Targets
        target_dist, target_align = prepare_targets(y_beh, reward_pos)

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
        append_stat(run_dir, "analyze_activty_lineardecoder.py", {
            "test": "ridge_linear_decoder_r2",
            "period": args.period,
            "distance_r2": float(res_dist["r2"]),
            "gaze_alignment_r2": float(res_align["r2"]),
        })
        report_path = write_stats_report(run_dir)
        print(f"Wrote statistics report: {report_path}")

        # 4. Visualize
        plot_decoder_results(res_dist, res_align, os.path.join(fig_dir, f"linear_decoder_fig_{args.period}"))
