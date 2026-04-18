from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from common import ensure_dir, load_behavior_activity, pivot_activity_matrix, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selectivity shuffle controls via label permutations.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/selectivity_shuffle_control")
    parser.add_argument("--spatial-grid", type=int, default=16)
    parser.add_argument("--gaze-bins", type=int, default=12)
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--max-neurons", type=int, default=100)
    parser.add_argument("--subset-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def bin_spatial(x: np.ndarray, y: np.ndarray, grid_size: int) -> np.ndarray:
    xb = np.clip(np.floor(x * grid_size).astype(int), 0, grid_size - 1)
    yb = np.clip(np.floor(y * grid_size).astype(int), 0, grid_size - 1)
    return xb * grid_size + yb


def bin_gaze(theta: np.ndarray, n_bins: int) -> np.ndarray:
    wrapped = (theta + np.pi) % (2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bins = np.digitize(wrapped, edges, right=False) - 1
    return np.clip(bins, 0, n_bins - 1)


def compute_selectivity(activity: np.ndarray, labels: np.ndarray, n_labels: int, eps: float = 1e-9) -> np.ndarray:
    n_t, n_neurons = activity.shape
    si = np.zeros(n_neurons, dtype=float)

    counts = np.bincount(labels, minlength=n_labels).astype(float)
    valid_bins = counts > 0

    for n in range(n_neurons):
        sums = np.bincount(labels, weights=activity[:, n], minlength=n_labels)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

        if np.sum(valid_bins) < 2:
            si[n] = 0.0
            continue

        pref_idx = int(np.argmax(means))
        pref_val = float(means[pref_idx])

        others_mask = np.ones(n_labels, dtype=bool)
        others_mask[pref_idx] = False
        others_mask &= valid_bins

        if np.sum(others_mask) == 0:
            rest_val = 0.0
        else:
            rest_val = float(np.mean(means[others_mask]))

        si[n] = (pref_val - rest_val) / (pref_val + rest_val + eps)

    return si


def run_shuffle(
    activity: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    n_shuffles: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    real = compute_selectivity(activity, labels, n_labels)
    shuffled = np.zeros((n_shuffles, activity.shape[1]), dtype=float)

    for i in range(n_shuffles):
        perm_labels = rng.permutation(labels)
        shuffled[i] = compute_selectivity(activity, perm_labels, n_labels)

    null_mean = np.mean(shuffled, axis=0)
    null_std = np.std(shuffled, axis=0, ddof=1)
    z = np.divide(real - null_mean, null_std, out=np.zeros_like(real), where=~np.isclose(null_std, 0.0))
    return real, shuffled, null_mean, z


def load_period_arrays(data_dir: Path, period: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    behavior, activity, _ = load_behavior_activity(data_dir, period)
    x_df, b_df = pivot_activity_matrix(activity, behavior)
    if b_df is None:
        raise RuntimeError("Behavior alignment failed.")

    # theta might not always exist in behavior. If missing, fill zeros.
    b_ext = behavior.groupby("timestep").first().sort_index()
    theta = b_ext["theta"].to_numpy(dtype=float) if "theta" in b_ext.columns else np.zeros(b_ext.shape[0], dtype=float)

    common = x_df.index.intersection(b_ext.index)
    x_df = x_df.loc[common]
    b_ext = b_ext.loc[common]

    x = b_ext["x"].to_numpy(dtype=float)
    y = b_ext["y"].to_numpy(dtype=float)
    theta = b_ext["theta"].to_numpy(dtype=float) if "theta" in b_ext.columns else np.zeros_like(x)
    activity_mat = x_df.to_numpy(dtype=float)

    return x, y, theta, activity_mat


def choose_neuron_subset(activity: np.ndarray, max_neurons: int, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_total = activity.shape[1]
    n_use = min(max_neurons, n_total)
    rng = np.random.default_rng(seed)

    if mode == "random" and n_use < n_total:
        idx = np.sort(rng.choice(np.arange(n_total), size=n_use, replace=False))
    else:
        idx = np.arange(n_use)
    return activity[:, idx], idx


def plot_hist(null_vals: np.ndarray, real_vals: np.ndarray, title: str, out_path: Path, color: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(null_vals, bins=60, density=True, alpha=0.55, color="gray", label="Null")
    plt.hist(real_vals, bins=35, density=True, alpha=0.6, color=color, label="Real")
    p95 = float(np.percentile(null_vals, 95))
    plt.axvline(p95, color="black", linestyle="--", linewidth=1.6, label="Null 95th percentile")
    plt.xlabel("Selectivity index")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def summarize_period(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    activity: np.ndarray,
    spatial_grid: int,
    gaze_bins: int,
    n_shuffles: int,
    seed: int,
) -> dict[str, Any]:
    spatial_labels = bin_spatial(x, y, spatial_grid)
    gaze_labels = bin_gaze(theta, gaze_bins)

    real_sp, sh_sp, null_sp_mean, z_sp = run_shuffle(
        activity=activity,
        labels=spatial_labels,
        n_labels=spatial_grid * spatial_grid,
        n_shuffles=n_shuffles,
        seed=seed,
    )
    real_gz, sh_gz, null_gz_mean, z_gz = run_shuffle(
        activity=activity,
        labels=gaze_labels,
        n_labels=gaze_bins,
        n_shuffles=n_shuffles,
        seed=seed + 1,
    )

    return {
        "spatial": {
            "real": real_sp,
            "shuffled": sh_sp,
            "null_mean_per_neuron": null_sp_mean,
            "z_scores": z_sp,
            "fraction_significant": float(np.mean(z_sp > 1.96)),
        },
        "gaze": {
            "real": real_gz,
            "shuffled": sh_gz,
            "null_mean_per_neuron": null_gz_mean,
            "z_scores": z_gz,
            "fraction_significant": float(np.mean(z_gz > 1.96)),
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    data_dir = Path(args.data_dir)
    x_b, y_b, th_b, act_b_full = load_period_arrays(data_dir, "before")
    x_a, y_a, th_a, act_a_full = load_period_arrays(data_dir, "after")
    n_use = min(args.max_neurons, act_b_full.shape[1], act_a_full.shape[1])
    act_b, idx_b = choose_neuron_subset(act_b_full, n_use, args.subset_mode, seed=args.seed)
    act_a, idx_a = choose_neuron_subset(act_a_full, n_use, args.subset_mode, seed=args.seed)
    res_before = summarize_period(
        x=x_b,
        y=y_b,
        theta=th_b,
        activity=act_b,
        spatial_grid=args.spatial_grid,
        gaze_bins=args.gaze_bins,
        n_shuffles=args.n_shuffles,
        seed=args.seed,
    )
    res_after = summarize_period(
        x=x_a,
        y=y_a,
        theta=th_a,
        activity=act_a,
        spatial_grid=args.spatial_grid,
        gaze_bins=args.gaze_bins,
        n_shuffles=args.n_shuffles,
        seed=args.seed + 11,
    )
    plot_hist(
        null_vals=res_before["spatial"]["shuffled"].reshape(-1),
        real_vals=res_before["spatial"]["real"],
        title="Spatial selectivity shuffle - Before RL",
        out_path=out_dir / "spatial_selectivity_shuffle_before.png",
        color="#2E5FA3",
    )
    plot_hist(
        null_vals=res_after["spatial"]["shuffled"].reshape(-1),
        real_vals=res_after["spatial"]["real"],
        title="Spatial selectivity shuffle - After RL",
        out_path=out_dir / "spatial_selectivity_shuffle_after.png",
        color="#E07A2D",
    )
    plot_hist(
        null_vals=res_before["gaze"]["shuffled"].reshape(-1),
        real_vals=res_before["gaze"]["real"],
        title="Gaze selectivity shuffle - Before RL",
        out_path=out_dir / "gaze_selectivity_shuffle_before.png",
        color="#2E5FA3",
    )
    plot_hist(
        null_vals=res_after["gaze"]["shuffled"].reshape(-1),
        real_vals=res_after["gaze"]["real"],
        title="Gaze selectivity shuffle - After RL",
        out_path=out_dir / "gaze_selectivity_shuffle_after.png",
        color="#E07A2D",
    )

    payload: dict[str, Any] = {
        "script": "selectivity_shuffle_control.py",
        "seed": int(args.seed),
        "n_shuffles": int(args.n_shuffles),
        "n_neurons_used": int(n_use),
        "neuron_indices_before": idx_b.tolist(),
        "neuron_indices_after": idx_a.tolist(),
        "fraction_significant_spatial_before": res_before["spatial"]["fraction_significant"],
        "fraction_significant_spatial_after": res_after["spatial"]["fraction_significant"],
        "fraction_significant_gaze_before": res_before["gaze"]["fraction_significant"],
        "fraction_significant_gaze_after": res_after["gaze"]["fraction_significant"],
        "before": {
            "spatial": {
                "real_selectivity": res_before["spatial"]["real"].tolist(),
                "null_mean_per_neuron": res_before["spatial"]["null_mean_per_neuron"].tolist(),
                "z_scores": res_before["spatial"]["z_scores"].tolist(),
            },
            "gaze": {
                "real_selectivity": res_before["gaze"]["real"].tolist(),
                "null_mean_per_neuron": res_before["gaze"]["null_mean_per_neuron"].tolist(),
                "z_scores": res_before["gaze"]["z_scores"].tolist(),
            },
        },
        "after": {
            "spatial": {
                "real_selectivity": res_after["spatial"]["real"].tolist(),
                "null_mean_per_neuron": res_after["spatial"]["null_mean_per_neuron"].tolist(),
                "z_scores": res_after["spatial"]["z_scores"].tolist(),
            },
            "gaze": {
                "real_selectivity": res_after["gaze"]["real"].tolist(),
                "null_mean_per_neuron": res_after["gaze"]["null_mean_per_neuron"].tolist(),
                "z_scores": res_after["gaze"]["z_scores"].tolist(),
            },
        },
    }

    save_json(out_dir / "selectivity_shuffle_results.json", payload)
    print(f"Saved selectivity shuffle outputs to: {out_dir}")


if __name__ == "__main__":
    main()
