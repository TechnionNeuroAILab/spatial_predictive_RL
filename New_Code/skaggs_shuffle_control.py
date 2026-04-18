from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from common import ensure_dir, load_behavior_activity, pivot_activity_matrix, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skaggs shuffle control via circular position shifts.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/skaggs_shuffle_control")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--max-neurons", type=int, default=100)
    parser.add_argument("--subset-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_skaggs_scores(flat_bins: np.ndarray, activity: np.ndarray, grid_size: int) -> np.ndarray:
    k_bins = grid_size * grid_size
    occ = np.bincount(flat_bins, minlength=k_bins).astype(float)
    p = occ / max(occ.sum(), 1.0)

    scores = np.zeros(activity.shape[1], dtype=float)
    for n in range(activity.shape[1]):
        a = activity[:, n]
        lam_bar = float(np.mean(a))
        if lam_bar <= 0:
            scores[n] = 0.0
            continue

        sum_a = np.bincount(flat_bins, weights=a, minlength=k_bins)
        lam_x = np.divide(sum_a, occ, out=np.zeros_like(sum_a), where=occ > 0)

        ratio = np.divide(lam_x, lam_bar, out=np.zeros_like(lam_x), where=lam_x > 0)
        mask = (ratio > 0) & (p > 0)

        info = np.zeros_like(ratio)
        info[mask] = ratio[mask] * np.log2(ratio[mask]) * p[mask]
        scores[n] = float(np.sum(info))

    return scores


def load_period_activity_matrix(data_dir: Path, period: str) -> tuple[np.ndarray, np.ndarray]:
    behavior, activity, _ = load_behavior_activity(data_dir, period)
    x_df, b_df = pivot_activity_matrix(activity, behavior)
    if b_df is None:
        raise RuntimeError("Behavior alignment failed.")
    positions = b_df[["x", "y"]].to_numpy(dtype=float)
    activity_mat = x_df.to_numpy(dtype=float)

    n = min(positions.shape[0], activity_mat.shape[0])
    return positions[:n], activity_mat[:n]


def bin_positions(positions: np.ndarray, grid_size: int) -> np.ndarray:
    x = np.clip(np.floor(positions[:, 0] * grid_size).astype(int), 0, grid_size - 1)
    y = np.clip(np.floor(positions[:, 1] * grid_size).astype(int), 0, grid_size - 1)
    return x * grid_size + y


def run_one_period(
    positions: np.ndarray,
    activity: np.ndarray,
    grid_size: int,
    n_shuffles: int,
    max_neurons: int,
    subset_mode: str,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_total = activity.shape[1]
    n_use = min(max_neurons, n_total)
    if subset_mode == "random" and n_use < n_total:
        use_idx = np.sort(rng.choice(np.arange(n_total), size=n_use, replace=False))
    else:
        use_idx = np.arange(n_use)

    act_use = activity[:, use_idx]
    flat_real = bin_positions(positions, grid_size)
    real_scores = compute_skaggs_scores(flat_real, act_use, grid_size)

    t = positions.shape[0]
    min_shift = max(1, int(0.1 * t))
    max_shift = max(min_shift + 1, int(0.9 * t))

    shuffled = np.zeros((n_shuffles, n_use), dtype=float)
    for i in range(n_shuffles):
        shift = int(rng.integers(min_shift, max_shift))
        flat_shifted = np.roll(flat_real, shift)
        shuffled[i] = compute_skaggs_scores(flat_shifted, act_use, grid_size)

    null = shuffled.reshape(-1)
    null_mean = float(np.mean(null))
    null_std = float(np.std(null, ddof=1))
    null_95 = float(np.percentile(null, 95))

    if np.isclose(null_std, 0.0):
        z = np.zeros_like(real_scores)
    else:
        z = (real_scores - null_mean) / null_std

    frac_sig = float(np.mean(z > 1.96))
    n_sig = int(np.sum(z > 1.96))
    ci_low_count, ci_high_count = binom.interval(0.95, n_use, frac_sig)
    frac_ci_low = float(ci_low_count / n_use)
    frac_ci_high = float(ci_high_count / n_use)

    return {
        "n_neurons_total": int(n_total),
        "n_neurons_used": int(n_use),
        "neuron_indices_used": use_idx.tolist(),
        "real_scores": real_scores.tolist(),
        "z_scores": z.tolist(),
        "null_mean": null_mean,
        "null_std": null_std,
        "null_95th_percentile": null_95,
        "fraction_significant": frac_sig,
        "fraction_significant_ci_low": frac_ci_low,
        "fraction_significant_ci_high": frac_ci_high,
        "n_significant": n_sig,
        "null_distribution": null,
    }


def plot_overlay(null_values: np.ndarray, real_scores: np.ndarray, null_95: float, title: str, out_path: Path, color: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(null_values, bins=60, density=True, alpha=0.55, color="gray", label="Null (circular shift)")
    plt.hist(real_scores, bins=35, density=True, alpha=0.6, color=color, label="Real")
    plt.axvline(null_95, color="black", linestyle="--", linewidth=1.8, label="Null 95th percentile")
    plt.xlabel("Skaggs information")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    data_dir = Path(args.data_dir)

    pos_b, act_b = load_period_activity_matrix(data_dir, "before")
    pos_a, act_a = load_period_activity_matrix(data_dir, "after")

    res_b = run_one_period(
        positions=pos_b,
        activity=act_b,
        grid_size=args.grid_size,
        n_shuffles=args.n_shuffles,
        max_neurons=args.max_neurons,
        subset_mode=args.subset_mode,
        seed=args.seed,
    )
    res_a = run_one_period(
        positions=pos_a,
        activity=act_a,
        grid_size=args.grid_size,
        n_shuffles=args.n_shuffles,
        max_neurons=args.max_neurons,
        subset_mode=args.subset_mode,
        seed=args.seed + 1,
    )

    null_b = np.asarray(res_b.pop("null_distribution"), dtype=float)
    null_a = np.asarray(res_a.pop("null_distribution"), dtype=float)
    np.save(out_dir / "skaggs_null_distribution_before.npy", null_b)
    np.save(out_dir / "skaggs_null_distribution_after.npy", null_a)
    np.save(out_dir / "skaggs_null_distribution.npy", np.concatenate([null_b, null_a]))

    plot_overlay(
        null_values=null_b,
        real_scores=np.asarray(res_b["real_scores"], dtype=float),
        null_95=res_b["null_95th_percentile"],
        title="Skaggs shuffle control - Before RL",
        out_path=out_dir / "skaggs_shuffle_before.png",
        color="#2E5FA3",
    )
    plot_overlay(
        null_values=null_a,
        real_scores=np.asarray(res_a["real_scores"], dtype=float),
        null_95=res_a["null_95th_percentile"],
        title="Skaggs shuffle control - After RL",
        out_path=out_dir / "skaggs_shuffle_after.png",
        color="#E07A2D",
    )

    payload = {
        "script": "skaggs_shuffle_control.py",
        "seed": int(args.seed),
        "n_shuffles": int(args.n_shuffles),
        "grid_size": int(args.grid_size),
        "before": res_b,
        "after": res_a,
        "fraction_significant_before": res_b["fraction_significant"],
        "fraction_significant_after": res_a["fraction_significant"],
    }
    save_json(out_dir / "skaggs_shuffle_results.json", payload)
    print(f"Saved Skaggs shuffle outputs to: {out_dir}")


if __name__ == "__main__":
    main()
