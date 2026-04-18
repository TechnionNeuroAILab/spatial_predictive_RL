from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, wilcoxon
from common import DEFAULT_REWARD, ensure_dir, extract_single_centers, load_pickle, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched and stratified shift shuffles toward reward.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/shift_shuffle_control")
    parser.add_argument("--centers-before", type=str, default="df_centers_before.pkl")
    parser.add_argument("--centers-after", type=str, default="df_centers_after.pkl")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--distance-bins", type=int, default=5)
    parser.add_argument("--reward-x", type=float, default=float(DEFAULT_REWARD[0]))
    parser.add_argument("--reward-y", type=float, default=float(DEFAULT_REWARD[1]))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def prepare_shift_data(data_dir: Path, centers_before: str, centers_after: str, reward_xy: np.ndarray) -> dict[str, np.ndarray]:
    df_b = load_pickle(data_dir / centers_before, required_columns=["neuron_id", "k", "center"])
    df_a = load_pickle(data_dir / centers_after, required_columns=["neuron_id", "k", "center"])

    b_xy = extract_single_centers(df_b, k_value=1).set_index("neuron_id")
    a_xy = extract_single_centers(df_a, k_value=1).set_index("neuron_id")

    common = sorted(set(b_xy.index).intersection(set(a_xy.index)))
    if len(common) < 5:
        raise ValueError("Too few common single-field neurons between before/after.")

    p1 = b_xy.loc[common, ["x", "y"]].to_numpy(dtype=float)
    p2 = a_xy.loc[common, ["x", "y"]].to_numpy(dtype=float)

    shift = p2 - p1
    mags = np.linalg.norm(shift, axis=1)

    to_reward = reward_xy[None, :] - p1
    reward_norm = np.linalg.norm(to_reward, axis=1, keepdims=True)
    reward_dir = np.divide(to_reward, reward_norm, out=np.zeros_like(to_reward), where=reward_norm > 0)

    directional = np.sum(shift * reward_dir, axis=1)
    distance_to_reward = np.linalg.norm(p1 - reward_xy[None, :], axis=1)

    return {
        "neuron_ids": np.asarray(common, dtype=int),
        "phase1_centroids": p1,
        "phase2_centroids": p2,
        "shift_vectors": shift,
        "shift_magnitudes": mags,
        "reward_dir": reward_dir,
        "directional_component": directional,
        "distance_to_reward": distance_to_reward,
    }


def magnitude_preserving_shuffle(
    mags: np.ndarray,
    reward_dir: np.ndarray,
    n_perm: int,
    seed: int,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm, dtype=float)
    last_surr_d = np.zeros_like(mags)

    for i in range(n_perm):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=mags.size)
        surr_shift = np.stack([mags * np.cos(theta), mags * np.sin(theta)], axis=1)
        surr_d = np.sum(surr_shift * reward_dir, axis=1)
        null[i] = float(np.mean(surr_d))
        last_surr_d = surr_d

    return null, float(np.mean(null)), float(np.std(null, ddof=1)), last_surr_d


def stratified_shuffle(
    shift_vectors: np.ndarray,
    reward_dir: np.ndarray,
    distance_to_reward: np.ndarray,
    n_perm: int,
    n_bins: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    max_dist = float(np.max(distance_to_reward))
    edges = np.linspace(0.0, max_dist + 1e-12, n_bins + 1)
    bin_id = np.digitize(distance_to_reward, edges, right=False) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)
    null_mean = np.zeros(n_perm, dtype=float)
    # Track per-bin mean directional components over permutations.
    null_bin_means = np.full((n_perm, n_bins), np.nan, dtype=float)

    for i in range(n_perm):
        assigned = shift_vectors.copy()
        for b in range(n_bins):
            idx = np.where(bin_id == b)[0]
            if idx.size <= 1:
                continue
            perm = rng.permutation(idx)
            assigned[idx] = shift_vectors[perm]

        surr_d = np.sum(assigned * reward_dir, axis=1)
        null_mean[i] = float(np.mean(surr_d))

        for b in range(n_bins):
            idx = np.where(bin_id == b)[0]
            if idx.size == 0:
                continue
            null_bin_means[i, b] = float(np.mean(surr_d[idx]))

    centers = 0.5 * (edges[:-1] + edges[1:])
    p5 = np.nanpercentile(null_bin_means, 5, axis=0)
    p95 = np.nanpercentile(null_bin_means, 95, axis=0)
    return null_mean, edges, centers, p5, p95


def cohens_d_one_sample(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    sd = np.std(values, ddof=1)
    if np.isclose(sd, 0.0):
        return np.nan
    return float(np.mean(values) / sd)


def plot_null_hist(null_vals: np.ndarray, observed: float, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(null_vals, bins=60, density=True, alpha=0.7, color="gray", label="Null mean directional shift")
    plt.axvline(observed, color="red", linestyle="--", linewidth=2.0, label="Observed mean directional shift")
    bins = np.linspace(float(np.min(null_vals)), float(np.max(null_vals)), 120)
    hist, edges = np.histogram(null_vals, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    tail_mask = centers >= observed
    if np.any(tail_mask):
        plt.fill_between(centers[tail_mask], 0, hist[tail_mask], color="red", alpha=0.25, label="Tail >= observed")

    plt.xlabel("Mean directional shift")
    plt.ylabel("Density")
    plt.title("Magnitude-preserving shuffle null")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_distance_envelope(
    distance_to_reward: np.ndarray,
    directional_real: np.ndarray,
    bin_centers: np.ndarray,
    p5: np.ndarray,
    p95: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(distance_to_reward, directional_real, s=20, alpha=0.55, color="#2E5FA3", label="Real neurons")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.fill_between(bin_centers, p5, p95, color="gray", alpha=0.3, label="Stratified null 5-95%")
    plt.plot(bin_centers, p5, color="gray", linewidth=1.2)
    plt.plot(bin_centers, p95, color="gray", linewidth=1.2)
    plt.xlabel("Distance to reward (Phase 1 centroid)")
    plt.ylabel("Directional shift component")
    plt.title("Directional shift vs distance with null envelope")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    data_dir = Path(args.data_dir)
    reward = np.array([args.reward_x, args.reward_y], dtype=float)

    data = prepare_shift_data(
        data_dir=data_dir,
        centers_before=args.centers_before,
        centers_after=args.centers_after,
        reward_xy=reward,
    )

    directional = data["directional_component"]
    d_bar = float(np.mean(directional))
    t_res = ttest_1samp(directional, popmean=0.0, alternative="greater")
    try:
        w_res = wilcoxon(directional, alternative="greater", zero_method="wilcox")
        w_stat = float(w_res.statistic)
        w_p = float(w_res.pvalue)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    coh_d = cohens_d_one_sample(directional)
    null_mag, null_mag_mean, null_mag_std, _ = magnitude_preserving_shuffle(
        mags=data["shift_magnitudes"],
        reward_dir=data["reward_dir"],
        n_perm=args.n_permutations,
        seed=args.seed,
    )
    p_perm = float(np.mean(null_mag >= d_bar))
    z_perm = float((d_bar - null_mag_mean) / null_mag_std) if not np.isclose(null_mag_std, 0.0) else np.nan
    null_strat, edges, centers, p5, p95 = stratified_shuffle(
        shift_vectors=data["shift_vectors"],
        reward_dir=data["reward_dir"],
        distance_to_reward=data["distance_to_reward"],
        n_perm=args.n_permutations,
        n_bins=args.distance_bins,
        seed=args.seed + 1,
    )
    p_perm_strat = float(np.mean(null_strat >= d_bar))
    null_strat_mean = float(np.mean(null_strat))
    null_strat_std = float(np.std(null_strat, ddof=1))
    z_perm_strat = float((d_bar - null_strat_mean) / null_strat_std) if not np.isclose(null_strat_std, 0.0) else np.nan

    plot_null_hist(null_mag, d_bar, out_dir / "shift_shuffle_histogram.png")
    plot_distance_envelope(
        distance_to_reward=data["distance_to_reward"],
        directional_real=directional,
        bin_centers=centers,
        p5=p5,
        p95=p95,
        out_path=out_dir / "shift_distance_null_envelope.png",
    )

    payload: dict[str, Any] = {
        "script": "shift_shuffle_control.py",
        "seed": int(args.seed),
        "phase": "before_to_after",
        "n_neurons": int(directional.size),
        "metric_name": "mean_directional_shift",
        "real_value": d_bar,
        "real_value_sd": float(np.std(directional, ddof=1)),
        "null_mean": null_mag_mean,
        "null_std": null_mag_std,
        "null_95th_pctile": float(np.percentile(null_mag, 95)),
        "null_99th_pctile": float(np.percentile(null_mag, 99)),
        "z_score": z_perm,
        "permutation_pval": p_perm,
        "n_permutations": int(args.n_permutations),
        "ttest_statistic": float(t_res.statistic),
        "ttest_pval": float(t_res.pvalue),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pval": w_p,
        "cohens_d": coh_d,
        "ci_95_low": float(np.percentile(directional, 2.5)),
        "ci_95_high": float(np.percentile(directional, 97.5)),
        "stratified_shuffle": {
            "distance_bins": int(args.distance_bins),
            "bin_edges": edges.tolist(),
            "bin_centers": centers.tolist(),
            "null_5th_by_bin": p5.tolist(),
            "null_95th_by_bin": p95.tolist(),
            "null_mean": null_strat_mean,
            "null_std": null_strat_std,
            "z_score": z_perm_strat,
            "permutation_pval": p_perm_strat,
        },
    }

    save_json(out_dir / "shift_shuffle_results.json", payload)
    np.save(out_dir / "shift_shuffle_null_magnitude.npy", null_mag)
    np.save(out_dir / "shift_shuffle_null_stratified.npy", null_strat)
    print(f"Saved shift shuffle outputs to: {out_dir}")

if __name__ == "__main__":
    main()
