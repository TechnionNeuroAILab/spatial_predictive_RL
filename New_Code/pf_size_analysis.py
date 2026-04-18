from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label
from scipy.stats import norm, spearmanr

from common import DEFAULT_REWARD, ensure_dir, load_behavior_activity, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Place-field size vs distance-from-reward analysis.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/pf_size_analysis")
    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=float(DEFAULT_REWARD[0]))
    parser.add_argument("--reward-y", type=float, default=float(DEFAULT_REWARD[1]))
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--smooth-sigma", type=float, default=1.5)
    parser.add_argument("--threshold-frac", type=float, default=0.5)
    parser.add_argument("--noise-threshold", type=float, default=0.05)
    parser.add_argument("--min-region-bins", type=int, default=2)
    parser.add_argument("--distance-bins", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_goal_position(data_dir: Path, goal_file: str, reward_x: float, reward_y: float) -> tuple[np.ndarray, str]:
    p = data_dir / goal_file
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float).reshape(-1)
        if arr.size >= 2:
            return np.array([float(arr[0]), float(arr[1])], dtype=float), str(p)
    return np.array([reward_x, reward_y], dtype=float), "fallback_args"


def build_rate_map(df_neuron: pd.DataFrame, grid_size: int) -> np.ndarray:
    x = np.clip(np.floor(df_neuron["x"].to_numpy(dtype=float) * grid_size).astype(int), 0, grid_size - 1)
    y = np.clip(np.floor(df_neuron["y"].to_numpy(dtype=float) * grid_size).astype(int), 0, grid_size - 1)
    a = df_neuron["activity"].to_numpy(dtype=float)
    occ = np.zeros((grid_size, grid_size), dtype=float)
    total = np.zeros((grid_size, grid_size), dtype=float)
    for i in range(x.shape[0]):
        occ[x[i], y[i]] += 1.0
        total[x[i], y[i]] += a[i]

    rate = np.divide(total, occ, out=np.zeros_like(total), where=occ > 0)
    return rate


def largest_component_mask(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    labeled, n_labels = label(binary_mask)
    if n_labels == 0:
        return np.zeros_like(binary_mask, dtype=bool), 0
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    comp = int(np.argmax(sizes))
    comp_size = int(sizes[comp])
    return labeled == comp, comp_size


def weighted_centroid_from_mask(rate_map: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    weights = rate_map * mask
    total_w = float(np.sum(weights))
    if total_w <= 0:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return np.nan, np.nan
        return float(np.mean(xs)), float(np.mean(ys))

    ys, xs = np.indices(rate_map.shape)
    cx = float(np.sum(xs * weights) / total_w)
    cy = float(np.sum(ys * weights) / total_w)
    return cx, cy


def partial_spearman(x: np.ndarray, y: np.ndarray, covar: np.ndarray) -> tuple[float, float, str]:
    df = pd.DataFrame({"x": x, "y": y, "covar": covar}).dropna()
    if df.shape[0] < 5:
        return np.nan, np.nan, "insufficient_data"
    try:
        import pingouin as pg  # type: ignore
        res = pg.partial_corr(data=df, x="x", y="y", covar="covar", method="spearman")
        return float(res["r"].iloc[0]), float(res["p-val"].iloc[0]), "pingouin"
    except Exception:
        xr = df["x"].rank(method="average").to_numpy(dtype=float)
        yr = df["y"].rank(method="average").to_numpy(dtype=float)
        cr = df["covar"].rank(method="average").to_numpy(dtype=float)
        X = np.column_stack([np.ones(cr.shape[0]), cr])
        bx, *_ = np.linalg.lstsq(X, xr, rcond=None)
        by, *_ = np.linalg.lstsq(X, yr, rcond=None)
        rx = xr - X @ bx
        ry = yr - X @ by
        rho, p = spearmanr(rx, ry)
        return float(rho), float(p), "rank_residual_fallback"


def fisher_z_compare(rho1: float, n1: int, rho2: float, n2: int) -> tuple[float, float]:
    if n1 <= 3 or n2 <= 3:
        return np.nan, np.nan
    r1 = np.clip(rho1, -0.999999, 0.999999)
    r2 = np.clip(rho2, -0.999999, 0.999999)
    z1 = 0.5 * np.log((1.0 + r1) / (1.0 - r1))
    z2 = 0.5 * np.log((1.0 + r2) / (1.0 - r2))
    z_diff = (z1 - z2) / np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    p = 2.0 * (1.0 - norm.cdf(abs(z_diff)))
    return float(z_diff), float(p)


def compute_period_table(
    merged: pd.DataFrame,
    reward: np.ndarray,
    grid_size: int,
    smooth_sigma: float,
    threshold_frac: float,
    noise_threshold: float,
    min_region_bins: int,
    period_name: str,
) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict[str, Any]] = []
    excluded = 0
    bin_size = 1.0 / grid_size
    for nid, df_n in merged.groupby("neuron_id"):
        rate_map = build_rate_map(df_n, grid_size=grid_size)
        rate_map = gaussian_filter(rate_map, sigma=smooth_sigma)
        peak = float(np.max(rate_map))
        if peak < noise_threshold:
            excluded += 1
            continue

        thresh = threshold_frac * peak
        mask = rate_map > thresh

        largest_mask, n_bins = largest_component_mask(mask)
        if n_bins < min_region_bins:
            excluded += 1
            continue
        cx_bin, cy_bin = weighted_centroid_from_mask(rate_map, largest_mask)
        if not np.isfinite(cx_bin) or not np.isfinite(cy_bin):
            excluded += 1
            continue
        cx = (cx_bin + 0.5) * bin_size
        cy = (cy_bin + 0.5) * bin_size

        area = n_bins * (bin_size ** 2)
        mean_rate = float(np.mean(df_n["activity"].to_numpy(dtype=float)))
        dist = float(np.linalg.norm(np.array([cx, cy], dtype=float) - reward))
        rows.append(
            {
                "neuron_id": int(nid),
                "period": period_name,
                "field_size_m2": float(area),
                "field_size_bins": int(n_bins),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "distance_to_reward": dist,
                "mean_rate": mean_rate,
                "peak_rate": peak,
            }
        )

    table = pd.DataFrame(rows)
    n_with_field = int(table.shape[0])
    return table, n_with_field, excluded


def binned_distance_summary(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["distance_bin", "distance_center", "mean", "sem", "count"])

    d = df["distance_to_reward"].to_numpy(dtype=float)
    edges = np.linspace(np.min(d), np.max(d) + 1e-12, n_bins + 1)

    b = np.digitize(d, edges, right=False) - 1
    b = np.clip(b, 0, n_bins - 1)
    tmp = df.copy()
    tmp["distance_bin"] = b

    out = (
        tmp.groupby("distance_bin")["field_size_m2"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    out["distance_center"] = centers[out["distance_bin"].to_numpy(dtype=int)]
    return out


def compute_size_ratio(df: pd.DataFrame) -> tuple[float, float, float]:
    if df.shape[0] < 8:
        return np.nan, np.nan, np.nan
    q1 = float(np.quantile(df["distance_to_reward"], 0.25))
    q3 = float(np.quantile(df["distance_to_reward"], 0.75))
    near = df[df["distance_to_reward"] <= q1]["field_size_m2"].to_numpy(dtype=float)
    far = df[df["distance_to_reward"] >= q3]["field_size_m2"].to_numpy(dtype=float)
    if near.size == 0 or far.size == 0:
        return np.nan, np.nan, np.nan

    m_near = float(np.mean(near))
    m_far = float(np.mean(far))
    ratio = float(m_near / m_far) if not np.isclose(m_far, 0.0) else np.nan
    return ratio, m_near, m_far


def plot_phase_scatter(df: pd.DataFrame, binned: pd.DataFrame, title: str, color: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["distance_to_reward"],
        df["field_size_m2"],
        s=16,
        alpha=0.35,
        color="gray",
        label="Neurons",
    )
    if not binned.empty:
        plt.errorbar(
            binned["distance_center"],
            binned["mean"],
            yerr=binned["sem"],
            color=color,
            marker="o",
            linewidth=2.0,
            capsize=4,
            label="Binned mean ± SEM",
        )

    plt.xlabel("Distance to reward")
    plt.ylabel("Place field size (m^2)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_combined(before_df: pd.DataFrame, after_df: pd.DataFrame, before_bin: pd.DataFrame, after_bin: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5.5))
    plt.scatter(before_df["distance_to_reward"], before_df["field_size_m2"], s=12, alpha=0.2, color="#2E5FA3")
    plt.scatter(after_df["distance_to_reward"], after_df["field_size_m2"], s=12, alpha=0.2, color="#E07A2D")
    if not before_bin.empty:
        plt.errorbar(
            before_bin["distance_center"],
            before_bin["mean"],
            yerr=before_bin["sem"],
            color="#2E5FA3",
            marker="o",
            linewidth=2.0,
            capsize=4,
            label="Phase 1",
        )

    if not after_bin.empty:
        plt.errorbar(
            after_bin["distance_center"],
            after_bin["mean"],
            yerr=after_bin["sem"],
            color="#E07A2D",
            marker="o",
            linewidth=2.0,
            capsize=4,
            label="Phase 2",
        )

    plt.xlabel("Distance to reward")
    plt.ylabel("Place field size (m^2)")
    plt.title("Place field size vs distance (combined)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_ratio_bar(ratio_before: float, ratio_after: float, out_path: Path) -> None:
    labels = ["Phase 1", "Phase 2"]
    vals = [ratio_before, ratio_after]

    plt.figure(figsize=(6, 4.5))
    plt.bar(labels, vals, color=["#2E5FA3", "#E07A2D"], alpha=0.9)
    plt.axhline(1.0, linestyle="--", color="black", linewidth=1.2)
    plt.ylabel("Near/Far field size ratio")
    plt.title("Near-vs-far place field size ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_correlations(df: pd.DataFrame) -> dict[str, Any]:
    if df.shape[0] < 4:
        return {
            "n": int(df.shape[0]),
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "partial_spearman_rho": np.nan,
            "partial_spearman_p": np.nan,
            "partial_method": "insufficient_data",
        }

    rho, p = spearmanr(df["distance_to_reward"], df["field_size_m2"])
    prho, pp, pm = partial_spearman(
        x=df["distance_to_reward"].to_numpy(dtype=float),
        y=df["field_size_m2"].to_numpy(dtype=float),
        covar=df["mean_rate"].to_numpy(dtype=float),
    )
    return {
        "n": int(df.shape[0]),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "partial_spearman_rho": float(prho),
        "partial_spearman_p": float(pp),
        "partial_method": pm,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir if args.output_dir else (data_dir / "pf_size_analysis"))
    fig_dir = ensure_dir(out_dir / "figures")
    reward, goal_source = load_goal_position(data_dir, args.goal_file, args.reward_x, args.reward_y)
    _, _, merged_before = load_behavior_activity(data_dir, "before")
    _, _, merged_after = load_behavior_activity(data_dir, "after")
    before_table, n_before, excl_before = compute_period_table(
        merged=merged_before,
        reward=reward,
        grid_size=args.grid_size,
        smooth_sigma=args.smooth_sigma,
        threshold_frac=args.threshold_frac,
        noise_threshold=args.noise_threshold,
        min_region_bins=args.min_region_bins,
        period_name="phase1",
    )

    after_table, n_after, excl_after = compute_period_table(
        merged=merged_after,
        reward=reward,
        grid_size=args.grid_size,
        smooth_sigma=args.smooth_sigma,
        threshold_frac=args.threshold_frac,
        noise_threshold=args.noise_threshold,
        min_region_bins=args.min_region_bins,
        period_name="phase2",
    )
    before_table.to_csv(out_dir / "pf_size_phase1_table.csv", index=False)
    after_table.to_csv(out_dir / "pf_size_phase2_table.csv", index=False)
    before_bin = binned_distance_summary(before_table, n_bins=args.distance_bins)
    after_bin = binned_distance_summary(after_table, n_bins=args.distance_bins)
    plot_phase_scatter(before_table, before_bin, "Phase 1: field size vs distance", "#2E5FA3", fig_dir / "pf_size_vs_distance_phase1.png")
    plot_phase_scatter(after_table, after_bin, "Phase 2: field size vs distance", "#E07A2D", fig_dir / "pf_size_vs_distance_phase2.png")
    plot_combined(before_table, after_table, before_bin, after_bin, fig_dir / "pf_size_vs_distance_combined.png")
    ratio_before, near_before, far_before = compute_size_ratio(before_table)
    ratio_after, near_after, far_after = compute_size_ratio(after_table)
    plot_ratio_bar(ratio_before, ratio_after, fig_dir / "pf_size_ratio_bar.png")
    corr_before = run_correlations(before_table)
    corr_after = run_correlations(after_table)
    z_diff, z_p = fisher_z_compare(
        rho1=corr_before["spearman_rho"],
        n1=corr_before["n"],
        rho2=corr_after["spearman_rho"],
        n2=corr_after["n"],
    )

    qualitative_match = bool(np.isfinite(ratio_after) and (0.4 <= ratio_after <= 0.7))

    payload: dict[str, Any] = {
        "script": "pf_size_analysis.py",
        "comparison_paper": "Lee et al. 2006; Dupret et al. 2010",
        "comparison_metric": "Near/Far place-field size ratio",
        "comparison_value_paper": 0.5,
        "comparison_value_paper_note": "Experimental near-reward fields reported ~40-60% smaller.",
        "your_value": float(ratio_after) if np.isfinite(ratio_after) else np.nan,
        "your_value_ci_95": [np.nan, np.nan],
        "qualitative_match": qualitative_match,
        "goal_position": reward.tolist(),
        "goal_position_source": goal_source,
        "params": {
            "grid_size": int(args.grid_size),
            "smooth_sigma": float(args.smooth_sigma),
            "threshold_frac": float(args.threshold_frac),
            "noise_threshold": float(args.noise_threshold),
            "min_region_bins": int(args.min_region_bins),
            "distance_bins": int(args.distance_bins),
        },
        "phase1": {
            **corr_before,
            "size_ratio": float(ratio_before) if np.isfinite(ratio_before) else np.nan,
            "mean_near_size": float(near_before) if np.isfinite(near_before) else np.nan,
            "mean_far_size": float(far_before) if np.isfinite(far_before) else np.nan,
            "n_neurons_with_field": int(n_before),
            "n_neurons_excluded": int(excl_before),
        },
        "phase2": {
            **corr_after,
            "size_ratio": float(ratio_after) if np.isfinite(ratio_after) else np.nan,
            "mean_near_size": float(near_after) if np.isfinite(near_after) else np.nan,
            "mean_far_size": float(far_after) if np.isfinite(far_after) else np.nan,
            "n_neurons_with_field": int(n_after),
            "n_neurons_excluded": int(excl_after),
        },
        "before_vs_after": {
            "fisher_z_diff": float(z_diff) if np.isfinite(z_diff) else np.nan,
            "fisher_z_p": float(z_p) if np.isfinite(z_p) else np.nan,
        },
    }

    save_json(out_dir / "pf_size_results.json", payload)
    print(f"Saved PF-size outputs to: {out_dir}")


if __name__ == "__main__":
    main()
