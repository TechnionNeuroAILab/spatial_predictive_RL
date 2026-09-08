from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from common import DEFAULT_REWARD, bootstrap_mean_ci, ensure_dir, load_behavior_activity, pivot_activity_matrix, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Population vector correlation across positions.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/population_vector")
    parser.add_argument("--pv-phase1", type=str, default="pv_phase1.npz")
    parser.add_argument("--pv-phase2", type=str, default="pv_phase2.npz")
    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=float(DEFAULT_REWARD[0]))
    parser.add_argument("--reward-y", type=float, default=float(DEFAULT_REWARD[1]))
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--min-visits", type=int, default=3)
    parser.add_argument("--distance-bins", type=int, default=6)
    parser.add_argument("--goal-area-fraction", type=float, default=0.20)
    parser.add_argument("--n-shuffle", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_goal_position(data_dir: Path, goal_file: str, reward_x: float, reward_y: float) -> tuple[np.ndarray, str]:
    p = data_dir / goal_file
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float).reshape(-1)
        if arr.size >= 2:
            return np.array([float(arr[0]), float(arr[1])], dtype=float), str(p)
    return np.array([reward_x, reward_y], dtype=float), "fallback_args"


def load_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    npz = np.load(path)
    if "positions" not in npz.files or "activations" not in npz.files:
        return None

    positions = np.asarray(npz["positions"], dtype=float)
    activations = np.asarray(npz["activations"], dtype=float)

    if positions.ndim != 2 or positions.shape[1] < 2:
        return None
    if activations.ndim != 2:
        return None

    n = min(positions.shape[0], activations.shape[0])
    return positions[:n, :2], activations[:n]


def load_from_pkl(data_dir: Path, period: str) -> tuple[np.ndarray, np.ndarray]:
    beh, act, _ = load_behavior_activity(data_dir, period)
    x_df, b_df = pivot_activity_matrix(act, beh)
    if b_df is None:
        raise RuntimeError("Behavior alignment failed.")
    positions = b_df[["x", "y"]].to_numpy(dtype=float)
    activity = x_df.to_numpy(dtype=float)
    n = min(positions.shape[0], activity.shape[0])
    return positions[:n], activity[:n]


def load_period_data(data_dir: Path, npz_name: str, period: str) -> tuple[np.ndarray, np.ndarray, str]:
    npz_path = data_dir / npz_name
    loaded = load_from_npz(npz_path)
    if loaded is not None:
        return loaded[0], loaded[1], str(npz_path)

    pos, act = load_from_pkl(data_dir, period)
    return pos, act, f"fallback_pkl_{period}"


def align_neurons(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(A.shape[1], B.shape[1])
    return A[:, :n], B[:, :n]


def build_pv_map(positions: np.ndarray, activations: np.ndarray, grid_size: int, min_visits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build KxKxN population-vector map and visit-count map.
    Bins with visits < min_visits are left as NaN vectors.
    """
    xb = np.clip(np.floor(positions[:, 0] * grid_size).astype(int), 0, grid_size - 1)
    yb = np.clip(np.floor(positions[:, 1] * grid_size).astype(int), 0, grid_size - 1)
    n_neurons = activations.shape[1]
    pv = np.full((grid_size, grid_size, n_neurons), np.nan, dtype=float)
    visits = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            idx = np.where((xb == i) & (yb == j))[0]
            visits[i, j] = int(idx.size)
            if idx.size < min_visits:
                continue
            pv[i, j] = np.mean(activations[idx], axis=0)

    return pv, visits


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(pearsonr(a, b)[0])


def pv_correlation_map(pv1: np.ndarray, pv2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid_size = pv1.shape[0]
    corr = np.full((grid_size, grid_size), np.nan, dtype=float)
    valid = np.zeros((grid_size, grid_size), dtype=bool)

    for i in range(grid_size):
        for j in range(grid_size):
            r = corr_safe(pv1[i, j], pv2[i, j])
            corr[i, j] = r
            valid[i, j] = np.isfinite(r)

    return corr, valid


def distance_map(grid_size: int, goal: np.ndarray) -> np.ndarray:
    centers = (np.arange(grid_size, dtype=float) + 0.5) / grid_size
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    return np.sqrt((xx - goal[0]) ** 2 + (yy - goal[1]) ** 2)


def binned_distance_curve(dist: np.ndarray, corr: np.ndarray, n_bins: int) -> pd.DataFrame:
    d = dist[np.isfinite(corr)]
    r = corr[np.isfinite(corr)]
    if d.size == 0:
        return pd.DataFrame(columns=["bin", "distance_center", "mean", "sem", "count"])

    edges = np.linspace(np.min(d), np.max(d) + 1e-12, n_bins + 1)
    b = np.digitize(d, edges, right=False) - 1
    b = np.clip(b, 0, n_bins - 1)

    df = pd.DataFrame({"bin": b, "distance": d, "corr": r})
    out = df.groupby("bin")["corr"].agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    out["distance_center"] = centers[out["bin"].to_numpy(dtype=int)]
    return out


def shuffle_mean_pv_corr(pv1: np.ndarray, pv2: np.ndarray, valid_mask: np.ndarray, n_shuffle: int, seed: int) -> np.ndarray:
    """
    Shuffle baseline: randomly permute Phase 2 bin assignments.
    Returns null distribution of mean PV correlations.
    """
    rng = np.random.default_rng(seed)

    idx = np.argwhere(valid_mask)
    if idx.shape[0] == 0:
        return np.array([], dtype=float)

    v1 = np.stack([pv1[i, j] for i, j in idx], axis=0)
    v2 = np.stack([pv2[i, j] for i, j in idx], axis=0)

    null = np.zeros(n_shuffle, dtype=float)
    for k in range(n_shuffle):
        perm = rng.permutation(v2.shape[0])
        vals = np.array([corr_safe(v1[i], v2[perm[i]]) for i in range(v1.shape[0])], dtype=float)
        null[k] = float(np.nanmean(vals))

    return null


def plot_pv_map(corr_map: np.ndarray, goal: np.ndarray, goal_radius: float, out_path: Path) -> None:
    plt.figure(figsize=(6, 5.5))
    ax = plt.gca()
    im = ax.imshow(corr_map.T, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    plt.colorbar(im, shrink=0.85, label="PV correlation (Phase1 vs Phase2)")

    gx = goal[0] * corr_map.shape[0] - 0.5
    gy = goal[1] * corr_map.shape[1] - 0.5
    gr = goal_radius * corr_map.shape[0]
    ax.add_patch(Circle((gx, gy), gr, fill=False, linestyle="--", linewidth=2.0, color="black"))

    ax.set_title("PV correlation map")
    ax.set_xlabel("x bin")
    ax.set_ylabel("y bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_corr_vs_distance(curve_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 4.8))
    if not curve_df.empty:
        plt.errorbar(
            curve_df["distance_center"],
            curve_df["mean"],
            yerr=curve_df["sem"],
            marker="o",
            linewidth=2.0,
            capsize=4,
            color="#2E5FA3",
        )
    plt.xlabel("Distance to goal")
    plt.ylabel("Mean PV correlation")
    plt.title("PV correlation vs distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_goal_vs_nongoal(goal_vals: np.ndarray, non_vals: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 4.8))
    data = [goal_vals[np.isfinite(goal_vals)], non_vals[np.isfinite(non_vals)]]
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks([1, 2], ["Goal zone", "Non-goal zone"])
    plt.ylabel("PV correlation")
    plt.title("Goal vs non-goal PV correlations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir if args.output_dir else (data_dir / "population_vector"))
    fig_dir = ensure_dir(out_dir / "figures")
    goal, goal_source = load_goal_position(data_dir, args.goal_file, args.reward_x, args.reward_y)
    goal_radius = float(np.sqrt(args.goal_area_fraction / np.pi))
    pos1, act1, src1 = load_period_data(data_dir, args.pv_phase1, "before")
    pos2, act2, src2 = load_period_data(data_dir, args.pv_phase2, "after")
    act1, act2 = align_neurons(act1, act2)
    pv1, visits1 = build_pv_map(pos1, act1, grid_size=args.grid_size, min_visits=args.min_visits)
    pv2, visits2 = build_pv_map(pos2, act2, grid_size=args.grid_size, min_visits=args.min_visits)
    corr_map, valid = pv_correlation_map(pv1, pv2)
    dist = distance_map(args.grid_size, goal)
    valid_corr = corr_map[np.isfinite(corr_map)]
    valid_dist = dist[np.isfinite(corr_map)]

    if valid_corr.size >= 3:
        rho, rho_p = spearmanr(valid_dist, valid_corr)
        rho = float(rho)
        rho_p = float(rho_p)
    else:
        rho, rho_p = np.nan, np.nan

    goal_mask = (dist <= goal_radius) & np.isfinite(corr_map)
    non_mask = (dist > goal_radius) & np.isfinite(corr_map)
    goal_vals = corr_map[goal_mask]
    non_vals = corr_map[non_mask]
    mean_goal = float(np.nanmean(goal_vals)) if goal_vals.size else np.nan
    mean_non = float(np.nanmean(non_vals)) if non_vals.size else np.nan
    if goal_vals.size > 0 and non_vals.size > 0:
        mw = mannwhitneyu(goal_vals, non_vals, alternative="less")
        mw_stat = float(mw.statistic)
        mw_p = float(mw.pvalue)
    else:
        mw_stat, mw_p = np.nan, np.nan

    frac_goal_low = float(np.mean(goal_vals < 0.5)) if goal_vals.size else np.nan
    frac_non_low = float(np.mean(non_vals < 0.5)) if non_vals.size else np.nan
    curve = binned_distance_curve(dist, corr_map, n_bins=args.distance_bins)
    null = shuffle_mean_pv_corr(pv1, pv2, valid_mask=valid, n_shuffle=args.n_shuffle, seed=args.seed)
    obs_mean = float(np.nanmean(valid_corr)) if valid_corr.size else np.nan
    null_mean = float(np.nanmean(null)) if null.size else np.nan
    p_null = float(np.mean(null >= obs_mean)) if null.size and np.isfinite(obs_mean) else np.nan
    plot_pv_map(corr_map, goal, goal_radius, fig_dir / "pv_correlation_map.png")
    plot_corr_vs_distance(curve, fig_dir / "pv_corr_vs_distance.png")
    plot_goal_vs_nongoal(goal_vals, non_vals, fig_dir / "pv_corr_goal_vs_non_goal.png")
    np.save(out_dir / "pv_corr_map.npy", corr_map)
    np.save(out_dir / "pv_shuffle_null.npy", null)
    qualitative_match = bool(np.isfinite(mean_goal) and np.isfinite(mean_non) and (mean_goal < mean_non))

    # Bootstrap 95% CI on the goal-zone mean PV correlation
    _goal_mean, _goal_ci_lo, _goal_ci_hi = bootstrap_mean_ci(goal_vals, n_boot=2000, seed=args.seed)

    payload: dict[str, Any] = {
        "script": "population_vector.py",
        "comparison_paper": "Leutgeb et al. 2005",
        "comparison_metric": "Goal-zone selective remapping (lower PV corr near goal)",
        "comparison_value_paper": None,
        "comparison_value_paper_note": "No single canonical scalar in Leutgeb et al.; directional comparison used.",
        "your_value": mean_goal,
        "your_value_ci_95": [float(_goal_ci_lo), float(_goal_ci_hi)],
        "qualitative_match": qualitative_match,
        "goal_position": goal.tolist(),
        "goal_position_source": goal_source,
        "goal_zone_radius": goal_radius,
        "params": {
            "grid_size": int(args.grid_size),
            "min_visits": int(args.min_visits),
            "distance_bins": int(args.distance_bins),
            "n_shuffle": int(args.n_shuffle),
        },
        "visit_coverage": {
            "phase1_fraction_bins_valid": float(np.mean(visits1 >= args.min_visits)),
            "phase2_fraction_bins_valid": float(np.mean(visits2 >= args.min_visits)),
            "fraction_bins_valid_both": float(np.mean(valid)),
        },
        "metrics": {
            "mean_pv_corr_goal_zone": mean_goal,
            "mean_pv_corr_non_goal_zone": mean_non,
            "mannwhitney_u_stat": mw_stat,
            "mannwhitney_u_p": mw_p,
            "spearman_rho_distance_vs_pv_corr": rho,
            "spearman_p_distance_vs_pv_corr": rho_p,
            "fraction_goal_zone_bins_pv_corr_lt_0_5": frac_goal_low,
            "fraction_non_goal_zone_bins_pv_corr_lt_0_5": frac_non_low,
            "shuffle_null_mean": null_mean,
            "shuffle_null_p_value": p_null,
            "observed_mean_pv_corr": obs_mean,
        },
        "sources": {
            "phase1_data_source": src1,
            "phase2_data_source": src2,
        },
    }

    save_json(out_dir / "pv_results.json", payload)
    print(f"Saved PV outputs to: {out_dir}")


if __name__ == "__main__":
    main()
