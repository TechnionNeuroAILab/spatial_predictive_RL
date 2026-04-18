from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.stats import chisquare

from common import DEFAULT_REWARD, ensure_dir, extract_single_centers, load_pickle, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hollup-style over-representation index analysis.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/overrepresentation_index")
    parser.add_argument("--centres-phase1", type=str, default="place_field_centres_phase1.npy")
    parser.add_argument("--centres-phase2", type=str, default="place_field_centres_phase2.npy")
    parser.add_argument("--centers-before-pkl", type=str, default="df_centers_before.pkl")
    parser.add_argument("--centers-after-pkl", type=str, default="df_centers_after.pkl")
    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=float(DEFAULT_REWARD[0]))
    parser.add_argument("--reward-y", type=float, default=float(DEFAULT_REWARD[1]))
    parser.add_argument("--goal-area-fraction", type=float, default=0.20)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trajectory-file",
        type=str,
        default="ori_trajectory.npz",
        help="Optional precomputed trajectory file with keys episodes, ori_per_episode, success_rate_per_episode.",
    )
    return parser.parse_args()


def load_goal_position(data_dir: Path, goal_file: str, reward_x: float, reward_y: float) -> tuple[np.ndarray, str]:
    goal_path = data_dir / goal_file
    if goal_path.exists():
        goal = np.asarray(np.load(goal_path), dtype=float).reshape(-1)
        if goal.size >= 2:
            return np.array([float(goal[0]), float(goal[1])], dtype=float), str(goal_path)
    return np.array([reward_x, reward_y], dtype=float), "fallback_args"


def load_centres_array(data_dir: Path, npy_name: str, pkl_name: str) -> tuple[np.ndarray, str]:
    npy_path = data_dir / npy_name
    if npy_path.exists():
        arr = np.asarray(np.load(npy_path), dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2], str(npy_path)

    pkl_path = data_dir / pkl_name
    df = load_pickle(pkl_path, required_columns=["neuron_id", "k", "center"])
    centres = extract_single_centers(df, k_value=1)
    if centres.empty:
        raise ValueError(f"No valid single-field centres found in {pkl_path}")
    arr = centres[["x", "y"]].to_numpy(dtype=float)
    return arr, str(pkl_path)


def compute_ori(centres: np.ndarray, goal: np.ndarray, radius: float, f_goal: float, f_non_goal: float) -> dict[str, float]:
    d = np.linalg.norm(centres - goal[None, :], axis=1)
    in_goal = d <= radius
    n_goal = int(np.sum(in_goal))
    n_non_goal = int(np.sum(~in_goal))
    n_total = int(centres.shape[0])

    if n_total == 0:
        return {
            "n_total": 0,
            "n_goal": 0,
            "n_non_goal": 0,
            "density_goal": np.nan,
            "density_non_goal": np.nan,
            "ori": np.nan,
        }

    density_goal = n_goal / (n_total * f_goal)
    density_non_goal = n_non_goal / (n_total * f_non_goal)
    if n_non_goal == 0:
        ori = np.inf
    else:
        ori = (n_goal * f_non_goal) / (n_non_goal * f_goal)

    return {
        "n_total": n_total,
        "n_goal": n_goal,
        "n_non_goal": n_non_goal,
        "density_goal": float(density_goal),
        "density_non_goal": float(density_non_goal),
        "ori": float(ori),
    }


def bootstrap_ori(centres: np.ndarray, goal: np.ndarray, radius: float, f_goal: float, f_non_goal: float, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = centres.shape[0]
    vals = np.zeros(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        c = centres[idx]
        res = compute_ori(c, goal, radius, f_goal, f_non_goal)
        vals[i] = float(res["ori"])

    return vals


def permutation_ori_uniform(n_fields: int, goal: np.ndarray, radius: float, f_goal: float, f_non_goal: float, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = np.zeros(n_perm, dtype=float)

    for i in range(n_perm):
        c = rng.uniform(0.0, 1.0, size=(n_fields, 2))
        res = compute_ori(c, goal, radius, f_goal, f_non_goal)
        vals[i] = float(res["ori"])

    return vals


def plot_ori_bar(before_ori: float, after_ori: float, before_ci: tuple[float, float], after_ci: tuple[float, float], out_path: Path) -> None:
    labels = ["Phase 1", "Phase 2"]
    vals = np.array([before_ori, after_ori], dtype=float)
    lows = np.array([before_ori - before_ci[0], after_ori - after_ci[0]], dtype=float)
    highs = np.array([before_ci[1] - before_ori, after_ci[1] - after_ori], dtype=float)
    plt.figure(figsize=(6, 4.5))
    plt.bar(labels, vals, yerr=[lows, highs], capsize=5, alpha=0.85, color=["#2E5FA3", "#E07A2D"])
    plt.axhline(1.0, linestyle="--", color="black", linewidth=1.2)
    plt.ylabel("Over-representation index (ORI)")
    plt.title("Goal-zone over-representation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_field_distribution(centres1: np.ndarray, centres2: np.ndarray, goal: np.ndarray, radius: float, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(centres1[:, 0], centres1[:, 1], s=16, alpha=0.5, label="Phase 1")
    plt.scatter(centres2[:, 0], centres2[:, 1], s=16, alpha=0.5, label="Phase 2")

    ax = plt.gca()
    ax.add_patch(Circle((goal[0], goal[1]), radius, fill=False, linestyle="--", linewidth=2.0, color="red"))
    ax.scatter([goal[0]], [goal[1]], marker="*", s=120, color="gold", edgecolors="k", zorder=5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Place-field centre distribution")
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_ori_trajectory(episodes: np.ndarray, ori_vals: np.ndarray, success: np.ndarray, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(episodes, ori_vals, color="#2E5FA3", linewidth=2.0, label="ORI")
    ax1.set_xlabel("RL Episode")
    ax1.set_ylabel("ORI", color="#2E5FA3")
    ax1.tick_params(axis="y", labelcolor="#2E5FA3")
    ax1.axhline(1.0, linestyle="--", color="black", linewidth=1.0)

    ax2 = ax1.twinx()
    ax2.plot(episodes, success, color="#1B8A5A", linewidth=2.0, alpha=0.85, label="Success rate")
    ax2.set_ylabel("Success rate", color="#1B8A5A")
    ax2.tick_params(axis="y", labelcolor="#1B8A5A")

    ax1.set_title("ORI trajectory with success overlay")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def maybe_load_trajectory(data_dir: Path, filename: str, before_ori: float, after_ori: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    path = data_dir / filename
    if path.exists():
        npz = np.load(path)
        required = {"episodes", "ori_per_episode", "success_rate_per_episode"}
        if required.issubset(set(npz.files)):
            return (
                np.asarray(npz["episodes"], dtype=float).reshape(-1),
                np.asarray(npz["ori_per_episode"], dtype=float).reshape(-1),
                np.asarray(npz["success_rate_per_episode"], dtype=float).reshape(-1),
                str(path),
            )
    # Fallback: placeholder 2-point trajectory with unknown success values.
    episodes = np.array([0, 1], dtype=float)
    ori_vals = np.array([before_ori, after_ori], dtype=float)
    success_vals = np.array([np.nan, np.nan], dtype=float)
    return episodes, ori_vals, success_vals, "fallback_two_point"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir if args.output_dir else (data_dir / "overrepresentation_index"))
    fig_dir = ensure_dir(out_dir / "figures")
    goal, goal_source = load_goal_position(data_dir, args.goal_file, args.reward_x, args.reward_y)
    centres1, source1 = load_centres_array(data_dir, args.centres_phase1, args.centers_before_pkl)
    centres2, source2 = load_centres_array(data_dir, args.centres_phase2, args.centers_after_pkl)
    f_goal = float(args.goal_area_fraction)
    f_non_goal = float(1.0 - f_goal)
    radius = float(np.sqrt(f_goal / np.pi))
    res1 = compute_ori(centres1, goal, radius, f_goal, f_non_goal)
    res2 = compute_ori(centres2, goal, radius, f_goal, f_non_goal)
    def chi_square_for_result(res: dict[str, float]) -> tuple[float, float]:
        n = int(res["n_total"])
        if n <= 0:
            return np.nan, np.nan
        exp_goal = n * f_goal
        exp_non_goal = n * f_non_goal
        chisq = chisquare(
            f_obs=[res["n_goal"], res["n_non_goal"]],
            f_exp=[exp_goal, exp_non_goal],
        )
        return float(chisq.statistic), float(chisq.pvalue)

    chi1_stat, chi1_p = chi_square_for_result(res1)
    chi2_stat, chi2_p = chi_square_for_result(res2)
    boot1 = bootstrap_ori(centres1, goal, radius, f_goal, f_non_goal, args.n_boot, args.seed)
    boot2 = bootstrap_ori(centres2, goal, radius, f_goal, f_non_goal, args.n_boot, args.seed + 1)
    perm1 = permutation_ori_uniform(int(res1["n_total"]), goal, radius, f_goal, f_non_goal, args.n_permutations, args.seed + 2)
    perm2 = permutation_ori_uniform(int(res2["n_total"]), goal, radius, f_goal, f_non_goal, args.n_permutations, args.seed + 3)
    p_perm1 = float(np.mean(perm1 >= float(res1["ori"])))
    p_perm2 = float(np.mean(perm2 >= float(res2["ori"])))
    ci1 = (float(np.percentile(boot1, 2.5)), float(np.percentile(boot1, 97.5)))
    ci2 = (float(np.percentile(boot2, 2.5)), float(np.percentile(boot2, 97.5)))

    # Optional trajectory arrays.
    episodes, ori_traj, success_traj, traj_source = maybe_load_trajectory(data_dir, args.trajectory_file, float(res1["ori"]), float(res2["ori"]))
    np.savez(
        out_dir / "ori_trajectory.npz",
        episodes=episodes,
        ori_per_episode=ori_traj,
        success_rate_per_episode=success_traj,
    )

    # Figures.
    plot_ori_bar(float(res1["ori"]), float(res2["ori"]), ci1, ci2, fig_dir / "ori_bar_before_after.png")
    plot_ori_trajectory(episodes, ori_traj, success_traj, fig_dir / "ori_trajectory.png")
    plot_field_distribution(centres1, centres2, goal, radius, fig_dir / "field_centre_distribution.png")
    paper_range = [1.5, 3.0]
    qualitative_match = bool(np.isfinite(res2["ori"]) and (paper_range[0] <= float(res2["ori"]) <= paper_range[1]))

    payload: dict[str, Any] = {
        "script": "overrepresentation_index.py",
        "comparison_paper": "Hollup et al. 2001",
        "comparison_metric": "over-representation index",
        "comparison_value_paper": 2.1,
        "comparison_value_paper_note": "Representative value within reported range ~1.5-3.0.",
        "your_value": float(res2["ori"]),
        "your_value_ci_95": [ci2[0], ci2[1]],
        "qualitative_match": qualitative_match,
        "goal": {
            "goal_position": goal.tolist(),
            "goal_position_source": goal_source,
            "goal_zone_radius": radius,
            "f_goal": f_goal,
            "f_non_goal": f_non_goal,
        },
        "phase1": {
            **res1,
            "chi2_stat": chi1_stat,
            "chi2_p": chi1_p,
            "bootstrap_ci_95": [ci1[0], ci1[1]],
            "permutation_p_value": p_perm1,
            "permutation_null_mean": float(np.mean(perm1)),
            "permutation_null_std": float(np.std(perm1, ddof=1)),
        },
        "phase2": {
            **res2,
            "chi2_stat": chi2_stat,
            "chi2_p": chi2_p,
            "bootstrap_ci_95": [ci2[0], ci2[1]],
            "permutation_p_value": p_perm2,
            "permutation_null_mean": float(np.mean(perm2)),
            "permutation_null_std": float(np.std(perm2, ddof=1)),
        },
        "sources": {
            "centres_phase1": source1,
            "centres_phase2": source2,
            "trajectory_source": traj_source,
        },
        "n_boot": int(args.n_boot),
        "n_permutations": int(args.n_permutations),
    }

    save_json(out_dir / "ori_results.json", payload)
    print(f"Saved ORI outputs to: {out_dir}")


if __name__ == "__main__":
    main()
