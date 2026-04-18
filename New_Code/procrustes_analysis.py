from __future__ import annotations
import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from common import DEFAULT_REWARD, bin_positions, ensure_dir, load_behavior_activity, pivot_activity_matrix, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section C4 Procrustes manifold analysis.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/procrustes_analysis")
    parser.add_argument("--pv-phase1", type=str, default="pv_phase1.npz")
    parser.add_argument("--pv-phase2", type=str, default="pv_phase2.npz")
    parser.add_argument("--pca-model", type=str, default="")
    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=float(DEFAULT_REWARD[0]))
    parser.add_argument("--reward-y", type=float, default=float(DEFAULT_REWARD[1]))
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--min-visits", type=int, default=3)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--sensitivity-min", type=int, default=5)
    parser.add_argument("--sensitivity-max", type=int, default=20)
    parser.add_argument("--goal-area-fraction", type=float, default=0.20)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--learning-curves", type=str, default="learning_curves.npz")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_goal_position(data_dir: Path, goal_file: str, reward_x: float, reward_y: float) -> tuple[np.ndarray, str]:
    p = data_dir / goal_file
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float).reshape(-1)
        if arr.size >= 2:
            return np.array([float(arr[0]), float(arr[1])], dtype=float), str(p)
    return np.array([reward_x, reward_y], dtype=float), "fallback_args"


def load_phase_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    npz = np.load(path)
    if "positions" not in npz.files or "activations" not in npz.files:
        return None

    pos = np.asarray(npz["positions"], dtype=float)
    act = np.asarray(npz["activations"], dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 2 or act.ndim != 2:
        return None

    n = min(pos.shape[0], act.shape[0])
    return pos[:n, :2], act[:n]


def load_phase_data(data_dir: Path, npz_name: str, period: str) -> tuple[np.ndarray, np.ndarray, str]:
    p = data_dir / npz_name
    loaded = load_phase_from_npz(p)
    if loaded is not None:
        return loaded[0], loaded[1], str(p)

    beh, act_df, _ = load_behavior_activity(data_dir, period)
    X_df, B_df = pivot_activity_matrix(act_df, beh)
    if B_df is None:
        raise RuntimeError(f"Could not align behavior and activity for period={period}.")

    pos = B_df[["x", "y"]].to_numpy(dtype=float)
    act = X_df.to_numpy(dtype=float)
    n = min(pos.shape[0], act.shape[0])
    return pos[:n], act[:n], f"fallback_pkl_{period}"


def align_features(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(A.shape[1], B.shape[1])
    return A[:, :n], B[:, :n]


def centered(X: np.ndarray) -> np.ndarray:
    return X - np.mean(X, axis=0, keepdims=True)


def fit_or_load_pca(X1c: np.ndarray, pca_model_path: str, n_fit: int) -> tuple[PCA, str]:
    if pca_model_path:
        p = Path(pca_model_path)
        if p.exists():
            with p.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, PCA):
                return obj, str(p)

    n_comp = min(n_fit, X1c.shape[0], X1c.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full").fit(X1c)
    return pca, "fitted_from_phase1"


def build_bin_pc_matrix(
    positions: np.ndarray,
    projected: np.ndarray,
    grid_size: int,
    min_visits: int,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    K = grid_size * grid_size
    mat = np.full((K, n_components), np.nan, dtype=float)
    counts = np.zeros(K, dtype=int)

    xb, yb, flat = bin_positions(positions[:, 0], positions[:, 1], grid_size=grid_size)

    for b in range(K):
        idx = np.where(flat == b)[0]
        counts[b] = int(idx.size)
        if idx.size < min_visits:
            continue
        mat[b] = np.mean(projected[idx, :n_components], axis=0)

    return mat, counts


def bin_centers(grid_size: int) -> np.ndarray:
    K = grid_size * grid_size
    c = np.zeros((K, 2), dtype=float)
    for b in range(K):
        i = b // grid_size
        j = b % grid_size
        c[b] = np.array([(i + 0.5) / grid_size, (j + 0.5) / grid_size], dtype=float)
    return c


def permutation_null_procrustes(M1: np.ndarray, M2: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(n_perm, dtype=float)

    for i in range(n_perm):
        perm = rng.permutation(M2.shape[0])
        _, _, d = procrustes(M1, M2[perm])
        out[i] = float(d)

    return out


def bootstrap_interaction(goal_vals: np.ndarray, non_vals: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    g = goal_vals[np.isfinite(goal_vals)]
    n = non_vals[np.isfinite(non_vals)]
    if g.size < 2 or n.size < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    d = np.zeros(n_boot, dtype=float)

    for i in range(n_boot):
        gb = rng.choice(g, size=g.size, replace=True)
        nb = rng.choice(n, size=n.size, replace=True)
        d[i] = float(np.mean(gb) - np.mean(nb))

    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def extract_episode_from_name(name: str) -> int | None:
    patterns = [
        r"episode[_-]?(\d+)",
        r"ep[_-]?(\d+)",
        r"checkpoint[_-]?(\d+)",
        r"ckpt[_-]?(\d+)",
    ]
    low = name.lower()
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            return int(m.group(1))

    nums = re.findall(r"(\d+)", low)
    if nums:
        return int(nums[-1])
    return None


def find_checkpoint_npz_files(data_dir: Path) -> list[tuple[int, Path]]:
    files = []
    for p in data_dir.rglob("*.npz"):
        if p.name in {"pv_phase1.npz", "pv_phase2.npz", "ori_trajectory.npz", "learning_curves.npz"}:
            continue
        name_low = p.name.lower()
        if ("checkpoint" in name_low) or ("ckpt" in name_low) or ("episode" in name_low):
            ep = extract_episode_from_name(p.name)
            if ep is not None:
                files.append((ep, p))

    files = sorted(files, key=lambda x: x[0])
    uniq: dict[int, Path] = {}
    for ep, p in files:
        uniq[ep] = p
    return sorted([(k, v) for k, v in uniq.items()], key=lambda x: x[0])


def load_success_curve(data_dir: Path, learning_curves_name: str) -> tuple[np.ndarray, str]:
    cands = [
        data_dir / learning_curves_name,
        data_dir / "learning_curves.npz",
        data_dir.parent / learning_curves_name,
        data_dir.parent / "learning_curves.npz",
    ]

    for p in cands:
        if not p.exists() or p.suffix.lower() != ".npz":
            continue
        npz = np.load(p)
        if "rl_success" in npz.files:
            return np.asarray(npz["rl_success"], dtype=float).reshape(-1), str(p)
    return np.array([], dtype=float), "missing"


def compute_disparity_for_components(
    M1_full: np.ndarray,
    M2_full: np.ndarray,
    n_components: int,
) -> float:
    c = min(n_components, M1_full.shape[1], M2_full.shape[1])
    _, _, d = procrustes(M1_full[:, :c], M2_full[:, :c])
    return float(d)


def plot_residual_map(residual_map: np.ndarray, goal: np.ndarray, goal_radius: float, out_path: Path) -> None:
    plt.figure(figsize=(6.2, 5.7))
    ax = plt.gca()
    im = ax.imshow(residual_map.T, origin="lower", cmap="magma")
    plt.colorbar(im, shrink=0.85, label="Procrustes residual")

    gx = goal[0] * residual_map.shape[0] - 0.5
    gy = goal[1] * residual_map.shape[1] - 0.5
    gr = goal_radius * residual_map.shape[0]
    ax.add_patch(Circle((gx, gy), gr, fill=False, linestyle="--", linewidth=2.0, color="cyan"))

    ax.set_title("Procrustes residual map")
    ax.set_xlabel("x bin")
    ax.set_ylabel("y bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_trajectory(episodes: np.ndarray, disparity: np.ndarray, success: np.ndarray, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax1.plot(episodes, disparity, color="#2E5FA3", linewidth=2.2, marker="o", label="Disparity")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Procrustes disparity", color="#2E5FA3")
    ax1.tick_params(axis="y", labelcolor="#2E5FA3")

    if np.any(np.isfinite(success)):
        ax2 = ax1.twinx()
        ax2.plot(episodes, success, color="#1B8A5A", linewidth=2.0, alpha=0.9, label="Success")
        ax2.set_ylabel("Success rate", color="#1B8A5A")
        ax2.tick_params(axis="y", labelcolor="#1B8A5A")

    ax1.set_title("Procrustes disparity trajectory")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_dimension_bar(dim_var: np.ndarray, out_path: Path) -> None:
    x = np.arange(1, dim_var.shape[0] + 1)
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(x, dim_var, color="#E07A2D", alpha=0.9)
    plt.xlabel("PC dimension")
    plt.ylabel("Residual variance")
    plt.title("Per-dimension residual variance")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir if args.output_dir else (data_dir / "procrustes_analysis"))
    fig_dir = ensure_dir(out_dir / "figures")
    out_data_dir = ensure_dir(out_dir / "data")
    goal, goal_source = load_goal_position(data_dir, args.goal_file, args.reward_x, args.reward_y)
    goal_radius = float(np.sqrt(args.goal_area_fraction / np.pi))
    pos1, act1, src1 = load_phase_data(data_dir, args.pv_phase1, "before")
    pos2, act2, src2 = load_phase_data(data_dir, args.pv_phase2, "after")
    act1, act2 = align_features(act1, act2)
    X1c = centered(act1)
    X2c = centered(act2)
    n_fit = max(args.n_components, args.sensitivity_max)
    pca, pca_source = fit_or_load_pca(X1c, args.pca_model, n_fit=n_fit)
    Z1 = pca.transform(X1c)
    Z2 = pca.transform(X2c)
    n_components = min(args.n_components, Z1.shape[1], Z2.shape[1])
    if n_components < 2:
        raise ValueError("Need at least 2 dimensions for Procrustes.")
    B1, c1 = build_bin_pc_matrix(pos1, Z1, args.grid_size, args.min_visits, n_components)
    B2, c2 = build_bin_pc_matrix(pos2, Z2, args.grid_size, args.min_visits, n_components)
    valid = (c1 >= args.min_visits) & (c2 >= args.min_visits)
    valid_idx = np.where(valid)[0]
    if valid_idx.size < 8:
        raise ValueError("Too few valid bins for Procrustes; adjust --min-visits or check data.")
    M1 = B1[valid_idx]
    M2 = B2[valid_idx]

    m1_std, m2_tx, disparity = procrustes(M1, M2)
    disparity = float(disparity)

    np.save(out_data_dir / "procrustes_m1_standardised.npy", m1_std)
    np.save(out_data_dir / "procrustes_m2_transformed.npy", m2_tx)

    null = permutation_null_procrustes(M1, M2, n_perm=args.n_permutations, seed=args.seed)
    p_less_equal = float(np.mean(null <= disparity))
    p_greater_equal = float(np.mean(null >= disparity))
    residual = np.linalg.norm(m1_std - m2_tx, axis=1)
    centers = bin_centers(args.grid_size)
    centers_valid = centers[valid_idx]
    dist_valid = np.sqrt((centers_valid[:, 0] - goal[0]) ** 2 + (centers_valid[:, 1] - goal[1]) ** 2)

    if residual.size >= 4:
        rho, p_rho = spearmanr(residual, dist_valid)
        rho = float(rho)
        p_rho = float(p_rho)
    else:
        rho, p_rho = np.nan, np.nan

    goal_mask = dist_valid <= goal_radius
    non_mask = ~goal_mask
    goal_vals = residual[goal_mask]
    non_vals = residual[non_mask]
    goal_effect = float(np.nanmean(goal_vals)) if goal_vals.size else np.nan
    non_effect = float(np.nanmean(non_vals)) if non_vals.size else np.nan
    interaction = goal_effect - non_effect if np.isfinite(goal_effect) and np.isfinite(non_effect) else np.nan
    ci_low, ci_high = bootstrap_interaction(goal_vals, non_vals, n_boot=args.n_bootstrap, seed=args.seed + 1)
    residual_map = np.full(args.grid_size * args.grid_size, np.nan, dtype=float)
    residual_map[valid_idx] = residual
    residual_map = residual_map.reshape(args.grid_size, args.grid_size)

    diff_dim = m1_std - m2_tx
    dim_var = np.var(diff_dim, axis=0, ddof=1)

    # Sensitivity: disparity vs number of PCs.
    sens_min = max(2, args.sensitivity_min)
    sens_max = min(args.sensitivity_max, Z1.shape[1], Z2.shape[1])
    sens_components = list(range(sens_min, sens_max + 1))
    sens_disparity = []
    B1_full, c1_full = build_bin_pc_matrix(pos1, Z1, args.grid_size, args.min_visits, sens_max)
    B2_full, c2_full = build_bin_pc_matrix(pos2, Z2, args.grid_size, args.min_visits, sens_max)
    valid_full = (c1_full >= args.min_visits) & (c2_full >= args.min_visits)
    M1_full = B1_full[valid_full]
    M2_full = B2_full[valid_full]

    if M1_full.shape[0] >= 8:
        for c in sens_components:
            sens_disparity.append(compute_disparity_for_components(M1_full, M2_full, n_components=c))

    # Trajectory over checkpoints.
    ckpts = find_checkpoint_npz_files(data_dir)
    ep_list: list[float] = []
    d_list: list[float] = []

    for ep, path in ckpts:
        loaded = load_phase_from_npz(path)
        if loaded is None:
            continue
        pos_ck, act_ck = loaded
        act_ck = act_ck[:, :act1.shape[1]]
        Xck = centered(act_ck)
        Zck = pca.transform(Xck)[:, :n_components]
        Bck, cck = build_bin_pc_matrix(pos_ck, Zck, args.grid_size, args.min_visits, n_components)
        valid_ck = (c1 >= args.min_visits) & (cck >= args.min_visits)
        idx_ck = np.where(valid_ck)[0]
        if idx_ck.size < 8:
            continue

        M1ck = B1[idx_ck]
        Mck = Bck[idx_ck]
        _, _, d_ck = procrustes(M1ck, Mck)
        ep_list.append(float(ep))
        d_list.append(float(d_ck))

    if len(ep_list) == 0:
        ep_arr = np.array([0.0, 1.0], dtype=float)
        disp_arr = np.array([0.0, disparity], dtype=float)
        success_arr = np.array([np.nan, np.nan], dtype=float)
        trajectory_source = "fallback_two_point"
    else:
        ep_arr = np.asarray(ep_list, dtype=float)
        disp_arr = np.asarray(d_list, dtype=float)
        success_curve, success_source = load_success_curve(data_dir, args.learning_curves)
        if success_curve.size > 0:
            idx = np.clip(ep_arr.astype(int), 0, success_curve.size - 1)
            success_arr = success_curve[idx]
        else:
            success_arr = np.full(ep_arr.shape, np.nan, dtype=float)
        trajectory_source = f"checkpoints:{len(ep_arr)}|success:{success_source}"
    plot_residual_map(residual_map, goal, goal_radius, fig_dir / "procrustes_residual_map.png")
    plot_trajectory(ep_arr, disp_arr, success_arr, fig_dir / "procrustes_disparity_trajectory.png")
    plot_dimension_bar(dim_var, fig_dir / "procrustes_dimension_bar.png")

    payload: dict[str, Any] = {
        "script": "procrustes_analysis.py",
        "comparison_paper": "Section C4 internal benchmark",
        "comparison_metric": "Residual shape difference after rigid alignment",
        "comparison_value_paper": None,
        "comparison_value_paper_note": "Directional benchmark: higher residual near reward.",
        "your_value": disparity,
        "your_value_ci_95": [np.nan, np.nan],
        "qualitative_match": bool(np.isfinite(rho) and (rho < 0.0)),
        "goal_position": goal.tolist(),
        "goal_position_source": goal_source,
        "goal_zone_radius": goal_radius,
        "sources": {
            "phase1_data": src1,
            "phase2_data": src2,
            "pca_source": pca_source,
            "trajectory_source": trajectory_source,
        },
        "procrustes": {
            "disparity": disparity,
            "null_mean": float(np.mean(null)) if null.size else np.nan,
            "null_std": float(np.std(null, ddof=1)) if null.size > 1 else np.nan,
            "null_95th_percentile": float(np.percentile(null, 95)) if null.size else np.nan,
            "permutation_p_less_equal": p_less_equal,
            "permutation_p_greater_equal": p_greater_equal,
            "n_permutations": int(args.n_permutations),
            "n_components": int(n_components),
            "n_valid_bins": int(valid_idx.size),
        },
        "spatial_test": {
            "spearman_rho_residual_vs_reward_distance": rho,
            "spearman_p_residual_vs_reward_distance": p_rho,
        },
        "per_dimension_residual_variance": dim_var.tolist(),
        "sensitivity": {
            "components": sens_components,
            "disparity": sens_disparity,
        },
        "trajectory": {
            "episodes": ep_arr.tolist(),
            "disparity": disp_arr.tolist(),
            "success_rate": success_arr.tolist(),
        },
        "spatial_specificity_rho": rho,
        "spatial_specificity_pval": p_rho,
        "goal_zone_effect_size": goal_effect,
        "non_goal_zone_effect_size": non_effect,
        "interaction_contrast": interaction,
        "interaction_bootstrap_ci_low": ci_low,
        "interaction_bootstrap_ci_high": ci_high,
    }

    save_json(out_dir / "procrustes_results.json", payload)
    print(f"Saved Section C4 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
