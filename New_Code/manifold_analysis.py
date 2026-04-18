from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA

from common import (
    DEFAULT_REWARD,
    bin_positions,
    ensure_dir,
    load_behavior_activity,
    maybe_subsample_indices,
    pivot_activity_matrix,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section C1 manifold analysis (PCA/UMAP/displacement).")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/manifold_analysis")
    parser.add_argument("--pv-phase1", type=str, default="pv_phase1.npz")
    parser.add_argument("--pv-phase2", type=str, default="pv_phase2.npz")
    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=float(DEFAULT_REWARD[0]))
    parser.add_argument("--reward-y", type=float, default=float(DEFAULT_REWARD[1]))
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--scree-components", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--min-visits-per-bin", type=int, default=3)
    parser.add_argument("--goal-area-fraction", type=float, default=0.20)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="euclidean")
    parser.add_argument("--umap-max-points", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_goal_position(data_dir: Path, goal_file: str, reward_x: float, reward_y: float) -> tuple[np.ndarray, str]:
    path = data_dir / goal_file
    if path.exists():
        arr = np.asarray(np.load(path), dtype=float).reshape(-1)
        if arr.size >= 2:
            return np.array([float(arr[0]), float(arr[1])], dtype=float), str(path)
    return np.array([reward_x, reward_y], dtype=float), "fallback_args"


def load_phase_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
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


def load_phase_data(data_dir: Path, npz_name: str, period: str) -> tuple[np.ndarray, np.ndarray, str]:
    npz_path = data_dir / npz_name
    loaded = load_phase_from_npz(npz_path)
    if loaded is not None:
        return loaded[0], loaded[1], str(npz_path)

    beh, act, _ = load_behavior_activity(data_dir, period)
    X_df, B_df = pivot_activity_matrix(act, beh)
    if B_df is None:
        raise RuntimeError(f"Could not align behavior and activity for period={period}.")

    positions = B_df[["x", "y"]].to_numpy(dtype=float)
    activations = X_df.to_numpy(dtype=float)
    n = min(positions.shape[0], activations.shape[0])
    return positions[:n], activations[:n], f"fallback_pkl_{period}"


def align_features(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(A.shape[1], B.shape[1])
    return A[:, :n], B[:, :n]


def centered(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0, keepdims=True)
    return X - mu, mu.reshape(-1)


def distance_to_goal(positions: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.sqrt((positions[:, 0] - goal[0]) ** 2 + (positions[:, 1] - goal[1]) ** 2)


def phase2_explained_ratio_same_basis(proj_phase2: np.ndarray, phase2_centered: np.ndarray) -> np.ndarray:
    var_pc = np.var(proj_phase2, axis=0, ddof=1)
    total_var = float(np.sum(np.var(phase2_centered, axis=0, ddof=1)))
    if not np.isfinite(total_var) or total_var <= 0:
        return np.full(var_pc.shape[0], np.nan)
    return var_pc / total_var


def bin_centroids_pc12(
    positions: np.ndarray,
    projected: np.ndarray,
    grid_size: int,
    min_visits_per_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = positions[:, 0]
    y = positions[:, 1]
    xb, yb, flat = bin_positions(x, y, grid_size=grid_size)

    centroids = np.full((grid_size, grid_size, 2), np.nan, dtype=float)
    counts = np.zeros((grid_size, grid_size), dtype=int)

    for i in range(grid_size):
        for j in range(grid_size):
            idx = np.where((xb == i) & (yb == j))[0]
            counts[i, j] = int(idx.size)
            if idx.size < min_visits_per_bin:
                continue
            centroids[i, j] = np.mean(projected[idx, :2], axis=0)

    return centroids, counts


def displacement_map(
    c1: np.ndarray,
    c2: np.ndarray,
    count1: np.ndarray,
    count2: np.ndarray,
    min_visits: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (count1 >= min_visits) & (count2 >= min_visits)
    disp = np.full(c1.shape[:2], np.nan, dtype=float)
    diff = c2 - c1
    mag = np.sqrt(np.sum(diff ** 2, axis=2))
    disp[valid] = mag[valid]
    return disp, valid


def bin_distance_map(grid_size: int, goal: np.ndarray) -> np.ndarray:
    c = (np.arange(grid_size, dtype=float) + 0.5) / grid_size
    xx, yy = np.meshgrid(c, c, indexing="ij")
    return np.sqrt((xx - goal[0]) ** 2 + (yy - goal[1]) ** 2)


def bootstrap_interaction(goal_vals: np.ndarray, non_vals: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if goal_vals.size < 2 or non_vals.size < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    d = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        g = rng.choice(goal_vals, size=goal_vals.size, replace=True)
        n = rng.choice(non_vals, size=non_vals.size, replace=True)
        d[i] = float(np.mean(g) - np.mean(n))

    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def scatter_reward_distance(z: np.ndarray, dist: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6.6, 5.6))
    sc = plt.scatter(z[:, 0], z[:, 1], c=dist, s=9, alpha=0.45, cmap="viridis")
    plt.colorbar(sc, shrink=0.85, label="Distance to reward")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def scatter_x_coordinate(z: np.ndarray, x: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6.6, 5.6))
    sc = plt.scatter(z[:, 0], z[:, 1], c=x, s=9, alpha=0.45, cmap="RdBu_r")
    plt.colorbar(sc, shrink=0.85, label="x position")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_scree(expl1: np.ndarray, expl2: np.ndarray, n_show: int, out_path: Path) -> None:
    n = min(n_show, expl1.shape[0], expl2.shape[0])
    x = np.arange(n)
    w = 0.42

    plt.figure(figsize=(8.2, 4.6))
    plt.bar(x - w / 2, expl1[:n], width=w, alpha=0.85, color="#2E5FA3", label="Phase 1")
    plt.bar(x + w / 2, expl2[:n], width=w, alpha=0.85, color="#E07A2D", label="Phase 2")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA scree plot (Phase 1 basis)")
    plt.xticks(x, [str(i + 1) for i in x], fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pca_overlay(z1: np.ndarray, z2: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7.0, 5.8))
    plt.scatter(z1[:, 0], z1[:, 1], s=8, alpha=0.30, color="#2E5FA3", label="Phase 1")
    plt.scatter(z2[:, 0], z2[:, 1], s=8, alpha=0.30, color="#E07A2D", label="Phase 2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA manifold overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_umap(
    X1: np.ndarray,
    X2: np.ndarray,
    positions1: np.ndarray,
    positions2: np.ndarray,
    goal: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Fit one UMAP embedding on concatenated phases.

    Returns embedding and position-aligned subsets used in that embedding.
    """
    idx1 = np.arange(X1.shape[0])
    idx2 = np.arange(X2.shape[0])

    if args.umap_max_points and args.umap_max_points > 0:
        idx1 = maybe_subsample_indices(X1.shape[0], args.umap_max_points, seed=args.seed)
        idx2 = maybe_subsample_indices(X2.shape[0], args.umap_max_points, seed=args.seed + 1)

    X1u = X1[idx1]
    X2u = X2[idx2]
    P1u = positions1[idx1]
    P2u = positions2[idx2]

    Xcat = np.vstack([X1u, X2u])

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.seed,
        )
        emb = reducer.fit_transform(Xcat)
        source = "umap"
    except Exception:
        # Deterministic fallback preserving comparable coordinates if umap-learn is missing.
        pca2 = PCA(n_components=2, random_state=args.seed)
        emb = pca2.fit_transform(Xcat)
        source = "pca_fallback"

    return emb, P1u, P2u, source


def plot_umap_overlay(emb: np.ndarray, n1: int, out_path: Path, title_suffix: str = "") -> None:
    plt.figure(figsize=(7.0, 5.8))
    plt.scatter(emb[:n1, 0], emb[:n1, 1], s=8, alpha=0.30, color="#2E5FA3", label="Phase 1")
    plt.scatter(emb[n1:, 0], emb[n1:, 1], s=8, alpha=0.30, color="#E07A2D", label="Phase 2")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("UMAP manifold overlay" + title_suffix)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_umap_reward_distance(emb: np.ndarray, n1: int, d1: np.ndarray, d2: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharex=True, sharey=True, constrained_layout=True)

    sc1 = axes[0].scatter(emb[:n1, 0], emb[:n1, 1], c=d1, s=9, alpha=0.45, cmap="viridis")
    axes[0].set_title("Phase 1")
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")

    sc2 = axes[1].scatter(emb[n1:, 0], emb[n1:, 1], c=d2, s=9, alpha=0.45, cmap="viridis")
    axes[1].set_title("Phase 2")
    axes[1].set_xlabel("Dim 1")

    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Distance to reward")

    fig.suptitle("UMAP embedding colored by reward distance")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_displacement_map(disp: np.ndarray, goal: np.ndarray, goal_radius: float, out_path: Path) -> None:
    plt.figure(figsize=(6.2, 5.7))
    ax = plt.gca()
    im = ax.imshow(disp.T, origin="lower", cmap="magma")
    plt.colorbar(im, shrink=0.85, label="|Δ centroid| in PC-space")

    gx = goal[0] * disp.shape[0] - 0.5
    gy = goal[1] * disp.shape[1] - 0.5
    gr = goal_radius * disp.shape[0]
    ax.add_patch(Circle((gx, gy), gr, fill=False, linestyle="--", linewidth=2.0, color="cyan"))

    ax.set_title("Manifold displacement map")
    ax.set_xlabel("x bin")
    ax.set_ylabel("y bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir if args.output_dir else (data_dir / "manifold_analysis"))
    fig_dir = ensure_dir(out_dir / "figures")
    out_data_dir = ensure_dir(out_dir / "data")

    goal, goal_source = load_goal_position(data_dir, args.goal_file, args.reward_x, args.reward_y)
    goal_radius = float(np.sqrt(args.goal_area_fraction / np.pi))

    pos1, act1, source1 = load_phase_data(data_dir, args.pv_phase1, "before")
    pos2, act2, source2 = load_phase_data(data_dir, args.pv_phase2, "after")
    act1, act2 = align_features(act1, act2)

    act1c, mean1 = centered(act1)
    act2c, mean2 = centered(act2)

    n_components = min(args.n_components, act1c.shape[0], act1c.shape[1])
    if n_components < 2:
        raise ValueError("Need at least 2 PCA components for C1 analysis.")

    pca = PCA(n_components=n_components, random_state=args.seed)
    z1 = pca.fit_transform(act1c)
    z2 = pca.transform(act2c)
    evr1 = np.asarray(pca.explained_variance_ratio_, dtype=float)
    evr2 = phase2_explained_ratio_same_basis(z2, act2c)
    np.save(out_data_dir / "pca_projections_phase1.npy", z1)
    np.save(out_data_dir / "pca_projections_phase2.npy", z2)
    with (out_data_dir / "pca_model.pkl").open("wb") as f:
        pickle.dump(pca, f)
    d1 = distance_to_goal(pos1, goal)
    d2 = distance_to_goal(pos2, goal)
    plot_scree(evr1, evr2, n_show=args.scree_components, out_path=fig_dir / "pca_scree_plot.png")
    scatter_reward_distance(z1, d1, "Phase 1 PCA colored by reward distance", fig_dir / "pca_scatter_phase1_reward_dist.png")
    scatter_reward_distance(z2, d2, "Phase 2 PCA colored by reward distance", fig_dir / "pca_scatter_phase2_reward_dist.png")
    plot_pca_overlay(z1, z2, fig_dir / "pca_scatter_overlay.png")

    # Additional x-coordinate views requested in the guide.
    scatter_x_coordinate(z1, pos1[:, 0], "Phase 1 PCA colored by x-position", fig_dir / "pca_scatter_phase1_xpos.png")
    scatter_x_coordinate(z2, pos2[:, 0], "Phase 2 PCA colored by x-position", fig_dir / "pca_scatter_phase2_xpos.png")
    emb, pos1u, pos2u, umap_source = run_umap(act1c, act2c, pos1, pos2, goal, args)
    np.save(out_data_dir / "umap_embedding.npy", emb)

    n1u = pos1u.shape[0]
    plot_umap_overlay(emb, n1u, fig_dir / "umap_scatter_overlay.png", title_suffix="" if umap_source == "umap" else " (fallback)")
    plot_umap_reward_distance(
        emb,
        n1u,
        distance_to_goal(pos1u, goal),
        distance_to_goal(pos2u, goal),
        fig_dir / "umap_scatter_reward_dist.png",
    )

    c1, n1 = bin_centroids_pc12(pos1, z1, grid_size=args.grid_size, min_visits_per_bin=args.min_visits_per_bin)
    c2, n2 = bin_centroids_pc12(pos2, z2, grid_size=args.grid_size, min_visits_per_bin=args.min_visits_per_bin)
    disp_map, valid = displacement_map(c1, c2, n1, n2, min_visits=args.min_visits_per_bin)

    dist_bins = bin_distance_map(args.grid_size, goal)
    valid_disp = disp_map[valid]
    valid_dist = dist_bins[valid]

    if valid_disp.size >= 4:
        rho, pval = spearmanr(valid_disp, valid_dist)
        rho = float(rho)
        pval = float(pval)
    else:
        rho, pval = np.nan, np.nan

    goal_mask = (dist_bins <= goal_radius) & valid
    non_mask = (dist_bins > goal_radius) & valid

    goal_vals = disp_map[goal_mask]
    non_vals = disp_map[non_mask]
    mean_goal = float(np.nanmean(goal_vals)) if goal_vals.size else np.nan
    mean_non = float(np.nanmean(non_vals)) if non_vals.size else np.nan

    if goal_vals.size > 0 and non_vals.size > 0:
        mw = mannwhitneyu(goal_vals, non_vals, alternative="greater")
        mw_stat = float(mw.statistic)
        mw_p = float(mw.pvalue)
    else:
        mw_stat, mw_p = np.nan, np.nan

    interaction = float(mean_goal - mean_non) if np.isfinite(mean_goal) and np.isfinite(mean_non) else np.nan
    ci_low, ci_high = bootstrap_interaction(goal_vals, non_vals, n_boot=args.n_bootstrap, seed=args.seed)

    plot_displacement_map(disp_map, goal, goal_radius, fig_dir / "manifold_displacement_map.png")

    payload: dict[str, Any] = {
        "script": "manifold_analysis.py",
        "comparison_paper": "Section C1 internal benchmark",
        "comparison_metric": "Reward-proximal manifold displacement",
        "comparison_value_paper": None,
        "comparison_value_paper_note": "Directional prediction: larger displacement near reward.",
        "your_value": interaction,
        "your_value_ci_95": [ci_low, ci_high],
        "qualitative_match": bool(np.isfinite(rho) and (rho < 0.0)),
        "goal_position": goal.tolist(),
        "goal_position_source": goal_source,
        "goal_zone_radius": goal_radius,
        "sources": {
            "phase1_data": source1,
            "phase2_data": source2,
            "umap_source": umap_source,
        },
        "pca": {
            "n_components": int(n_components),
            "phase1_mean_activation_shape": list(mean1.shape),
            "phase2_mean_activation_shape": list(mean2.shape),
            "explained_variance_ratio_phase1": evr1.tolist(),
            "explained_variance_ratio_phase2": evr2.tolist(),
        },
        "displacement": {
            "mean_displacement_all_valid": float(np.nanmean(valid_disp)) if valid_disp.size else np.nan,
            "mean_displacement_goal_zone": mean_goal,
            "mean_displacement_non_goal_zone": mean_non,
            "mannwhitney_u_stat": mw_stat,
            "mannwhitney_u_p": mw_p,
            "n_valid_bins": int(np.sum(valid)),
            "n_goal_bins": int(np.sum(goal_mask)),
            "n_non_goal_bins": int(np.sum(non_mask)),
        },
        "spatial_specificity_rho": rho,
        "spatial_specificity_pval": pval,
        "goal_zone_effect_size": mean_goal,
        "non_goal_zone_effect_size": mean_non,
        "interaction_contrast": interaction,
        "interaction_bootstrap_ci_low": ci_low,
        "interaction_bootstrap_ci_high": ci_high,
    }

    save_json(out_dir / "manifold_pca_results.json", payload)
    print(f"Saved Section C1 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
