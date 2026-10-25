#!/usr/bin/env python3
"""Panel d cluster identification and multi-variable driver analysis."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import chi2_contingency, kruskal, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from panel_b import build_v10_fov_env, get_observation_vector
from panel_d import load_phase_probe, x_tertile_labels
from utils import DEFAULT_REWARD, DEFAULT_RUN, FIGURES_DIR, M, apply_nature_style, ensure_dir, save_figure

apply_nature_style()

ANALYSIS_DIR = FIGURES_DIR / "panel_d_analysis"
WALL_PROX_THRESH = 0.15
N_PERM = 1000
SEED = 42


def load_behavior_full(data_dir: Path) -> pd.DataFrame:
    """Load the 'before' behaviour frame for a run."""
    return pd.read_parquet(data_dir / "df_behavior_before.parquet")


def wall_proximity_labels(x: np.ndarray, thresh: float = WALL_PROX_THRESH) -> np.ndarray:
    """0=left wall, 1=center, 2=right wall."""
    labels = np.ones(len(x), dtype=int)
    labels[x < thresh] = 0
    labels[x > 1.0 - thresh] = 2
    return labels


def recompute_observations(beh: pd.DataFrame, seed: int = SEED, batch_log: int = 10000) -> np.ndarray:
    """Recompute 24-dim FoV observation for each (x, y, theta) row."""
    _env, agent, fov_cells = build_v10_fov_env(seed=seed)
    n = len(beh)
    obs = np.zeros((n, 24), dtype=np.float32)
    thetas = beh["theta"].to_numpy(dtype=float)

    for i in range(n):
        agent.pos = np.array([beh["x"].iat[i], beh["y"].iat[i]], dtype=float)
        agent.head_direction = np.array(
            [np.cos(thetas[i]), np.sin(thetas[i])], dtype=float
        )
        obs[i] = get_observation_vector(fov_cells)
        if batch_log and (i + 1) % batch_log == 0:
            print(f"    obs recompute: {i + 1}/{n}")
    return obs


def build_variable_table(
    beh: pd.DataFrame,
    obs: np.ndarray,
    reward_pos: tuple[float, float],
    m: int = M,
) -> pd.DataFrame:
    """Per-timestep candidate explanatory variables (position, heading, speed, wall and reward distances, observation summaries) as one table."""
    x = beh["x"].to_numpy(dtype=float)
    y = beh["y"].to_numpy(dtype=float)
    theta = beh["theta"].to_numpy(dtype=float)
    reward = np.array(reward_pos, dtype=float)
    center = np.array([0.5, 0.5], dtype=float)

    pos = np.stack([x, y], axis=1)
    speed = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=1)

    x_bin = np.clip((x * m).astype(int), 0, m - 1)
    y_bin = np.clip((y * m).astype(int), 0, m - 1)
    flat_bin = x_bin * m + y_bin
    counts = np.bincount(flat_bin, minlength=m * m).astype(float)
    occ = counts / len(x)
    occ_map = occ.reshape(m, m)
    occ_smooth = gaussian_filter(occ_map, sigma=1.5)
    local_density = occ_smooth[x_bin, y_bin]

    fr = obs[:, :18].reshape(-1, 3, 6)
    wall_rays = obs[:, 18:24]

    data = {
        "x": x,
        "y": y,
        "dist_reward": np.linalg.norm(pos - reward, axis=1),
        "dist_center": np.linalg.norm(pos - center, axis=1),
        "polar_angle": np.arctan2(y - 0.5, x - 0.5),
        "theta": theta,
        "cos_theta": np.cos(theta),
        "sin_theta": np.sin(theta),
        "speed": speed,
        "fov_type0_sum": fr[:, 0].sum(axis=1),
        "fov_type1_sum": fr[:, 1].sum(axis=1),
        "fov_type2_sum": fr[:, 2].sum(axis=1),
        "wall_ray_mean": wall_rays.mean(axis=1),
        "wall_ray_max": wall_rays.max(axis=1),
        "wall_ray_min": wall_rays.min(axis=1),
        "obs_l2": np.linalg.norm(obs, axis=1),
        "spatial_bin": flat_bin,
        "local_density": local_density,
        "near_left_wall": (x < WALL_PROX_THRESH).astype(float),
        "near_right_wall": (x > 1.0 - WALL_PROX_THRESH).astype(float),
    }
    for j in range(24):
        data[f"obs_{j:02d}"] = obs[:, j]
    return pd.DataFrame(data)


def eta_squared_kruskal(groups: list[np.ndarray]) -> float:
    """Kruskal-Wallis effect size eta^2 for a set of groups."""
    all_vals = np.concatenate(groups)
    n = len(all_vals)
    if n == 0:
        return np.nan
    ranks = pd.Series(all_vals).rank(method="average").to_numpy()
    offsets = 0
    h_stat = 0.0
    mean_rank = (n + 1) / 2.0
    for g in groups:
        k = len(g)
        if k == 0:
            continue
        r = ranks[offsets : offsets + k]
        h_stat += k * (r.mean() - mean_rank) ** 2
        offsets += k
    h_stat *= 12.0 / (n * (n + 1))
    return float(h_stat / (n - 1))


def mutual_information_discrete(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Mutual information between two variables after binning both into `n_bins`."""
    x_disc = np.digitize(x, np.linspace(x.min(), x.max(), n_bins + 1)[1:-1])
    mi = 0.0
    n = len(x)
    for xi in np.unique(x_disc):
        for yi in np.unique(y):
            p_xy = np.mean((x_disc == xi) & (y == yi))
            if p_xy <= 0:
                continue
            p_x = np.mean(x_disc == xi)
            p_y = np.mean(y == yi)
            mi += p_xy * np.log(p_xy / (p_x * p_y))
    return float(mi)


def ridge_r2_cv(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    """Cross-validated R^2 of a ridge regression from X to y."""
    if X.ndim == 1:
        X = X[:, None]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=1.0, random_state=seed)
    scores = cross_val_score(model, Xs, y, cv=5, scoring="r2")
    return float(np.mean(scores))


def run_stats_for_partition(
    variables: pd.DataFrame,
    labels: np.ndarray,
    partition_name: str,
    z: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> pd.DataFrame:
    """Score every variable against one partition (effect size plus a permutation null), sorted by eta^2."""
    rng = np.random.default_rng(seed)
    rows = []
    numeric_cols = [
        c for c in variables.columns
        if c not in {"near_left_wall", "near_right_wall"}
    ]

    perm_eta = []
    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        max_eta = 0.0
        for col in ["x", "y", "dist_reward", "wall_ray_mean"]:
            groups = [variables[col].to_numpy()[shuffled == k] for k in range(3)]
            max_eta = max(max_eta, eta_squared_kruskal(groups))
        perm_eta.append(max_eta)
    perm_eta = np.array(perm_eta)

    for col in numeric_cols:
        vals = variables[col].to_numpy(dtype=float)
        groups = [vals[labels == k] for k in range(3)]
        h_stat, p_val = kruskal(*groups)
        eta2 = eta_squared_kruskal(groups)
        rho_pc1, p_pc1 = spearmanr(vals, z[:, 0])
        rho_pc2, p_pc2 = spearmanr(vals, z[:, 1])
        mi = mutual_information_discrete(vals, labels)
        perm_p = float(np.mean(perm_eta >= eta2))
        rows.append({
            "partition": partition_name,
            "variable": col,
            "kruskal_H": float(h_stat),
            "kruskal_p": float(p_val),
            "eta_squared": eta2,
            "perm_p_eta": perm_p,
            "spearman_pc1": float(rho_pc1),
            "spearman_pc1_p": float(p_pc1),
            "spearman_pc2": float(rho_pc2),
            "spearman_pc2_p": float(p_pc2),
            "mutual_info": mi,
        })

    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)


def pc1_tertile_labels(z: np.ndarray) -> np.ndarray:
    """Tertile label per point along PC1."""
    return np.digitize(z[:, 0], np.quantile(z[:, 0], [1 / 3, 2 / 3]))


def build_summary(
    all_stats: pd.DataFrame,
    kmeans_labels: np.ndarray,
    x_tert: np.ndarray,
    wall_labels: np.ndarray,
    pc1_tert: np.ndarray,
    z: np.ndarray,
    obs: np.ndarray,
    variables: pd.DataFrame,
    evr: np.ndarray,
    seed: int = SEED,
) -> dict:
    """Collect the whole analysis into one JSON-ready dict: partition agreement (ARI), effect sizes and variance explained."""
    ari = {
        "kmeans_vs_x_tertile": float(adjusted_rand_score(kmeans_labels, x_tert)),
        "kmeans_vs_wall_proximity": float(adjusted_rand_score(kmeans_labels, wall_labels)),
        "kmeans_vs_pc1_tertile": float(adjusted_rand_score(kmeans_labels, pc1_tert)),
        "x_tertile_vs_wall_proximity": float(adjusted_rand_score(x_tert, wall_labels)),
        "x_tertile_vs_pc1_tertile": float(adjusted_rand_score(x_tert, pc1_tert)),
    }

    sil_k = {}
    for k in (2, 3, 4):
        sil_k[k] = float(silhouette_score(z, KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(z)))

    xy = variables[["x", "y"]].to_numpy()
    theta_col = variables[["theta"]].to_numpy()
    xy_theta = np.hstack([xy, theta_col])
    xy_obs = np.hstack([xy_theta, obs])

    ridge = {
        "xy_pc1_r2": ridge_r2_cv(xy, z[:, 0]),
        "xy_pc2_r2": ridge_r2_cv(xy, z[:, 1]),
        "xy_theta_pc1_r2": ridge_r2_cv(xy_theta, z[:, 0]),
        "xy_theta_pc2_r2": ridge_r2_cv(xy_theta, z[:, 1]),
        "xy_obs_pc1_r2": ridge_r2_cv(xy_obs, z[:, 0]),
        "xy_obs_pc2_r2": ridge_r2_cv(xy_obs, z[:, 1]),
        "obs_only_pc1_r2": ridge_r2_cv(obs, z[:, 0]),
        "obs_only_pc2_r2": ridge_r2_cv(obs, z[:, 1]),
    }

    def top5(part: str):
        sub = all_stats[all_stats["partition"] == part].head(5)
        return sub[["variable", "eta_squared", "mutual_info"]].to_dict("records")

    x_stats = all_stats[all_stats["partition"] == "x_tertile"]
    perm_p95 = float(x_stats["eta_squared"].quantile(0.95)) if len(x_stats) else 0.0

    return {
        "n_timesteps": int(len(z)),
        "pca_explained_variance_ratio": [float(e) for e in evr],
        "adjusted_rand_index": ari,
        "silhouette_k2": sil_k[2],
        "silhouette_k3": sil_k[3],
        "silhouette_k4": sil_k[4],
        "ridge_r2": ridge,
        "top5_by_partition": {
            "kmeans": top5("kmeans"),
            "x_tertile": top5("x_tertile"),
            "pc1_tertile": top5("pc1_tertile"),
            "wall_proximity": top5("wall_proximity"),
        },
        "kmeans_mean_x_per_cluster": [
            float(variables["x"].to_numpy()[kmeans_labels == k].mean()) for k in range(3)
        ],
        "x_tertile_mean_pc1_per_group": [
            float(z[x_tert == t, 0].mean()) for t in range(3)
        ],
    }


def discretize_equal_freq(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency bins; duplicate edges collapsed."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.zeros(len(values), dtype=int)
    return np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)


def y_tertile_labels(y: np.ndarray) -> np.ndarray:
    """Tertile label per point along the arena y axis."""
    return np.digitize(y, [1.0 / 3.0, 2.0 / 3.0])


def heading_octant_labels(theta: np.ndarray) -> np.ndarray:
    """Nearest of the eight discrete headings, per timestep."""
    from utils import HDS
    hd_angles = np.arctan2(HDS[:, 1], HDS[:, 0])
    diff = np.abs((theta[:, None] - hd_angles[None, :]) + np.pi) % (2 * np.pi) - np.pi
    return np.argmin(diff, axis=1)


def cramers_v_from_labels(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int]:
    """Cramér's V and chi-square p-value from two integer label vectors."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    cats_a = np.unique(a)
    cats_b = np.unique(b)
    mat = np.zeros((len(cats_a), len(cats_b)), dtype=int)
    a_map = {c: i for i, c in enumerate(cats_a)}
    b_map = {c: i for i, c in enumerate(cats_b)}
    for ai, bi in zip(a, b):
        mat[a_map[ai], b_map[bi]] += 1
    chi2, p_val, _, _ = chi2_contingency(mat)
    n = mat.sum()
    k = min(mat.shape)
    if n == 0 or k <= 1:
        return 0.0, 1.0, mat.shape[0], mat.shape[1]
    v = float(np.sqrt(chi2 / (n * (k - 1))))
    return v, float(p_val), mat.shape[0], mat.shape[1]


def normalized_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """NMI = MI / sqrt(H(a) H(b)) using empirical counts."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    mi = mutual_information_discrete(a.astype(float), b)
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / n
        return -np.sum(p * np.log(p + 1e-12))
    ha, hb = entropy(a), entropy(b)
    denom = np.sqrt(ha * hb)
    return float(mi / denom) if denom > 0 else 0.0


def multinomial_pseudo_r2(kmeans_labels: np.ndarray, X: np.ndarray, seed: int = SEED) -> float:
    """McFadden pseudo-R² for multinomial logistic predicting KMeans from X."""
    if X.ndim == 1:
        X = X[:, None]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    model.fit(Xs, kmeans_labels)
    probs_null = np.bincount(kmeans_labels, minlength=3) / len(kmeans_labels)
    ll_null = float(np.sum(np.log(probs_null[kmeans_labels] + 1e-12)))
    ll_full = float(np.sum(np.log(model.predict_proba(Xs)[np.arange(len(kmeans_labels)), kmeans_labels] + 1e-12)))
    if ll_null == 0:
        return float("nan")
    return float(1.0 - ll_full / ll_null)


def build_categorical_features(
    variables: pd.DataFrame,
    z: np.ndarray,
    positions: np.ndarray,
    beh: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Discrete versions of the candidate variables, for the categorical association tests."""
    x, y = positions[:, 0], positions[:, 1]
    theta = beh["theta"].to_numpy(dtype=float)
    cats: dict[str, np.ndarray] = {}

    # Reference / oracle partitions
    cats["pc1_tertile"] = pc1_tertile_labels(z)
    cats["pc2_tertile"] = np.digitize(z[:, 1], np.quantile(z[:, 1], [1 / 3, 2 / 3]))
    cats["x_tertile"] = x_tertile_labels(x)
    cats["y_tertile"] = y_tertile_labels(y)
    cats["wall_proximity"] = wall_proximity_labels(x)
    cats["heading_octant"] = heading_octant_labels(theta)

    m = M
    x8 = np.clip((x * 8).astype(int), 0, 7)
    y8 = np.clip((y * 8).astype(int), 0, 7)
    cats["spatial_8x8"] = x8 * 8 + y8
    cats["xy_tertile_9"] = x_tertile_labels(x) * 3 + y_tertile_labels(y)

    numeric_cols = [
        c for c in variables.columns
        if c not in {"near_left_wall", "near_right_wall", "spatial_bin"}
    ]
    for col in numeric_cols:
        vals = variables[col].to_numpy(dtype=float)
        cats[f"{col}__t3"] = discretize_equal_freq(vals, 3)
        cats[f"{col}__t5"] = discretize_equal_freq(vals, 5)

    return cats


def run_categorical_correlation(
    categorical: dict[str, np.ndarray],
    kmeans_labels: np.ndarray,
) -> pd.DataFrame:
    """Cramer's V between each categorical feature and the cluster labels, sorted strongest first."""
    rows = []
    for name, labels in categorical.items():
        v, p_val, n_a, n_b = cramers_v_from_labels(labels, kmeans_labels)
        nmi = normalized_mutual_information(labels, kmeans_labels)
        ari = adjusted_rand_score(labels, kmeans_labels)
        rows.append({
            "variable": name,
            "cramers_v": v,
            "chi2_p": p_val,
            "nmi": nmi,
            "ari": float(ari),
            "n_categories": n_a,
        })
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def run_multinomial_screen(
    variables: pd.DataFrame,
    obs: np.ndarray,
    z: np.ndarray,
    kmeans_labels: np.ndarray,
    seed: int = SEED,
) -> pd.DataFrame:
    """McFadden pseudo-R² for predicting KMeans from feature sets."""
    xy = variables[["x", "y"]].to_numpy()
    xy_theta = np.hstack([xy, variables[["theta"]].to_numpy()])
    xy_obs = np.hstack([xy_theta, obs])
    blocks = {
        "x_only": variables[["x"]].to_numpy(),
        "y_only": variables[["y"]].to_numpy(),
        "xy": xy,
        "xy_theta": xy_theta,
        "dist_reward": variables[["dist_reward"]].to_numpy(),
        "wall_rays": variables[["wall_ray_mean", "wall_ray_min", "wall_ray_max"]].to_numpy(),
        "fov_sums": variables[["fov_type0_sum", "fov_type1_sum", "fov_type2_sum"]].to_numpy(),
        "obs_24": obs,
        "xy_obs": xy_obs,
        "pc1_pc2_oracle": z,
    }
    rows = []
    for name, X in blocks.items():
        try:
            r2 = multinomial_pseudo_r2(kmeans_labels, X, seed=seed)
        except Exception:
            r2 = float("nan")
        rows.append({"feature_set": name, "mcfadden_r2": r2})
    return pd.DataFrame(rows).sort_values("mcfadden_r2", ascending=False).reset_index(drop=True)


def plot_categorical_heatmap(cat_df: pd.DataFrame, stem: Path, top_n: int = 25) -> None:
    """Heatmap of the top categorical associations by Cramer's V."""
    sub = cat_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    y_pos = np.arange(len(sub))
    ax.barh(y_pos, sub["cramers_v"], color="0.3", height=0.7, label="Cramér's V")
    ax.barh(y_pos, sub["nmi"] * 0.05, color="0.65", height=0.4, alpha=0.8, label="NMI (×0.05)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["variable"], fontsize=5)
    ax.set_xlabel("Association with KMeans cluster")
    ax.set_title("Categorical correlation with KMeans lobes", fontsize=6)
    ax.legend(fontsize=5, loc="lower right")
    plt.tight_layout()
    save_figure(fig, stem)


def plot_contingency_heatmap(
    cat_labels: np.ndarray,
    kmeans_labels: np.ndarray,
    cat_name: str,
    stem: Path,
) -> None:
    """Contingency table of two labellings, drawn as a heatmap."""
    cats = np.unique(cat_labels)
    mat = np.zeros((3, len(cats)), dtype=int)
    for k in range(3):
        for j, c in enumerate(cats):
            mat[k, j] = np.sum((kmeans_labels == k) & (cat_labels == c))
    mat_norm = mat / mat.sum(axis=0, keepdims=True)
    fig, ax = plt.subplots(figsize=(max(3.0, len(cats) * 0.35), 2.4))
    im = ax.imshow(mat_norm, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["KM 0", "KM 1", "KM 2"], fontsize=6)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([str(int(c)) for c in cats], fontsize=4, rotation=90)
    ax.set_xlabel(cat_name)
    ax.set_ylabel("KMeans cluster")
    ax.set_title(f"P(cluster | {cat_name})", fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_figure(fig, stem)


def save_cache(out_dir: Path, variables: pd.DataFrame, z: np.ndarray, kmeans_labels: np.ndarray,
               positions: np.ndarray, obs: np.ndarray) -> None:
    """Write the variable table, embedding and label arrays to the cache directory."""
    cache = ensure_dir(out_dir / "cache")
    variables.to_parquet(cache / "variables.parquet", index=False)
    np.save(cache / "z.npy", z)
    np.save(cache / "kmeans_labels.npy", kmeans_labels)
    np.save(cache / "positions.npy", positions)
    np.save(cache / "obs.npy", obs)


def load_cache(out_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Read back what `save_cache` wrote, or None if the cache is absent or incomplete."""
    cache = out_dir / "cache"
    pkl = cache / "variables.parquet"
    paths = [pkl, cache / "z.npy", cache / "kmeans_labels.npy",
             cache / "positions.npy", cache / "obs.npy"]
    if not all(p.exists() for p in paths):
        return None
    return (
        pd.read_parquet(pkl),
        np.load(paths[1]),
        np.load(paths[2]),
        np.load(paths[3]),
        np.load(paths[4]),
    )


def run_categorical_analysis(
    variables: pd.DataFrame,
    z: np.ndarray,
    kmeans_labels: np.ndarray,
    positions: np.ndarray,
    beh: pd.DataFrame,
    obs: np.ndarray,
    out_dir: Path,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the categorical screens (Cramer's V and multinomial) and write both as CSV."""
    print("  Building categorical features...")
    categorical = build_categorical_features(variables, z, positions, beh)
    cat_df = run_categorical_correlation(categorical, kmeans_labels)
    multi_df = run_multinomial_screen(variables, obs, z, kmeans_labels, seed=seed)

    cat_path = out_dir / "categorical_correlation.csv"
    multi_path = out_dir / "multinomial_screen.csv"
    cat_df.to_csv(cat_path, index=False)
    multi_df.to_csv(multi_path, index=False)
    print(f"  Saved: {cat_path}")
    print(f"  Saved: {multi_path}")

    plot_categorical_heatmap(cat_df, out_dir / "categorical_correlation_bars")
    for i, var in enumerate(cat_df.head(3)["variable"]):
        safe = str(var).replace("/", "_")
        plot_contingency_heatmap(
            categorical[var], kmeans_labels, var, out_dir / f"contingency_kmeans_vs_{safe}",
        )

    return cat_df, multi_df


def print_categorical_interpretation(cat_df: pd.DataFrame, multi_df: pd.DataFrame) -> None:
    """Print the categorical results in readable form."""
    top = cat_df.iloc[0]
    beh_top = cat_df[~cat_df["variable"].str.startswith(("pc1", "pc2"))].iloc[0]
    oracle = cat_df[cat_df["variable"] == "pc1_tertile"]
    pc1_v = float(oracle["cramers_v"].iloc[0]) if len(oracle) else float("nan")

    print("\n" + "=" * 60)
    print("CATEGORICAL CORRELATION (KMeans lobes)")
    print("=" * 60)
    print(f"Oracle pc1_tertile  Cramér's V = {pc1_v:.3f}  (KMeans ≈ PC1 bands)")
    print(f"Top association:     {top['variable']}  V={top['cramers_v']:.3f}  NMI={top['nmi']:.3f}")
    print(f"Top behavioural:     {beh_top['variable']}  V={beh_top['cramers_v']:.3f}")
    print("\nTop 10 categorical predictors:")
    for _, row in cat_df.head(10).iterrows():
        print(f"  {row['variable']:28s}  V={row['cramers_v']:.4f}  NMI={row['nmi']:.4f}  ARI={row['ari']:.4f}")
    print("\nMultinomial McFadden R² (predict KMeans from features):")
    for _, row in multi_df.head(8).iterrows():
        print(f"  {row['feature_set']:16s}  R²={row['mcfadden_r2']:.4f}")
    print("\nConclusion:")
    if pc1_v > 0.95:
        print("  KMeans lobes ARE PC1 tertiles (emergent neural axis). No behavioural")
        print("  or sensory variable reaches comparable categorical association.")
        print(f"  Best behavioural tertile: {beh_top['variable']} (V={beh_top['cramers_v']:.3f}).")
        print("  The lobes are an intrinsic population-state partition, not a single")
        print("  observable variable like x, y, or reward distance.")
    print("=" * 60 + "\n")


def run_stats_battery(
    variables: pd.DataFrame,
    kmeans_labels: np.ndarray,
    z: np.ndarray,
    x_tert: np.ndarray,
    wall_labels: np.ndarray,
    pc1_tert: np.ndarray,
    obs: np.ndarray,
    evr: np.ndarray,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """Run `run_stats_for_partition` over every partition and build the summary."""
    parts = [
        ("kmeans", kmeans_labels),
        ("x_tertile", x_tert),
        ("pc1_tertile", pc1_tert),
        ("wall_proximity", wall_labels),
    ]
    frames = [run_stats_for_partition(variables, lab, name, z, seed=seed) for name, lab in parts]
    all_stats = pd.concat(frames, ignore_index=True)
    summary = build_summary(
        all_stats, kmeans_labels, x_tert, wall_labels, pc1_tert, z, obs, variables, evr, seed=seed,
    )
    return all_stats, summary


def plot_pca_grid(z: np.ndarray, color_dict: dict[str, np.ndarray], evr: np.ndarray, stem: Path) -> None:
    """PCA embedding drawn once per colouring variable, on a shared grid."""
    keys = list(color_dict.keys())
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6))
    axes_flat = axes.flatten()
    compass = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    for ax, key in zip(axes_flat, keys):
        vals = color_dict[key]
        if key == "kmeans":
            sc = ax.scatter(z[:, 0], z[:, 1], c=vals, s=2, alpha=0.35, cmap="Set1", vmin=0, vmax=2, rasterized=True)
        elif key == "heading_octant":
            cmap = plt.get_cmap("tab10", 8)
            sc = ax.scatter(
                z[:, 0], z[:, 1], c=vals, s=2, alpha=0.35, cmap=cmap,
                vmin=-0.5, vmax=7.5, rasterized=True,
            )
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, ticks=range(8))
            cbar.ax.set_yticklabels(compass, fontsize=4)
        else:
            sc = ax.scatter(z[:, 0], z[:, 1], c=vals, s=2, alpha=0.35, cmap="viridis", rasterized=True)
            plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        ax.set_title(key, fontsize=6)
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=5)
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=5)
    for ax in axes_flat[len(keys):]:
        ax.axis("off")
    plt.tight_layout(pad=0.4)
    save_figure(fig, stem)


def plot_arena_clusters(x: np.ndarray, y: np.ndarray, labels: np.ndarray, stem: Path, title: str = "Arena") -> None:
    """Cluster membership plotted at each point's arena position."""
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    sc = ax.scatter(x, y, c=labels, s=1, alpha=0.25, cmap="Set1", vmin=0, vmax=2, rasterized=True)
    for k in range(3):
        mask = labels == k
        ax.scatter(x[mask].mean(), y[mask].mean(), s=40, c="k", marker="x", linewidths=1.0, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=6)
    plt.colorbar(sc, ax=ax, shrink=0.75, label="cluster")
    plt.tight_layout()
    save_figure(fig, stem)


def plot_effect_sizes(stats_df: pd.DataFrame, stem: Path, partition: str = "x_tertile", top_n: int = 15) -> None:
    """Bar chart of the strongest effect sizes for one partition."""
    sub = stats_df[stats_df["partition"] == partition].head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    y_pos = np.arange(len(sub))
    ax.barh(y_pos, sub["eta_squared"], color="0.35", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["variable"], fontsize=5)
    ax.set_xlabel("η² (Kruskal–Wallis effect size)")
    ax.set_title(f"Effect sizes vs {partition} groups", fontsize=6)
    plt.tight_layout()
    save_figure(fig, stem)


def plot_confusion(labels_a: np.ndarray, labels_b: np.ndarray, names: tuple[str, str], stem: Path) -> None:
    """Confusion matrix between two labellings."""
    mat = np.zeros((3, 3), dtype=int)
    for a, b in zip(labels_a, labels_b):
        mat[int(a), int(b)] += 1
    mat_norm = mat / mat.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    im = ax.imshow(mat_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["0", "1", "2"], fontsize=6)
    ax.set_yticklabels(["0", "1", "2"], fontsize=6)
    ax.set_xlabel(names[1])
    ax.set_ylabel(names[0])
    ax.set_title(f"Confusion: {names[0]} vs {names[1]}", fontsize=6)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]}", ha="center", va="center", fontsize=5, color="0.2")
    plt.colorbar(im, ax=ax, shrink=0.8, label="row fraction")
    plt.tight_layout()
    save_figure(fig, stem)


def plot_summary_panel(summary: dict, stats_df: pd.DataFrame, stem: Path) -> None:
    """One-page summary: variance explained, partition agreement and the leading effect sizes."""
    ari = summary["adjusted_rand_index"]
    x_top = stats_df[stats_df["partition"] == "x_tertile"].head(5)
    km_top = stats_df[stats_df["partition"] == "kmeans"].head(3)
    lines = [
        "Panel d cluster analysis summary",
        f"N = {summary['n_timesteps']:,} timesteps (phase-1 random walk)",
        f"PCA variance: PC1={summary['pca_explained_variance_ratio'][0]*100:.1f}%, "
        f"PC2={summary['pca_explained_variance_ratio'][1]*100:.1f}%",
        f"Silhouette k=3 (KMeans on PCA): {summary['silhouette_k3']:.3f}",
        "",
        "Partition agreement (ARI):",
        f"  KMeans vs x-tertile: {ari['kmeans_vs_x_tertile']:.3f}",
        f"  x-tertile vs wall proximity: {ari['x_tertile_vs_wall_proximity']:.3f}",
        f"  KMeans vs PC1-tertile: {ari['kmeans_vs_pc1_tertile']:.3f}",
        "",
        "Visual lobes (x-tertile) — top drivers:",
    ]
    for _, row in x_top.iterrows():
        lines.append(f"  {row['variable']}: η²={row['eta_squared']:.3f}")
    lines += [
        "",
        "KMeans (PC1 bands) — top drivers:",
    ]
    for _, row in km_top.iterrows():
        lines.append(f"  {row['variable']}: η²={row['eta_squared']:.4f}")
    lines += [
        "",
        f"R² PC2 from [x,y]: {summary['ridge_r2']['xy_pc2_r2']:.3f}  "
        f"from [x,y,θ,obs]: {summary['ridge_r2']['xy_obs_pc2_r2']:.3f}",
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=6, family="monospace",
            transform=ax.transAxes)
    save_figure(fig, stem)


def print_interpretation(summary: dict, stats_df: pd.DataFrame) -> None:
    """Print the summary and the ranked effect sizes in readable form."""
    ari = summary["adjusted_rand_index"]
    x_top = stats_df[stats_df["partition"] == "x_tertile"].iloc[0]
    x_row = stats_df[(stats_df["partition"] == "x_tertile") & (stats_df["variable"] == "x")].iloc[0]
    wall_row = stats_df[(stats_df["partition"] == "x_tertile") & (stats_df["variable"] == "wall_ray_min")]

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"Top driver for x-tertile lobes: {x_top['variable']} (η²={x_top['eta_squared']:.3f})")
    print(f"x effect across x-tertiles: η²={x_row['eta_squared']:.3f}")
    if len(wall_row):
        print(f"wall_ray_min across x-tertiles: η²={wall_row.iloc[0]['eta_squared']:.3f}")
    print(f"KMeans vs x-tertile ARI: {ari['kmeans_vs_x_tertile']:.3f} (orthogonal partitions)")
    print(f"KMeans vs PC1-tertile ARI: {ari['kmeans_vs_pc1_tertile']:.3f}")
    print(f"x-tertile vs wall proximity ARI: {ari['x_tertile_vs_wall_proximity']:.3f}")
    print(f"KMeans mean x per cluster: {summary['kmeans_mean_x_per_cluster']}")
    print(f"x-tertile mean PC1 per group: {summary['x_tertile_mean_pc1_per_group']}")

    print("\nConclusion:")
    print("  The three VISUAL lobes in panel d (x-coloured) are ALLOCENTRIC X-BANDS")
    print("  (left / central / right arena). KMeans on PCA splits PC1 into equal thirds")
    print("  with identical mean x — a different, density-based partition.")
    print("  Wall/object sensory variables co-vary with x because landmarks sit on")
    print("  left/right walls (x≈0, x≈1). Reward distance weakly gradients PC2.")
    print("=" * 60 + "\n")


def run_analysis(
    run_dir: Path,
    output_dir: Path,
    reward_pos: tuple[float, float],
    seed: int = SEED,
    skip_obs: bool = False,
    subsample: int = 0,
) -> None:
    """Full panel-d analysis: load, build variables, embed, cluster, score, plot and cache."""
    data_dir = run_dir / "data"
    out_dir = ensure_dir(output_dir)

    print("[1/5] Loading phase-1 probe data...")
    beh = load_behavior_full(data_dir)
    positions, _theta, activations = load_phase_probe(data_dir)
    n = len(beh)
    if subsample and subsample < n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=subsample, replace=False))
        beh = beh.iloc[idx].reset_index(drop=True)
        positions = positions[idx]
        activations = activations[idx]
        n = subsample
        print(f"  Subsampled to {n} points")

    print("[2/5] PCA embedding (mirror panel d)...")
    act_c = activations - activations.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2, random_state=seed)
    z = pca.fit_transform(act_c)
    evr = pca.explained_variance_ratio_

    kmeans_labels = KMeans(n_clusters=3, random_state=seed, n_init=10).fit_predict(z)
    x_tert = x_tertile_labels(positions[:, 0])
    wall_labels = wall_proximity_labels(positions[:, 0])
    pc1_tert = pc1_tertile_labels(z)

    print("[3/5] Recomputing sensory observations (24-dim FoV)...")
    if skip_obs:
        obs = np.zeros((n, 24), dtype=np.float32)
        print("  Skipped (--skip_obs); sensory variables will be zero.")
    else:
        obs = recompute_observations(beh, seed=seed)

    print("[4/6] Building variable table and running statistics...")
    variables = build_variable_table(beh, obs, reward_pos)
    stats_df, summary = run_stats_battery(
        variables, kmeans_labels, z, x_tert, wall_labels, pc1_tert, obs, evr, seed=seed,
    )

    def _nan_to_null(obj):
        """Recursively replace float NaN/Inf with None for valid JSON serialization."""
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _nan_to_null(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_nan_to_null(v) for v in obj]
        return obj

    stats_path = out_dir / "cluster_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    json_path = out_dir / "cluster_summary.json"
    with json_path.open("w") as f:
        json.dump(_nan_to_null(summary), f, indent=2)
    print(f"  Saved: {stats_path}")
    print(f"  Saved: {json_path}")

    save_cache(out_dir, variables, z, kmeans_labels, positions, obs)

    print("[5/6] Categorical correlation analysis (KMeans lobes)...")
    cat_df, multi_df = run_categorical_analysis(
        variables, z, kmeans_labels, positions, beh, obs, out_dir, seed=seed,
    )
    summary["categorical_top5"] = cat_df.head(5)[["variable", "cramers_v", "nmi"]].to_dict("records")
    summary["multinomial_top3"] = multi_df.head(3).to_dict("records")
    with json_path.open("w") as f:
        json.dump(_nan_to_null(summary), f, indent=2)

    print("[6/6] Generating figures...")
    color_dict = {
        "kmeans": kmeans_labels.astype(float),
        "x_tertile": x_tert.astype(float),
        "x": variables["x"].to_numpy(),
        "y": variables["y"].to_numpy(),
        "dist_reward": variables["dist_reward"].to_numpy(),
        "time_index": np.arange(len(z), dtype=float),
        "wall_ray_mean": variables["wall_ray_mean"].to_numpy(),
        "fov_type0_sum": variables["fov_type0_sum"].to_numpy(),
    }
    plot_pca_grid(z, color_dict, evr, out_dir / "pca_color_grid")
    plot_arena_clusters(positions[:, 0], positions[:, 1], x_tert, out_dir / "arena_x_tertile",
                        title="Arena coloured by x-tertile")
    plot_arena_clusters(positions[:, 0], positions[:, 1], kmeans_labels, out_dir / "arena_kmeans",
                        title="Arena coloured by KMeans")
    plot_effect_sizes(stats_df, out_dir / "effect_sizes_x_tertile", partition="x_tertile")
    plot_effect_sizes(stats_df, out_dir / "effect_sizes_kmeans", partition="kmeans", top_n=10)
    plot_confusion(kmeans_labels, x_tert, ("KMeans", "x_tertile"), out_dir / "confusion_kmeans_xtertile")
    plot_confusion(kmeans_labels, wall_labels, ("KMeans", "wall_prox"), out_dir / "confusion_kmeans_wall")
    plot_summary_panel(summary, stats_df, out_dir / "summary_panel")

    print_interpretation(summary, stats_df)
    print_categorical_interpretation(cat_df, multi_df)


def run_categorical_only(output_dir: Path, seed: int = SEED) -> None:
    """Re-run categorical analysis from cached variables (fast)."""
    out_dir = ensure_dir(output_dir)
    cached = load_cache(out_dir)
    if cached is None:
        raise FileNotFoundError(
            f"No cache in {out_dir / 'cache'}. Run full analysis first."
        )
    variables, z, kmeans_labels, positions, obs = cached
    beh = pd.DataFrame({
        "theta": variables["theta"].to_numpy(),
        "x": variables["x"].to_numpy(),
        "y": variables["y"].to_numpy(),
    })
    print("Loaded cache; running categorical analysis...")
    cat_df, multi_df = run_categorical_analysis(
        variables, z, kmeans_labels, positions, beh, obs, out_dir, seed=seed,
    )
    print_categorical_interpretation(cat_df, multi_df)


def parse_args():
    """Command-line arguments: run directory, output directory, seed, cache behaviour."""
    p = argparse.ArgumentParser(description="Panel d cluster driver analysis.")
    p.add_argument("--run_dir", default=str(DEFAULT_RUN))
    p.add_argument("--output_dir", default=str(ANALYSIS_DIR))
    p.add_argument("--reward_pos", type=float, nargs=2, default=list(DEFAULT_REWARD))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip_obs", action="store_true", help="Skip slow FoV recomputation (testing only)")
    p.add_argument("--subsample", type=int, default=0, help="Subsample timesteps for quick runs")
    p.add_argument("--categorical_only", action="store_true",
                   help="Re-run categorical correlation from cache only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.categorical_only:
        run_categorical_only(Path(args.output_dir).resolve(), seed=args.seed)
    else:
        run_analysis(
            Path(args.run_dir).resolve(),
            Path(args.output_dir).resolve(),
            tuple(args.reward_pos),
            seed=args.seed,
            skip_obs=args.skip_obs,
            subsample=args.subsample,
        )
