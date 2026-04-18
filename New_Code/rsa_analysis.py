from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

from common import DEFAULT_REWARD, bin_positions, ensure_dir, load_behavior_activity, pivot_activity_matrix, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section C3 representational similarity analysis.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/rsa_analysis")
    parser.add_argument("--pv-phase1", type=str, default="pv_phase1.npz")
    parser.add_argument("--pv-phase2", type=str, default="pv_phase2.npz")
    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=float(DEFAULT_REWARD[0]))
    parser.add_argument("--reward-y", type=float, default=float(DEFAULT_REWARD[1]))
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--min-visits", type=int, default=3)
    parser.add_argument("--goal-area-fraction", type=float, default=0.20)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-pairs", type=int, default=5000)
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


def build_population_response(
    positions: np.ndarray,
    activations: np.ndarray,
    grid_size: int,
    min_visits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K = grid_size * grid_size
    N = activations.shape[1]

    xb, yb, flat = bin_positions(positions[:, 0], positions[:, 1], grid_size=grid_size)
    R = np.full((K, N), np.nan, dtype=float)
    counts = np.zeros(K, dtype=int)

    for b in range(K):
        idx = np.where(flat == b)[0]
        counts[b] = int(idx.size)
        if idx.size < min_visits:
            continue
        R[b] = np.mean(activations[idx], axis=0)

    centers = np.zeros((K, 2), dtype=float)
    for b in range(K):
        i = b // grid_size
        j = b % grid_size
        centers[b, 0] = (i + 0.5) / grid_size
        centers[b, 1] = (j + 0.5) / grid_size

    valid = counts >= min_visits
    return R, valid, counts, centers


def upper_values(mat: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    tri = np.triu_indices(mat.shape[0], k=1)
    return mat[tri], tri


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 5:
        return np.nan, np.nan
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def partial_spearman(neural: np.ndarray, reward: np.ndarray, spatial: np.ndarray) -> tuple[float, float, str]:
    df = pd.DataFrame({"neural": neural, "reward": reward, "spatial": spatial}).dropna()
    if df.shape[0] < 10:
        return np.nan, np.nan, "insufficient_data"

    try:
        import pingouin as pg  # type: ignore

        res = pg.partial_corr(data=df, x="neural", y="reward", covar="spatial", method="spearman")
        return float(res["r"].iloc[0]), float(res["p-val"].iloc[0]), "pingouin"
    except Exception:
        xr = df["neural"].rank(method="average").to_numpy(dtype=float)
        yr = df["reward"].rank(method="average").to_numpy(dtype=float)
        cr = df["spatial"].rank(method="average").to_numpy(dtype=float)

        X = np.column_stack([np.ones(cr.shape[0]), cr])
        bx, *_ = np.linalg.lstsq(X, xr, rcond=None)
        by, *_ = np.linalg.lstsq(X, yr, rcond=None)

        rx = xr - X @ bx
        ry = yr - X @ by

        rho, p = spearmanr(rx, ry)
        return float(rho), float(p), "rank_residual_fallback"


def permutation_diff_test(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    reward_rdm: np.ndarray,
    n_perm: int,
    seed: int,
) -> tuple[float, np.ndarray, float]:
    tri = np.triu_indices(rdm1.shape[0], k=1)
    reward_vals = reward_rdm[tri]

    rho1, _ = safe_spearman(rdm1[tri], reward_vals)
    rho2, _ = safe_spearman(rdm2[tri], reward_vals)
    observed = rho2 - rho1

    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm, dtype=float)

    for i in range(n_perm):
        perm = rng.permutation(rdm1.shape[0])
        rdm1p = rdm1[perm][:, perm]
        rp, _ = safe_spearman(rdm1p[tri], reward_vals)
        null[i] = rho2 - rp

    p = float(np.mean(null >= observed))
    return float(observed), null, p


def bootstrap_rho_ci(x: np.ndarray, y: np.ndarray, n_boot: int, n_pairs: int, seed: int) -> tuple[tuple[float, float], np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 10:
        return (np.nan, np.nan), np.array([], dtype=float)

    rng = np.random.default_rng(seed)
    m = x.size
    k = min(max(50, n_pairs), m)
    vals = np.zeros(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.choice(m, size=k, replace=True)
        vals[i], _ = safe_spearman(x[idx], y[idx])

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan), vals
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))), vals


def bootstrap_interaction(goal_delta: np.ndarray, non_delta: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    g = goal_delta[np.isfinite(goal_delta)]
    n = non_delta[np.isfinite(non_delta)]
    if g.size < 2 or n.size < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    out = np.zeros(n_boot, dtype=float)

    for i in range(n_boot):
        gb = rng.choice(g, size=g.size, replace=True)
        nb = rng.choice(n, size=n.size, replace=True)
        out[i] = float(np.mean(gb) - np.mean(nb))

    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def plot_heatmaps(rdm1: np.ndarray, rdm2: np.ndarray, spatial_rdm: np.ndarray, reward_rdm: np.ndarray, out_path: Path) -> None:
    mats = [rdm1, rdm2, spatial_rdm, reward_rdm]
    vmin = float(np.nanmin([np.nanmin(m) for m in mats]))
    vmax = float(np.nanmax([np.nanmax(m) for m in mats]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
    titles = ["Neural RDM Phase 1", "Neural RDM Phase 2", "Spatial model RDM", "Reward-proximity model RDM"]

    for ax, m, t in zip(axes.ravel(), mats, titles):
        im = ax.imshow(m, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(t)
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Bin index")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("Dissimilarity")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_rsa_bar(values: dict[str, float], cis: dict[str, tuple[float, float]], out_path: Path) -> None:
    labels = ["Spatial P1", "Spatial P2", "Reward P1", "Reward P2"]
    vals = [values["spatial1"], values["spatial2"], values["reward1"], values["reward2"]]

    lows = [
        vals[0] - cis["spatial1"][0],
        vals[1] - cis["spatial2"][0],
        vals[2] - cis["reward1"][0],
        vals[3] - cis["reward2"][0],
    ]
    highs = [
        cis["spatial1"][1] - vals[0],
        cis["spatial2"][1] - vals[1],
        cis["reward1"][1] - vals[2],
        cis["reward2"][1] - vals[3],
    ]

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(labels, vals, yerr=[lows, highs], capsize=5, color=["#2E5FA3", "#E07A2D", "#2E5FA3", "#E07A2D"], alpha=0.9)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    plt.ylabel("Spearman rho")
    plt.title("RSA summary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir if args.output_dir else (data_dir / "rsa_analysis"))
    fig_dir = ensure_dir(out_dir / "figures")
    out_data_dir = ensure_dir(out_dir / "data")

    goal, goal_source = load_goal_position(data_dir, args.goal_file, args.reward_x, args.reward_y)
    goal_radius = float(np.sqrt(args.goal_area_fraction / np.pi))

    pos1, act1, src1 = load_phase_data(data_dir, args.pv_phase1, "before")
    pos2, act2, src2 = load_phase_data(data_dir, args.pv_phase2, "after")
    act1, act2 = align_features(act1, act2)

    R1, valid1, _, centers = build_population_response(pos1, act1, args.grid_size, args.min_visits)
    R2, valid2, _, _ = build_population_response(pos2, act2, args.grid_size, args.min_visits)

    common_valid = valid1 & valid2
    valid_idx = np.where(common_valid)[0]

    if valid_idx.size < 8:
        raise ValueError("Too few valid bins for RSA; adjust --min-visits or check data coverage.")

    R1v = R1[valid_idx]
    R2v = R2[valid_idx]
    C = centers[valid_idx]

    rdm1 = pairwise_distances(R1v, metric="correlation")
    rdm2 = pairwise_distances(R2v, metric="correlation")

    spatial_rdm = pairwise_distances(C, metric="euclidean")
    dist_goal = np.sqrt((C[:, 0] - goal[0]) ** 2 + (C[:, 1] - goal[1]) ** 2)
    reward_rdm = np.abs(dist_goal[:, None] - dist_goal[None, :])

    tri = np.triu_indices(rdm1.shape[0], k=1)
    n1_vals = rdm1[tri]
    n2_vals = rdm2[tri]
    s_vals = spatial_rdm[tri]
    r_vals = reward_rdm[tri]

    rho_sp1, p_sp1 = safe_spearman(n1_vals, s_vals)
    rho_sp2, p_sp2 = safe_spearman(n2_vals, s_vals)
    rho_rw1, p_rw1 = safe_spearman(n1_vals, r_vals)
    rho_rw2, p_rw2 = safe_spearman(n2_vals, r_vals)

    prho1, pp1, pm1 = partial_spearman(n1_vals, r_vals, s_vals)
    prho2, pp2, pm2 = partial_spearman(n2_vals, r_vals, s_vals)

    obs_diff, null_diff, perm_p = permutation_diff_test(rdm1, rdm2, reward_rdm, n_perm=args.n_permutations, seed=args.seed)

    ci_sp1, _ = bootstrap_rho_ci(n1_vals, s_vals, args.n_bootstrap, args.bootstrap_pairs, args.seed + 10)
    ci_sp2, _ = bootstrap_rho_ci(n2_vals, s_vals, args.n_bootstrap, args.bootstrap_pairs, args.seed + 11)
    ci_rw1, _ = bootstrap_rho_ci(n1_vals, r_vals, args.n_bootstrap, args.bootstrap_pairs, args.seed + 12)
    ci_rw2, _ = bootstrap_rho_ci(n2_vals, r_vals, args.n_bootstrap, args.bootstrap_pairs, args.seed + 13)

    delta = n2_vals - n1_vals
    near = dist_goal <= goal_radius
    far = ~near
    near_pair = near[tri[0]] & near[tri[1]]
    far_pair = far[tri[0]] & far[tri[1]]

    goal_delta = delta[near_pair]
    non_delta = delta[far_pair]

    goal_effect = float(np.nanmean(goal_delta)) if goal_delta.size else np.nan
    non_effect = float(np.nanmean(non_delta)) if non_delta.size else np.nan
    interaction = goal_effect - non_effect if np.isfinite(goal_effect) and np.isfinite(non_effect) else np.nan
    ci_inter_low, ci_inter_high = bootstrap_interaction(goal_delta, non_delta, args.n_bootstrap, args.seed + 20)

    np.save(out_data_dir / "rdm_phase1.npy", rdm1)
    np.save(out_data_dir / "rdm_phase2.npy", rdm2)
    np.save(out_data_dir / "spatial_rdm.npy", spatial_rdm)

    plot_heatmaps(rdm1, rdm2, spatial_rdm, reward_rdm, fig_dir / "rdm_heatmaps.png")
    plot_rsa_bar(
        values={"spatial1": rho_sp1, "spatial2": rho_sp2, "reward1": rho_rw1, "reward2": rho_rw2},
        cis={"spatial1": ci_sp1, "spatial2": ci_sp2, "reward1": ci_rw1, "reward2": ci_rw2},
        out_path=fig_dir / "rsa_bar_summary.png",
    )

    payload: dict[str, Any] = {
        "script": "rsa_analysis.py",
        "comparison_paper": "Kriegeskorte et al. 2008",
        "comparison_metric": "Reward-model RSA increase after RL",
        "comparison_value_paper": None,
        "comparison_value_paper_note": "Directional benchmark: reward-model RSA should increase post-RL.",
        "your_value": obs_diff,
        "your_value_ci_95": [np.nan, np.nan],
        "qualitative_match": bool(np.isfinite(obs_diff) and (obs_diff > 0.0)),
        "goal_position": goal.tolist(),
        "goal_position_source": goal_source,
        "goal_zone_radius": goal_radius,
        "sources": {
            "phase1_data": src1,
            "phase2_data": src2,
        },
        "rsa_spatial_phase1": {"rho": rho_sp1, "p": p_sp1, "ci_95": [ci_sp1[0], ci_sp1[1]]},
        "rsa_spatial_phase2": {"rho": rho_sp2, "p": p_sp2, "ci_95": [ci_sp2[0], ci_sp2[1]]},
        "rsa_reward_phase1": {"rho": rho_rw1, "p": p_rw1, "ci_95": [ci_rw1[0], ci_rw1[1]]},
        "rsa_reward_phase2": {"rho": rho_rw2, "p": p_rw2, "ci_95": [ci_rw2[0], ci_rw2[1]]},
        "partial_rsa_reward_phase1": {"rho": prho1, "p": pp1, "method": pm1},
        "partial_rsa_reward_phase2": {"rho": prho2, "p": pp2, "method": pm2},
        "phase_difference_reward_rsa": {
            "observed_diff": obs_diff,
            "permutation_p": perm_p,
            "null_mean": float(np.nanmean(null_diff)) if null_diff.size else np.nan,
            "null_std": float(np.nanstd(null_diff, ddof=1)) if null_diff.size > 1 else np.nan,
            "n_permutations": int(args.n_permutations),
        },
        "n_valid_bins": int(valid_idx.size),
        "spatial_specificity_rho": rho_rw2,
        "spatial_specificity_pval": p_rw2,
        "goal_zone_effect_size": goal_effect,
        "non_goal_zone_effect_size": non_effect,
        "interaction_contrast": interaction,
        "interaction_bootstrap_ci_low": ci_inter_low,
        "interaction_bootstrap_ci_high": ci_inter_high,
    }

    save_json(out_dir / "rsa_results.json", payload)
    print(f"Saved Section C3 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
