from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from common import DEFAULT_REWARD, ensure_dir, load_behavior_activity, pivot_activity_matrix, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section C2 dimensionality metrics (PR and D_eff).")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/dimensionality_analysis")

    parser.add_argument("--pv-phase1", type=str, default="pv_phase1.npz")
    parser.add_argument("--pv-phase2", type=str, default="pv_phase2.npz")
    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=float(DEFAULT_REWARD[0]))
    parser.add_argument("--reward-y", type=float, default=float(DEFAULT_REWARD[1]))

    parser.add_argument("--goal-area-fraction", type=float, default=0.20)
    parser.add_argument("--n-bootstrap", type=int, default=200)
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


def pr_deff_from_eigs(eigs: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(eigs, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 1e-12)]
    if vals.size == 0:
        return np.nan, np.nan

    s1 = float(np.sum(vals))
    s2 = float(np.sum(vals ** 2))
    pr = (s1 ** 2) / s2 if s2 > 0 else np.nan

    p = vals / s1
    deff = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    return float(pr), deff


def fit_pca_full(X: np.ndarray, max_components: int = 128) -> PCA:
    n_comp = min(X.shape[0], X.shape[1], max_components)
    return PCA(n_components=n_comp, svd_solver="randomized", random_state=42).fit(X)


def global_metrics_phase1_basis(X1c: np.ndarray, X2c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca1 = fit_pca_full(X1c)
    eig1 = np.asarray(pca1.explained_variance_, dtype=float)
    proj2 = pca1.transform(X2c)
    eig2 = np.var(proj2, axis=0, ddof=1)
    return eig1, eig2, np.asarray(pca1.explained_variance_ratio_, dtype=float)


def local_pr(X: np.ndarray, mask: np.ndarray) -> float:
    Xs = X[mask]
    if Xs.shape[0] < 5:
        return np.nan
    pca = fit_pca_full(Xs)
    pr, _ = pr_deff_from_eigs(np.asarray(pca.explained_variance_, dtype=float))
    return pr


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def distance_to_goal(pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.sqrt((pos[:, 0] - goal[0]) ** 2 + (pos[:, 1] - goal[1]) ** 2)


def bootstrap_global(
    X1c: np.ndarray,
    X2c: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {
        "pr1": np.full(n_boot, np.nan, dtype=float),
        "pr2": np.full(n_boot, np.nan, dtype=float),
        "deff1": np.full(n_boot, np.nan, dtype=float),
        "deff2": np.full(n_boot, np.nan, dtype=float),
        "pr_diff": np.full(n_boot, np.nan, dtype=float),
        "deff_diff": np.full(n_boot, np.nan, dtype=float),
    }

    n1 = X1c.shape[0]
    n2 = X2c.shape[0]

    for i in tqdm(range(n_boot), desc="Bootstrap (global PR)", unit="iter"):
        idx1 = rng.choice(n1, size=n1, replace=True)
        idx2 = rng.choice(n2, size=n2, replace=True)

        X1b = X1c[idx1]
        X2b = X2c[idx2]

        eig1, eig2, _ = global_metrics_phase1_basis(X1b, X2b)
        pr1, deff1 = pr_deff_from_eigs(eig1)
        pr2, deff2 = pr_deff_from_eigs(eig2)

        out["pr1"][i] = pr1
        out["pr2"][i] = pr2
        out["deff1"][i] = deff1
        out["deff2"][i] = deff2
        out["pr_diff"][i] = pr2 - pr1
        out["deff_diff"][i] = deff2 - deff1

    return out


def bootstrap_local_interaction(
    X1c: np.ndarray,
    X2c: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    goal_radius: float,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inter = np.full(n_boot, np.nan, dtype=float)
    goal_eff = np.full(n_boot, np.nan, dtype=float)
    non_eff = np.full(n_boot, np.nan, dtype=float)

    n1 = X1c.shape[0]
    n2 = X2c.shape[0]

    for i in tqdm(range(n_boot), desc="Bootstrap (local PR)", unit="iter"):
        idx1 = rng.choice(n1, size=n1, replace=True)
        idx2 = rng.choice(n2, size=n2, replace=True)

        X1b = X1c[idx1]
        X2b = X2c[idx2]
        d1b = d1[idx1]
        d2b = d2[idx2]

        g1 = d1b <= goal_radius
        g2 = d2b <= goal_radius
        ng1 = ~g1
        ng2 = ~g2

        pr_g1 = local_pr(X1b, g1)
        pr_g2 = local_pr(X2b, g2)
        pr_ng1 = local_pr(X1b, ng1)
        pr_ng2 = local_pr(X2b, ng2)

        ge = pr_g2 - pr_g1
        ne = pr_ng2 - pr_ng1

        goal_eff[i] = ge
        non_eff[i] = ne
        inter[i] = ge - ne

    return inter, goal_eff, non_eff


def plot_pr_bar(metrics: dict[str, float], cis: dict[str, tuple[float, float]], out_path: Path) -> None:
    labels = ["PR P1", "PR P2", "D_eff P1", "D_eff P2"]
    vals = [metrics["pr1"], metrics["pr2"], metrics["deff1"], metrics["deff2"]]

    yerr_low = [
        metrics["pr1"] - cis["pr1"][0],
        metrics["pr2"] - cis["pr2"][0],
        metrics["deff1"] - cis["deff1"][0],
        metrics["deff2"] - cis["deff2"][0],
    ]
    yerr_high = [
        cis["pr1"][1] - metrics["pr1"],
        cis["pr2"][1] - metrics["pr2"],
        cis["deff1"][1] - metrics["deff1"],
        cis["deff2"][1] - metrics["deff2"],
    ]

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(labels, vals, yerr=[yerr_low, yerr_high], capsize=5, color=["#2E5FA3", "#E07A2D", "#2E5FA3", "#E07A2D"], alpha=0.9)
    plt.ylabel("Dimensionality")
    plt.title("Participation ratio and effective dimensionality")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_eigen_spectrum(eig1: np.ndarray, eig2: np.ndarray, out_path: Path) -> None:
    r1 = np.arange(1, eig1.shape[0] + 1)
    r2 = np.arange(1, eig2.shape[0] + 1)

    plt.figure(figsize=(7.5, 5.0))
    plt.loglog(r1, np.maximum(eig1, 1e-12), color="#2E5FA3", linewidth=2, label="Phase 1")
    plt.loglog(r2, np.maximum(eig2, 1e-12), color="#E07A2D", linewidth=2, label="Phase 2")
    plt.xlabel("Eigenvalue rank")
    plt.ylabel("Eigenvalue magnitude")
    plt.title("Eigenvalue spectrum (log-log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_local_pr(pr_goal_1: float, pr_goal_2: float, pr_non_1: float, pr_non_2: float, out_path: Path) -> None:
    x = np.arange(2)
    w = 0.38

    plt.figure(figsize=(7.0, 4.8))
    plt.bar(x - w / 2, [pr_goal_1, pr_non_1], width=w, color="#2E5FA3", alpha=0.9, label="Phase 1")
    plt.bar(x + w / 2, [pr_goal_2, pr_non_2], width=w, color="#E07A2D", alpha=0.9, label="Phase 2")
    plt.xticks(x, ["Goal zone", "Non-goal zone"])
    plt.ylabel("Local participation ratio")
    plt.title("Local dimensionality near vs far from reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir if args.output_dir else (data_dir / "dimensionality_analysis"))
    fig_dir = ensure_dir(out_dir / "figures")

    goal, goal_source = load_goal_position(data_dir, args.goal_file, args.reward_x, args.reward_y)
    goal_radius = float(np.sqrt(args.goal_area_fraction / np.pi))

    print("[1/6] Loading phase data...")
    pos1, act1, src1 = load_phase_data(data_dir, args.pv_phase1, "before")
    pos2, act2, src2 = load_phase_data(data_dir, args.pv_phase2, "after")
    act1, act2 = align_features(act1, act2)
    print(f"      Phase1: {act1.shape}  Phase2: {act2.shape}")

    X1c = centered(act1)
    X2c = centered(act2)

    print("[2/6] Computing global PCA metrics...")
    eig1, eig2, evr1 = global_metrics_phase1_basis(X1c, X2c)
    pr1, deff1 = pr_deff_from_eigs(eig1)
    pr2, deff2 = pr_deff_from_eigs(eig2)
    print(f"      PR1={pr1:.2f}  PR2={pr2:.2f}  D_eff1={deff1:.2f}  D_eff2={deff2:.2f}")

    print("[3/6] Computing local PR (goal vs non-goal zone)...")
    d1 = distance_to_goal(pos1, goal)
    d2 = distance_to_goal(pos2, goal)

    g1 = d1 <= goal_radius
    g2 = d2 <= goal_radius
    ng1 = ~g1
    ng2 = ~g2

    pr_goal_1 = local_pr(X1c, g1)
    pr_goal_2 = local_pr(X2c, g2)
    pr_non_1 = local_pr(X1c, ng1)
    pr_non_2 = local_pr(X2c, ng2)

    goal_effect = pr_goal_2 - pr_goal_1 if np.isfinite(pr_goal_1) and np.isfinite(pr_goal_2) else np.nan
    non_effect = pr_non_2 - pr_non_1 if np.isfinite(pr_non_1) and np.isfinite(pr_non_2) else np.nan
    interaction = goal_effect - non_effect if np.isfinite(goal_effect) and np.isfinite(non_effect) else np.nan

    print(f"[4/6] Running global bootstrap ({args.n_bootstrap} iterations)...")
    boot = bootstrap_global(X1c, X2c, n_boot=args.n_bootstrap, seed=args.seed)
    print(f"[5/6] Running local bootstrap ({args.n_bootstrap} iterations)...")
    inter_dist, goal_eff_dist, non_eff_dist = bootstrap_local_interaction(
        X1c,
        X2c,
        d1,
        d2,
        goal_radius,
        n_boot=args.n_bootstrap,
        seed=args.seed + 1,
    )

    ci_pr1 = percentile_ci(boot["pr1"])
    ci_pr2 = percentile_ci(boot["pr2"])
    ci_deff1 = percentile_ci(boot["deff1"])
    ci_deff2 = percentile_ci(boot["deff2"])
    ci_pr_diff = percentile_ci(boot["pr_diff"])
    ci_deff_diff = percentile_ci(boot["deff_diff"])
    ci_inter = percentile_ci(inter_dist)

    print("[6/6] Saving figures and JSON...")
    plot_pr_bar(
        metrics={"pr1": pr1, "pr2": pr2, "deff1": deff1, "deff2": deff2},
        cis={"pr1": ci_pr1, "pr2": ci_pr2, "deff1": ci_deff1, "deff2": ci_deff2},
        out_path=fig_dir / "participation_ratio_bar.png",
    )
    plot_eigen_spectrum(eig1, eig2, fig_dir / "eigenvalue_spectrum.png")
    plot_local_pr(pr_goal_1, pr_goal_2, pr_non_1, pr_non_2, fig_dir / "local_pr_comparison.png")

    payload: dict[str, Any] = {
        "script": "dimensionality_analysis.py",
        "comparison_paper": "Mazzucato et al. 2016",
        "comparison_metric": "Participation ratio and effective dimensionality shift",
        "comparison_value_paper": None,
        "comparison_value_paper_note": "Directional benchmark: significant pre/post dimensionality change.",
        "your_value": float(pr2 - pr1) if np.isfinite(pr1) and np.isfinite(pr2) else np.nan,
        "your_value_ci_95": [ci_pr_diff[0], ci_pr_diff[1]],
        "qualitative_match": bool(np.isfinite(ci_pr_diff[0]) and np.isfinite(ci_pr_diff[1]) and ((ci_pr_diff[0] > 0.0) or (ci_pr_diff[1] < 0.0))),
        "goal_position": goal.tolist(),
        "goal_position_source": goal_source,
        "goal_zone_radius": goal_radius,
        "sources": {
            "phase1_data": src1,
            "phase2_data": src2,
        },
        "global": {
            "PR_phase1": pr1,
            "PR_phase2": pr2,
            "D_eff_phase1": deff1,
            "D_eff_phase2": deff2,
            "PR_phase1_ci_95": [ci_pr1[0], ci_pr1[1]],
            "PR_phase2_ci_95": [ci_pr2[0], ci_pr2[1]],
            "D_eff_phase1_ci_95": [ci_deff1[0], ci_deff1[1]],
            "D_eff_phase2_ci_95": [ci_deff2[0], ci_deff2[1]],
            "PR_diff_phase2_minus_phase1_ci_95": [ci_pr_diff[0], ci_pr_diff[1]],
            "D_eff_diff_phase2_minus_phase1_ci_95": [ci_deff_diff[0], ci_deff_diff[1]],
            "phase1_cumulative_explained_var": float(np.sum(evr1)),
        },
        "local": {
            "PR_goal_phase1": pr_goal_1,
            "PR_goal_phase2": pr_goal_2,
            "PR_nongol_phase1": pr_non_1,
            "PR_nongol_phase2": pr_non_2,
            "interaction_contrast": interaction,
            "interaction_ci_95": [ci_inter[0], ci_inter[1]],
        },
        "eigenvalues_phase1": eig1.tolist(),
        "eigenvalues_phase2": eig2.tolist(),
        "spatial_specificity_rho": np.nan,
        "spatial_specificity_pval": np.nan,
        "goal_zone_effect_size": float(np.nanmean(goal_eff_dist)),
        "non_goal_zone_effect_size": float(np.nanmean(non_eff_dist)),
        "interaction_contrast": interaction,
        "interaction_bootstrap_ci_low": ci_inter[0],
        "interaction_bootstrap_ci_high": ci_inter[1],
    }

    save_json(out_dir / "dimensionality_results.json", payload)
    print(f"Saved Section C2 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
