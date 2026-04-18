from __future__ import annotations
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, ttest_rel, norm
matplotlib.use("Agg")
REWARD = np.array([0.5, 0.5], dtype=float)
DATA_DIR = Path("data")
OUT_DIR  = Path("../New_Results/skaggs_distance_analysis")
N_BOOT   = 5_000
N_PERM   = 10_000
SEED     = 42
RNG      = np.random.default_rng(SEED)
C_BEFORE = "#2E5FA3"
C_AFTER  = "#E07A2D"
C_GOLD   = "#F4D03F"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def center_to_xy(center) -> tuple[float, float] | None:
    if center is None:
        return None
    try:
        arr = np.asarray(center, dtype=float)
    except Exception:
        return None
    if arr.ndim == 1 and arr.size == 2:
        return float(arr[0]), float(arr[1])
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
        return float(arr[0, 0]), float(arr[0, 1])
    return None


def dist_to_reward(x: float, y: float) -> float:
    return float(np.sqrt((x - REWARD[0])**2 + (y - REWARD[1])**2))


def fisher_z_compare(rho1: float, n1: int, rho2: float, n2: int) -> tuple[float, float]:
    if n1 <= 3 or n2 <= 3:
        return np.nan, np.nan
    r1 = np.clip(rho1, -0.999999, 0.999999)
    r2 = np.clip(rho2, -0.999999, 0.999999)
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    z_diff = (z1 - z2) / np.sqrt(1/(n1-3) + 1/(n2-3))
    p = 2.0 * (1.0 - norm.cdf(abs(z_diff)))
    return float(z_diff), float(p)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    means = np.array([np.mean(RNG.choice(vals, size=vals.size, replace=True)) for _ in range(n_boot)])
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def build_dataset(phase: str) -> pd.DataFrame:
    skaggs_path  = DATA_DIR / f"df_skaggs_{phase}.pkl"
    centers_path = DATA_DIR / f"df_centers_{phase}.pkl"
    skaggs  = pd.read_pickle(skaggs_path)[["neuron_id", "skaggs_score"]]
    centers = pd.read_pickle(centers_path)
    centers_k1 = centers[centers["k"] == 1].copy()
    parsed = []
    for _, row in centers_k1.iterrows():
        xy = center_to_xy(row["center"])
        if xy is None:
            continue
        parsed.append({"neuron_id": int(row["neuron_id"]),
                       "cx": xy[0], "cy": xy[1]})
    centers_parsed = pd.DataFrame(parsed).drop_duplicates("neuron_id")

    merged = skaggs.merge(centers_parsed, on="neuron_id", how="inner")
    merged["dist_reward"] = merged.apply(lambda r: dist_to_reward(r["cx"], r["cy"]), axis=1)
    merged["phase"] = phase
    return merged


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_b = build_dataset("before")
    df_a = build_dataset("after")
    df_paired = df_b.merge(
        df_a[["neuron_id", "skaggs_score", "dist_reward"]].rename(
            columns={"skaggs_score": "skaggs_after", "dist_reward": "dist_after"}),
        on="neuron_id", how="inner"
    ).rename(columns={"skaggs_score": "skaggs_before", "dist_reward": "dist_before"})
    df_paired["delta_skaggs"] = df_paired["skaggs_after"] - df_paired["skaggs_before"]

    print(f"  Phase 1 neurons: {len(df_b)}")
    print(f"  Phase 2 neurons: {len(df_a)}")
    print(f"  Paired (both phases): {len(df_paired)}")
    return df_b, df_a, df_paired


def spearman_with_ci(x: np.ndarray, y: np.ndarray) -> dict:
    rho, p = spearmanr(x, y)
    n = len(x)
    boot_rhos = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        idx = RNG.choice(n, size=n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        boot_rhos[i] = r
    return {
        "rho": float(rho),
        "p": float(p),
        "ci_95": [float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))],
        "n": int(n),
    }


def near_far_ratio(df: pd.DataFrame, skaggs_col: str = "skaggs_score") -> dict:
    q25 = float(df["dist_reward"].quantile(0.25))
    q75 = float(df["dist_reward"].quantile(0.75))
    near = df[df["dist_reward"] <= q25][skaggs_col].to_numpy(dtype=float)
    far  = df[df["dist_reward"] >= q75][skaggs_col].to_numpy(dtype=float)
    mean_near, ci_low_near, ci_high_near = bootstrap_ci(near)
    mean_far,  ci_low_far,  ci_high_far  = bootstrap_ci(far)
    stat, p_mw = mannwhitneyu(near, far, alternative="greater")
    ratio = float(mean_near / mean_far) if (np.isfinite(mean_far) and mean_far > 0) else np.nan
    return {
        "q25_dist": q25, "q75_dist": q75,
        "n_near": int(len(near)), "n_far": int(len(far)),
        "mean_near": mean_near, "ci_near": [ci_low_near, ci_high_near],
        "mean_far":  mean_far,  "ci_far":  [ci_low_far,  ci_high_far],
        "ratio_near_over_far": ratio,
        "mannwhitney_u": float(stat), "p_near_gt_far": float(p_mw),
    }


def interaction_test(df_paired: pd.DataFrame) -> dict:
    q25 = float(df_paired["dist_before"].quantile(0.25))
    q75 = float(df_paired["dist_before"].quantile(0.75))
    near_delta = df_paired[df_paired["dist_before"] <= q25]["delta_skaggs"].to_numpy(dtype=float)
    far_delta  = df_paired[df_paired["dist_before"] >= q75]["delta_skaggs"].to_numpy(dtype=float)
    stat, p_mw = mannwhitneyu(near_delta, far_delta, alternative="two-sided")
    m_near, ci_low, ci_high = bootstrap_ci(near_delta)
    m_far,  ci_low_f, ci_high_f = bootstrap_ci(far_delta)
    return {
        "q25_dist": q25, "q75_dist": q75,
        "mean_delta_near": m_near, "ci_delta_near": [ci_low, ci_high],
        "mean_delta_far":  m_far,  "ci_delta_far":  [ci_low_f, ci_high_f],
        "interaction_mannwhitney_p": float(p_mw),
        "n_near": int(len(near_delta)), "n_far": int(len(far_delta)),
    }


def compute_all_stats(df_b: pd.DataFrame, df_a: pd.DataFrame,
                      df_paired: pd.DataFrame) -> dict:
    print("Computing statistics …")
    corr_b = spearman_with_ci(
        df_b["dist_reward"].to_numpy(), df_b["skaggs_score"].to_numpy())
    corr_a = spearman_with_ci(
        df_a["dist_reward"].to_numpy(), df_a["skaggs_score"].to_numpy())

    fz_diff, fz_p = fisher_z_compare(corr_b["rho"], corr_b["n"],
                                     corr_a["rho"], corr_a["n"])
    nf_b = near_far_ratio(df_b)
    nf_a = near_far_ratio(df_a)

    delta = df_paired["delta_skaggs"].to_numpy(dtype=float)
    t_stat, p_paired = ttest_rel(
        df_paired["skaggs_after"].to_numpy(),
        df_paired["skaggs_before"].to_numpy(),
        alternative="two-sided")
    delta_mean, delta_ci_low, delta_ci_high = bootstrap_ci(delta)
    interact = interaction_test(df_paired)

    return {
        "spearman_before": corr_b,
        "spearman_after":  corr_a,
        "fisher_z_rho_change": {"z_diff": fz_diff, "p": fz_p},
        "near_far_ratio_before": nf_b,
        "near_far_ratio_after":  nf_a,
        "paired_delta_skaggs": {
            "mean_delta": delta_mean,
            "ci_95": [delta_ci_low, delta_ci_high],
            "ttest_t": float(t_stat),
            "ttest_p": float(p_paired),
            "n": int(len(delta)),
        },
        "interaction_near_vs_far_delta": interact,
    }


def _binned_summary(df: pd.DataFrame, x_col: str, y_col: str, n_bins: int = 8) -> pd.DataFrame:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    edges = np.linspace(x.min(), x.max() + 1e-10, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means, sems, ns = [], [], []
    for i in range(n_bins):
        mask = (x >= edges[i]) & (x < edges[i+1])
        vals = y[mask]
        if vals.size > 0:
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
            ns.append(int(vals.size))
        else:
            means.append(np.nan); sems.append(np.nan); ns.append(0)
    return pd.DataFrame({"center": centers, "mean": means, "sem": sems, "n": ns})


def plot_scatter(df_b: pd.DataFrame, df_a: pd.DataFrame,stats: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, df, phase, color, label in [
        (axes[0], df_b, "before", C_BEFORE, "Phase 1 (before RL)"),
        (axes[1], df_a, "after",  C_AFTER,  "Phase 2 (after RL)"),
    ]:
        ax.scatter(df["dist_reward"], df["skaggs_score"],s=14, alpha=0.3, color=color, label=label)

        binned = _binned_summary(df, "dist_reward", "skaggs_score", n_bins=8)
        valid = binned[binned["n"] > 0]
        ax.errorbar(valid["center"], valid["mean"], yerr=valid["sem"],
                    color=color, marker="o", lw=2.0, capsize=4,
                    label="Binned mean ± SEM")
        rho = stats[f"spearman_{phase}"]["rho"]
        p   = stats[f"spearman_{phase}"]["p"]
        ax.set_xlabel("Distance to reward (field centroid)", fontsize=10)
        ax.set_ylabel("Skaggs information score (bits/spike)", fontsize=10)
        ax.set_title(f"{label}\nSpearman ρ = {rho:.3f},  p = {p:.2e}", fontsize=10)
        ax.legend(fontsize=9)
        ax.axvline(0.12, color=C_GOLD, ls="--", lw=1.2, alpha=0.7, label="Goal zone radius")

    plt.suptitle("Skaggs Score vs Distance from Reward", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "skaggs_vs_distance_scatter.png", dpi=150)
    plt.close()
    print("  Saved: skaggs_vs_distance_scatter.png")


def plot_change_by_zone(df_paired: pd.DataFrame, stats: dict, fig_dir: Path) -> None:
    df = df_paired.copy()
    q25 = float(df["dist_before"].quantile(0.25))
    q75 = float(df["dist_before"].quantile(0.75))
    df["zone"] = "Mid"
    df.loc[df["dist_before"] <= q25, "zone"] = f"Near\n(d≤{q25:.2f})"
    df.loc[df["dist_before"] >= q75, "zone"] = f"Far\n(d≥{q75:.2f})"
    group_order = [f"Near\n(d≤{q25:.2f})", "Mid", f"Far\n(d≥{q75:.2f})"]
    colors = {"Near": "#E63946", "Mid": "#A8DADC", "Far": "#457B9D"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for j, grp in enumerate(group_order):
        subset = df[df["zone"] == grp]["delta_skaggs"].to_numpy(dtype=float)
        if len(subset) < 3:
            continue
        color_key = grp.split("\n")[0]
        parts = ax.violinplot(subset, positions=[j], showmedians=True, widths=0.65)
        for pc in parts["bodies"]:
            pc.set_facecolor(colors[color_key])
            pc.set_alpha(0.7)
        for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[part].set_color("black")
            parts[part].set_linewidth(1.2)
        m, lo, hi = bootstrap_ci(subset)
        ax.errorbar(j, m, yerr=[[m-lo], [hi-m]], fmt="D",
                    color="black", ms=8, capsize=5, lw=2, zorder=5)
        ax.text(j, hi + 0.005, f"μ={m:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order, fontsize=10)
    ax.set_ylabel("ΔSkaggs score (after − before RL)", fontsize=10)

    interact = stats["interaction_near_vs_far_delta"]
    ax.set_title(
        "Change in Skaggs Score by Distance Zone\n"
        f"Near vs Far interaction: p = {interact['interaction_mannwhitney_p']:.4f}",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "skaggs_change_by_zone.png", dpi=150)
    plt.close()
    print("  Saved: skaggs_change_by_zone.png")


def plot_near_far_bar(stats: dict, fig_dir: Path) -> None:
    nf_b = stats["near_far_ratio_before"]
    nf_a = stats["near_far_ratio_after"]
    labels  = ["Phase 1\nNear", "Phase 1\nFar", "Phase 2\nNear", "Phase 2\nFar"]
    values  = [nf_b["mean_near"], nf_b["mean_far"],
               nf_a["mean_near"], nf_a["mean_far"]]
    ci_lows = [nf_b["ci_near"][0], nf_b["ci_far"][0],
               nf_a["ci_near"][0], nf_a["ci_far"][0]]
    ci_highs= [nf_b["ci_near"][1], nf_b["ci_far"][1],
               nf_a["ci_near"][1], nf_a["ci_far"][1]]
    bar_colors = [C_BEFORE, C_BEFORE, C_AFTER, C_AFTER]
    alphas = [0.9, 0.5, 0.9, 0.5]
    hatches = [None, "//", None, "//"]
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [0, 1, 3, 4]
    for i, (x, v, lo, hi, c, a, h) in enumerate(
            zip(xs, values, ci_lows, ci_highs, bar_colors, alphas, hatches)):
        bar = ax.bar(x, v, color=c, alpha=a, hatch=h, edgecolor="black",
                     width=0.75, label=labels[i])
        err_low  = v - lo if (np.isfinite(lo) and lo is not None) else 0
        err_high = hi - v if (np.isfinite(hi) and hi is not None) else 0
        ax.errorbar(x, v, yerr=[[err_low], [err_high]], fmt="none",color="black", capsize=5, lw=2)
    for phase_label, nf, x_near, x_far in [
        ("Phase 1", nf_b, 0, 1), ("Phase 2", nf_a, 3, 4)
    ]:
        ratio = nf["ratio_near_over_far"]
        p_val = nf["p_near_gt_far"]
        ax.annotate(
            f"Near/Far = {ratio:.2f}\n(MW p={p_val:.4f})",
            xy=((x_near + x_far) / 2, max(nf["mean_near"], nf["mean_far"]) * 1.05),
            ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Skaggs score (bits/spike)", fontsize=10)
    ax.set_title("Skaggs Information Near vs Far from Reward\n"
                 "(filled = near-reward cells, hatched = far-reward cells)", fontsize=10)
    ax.axhline(0, color="black", lw=0.8)
    plt.tight_layout()
    plt.savefig(fig_dir / "skaggs_near_far_ratio.png", dpi=150)
    plt.close()
    print("  Saved: skaggs_near_far_ratio.png")


def plot_4panel(df_b: pd.DataFrame, df_a: pd.DataFrame,
                df_paired: pd.DataFrame, stats: dict, fig_dir: Path) -> None:
    fig = plt.figure(figsize=(14, 11))
    ax_a = fig.add_subplot(2, 2, 1)
    ax_a.scatter(df_b["dist_reward"], df_b["skaggs_score"],s=12, alpha=0.3, color=C_BEFORE)
    binned_b = _binned_summary(df_b, "dist_reward", "skaggs_score")
    valid_b  = binned_b[binned_b["n"] > 0]
    ax_a.errorbar(valid_b["center"], valid_b["mean"], yerr=valid_b["sem"],
                  color=C_BEFORE, marker="o", lw=2, capsize=4, label="Binned mean")
    ax_a.axvline(0.12, color=C_GOLD, ls="--", lw=1.2, alpha=0.7)
    rho_b = stats["spearman_before"]["rho"]
    p_b   = stats["spearman_before"]["p"]
    ax_a.set_title(f"A  Phase 1 (before RL)\nSpearman ρ = {rho_b:.3f},  p = {p_b:.2e}", fontsize=10)
    ax_a.set_xlabel("Distance to reward"); ax_a.set_ylabel("Skaggs score")
    ax_a.legend(fontsize=8)
    ax_b = fig.add_subplot(2, 2, 2)
    ax_b.scatter(df_a["dist_reward"], df_a["skaggs_score"],s=12, alpha=0.3, color=C_AFTER)
    binned_a = _binned_summary(df_a, "dist_reward", "skaggs_score")
    valid_a  = binned_a[binned_a["n"] > 0]
    ax_b.errorbar(valid_a["center"], valid_a["mean"], yerr=valid_a["sem"],
                  color=C_AFTER, marker="o", lw=2, capsize=4, label="Binned mean")
    ax_b.axvline(0.12, color=C_GOLD, ls="--", lw=1.2, alpha=0.7)
    rho_a = stats["spearman_after"]["rho"]
    p_a   = stats["spearman_after"]["p"]
    ax_b.set_title(f"B  Phase 2 (after RL)\nSpearman ρ = {rho_a:.3f},  p = {p_a:.2e}", fontsize=10)
    ax_b.set_xlabel("Distance to reward"); ax_b.set_ylabel("Skaggs score")
    ax_b.legend(fontsize=8)
    ax_c = fig.add_subplot(2, 2, 3)
    df2 = df_paired.copy()
    q25 = float(df2["dist_before"].quantile(0.25))
    q75 = float(df2["dist_before"].quantile(0.75))
    df2["zone"] = "Mid"
    df2.loc[df2["dist_before"] <= q25, "zone"] = f"Near\n(d≤{q25:.2f})"
    df2.loc[df2["dist_before"] >= q75, "zone"] = f"Far\n(d≥{q75:.2f})"
    group_order_c = [f"Near\n(d≤{q25:.2f})", "Mid", f"Far\n(d≥{q75:.2f})"]
    zone_colors = {f"Near\n(d≤{q25:.2f})": "#E63946",
                   "Mid": "#A8DADC",
                   f"Far\n(d≥{q75:.2f})": "#457B9D"}
    for j, grp in enumerate(group_order_c):
        subset = df2[df2["zone"] == grp]["delta_skaggs"].to_numpy(dtype=float)
        if len(subset) < 3:
            continue
        parts = ax_c.violinplot(subset, positions=[j], showmedians=True, widths=0.65)
        for pc in parts["bodies"]:
            pc.set_facecolor(zone_colors[grp])
            pc.set_alpha(0.7)
        for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[part].set_color("black"); parts[part].set_linewidth(1.0)
        m, lo, hi = bootstrap_ci(subset)
        ax_c.text(j, hi + 0.005, f"μ={m:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax_c.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax_c.set_xticks(range(len(group_order_c)))
    ax_c.set_xticklabels(group_order_c, fontsize=9)
    ax_c.set_ylabel("ΔSkaggs (after − before)")
    interact_p = stats["interaction_near_vs_far_delta"]["interaction_mannwhitney_p"]
    ax_c.set_title(f"C  ΔSkaggs by distance zone\nNear vs Far interaction p = {interact_p:.4f}", fontsize=10)
    ax_d = fig.add_subplot(2, 2, 4)
    nf_b2 = stats["near_far_ratio_before"]
    nf_a2 = stats["near_far_ratio_after"]
    xs = [0, 1, 3, 4]
    values  = [nf_b2["mean_near"], nf_b2["mean_far"],
               nf_a2["mean_near"], nf_a2["mean_far"]]
    ci_lows = [nf_b2["ci_near"][0], nf_b2["ci_far"][0],
               nf_a2["ci_near"][0], nf_a2["ci_far"][0]]
    ci_highs= [nf_b2["ci_near"][1], nf_b2["ci_far"][1],
               nf_a2["ci_near"][1], nf_a2["ci_far"][1]]
    clrs    = [C_BEFORE, C_BEFORE, C_AFTER, C_AFTER]
    alps    = [0.9, 0.5, 0.9, 0.5]
    htchs   = [None, "//", None, "//"]
    xlbls   = ["Ph1\nNear", "Ph1\nFar", "Ph2\nNear", "Ph2\nFar"]
    for i, (x, v, lo, hi, c, a, h, lbl) in enumerate(
            zip(xs, values, ci_lows, ci_highs, clrs, alps, htchs, xlbls)):
        ax_d.bar(x, v, color=c, alpha=a, hatch=h, edgecolor="black", width=0.75)
        err_low  = v - lo if (np.isfinite(lo) and lo is not None) else 0
        err_high = hi - v if (np.isfinite(hi) and hi is not None) else 0
        ax_d.errorbar(x, v, yerr=[[err_low], [err_high]], fmt="none",
                      color="black", capsize=5, lw=1.5)
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels(xlbls, fontsize=9)
    ax_d.set_ylabel("Mean Skaggs score")
    r1 = nf_b2["ratio_near_over_far"]
    r2 = nf_a2["ratio_near_over_far"]
    ax_d.set_title(f"D  Near/Far Skaggs ratio\nPhase 1: {r1:.2f}×  |  Phase 2: {r2:.2f}×", fontsize=10)

    plt.suptitle("Skaggs Information Score vs Distance from Reward", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "skaggs_4panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: skaggs_4panel.png  ← main figure")


def main() -> None:
    out_dir = ensure_dir(OUT_DIR)
    fig_dir = ensure_dir(out_dir / "figures")

    print("Loading data …")
    df_b, df_a, df_paired = load_all_data()
    stats = compute_all_stats(df_b, df_a, df_paired)
    print("\n── Key Results ──────────────────────────────────────────────────")
    print(f"  Spearman ρ (before RL): {stats['spearman_before']['rho']:.4f}  "
          f"p={stats['spearman_before']['p']:.2e}")
    print(f"  Spearman ρ (after RL):  {stats['spearman_after']['rho']:.4f}  "
          f"p={stats['spearman_after']['p']:.2e}")
    print(f"  Fisher z change in ρ:   z={stats['fisher_z_rho_change']['z_diff']:.3f}  "
          f"p={stats['fisher_z_rho_change']['p']:.4f}")
    nf_b = stats["near_far_ratio_before"]
    nf_a = stats["near_far_ratio_after"]
    print(f"  Near/Far Skaggs ratio (Phase 1): {nf_b['ratio_near_over_far']:.3f}  "
          f"(MW p={nf_b['p_near_gt_far']:.4f})")
    print(f"  Near/Far Skaggs ratio (Phase 2): {nf_a['ratio_near_over_far']:.3f}  "
          f"(MW p={nf_a['p_near_gt_far']:.4f})")
    interact = stats["interaction_near_vs_far_delta"]
    print(f"  Interaction (near vs far ΔSkaggs): p={interact['interaction_mannwhitney_p']:.4f}")
    print(f"  Mean ΔSkaggs near-reward: {interact['mean_delta_near']:.4f}")
    print(f"  Mean ΔSkaggs far-reward:  {interact['mean_delta_far']:.4f}")
    print("──────────────────────────────────────────────────────────────")
    print("\nGenerating figures …")
    plot_scatter(df_b, df_a, stats, fig_dir)
    plot_change_by_zone(df_paired, stats, fig_dir)
    plot_near_far_bar(stats, fig_dir)
    plot_4panel(df_b, df_a, df_paired, stats, fig_dir)

    save_json(out_dir / "skaggs_distance_results.json", stats)
    print(f"\nAll outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
