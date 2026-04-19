"""
Tier 1.1 - De-trended directional shift analysis.

Subtracts the global population drift vector from each neuron's shift before
computing the cosine similarity with the toward-reward direction. The raw
analysis was dominated by a rightward occupancy drift (mean dx=+0.032);
removing it isolates the reward-specific component of remapping.

Inputs : ../New_Results/shift_directional_analysis/shift_vectors.csv
Outputs: ../New_Results/shift_detrended_analysis/
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

matplotlib.use("Agg")

REWARD = np.array([0.5, 0.5], dtype=float)
GOAL_RADIUS = 0.12
IN_CSV = Path("../New_Results/shift_directional_analysis/shift_vectors.csv")
OUT_DIR = Path("../New_Results/shift_detrended_analysis")
N_PERM = 10_000
N_BOOT = 5_000
SEED = 42
RNG = np.random.default_rng(SEED)

C_REWARD = "#E63946"
C_AWAY = "#457B9D"
C_GOLD = "#F4D03F"
C_RAW = "#8D99AE"
C_DET = "#2A9D8F"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def rayleigh_test(angles_rad: np.ndarray) -> tuple[float, float]:
    n = angles_rad.size
    if n < 3:
        return np.nan, np.nan
    C = np.mean(np.cos(angles_rad))
    S = np.mean(np.sin(angles_rad))
    R = np.sqrt(C**2 + S**2)
    z = n * R**2
    p = np.exp(-z) * (
        1.0
        + (2.0 * z - z**2) / (4.0 * n)
        - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n**2)
    )
    return float(R), float(np.clip(p, 0.0, 1.0))


def load_shift_vectors() -> pd.DataFrame:
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"Expected {IN_CSV}. Run shift_directional_analysis.py first."
        )
    df = pd.read_csv(IN_CSV)
    df["dx"] = df["ax"] - df["bx"]
    df["dy"] = df["ay"] - df["by"]
    return df


def detrend(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mean_dx = float(df["dx"].mean())
    mean_dy = float(df["dy"].mean())

    df = df.copy()
    df["dx_det"] = df["dx"] - mean_dx
    df["dy_det"] = df["dy"] - mean_dy
    df["shift_mag_det"] = np.sqrt(df["dx_det"] ** 2 + df["dy_det"] ** 2)
    df = df[df["shift_mag_det"] > 1e-8].copy()

    tnorm = np.sqrt((REWARD[0] - df["bx"]) ** 2 + (REWARD[1] - df["by"]) ** 2)
    df = df[tnorm > 1e-8].copy()
    tx = (REWARD[0] - df["bx"]) / np.sqrt(
        (REWARD[0] - df["bx"]) ** 2 + (REWARD[1] - df["by"]) ** 2
    )
    ty = (REWARD[1] - df["by"]) / np.sqrt(
        (REWARD[0] - df["bx"]) ** 2 + (REWARD[1] - df["by"]) ** 2
    )

    ux = df["dx_det"] / df["shift_mag_det"]
    uy = df["dy_det"] / df["shift_mag_det"]
    df["cos_theta_det"] = ux * tx + uy * ty
    df["shift_angle_det"] = np.arctan2(df["dy_det"], df["dx_det"])
    toward_angle = np.arctan2(ty, tx)
    rel = df["shift_angle_det"] - toward_angle
    df["rel_angle_det"] = (rel + np.pi) % (2 * np.pi) - np.pi
    df["ax_det"] = df["bx"] + df["dx_det"]
    df["ay_det"] = df["by"] + df["dy_det"]

    drift = {
        "mean_dx_raw": mean_dx,
        "mean_dy_raw": mean_dy,
        "drift_magnitude": float(np.sqrt(mean_dx**2 + mean_dy**2)),
        "drift_angle_deg": float(np.degrees(np.arctan2(mean_dy, mean_dx))),
        "n_after_filter": int(len(df)),
    }
    return df, drift


def cosine_statistics(cosines: np.ndarray, rel_angles: np.ndarray,
                      shift_angles: np.ndarray) -> dict:
    n = cosines.size
    R_all, p_ray_all = rayleigh_test(shift_angles)
    R_rel, p_ray_rel = rayleigh_test(rel_angles)
    mean_cos = float(np.mean(cosines))
    mean_rel_angle_deg = float(
        np.degrees(np.arctan2(np.mean(np.sin(rel_angles)), np.mean(np.cos(rel_angles))))
    )
    t_stat, p_ttest = ttest_1samp(cosines, popmean=0.0, alternative="greater")

    perm_means = np.empty(N_PERM, dtype=float)
    for i in range(N_PERM):
        signs = RNG.choice([-1.0, 1.0], size=n)
        perm_means[i] = float(np.mean(signs * cosines))
    perm_p = float(np.mean(perm_means >= mean_cos))

    boot_means = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        boot_means[i] = float(np.mean(RNG.choice(cosines, size=n, replace=True)))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    return {
        "n": int(n),
        "mean_cosine": mean_cos,
        "bootstrap_ci_95": [ci_low, ci_high],
        "mean_rel_angle_deg": mean_rel_angle_deg,
        "fraction_toward_reward": float(np.mean(cosines > 0)),
        "rayleigh_all_directions": {"R": R_all, "p": p_ray_all},
        "rayleigh_toward_reward": {"R": R_rel, "p": p_ray_rel},
        "ttest_cos_vs_0": {"t": float(t_stat), "p_one_tailed": float(p_ttest)},
        "permutation_test": {
            "observed_mean_cos": mean_cos,
            "p": perm_p,
            "null_95th": float(np.percentile(perm_means, 95)),
        },
    }


def near_far_breakdown(df: pd.DataFrame, cos_col: str) -> dict:
    q25 = float(df["dist_before"].quantile(0.25))
    q75 = float(df["dist_before"].quantile(0.75))
    near = df[df["dist_before"] <= q25][cos_col].to_numpy(dtype=float)
    far = df[df["dist_before"] >= q75][cos_col].to_numpy(dtype=float)

    def _boot(v):
        if v.size == 0:
            return np.nan, np.nan, np.nan
        means = np.array([np.mean(RNG.choice(v, size=v.size, replace=True))
                          for _ in range(N_BOOT)])
        return float(np.mean(v)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    m_n, lo_n, hi_n = _boot(near)
    m_f, lo_f, hi_f = _boot(far)
    return {
        "q25_dist": q25, "q75_dist": q75,
        "n_near": int(near.size), "n_far": int(far.size),
        "mean_cos_near": m_n, "ci_near": [lo_n, hi_n],
        "mean_cos_far":  m_f, "ci_far":  [lo_f, hi_f],
    }


def plot_drift_arrow(mean_dx: float, mean_dy: float, df: pd.DataFrame,
                     fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["bx"], df["by"], s=12, c=C_RAW, alpha=0.5,
               label="Field centre (before)")
    ax.annotate(
        "", xy=(0.5 + mean_dx * 10, 0.5 + mean_dy * 10), xytext=(0.5, 0.5),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=3, mutation_scale=16),
    )
    ax.text(0.5 + mean_dx * 10 + 0.02, 0.5 + mean_dy * 10 + 0.02,
            f"drift x 10\n({mean_dx:+.3f}, {mean_dy:+.3f})", fontsize=9)
    circle = plt.Circle(REWARD, GOAL_RADIUS, fill=False, edgecolor=C_GOLD, lw=2, ls="--")
    ax.add_patch(circle)
    ax.scatter(*REWARD, s=220, marker="*", color=C_GOLD, zorder=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Global population drift (subtracted from each neuron)")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "drift_vector.png", dpi=150)
    plt.close()
    print("  Saved: drift_vector.png")


def plot_raw_vs_detrended_compare(cos_raw: np.ndarray, cos_det: np.ndarray,
                                  stats_raw: dict, stats_det: dict,
                                  fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(cos_raw, bins=30, density=True, alpha=0.55, color=C_RAW,
            label=f"Raw  (mu={stats_raw['mean_cosine']:+.3f})")
    ax.hist(cos_det, bins=30, density=True, alpha=0.55, color=C_DET,
            label=f"Detrended  (mu={stats_det['mean_cosine']:+.3f})")
    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.axvline(stats_raw["mean_cosine"], color=C_RAW, lw=2)
    ax.axvline(stats_det["mean_cosine"], color=C_DET, lw=2)
    ax.set_xlabel("cos(theta)  [+ = shift toward reward]")
    ax.set_ylabel("Density")
    ax.set_title("Raw vs detrended cosine distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "raw_vs_detrended_cosine.png", dpi=150)
    plt.close()
    print("  Saved: raw_vs_detrended_cosine.png")


def plot_4panel(df: pd.DataFrame, stats: dict, near_far: dict,
                drift: dict, fig_dir: Path) -> None:
    fig = plt.figure(figsize=(14, 12))

    ax_a = fig.add_subplot(2, 2, 1)
    cmap = matplotlib.colormaps.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    for _, row in df.iterrows():
        c = cmap(norm(row["cos_theta_det"]))
        ax_a.annotate(
            "", xy=(row["ax_det"], row["ay_det"]), xytext=(row["bx"], row["by"]),
            arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, mutation_scale=5),
        )
    ax_a.scatter(df["bx"], df["by"], s=6, c="gray", alpha=0.4, zorder=3)
    circle = plt.Circle(REWARD, GOAL_RADIUS, fill=False, edgecolor=C_GOLD, lw=2, ls="--")
    ax_a.add_patch(circle)
    ax_a.scatter(*REWARD, s=180, marker="*", color=C_GOLD, zorder=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax_a, fraction=0.04, pad=0.02).set_label(
        "cos(theta) detrended", fontsize=9)
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.set_aspect("equal")
    ax_a.set_xlabel("x"); ax_a.set_ylabel("y")
    ax_a.set_title(
        "A  Detrended shift vectors\n"
        f"(drift removed: [{drift['mean_dx_raw']:+.3f}, {drift['mean_dy_raw']:+.3f}])",
        fontsize=10,
    )

    ax_b = fig.add_subplot(2, 2, 2, projection="polar")
    rel = df["rel_angle_det"].to_numpy(dtype=float)
    n_bins = 24
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(rel, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = 2 * np.pi / n_bins
    bar_colors = [C_REWARD if abs(c) < np.pi / 2 else C_AWAY for c in centers]
    ax_b.bar(centers, counts, width=width * 0.85, color=bar_colors,
             alpha=0.75, edgecolor="white", linewidth=0.5)
    ax_b.plot(np.linspace(-np.pi, np.pi, 200), [len(rel) / n_bins] * 200,
              "k--", lw=1.5)
    mean_dir = stats["mean_rel_angle_deg"] * np.pi / 180
    ax_b.annotate("", xy=(mean_dir, float(np.max(counts)) * 0.85), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax_b.set_theta_zero_location("E"); ax_b.set_theta_direction(1)
    ax_b.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax_b.set_xticklabels(["-> Reward", "", "perp", "", "<- Away", "", "perp", ""],
                         fontsize=8)
    rr = stats["rayleigh_toward_reward"]
    ax_b.set_title(
        f"B  Detrended direction vs reward\nRayleigh R={rr['R']:.3f}, p={rr['p']:.4f}",
        fontsize=10, pad=15,
    )

    ax_c = fig.add_subplot(2, 2, 3)
    cosines = df["cos_theta_det"].to_numpy(dtype=float)
    null_means = np.empty(2000, dtype=float)
    for i in range(2000):
        s = RNG.choice([-1.0, 1.0], size=cosines.size)
        null_means[i] = float(np.mean(s * cosines))
    ax_c.hist(cosines, bins=25, density=True, color=C_DET, alpha=0.65,
              label="Detrended cos(theta)")
    ax_c.axvline(0, color="black", lw=1.2, ls="--")
    ax_c.axvline(stats["mean_cosine"], color=C_REWARD, lw=2.5,
                 label=f"Mean = {stats['mean_cosine']:+.3f}")
    ci = stats["bootstrap_ci_95"]
    ax_c.axvspan(ci[0], ci[1], alpha=0.15, color=C_REWARD, label="95% CI")
    ax_c2 = ax_c.twinx()
    ax_c2.hist(null_means, bins=30, density=True, color="gray", alpha=0.30,
               label="Permutation null")
    ax_c2.set_ylabel("Null density", fontsize=8, color="gray")
    ax_c2.tick_params(axis="y", labelcolor="gray")
    p_t = stats["ttest_cos_vs_0"]["p_one_tailed"]
    p_perm = stats["permutation_test"]["p"]
    ax_c.set_xlabel("cos(theta) detrended", fontsize=9)
    ax_c.set_ylabel("Density", fontsize=9)
    ax_c.set_title(
        f"C  Detrended cosine distribution\nt-test p={p_t:.4f} | perm p={p_perm:.4f}",
        fontsize=10,
    )
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax_d = fig.add_subplot(2, 2, 4)
    q25 = near_far["q25_dist"]; q75 = near_far["q75_dist"]
    df2 = df.copy()
    df2["zone"] = "Mid"
    df2.loc[df2["dist_before"] <= q25, "zone"] = f"Near\n(d<={q25:.2f})"
    df2.loc[df2["dist_before"] >= q75, "zone"] = f"Far\n(d>={q75:.2f})"
    order = [f"Near\n(d<={q25:.2f})", "Mid", f"Far\n(d>={q75:.2f})"]
    pal = {f"Near\n(d<={q25:.2f})": C_REWARD, "Mid": "#A8DADC",
           f"Far\n(d>={q75:.2f})": C_AWAY}
    for j, grp in enumerate(order):
        subset = df2[df2["zone"] == grp]["cos_theta_det"].to_numpy(dtype=float)
        if subset.size < 3:
            continue
        parts = ax_d.violinplot(subset, positions=[j], showmedians=True, widths=0.65)
        for pc in parts["bodies"]:
            pc.set_facecolor(pal[grp]); pc.set_alpha(0.7)
        for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[part].set_color("black"); parts[part].set_linewidth(1.0)
        m = float(np.mean(subset))
        ax_d.text(j, m + 0.03, f"mu={m:+.2f}", ha="center", fontsize=9,
                  fontweight="bold")
    ax_d.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax_d.set_xticks(range(len(order)))
    ax_d.set_xticklabels(order, fontsize=9)
    ax_d.set_ylabel("cos(theta) detrended", fontsize=9)
    ax_d.set_title(
        "D  Detrended bias by initial distance\n"
        f"Near mu={near_far['mean_cos_near']:+.3f} | "
        f"Far mu={near_far['mean_cos_far']:+.3f}",
        fontsize=10,
    )

    plt.suptitle(
        "Detrended Directional Place-Field Shift Analysis\n"
        f"(N={len(df)} neurons; global drift subtracted before cosine)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "shift_detrended_4panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shift_detrended_4panel.png  <- main figure")


def main() -> None:
    out_dir = ensure_dir(OUT_DIR)
    fig_dir = ensure_dir(out_dir / "figures")

    print("Loading raw shift vectors...")
    df_raw = load_shift_vectors()
    print(f"  Loaded {len(df_raw)} neurons.")

    print("Detrending by global mean drift...")
    df, drift = detrend(df_raw)
    print(f"  Mean drift: dx={drift['mean_dx_raw']:+.4f}, dy={drift['mean_dy_raw']:+.4f}")
    print(f"  Drift magnitude: {drift['drift_magnitude']:.4f}, "
          f"angle: {drift['drift_angle_deg']:+.1f} deg")
    print(f"  Neurons retained: {drift['n_after_filter']}")

    print("Computing statistics...")
    stats_det = cosine_statistics(
        df["cos_theta_det"].to_numpy(dtype=float),
        df["rel_angle_det"].to_numpy(dtype=float),
        df["shift_angle_det"].to_numpy(dtype=float),
    )
    stats_raw = cosine_statistics(
        df_raw["cos_theta"].to_numpy(dtype=float),
        df_raw["rel_angle"].to_numpy(dtype=float),
        np.arctan2(df_raw["dy"].to_numpy(dtype=float),
                   df_raw["dx"].to_numpy(dtype=float)),
    )
    near_far = near_far_breakdown(df, "cos_theta_det")

    print("\n-- Detrended Results ----------------------------------------")
    print(f"  Raw mean cos       : {stats_raw['mean_cosine']:+.4f}  "
          f"[{stats_raw['bootstrap_ci_95'][0]:+.3f}, {stats_raw['bootstrap_ci_95'][1]:+.3f}]")
    print(f"  Detrended mean cos : {stats_det['mean_cosine']:+.4f}  "
          f"[{stats_det['bootstrap_ci_95'][0]:+.3f}, {stats_det['bootstrap_ci_95'][1]:+.3f}]")
    print(f"  Detrended frac toward reward: {stats_det['fraction_toward_reward']*100:.1f}%")
    print(f"  Detrended t-test p : {stats_det['ttest_cos_vs_0']['p_one_tailed']:.5f}")
    print(f"  Detrended perm p   : {stats_det['permutation_test']['p']:.5f}")
    print(f"  Rayleigh (rel)     : R={stats_det['rayleigh_toward_reward']['R']:.4f}, "
          f"p={stats_det['rayleigh_toward_reward']['p']:.5f}")
    print(f"  Near (d<={near_far['q25_dist']:.2f}) mean cos: "
          f"{near_far['mean_cos_near']:+.4f}  (n={near_far['n_near']})")
    print(f"  Far  (d>={near_far['q75_dist']:.2f}) mean cos: "
          f"{near_far['mean_cos_far']:+.4f}  (n={near_far['n_far']})")
    print("-------------------------------------------------------------")

    print("\nGenerating figures...")
    plot_drift_arrow(drift["mean_dx_raw"], drift["mean_dy_raw"], df, fig_dir)
    plot_raw_vs_detrended_compare(
        df_raw["cos_theta"].to_numpy(dtype=float),
        df["cos_theta_det"].to_numpy(dtype=float),
        stats_raw, stats_det, fig_dir,
    )
    plot_4panel(df, stats_det, near_far, drift, fig_dir)

    df[["neuron_id", "bx", "by", "ax", "ay",
        "dx", "dy", "dx_det", "dy_det",
        "cos_theta", "cos_theta_det",
        "rel_angle", "rel_angle_det",
        "dist_before", "dist_after"]].to_csv(
        out_dir / "shift_detrended_vectors.csv", index=False
    )
    save_json(out_dir / "shift_detrended_results.json", {
        "drift": drift,
        "statistics_raw": stats_raw,
        "statistics_detrended": stats_det,
        "near_far_detrended": near_far,
    })
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
