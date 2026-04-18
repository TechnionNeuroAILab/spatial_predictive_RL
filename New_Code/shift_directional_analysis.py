from __future__ import annotations
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
matplotlib.use("Agg")
REWARD = np.array([0.5, 0.5], dtype=float)
GOAL_RADIUS = 0.12          
SKAGGS_THRESHOLD = 0.05     
DATA_DIR = Path("data")
OUT_DIR = Path("../New_Results/shift_directional_analysis")
N_PERM = 10_000
N_BOOT = 5_000
SEED = 42
RNG = np.random.default_rng(SEED)

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


def load_centers_k1(pkl_path: Path) -> pd.DataFrame:
    df = pd.read_pickle(pkl_path)
    df = df[df["k"] == 1].copy()
    rows = []
    for _, row in df.iterrows():
        xy = center_to_xy(row["center"])
        if xy is None:
            continue
        rows.append({"neuron_id": int(row["neuron_id"]), "cx": xy[0], "cy": xy[1]})
    return pd.DataFrame(rows).drop_duplicates("neuron_id")


def load_skaggs(pkl_path: Path) -> pd.DataFrame:
    return pd.read_pickle(pkl_path)[["neuron_id", "skaggs_score"]]


def rayleigh_test(angles_rad: np.ndarray) -> tuple[float, float]:
    n = angles_rad.size
    if n < 3:
        return np.nan, np.nan
    C = np.mean(np.cos(angles_rad))
    S = np.mean(np.sin(angles_rad))
    R = np.sqrt(C**2 + S**2)
    # Rayleigh's z statistic
    z = n * R**2
    # Approximation for p-value (Zar 2010, eq. 27.4)
    p = np.exp(-z) * (1.0 + (2.0*z - z**2) / (4.0*n) - (24.0*z - 132.0*z**2 + 76.0*z**3 - 9.0*z**4) / (288.0*n**2))
    p = float(np.clip(p, 0.0, 1.0))
    return float(R), p


def load_paired_shifts() -> pd.DataFrame:
    """
    Load paired shift data: neurons that have k=1 place fields in BOTH
    before and after, passing the Skaggs threshold before RL.
    Returns DataFrame with columns:
        neuron_id, bx, by, ax, ay,
        dx, dy,          # shift vector
        toward_x, toward_y,   # unit vector toward reward from before-center
        cos_theta,       # cosine between shift and toward-reward
        shift_angle,     # absolute angle of shift vector (radians)
        rel_angle,       # shift angle relative to toward-reward direction
        shift_mag,       # Euclidean magnitude of shift
        dist_before,     # Euclidean distance from reward at before-center
        dist_after,      # Euclidean distance from reward at after-center
        scalar_shift     # dist_before - dist_after (positive = closer)
    """
    print("Loading data …")
    centers_b = load_centers_k1(DATA_DIR / "df_centers_before.pkl")
    centers_a = load_centers_k1(DATA_DIR / "df_centers_after.pkl")
    skaggs_b  = load_skaggs(DATA_DIR / "df_skaggs_before.pkl")
    # Keep only neurons with high enough Skaggs score before RL
    valid_ids = skaggs_b.loc[skaggs_b["skaggs_score"] >= SKAGGS_THRESHOLD, "neuron_id"]
    centers_b = centers_b[centers_b["neuron_id"].isin(valid_ids)]
    # Inner join: only neurons present in both
    merged = centers_b.merge(
        centers_a.rename(columns={"cx": "ax", "cy": "ay"}),
        on="neuron_id", how="inner"
    ).rename(columns={"cx": "bx", "cy": "by"})

    print(f"  Neurons tracked (k=1 in both phases, Skaggs≥{SKAGGS_THRESHOLD}): {len(merged)}")

    # Shift vector
    merged["dx"] = merged["ax"] - merged["bx"]
    merged["dy"] = merged["ay"] - merged["by"]
    merged["shift_mag"] = np.sqrt(merged["dx"]**2 + merged["dy"]**2)
    # Only keep neurons whose fields actually moved (avoid zero-division)
    merged = merged[merged["shift_mag"] > 1e-8].copy()
    print(f"  Neurons with non-zero shift: {len(merged)}")
    # Unit vector toward reward from before-center
    to_reward_x = REWARD[0] - merged["bx"]
    to_reward_y = REWARD[1] - merged["by"]
    to_reward_norm = np.sqrt(to_reward_x**2 + to_reward_y**2)
    # Edge case: neuron already AT reward — skip
    merged = merged[to_reward_norm > 1e-8].copy()
    to_reward_x = (REWARD[0] - merged["bx"]) / (np.sqrt((REWARD[0]-merged["bx"])**2+(REWARD[1]-merged["by"])**2))
    to_reward_y = (REWARD[1] - merged["by"]) / (np.sqrt((REWARD[0]-merged["bx"])**2+(REWARD[1]-merged["by"])**2))

    # Cosine similarity
    unit_dx = merged["dx"] / merged["shift_mag"]
    unit_dy = merged["dy"] / merged["shift_mag"]
    merged["cos_theta"] = unit_dx * to_reward_x + unit_dy * to_reward_y

    # Absolute angle of shift and relative angle
    merged["shift_angle"] = np.arctan2(merged["dy"], merged["dx"])
    toward_angle = np.arctan2(to_reward_y, to_reward_x)
    # Relative angle: 0 = exactly toward reward
    rel = merged["shift_angle"] - toward_angle
    # Wrap to [-π, π]
    merged["rel_angle"] = (rel + np.pi) % (2 * np.pi) - np.pi

    # Distances
    merged["dist_before"] = np.sqrt((merged["bx"] - REWARD[0])**2 + (merged["by"] - REWARD[1])**2)
    merged["dist_after"]  = np.sqrt((merged["ax"] - REWARD[0])**2 + (merged["ay"] - REWARD[1])**2)
    merged["scalar_shift"] = merged["dist_before"] - merged["dist_after"]

    return merged.reset_index(drop=True)


def compute_statistics(df: pd.DataFrame) -> dict:
    cosines = df["cos_theta"].to_numpy(dtype=float)
    rel_angles = df["rel_angle"].to_numpy(dtype=float)
    n = len(cosines)
    # Rayleigh test on absolute shift directions
    R_all, p_ray_all = rayleigh_test(df["shift_angle"].to_numpy())
    # Rayleigh test on relative angles (toward-reward-centred)
    R_rel, p_ray_rel = rayleigh_test(rel_angles)
    # Mean cos and direction
    mean_cos = float(np.mean(cosines))
    mean_rel_angle_deg = float(np.degrees(np.arctan2(
        np.mean(np.sin(rel_angles)), np.mean(np.cos(rel_angles)))))

    # One-sample t-test: cos(θ) vs 0
    t_stat, p_ttest = ttest_1samp(cosines, popmean=0.0, alternative="greater")
    # Permutation test: randomly flip the sign of rel_angle_i
    obs_mean = mean_cos
    perm_means = np.empty(N_PERM, dtype=float)
    for i in range(N_PERM):
        signs = RNG.choice([-1.0, 1.0], size=n)
        perm_means[i] = float(np.mean(signs * cosines))
    perm_p = float(np.mean(perm_means >= obs_mean))

    # Bootstrap 95% CI on mean cos
    boot_means = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        sample = RNG.choice(cosines, size=n, replace=True)
        boot_means[i] = float(np.mean(sample))
    ci_low  = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # Fraction of neurons shifting toward reward (cos > 0)
    frac_toward = float(np.mean(cosines > 0))

    return {
        "n": int(n),
        "mean_cosine": mean_cos,
        "bootstrap_ci_95": [ci_low, ci_high],
        "mean_rel_angle_deg": mean_rel_angle_deg,
        "fraction_toward_reward": frac_toward,
        "rayleigh_all_directions": {"R": R_all, "p": p_ray_all},
        "rayleigh_toward_reward":  {"R": R_rel, "p": p_ray_rel},
        "ttest_cos_vs_0": {"t": float(t_stat), "p_one_tailed": float(p_ttest)},
        "permutation_test": {"observed_mean_cos": obs_mean, "p": perm_p,
                             "null_95th": float(np.percentile(perm_means, 95))},
    }


C_REWARD = "#E63946"   
C_AWAY   = "#457B9D"
C_GOLD   = "#F4D03F"

def _norm_to_cmap(vals: np.ndarray) -> np.ndarray:
    return (vals + 1.0) / 2.0


def plot_quiver_map(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = matplotlib.colormaps.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cosines = df["cos_theta"].to_numpy(dtype=float)
    colors  = cmap(norm(cosines))
    for _, row in df.iterrows():
        c = cmap(norm(row["cos_theta"]))
        ax.annotate(
            "", xy=(row["ax"], row["ay"]), xytext=(row["bx"], row["by"]),
            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, mutation_scale=6),
        )

    ax.scatter(df["bx"], df["by"], s=8, c="gray", alpha=0.5, zorder=3, label="Field centre (before)")
    circle = plt.Circle(REWARD, GOAL_RADIUS, fill=False, edgecolor=C_GOLD, lw=2, ls="--", zorder=5)
    ax.add_patch(circle)
    ax.scatter(*REWARD, s=200, marker="*", color=C_GOLD, zorder=6, label="Reward")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("cos(θ): toward reward →", fontsize=10)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x position"); ax.set_ylabel("y position")
    ax.set_title("A  Place-field shifts in arena space\n(arrows: before → after, colour = direction bias)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "shift_quiver_map.png", dpi=150)
    plt.close()
    print("  Saved: shift_quiver_map.png")


def plot_rose_diagram(df: pd.DataFrame, stats: dict, fig_dir: Path) -> None:
    rel_angles = df["rel_angle"].to_numpy(dtype=float)
    N = len(rel_angles)
    n_bins = 24
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(rel_angles, bins=bin_edges)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"projection": "polar"})

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = 2 * np.pi / n_bins
    # Colour bars by whether they are in the toward-reward half
    bar_colors = [C_REWARD if abs(c) < np.pi / 2 else C_AWAY for c in bin_centers]
    bars = ax.bar(bin_centers, counts, width=width * 0.85, bottom=0,
                  color=bar_colors, alpha=0.75, edgecolor="white", linewidth=0.5)
    # Expected count under uniform distribution
    uniform_count = N / n_bins
    ax.plot(np.linspace(-np.pi, np.pi, 200),
            [uniform_count] * 200, color="black", lw=1.5, ls="--", label="Uniform null")

    # Mean direction arrow
    mean_dir = stats["mean_rel_angle_deg"] * np.pi / 180
    max_count = float(np.max(counts))
    ax.annotate("", xy=(mean_dir, max_count * 0.9), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5))

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)     
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels(["→ Reward", "45°", "⊥", "135°", "← Away", "225°", "⊥", "315°"], fontsize=8)

    r_rel = stats["rayleigh_toward_reward"]["R"]
    p_rel = stats["rayleigh_toward_reward"]["p"]
    frac  = stats["fraction_toward_reward"]
    ax.set_title(
        f"B  Shift direction relative to reward\n"
        f"Rayleigh R={r_rel:.3f}, p={p_rel:.4f} | "
        f"{frac*100:.1f}% toward",
        fontsize=9, pad=15
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "shift_rose_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shift_rose_diagram.png")


def plot_cosine_distribution(df: pd.DataFrame, stats: dict, fig_dir: Path) -> None:
    cosines = df["cos_theta"].to_numpy(dtype=float)
    n = len(cosines)
    null_means = np.empty(N_PERM, dtype=float)
    for i in range(N_PERM):
        s = RNG.choice([-1.0, 1.0], size=n)
        null_means[i] = float(np.mean(s * cosines))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.hist(cosines, bins=30, density=True, color="#2E5FA3", alpha=0.65,
            label="Observed cos(θ) per neuron")
    ax.axvline(0, color="black", lw=1.2, ls="--", label="Null (cos=0)")
    ax.axvline(stats["mean_cosine"], color=C_REWARD, lw=2.5,
               label=f"Mean cos = {stats['mean_cosine']:.3f}")

    # 95% CI shading on mean from bootstrap
    ci = stats["bootstrap_ci_95"]
    ax.axvspan(ci[0], ci[1], alpha=0.15, color=C_REWARD, label="95% CI (bootstrap)")

    # Null distribution of means
    ax2 = ax.twinx()
    ax2.hist(null_means, bins=40, density=True, color="gray", alpha=0.35,
             label="Permutation null\n(mean cos)")
    ax2.set_ylabel("Null density", fontsize=9, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Statistics annotation
    mc = stats["mean_cosine"]
    p_t = stats["ttest_cos_vs_0"]["p_one_tailed"]
    p_perm = stats["permutation_test"]["p"]
    ax.set_xlabel("cos(θ): shift projected onto toward-reward direction", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"C  Cosine similarity: shift vs toward-reward direction\n"
        f"Mean={mc:.3f} | t-test p={p_t:.4f} | permutation p={p_perm:.4f}",
        fontsize=10
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "cosine_distribution.png", dpi=150)
    plt.close()
    print("  Saved: cosine_distribution.png")


def plot_cosine_by_distance(df: pd.DataFrame, fig_dir: Path) -> None:
    df = df.copy()
    q25 = float(df["dist_before"].quantile(0.25))
    q75 = float(df["dist_before"].quantile(0.75))
    df["distance_group"] = "Mid"
    df.loc[df["dist_before"] <= q25, "distance_group"] = f"Near\n(d ≤ {q25:.2f})"
    df.loc[df["dist_before"] >= q75, "distance_group"] = f"Far\n(d ≥ {q75:.2f})"

    group_order = [f"Near\n(d ≤ {q25:.2f})", "Mid", f"Far\n(d ≥ {q75:.2f})"]
    palette = {f"Near\n(d ≤ {q25:.2f})": C_REWARD, "Mid": "#A8DADC",
               f"Far\n(d ≥ {q75:.2f})": C_AWAY}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for j, grp in enumerate(group_order):
        subset = df[df["distance_group"] == grp]["cos_theta"].to_numpy()
        if len(subset) < 3:
            continue
        parts = ax.violinplot(subset, positions=[j], showmedians=True,
                              widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(palette[grp])
            pc.set_alpha(0.7)
        for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[part].set_color("black")
            parts[part].set_linewidth(1.2)

    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7,
               label="cos=0 (random direction)")
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order, fontsize=10)
    ax.set_ylabel("cos(θ): shift projected toward reward", fontsize=10)
    ax.set_title("D  Directional bias by initial distance from reward\n"
                 "(positive = shift toward reward)", fontsize=10)
    ax.legend(fontsize=8)

    # Annotate group means
    for j, grp in enumerate(group_order):
        subset = df[df["distance_group"] == grp]["cos_theta"].to_numpy()
        if len(subset) < 3:
            continue
        m = float(np.mean(subset))
        ax.text(j, m + 0.03, f"μ={m:.2f}", ha="center", va="bottom",
                fontsize=9, color="black", fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_dir / "cosine_by_distance.png", dpi=150)
    plt.close()
    print("  Saved: cosine_by_distance.png")


def plot_4panel(df: pd.DataFrame, stats: dict, fig_dir: Path) -> None:
    fig = plt.figure(figsize=(14, 12))
    ax_a = fig.add_subplot(2, 2, 1)
    cmap = matplotlib.colormaps.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    for _, row in df.iterrows():
        c = cmap(norm(row["cos_theta"]))
        ax_a.annotate(
            "", xy=(row["ax"], row["ay"]), xytext=(row["bx"], row["by"]),
            arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, mutation_scale=5),
        )
    ax_a.scatter(df["bx"], df["by"], s=6, c="gray", alpha=0.4, zorder=3)
    circle = plt.Circle(REWARD, GOAL_RADIUS, fill=False, edgecolor=C_GOLD, lw=2, ls="--", zorder=5)
    ax_a.add_patch(circle)
    ax_a.scatter(*REWARD, s=180, marker="*", color=C_GOLD, zorder=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_a, fraction=0.04, pad=0.02)
    cb.set_label("cos(θ)", fontsize=9)
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("x"); ax_a.set_ylabel("y")
    ax_a.set_title("A  Shift vectors in arena space\n(red=toward reward, blue=away)", fontsize=10)
    ax_b = fig.add_subplot(2, 2, 2, projection="polar")
    rel_angles = df["rel_angle"].to_numpy(dtype=float)
    N = len(rel_angles)
    n_bins = 24
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(rel_angles, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = 2 * np.pi / n_bins
    bar_colors = [C_REWARD if abs(c) < np.pi / 2 else C_AWAY for c in bin_centers]
    ax_b.bar(bin_centers, counts, width=width * 0.85, bottom=0,
             color=bar_colors, alpha=0.75, edgecolor="white", linewidth=0.5)
    uniform_count = N / n_bins
    ax_b.plot(np.linspace(-np.pi, np.pi, 200), [uniform_count] * 200,
              "k--", lw=1.5)
    mean_dir = stats["mean_rel_angle_deg"] * np.pi / 180
    max_c = float(np.max(counts))
    ax_b.annotate("", xy=(mean_dir, max_c * 0.85), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0))
    ax_b.set_theta_zero_location("E")
    ax_b.set_theta_direction(1)
    ax_b.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax_b.set_xticklabels(["→ Reward", "", "⊥", "", "← Away", "", "⊥", ""], fontsize=8)
    rr = stats["rayleigh_toward_reward"]
    ax_b.set_title(f"B  Shift direction (relative to reward)\n"
                   f"Rayleigh R={rr['R']:.3f}, p={rr['p']:.4f}", fontsize=10, pad=15)

    ax_c = fig.add_subplot(2, 2, 3)
    cosines = df["cos_theta"].to_numpy(dtype=float)
    null_means = np.empty(min(N_PERM, 2000), dtype=float)
    for i in range(len(null_means)):
        s = RNG.choice([-1.0, 1.0], size=len(cosines))
        null_means[i] = float(np.mean(s * cosines))

    ax_c.hist(cosines, bins=25, density=True, color="#2E5FA3", alpha=0.65,
              label="Observed cos(θ)")
    ax_c.axvline(0, color="black", lw=1.2, ls="--")
    ax_c.axvline(stats["mean_cosine"], color=C_REWARD, lw=2.5,
                 label=f"Mean = {stats['mean_cosine']:.3f}")
    ci = stats["bootstrap_ci_95"]
    ax_c.axvspan(ci[0], ci[1], alpha=0.15, color=C_REWARD, label="95% CI")

    ax_c2 = ax_c.twinx()
    ax_c2.hist(null_means, bins=30, density=True, color="gray", alpha=0.30,
               label="Permutation null (means)")
    ax_c2.set_ylabel("Null density", fontsize=8, color="gray")
    ax_c2.tick_params(axis="y", labelcolor="gray")

    p_t = stats["ttest_cos_vs_0"]["p_one_tailed"]
    p_perm = stats["permutation_test"]["p"]
    ax_c.set_xlabel("cos(θ)  [positive = toward reward]", fontsize=9)
    ax_c.set_ylabel("Density", fontsize=9)
    ax_c.set_title(f"C  Cosine distribution\n"
                   f"t-test p={p_t:.4f} | perm p={p_perm:.4f}", fontsize=10)
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax_d = fig.add_subplot(2, 2, 4)
    df2 = df.copy()
    q25 = float(df2["dist_before"].quantile(0.25))
    q75 = float(df2["dist_before"].quantile(0.75))
    df2["distance_group"] = "Mid"
    df2.loc[df2["dist_before"] <= q25, "distance_group"] = f"Near\n(d≤{q25:.2f})"
    df2.loc[df2["dist_before"] >= q75, "distance_group"] = f"Far\n(d≥{q75:.2f})"
    group_order = [f"Near\n(d≤{q25:.2f})", "Mid", f"Far\n(d≥{q75:.2f})"]
    palette = {f"Near\n(d≤{q25:.2f})": C_REWARD, "Mid": "#A8DADC",
               f"Far\n(d≥{q75:.2f})": C_AWAY}
    for j, grp in enumerate(group_order):
        subset = df2[df2["distance_group"] == grp]["cos_theta"].to_numpy()
        if len(subset) < 3:
            continue
        parts = ax_d.violinplot(subset, positions=[j], showmedians=True, widths=0.65)
        for pc in parts["bodies"]:
            pc.set_facecolor(palette[grp])
            pc.set_alpha(0.7)
        for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[part].set_color("black")
            parts[part].set_linewidth(1.0)
        m = float(np.mean(subset))
        ax_d.text(j, m + 0.03, f"μ={m:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax_d.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax_d.set_xticks(range(len(group_order)))
    ax_d.set_xticklabels(group_order, fontsize=9)
    ax_d.set_ylabel("cos(θ)", fontsize=9)
    ax_d.set_title("D  Directional bias by initial distance\n"
                   "(+= toward reward)", fontsize=10)

    plt.suptitle("Directional Place-Field Shift Analysis\n"
                 f"(N={len(df)} neurons, reward at centre of arena)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_dir / "shift_4panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shift_4panel.png  ← main figure")


def main() -> None:
    out_dir = ensure_dir(OUT_DIR)
    fig_dir = ensure_dir(out_dir / "figures")

    df = load_paired_shifts()
    stats = compute_statistics(df)
    print("\n── Results ──────────────────────────────────────────────────")
    print(f"  N neurons               : {stats['n']}")
    print(f"  Mean cos(θ)             : {stats['mean_cosine']:.4f}")
    print(f"  95% CI (bootstrap)      : [{stats['bootstrap_ci_95'][0]:.4f}, {stats['bootstrap_ci_95'][1]:.4f}]")
    print(f"  Fraction toward reward  : {stats['fraction_toward_reward']*100:.1f}%")
    print(f"  t-test (cos>0), p       : {stats['ttest_cos_vs_0']['p_one_tailed']:.5f}")
    print(f"  Permutation p           : {stats['permutation_test']['p']:.5f}")
    print(f"  Rayleigh R (rel. angle) : {stats['rayleigh_toward_reward']['R']:.4f}")
    print(f"  Rayleigh p (rel. angle) : {stats['rayleigh_toward_reward']['p']:.5f}")
    print("──────────────────────────────────────────────────────────────")

    print("\nGenerating figures …")
    plot_quiver_map(df, fig_dir)
    plot_rose_diagram(df, stats, fig_dir)
    plot_cosine_distribution(df, stats, fig_dir)
    plot_cosine_by_distance(df, fig_dir)
    plot_4panel(df, stats, fig_dir)
    df[["neuron_id", "bx", "by", "ax", "ay",
        "cos_theta", "rel_angle", "shift_mag",
        "dist_before", "dist_after", "scalar_shift"]].to_csv(
        out_dir / "shift_vectors.csv", index=False
    )

    save_json(out_dir / "shift_directional_results.json", stats)
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
