"""
Tier 1.4 - Reframed Skaggs interaction analysis.

The original Skaggs-vs-distance plot led on a flat post-RL Spearman gradient
(rho ~ -0.01, p=0.82) which reads as a null. The actual signal is the
distance-dependent loss of Skaggs information: near-reward cells lose less
information than far cells (delta_near = -0.026 vs delta_far = -0.064,
Mann-Whitney p = 2.5e-5).

This script makes that interaction the primary visual story:
  - paired before->after Skaggs lines colour-coded by distance zone,
  - near/mid/far delta bars with bootstrap 95% CIs,
  - dSkaggs vs distance scatter with binned means,
  - a compact stats summary panel.

Inputs : data/df_skaggs_{before,after}.pkl
         data/df_centers_{before,after}.pkl
Outputs: ../New_Results/skaggs_interaction_reframe/
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, ttest_rel, ttest_ind

matplotlib.use("Agg")

REWARD = np.array([0.5, 0.5], dtype=float)
GOAL_RADIUS = 0.12
DATA_DIR = Path("data")
OUT_DIR = Path("../New_Results/skaggs_interaction_reframe")
N_BOOT = 5_000
SEED = 42
RNG = np.random.default_rng(SEED)

C_NEAR = "#E63946"
C_MID = "#A8DADC"
C_FAR = "#457B9D"
C_GOLD = "#F4D03F"
C_BEFORE = "#2E5FA3"
C_AFTER = "#E07A2D"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def center_to_xy(c) -> tuple[float, float] | None:
    if c is None:
        return None
    try:
        arr = np.asarray(c, dtype=float)
    except Exception:
        return None
    if arr.ndim == 1 and arr.size == 2:
        return float(arr[0]), float(arr[1])
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
        return float(arr[0, 0]), float(arr[0, 1])
    return None


def load_centres_k1(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path); df = df[df["k"] == 1].copy()
    rows = []
    for _, r in df.iterrows():
        xy = center_to_xy(r["center"])
        if xy is None:
            continue
        rows.append({"neuron_id": int(r["neuron_id"]), "cx": xy[0], "cy": xy[1]})
    return pd.DataFrame(rows).drop_duplicates("neuron_id")


def bootstrap_mean_ci(v: np.ndarray) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([np.mean(RNG.choice(v, size=v.size, replace=True))
                      for _ in range(N_BOOT)])
    return float(np.mean(v)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_paired() -> pd.DataFrame:
    sk_b = pd.read_pickle(DATA_DIR / "df_skaggs_before.pkl")[["neuron_id", "skaggs_score"]]
    sk_a = pd.read_pickle(DATA_DIR / "df_skaggs_after.pkl")[["neuron_id", "skaggs_score"]]
    ctr_b = load_centres_k1(DATA_DIR / "df_centers_before.pkl")
    ctr_a = load_centres_k1(DATA_DIR / "df_centers_after.pkl")

    b = sk_b.merge(ctr_b, on="neuron_id", how="inner")
    b["dist_before"] = np.sqrt((b["cx"] - REWARD[0])**2 + (b["cy"] - REWARD[1])**2)
    b = b.rename(columns={"skaggs_score": "skaggs_before",
                          "cx": "bx", "cy": "by"})

    a = sk_a.merge(ctr_a, on="neuron_id", how="inner")
    a["dist_after"] = np.sqrt((a["cx"] - REWARD[0])**2 + (a["cy"] - REWARD[1])**2)
    a = a.rename(columns={"skaggs_score": "skaggs_after",
                          "cx": "ax", "cy": "ay"})

    paired = b.merge(a, on="neuron_id", how="inner")
    paired["delta"] = paired["skaggs_after"] - paired["skaggs_before"]
    return paired


def assign_zone(paired: pd.DataFrame) -> pd.DataFrame:
    q25 = float(paired["dist_before"].quantile(0.25))
    q75 = float(paired["dist_before"].quantile(0.75))
    zone = np.where(
        paired["dist_before"] <= q25, "Near",
        np.where(paired["dist_before"] >= q75, "Far", "Mid"),
    )
    paired = paired.copy()
    paired["zone"] = zone
    paired["q25_dist"] = q25
    paired["q75_dist"] = q75
    return paired


def compute_stats(paired: pd.DataFrame) -> dict:
    delta = paired["delta"].to_numpy(dtype=float)
    t_stat, p_t = ttest_rel(paired["skaggs_after"].to_numpy(dtype=float),
                            paired["skaggs_before"].to_numpy(dtype=float),
                            alternative="two-sided")
    m, lo, hi = bootstrap_mean_ci(delta)

    zones = {}
    for z in ["Near", "Mid", "Far"]:
        v = paired[paired["zone"] == z]["delta"].to_numpy(dtype=float)
        mm, llo, hhi = bootstrap_mean_ci(v)
        zones[z] = {"n": int(v.size), "mean_delta": mm, "ci_95": [llo, hhi]}

    near = paired[paired["zone"] == "Near"]["delta"].to_numpy(dtype=float)
    far = paired[paired["zone"] == "Far"]["delta"].to_numpy(dtype=float)
    mid = paired[paired["zone"] == "Mid"]["delta"].to_numpy(dtype=float)

    mw = mannwhitneyu(near, far, alternative="two-sided") if near.size > 2 and far.size > 2 else (np.nan, np.nan)
    t_nf = ttest_ind(near, far, equal_var=False) if near.size > 2 and far.size > 2 else (np.nan, np.nan)

    rho, p_rho = spearmanr(paired["dist_before"].to_numpy(dtype=float), delta)

    return {
        "n_paired": int(len(paired)),
        "q25_dist": float(paired["q25_dist"].iloc[0]),
        "q75_dist": float(paired["q75_dist"].iloc[0]),
        "overall_delta": {"mean": m, "ci_95": [lo, hi],
                          "ttest_t": float(t_stat), "ttest_p": float(p_t)},
        "by_zone": zones,
        "near_vs_far": {
            "mannwhitney_u": float(mw[0]) if np.isfinite(mw[0]) else None,
            "mannwhitney_p": float(mw[1]) if np.isfinite(mw[1]) else None,
            "welch_t": float(t_nf[0]) if np.isfinite(t_nf[0]) else None,
            "welch_p": float(t_nf[1]) if np.isfinite(t_nf[1]) else None,
            "mean_diff_near_minus_far": (
                float(np.mean(near) - np.mean(far))
                if near.size and far.size else None
            ),
        },
        "delta_vs_distance_spearman": {"rho": float(rho), "p": float(p_rho)},
    }


def plot_trajectories(paired: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"Near": C_NEAR, "Mid": C_MID, "Far": C_FAR}
    jitter_map = {"Near": -0.08, "Mid": 0.0, "Far": 0.08}
    rng = np.random.default_rng(0)
    for zone in ["Near", "Mid", "Far"]:
        sub = paired[paired["zone"] == zone]
        for _, r in sub.iterrows():
            x0 = 0 + jitter_map[zone] + rng.uniform(-0.02, 0.02)
            x1 = 1 + jitter_map[zone] + rng.uniform(-0.02, 0.02)
            ax.plot([x0, x1], [r["skaggs_before"], r["skaggs_after"]],
                    color=colors[zone], alpha=0.35, lw=0.7)
    for zone in ["Near", "Mid", "Far"]:
        sub = paired[paired["zone"] == zone]
        mb = sub["skaggs_before"].mean(); ma = sub["skaggs_after"].mean()
        ax.plot([0 + jitter_map[zone], 1 + jitter_map[zone]], [mb, ma],
                color=colors[zone], lw=3.0, marker="o", markersize=9,
                markeredgecolor="black", label=f"{zone} (n={len(sub)})")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Phase 1\n(before RL)", "Phase 2\n(after RL)"])
    ax.set_ylabel("Skaggs score (bits/spike)")
    ax.set_title("Skaggs trajectories by distance zone\n"
                 "(group means overlaid; thin lines = individual cells)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "skaggs_trajectories_by_zone.png", dpi=150)
    plt.close()
    print("  Saved: skaggs_trajectories_by_zone.png")


def plot_delta_bars(stats: dict, fig_dir: Path) -> None:
    zones = ["Near", "Mid", "Far"]
    colors = [C_NEAR, C_MID, C_FAR]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for i, z in enumerate(zones):
        d = stats["by_zone"][z]
        m, lo, hi = d["mean_delta"], d["ci_95"][0], d["ci_95"][1]
        ax.bar(i, m, color=colors[i], edgecolor="black", width=0.65,
               label=f"{z} (n={d['n']})")
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", color="black",
                    lw=1.6, capsize=5)
        ax.text(i, m + (0.002 if m >= 0 else -0.006),
                f"{m:+.3f}", ha="center",
                va="bottom" if m >= 0 else "top",
                fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(
        [f"Near\n(d<={stats['q25_dist']:.2f})", "Mid",
         f"Far\n(d>={stats['q75_dist']:.2f})"]
    )
    ax.set_ylabel("dSkaggs = after - before  (bits/spike)")
    nf = stats["near_vs_far"]
    p_mw = nf.get("mannwhitney_p")
    p_str = f"{p_mw:.2e}" if p_mw is not None else "n/a"
    ax.set_title(f"dSkaggs by distance zone\nNear vs Far: MW p = {p_str}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "delta_skaggs_bars.png", dpi=150)
    plt.close()
    print("  Saved: delta_skaggs_bars.png")


def plot_delta_vs_distance(paired: pd.DataFrame, stats: dict, fig_dir: Path) -> None:
    x = paired["dist_before"].to_numpy(dtype=float)
    y = paired["delta"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    zone_c = {"Near": C_NEAR, "Mid": C_MID, "Far": C_FAR}
    for z, c in zone_c.items():
        sub = paired[paired["zone"] == z]
        ax.scatter(sub["dist_before"], sub["delta"], s=22, color=c, alpha=0.65,
                   edgecolor="black", linewidth=0.3, label=z)

    bins = np.linspace(x.min(), x.max(), 9)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means, sems = [], []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        v = y[mask]
        if v.size > 1:
            means.append(np.mean(v))
            sems.append(np.std(v, ddof=1) / np.sqrt(v.size))
        else:
            means.append(np.nan); sems.append(np.nan)
    ax.errorbar(centers, means, yerr=sems, color="black", marker="o", lw=2,
                capsize=4, label="Binned mean +/- SEM")

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(GOAL_RADIUS, color=C_GOLD, lw=1.5, ls="--", alpha=0.7,
               label="Goal radius")
    sp = stats["delta_vs_distance_spearman"]
    ax.set_xlabel("Distance to reward (before RL)")
    ax.set_ylabel("dSkaggs = after - before")
    ax.set_title(f"dSkaggs vs distance-from-reward\nSpearman rho={sp['rho']:+.3f}, p={sp['p']:.2e}")
    ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig(fig_dir / "delta_vs_distance_scatter.png", dpi=150)
    plt.close()
    print("  Saved: delta_vs_distance_scatter.png")


def plot_4panel(paired: pd.DataFrame, stats: dict, fig_dir: Path) -> None:
    fig = plt.figure(figsize=(14, 11))
    colors = {"Near": C_NEAR, "Mid": C_MID, "Far": C_FAR}

    ax_a = fig.add_subplot(2, 2, 1)
    jitter_map = {"Near": -0.08, "Mid": 0.0, "Far": 0.08}
    rng = np.random.default_rng(0)
    for zone in ["Near", "Mid", "Far"]:
        sub = paired[paired["zone"] == zone]
        for _, r in sub.iterrows():
            x0 = 0 + jitter_map[zone] + rng.uniform(-0.015, 0.015)
            x1 = 1 + jitter_map[zone] + rng.uniform(-0.015, 0.015)
            ax_a.plot([x0, x1], [r["skaggs_before"], r["skaggs_after"]],
                      color=colors[zone], alpha=0.3, lw=0.6)
    for zone in ["Near", "Mid", "Far"]:
        sub = paired[paired["zone"] == zone]
        mb = sub["skaggs_before"].mean(); ma = sub["skaggs_after"].mean()
        ax_a.plot([0 + jitter_map[zone], 1 + jitter_map[zone]], [mb, ma],
                  color=colors[zone], lw=3.0, marker="o", markersize=9,
                  markeredgecolor="black", label=f"{zone} (n={len(sub)})")
    ax_a.set_xticks([0, 1]); ax_a.set_xticklabels(["Before RL", "After RL"])
    ax_a.set_ylabel("Skaggs score (bits/spike)")
    ax_a.set_title("A  Skaggs trajectories by distance zone")
    ax_a.legend(fontsize=9)

    ax_b = fig.add_subplot(2, 2, 2)
    zones = ["Near", "Mid", "Far"]
    for i, z in enumerate(zones):
        d = stats["by_zone"][z]
        m, lo, hi = d["mean_delta"], d["ci_95"][0], d["ci_95"][1]
        ax_b.bar(i, m, color=colors[z], edgecolor="black", width=0.65)
        ax_b.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none",
                      color="black", lw=1.6, capsize=5)
        ax_b.text(i, m + (0.002 if m >= 0 else -0.006), f"{m:+.3f}",
                  ha="center", va="bottom" if m >= 0 else "top",
                  fontsize=10, fontweight="bold")
    ax_b.axhline(0, color="black", lw=0.8)
    ax_b.set_xticks(range(3))
    ax_b.set_xticklabels([f"Near\n(d<={stats['q25_dist']:.2f})", "Mid",
                          f"Far\n(d>={stats['q75_dist']:.2f})"])
    ax_b.set_ylabel("dSkaggs")
    p_mw = stats["near_vs_far"].get("mannwhitney_p")
    p_str = f"{p_mw:.2e}" if p_mw is not None else "n/a"
    ax_b.set_title(f"B  dSkaggs by zone (bootstrap 95% CI)\nNear vs Far MW p = {p_str}")

    ax_c = fig.add_subplot(2, 2, 3)
    for z in zones:
        sub = paired[paired["zone"] == z]
        ax_c.scatter(sub["dist_before"], sub["delta"], s=18, color=colors[z],
                     alpha=0.6, edgecolor="black", linewidth=0.3, label=z)
    x = paired["dist_before"].to_numpy(dtype=float)
    y = paired["delta"].to_numpy(dtype=float)
    bins = np.linspace(x.min(), x.max(), 9)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means, sems = [], []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        v = y[mask]
        if v.size > 1:
            means.append(np.mean(v)); sems.append(np.std(v, ddof=1) / np.sqrt(v.size))
        else:
            means.append(np.nan); sems.append(np.nan)
    ax_c.errorbar(centers, means, yerr=sems, color="black", marker="o",
                  lw=2, capsize=4, label="Binned mean")
    ax_c.axhline(0, color="black", lw=0.8, ls="--")
    ax_c.axvline(GOAL_RADIUS, color=C_GOLD, lw=1.5, ls="--", alpha=0.7)
    sp = stats["delta_vs_distance_spearman"]
    ax_c.set_xlabel("Distance to reward (before RL)")
    ax_c.set_ylabel("dSkaggs")
    ax_c.set_title(f"C  dSkaggs vs distance\nSpearman rho={sp['rho']:+.3f}, p={sp['p']:.2e}")
    ax_c.legend(fontsize=8, loc="lower left")

    ax_d = fig.add_subplot(2, 2, 4); ax_d.axis("off")
    nf = stats["near_vs_far"]
    ov = stats["overall_delta"]
    by = stats["by_zone"]
    lines = [
        f"N paired cells: {stats['n_paired']}",
        f"Zones split at quartiles of dist_before:",
        f"   q25 = {stats['q25_dist']:.3f}   q75 = {stats['q75_dist']:.3f}",
        "",
        f"Overall dSkaggs:  {ov['mean']:+.4f}  [95% CI {ov['ci_95'][0]:+.4f}, {ov['ci_95'][1]:+.4f}]",
        f"   paired t-test  t={ov['ttest_t']:+.2f}  p={ov['ttest_p']:.2e}",
        "",
        f"By zone (mean dSkaggs):",
        f"   Near (n={by['Near']['n']}):  {by['Near']['mean_delta']:+.4f}",
        f"   Mid  (n={by['Mid']['n']}):  {by['Mid']['mean_delta']:+.4f}",
        f"   Far  (n={by['Far']['n']}):  {by['Far']['mean_delta']:+.4f}",
        "",
        f"Near vs Far (interaction test):",
        f"   MW p = {nf['mannwhitney_p']}",
        f"   Welch t p = {nf['welch_p']}",
        f"   mean diff (Near-Far) = {nf['mean_diff_near_minus_far']}",
    ]
    ax_d.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=9,
              va="top", ha="left")
    ax_d.set_title("D  Statistical summary", loc="left")

    plt.suptitle("Skaggs Information Loss by Distance Zone (Interaction Analysis)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_dir / "skaggs_interaction_4panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: skaggs_interaction_4panel.png  <- main figure")


def main() -> None:
    out_dir = ensure_dir(OUT_DIR)
    fig_dir = ensure_dir(out_dir / "figures")

    print("Loading data and building paired dataset...")
    paired = build_paired()
    paired = assign_zone(paired)
    print(f"  Paired cells: {len(paired)}")
    print(f"  Zones: Near={int((paired['zone']=='Near').sum())}  "
          f"Mid={int((paired['zone']=='Mid').sum())}  "
          f"Far={int((paired['zone']=='Far').sum())}")

    print("Computing statistics...")
    stats = compute_stats(paired)
    print("\n-- Interaction Results --------------------------------------")
    print(f"  Overall dSkaggs: {stats['overall_delta']['mean']:+.4f}  "
          f"[{stats['overall_delta']['ci_95'][0]:+.4f}, "
          f"{stats['overall_delta']['ci_95'][1]:+.4f}]  "
          f"t-test p={stats['overall_delta']['ttest_p']:.2e}")
    for z in ["Near", "Mid", "Far"]:
        d = stats["by_zone"][z]
        print(f"  {z}: n={d['n']}  mean dSkaggs={d['mean_delta']:+.4f}  "
              f"CI[{d['ci_95'][0]:+.4f}, {d['ci_95'][1]:+.4f}]")
    nf = stats["near_vs_far"]
    print(f"  Near vs Far: MW p={nf['mannwhitney_p']}  "
          f"Welch p={nf['welch_p']}")
    sp = stats["delta_vs_distance_spearman"]
    print(f"  Spearman(dSkaggs, dist_before): rho={sp['rho']:+.4f}, p={sp['p']:.4f}")
    print("-------------------------------------------------------------")

    print("\nGenerating figures...")
    plot_trajectories(paired, fig_dir)
    plot_delta_bars(stats, fig_dir)
    plot_delta_vs_distance(paired, stats, fig_dir)
    plot_4panel(paired, stats, fig_dir)

    paired[["neuron_id", "bx", "by", "ax", "ay",
            "skaggs_before", "skaggs_after", "delta",
            "dist_before", "dist_after", "zone"]].to_csv(
        out_dir / "paired_skaggs.csv", index=False
    )
    save_json(out_dir / "skaggs_interaction_results.json", stats)
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
