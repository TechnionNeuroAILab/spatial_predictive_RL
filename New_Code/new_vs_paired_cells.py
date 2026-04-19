"""
Tier 1.3 - New vs paired place-cell analysis.

The 46 cells recruited after RL (Phase 2 only) are the actual driver of the
over-representation effect; the 208 cells present in both phases mostly track
occupancy drift. This script separates the populations and characterises:
  - where newly recruited cells are located relative to the reward,
  - the distribution of their Skaggs scores,
  - how their spatial density compares to the paired population,
  - a formal Mann-Whitney test: are new cells closer to reward than paired?

Inputs : data/df_centers_{before,after}.pkl
         data/df_skaggs_{before,after}.pkl
Outputs: ../New_Results/new_vs_paired_cells/
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp

matplotlib.use("Agg")

REWARD = np.array([0.5, 0.5], dtype=float)
GOAL_RADIUS = 0.12
DATA_DIR = Path("data")
OUT_DIR = Path("../New_Results/new_vs_paired_cells")
N_BOOT = 5_000
SEED = 42
RNG = np.random.default_rng(SEED)

C_PAIRED = "#2E5FA3"
C_NEW = "#E63946"
C_LOST = "#8D99AE"
C_GOLD = "#F4D03F"


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
    df = pd.read_pickle(path)
    df = df[df["k"] == 1].copy()
    rows = []
    for _, r in df.iterrows():
        xy = center_to_xy(r["center"])
        if xy is None:
            continue
        rows.append({"neuron_id": int(r["neuron_id"]), "cx": xy[0], "cy": xy[1]})
    return pd.DataFrame(rows).drop_duplicates("neuron_id")


def dist_reward(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt((x - REWARD[0])**2 + (y - REWARD[1])**2)


def bootstrap_mean_ci(v: np.ndarray) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([np.mean(RNG.choice(v, size=v.size, replace=True))
                      for _ in range(N_BOOT)])
    return float(np.mean(v)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def classify_cells(ctr_b: pd.DataFrame, ctr_a: pd.DataFrame,
                   sk_b: pd.DataFrame, sk_a: pd.DataFrame) -> dict:
    ids_b = set(ctr_b["neuron_id"])
    ids_a = set(ctr_a["neuron_id"])
    ids_paired = sorted(ids_b & ids_a)
    ids_new = sorted(ids_a - ids_b)
    ids_lost = sorted(ids_b - ids_a)

    paired_after = ctr_a[ctr_a["neuron_id"].isin(ids_paired)].merge(
        sk_a, on="neuron_id", how="left"
    )
    paired_after["dist_reward"] = dist_reward(paired_after["cx"].to_numpy(),
                                              paired_after["cy"].to_numpy())
    paired_after["group"] = "paired"

    new_after = ctr_a[ctr_a["neuron_id"].isin(ids_new)].merge(
        sk_a, on="neuron_id", how="left"
    )
    new_after["dist_reward"] = dist_reward(new_after["cx"].to_numpy(),
                                           new_after["cy"].to_numpy())
    new_after["group"] = "new"

    lost_before = ctr_b[ctr_b["neuron_id"].isin(ids_lost)].merge(
        sk_b, on="neuron_id", how="left"
    )
    lost_before["dist_reward"] = dist_reward(lost_before["cx"].to_numpy(),
                                             lost_before["cy"].to_numpy())
    lost_before["group"] = "lost"

    return {
        "paired_after": paired_after,
        "new_after": new_after,
        "lost_before": lost_before,
        "counts": {
            "n_before": len(ids_b),
            "n_after": len(ids_a),
            "n_paired": len(ids_paired),
            "n_new": len(ids_new),
            "n_lost": len(ids_lost),
        },
    }


def compare_populations(new_df: pd.DataFrame, paired_df: pd.DataFrame) -> dict:
    d_new = new_df["dist_reward"].to_numpy(dtype=float)
    d_p = paired_df["dist_reward"].to_numpy(dtype=float)
    s_new = new_df["skaggs_score"].to_numpy(dtype=float)
    s_p = paired_df["skaggs_score"].to_numpy(dtype=float)
    s_new = s_new[np.isfinite(s_new)]; s_p = s_p[np.isfinite(s_p)]

    mw_dist = mannwhitneyu(d_new, d_p, alternative="less") if d_new.size > 2 and d_p.size > 2 else (np.nan, np.nan)
    ks_dist = ks_2samp(d_new, d_p) if d_new.size > 2 and d_p.size > 2 else (np.nan, np.nan)
    mw_sk = mannwhitneyu(s_new, s_p, alternative="two-sided") if s_new.size > 2 and s_p.size > 2 else (np.nan, np.nan)

    mean_new, lo_new, hi_new = bootstrap_mean_ci(d_new)
    mean_p, lo_p, hi_p = bootstrap_mean_ci(d_p)

    # Proportion near reward (inside goal zone)
    frac_near_new = float(np.mean(d_new <= GOAL_RADIUS)) if d_new.size else float("nan")
    frac_near_p = float(np.mean(d_p <= GOAL_RADIUS)) if d_p.size else float("nan")

    return {
        "distance_to_reward": {
            "new_mean": mean_new, "new_ci": [lo_new, hi_new], "n_new": int(d_new.size),
            "paired_mean": mean_p, "paired_ci": [lo_p, hi_p], "n_paired": int(d_p.size),
            "mannwhitney_u": float(mw_dist[0]) if np.isfinite(mw_dist[0]) else None,
            "p_new_closer": float(mw_dist[1]) if np.isfinite(mw_dist[1]) else None,
            "ks_stat": float(ks_dist[0]) if np.isfinite(ks_dist[0]) else None,
            "ks_p": float(ks_dist[1]) if np.isfinite(ks_dist[1]) else None,
        },
        "skaggs_comparison": {
            "new_mean": float(np.mean(s_new)) if s_new.size else float("nan"),
            "paired_mean": float(np.mean(s_p)) if s_p.size else float("nan"),
            "mannwhitney_u": float(mw_sk[0]) if np.isfinite(mw_sk[0]) else None,
            "p_two_sided": float(mw_sk[1]) if np.isfinite(mw_sk[1]) else None,
        },
        "fraction_inside_goal_zone": {
            "new": frac_near_new, "paired": frac_near_p,
            "ratio_new_over_paired": float(frac_near_new / frac_near_p)
            if frac_near_p > 0 else None,
        },
    }


def plot_spatial_scatter(groups: dict, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(groups["paired_after"]["cx"], groups["paired_after"]["cy"],
               s=18, color=C_PAIRED, alpha=0.55,
               label=f"Paired (n={groups['counts']['n_paired']})")
    ax.scatter(groups["new_after"]["cx"], groups["new_after"]["cy"],
               s=45, color=C_NEW, alpha=0.85, marker="^",
               edgecolor="black", linewidth=0.5,
               label=f"New (n={groups['counts']['n_new']})")
    ax.scatter(groups["lost_before"]["cx"], groups["lost_before"]["cy"],
               s=28, color=C_LOST, alpha=0.5, marker="x",
               label=f"Lost (n={groups['counts']['n_lost']})")
    circle = plt.Circle(REWARD, GOAL_RADIUS, fill=False, edgecolor=C_GOLD, lw=2, ls="--")
    ax.add_patch(circle)
    ax.scatter(*REWARD, s=280, marker="*", color=C_GOLD, zorder=6, label="Reward")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Place-cell centres by recruitment class")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "recruitment_spatial_scatter.png", dpi=150)
    plt.close()
    print("  Saved: recruitment_spatial_scatter.png")


def plot_distance_distributions(groups: dict, comp: dict, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bins = np.linspace(0, np.sqrt(2) / 2 + 0.05, 25)
    ax.hist(groups["paired_after"]["dist_reward"], bins=bins, alpha=0.55,
            color=C_PAIRED, density=True,
            label=f"Paired (n={groups['counts']['n_paired']})")
    ax.hist(groups["new_after"]["dist_reward"], bins=bins, alpha=0.65,
            color=C_NEW, density=True,
            label=f"New (n={groups['counts']['n_new']})")
    dd = comp["distance_to_reward"]
    ax.axvline(dd["paired_mean"], color=C_PAIRED, lw=2,
               label=f"Paired mean={dd['paired_mean']:.3f}")
    ax.axvline(dd["new_mean"], color=C_NEW, lw=2,
               label=f"New mean={dd['new_mean']:.3f}")
    ax.axvline(GOAL_RADIUS, color=C_GOLD, lw=1.5, ls="--", label="Goal radius")
    p_val = dd.get("p_new_closer")
    p_str = f"{p_val:.4f}" if p_val is not None else "n/a"
    ax.set_title(f"Distance to reward: new vs paired\nMann-Whitney p(new<paired) = {p_str}")
    ax.set_xlabel("Distance to reward"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "distance_histograms.png", dpi=150)
    plt.close()
    print("  Saved: distance_histograms.png")


def plot_inside_zone_bar(comp: dict, fig_dir: Path) -> None:
    dz = comp["fraction_inside_goal_zone"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(["Paired", "New"], [dz["paired"], dz["new"]],
                  color=[C_PAIRED, C_NEW], edgecolor="black", width=0.6)
    for b, v in zip(bars, [dz["paired"], dz["new"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Fraction of cells with centre inside goal zone")
    ratio = dz.get("ratio_new_over_paired")
    ratio_str = f"{ratio:.2f}x" if ratio is not None else "n/a"
    ax.set_title(f"Over-representation inside goal zone\n(new/paired ratio = {ratio_str})")
    plt.tight_layout()
    plt.savefig(fig_dir / "inside_goal_zone_bar.png", dpi=150)
    plt.close()
    print("  Saved: inside_goal_zone_bar.png")


def plot_skaggs_comparison(groups: dict, comp: dict, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    s_new = groups["new_after"]["skaggs_score"].dropna().to_numpy(dtype=float)
    s_p = groups["paired_after"]["skaggs_score"].dropna().to_numpy(dtype=float)
    parts = ax.violinplot([s_p, s_new], positions=[0, 1], showmedians=True,
                          widths=0.7)
    for pc, c in zip(parts["bodies"], [C_PAIRED, C_NEW]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts[part].set_color("black"); parts[part].set_linewidth(1.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Paired (after)", "New (after)"])
    ax.set_ylabel("Skaggs score (bits/spike)")
    p_val = comp["skaggs_comparison"].get("p_two_sided")
    p_str = f"{p_val:.4f}" if p_val is not None else "n/a"
    ax.set_title(f"Skaggs score by recruitment class\nMann-Whitney p = {p_str}")
    plt.tight_layout()
    plt.savefig(fig_dir / "skaggs_by_recruitment_class.png", dpi=150)
    plt.close()
    print("  Saved: skaggs_by_recruitment_class.png")


def plot_4panel(groups: dict, comp: dict, fig_dir: Path) -> None:
    fig = plt.figure(figsize=(14, 11))

    ax_a = fig.add_subplot(2, 2, 1)
    ax_a.scatter(groups["paired_after"]["cx"], groups["paired_after"]["cy"],
                 s=16, color=C_PAIRED, alpha=0.55,
                 label=f"Paired (n={groups['counts']['n_paired']})")
    ax_a.scatter(groups["new_after"]["cx"], groups["new_after"]["cy"],
                 s=40, color=C_NEW, alpha=0.85, marker="^",
                 edgecolor="black", linewidth=0.5,
                 label=f"New (n={groups['counts']['n_new']})")
    circle = plt.Circle(REWARD, GOAL_RADIUS, fill=False, edgecolor=C_GOLD, lw=2, ls="--")
    ax_a.add_patch(circle)
    ax_a.scatter(*REWARD, s=220, marker="*", color=C_GOLD, zorder=6)
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.set_aspect("equal")
    ax_a.set_xlabel("x"); ax_a.set_ylabel("y")
    ax_a.set_title("A  Post-RL centres: paired vs newly recruited")
    ax_a.legend(fontsize=9)

    ax_b = fig.add_subplot(2, 2, 2)
    bins = np.linspace(0, np.sqrt(2) / 2 + 0.05, 22)
    ax_b.hist(groups["paired_after"]["dist_reward"], bins=bins, alpha=0.55,
              color=C_PAIRED, density=True, label="Paired")
    ax_b.hist(groups["new_after"]["dist_reward"], bins=bins, alpha=0.65,
              color=C_NEW, density=True, label="New")
    dd = comp["distance_to_reward"]
    ax_b.axvline(dd["paired_mean"], color=C_PAIRED, lw=2)
    ax_b.axvline(dd["new_mean"], color=C_NEW, lw=2)
    ax_b.axvline(GOAL_RADIUS, color=C_GOLD, lw=1.5, ls="--")
    p_val = dd.get("p_new_closer")
    p_str = f"{p_val:.4f}" if p_val is not None else "n/a"
    ax_b.set_title(f"B  Distance to reward\nMW p(new<paired) = {p_str}")
    ax_b.set_xlabel("Distance to reward"); ax_b.set_ylabel("Density")
    ax_b.legend(fontsize=9)

    ax_c = fig.add_subplot(2, 2, 3)
    dz = comp["fraction_inside_goal_zone"]
    ax_c.bar(["Paired", "New"], [dz["paired"], dz["new"]],
             color=[C_PAIRED, C_NEW], edgecolor="black", width=0.6)
    for i, v in enumerate([dz["paired"], dz["new"]]):
        ax_c.text(i, v + 0.01, f"{v*100:.1f}%", ha="center", fontsize=11,
                  fontweight="bold")
    ratio = dz.get("ratio_new_over_paired")
    ratio_str = f"{ratio:.2f}x" if ratio is not None else "n/a"
    ax_c.set_ylabel("Fraction of cells inside goal zone")
    ax_c.set_title(f"C  Inside goal zone\n(new/paired = {ratio_str})")

    ax_d = fig.add_subplot(2, 2, 4)
    s_new = groups["new_after"]["skaggs_score"].dropna().to_numpy(dtype=float)
    s_p = groups["paired_after"]["skaggs_score"].dropna().to_numpy(dtype=float)
    if s_new.size > 2 and s_p.size > 2:
        parts = ax_d.violinplot([s_p, s_new], positions=[0, 1],
                                showmedians=True, widths=0.7)
        for pc, c in zip(parts["bodies"], [C_PAIRED, C_NEW]):
            pc.set_facecolor(c); pc.set_alpha(0.7)
        for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[part].set_color("black"); parts[part].set_linewidth(1.2)
    ax_d.set_xticks([0, 1]); ax_d.set_xticklabels(["Paired", "New"])
    ax_d.set_ylabel("Skaggs score (bits/spike)")
    p_sk = comp["skaggs_comparison"].get("p_two_sided")
    p_sk_str = f"{p_sk:.4f}" if p_sk is not None else "n/a"
    ax_d.set_title(f"D  Skaggs score comparison\nMW p = {p_sk_str}")

    plt.suptitle(
        "Recruitment-Class Analysis: Paired vs Newly Recruited Place Cells",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "recruitment_4panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: recruitment_4panel.png  <- main figure")


def main() -> None:
    out_dir = ensure_dir(OUT_DIR)
    fig_dir = ensure_dir(out_dir / "figures")

    print("Loading centres and Skaggs scores...")
    ctr_b = load_centres_k1(DATA_DIR / "df_centers_before.pkl")
    ctr_a = load_centres_k1(DATA_DIR / "df_centers_after.pkl")
    sk_b = pd.read_pickle(DATA_DIR / "df_skaggs_before.pkl")[["neuron_id", "skaggs_score"]]
    sk_a = pd.read_pickle(DATA_DIR / "df_skaggs_after.pkl")[["neuron_id", "skaggs_score"]]
    print(f"  Before: {len(ctr_b)} k=1 cells  After: {len(ctr_a)} k=1 cells")

    print("Classifying cells (paired / new / lost)...")
    groups = classify_cells(ctr_b, ctr_a, sk_b, sk_a)
    c = groups["counts"]
    print(f"  Paired: {c['n_paired']}  New: {c['n_new']}  Lost: {c['n_lost']}")

    print("Comparing populations...")
    comp = compare_populations(groups["new_after"], groups["paired_after"])
    dd = comp["distance_to_reward"]
    dz = comp["fraction_inside_goal_zone"]
    sk = comp["skaggs_comparison"]

    print("\n-- Recruitment Results --------------------------------------")
    print(f"  New cells (n={dd['n_new']})   dist_reward mean="
          f"{dd['new_mean']:.3f} CI[{dd['new_ci'][0]:.3f}, {dd['new_ci'][1]:.3f}]")
    print(f"  Paired cells (n={dd['n_paired']}) dist_reward mean="
          f"{dd['paired_mean']:.3f} CI[{dd['paired_ci'][0]:.3f}, {dd['paired_ci'][1]:.3f}]")
    p_clos = dd.get("p_new_closer")
    print(f"  MW p(new < paired distance): "
          f"{p_clos if p_clos is None else f'{p_clos:.5f}'}")
    ks_p = dd.get("ks_p")
    print(f"  KS p (distribution): {ks_p if ks_p is None else f'{ks_p:.5f}'}")
    _ratio_val = dz['ratio_new_over_paired']
    _ratio_str = "None" if _ratio_val is None else f"{_ratio_val:.2f}x"
    print(f"  Frac inside goal zone: new={dz['new']*100:.1f}%  "
          f"paired={dz['paired']*100:.1f}%  "
          f"ratio={_ratio_str}")
    p_sk = sk.get("p_two_sided")
    print(f"  Skaggs MW p: {p_sk if p_sk is None else f'{p_sk:.5f}'}")
    print("-------------------------------------------------------------")

    print("\nGenerating figures...")
    plot_spatial_scatter(groups, fig_dir)
    plot_distance_distributions(groups, comp, fig_dir)
    plot_inside_zone_bar(comp, fig_dir)
    plot_skaggs_comparison(groups, comp, fig_dir)
    plot_4panel(groups, comp, fig_dir)

    for name, df in [
        ("paired_after", groups["paired_after"]),
        ("new_after", groups["new_after"]),
        ("lost_before", groups["lost_before"]),
    ]:
        df.to_csv(out_dir / f"{name}.csv", index=False)

    save_json(out_dir / "new_vs_paired_results.json", {
        "counts": groups["counts"],
        "comparison": comp,
    })
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
