"""
Tier 1.2 - Occupancy-matched resampling.

Matches the (x,y) occupancy distributions between Phase 1 (before RL) and
Phase 2 (after RL) by subsampling timesteps inside each spatial bin to the
minimum count across phases. From the matched data we recompute:
  - per-neuron rate maps
  - Skaggs information scores
  - place-field centres (smoothed rate-map peak)
  - directional shift + cosine-with-reward statistics

This isolates representational change from behavioural change. The original
analysis was confounded by pre-RL left-biased exploration; matched occupancy
removes that confound at its source.

Inputs : data/df_behavior_{before,after}.pkl
         data/df_activations_{before,after}.pkl
Outputs: ../New_Results/shift_occupancy_matched/
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr, ttest_1samp, mannwhitneyu

matplotlib.use("Agg")

REWARD = np.array([0.5, 0.5], dtype=float)
GOAL_RADIUS = 0.12
DATA_DIR = Path("data")
OUT_DIR = Path("../New_Results/shift_occupancy_matched")
GRID = 20                # NxN spatial grid
SIGMA_BINS = 1.0         # Gaussian smoothing (in bin units) for rate maps
RATE_MAP_FLOOR = 1e-8
SKAGGS_THRESHOLD = 0.05  # same as prior analysis for comparability
N_BOOT = 5_000
N_PERM = 10_000
SEED = 42
RNG = np.random.default_rng(SEED)

C_BEFORE = "#2E5FA3"
C_AFTER = "#E07A2D"
C_MATCH = "#2A9D8F"
C_REWARD = "#E63946"
C_GOLD = "#F4D03F"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def bin_coord(v: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(v * GRID).astype(int), 0, GRID - 1)


def load_phase(period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    beh = pd.read_pickle(DATA_DIR / f"df_behavior_{period}.pkl")
    act = pd.read_pickle(DATA_DIR / f"df_activations_{period}.pkl")
    beh = beh[["timestep", "x", "y"]].drop_duplicates("timestep").reset_index(drop=True)
    return beh, act


def match_occupancy(beh_b: pd.DataFrame, beh_a: pd.DataFrame
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return arrays of kept timesteps for each phase such that per-bin
    occupancy counts are equal (= min of the two)."""
    xb = bin_coord(beh_b["x"].to_numpy(dtype=float))
    yb = bin_coord(beh_b["y"].to_numpy(dtype=float))
    xa = bin_coord(beh_a["x"].to_numpy(dtype=float))
    ya = bin_coord(beh_a["y"].to_numpy(dtype=float))
    tb = beh_b["timestep"].to_numpy()
    ta = beh_a["timestep"].to_numpy()

    occ_b = np.zeros((GRID, GRID), dtype=int)
    occ_a = np.zeros((GRID, GRID), dtype=int)
    np.add.at(occ_b, (xb, yb), 1)
    np.add.at(occ_a, (xa, ya), 1)
    min_occ = np.minimum(occ_b, occ_a)

    kept_b, kept_a = [], []
    for bi in range(GRID):
        for bj in range(GRID):
            k = int(min_occ[bi, bj])
            if k == 0:
                continue
            idx_b = np.where((xb == bi) & (yb == bj))[0]
            idx_a = np.where((xa == bi) & (ya == bj))[0]
            kept_b.append(RNG.choice(idx_b, size=k, replace=False))
            kept_a.append(RNG.choice(idx_a, size=k, replace=False))

    kept_b = np.sort(np.concatenate(kept_b)) if kept_b else np.array([], dtype=int)
    kept_a = np.sort(np.concatenate(kept_a)) if kept_a else np.array([], dtype=int)
    info = {
        "grid": GRID,
        "n_before_raw": int(len(beh_b)),
        "n_after_raw": int(len(beh_a)),
        "n_before_matched": int(kept_b.size),
        "n_after_matched": int(kept_a.size),
        "n_bins_nonempty_before": int((occ_b > 0).sum()),
        "n_bins_nonempty_after": int((occ_a > 0).sum()),
        "n_bins_common": int((min_occ > 0).sum()),
        "occ_before_sum": int(occ_b.sum()),
        "occ_after_sum": int(occ_a.sum()),
        "occ_matched_sum": int(min_occ.sum()),
    }
    return tb[kept_b], ta[kept_a], info, occ_b, occ_a, min_occ


def rate_maps(act: pd.DataFrame, beh: pd.DataFrame,
              kept_timesteps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rate_maps [N, GRID, GRID], occupancy [GRID, GRID], neuron_ids [N])."""
    keep = set(kept_timesteps.tolist())
    beh_k = beh[beh["timestep"].isin(keep)]
    act_k = act[act["timestep"].isin(keep)]

    pos_map = beh_k.set_index("timestep")[["x", "y"]]
    merged = act_k.merge(pos_map, left_on="timestep", right_index=True, how="inner")
    merged["xb"] = bin_coord(merged["x"].to_numpy(dtype=float))
    merged["yb"] = bin_coord(merged["y"].to_numpy(dtype=float))

    occ = np.zeros((GRID, GRID), dtype=float)
    np.add.at(occ, (bin_coord(beh_k["x"].to_numpy(dtype=float)),
                    bin_coord(beh_k["y"].to_numpy(dtype=float))), 1.0)

    neuron_ids = np.sort(merged["neuron_id"].unique())
    n = len(neuron_ids)
    sums = np.zeros((n, GRID, GRID), dtype=float)
    nid_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}

    nid_arr = merged["neuron_id"].to_numpy()
    xb_arr = merged["xb"].to_numpy()
    yb_arr = merged["yb"].to_numpy()
    act_arr = merged["activity"].to_numpy(dtype=float)
    idx_arr = np.array([nid_to_idx[n] for n in nid_arr])
    np.add.at(sums, (idx_arr, xb_arr, yb_arr), act_arr)

    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(occ > 0, sums / occ, 0.0)

    rates_smooth = np.stack(
        [gaussian_filter(r, sigma=SIGMA_BINS, mode="nearest") for r in rates], axis=0
    )
    return rates_smooth, occ, neuron_ids


def skaggs_from_rates(rates: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """Skaggs information (bits/spike) from smoothed rate maps & occupancy."""
    p = occ / max(occ.sum(), 1.0)
    mean_rate = np.einsum("ij,nij->n", p, rates)
    mean_rate = np.where(mean_rate > RATE_MAP_FLOOR, mean_rate, RATE_MAP_FLOOR)
    ratio = rates / mean_rate[:, None, None]
    ratio = np.where(ratio > RATE_MAP_FLOOR, ratio, RATE_MAP_FLOOR)
    log_r = np.log2(ratio)
    contrib = rates / mean_rate[:, None, None] * log_r
    contrib = np.where(np.isfinite(contrib), contrib, 0.0)
    info = np.einsum("ij,nij->n", p, contrib)
    return info


def peak_centres(rates: np.ndarray) -> np.ndarray:
    """(N,2) centre coords in [0,1]^2 from argmax of smoothed rate map."""
    N = rates.shape[0]
    coords = np.zeros((N, 2), dtype=float)
    for i in range(N):
        flat = np.argmax(rates[i])
        bi, bj = np.unravel_index(flat, rates[i].shape)
        coords[i, 0] = (bi + 0.5) / GRID
        coords[i, 1] = (bj + 0.5) / GRID
    return coords


def rayleigh(angles: np.ndarray) -> tuple[float, float]:
    n = angles.size
    if n < 3:
        return np.nan, np.nan
    C = np.mean(np.cos(angles)); S = np.mean(np.sin(angles))
    R = np.sqrt(C**2 + S**2); z = n * R**2
    p = np.exp(-z) * (
        1.0 + (2.0 * z - z**2) / (4.0 * n)
        - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n**2)
    )
    return float(R), float(np.clip(p, 0.0, 1.0))


def shift_analysis(centres_b: pd.DataFrame, centres_a: pd.DataFrame,
                   skaggs_b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    valid = skaggs_b[skaggs_b["skaggs_score"] >= SKAGGS_THRESHOLD]["neuron_id"]
    b = centres_b[centres_b["neuron_id"].isin(valid)]
    merged = b.merge(
        centres_a.rename(columns={"cx": "ax", "cy": "ay"}), on="neuron_id", how="inner"
    ).rename(columns={"cx": "bx", "cy": "by"})
    merged["dx"] = merged["ax"] - merged["bx"]
    merged["dy"] = merged["ay"] - merged["by"]
    merged["shift_mag"] = np.sqrt(merged["dx"]**2 + merged["dy"]**2)
    merged = merged[merged["shift_mag"] > 1e-8].copy()
    tnorm = np.sqrt((REWARD[0] - merged["bx"])**2 + (REWARD[1] - merged["by"])**2)
    merged = merged[tnorm > 1e-8].copy()
    tx = (REWARD[0] - merged["bx"]) / np.sqrt(
        (REWARD[0] - merged["bx"])**2 + (REWARD[1] - merged["by"])**2)
    ty = (REWARD[1] - merged["by"]) / np.sqrt(
        (REWARD[0] - merged["bx"])**2 + (REWARD[1] - merged["by"])**2)
    ux = merged["dx"] / merged["shift_mag"]; uy = merged["dy"] / merged["shift_mag"]
    merged["cos_theta"] = ux * tx + uy * ty
    merged["shift_angle"] = np.arctan2(merged["dy"], merged["dx"])
    toward = np.arctan2(ty, tx)
    rel = merged["shift_angle"] - toward
    merged["rel_angle"] = (rel + np.pi) % (2 * np.pi) - np.pi
    merged["dist_before"] = np.sqrt((merged["bx"] - REWARD[0])**2 + (merged["by"] - REWARD[1])**2)
    merged["dist_after"] = np.sqrt((merged["ax"] - REWARD[0])**2 + (merged["ay"] - REWARD[1])**2)

    cos = merged["cos_theta"].to_numpy(dtype=float)
    rel_a = merged["rel_angle"].to_numpy(dtype=float)
    t_stat, p_t = ttest_1samp(cos, popmean=0.0, alternative="greater")
    R_rel, p_rel = rayleigh(rel_a)
    R_all, p_all = rayleigh(merged["shift_angle"].to_numpy(dtype=float))
    boot = np.array([np.mean(RNG.choice(cos, size=cos.size, replace=True))
                     for _ in range(N_BOOT)])
    perm = np.array([np.mean(RNG.choice([-1.0, 1.0], size=cos.size) * cos)
                     for _ in range(N_PERM)])

    stats = {
        "n_paired_matched": int(len(merged)),
        "mean_cosine": float(np.mean(cos)),
        "bootstrap_ci_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "mean_rel_angle_deg": float(np.degrees(np.arctan2(
            np.mean(np.sin(rel_a)), np.mean(np.cos(rel_a))))),
        "fraction_toward_reward": float(np.mean(cos > 0)),
        "rayleigh_all_directions": {"R": R_all, "p": p_all},
        "rayleigh_toward_reward": {"R": R_rel, "p": p_rel},
        "ttest_cos_vs_0": {"t": float(t_stat), "p_one_tailed": float(p_t)},
        "permutation_test": {"p": float(np.mean(perm >= float(np.mean(cos))))},
        "mean_dx": float(merged["dx"].mean()),
        "mean_dy": float(merged["dy"].mean()),
    }
    return merged, stats


def skaggs_distance_stats(sk_b: pd.DataFrame, ctr_b: pd.DataFrame,
                          sk_a: pd.DataFrame, ctr_a: pd.DataFrame) -> dict:
    def _joined(ctr, sk):
        j = ctr.merge(sk, on="neuron_id", how="inner")
        j["dist_reward"] = np.sqrt((j["cx"] - REWARD[0])**2 + (j["cy"] - REWARD[1])**2)
        return j

    jb = _joined(ctr_b, sk_b); ja = _joined(ctr_a, sk_a)
    rho_b, p_b = spearmanr(jb["dist_reward"], jb["skaggs_score"])
    rho_a, p_a = spearmanr(ja["dist_reward"], ja["skaggs_score"])

    paired = jb.merge(
        ja[["neuron_id", "skaggs_score", "dist_reward"]].rename(
            columns={"skaggs_score": "skaggs_after", "dist_reward": "dist_after"}),
        on="neuron_id", how="inner"
    ).rename(columns={"skaggs_score": "skaggs_before", "dist_reward": "dist_before"})
    paired["delta"] = paired["skaggs_after"] - paired["skaggs_before"]
    q25 = float(paired["dist_before"].quantile(0.25))
    q75 = float(paired["dist_before"].quantile(0.75))
    near = paired[paired["dist_before"] <= q25]["delta"].to_numpy(dtype=float)
    far = paired[paired["dist_before"] >= q75]["delta"].to_numpy(dtype=float)
    if near.size > 2 and far.size > 2:
        _, p_int = mannwhitneyu(near, far, alternative="two-sided")
    else:
        p_int = np.nan

    return {
        "spearman_before_matched": {"rho": float(rho_b), "p": float(p_b), "n": int(len(jb))},
        "spearman_after_matched": {"rho": float(rho_a), "p": float(p_a), "n": int(len(ja))},
        "paired_n": int(len(paired)),
        "interaction_near_vs_far": {
            "q25_dist": q25, "q75_dist": q75,
            "mean_delta_near": float(np.mean(near)) if near.size else float("nan"),
            "mean_delta_far": float(np.mean(far)) if far.size else float("nan"),
            "mannwhitney_p": float(p_int) if np.isfinite(p_int) else None,
            "n_near": int(near.size), "n_far": int(far.size),
        },
    }


def plot_occupancy_heatmaps(occ_b: np.ndarray, occ_a: np.ndarray,
                            occ_m: np.ndarray, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, occ, title in zip(
        axes, [occ_b, occ_a, occ_m],
        ["Phase 1 (raw)", "Phase 2 (raw)", "Matched (min)"],
    ):
        im = ax.imshow(occ.T, origin="lower", cmap="viridis",
                       extent=[0, 1, 0, 1], aspect="equal")
        circle = plt.Circle(REWARD, GOAL_RADIUS, fill=False, edgecolor=C_GOLD,
                            lw=2, ls="--")
        ax.add_patch(circle)
        ax.scatter(*REWARD, s=160, marker="*", color=C_GOLD)
        ax.set_title(f"{title}  (sum={int(occ.sum())})", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    plt.suptitle("Spatial occupancy before / after / matched", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "occupancy_heatmaps.png", dpi=150)
    plt.close()
    print("  Saved: occupancy_heatmaps.png")


def plot_quiver_matched(paired: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap = matplotlib.colormaps.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    for _, row in paired.iterrows():
        c = cmap(norm(row["cos_theta"]))
        ax.annotate("", xy=(row["ax"], row["ay"]), xytext=(row["bx"], row["by"]),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, mutation_scale=6))
    ax.scatter(paired["bx"], paired["by"], s=8, c="gray", alpha=0.45, zorder=3)
    circle = plt.Circle(REWARD, GOAL_RADIUS, fill=False, edgecolor=C_GOLD, lw=2, ls="--")
    ax.add_patch(circle)
    ax.scatter(*REWARD, s=220, marker="*", color=C_GOLD, zorder=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("cos(theta)", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Place-field shifts under matched occupancy")
    plt.tight_layout()
    plt.savefig(fig_dir / "matched_shift_quiver.png", dpi=150)
    plt.close()
    print("  Saved: matched_shift_quiver.png")


def plot_cosine_hist(paired: pd.DataFrame, stats: dict, fig_dir: Path) -> None:
    cos = paired["cos_theta"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(cos, bins=25, density=True, color=C_MATCH, alpha=0.7,
            label=f"Matched cos  (mu={stats['mean_cosine']:+.3f})")
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.axvline(stats["mean_cosine"], color=C_REWARD, lw=2.5,
               label=f"Mean = {stats['mean_cosine']:+.3f}")
    ci = stats["bootstrap_ci_95"]
    ax.axvspan(ci[0], ci[1], alpha=0.15, color=C_REWARD, label="95% CI")
    p_t = stats["ttest_cos_vs_0"]["p_one_tailed"]
    p_perm = stats["permutation_test"]["p"]
    ax.set_xlabel("cos(theta)"); ax.set_ylabel("Density")
    ax.set_title(f"Matched-occupancy cosine distribution\nt-test p={p_t:.4f} | perm p={p_perm:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "matched_cosine_hist.png", dpi=150)
    plt.close()
    print("  Saved: matched_cosine_hist.png")


def plot_skaggs_scatter(ctr_b: pd.DataFrame, sk_b: pd.DataFrame,
                        ctr_a: pd.DataFrame, sk_a: pd.DataFrame,
                        stats: dict, fig_dir: Path) -> None:
    def _prep(ctr, sk):
        j = ctr.merge(sk, on="neuron_id", how="inner")
        j["dist_reward"] = np.sqrt((j["cx"] - REWARD[0])**2 + (j["cy"] - REWARD[1])**2)
        return j
    jb = _prep(ctr_b, sk_b); ja = _prep(ctr_a, sk_a)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, j, col, phase in [(axes[0], jb, C_BEFORE, "before"),
                              (axes[1], ja, C_AFTER, "after")]:
        ax.scatter(j["dist_reward"], j["skaggs_score"], s=14, alpha=0.3, color=col)
        rho = stats[f"spearman_{phase}_matched"]["rho"]
        p = stats[f"spearman_{phase}_matched"]["p"]
        ax.axvline(GOAL_RADIUS, color=C_GOLD, ls="--", lw=1.2, alpha=0.7)
        ax.set_xlabel("Distance to reward")
        ax.set_ylabel("Skaggs score (bits/spike)")
        ax.set_title(f"Matched {phase}: rho={rho:.3f}, p={p:.2e}")
    plt.suptitle("Skaggs vs distance (occupancy-matched)")
    plt.tight_layout()
    plt.savefig(fig_dir / "matched_skaggs_scatter.png", dpi=150)
    plt.close()
    print("  Saved: matched_skaggs_scatter.png")


def main() -> None:
    out_dir = ensure_dir(OUT_DIR)
    fig_dir = ensure_dir(out_dir / "figures")

    print("Loading raw data...")
    beh_b, act_b = load_phase("before")
    beh_a, act_a = load_phase("after")
    print(f"  Phase 1: {len(beh_b)} timesteps, "
          f"{act_b['neuron_id'].nunique()} neurons")
    print(f"  Phase 2: {len(beh_a)} timesteps, "
          f"{act_a['neuron_id'].nunique()} neurons")

    print(f"Matching occupancy on {GRID}x{GRID} grid...")
    kept_b, kept_a, info, occ_b, occ_a, occ_m = match_occupancy(beh_b, beh_a)
    print(f"  Matched: {info['n_before_matched']}/{info['n_before_raw']} kept in Phase 1, "
          f"{info['n_after_matched']}/{info['n_after_raw']} kept in Phase 2")
    print(f"  Common bins: {info['n_bins_common']}/{GRID*GRID}")

    print("Computing smoothed rate maps & place-field centres...")
    rates_b, occ_map_b, nids_b = rate_maps(act_b, beh_b, kept_b)
    rates_a, occ_map_a, nids_a = rate_maps(act_a, beh_a, kept_a)
    print(f"  Rate maps: before={rates_b.shape}, after={rates_a.shape}")

    print("Computing matched Skaggs scores & centres...")
    sk_b_vals = skaggs_from_rates(rates_b, occ_map_b)
    sk_a_vals = skaggs_from_rates(rates_a, occ_map_a)
    ctrs_b = peak_centres(rates_b)
    ctrs_a = peak_centres(rates_a)
    sk_b = pd.DataFrame({"neuron_id": nids_b, "skaggs_score": sk_b_vals})
    sk_a = pd.DataFrame({"neuron_id": nids_a, "skaggs_score": sk_a_vals})
    ctr_b = pd.DataFrame({"neuron_id": nids_b, "cx": ctrs_b[:, 0], "cy": ctrs_b[:, 1]})
    ctr_a = pd.DataFrame({"neuron_id": nids_a, "cx": ctrs_a[:, 0], "cy": ctrs_a[:, 1]})

    print("Running directional shift analysis on matched centres...")
    paired, shift_stats = shift_analysis(ctr_b, ctr_a, sk_b)

    print("Running Skaggs-vs-distance analysis on matched centres...")
    sk_dist_stats = skaggs_distance_stats(sk_b, ctr_b, sk_a, ctr_a)

    print("\n-- Matched-Occupancy Results --------------------------------")
    print(f"  N paired  : {shift_stats['n_paired_matched']}")
    print(f"  Mean cos  : {shift_stats['mean_cosine']:+.4f}  "
          f"[{shift_stats['bootstrap_ci_95'][0]:+.3f}, {shift_stats['bootstrap_ci_95'][1]:+.3f}]")
    print(f"  Mean drift: dx={shift_stats['mean_dx']:+.4f}, dy={shift_stats['mean_dy']:+.4f}")
    print(f"  Frac toward reward : {shift_stats['fraction_toward_reward']*100:.1f}%")
    print(f"  t-test p (cos>0)   : {shift_stats['ttest_cos_vs_0']['p_one_tailed']:.5f}")
    print(f"  Rayleigh rel       : R={shift_stats['rayleigh_toward_reward']['R']:.4f}, "
          f"p={shift_stats['rayleigh_toward_reward']['p']:.5f}")
    sb = sk_dist_stats["spearman_before_matched"]
    sa = sk_dist_stats["spearman_after_matched"]
    print(f"  Spearman matched (before): rho={sb['rho']:+.4f}, p={sb['p']:.4f}")
    print(f"  Spearman matched (after):  rho={sa['rho']:+.4f}, p={sa['p']:.4f}")
    iv = sk_dist_stats["interaction_near_vs_far"]
    print(f"  Near dSkaggs: {iv['mean_delta_near']:+.4f}  (n={iv['n_near']})")
    print(f"  Far  dSkaggs: {iv['mean_delta_far']:+.4f}  (n={iv['n_far']})")
    print(f"  Interaction p: {iv['mannwhitney_p']}")
    print("-------------------------------------------------------------")

    print("\nGenerating figures...")
    plot_occupancy_heatmaps(occ_b, occ_a, occ_m, fig_dir)
    plot_quiver_matched(paired, fig_dir)
    plot_cosine_hist(paired, shift_stats, fig_dir)
    plot_skaggs_scatter(ctr_b, sk_b, ctr_a, sk_a, sk_dist_stats, fig_dir)

    paired[["neuron_id", "bx", "by", "ax", "ay", "dx", "dy",
            "cos_theta", "rel_angle", "shift_mag",
            "dist_before", "dist_after"]].to_csv(
        out_dir / "matched_shift_vectors.csv", index=False
    )
    sk_b.to_csv(out_dir / "matched_skaggs_before.csv", index=False)
    sk_a.to_csv(out_dir / "matched_skaggs_after.csv", index=False)
    ctr_b.to_csv(out_dir / "matched_centres_before.csv", index=False)
    ctr_a.to_csv(out_dir / "matched_centres_after.csv", index=False)

    save_json(out_dir / "shift_occupancy_matched_results.json", {
        "matching": info,
        "shift_statistics_matched": shift_stats,
        "skaggs_distance_matched": sk_dist_stats,
    })
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
