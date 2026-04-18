from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle
from scipy.stats import chi2_contingency, spearmanr, wilcoxon

from common import DEFAULT_REWARD, ensure_dir, load_behavior_activity, pivot_activity_matrix, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distance-to-reward ramping analysis.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/ramping_analysis")

    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=float(DEFAULT_REWARD[0]))
    parser.add_argument("--reward-y", type=float, default=float(DEFAULT_REWARD[1]))

    parser.add_argument("--n-distance-bins", type=int, default=20)
    parser.add_argument("--distance-max", type=float, default=0.8)
    parser.add_argument("--min-samples-per-bin", type=int, default=3)
    parser.add_argument("--min-valid-bins", type=int, default=12)

    parser.add_argument("--rho-threshold", type=float, default=0.30)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--max-monotonicity-violations", type=int, default=3)
    parser.add_argument("--require-approach", action="store_true", default=True)
    parser.add_argument("--no-require-approach", dest="require_approach", action="store_false")
    parser.add_argument("--n-shuffle", type=int, default=200)
    parser.add_argument("--shuffle-neurons", type=int, default=120)
    parser.add_argument("--n-example-cells", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_goal_position(data_dir: Path, goal_file: str, reward_x: float, reward_y: float) -> tuple[np.ndarray, str]:
    p = data_dir / goal_file
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float).reshape(-1)
        if arr.size >= 2:
            return np.array([float(arr[0]), float(arr[1])], dtype=float), str(p)
    return np.array([reward_x, reward_y], dtype=float), "fallback_args"


def load_period_matrix(data_dir: Path, period: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beh, act, _ = load_behavior_activity(data_dir, period)
    x_df, b_df = pivot_activity_matrix(act, beh)
    if b_df is None:
        raise RuntimeError(f"Could not align behavior and activity for period={period}.")

    neuron_ids = x_df.columns.to_numpy(dtype=int)
    activity = x_df.to_numpy(dtype=float)
    positions = b_df[["x", "y"]].to_numpy(dtype=float)
    return neuron_ids, activity, positions


def approach_mask(dist: np.ndarray, distance_max: float, require_approach: bool) -> np.ndarray:
    mask = dist <= distance_max
    if not require_approach:
        return mask
    moving_toward = np.ones_like(mask, dtype=bool)
    moving_toward[1:] = dist[1:] <= dist[:-1]
    return mask & moving_toward


def distance_labels(dist: np.ndarray, n_bins: int, distance_max: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, distance_max, n_bins + 1)
    labels = np.digitize(dist, edges, right=False) - 1
    labels = np.clip(labels, 0, n_bins - 1)
    return labels, edges


def interpolate_row(row: np.ndarray) -> np.ndarray:
    idx = np.arange(row.shape[0])
    finite = np.isfinite(row)
    if np.sum(finite) < 2:
        return row
    return np.interp(idx, idx[finite], row[finite])


def classify_profile(
    z_profile: np.ndarray,
    rho_threshold: float,
    p_threshold: float,
    max_monotonicity_violations: int,
) -> dict[str, Any]:
    bins = np.arange(z_profile.shape[0], dtype=float)
    rho, p = spearmanr(bins, z_profile)
    rho = float(rho)
    p = float(p)

    d = np.diff(z_profile)
    neg_ramp_viol = int(np.sum(d > 0))
    pos_ramp_viol = int(np.sum(d < 0))

    is_negative = bool((rho < -rho_threshold) and (p < p_threshold) and (neg_ramp_viol <= max_monotonicity_violations))
    is_positive = bool((rho > rho_threshold) and (p < p_threshold) and (pos_ramp_viol <= max_monotonicity_violations))

    return {
        "rho": rho,
        "p": p,
        "neg_viol": neg_ramp_viol,
        "pos_viol": pos_ramp_viol,
        "is_negative_ramp": is_negative,
        "is_positive_ramp": is_positive,
    }


def analyze_period(
    neuron_ids: np.ndarray,
    activity: np.ndarray,
    positions: np.ndarray,
    goal: np.ndarray,
    n_bins: int,
    distance_max: float,
    min_samples_per_bin: int,
    min_valid_bins: int,
    rho_threshold: float,
    p_threshold: float,
    max_monotonicity_violations: int,
    require_approach: bool,
) -> dict[str, Any]:
    """Compute per-neuron ramping stats for one phase."""
    dist = np.linalg.norm(positions - goal[None, :], axis=1)
    labels, edges = distance_labels(dist, n_bins=n_bins, distance_max=distance_max)
    mask = approach_mask(dist, distance_max=distance_max, require_approach=require_approach)

    n_neurons = neuron_ids.shape[0]
    raw_profiles = np.full((n_neurons, n_bins), np.nan, dtype=float)

    for b in range(n_bins):
        idx = np.where(mask & (labels == b))[0]
        if idx.size < min_samples_per_bin:
            continue
        raw_profiles[:, b] = np.mean(activity[idx], axis=0)

    valid_bins_count = np.sum(np.isfinite(raw_profiles), axis=1)
    valid_neurons = valid_bins_count >= min_valid_bins
    z_profiles = np.full_like(raw_profiles, np.nan)
    rho_arr = np.full(n_neurons, np.nan, dtype=float)
    p_arr = np.full(n_neurons, np.nan, dtype=float)
    neg_viol_arr = np.full(n_neurons, np.nan, dtype=float)
    pos_viol_arr = np.full(n_neurons, np.nan, dtype=float)
    neg_ramp = np.zeros(n_neurons, dtype=bool)
    pos_ramp = np.zeros(n_neurons, dtype=bool)
    for i in range(n_neurons):
        if not valid_neurons[i]:
            continue
        row = interpolate_row(raw_profiles[i].copy())
        if np.any(~np.isfinite(row)):
            continue

        mu = float(np.mean(row))
        sd = float(np.std(row, ddof=1))
        if sd < 1e-12:
            continue

        z = (row - mu) / sd
        cls = classify_profile(
            z_profile=z,
            rho_threshold=rho_threshold,
            p_threshold=p_threshold,
            max_monotonicity_violations=max_monotonicity_violations,
        )

        z_profiles[i] = z
        rho_arr[i] = cls["rho"]
        p_arr[i] = cls["p"]
        neg_viol_arr[i] = cls["neg_viol"]
        pos_viol_arr[i] = cls["pos_viol"]
        neg_ramp[i] = cls["is_negative_ramp"]
        pos_ramp[i] = cls["is_positive_ramp"]

    used = np.isfinite(rho_arr)
    frac_negative = float(np.mean(neg_ramp[used])) if np.any(used) else np.nan
    frac_positive = float(np.mean(pos_ramp[used])) if np.any(used) else np.nan

    return {
        "neuron_ids": neuron_ids,
        "distance": dist,
        "labels": labels,
        "edges": edges,
        "mask": mask,
        "raw_profiles": raw_profiles,
        "z_profiles": z_profiles,
        "valid_neurons": used,
        "rho": rho_arr,
        "p": p_arr,
        "neg_viol": neg_viol_arr,
        "pos_viol": pos_viol_arr,
        "is_negative_ramp": neg_ramp,
        "is_positive_ramp": pos_ramp,
        "fraction_negative_ramp": frac_negative,
        "fraction_positive_ramp": frac_positive,
        "mean_rho": float(np.nanmean(rho_arr)) if np.any(used) else np.nan,
        "n_analyzed": int(np.sum(used)),
        "n_total": int(neuron_ids.shape[0]),
    }


def expected_ramp_fraction_shuffle(
    activity: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    n_bins: int,
    min_samples_per_bin: int,
    min_valid_bins: int,
    rho_threshold: float,
    p_threshold: float,
    max_monotonicity_violations: int,
    n_shuffle: int,
    shuffle_neurons: int,
    seed: int,
) -> float:
    """
    Estimate expected negative-ramp fraction under shuffled distance labels.
    """
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    if idx.size < (min_samples_per_bin * 2):
        return np.nan
    labels_sub = labels[idx]
    act_sub = activity[idx]
    n_total_neurons = act_sub.shape[1]
    n_use = min(shuffle_neurons, n_total_neurons)
    neuron_sel = rng.choice(n_total_neurons, size=n_use, replace=False)
    fracs = np.zeros(n_shuffle, dtype=float)

    for s in range(n_shuffle):
        n_ramp = 0
        n_valid = 0

        for j in neuron_sel:
            perm_labels = rng.permutation(labels_sub)
            prof = np.full(n_bins, np.nan, dtype=float)
            for b in range(n_bins):
                bidx = np.where(perm_labels == b)[0]
                if bidx.size < min_samples_per_bin:
                    continue
                prof[b] = float(np.mean(act_sub[bidx, j]))

            if np.sum(np.isfinite(prof)) < min_valid_bins:
                continue

            prof = interpolate_row(prof)
            if np.any(~np.isfinite(prof)):
                continue

            sd = float(np.std(prof, ddof=1))
            if sd < 1e-12:
                continue

            z = (prof - float(np.mean(prof))) / sd
            cls = classify_profile(
                z_profile=z,
                rho_threshold=rho_threshold,
                p_threshold=p_threshold,
                max_monotonicity_violations=max_monotonicity_violations,
            )

            n_valid += 1
            if cls["is_negative_ramp"]:
                n_ramp += 1

        fracs[s] = (n_ramp / n_valid) if n_valid > 0 else np.nan

    return float(np.nanmean(fracs))


def load_centers_phase2(data_dir: Path) -> tuple[pd.DataFrame, str]:
    npy = data_dir / "place_field_centres_phase2.npy"
    if npy.exists():
        arr = np.asarray(np.load(npy))
        if arr.ndim == 2 and arr.shape[1] >= 3:
            df = pd.DataFrame({"neuron_id": arr[:, 0].astype(int), "x": arr[:, 1].astype(float), "y": arr[:, 2].astype(float)})
            return df, str(npy)

    pkl = data_dir / "df_centers_after.pkl"
    if pkl.exists():
        df = pd.read_pickle(pkl)
        rename_map = {"center_x": "x", "center_y": "y", "neuron": "neuron_id"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "neuron_id" in df.columns and "x" in df.columns and "y" in df.columns:
            return df[["neuron_id", "x", "y"]].copy(), str(pkl)

    return pd.DataFrame(columns=["neuron_id", "x", "y"]), "missing"


def plot_examples(neuron_ids: np.ndarray, z_profiles: np.ndarray, rho: np.ndarray, ramp_mask: np.ndarray, n_examples: int, out_path: Path) -> list[int]:
    idx = np.where(ramp_mask & np.isfinite(rho))[0]
    if idx.size == 0:
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No ramp cells found", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return []

    idx_sorted = idx[np.argsort(rho[idx])]  # strongest negative first
    pick = idx_sorted[: min(n_examples, idx_sorted.size)]

    n_cols = 5
    n_rows = int(np.ceil(pick.size / n_cols))
    plt.figure(figsize=(3.0 * n_cols, 2.4 * n_rows))

    x = np.arange(z_profiles.shape[1])
    selected_ids: list[int] = []
    for k, i in enumerate(pick):
        plt.subplot(n_rows, n_cols, k + 1)
        plt.plot(x, z_profiles[i], color="#D95319", linewidth=2)
        plt.gca().invert_xaxis()  # Near reward on right for visual consistency
        plt.title(f"n={int(neuron_ids[i])}, rho={rho[i]:.2f}", fontsize=9)
        plt.xlabel("Distance bin")
        plt.ylabel("z")
        plt.ylim(-2.5, 2.5)
        selected_ids.append(int(neuron_ids[i]))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return selected_ids


def plot_profiles(phase1: dict[str, Any], phase2: dict[str, Any], out_path: Path) -> None:
    x = np.arange(phase1["z_profiles"].shape[1])

    def mean_sem(z: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = z[mask]
        if arr.shape[0] == 0:
            return np.full(z.shape[1], np.nan), np.full(z.shape[1], np.nan)
        m = np.nanmean(arr, axis=0)
        s = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(max(arr.shape[0], 1))
        return m, s

    m_ramp, s_ramp = mean_sem(phase2["z_profiles"], phase2["is_negative_ramp"] & phase2["valid_neurons"])
    m_non, s_non = mean_sem(phase2["z_profiles"], (~phase2["is_negative_ramp"]) & phase2["valid_neurons"])

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x, m_ramp, color="#D95319", linewidth=2.2, label="Ramping cells")
    plt.fill_between(x, m_ramp - s_ramp, m_ramp + s_ramp, color="#D95319", alpha=0.25)

    plt.plot(x, m_non, color="#1F77B4", linewidth=2.0, label="Non-ramping cells")
    plt.fill_between(x, m_non - s_non, m_non + s_non, color="#1F77B4", alpha=0.22)

    plt.gca().invert_xaxis()
    plt.xlabel("Distance bin")
    plt.ylabel("Mean z-scored activity")
    plt.title("Ramping vs non-ramping population profiles (Phase 2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_fraction_bar(frac1: float, frac2: float, frac_shuf: float, out_path: Path) -> None:
    labels = ["Phase 1", "Phase 2", "Shuffle expected"]
    vals = [frac1, frac2, frac_shuf]

    plt.figure(figsize=(6.8, 4.6))
    plt.bar(labels, vals, color=["#4C78A8", "#F58518", "#9E9E9E"], alpha=0.9)
    plt.ylabel("Fraction negative-ramping")
    plt.title("Ramping-cell fraction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def preferred_distance(z_profiles: np.ndarray, mask: np.ndarray, edges: np.ndarray) -> dict[int, float]:
    """Preferred (near-reward biased) distance estimate from top-activity bins."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    out: dict[int, float] = {}
    idx = np.where(mask)[0]
    for i in idx:
        row = z_profiles[i]
        if np.any(~np.isfinite(row)):
            continue
        thr = float(np.quantile(row, 0.9))
        top = np.where(row >= thr)[0]
        if top.size == 0:
            continue
        out[int(i)] = float(np.min(centers[top]))
    return out


def plot_spatial_map(
    centers_df: pd.DataFrame,
    center_source: str,
    neuron_ids: np.ndarray,
    ramp_mask: np.ndarray,
    z_profiles: np.ndarray,
    edges: np.ndarray,
    goal: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(6.5, 6.0))
    ax = plt.gca()

    if not centers_df.empty:
        ax.scatter(centers_df["x"], centers_df["y"], s=10, color="gray", alpha=0.45, label="All PF centers")

    pref = preferred_distance(z_profiles, ramp_mask, edges)

    ramp_ids = set(neuron_ids[np.where(ramp_mask)[0]].astype(int).tolist())
    ramp_df = centers_df[centers_df["neuron_id"].isin(ramp_ids)].copy() if not centers_df.empty else pd.DataFrame()

    if not ramp_df.empty:
        ax.scatter(ramp_df["x"], ramp_df["y"], s=22, color="#D95319", alpha=0.9, label="Ramping PF centers")

        nid_to_row = {int(n): i for i, n in enumerate(neuron_ids.astype(int).tolist())}
        for _, r in ramp_df.iterrows():
            nid = int(r["neuron_id"])
            if nid not in nid_to_row:
                continue
            i = nid_to_row[nid]
            d_pref = pref.get(i, None)
            if d_pref is None:
                continue

            angle = (nid * 137) % 360
            arc = Arc(
                (goal[0], goal[1]),
                width=2 * d_pref,
                height=2 * d_pref,
                theta1=angle - 22,
                theta2=angle + 22,
                linewidth=1.5,
                color="#D95319",
                alpha=0.5,
            )
            ax.add_patch(arc)

    ax.add_patch(Circle((goal[0], goal[1]), radius=0.02, color="black", alpha=0.85, label="Goal"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Ramping spatial map (centers source: {center_source})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def paired_rho_test(phase1: dict[str, Any], phase2: dict[str, Any]) -> tuple[float, float, float, int]:
    ids1 = phase1["neuron_ids"].astype(int)
    ids2 = phase2["neuron_ids"].astype(int)

    map1 = {int(n): i for i, n in enumerate(ids1.tolist())}
    map2 = {int(n): i for i, n in enumerate(ids2.tolist())}
    common = sorted(set(map1.keys()).intersection(set(map2.keys())))

    if len(common) < 5:
        return np.nan, np.nan, np.nan, len(common)

    r1 = []
    r2 = []
    for n in common:
        i = map1[n]
        j = map2[n]
        a = phase1["rho"][i]
        b = phase2["rho"][j]
        if np.isfinite(a) and np.isfinite(b):
            r1.append(float(a))
            r2.append(float(b))

    if len(r1) < 5:
        return np.nan, np.nan, np.nan, len(r1)

    stat, p = wilcoxon(np.asarray(r1), np.asarray(r2), alternative="greater")
    return float(np.mean(r1)), float(np.mean(r2)), float(p), len(r1)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir if args.output_dir else (data_dir / "ramping_analysis"))
    fig_dir = ensure_dir(out_dir / "figures")

    goal, goal_source = load_goal_position(data_dir, args.goal_file, args.reward_x, args.reward_y)

    n1, a1, p1 = load_period_matrix(data_dir, "before")
    n2, a2, p2 = load_period_matrix(data_dir, "after")

    phase1 = analyze_period(
        neuron_ids=n1,
        activity=a1,
        positions=p1,
        goal=goal,
        n_bins=args.n_distance_bins,
        distance_max=args.distance_max,
        min_samples_per_bin=args.min_samples_per_bin,
        min_valid_bins=args.min_valid_bins,
        rho_threshold=args.rho_threshold,
        p_threshold=args.p_threshold,
        max_monotonicity_violations=args.max_monotonicity_violations,
        require_approach=args.require_approach,
    )

    phase2 = analyze_period(
        neuron_ids=n2,
        activity=a2,
        positions=p2,
        goal=goal,
        n_bins=args.n_distance_bins,
        distance_max=args.distance_max,
        min_samples_per_bin=args.min_samples_per_bin,
        min_valid_bins=args.min_valid_bins,
        rho_threshold=args.rho_threshold,
        p_threshold=args.p_threshold,
        max_monotonicity_violations=args.max_monotonicity_violations,
        require_approach=args.require_approach,
    )

    frac1 = phase1["fraction_negative_ramp"]
    frac2 = phase2["fraction_negative_ramp"]

    # Proportion test across analyzed neurons.
    n1_used = max(int(phase1["n_analyzed"]), 1)
    n2_used = max(int(phase2["n_analyzed"]), 1)
    k1 = int(np.sum(phase1["is_negative_ramp"] & phase1["valid_neurons"]))
    k2 = int(np.sum(phase2["is_negative_ramp"] & phase2["valid_neurons"]))

    table = np.array([[k1, n1_used - k1], [k2, n2_used - k2]], dtype=float)
    chi2, chi2_p, _, _ = chi2_contingency(table)

    mean_rho1, mean_rho2, wilcoxon_p, n_paired = paired_rho_test(phase1, phase2)

    frac_shuf = expected_ramp_fraction_shuffle(
        activity=a2,
        labels=phase2["labels"],
        mask=phase2["mask"],
        n_bins=args.n_distance_bins,
        min_samples_per_bin=args.min_samples_per_bin,
        min_valid_bins=args.min_valid_bins,
        rho_threshold=args.rho_threshold,
        p_threshold=args.p_threshold,
        max_monotonicity_violations=args.max_monotonicity_violations,
        n_shuffle=args.n_shuffle,
        shuffle_neurons=args.shuffle_neurons,
        seed=args.seed,
    )
    selected_ids = plot_examples(
        neuron_ids=phase2["neuron_ids"],
        z_profiles=phase2["z_profiles"],
        rho=phase2["rho"],
        ramp_mask=phase2["is_negative_ramp"] & phase2["valid_neurons"],
        n_examples=args.n_example_cells,
        out_path=fig_dir / "ramping_examples.png",
    )
    plot_profiles(phase1, phase2, fig_dir / "ramping_profile.png")
    plot_fraction_bar(frac1, frac2, frac_shuf, fig_dir / "ramping_fraction_bar.png")
    centers_df, center_source = load_centers_phase2(data_dir)
    plot_spatial_map(
        centers_df=centers_df,
        center_source=center_source,
        neuron_ids=phase2["neuron_ids"],
        ramp_mask=phase2["is_negative_ramp"] & phase2["valid_neurons"],
        z_profiles=phase2["z_profiles"],
        edges=phase2["edges"],
        goal=goal,
        out_path=fig_dir / "ramping_spatial_map.png",
    )

    payload: dict[str, Any] = {
        "script": "ramping_analysis.py",
        "comparison_paper": "Pothuizen et al. 2005",
        "comparison_metric": "Fraction of ramping neurons",
        "comparison_value_paper": 0.20,
        "comparison_value_paper_note": "Experimental reports vary; ~20% used as anchor.",
        "your_value": float(frac2) if np.isfinite(frac2) else np.nan,
        "your_value_ci_95": [np.nan, np.nan],
        "qualitative_match": bool(np.isfinite(frac2) and (frac2 > frac1)),
        "goal_position": goal.tolist(),
        "goal_position_source": goal_source,
        "params": {
            "n_distance_bins": int(args.n_distance_bins),
            "distance_max": float(args.distance_max),
            "min_samples_per_bin": int(args.min_samples_per_bin),
            "min_valid_bins": int(args.min_valid_bins),
            "rho_threshold": float(args.rho_threshold),
            "p_threshold": float(args.p_threshold),
            "max_monotonicity_violations": int(args.max_monotonicity_violations),
            "require_approach": bool(args.require_approach),
            "n_shuffle": int(args.n_shuffle),
            "shuffle_neurons": int(args.shuffle_neurons),
        },
        "phase1": {
            "fraction_negative_ramp": float(frac1) if np.isfinite(frac1) else np.nan,
            "fraction_positive_ramp": float(phase1["fraction_positive_ramp"]) if np.isfinite(phase1["fraction_positive_ramp"]) else np.nan,
            "n_analyzed": int(phase1["n_analyzed"]),
            "n_total": int(phase1["n_total"]),
            "mean_rho": float(phase1["mean_rho"]) if np.isfinite(phase1["mean_rho"]) else np.nan,
        },
        "phase2": {
            "fraction_negative_ramp": float(frac2) if np.isfinite(frac2) else np.nan,
            "fraction_positive_ramp": float(phase2["fraction_positive_ramp"]) if np.isfinite(phase2["fraction_positive_ramp"]) else np.nan,
            "n_analyzed": int(phase2["n_analyzed"]),
            "n_total": int(phase2["n_total"]),
            "n_ramp_cells": int(np.sum(phase2["is_negative_ramp"] & phase2["valid_neurons"])),
            "mean_rho": float(phase2["mean_rho"]) if np.isfinite(phase2["mean_rho"]) else np.nan,
            "example_cell_ids": selected_ids,
        },
        "before_vs_after": {
            "chi2_stat": float(chi2),
            "chi2_p": float(chi2_p),
            "wilcoxon_paired_rho_p": float(wilcoxon_p) if np.isfinite(wilcoxon_p) else np.nan,
            "mean_rho_phase1_common": float(mean_rho1) if np.isfinite(mean_rho1) else np.nan,
            "mean_rho_phase2_common": float(mean_rho2) if np.isfinite(mean_rho2) else np.nan,
            "n_paired_neurons": int(n_paired),
        },
        "shuffle_expected_fraction": float(frac_shuf) if np.isfinite(frac_shuf) else np.nan,
    }

    save_json(out_dir / "ramping_results.json", payload)
    print(f"Saved ramping outputs to: {out_dir}")


if __name__ == "__main__":
    main()
