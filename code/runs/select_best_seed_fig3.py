#!/usr/bin/env python3
"""
Score 10 head-condition (Fig 3) multi-seed runs by COM uniformity, pick the
best example, and regenerate Figure 3 with aggregate supplementary panels.

Usage (run from repo root after run_fig34_parallel.py completes):
    python code/runs/select_best_seed_fig3.py
    python code/runs/select_best_seed_fig3.py --seeds 0 1 2 3 4 5 6 7 8 9
    python code/runs/select_best_seed_fig3.py --list_only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon as _wilcoxon

REPO = Path(__file__).resolve().parents[2]
FIG3_DIR = REPO / "code" / "figures" / "fig3"
FIG2_DIR = REPO / "code" / "figures" / "fig2"  # for shared analysis functions
FIG1_DIR = REPO / "code" / "figures" / "fig1"
MULTI_SEED_DIR = REPO / "results" / "multi_seed_head"

# Fig2's analysis.py has the right load_phases/paired_distance_stats for this data structure
sys.path.insert(0, str(FIG1_DIR))
sys.path.insert(0, str(FIG2_DIR))

from analysis import load_phases, shared_place_cell_ids, paired_distance_stats  # noqa: E402
from utils import DEFAULT_REWARD, apply_nature_style  # noqa: E402

apply_nature_style()

REWARD_POS = np.asarray(DEFAULT_REWARD, dtype=float)


# ---------------------------------------------------------------------------
# Uniformity scoring (same metric as Fig 2)
# ---------------------------------------------------------------------------

def com_spatial_entropy(coms: np.ndarray, bins: int = 5) -> float:
    coms = np.asarray(coms, dtype=float)
    mask = np.isfinite(coms).all(axis=1)
    coms = coms[mask]
    if len(coms) < 5:
        return -np.inf
    hist, _, _ = np.histogram2d(
        coms[:, 0], coms[:, 1], bins=bins, range=[[0.0, 1.0], [0.0, 1.0]],
    )
    p = hist / hist.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))


def load_seed_data(seed: int):
    run_dir = MULTI_SEED_DIR / f"head_seed{seed}"
    data_dir = run_dir / "data"
    if not data_dir.exists():
        return None
    try:
        phases = load_phases(data_dir)
        ids = shared_place_cell_ids(phases)
        stats = paired_distance_stats(phases, REWARD_POS, ids)
        return phases, stats, ids
    except Exception as exc:
        print(f"  [seed {seed}] load error: {exc}")
        return None


def score_seeds(seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        data = load_seed_data(seed)
        if data is None:
            print(f"Seed {seed}: missing / failed — skipping")
            continue
        phases, stats, ids = data
        com_a = phases["A"]["com"][ids]
        entropy = com_spatial_entropy(com_a)
        rows.append({
            "seed": seed,
            "entropy": entropy,
            "n_cells": int(len(ids)),
            "com_mean_delta_AB": float(stats["com_mean_delta_AB"]),
            "com_frac_toward_AB": float(stats["com_frac_toward_AB"]),
            "phases": phases,
            "stats": stats,
            "ids": ids,
        })
        print(
            f"  seed={seed:2d}  n={len(ids):3d}  "
            f"entropy={entropy:.3f}  "
            f"Δd(A→B)={stats['com_mean_delta_AB']:+.3f}  "
            f"frac_toward={100*stats['com_frac_toward_AB']:.0f}%"
        )
    rows.sort(key=lambda r: r["entropy"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Aggregate panels
# ---------------------------------------------------------------------------

def fig_aggregate_panel(rows: list[dict], out_stem: Path):
    seeds = [r["seed"] for r in rows]
    delta_ab_per_seed = [np.asarray(r["stats"]["delta_com_AB"], dtype=float) for r in rows]
    frac_toward = np.array([r["com_frac_toward_AB"] for r in rows])
    mean_delta = np.array([d.mean() for d in delta_ab_per_seed])

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.2))

    ax = axes[0]
    x = np.zeros(len(seeds))
    ax.scatter(x + np.random.default_rng(0).uniform(-0.08, 0.08, len(seeds)),
               mean_delta, s=12, color="#8B4513", alpha=0.7, zorder=3, linewidths=0)
    ax.errorbar([0], [mean_delta.mean()], yerr=[mean_delta.std(ddof=1)],
                fmt="D", color="black", markersize=5, capsize=4, linewidth=1.2, zorder=5,
                label=f"mean±SD\n{mean_delta.mean():+.3f}±{mean_delta.std(ddof=1):.3f}")
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks([0])
    ax.set_xticklabels(["A→B"])
    ax.set_ylabel(r"mean $\Delta$ COM dist to reward")
    ax.set_title(f"Per-seed mean shift (head ON)\n(n={len(seeds)} seeds)", fontsize=6)
    ax.legend(fontsize=5, loc="lower right")

    ax = axes[1]
    ax.scatter(np.zeros(len(seeds)) + np.random.default_rng(1).uniform(-0.08, 0.08, len(seeds)),
               frac_toward * 100, s=12, color="#CC6677", alpha=0.7, zorder=3, linewidths=0)
    ax.errorbar([0], [frac_toward.mean() * 100], yerr=[frac_toward.std(ddof=1) * 100],
                fmt="D", color="black", markersize=5, capsize=4, linewidth=1.2, zorder=5,
                label=f"{frac_toward.mean()*100:.1f}±{frac_toward.std(ddof=1)*100:.1f}%")
    ax.axhline(50, color="0.4", lw=0.8, ls="--", label="chance")
    ax.set_xticks([0])
    ax.set_xticklabels(["A→B"])
    ax.set_ylabel("% cells shifting toward reward")
    ax.set_title(f"Fraction toward reward (head ON)\n(n={len(seeds)} seeds)", fontsize=6)
    ax.legend(fontsize=5, loc="lower right")
    ax.set_ylim(0, 100)

    fig.suptitle("Aggregate across seeds (reward-prediction head)", fontsize=6.5, y=1.02)
    fig.tight_layout(pad=0.4, w_pad=0.5)

    out_stem = Path(out_stem)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_stem.with_suffix('.pdf')}")


def save_aggregate_json(rows: list[dict], best_seed: int, out_path: Path):
    all_delta_ab = np.concatenate(
        [np.asarray(r["stats"]["delta_com_AB"], dtype=float) for r in rows]
    )
    per_seed_means = [float(np.asarray(r["stats"]["delta_com_AB"], dtype=float).mean())
                      for r in rows]
    frac_toward = [float(r["com_frac_toward_AB"]) for r in rows]

    if len(per_seed_means) >= 5:
        try:
            wstat, wp = _wilcoxon(per_seed_means, alternative="less")
        except Exception:
            wstat, wp = float("nan"), float("nan")
    else:
        wstat, wp = float("nan"), float("nan")

    summary = {
        "figure": "3_pred_head",
        "best_seed": int(best_seed),
        "n_seeds": int(len(rows)),
        "per_seed_entropy": {str(r["seed"]): float(r["entropy"]) for r in rows},
        "per_seed_n_cells": {str(r["seed"]): int(r["n_cells"]) for r in rows},
        "per_seed_com_mean_delta_AB": {str(r["seed"]): float(r["com_mean_delta_AB"]) for r in rows},
        "per_seed_frac_toward_AB": {str(r["seed"]): float(r["com_frac_toward_AB"]) for r in rows},
        "aggregate": {
            "pooled_n_cell_pairs": int(len(all_delta_ab)),
            "pooled_mean_delta_AB": float(all_delta_ab.mean()),
            "pooled_std_delta_AB": float(all_delta_ab.std(ddof=1)),
            "per_seed_mean_delta_AB_mean": float(np.mean(per_seed_means)),
            "per_seed_mean_delta_AB_std": float(np.std(per_seed_means, ddof=1)),
            "per_seed_frac_toward_mean": float(np.mean(frac_toward)),
            "per_seed_frac_toward_std": float(np.std(frac_toward, ddof=1)),
            "wilcoxon_vs_zero_stat": float(wstat),
            "wilcoxon_vs_zero_p": float(wp),
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--list_only", action="store_true")
    p.add_argument("--best_seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    agg_dir = MULTI_SEED_DIR / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    print("Scoring head-condition seeds by COM uniformity (Phase A spatial entropy)...")
    rows = score_seeds(args.seeds)

    if not rows:
        print("ERROR: no successful seeds found.")
        sys.exit(1)

    print(f"\nRanking (best first by COM spatial entropy):")
    for i, r in enumerate(rows):
        marker = " ← BEST" if i == 0 else ""
        print(f"  #{i+1}  seed={r['seed']:2d}  entropy={r['entropy']:.3f}  "
              f"n={r['n_cells']}  Δd={r['com_mean_delta_AB']:+.3f}{marker}")

    best = rows[0] if args.best_seed is None else next(
        (r for r in rows if r["seed"] == args.best_seed), rows[0]
    )
    print(f"\nBest seed: {best['seed']}  (entropy={best['entropy']:.3f}, n={best['n_cells']})")

    if args.list_only:
        return

    print("\nSaving aggregate stats...")
    agg_json = save_aggregate_json(rows, best["seed"], agg_dir / "fig3_multi_seed_summary.json")

    print("\nGenerating aggregate panels...")
    fig_aggregate_panel(rows, agg_dir / "panel_e_aggregate_strip")

    # Auto-regenerate Figure 3 with best seed
    best_run_dir = MULTI_SEED_DIR / f"head_seed{best['seed']}"
    fig3_out = FIG3_DIR / "figures"
    cmd = [
        sys.executable,
        str(FIG3_DIR / "generate_figure3.py"),
        "--run_dir", str(best_run_dir),
        "--output_dir", str(fig3_out),
        "--reward_pos", "0.3", "0.3",
        "--no_show",
    ]
    print(f"\nAuto-regenerating Figure 3 with best seed ({best['seed']})...")
    result = subprocess.run(cmd, cwd=str(FIG3_DIR))
    if result.returncode != 0:
        print("WARNING: Figure 3 generation failed. Run generate_figure3.py manually.")
    else:
        print("Figure 3 regenerated.")

    # Also run supp figure
    supp_cmd = [
        sys.executable,
        str(FIG3_DIR / "generate_figure3_supp.py"),
        "--multi_seed_dir", str(MULTI_SEED_DIR),
        "--nohead_dir", str(REPO / "results" / "multi_seed"),
        "--seeds", *[str(s) for s in args.seeds],
        "--no_show",
    ]
    print("Auto-regenerating Supplementary Figure 3 (10-seed + head vs no-head)...")
    result = subprocess.run(supp_cmd, cwd=str(FIG3_DIR))
    if result.returncode != 0:
        print("WARNING: Supp Figure 3 generation failed.")
    else:
        print("Supplementary Figure 3 regenerated.")

    agg = agg_json["aggregate"]
    print(f"\nAggregate summary (head condition):")
    print(f"  Seeds evaluated:      {agg_json['n_seeds']}")
    print(f"  Pooled cell-pairs:    {agg['pooled_n_cell_pairs']}")
    print(f"  Pooled mean Δd(A→B):  {agg['pooled_mean_delta_AB']:+.4f}")
    print(f"  Per-seed mean Δd:     {agg['per_seed_mean_delta_AB_mean']:+.4f} ± {agg['per_seed_mean_delta_AB_std']:.4f}")
    print(f"  Frac toward reward:   {agg['per_seed_frac_toward_mean']*100:.1f}% ± {agg['per_seed_frac_toward_std']*100:.1f}%")
    print(f"  Wilcoxon p (vs 0):    {agg['wilcoxon_vs_zero_p']:.4f}")


if __name__ == "__main__":
    main()
