#!/usr/bin/env python3
"""
Score 10 multi-seed runs by COM uniformity, pick the best example for Figure 2,
and regenerate figure 2 with:
  - Panels a–d from the best seed (replacing nohead_seed42)
  - A new panel e: aggregate statistics across all seeds

Usage (run from repo root after run_fig2_seeds.py completes):
    python code/runs/select_best_seed_fig2.py
    python code/runs/select_best_seed_fig2.py --seeds 0 1 2 3 4 5 6 7 8 9
    python code/runs/select_best_seed_fig2.py --list_only   # just print rankings
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon as _wilcoxon

REPO = Path(__file__).resolve().parents[2]
FIG2_DIR = REPO / "code" / "figures" / "fig2"
FIG1_DIR = REPO / "code" / "figures" / "fig1"
MULTI_SEED_DIR = REPO / "results" / "multi_seed"

# FIG2_DIR must come first so figure 2's utils.py shadows figure 1's
sys.path.insert(0, str(FIG1_DIR))
sys.path.insert(0, str(FIG2_DIR))

from analysis import load_phases, shared_place_cell_ids, paired_distance_stats  # noqa: E402
from utils import DEFAULT_REWARD, apply_nature_style  # noqa: E402

apply_nature_style()

REWARD_POS = np.asarray(DEFAULT_REWARD, dtype=float)


# ---------------------------------------------------------------------------
# Uniformity scoring
# ---------------------------------------------------------------------------

def com_spatial_entropy(coms: np.ndarray, bins: int = 5) -> float:
    """Spatial entropy of COM positions on a bins×bins grid over [0,1]^2.

    Higher = more uniform coverage of the arena.
    Only uses valid (non-NaN) COMs.
    """
    coms = np.asarray(coms, dtype=float)
    mask = np.isfinite(coms).all(axis=1)
    coms = coms[mask]
    if len(coms) < 5:
        return -np.inf
    hist, _, _ = np.histogram2d(
        coms[:, 0], coms[:, 1],
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    p = hist / hist.sum()
    # Shannon entropy; treat 0 as 0*log(0)=0
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = float(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))
    return ent


def load_seed_data(seed: int):
    """Return (phases, stats, ids) for one seed, or None on failure."""
    run_dir = MULTI_SEED_DIR / f"nohead_seed{seed}"
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
# Aggregate panel across all seeds
# ---------------------------------------------------------------------------

def fig_aggregate_panel(rows: list[dict], out_stem: Path):
    """Panel e — mean ± SD of COM Δd (A→B) and fraction toward reward across seeds."""
    seeds = [r["seed"] for r in rows]
    delta_ab_per_seed = [np.asarray(r["stats"]["delta_com_AB"], dtype=float) for r in rows]
    frac_toward = np.array([r["com_frac_toward_AB"] for r in rows])

    # Per-seed mean Δd
    mean_delta = np.array([d.mean() for d in delta_ab_per_seed])

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.2))

    # Left: per-seed mean COM Δd strip chart + grand mean
    ax = axes[0]
    x = np.zeros(len(seeds))
    ax.scatter(x + np.random.default_rng(0).uniform(-0.08, 0.08, len(seeds)),
               mean_delta, s=12, color="#2E5FA3", alpha=0.7, zorder=3, linewidths=0)
    ax.errorbar([0], [mean_delta.mean()], yerr=[mean_delta.std(ddof=1)],
                fmt="D", color="black", markersize=5, capsize=4, linewidth=1.2, zorder=5,
                label=f"mean±SD\n{mean_delta.mean():+.3f}±{mean_delta.std(ddof=1):.3f}")
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks([0])
    ax.set_xticklabels(["A→B"])
    ax.set_ylabel(r"mean $\Delta$ COM dist to reward")
    ax.set_title(f"Per-seed mean shift\n(n={len(seeds)} seeds)", fontsize=6)
    ax.legend(fontsize=5, loc="lower right")

    # Right: fraction toward reward per seed
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
    ax.set_title(f"Fraction toward reward\n(n={len(seeds)} seeds)", fontsize=6)
    ax.legend(fontsize=5, loc="lower right")
    ax.set_ylim(0, 100)

    fig.suptitle("Aggregate across seeds", fontsize=6.5, y=1.02)
    fig.tight_layout(pad=0.4, w_pad=0.5)

    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")
    return png, pdf


def fig_aggregate_violin(rows: list[dict], out_stem: Path):
    """Alternative: pooled violin of Δd across all seeds, overlaid per-seed medians."""
    all_deltas = np.concatenate([np.asarray(r["stats"]["delta_com_AB"], dtype=float) for r in rows])
    per_seed_means = np.array([np.asarray(r["stats"]["delta_com_AB"], dtype=float).mean() for r in rows])

    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    parts = ax.violinplot([all_deltas], positions=[1], showmeans=False,
                          showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#2E5FA3")
        body.set_alpha(0.55)
        body.set_edgecolor("0.2")
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("0.25")
            parts[key].set_linewidth(0.9)

    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(rows))
    ax.scatter(np.ones(len(rows)) + jitter, per_seed_means,
               s=14, color="black", zorder=5, linewidths=0, alpha=0.85,
               label="seed mean")

    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks([1])
    ax.set_xticklabels(["A→B"])
    ax.set_ylabel(r"$\Delta$ COM dist to reward")
    n_cells_total = sum(r["n_cells"] for r in rows)
    ax.set_title(
        f"Pooled across {len(rows)} seeds\n"
        f"(n={n_cells_total} cell-pairs total)\n"
        f"mean={all_deltas.mean():+.3f}",
        fontsize=5.5, pad=2,
    )
    ax.legend(fontsize=5)
    fig.tight_layout(pad=0.4)

    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")
    return png, pdf


def save_aggregate_json(rows: list[dict], best_seed: int, out_path: Path):
    all_delta_ab = np.concatenate(
        [np.asarray(r["stats"]["delta_com_AB"], dtype=float) for r in rows]
    )
    per_seed_means = [float(np.asarray(r["stats"]["delta_com_AB"], dtype=float).mean())
                      for r in rows]
    frac_toward = [float(r["com_frac_toward_AB"]) for r in rows]

    # Wilcoxon signed-rank test on per-seed means vs 0
    if len(per_seed_means) >= 5:
        try:
            wstat, wp = _wilcoxon(per_seed_means, alternative="less")
        except Exception:
            wstat, wp = float("nan"), float("nan")
    else:
        wstat, wp = float("nan"), float("nan")

    summary = {
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
    p.add_argument("--list_only", action="store_true", help="Print rankings without generating figure")
    p.add_argument("--best_seed", type=int, default=None,
                   help="Override best seed (skip scoring, use this one)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    agg_dir = MULTI_SEED_DIR / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    print("Scoring seeds by COM uniformity (Phase A spatial entropy)...")
    rows = score_seeds(args.seeds)

    if not rows:
        print("ERROR: no successful seeds found.")
        sys.exit(1)

    print(f"\nRanking (best first by COM spatial entropy):")
    for i, r in enumerate(rows):
        marker = " ← BEST" if i == 0 else ""
        print(f"  #{i+1}  seed={r['seed']:2d}  entropy={r['entropy']:.3f}  "
              f"n_cells={r['n_cells']}  Δd={r['com_mean_delta_AB']:+.3f}{marker}")

    best = rows[0] if args.best_seed is None else next(
        (r for r in rows if r["seed"] == args.best_seed), rows[0]
    )
    print(f"\nBest seed: {best['seed']}  (entropy={best['entropy']:.3f}, n={best['n_cells']})")

    if args.list_only:
        return

    # --- Save aggregate JSON ---
    print("\nSaving aggregate stats...")
    agg_json = save_aggregate_json(rows, best["seed"], agg_dir / "fig2_multi_seed_summary.json")

    # --- Generate aggregate panels ---
    print("\nGenerating aggregate panels...")
    fig_aggregate_panel(rows, agg_dir / "panel_e_aggregate_strip")
    fig_aggregate_violin(rows, agg_dir / "panel_e_aggregate_violin")

    # --- Print instructions for regenerating figure 2 with best seed ---
    best_run_dir = MULTI_SEED_DIR / f"nohead_seed{best['seed']}"
    fig2_out = FIG2_DIR / "figures"
    print(f"""
{'='*60}
To regenerate Figure 2 with the best seed ({best['seed']}), run:

    cd {REPO}
    python code/figures/fig2/generate_figure2.py \\
        --run_dir "{best_run_dir}" \\
        --output_dir "{fig2_out}" \\
        --reward_pos 0.3 0.3 \\
        --no_show

Aggregate panels are in:
    {agg_dir}

Aggregate summary JSON:
    {agg_dir / 'fig2_multi_seed_summary.json'}
{'='*60}
""")

    # --- Auto-regenerate figure 2 with best seed ---
    import subprocess
    cmd = [
        sys.executable,
        str(FIG2_DIR / "generate_figure2.py"),
        "--run_dir", str(best_run_dir),
        "--output_dir", str(fig2_out),
        "--reward_pos", "0.3", "0.3",
        "--no_show",
    ]
    print("Auto-regenerating Figure 2 with best seed...")
    result = subprocess.run(cmd, cwd=str(FIG2_DIR))
    if result.returncode != 0:
        print("WARNING: figure 2 generation failed.")
    else:
        print("Figure 2 regenerated.")

    print("\nAggregate summary:")
    agg = agg_json["aggregate"]
    print(f"  Seeds evaluated:      {agg_json['n_seeds']}")
    print(f"  Pooled cell-pairs:    {agg['pooled_n_cell_pairs']}")
    print(f"  Pooled mean Δd(A→B):  {agg['pooled_mean_delta_AB']:+.4f}")
    print(f"  Per-seed mean Δd:     {agg['per_seed_mean_delta_AB_mean']:+.4f} ± {agg['per_seed_mean_delta_AB_std']:.4f}")
    print(f"  Frac toward reward:   {agg['per_seed_frac_toward_mean']*100:.1f}% ± {agg['per_seed_frac_toward_std']*100:.1f}%")
    print(f"  Wilcoxon p (vs 0):    {agg['wilcoxon_vs_zero_p']:.4f}")


if __name__ == "__main__":
    main()
