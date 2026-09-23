#!/usr/bin/env python3
"""
Score relocation (Fig 4) multi-seed runs by tracking quality, pick the
best example, and regenerate Figure 4 with aggregate supplementary panels.

Seeds are first filtered through the entropy-collapse gate (seed_is_clean,
shared with Figure 5's diagnostic) — a seed whose post-relocation field
distribution has collapsed into a tiny region gives an artificially high
tracking score without genuine tracking, so it's excluded before ranking.

Tracking quality = harmonic mean of:
  - fraction of cells whose fields move toward A (pre → postA)
  - fraction of cells whose fields move toward B (postA → postB)

A good example seed has both fractions well above chance.

Usage (run from repo root after run_fig34_parallel.py completes):
    python code/runs/select_best_seed_fig4.py
    python code/runs/select_best_seed_fig4.py --seeds 0 1 2 3 4 5 6 7 8 9
    python code/runs/select_best_seed_fig4.py --list_only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
FIG4_DIR = REPO / "code" / "figures" / "fig4"
MULTI_SEED_DIR = REPO / "results" / "multi_seed_reloc"
NEW_CODE = REPO / "code" / "lib"

sys.path.insert(0, str(NEW_CODE))
from analysis_helpers import load_period_centers, has_period, seed_is_clean  # noqa: E402

A = np.array([0.3, 0.3])
B = np.array([0.7, 0.7])
SKAGGS_THRESHOLD = 0.15
PERIODS = ["replay_reloc_pred", "replay_reloc_postA", "replay_reloc_postB"]


def _wilcoxon_p(a, b):
    if len(a) < 5 or np.all(np.abs(a - b) < 1e-12):
        return float("nan")
    try:
        return float(wilcoxon(a, b)[1])
    except Exception:
        return float("nan")


def load_seed_data(seed: int):
    run_dir = MULTI_SEED_DIR / f"reloc_seed{seed}"
    data_dir = str(run_dir / "data")
    if not os.path.exists(data_dir):
        return None
    for period in PERIODS:
        if not has_period(data_dir, period):
            return None
    try:
        states = {}
        for period in PERIODS:
            sk, com, peak = load_period_centers(data_dir, period)
            states[period] = {"sk": sk, "peak": peak, "com": com}
        return states
    except Exception as exc:
        print(f"  [seed {seed}] load error: {exc}")
        return None


def compute_tracking_stats(states: dict) -> dict:
    """Compute tracking metrics (same as reward_relocation_analysis.py logic)."""
    pre, pA, pB = PERIODS

    def paired_shift(p_from, p_to, reward):
        sk_f = states[p_from]["sk"]
        sk_t = states[p_to]["sk"]
        active = np.where(np.maximum(sk_f, sk_t) >= SKAGGS_THRESHOLD)[0]
        d_f = np.linalg.norm(states[p_from]["peak"][active] - reward[None, :], axis=1)
        d_t = np.linalg.norm(states[p_to]["peak"][active] - reward[None, :], axis=1)
        d = d_t - d_f
        n = int(len(active))
        frac = float(np.mean(d < 0)) if n > 0 else float("nan")
        mean_d = float(np.mean(d)) if n > 0 else float("nan")
        p = _wilcoxon_p(d_t, d_f)
        return {"mean_delta": mean_d, "frac_toward": frac, "wilcoxon_p": p, "n": n,
                "delta": d}

    shiftA = paired_shift(pre, pA, A)
    shiftB = paired_shift(pA, pB, B)

    # Harmonic mean of the two fractions (tracking quality score)
    fa, fb = shiftA["frac_toward"], shiftB["frac_toward"]
    if fa > 0 and fb > 0:
        tracking_score = 2 * fa * fb / (fa + fb)
    else:
        tracking_score = 0.0

    return {
        "toward_A": shiftA,
        "toward_B": shiftB,
        "tracking_score": tracking_score,
    }


def score_seeds(seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        data_dir = MULTI_SEED_DIR / f"reloc_seed{seed}" / "data"
        if not seed_is_clean(data_dir, PERIODS):
            print(f"Seed {seed}: FAILS entropy gate (collapsed field distribution) — excluded from ranking")
            continue
        data = load_seed_data(seed)
        if data is None:
            print(f"Seed {seed}: missing / failed — skipping")
            continue
        ts = compute_tracking_stats(data)
        row = {
            "seed": seed,
            "tracking_score": ts["tracking_score"],
            "frac_toward_A": ts["toward_A"]["frac_toward"],
            "frac_toward_B": ts["toward_B"]["frac_toward"],
            "mean_delta_A": ts["toward_A"]["mean_delta"],
            "mean_delta_B": ts["toward_B"]["mean_delta"],
            "wilcoxon_p_A": ts["toward_A"]["wilcoxon_p"],
            "wilcoxon_p_B": ts["toward_B"]["wilcoxon_p"],
            "n_A": ts["toward_A"]["n"],
            "n_B": ts["toward_B"]["n"],
            "delta_A": ts["toward_A"]["delta"],
            "delta_B": ts["toward_B"]["delta"],
        }
        rows.append(row)
        print(
            f"  seed={seed:2d}  score={ts['tracking_score']:.3f}  "
            f"→A: {100*ts['toward_A']['frac_toward']:.0f}% (Δ={ts['toward_A']['mean_delta']:+.3f}, p={ts['toward_A']['wilcoxon_p']:.1e})  "
            f"→B: {100*ts['toward_B']['frac_toward']:.0f}% (Δ={ts['toward_B']['mean_delta']:+.3f}, p={ts['toward_B']['wilcoxon_p']:.1e})"
        )
    rows.sort(key=lambda r: r["tracking_score"], reverse=True)
    return rows


def save_aggregate_json(rows: list[dict], best_seed: int, out_path: Path):
    per_seed_fa = [r["frac_toward_A"] for r in rows]
    per_seed_fb = [r["frac_toward_B"] for r in rows]
    all_delta_A = np.concatenate([r["delta_A"] for r in rows])
    all_delta_B = np.concatenate([r["delta_B"] for r in rows])

    def _wil(vals, alternative="less"):
        try:
            _, p = wilcoxon(vals, alternative=alternative)
            return float(p)
        except Exception:
            return float("nan")

    summary = {
        "figure": "4_relocation",
        "best_seed": int(best_seed),
        "n_seeds": int(len(rows)),
        "reward_A": A.tolist(),
        "reward_B": B.tolist(),
        "per_seed": {
            str(r["seed"]): {
                "tracking_score": float(r["tracking_score"]),
                "frac_toward_A": float(r["frac_toward_A"]),
                "frac_toward_B": float(r["frac_toward_B"]),
                "mean_delta_A": float(r["mean_delta_A"]),
                "mean_delta_B": float(r["mean_delta_B"]),
                "wilcoxon_p_A": float(r["wilcoxon_p_A"]),
                "wilcoxon_p_B": float(r["wilcoxon_p_B"]),
                "n_A": int(r["n_A"]),
                "n_B": int(r["n_B"]),
            }
            for r in rows
        },
        "aggregate": {
            "frac_toward_A_mean": float(np.mean(per_seed_fa)),
            "frac_toward_A_std": float(np.std(per_seed_fa, ddof=1)),
            "frac_toward_B_mean": float(np.mean(per_seed_fb)),
            "frac_toward_B_std": float(np.std(per_seed_fb, ddof=1)),
            "pooled_n_A": int(len(all_delta_A)),
            "pooled_n_B": int(len(all_delta_B)),
            "pooled_mean_delta_A": float(all_delta_A.mean()),
            "pooled_mean_delta_B": float(all_delta_B.mean()),
            "wilcoxon_p_A_pooled": _wil(per_seed_fa, alternative="greater"),
            "wilcoxon_p_B_pooled": _wil(per_seed_fb, alternative="greater"),
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")
    return summary


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

    print("Scoring relocation seeds by tracking quality (harmonic mean frac→A, frac→B)...")
    rows = score_seeds(args.seeds)

    if not rows:
        print("ERROR: no successful seeds found.")
        sys.exit(1)

    print(f"\nRanking (best first by tracking score):")
    for i, r in enumerate(rows):
        marker = " ← BEST" if i == 0 else ""
        print(f"  #{i+1}  seed={r['seed']:2d}  score={r['tracking_score']:.3f}  "
              f"→A={100*r['frac_toward_A']:.0f}%  →B={100*r['frac_toward_B']:.0f}%{marker}")

    best = rows[0] if args.best_seed is None else next(
        (r for r in rows if r["seed"] == args.best_seed), rows[0]
    )
    if args.best_seed is not None:
        override_dir = MULTI_SEED_DIR / f"reloc_seed{args.best_seed}" / "data"
        if not seed_is_clean(override_dir, PERIODS):
            print(f"WARNING: manually-selected seed {args.best_seed} FAILS the entropy gate.")
    print(f"\nBest seed: {best['seed']}  (score={best['tracking_score']:.3f})")

    if args.list_only:
        return

    print("\nSaving aggregate stats...")
    agg_json = save_aggregate_json(rows, best["seed"], agg_dir / "fig4_multi_seed_summary.json")

    # Auto-regenerate Figure 4 with best seed
    best_run_dir = MULTI_SEED_DIR / f"reloc_seed{best['seed']}"
    fig4_out = FIG4_DIR / "figures"
    cmd = [
        sys.executable,
        str(FIG4_DIR / "generate_figure4.py"),
        "--run_dir", str(best_run_dir),
        "--output_dir", str(fig4_out),
        "--no_show",
    ]
    print(f"\nAuto-regenerating Figure 4 with best seed ({best['seed']})...")
    result = subprocess.run(cmd, cwd=str(FIG4_DIR))
    if result.returncode != 0:
        print("WARNING: Figure 4 generation failed. Run generate_figure4.py manually.")
    else:
        print("Figure 4 regenerated.")

    # Also run supp figure
    supp_cmd = [
        sys.executable,
        str(FIG4_DIR / "generate_figure4_supp.py"),
        "--multi_seed_dir", str(MULTI_SEED_DIR),
        "--seeds", *[str(s) for s in args.seeds],
        "--no_show",
    ]
    print("Auto-regenerating Supplementary Figure 4 (10-seed tracking stats)...")
    result = subprocess.run(supp_cmd, cwd=str(FIG4_DIR))
    if result.returncode != 0:
        print("WARNING: Supp Figure 4 generation failed.")
    else:
        print("Supplementary Figure 4 regenerated.")

    agg = agg_json["aggregate"]
    print(f"\nAggregate summary (relocation):")
    print(f"  Seeds evaluated:    {agg_json['n_seeds']}")
    print(f"  Frac toward A:      {agg['frac_toward_A_mean']*100:.1f}% ± {agg['frac_toward_A_std']*100:.1f}%")
    print(f"  Frac toward B:      {agg['frac_toward_B_mean']*100:.1f}% ± {agg['frac_toward_B_std']*100:.1f}%")
    print(f"  Mean ΔdA (pooled):  {agg['pooled_mean_delta_A']:+.4f}")
    print(f"  Mean ΔdB (pooled):  {agg['pooled_mean_delta_B']:+.4f}")


if __name__ == "__main__":
    main()
