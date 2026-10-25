#!/usr/bin/env python3
"""
Drive occupancy_matched_estimator over all no-head + head seeds, both post
periods, and cache per-seed rows + validation summary to
results/aggregate/occupancy_matched_estimator.json.

Post periods:
  - cycle_1_goal      : same weights as the ground-truth post snapshot
                        (replay_c1_postrl) -- exact weight alignment, few steps.
  - goal_conditioned  : post-interleave weights (the interleave is the map
                        stabilizer, so approximately aligned), 10x more steps.

Usage:
    python code/aggregate/run_estimator_validation.py [--seeds 0 1 2] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "lib"))

from analysis_helpers import campaign_passers  # noqa: E402

CONDS = [("nohead", REPO / "results" / "multi_seed", "nohead_"),
         ("head", REPO / "results" / "multi_seed_head", "head_")]
POST_PERIODS = ["cycle_1_goal", "goal_conditioned"]
GATE_PERIODS = ["replay_c1_pred", "replay_c1_postrl"]
OUT = REPO / "results" / "aggregate" / "occupancy_matched_estimator.json"


def one_job(cond: str, run_dir: str, seed: int, post_period: str, n_reps: int):
    # imported inside the worker so the pool forks cleanly
    from occupancy_matched_estimator import estimator_row
    row = estimator_row(Path(run_dir), seed, n_reps=n_reps, post_period=post_period)
    return cond, post_period, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n_reps", type=int, default=20)
    args = ap.parse_args()

    jobs = []
    for cond, cdir, prefix in CONDS:
        if args.seeds is not None:
            seeds = list(args.seeds)
        else:
            # Every finished seed directory, not only 0–49. The cohort is
            # collected until 50 gate-passing networks, so later indices
            # are part of the result.
            seeds = []
            stem = prefix + "seed"
            for p in sorted(cdir.glob(f"{stem}*")):
                if p.is_dir() and p.name[len(stem):].isdigit():
                    seeds.append(int(p.name[len(stem):]))
        for s in seeds:
            rd = cdir / f"{prefix}seed{s}"
            if not (rd / "data").exists():
                continue
            if s not in campaign_passers("paired"):     # the pre-specified cohort
                continue
            for pp in POST_PERIODS:
                jobs.append((cond, str(rd), s, pp))
    print(f"{len(jobs)} jobs ({len(jobs)//2} seed-conditions x {len(POST_PERIODS)} post periods)")

    results: dict = {pp: {"nohead": [], "head": []} for pp in POST_PERIODS}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one_job, c, rd, s, pp, args.n_reps) for c, rd, s, pp in jobs]
        done = 0
        for f in as_completed(futs):
            cond, pp, row = f.result()
            done += 1
            if row:
                results[pp][cond].append(row)
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} done", flush=True)

    # validation summary per post period
    summary: dict = {}
    for pp in POST_PERIODS:
        s: dict = {}
        for cond in ("nohead", "head"):
            rows = sorted(results[pp][cond], key=lambda r: r["seed"])
            results[pp][cond] = rows
            if not rows:
                continue
            gt = np.array([r["gt_mean_dd"] for r in rows])
            for mode in ("naive", "pos", "joint", "reweight"):
                est = np.array([r.get(f"est_{mode}") for r in rows], dtype=float)
                ok = np.isfinite(est)
                if ok.sum() > 2:
                    r_p = float(np.corrcoef(est[ok], gt[ok])[0, 1])
                    slope = float(np.polyfit(gt[ok], est[ok], 1)[0])
                else:
                    r_p, slope = float("nan"), float("nan")
                s[f"{cond}_{mode}"] = {
                    "mean": float(np.nanmean(est)), "sd": float(np.nanstd(est, ddof=1)),
                    "r_vs_gt": r_p, "slope_vs_gt": slope, "n": int(ok.sum()),
                }
            s[f"{cond}_gt"] = {"mean": float(gt.mean()), "sd": float(gt.std(ddof=1))}
            nulls = np.array([r.get("null_joint") for r in rows], dtype=float)
            s[f"{cond}_null_joint"] = {"mean": float(np.nanmean(nulls)),
                                       "sd": float(np.nanstd(nulls, ddof=1))}
            nulls_nv = np.array([r.get("null_naive") for r in rows], dtype=float)
            s[f"{cond}_null_naive"] = {"mean": float(np.nanmean(nulls_nv)),
                                       "sd": float(np.nanstd(nulls_nv, ddof=1))}

        # paired ordering across matched seeds (paired: same seed = same
        # pre-learning network in both conditions)
        nh = {r["seed"]: r for r in results[pp]["nohead"]}
        hd = {r["seed"]: r for r in results[pp]["head"]}
        common = sorted(set(nh) & set(hd))
        if len(common) >= 5:
            from scipy.stats import wilcoxon
            ords = {}
            for mode in ("naive", "joint"):
                a = np.array([nh[s_]["est_" + mode] for s_ in common], dtype=float)
                b = np.array([hd[s_]["est_" + mode] for s_ in common], dtype=float)
                g_a = np.array([nh[s_]["gt_mean_dd"] for s_ in common])
                g_b = np.array([hd[s_]["gt_mean_dd"] for s_ in common])
                ok = np.isfinite(a) & np.isfinite(b)
                ords[mode] = {
                    "n_pairs": int(ok.sum()),
                    "nohead_mean": float(np.nanmean(a)), "head_mean": float(np.nanmean(b)),
                    "head_more_toward": bool(np.nanmean(b) < np.nanmean(a)),
                    "paired_p": float(wilcoxon(a[ok], b[ok]).pvalue) if ok.sum() > 4 else None,
                }
                ords.setdefault("gt", {
                    "nohead_mean": float(g_a.mean()), "head_mean": float(g_b.mean()),
                    "head_more_toward": bool(g_b.mean() < g_a.mean()),
                    "paired_p": float(wilcoxon(g_a[ok], g_b[ok]).pvalue),
                })
            s["ordering"] = ords
        summary[pp] = s

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"params": {"n_reps": args.n_reps, "grid": 32, "headings": 8,
                              "trim_after_reset": 5, "skaggs_thr": 0.3,
                              "convention": "negative = toward reward"},
                   "per_seed": results, "summary": summary}, f, indent=2)
    print(f"\nSaved: {OUT}")

    for pp in POST_PERIODS:
        print(f"\n===== post = {pp} =====")
        s = summary[pp]
        for k in ("nohead_gt", "head_gt", "nohead_naive", "head_naive",
                  "nohead_joint", "head_joint", "nohead_null_naive", "nohead_null_joint"):
            if k in s:
                v = s[k]
                extra = f"  r_vs_gt={v.get('r_vs_gt'):.3f} slope={v.get('slope_vs_gt'):.3f}" \
                    if "r_vs_gt" in v and np.isfinite(v.get("r_vs_gt", float("nan"))) else ""
                print(f"  {k:22s} mean={v['mean']:+.4f} sd={v['sd']:.4f}{extra}")
        if "ordering" in s:
            for mode, o in s["ordering"].items():
                print(f"  ordering[{mode}]: nohead={o['nohead_mean']:+.4f} head={o['head_mean']:+.4f} "
                      f"head_more_toward={o['head_more_toward']} p={o['paired_p']}")


if __name__ == "__main__":
    main()
