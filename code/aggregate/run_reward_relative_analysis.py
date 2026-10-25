#!/usr/bin/env python3
"""
Aggregate code/lib/reward_relative_analysis.per_seed_row over all
entropy-gated relocation seeds and cache group-level statistics to
results/multi_seed_reloc/aggregate/reward_relative_summary.json.

why this file exists: that JSON is read by figures/fig5/generate_figure5_data.py
(panel comparing model reward-relative fractions against the Sosa 2025
reanalysis), and it is committed to the repo -- but no driver that produces it
survived. `reward_relative_analysis.py` states "entropy gate handled by the
caller" and exposes `per_seed_row`, with nothing calling it. This script is that
missing caller, reconstructed to match the existing cache's schema. It closes
the one place where a shipped figure could not be regenerated from scratch.

Cohort: the retained relocation networks, analysis_helpers.campaign_passers("reloc")
-- the same entropy-collapse gate the other relocation analyses use (see
figures/figS_reward_vector/analysis.py) -- applied before the per-cell
Skaggs>=0.15 gate inside reward_relative_analysis itself.

Reported per read-out (behavioural = cycle_1_goal vs cycle_2_goal, what an
experimenter without occupancy control sees; controlled = replay_reloc_postA vs
postB on an identical frozen trajectory, primary):
  frac_reward_relative / frac_place_anchored group mean+SD, and for the
  controlled read-out the RRI mean, the count of seeds with
  place-anchored > reward-relative, and the paired Wilcoxon between them.
Also the paired Wilcoxon of frac_reward_relative behavioural-vs-controlled.

The heavy per-cell detail (`cells`) that per_seed_row returns is stripped before
writing, matching the committed cache.

NOTE: not yet validated against the committed cache -- run with --out to a
scratch path and diff before trusting it (the existing file is protected: this
script refuses to overwrite it without --force).

Usage:
    python code/aggregate/run_reward_relative_analysis.py --out /tmp/rr.json
    python code/aggregate/run_reward_relative_analysis.py --seeds 0 1 --out /tmp/rr2.json
    python code/aggregate/run_reward_relative_analysis.py --force   # overwrite the cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "lib"))

from analysis_helpers import campaign_passers  # noqa: E402
from reward_relative_analysis import per_seed_row  # noqa: E402

RELOC = REPO / "results" / "multi_seed_reloc"
OUT = RELOC / "aggregate" / "reward_relative_summary.json"
GATE_PERIODS = ["cycle_1_goal", "cycle_2_goal", "goal_conditioned"]
from analysis_helpers import MIN_ENTROPY  # noqa: E402,F401  (single source)


def _mean_sd(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "sd": None}
    return {"mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}


def _paired_p(a, b) -> float | None:
    """Paired Wilcoxon over seeds, keeping only pairs where both are finite."""
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 2:
        return None
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    if np.allclose(x, y):
        return None
    try:
        return float(wilcoxon(x, y).pvalue)
    except Exception:
        return None


def _get(row, readout, key):
    r = row.get(readout)
    return None if r is None else r.get(key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reloc_dir", type=str, default=str(RELOC))
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="restrict to these seeds (default: every seed dir found)")
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing summary JSON")
    ap.add_argument("--min_entropy", type=float, default=MIN_ENTROPY)
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"REFUSING TO OVERWRITE {out_path}\n"
              f"  This cache is committed and consumed by fig5. Re-run with "
              f"--out <scratch path> and diff, or pass --force.")
        return 1

    reloc_dir = Path(args.reloc_dir)
    # sorted by directory name, matching the seed order in the committed cache
    seed_dirs = sorted((d for d in reloc_dir.iterdir()
                        if d.is_dir() and d.name.startswith("reloc_seed")),
                       key=lambda d: d.name)
    if not seed_dirs:
        print(f"ERROR: no reloc_seed* directories under {reloc_dir}")
        return 1

    rows, skipped = [], []
    for d in seed_dirs:
        try:
            seed = int(d.name.replace("reloc_seed", ""))
        except ValueError:
            continue
        if args.seeds is not None and seed not in args.seeds:
            continue
        data_dir = d / "data"
        if seed not in campaign_passers("reloc"):    # the pre-specified cohort
            skipped.append({"seed": seed, "reason": "not in retained cohort"})
            continue
        row = per_seed_row(d, seed)
        if row is None:
            skipped.append({"seed": seed, "reason": "readout failed"})
            continue
        # drop the per-cell detail (used only for example-cell selection)
        for ro in ("behavioural", "controlled"):
            if isinstance(row.get(ro), dict):
                row[ro].pop("cells", None)
        rows.append(row)
        print(f"  seed {seed:3d}  "
              f"ctl RR={_get(row,'controlled','frac_reward_relative'):.4f}  "
              f"ctl PA={_get(row,'controlled','frac_place_anchored'):.4f}")

    if not rows:
        print("ERROR: no seed passed the gate")
        return 1

    ctl_rr = [_get(r, "controlled", "frac_reward_relative") for r in rows]
    ctl_pa = [_get(r, "controlled", "frac_place_anchored") for r in rows]
    ctl_rri = [_get(r, "controlled", "rri_mean") for r in rows]
    beh_rr = [_get(r, "behavioural", "frac_reward_relative") for r in rows]
    beh_pa = [_get(r, "behavioural", "frac_place_anchored") for r in rows]

    result = {
        "summary": {
            "n_seeds": len(rows),
            "controlled": {
                "frac_reward_relative": _mean_sd(ctl_rr),
                "frac_place_anchored": _mean_sd(ctl_pa),
                "rri_mean": _mean_sd(ctl_rri),
                "pa_gt_rr_seeds": int(sum(
                    1 for a, b in zip(ctl_pa, ctl_rr)
                    if a is not None and b is not None and a > b)),
                "pa_vs_rr_paired_p": _paired_p(ctl_pa, ctl_rr),
            },
            "behavioural": {
                "frac_reward_relative": _mean_sd(beh_rr),
                "frac_place_anchored": _mean_sd(beh_pa),
            },
            "readout_paired_p_rr": _paired_p(beh_rr, ctl_rr),
        },
        "per_seed": rows,
    }
    if skipped:
        result["skipped"] = skipped

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    s = result["summary"]
    print(f"\nn_seeds={s['n_seeds']}  "
          f"ctl RR={s['controlled']['frac_reward_relative']['mean']}  "
          f"ctl PA={s['controlled']['frac_place_anchored']['mean']}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
