#!/usr/bin/env python3
"""
Across-network paired and relocation statistics quoted in the main text, for
field peak and centre of mass, written to
results/aggregate/headline_stats.json. Reward-omission statistics have
different three-period survivor gates and are generated only by
run_extinction_analysis.py in extinction_summary.json.

One row per launched seed is computed once (centres cached on disk by
analysis_helpers.load_period_centers_cached); cohorts are then formed two ways:

  cohort  -- the retained networks of each family (campaign_passers), which
             is what the text and figures report;
  sweep   -- every seed whose minimum field-peak entropy over the family's
             gate periods is >= T, for T in THRESHOLDS (Methods robustness
             statement and Fig. S2a).

Each summary carries mean, s.d., s.e.m. (= s.d./sqrt(n)), n, the number of
networks with a negative mean (toward reward) and Wilcoxon p-values.
Convention: delta = d_post - d_pre, negative = toward reward.

Usage:
    python code/aggregate/headline_stats.py [--workers 64]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "lib"))

from analysis_helpers import (  # noqa: E402
    _mean_delta_to, campaign_passers, peak_entropy, MIN_ENTROPY,
)

RES = REPO / "results"
OUT = RES / "aggregate" / "headline_stats.json"
THRESHOLDS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
CENTERS = ("peak", "com")
REWARD_A = (0.3, 0.3)
REWARD_B = (0.7, 0.7)

# Behavioural contrast: pre-learning random walk -> post-learning goal-directed
# probe (reset transients dropped). Parameters and trajectory both differ, as in
# a recording before and after learning. Frozen replay changes only the weights.
BEH = ("replay_c1_pred", "cycle_1_goal")
FRZ = ("replay_c1_pred", "replay_c1_postrl")
PAIRED_PLASTIC_PERIODS = ["cycle_1", "cycle_1_goal", *FRZ]   # the campaign gate
PAIRED_RIGID_PERIODS = list(FRZ)
RELOC_TRIPLE = ("replay_reloc_pred", "replay_reloc_postA", "replay_reloc_postB")
RELOC_PERIODS = [*RELOC_TRIPLE, "cycle_1_goal", "cycle_2_goal", "goal_conditioned"]
def _launched(family: str) -> list[int]:
    data = json.loads((RES / "aggregate" / f"until_clean_{family}.json").read_text())
    return sorted(int(e["seed"]) for e in data["entries"])


def _min_entropy(data_dir: Path, periods) -> float | None:
    ents = [peak_entropy(str(data_dir), p, cached=True) for p in periods]
    return None if any(e is None for e in ents) else float(min(ents))


def _shift(data_dir, pre, post, reward, thr, center, trim_post=False):
    r = _mean_delta_to(str(data_dir), pre, post, reward, thr, center=center, cached=True,
                       trim_post=trim_post)
    if r is None:
        return None
    return {"mean": r["mean"], "frac": r["frac_toward"], "n": r["n"], "abs_disp": r["abs_disp"]}


def paired_row(seed: int) -> dict:
    dirs = {"nh": RES / "multi_seed" / f"nohead_seed{seed}" / "data",
            "hd": RES / "multi_seed_head" / f"head_seed{seed}" / "data",
            "rg": RES / "multi_seed_rigid" / f"rigid_seed{seed}" / "data"}
    row: dict = {"seed": seed}
    ents = [_min_entropy(dirs["nh"], PAIRED_PLASTIC_PERIODS),
            _min_entropy(dirs["hd"], PAIRED_PLASTIC_PERIODS),
            _min_entropy(dirs["rg"], PAIRED_RIGID_PERIODS)]
    row["entropy"] = None if any(e is None for e in ents) else min(ents)
    for c in CENTERS:
        for arm in ("nh", "hd"):
            row[f"{arm}_beh_{c}"] = _shift(dirs[arm], *BEH, REWARD_A, 0.3, c, trim_post=True)
            row[f"{arm}_frz_{c}"] = _shift(dirs[arm], *FRZ, REWARD_A, 0.3, c)
        row[f"rg_frz_{c}"] = _shift(dirs["rg"], *FRZ, REWARD_A, 0.3, c)
    return row


def reloc_row(seed: int) -> dict:
    d = RES / "multi_seed_reloc" / f"reloc_seed{seed}" / "data"
    pre, pa, pb = RELOC_TRIPLE
    row: dict = {"seed": seed, "entropy": _min_entropy(d, RELOC_PERIODS)}
    for c in CENTERS:
        row[f"A_{c}"] = _shift(d, pre, pa, REWARD_A, 0.15, c)
        row[f"B_{c}"] = _shift(d, pa, pb, REWARD_B, 0.15, c)
    return row


def _p(x, alternative="two-sided"):
    x = np.asarray(x, float)
    if len(x) < 2 or np.allclose(x, 0):
        return None
    return float(wilcoxon(x, alternative=alternative).pvalue)


def summarize(values, fracs=None) -> dict:
    v = np.asarray(values, float)
    n = len(v)
    sd = float(np.std(v, ddof=1)) if n > 1 else float("nan")
    out = {"n": n, "mean": float(v.mean()), "median": float(np.median(v)),
           "sd": sd, "sem": sd / np.sqrt(n), "n_negative": int((v < 0).sum()),
           "p_two_sided": _p(v), "p_less": _p(v, "less"), "p_greater": _p(v, "greater")}
    if fracs is not None:
        f = 100 * np.asarray(fracs, float)
        out["pct_toward_mean"] = float(f.mean())
        out["pct_toward_sd"] = float(np.std(f, ddof=1))
        out["pct_toward_sem"] = out["pct_toward_sd"] / np.sqrt(n)
    return out


def _complete(rows, keys):
    return [r for r in rows if all(r.get(k) is not None for k in keys)]


def paired_summary(rows) -> dict:
    out: dict = {"n": len(rows), "seeds": [r["seed"] for r in rows]}
    for c in CENTERS:
        keys = [f"{a}_{c}" for a in ("nh_beh", "hd_beh", "nh_frz", "hd_frz", "rg_frz")]
        rr = _complete(rows, keys)
        s: dict = {}
        for k in keys:
            s[k.rsplit("_", 1)[0]] = summarize([r[k]["mean"] for r in rr], [r[k]["frac"] for r in rr])
            s[k.rsplit("_", 1)[0]]["abs_disp"] = summarize([r[k]["abs_disp"] for r in rr])
        nh_f = np.array([r[f"nh_frz_{c}"]["mean"] for r in rr])
        hd_f = np.array([r[f"hd_frz_{c}"]["mean"] for r in rr])
        nh_b = np.array([r[f"nh_beh_{c}"]["mean"] for r in rr])
        hd_b = np.array([r[f"hd_beh_{c}"]["mean"] for r in rr])
        s["frz_hd_minus_nh"] = summarize(hd_f - nh_f)
        s["frz_hd_minus_nh"]["pct_pairs_hd_larger"] = float(100 * (hd_f < nh_f).mean())
        s["beh_hd_minus_nh"] = summarize(hd_b - nh_b)
        s["interaction"] = summarize((hd_b - hd_f) - (nh_b - nh_f))
        s["nh_beh_minus_frz"] = summarize(nh_b - nh_f)
        s["hd_beh_minus_frz"] = summarize(hd_b - hd_f)
        # how much of the behavioural shift fixed-trajectory replay retains
        for arm, b_, f_ in (("nh", nh_b, nh_f), ("hd", hd_b, hd_f)):
            s[f"{arm}_retained"] = {
                "ratio_of_means": float(f_.mean() / b_.mean()),
                "median_per_network": float(np.median(f_ / b_)),
                "n_replay_smaller": int((np.abs(f_) < np.abs(b_)).sum()),
                "n": int(len(b_)),
            }
        s["sd_over_abs_mean"] = {"nh_frz": float(np.std(nh_f, ddof=1) / abs(nh_f.mean())),
                                 "hd_frz": float(np.std(hd_f, ddof=1) / abs(hd_f.mean()))}
        out[c] = s
    return out


def reloc_summary(rows) -> dict:
    out: dict = {"n": len(rows), "seeds": [r["seed"] for r in rows]}
    for c in CENTERS:
        rr = _complete(rows, [f"A_{c}", f"B_{c}"])
        a = np.array([r[f"A_{c}"]["mean"] for r in rr])
        b = np.array([r[f"B_{c}"]["mean"] for r in rr])
        out[c] = {"toward_A": summarize(a, [r[f"A_{c}"]["frac"] for r in rr]),
                  "toward_B": summarize(b, [r[f"B_{c}"]["frac"] for r in rr]),
                  # Wilcoxon on dA - dB (negative = larger shift toward A)
                  "A_minus_B": {**summarize(a - b), "n_A_larger": int((a < b).sum())}}
    return out


FAMILIES = {
    "paired": (paired_row, paired_summary),
    "reloc": (reloc_row, reloc_summary),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    out: dict = {"convention": "delta = d_post - d_pre, negative = toward reward; "
                               "sem = sd/sqrt(n) across networks; Wilcoxon signed-rank on per-network means",
                 "thresholds": THRESHOLDS, "min_entropy": MIN_ENTROPY}
    with ProcessPoolExecutor(args.workers) as pool:
        for fam, (row_fn, summ_fn) in FAMILIES.items():
            seeds = _launched(fam)
            rows = list(pool.map(row_fn, seeds))
            passers = campaign_passers(fam)
            cohort = [r for r in rows if r["seed"] in passers]
            sweep = {}
            for t in THRESHOLDS:
                kept = [r for r in rows if r["entropy"] is not None and r["entropy"] >= t]
                sweep[str(t)] = summ_fn(kept)
            out[fam] = {"cohort": summ_fn(cohort), "sweep": sweep,
                        "rows": [{k: v for k, v in r.items()} for r in rows]}
            print(f"[{fam}] launched {len(seeds)}, cohort {len(cohort)}, "
                  f"gate-{MIN_ENTROPY} {sweep[str(MIN_ENTROPY)]['n']}")
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
