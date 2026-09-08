#!/usr/bin/env python3
"""
Aggregate New_Code/extinction_analysis.per_seed_row over all entropy-gated
extinction seeds and cache group-level statistics to
Repro/aggregate/extinction_summary.json.

Gate: seed_is_clean(GATE_PERIODS, 5.5) on the frozen-replay triple (same
collapse gate used everywhere else in this project), applied BEFORE the
per-cell Skaggs>=0.15 gate inside extinction_analysis itself.

K1 (over-representation collapse): per-seed delta_learn/delta_ext/delta_net;
Wilcoxon vs 0 for delta_learn and delta_ext; pooled recovery fraction
-sum(delta_ext)/sum(delta_learn) with a seed-level bootstrap CI (ratios of
per-seed means are unstable -- never average per-seed ratios).

K2 (reliability): per-period (pred/postLearn/postExt) x stratum (all/near/far)
group means; paired Wilcoxon postLearn vs postExt per stratum (pairing: seed).

K3 (dopamine proxy): group-mean head-output-vs-distance profile (postLearn vs
postExt, same head, same trajectory); per-seed near/far contrast ratio
(profile at nearest bin / profile at farthest valid bin); paired Wilcoxon
contrast postLearn vs postExt (pairing: seed).

Usage:
    python New_Code/run_extinction_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "New_Code"))

from analysis_helpers import seed_is_clean  # noqa: E402
from extinction_analysis import PERIODS, per_seed_row  # noqa: E402

EXT = REPO / "Repro" / "multi_seed_ext"
OUT = REPO / "Repro" / "aggregate" / "extinction_summary.json"
GATE_PERIODS = PERIODS  # replay_ext_pred/postLearn/postExt
N_BOOT = 10000


def _wstat(x, alternative="two-sided"):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0):
        return {"n": int(len(x)), "mean": float(np.mean(x)) if len(x) else None,
                "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None, "p": None}
    try:
        p = float(wilcoxon(x, alternative=alternative).pvalue)
    except Exception:
        p = None
    return {"n": int(len(x)), "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)), "p": p}


def _wstat_ratio(x):
    """Like _wstat (two-sided vs 0) but for heavy-tailed positive ratios: adds
    median/IQR as the headline (mean/SD are skewed by 1-2 near-zero-denominator
    seeds; the paired/one-sample Wilcoxon is rank-based and unaffected)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "iqr": None, "p": None}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
            "median": float(med), "iqr": [float(q1), float(q3)]}


def _paired_wstat(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return {"n": int(len(a)), "mean_a": None, "mean_b": None, "p": None}
    d = a - b
    try:
        p = float(wilcoxon(d).pvalue) if not np.allclose(d, 0) else 1.0
    except Exception:
        p = None
    return {"n": int(len(a)), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": float(d.mean()), "p": p}


def bootstrap_recovery_ratio(delta_learn, delta_ext, n_boot=N_BOOT, seed=0):
    dl = np.asarray(delta_learn, dtype=float)
    de = np.asarray(delta_ext, dtype=float)
    ok = np.isfinite(dl) & np.isfinite(de)
    dl, de = dl[ok], de[ok]
    n = len(dl)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        denom = dl[idx].sum()
        if abs(denom) < 1e-12:
            continue
        vals.append(-de[idx].sum() / denom)
    vals = np.array(vals)
    return {
        "point": float(-de.sum() / dl.sum()),
        "ci_lo": float(np.percentile(vals, 2.5)) if len(vals) else None,
        "ci_hi": float(np.percentile(vals, 97.5)) if len(vals) else None,
        "n_boot_valid": int(len(vals)),
    }


def main():
    all_seeds = sorted(int(p.name.replace("ext_seed", ""))
                       for p in EXT.glob("ext_seed*") if p.is_dir())
    gated = [s for s in all_seeds
             if seed_is_clean(str(EXT / f"ext_seed{s}" / "data"), GATE_PERIODS, 5.5)]
    print(f"{len(gated)}/{len(all_seeds)} seeds pass entropy gate 5.5: {gated}")

    rows = []
    for s in gated:
        row = per_seed_row(EXT / f"ext_seed{s}", s)
        if row is None:
            print(f"  seed {s}: dropped (Skaggs<10 gated cells or load failure)")
            continue
        rows.append(row)
    print(f"{len(rows)}/{len(gated)} seeds usable after Skaggs>=0.15 triple gate")

    # ---------------- K1 ----------------
    k1_rows = [r["k1"] for r in rows]
    delta_learn = [k["delta_learn"] for k in k1_rows]
    delta_ext = [k["delta_ext"] for k in k1_rows]
    delta_net = [k["delta_net"] for k in k1_rows]
    near_frac = {lbl: [k["near_frac"][i] for k in k1_rows]
                 for i, lbl in enumerate(["pred", "postLearn", "postExt"])}
    mean_dist = {lbl: [k["mean_dist"][i] for k in k1_rows]
                 for i, lbl in enumerate(["pred", "postLearn", "postExt"])}
    k1_summary = {
        "n_seeds": len(k1_rows),
        "delta_learn": _wstat(delta_learn, alternative="less"),      # expect < 0
        "delta_ext": _wstat(delta_ext, alternative="greater"),        # expect > 0
        "delta_net": _wstat(delta_net),
        "recovery_pooled": bootstrap_recovery_ratio(delta_learn, delta_ext),
        "near_frac_by_period": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                                for k, v in near_frac.items()},
        "mean_dist_by_period": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                                for k, v in mean_dist.items()},
    }

    # ---------------- K2 ----------------
    k2_rows = [r["k2"] for r in rows]
    k2_summary = {"n_seeds": len(k2_rows)}
    for stratum in ("all", "near", "far"):
        vals = {p: [k2r[p][stratum] for k2r in k2_rows] for p in ["pred", "postLearn", "postExt"]}
        k2_summary[stratum] = {
            p: {"mean": float(np.nanmean(vals[p])) if any(v is not None for v in vals[p]) else None,
                "sd": float(np.nanstd(np.array(vals[p], dtype=float), ddof=1))
                if sum(v is not None for v in vals[p]) > 1 else None}
            for p in ["pred", "postLearn", "postExt"]
        }
        k2_summary[stratum]["postLearn_vs_postExt"] = _paired_wstat(vals["postLearn"], vals["postExt"])

    # ---------------- K3 ----------------
    k3_rows = [r["k3"] for r in rows if r["k3"] is not None]
    k3_summary = {"n_seeds": len(k3_rows)}
    if k3_rows:
        centres = k3_rows[0]["dist_bin_centres"]
        nb = len(centres)

        def mean_profile(label):
            mat = np.full((len(k3_rows), nb), np.nan)
            for i, k3r in enumerate(k3_rows):
                prof = k3r[label]
                for b in range(nb):
                    if prof[b] is not None:
                        mat[i, b] = prof[b]
            return mat

        mat_learn = mean_profile("postLearn")
        mat_ext = mean_profile("postExt")

        def contrast_ratio(mat):
            out = []
            for row_ in mat:
                valid = np.where(np.isfinite(row_))[0]
                if len(valid) < 2:
                    out.append(np.nan)
                    continue
                near_b, far_b = valid[0], valid[-1]
                if row_[far_b] <= 1e-9:
                    out.append(np.nan)
                    continue
                out.append(float(row_[near_b] / row_[far_b]))
            return out

        contrast_learn = contrast_ratio(mat_learn)
        contrast_ext = contrast_ratio(mat_ext)
        seeds_used = [r["seed"] for r in rows if r["k3"] is not None]
        # Ratios are heavy-tailed (a couple of seeds have a near-zero far-bin
        # denominator); report median/IQR as the headline. Also flag any seed
        # where the ratio INCREASES under extinction (wrong-direction anomaly)
        # rather than silently averaging over it.
        reversals = [s for s, cl, ce in zip(seeds_used, contrast_learn, contrast_ext)
                     if np.isfinite(cl) and np.isfinite(ce) and ce > cl]

        k3_summary.update({
            "dist_bin_centres": centres,
            "target_profile": k3_rows[0]["target"],
            "mean_profile": {
                "postLearn": np.nanmean(mat_learn, axis=0).tolist(),
                "postExt": np.nanmean(mat_ext, axis=0).tolist(),
                "postLearn_sd": np.nanstd(mat_learn, axis=0, ddof=1).tolist(),
                "postExt_sd": np.nanstd(mat_ext, axis=0, ddof=1).tolist(),
            },
            "contrast_near_over_far": {
                "postLearn": _wstat_ratio(contrast_learn),
                "postExt": _wstat_ratio(contrast_ext),
                "postLearn_vs_postExt": _paired_wstat(contrast_learn, contrast_ext),
                "seeds_used": seeds_used,
                "per_seed_postLearn": contrast_learn,
                "per_seed_postExt": contrast_ext,
                "wrong_direction_seeds": reversals,
                "note": "mean/SD are skewed by 1-2 seeds with a near-zero far-bin "
                        "denominator; median/IQR is the headline statistic. Paired "
                        "Wilcoxon (rank-based) is unaffected by this skew.",
            },
        })

    out = {
        "params": {"skaggs_thr": 0.15, "near_radius": 0.15, "entropy_gate": 5.5,
                   "block_size_k2": 1000, "n_boot_k1": N_BOOT,
                   "convention": "negative delta = toward reward A; "
                                 "delta_learn = postLearn-pred; delta_ext = postExt-postLearn"},
        "all_seeds": all_seeds,
        "gated_seeds": gated,
        "usable_seeds": [r["seed"] for r in rows],
        "per_seed": rows,
        "k1": k1_summary,
        "k2": k2_summary,
        "k3": k3_summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")

    print("\n===== K1: over-representation collapse =====")
    print(f"  delta_learn (pred->postLearn): {k1_summary['delta_learn']['mean']:+.4f} "
          f"+- {k1_summary['delta_learn']['sd']:.4f}  (p={k1_summary['delta_learn']['p']}, "
          f"n={k1_summary['delta_learn']['n']}) [expect <0]")
    print(f"  delta_ext   (postLearn->postExt): {k1_summary['delta_ext']['mean']:+.4f} "
          f"+- {k1_summary['delta_ext']['sd']:.4f}  (p={k1_summary['delta_ext']['p']}, "
          f"n={k1_summary['delta_ext']['n']}) [expect >0]")
    rec = k1_summary["recovery_pooled"]
    print(f"  pooled recovery fraction: {rec['point']:.3f}  95% CI [{rec['ci_lo']:.3f}, {rec['ci_hi']:.3f}]")
    for lbl in ("pred", "postLearn", "postExt"):
        nf = k1_summary["near_frac_by_period"][lbl]
        md = k1_summary["mean_dist_by_period"][lbl]
        print(f"  {lbl:>10}: near_frac={nf['mean']*100:.1f}%+-{nf['sd']*100:.1f}%  "
              f"mean_dist={md['mean']:.4f}+-{md['sd']:.4f}")

    print("\n===== K2: reliability =====")
    for stratum in ("all", "near", "far"):
        s = k2_summary[stratum]
        pl, pe = s["postLearn"], s["postExt"]
        pv = s["postLearn_vs_postExt"]
        if pl["mean"] is None or pe["mean"] is None:
            print(f"  {stratum:>4}: insufficient data")
            continue
        print(f"  {stratum:>4}: postLearn={pl['mean']:.3f}+-{(pl['sd'] or 0):.3f}  "
              f"postExt={pe['mean']:.3f}+-{(pe['sd'] or 0):.3f}  "
              f"paired p={pv['p']} (n={pv['n']})")

    print("\n===== K3: dopamine proxy (near/far contrast) =====")
    if k3_rows:
        c = k3_summary["contrast_near_over_far"]
        print(f"  postLearn contrast: median={c['postLearn']['median']:.2f}x "
              f"IQR=[{c['postLearn']['iqr'][0]:.2f}, {c['postLearn']['iqr'][1]:.2f}]  "
              f"(mean={c['postLearn']['mean']:.1f}x, skewed; n={c['postLearn']['n']})")
        print(f"  postExt   contrast: median={c['postExt']['median']:.2f}x "
              f"IQR=[{c['postExt']['iqr'][0]:.2f}, {c['postExt']['iqr'][1]:.2f}]  "
              f"(mean={c['postExt']['mean']:.1f}x, skewed; n={c['postExt']['n']})")
        print(f"  paired postLearn vs postExt: p={c['postLearn_vs_postExt']['p']} "
              f"(mean diff {c['postLearn_vs_postExt']['mean_diff']:+.2f})")
        if c["wrong_direction_seeds"]:
            print(f"  NOTE wrong-direction seeds (contrast INCREASED under extinction): "
                  f"{c['wrong_direction_seeds']}")
    else:
        print("  no seeds with usable checkpoints")


if __name__ == "__main__":
    main()
