#!/usr/bin/env python3
"""
Verify that the rigid-weight control seeds are matched to their plastic
counterparts on everything except the recurrent learning rate.

Three independent checks per seed:

1. **Config symmetric difference.** rigid vs no-head must differ ONLY in
   {output_dir, rnn_rl_learning_rate}; rigid vs head additionally in the two
   reward-head keys. Anything else means the arg lists have drifted apart.

2. **Bit-identical phase-1 final loss.** Phase 1 is re-seeded immediately before
   data generation, so a given seed's pre-learning network is deterministic and
   independent of the Phase-2 condition. If the rigid, no-head and head runs for
   a seed share a bit-identical `phase1_final_loss`, they provably started from
   the same map. This is the strong check: it cannot be satisfied by accident.

3. **Rigidity signature.** The frozen-replay shift should be ~0. Crucially we
   also report the mean per-cell DISPLACEMENT MAGNITUDE, which distinguishes
   "fields moved but not net-toward-reward" from "fields did not move at all".
   Only the latter supports the claim that plasticity is the gate.

Usage:
    python New_Code/verify_rigid_match.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "New_Code"))

from analysis_helpers import load_period_centers, seed_is_clean  # noqa: E402

RIGID = REPO / "Repro" / "multi_seed_rigid"
NOHEAD = REPO / "Repro" / "multi_seed"
HEAD = REPO / "Repro" / "multi_seed_head"
REWARD = np.array([0.3, 0.3])
THR = 0.3
PRE, POST = "replay_c1_pred", "replay_c1_postrl"

EXPECTED_VS_NOHEAD = {"output_dir", "rnn_rl_learning_rate"}
EXPECTED_VS_HEAD = EXPECTED_VS_NOHEAD | {"reward_pred_head", "reward_pred_weight"}


def cfg(p: Path):
    try:
        return json.load(open(p / "config.json"))
    except Exception:
        return None


def diff_keys(a, b):
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def rigidity(run_dir: Path):
    d = run_dir / "data"
    try:
        sa, _, pa = load_period_centers(str(d), PRE)
        sb, _, pb = load_period_centers(str(d), POST)
    except Exception:
        return None
    g = (sa >= THR) & (sb >= THR)
    if g.sum() < 10:
        return None
    dd = np.linalg.norm(pb[g] - REWARD, axis=1) - np.linalg.norm(pa[g] - REWARD, axis=1)
    disp = np.linalg.norm(pb[g] - pa[g], axis=1)
    return {"n": int(g.sum()), "mean_dd": float(dd.mean()),
            "mean_disp": float(disp.mean()), "frac_toward": float((dd < 0).mean())}


def main():
    seeds = sorted(int(p.name.replace("rigid_seed", ""))
                   for p in RIGID.glob("rigid_seed*") if p.is_dir())
    # Seed 42 began as the reproducibility smoke test, but it is a fully valid
    # matched triple (gate-clean, with plastic partners in both conditions), so
    # it is INCLUDED here — excluding valid data to obtain a round n would be
    # worse than reporting it. This keeps the verifier's seed set identical to
    # the one code_analysis/figure_sr/ discovers, so their counts agree.
    if not seeds:
        print("No rigid seeds found yet.")
        return

    ok_cfg = ok_p1 = 0
    rows = []
    print(f"{'seed':>4} {'cfg':>5} {'phase1':>7} {'gate':>6} {'n':>5} {'mean_dd':>9} {'disp':>7} {'toward':>7}")
    for s in seeds:
        r, nh, hd = RIGID / f"rigid_seed{s}", NOHEAD / f"nohead_seed{s}", HEAD / f"head_seed{s}"
        cr, cn, ch = cfg(r), cfg(nh), cfg(hd)
        if cr is None:
            continue

        cfg_ok = True
        if cn is not None:
            cfg_ok &= diff_keys(cr, cn) == EXPECTED_VS_NOHEAD
        if ch is not None:
            cfg_ok &= diff_keys(cr, ch) == EXPECTED_VS_HEAD
        cfg_ok &= float(cr.get("rnn_rl_learning_rate", 0)) == 2e-06

        def p1(p):
            try:
                return json.load(open(p / "metrics.json")).get("phase1_final_loss")
            except Exception:
                return None
        lr_, ln_, lh_ = p1(r), p1(nh), p1(hd)
        p1_ok = lr_ is not None and (ln_ is None or lr_ == ln_) and (lh_ is None or lr_ == lh_)

        rg = rigidity(r)
        ok_cfg += cfg_ok
        ok_p1 += p1_ok
        # The paper's rigid statistic must be filtered by the SAME pre-specified
        # entropy gate as every other condition, so apply it here rather than
        # reporting an unfiltered number that would not match the manuscript.
        gated = seed_is_clean(str(r / "data"), [PRE, POST], 5.5)
        if rg:
            if gated:
                rows.append(rg)
            print(f"{s:>4} {'OK' if cfg_ok else 'FAIL':>5} {'OK' if p1_ok else 'FAIL':>7} "
                  f"{'pass' if gated else 'GATED':>6} "
                  f"{rg['n']:>5} {rg['mean_dd']:>+9.4f} {rg['mean_disp']:>7.4f} {rg['frac_toward']*100:>6.1f}%")
        else:
            print(f"{s:>4} {'OK' if cfg_ok else 'FAIL':>5} {'OK' if p1_ok else 'FAIL':>7} "
                  f"{'-':>6}   (no replay data)")

    print(f"\nconfig match: {ok_cfg}/{len(seeds)}   phase1 bit-identical: {ok_p1}/{len(seeds)}")
    print(f"entropy-gate survivors used below: {len(rows)}")
    if rows:
        dd = np.array([r["mean_dd"] for r in rows])
        disp = np.array([r["mean_disp"] for r in rows])
        from scipy.stats import wilcoxon
        try:
            p = wilcoxon(dd).pvalue
        except Exception:
            p = float("nan")
        print(f"rigid frozen-replay shift: {dd.mean():+.4f} +- {dd.std(ddof=1):.4f}  "
              f"(Wilcoxon vs 0: p={p:.3g}, n={len(dd)})")
        print(f"rigid per-cell displacement magnitude: {disp.mean():.4f} +- {disp.std(ddof=1):.4f}")
        print("\nInterpretation: a near-zero mean_dd WITH a near-zero displacement magnitude "
              "supports 'the weights did not move'. A near-zero mean_dd with LARGE displacement "
              "would instead mean the fields moved randomly, which is a different claim.")


if __name__ == "__main__":
    main()
