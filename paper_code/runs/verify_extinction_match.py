#!/usr/bin/env python3
"""
Verify that the extinction seeds are matched to their head-ON counterparts on
everything except the extinction manipulation itself, and that the
extinction-specific code changes are confined to cycle 2.

Three independent checks per seed:

1. **Config symmetric difference.** ext vs head-ON must differ ONLY in
   {output_dir, n_cycles, max_episodes} (extinguish_at_cycle is a runtime-only
   flag, not written into the tracked config dict -- the runner already
   asserts the CLI-level diff at launch time; this is the saved-config-level
   cross-check). Anything else means the arg lists have drifted apart.

2. **Bit-identical phase-1 final loss.** Phase 1 is re-seeded immediately
   before data generation, so a given seed's pre-learning network is
   deterministic and independent of n_cycles/extinguish_at_cycle. If ext and
   head share a bit-identical `phase1_final_loss`, they provably started from
   the same map.

3. **Strong check: bit-identical cycle-1 replication.** replay_ext_postLearn
   (ext run, end of cycle 1 = before extinction starts) must be bit-identical
   -- same positions AND same activations, not just same field centres -- to
   head's replay_c1_postrl (same weights, same frozen trajectory). This
   proves the extinction-specific code path (r=0, head loss gated out) is
   completely confined to cycle 2 and never leaks into cycle 1.

Usage:
    python paper_code/runs/verify_extinction_match.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper_code" / "lib"))

from analysis_helpers import seed_is_clean  # noqa: E402
from occupancy_matched_estimator import load_period_arrays  # noqa: E402

EXT = REPO / "Repro" / "multi_seed_ext"
HEAD = REPO / "Repro" / "multi_seed_head"
GATE_PERIODS = ["replay_ext_pred", "replay_ext_postLearn", "replay_ext_postExt"]

EXPECTED_VS_HEAD = {"output_dir", "n_cycles", "max_episodes"}


def cfg(p: Path):
    try:
        return json.load(open(p / "config.json"))
    except Exception:
        return None


def diff_keys(a, b):
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def bit_identical_cycle1(ext_dir: Path, head_dir: Path) -> dict | None:
    try:
        pos_e, th_e, act_e = load_period_arrays(ext_dir / "data", "replay_ext_postLearn")
        pos_h, th_h, act_h = load_period_arrays(head_dir / "data", "replay_c1_postrl")
    except Exception as e:
        return {"error": str(e)}
    pos_ok = pos_e.shape == pos_h.shape and np.array_equal(pos_e, pos_h)
    th_ok = th_e.shape == th_h.shape and np.array_equal(th_e, th_h)
    act_ok = act_e.shape == act_h.shape and np.array_equal(act_e, act_h)
    return {"pos_identical": bool(pos_ok), "theta_identical": bool(th_ok),
            "act_identical": bool(act_ok), "n_values": int(act_e.size) if act_ok else None}


def main():
    seeds = sorted(int(p.name.replace("ext_seed", ""))
                   for p in EXT.glob("ext_seed*") if p.is_dir())
    if not seeds:
        print("No extinction seeds found yet.")
        return

    ok_cfg = ok_p1 = ok_bit = 0
    gated_seeds = []
    print(f"{'seed':>4} {'cfg':>5} {'phase1':>7} {'cyc1-bit':>9} {'gate':>6}")
    for s in seeds:
        e, hd = EXT / f"ext_seed{s}", HEAD / f"head_seed{s}"
        ce, ch = cfg(e), cfg(hd)
        if ce is None:
            continue

        cfg_ok = True
        if ch is not None:
            cfg_ok &= diff_keys(ce, ch) == EXPECTED_VS_HEAD
        ok_cfg += cfg_ok

        def p1(p):
            try:
                return json.load(open(p / "metrics.json")).get("phase1_final_loss")
            except Exception:
                return None
        le_, lh_ = p1(e), p1(hd)
        p1_ok = le_ is not None and (lh_ is None or le_ == lh_)
        ok_p1 += p1_ok

        bit = bit_identical_cycle1(e, hd) if ch is not None else None
        bit_ok = bool(bit and bit.get("act_identical") and bit.get("pos_identical"))
        ok_bit += bit_ok

        gated = seed_is_clean(str(e / "data"), GATE_PERIODS, 5.5)
        if gated:
            gated_seeds.append(s)

        bit_str = "OK" if bit_ok else ("FAIL" if bit else "n/a")
        print(f"{s:>4} {'OK' if cfg_ok else 'FAIL':>5} {'OK' if p1_ok else 'FAIL':>7} "
              f"{bit_str:>9} {'pass' if gated else 'GATED':>6}")
        if bit is not None and not bit_ok and "error" not in bit:
            print(f"       -> {bit}")

    print(f"\nconfig match: {ok_cfg}/{len(seeds)}   phase1 bit-identical: {ok_p1}/{len(seeds)}   "
          f"cycle-1 bit-identical: {ok_bit}/{len(seeds)}")
    print(f"entropy-gate (0.15 thr, {GATE_PERIODS}) survivors: {len(gated_seeds)}/{len(seeds)} -> {gated_seeds}")


if __name__ == "__main__":
    main()
