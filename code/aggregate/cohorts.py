#!/usr/bin/env python3
"""
The analysed cohort of each model family, written to results/aggregate/cohorts.json.

A network is retained when its minimum field-peak entropy over the family's
gate periods is >= analysis_helpers.MIN_ENTROPY. Entropy is computed from the
same rate maps as every analysis (occupancy-weighted spatial information,
goal-directed periods without reset steps). The entropy gate always uses the
shared 0.3 Skaggs threshold; relocation and omission effect estimates apply
their 0.15 cell gates after network selection. Every launched network is evaluated
(results/aggregate/until_clean_<family>.json lists the launches).

Usage:
    python code/aggregate/cohorts.py [--workers 96]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "lib"))

from analysis_helpers import MIN_ENTROPY, peak_entropy  # noqa: E402

RES = REPO / "results"
OUT = RES / "aggregate" / "cohorts.json"
PLASTIC = ["cycle_1", "cycle_1_goal", "replay_c1_pred", "replay_c1_postrl"]
FAMILIES = {
    # all three arms of a matched triple must pass
    "paired": [("multi_seed/nohead_seed", PLASTIC), ("multi_seed_head/head_seed", PLASTIC),
               ("multi_seed_rigid/rigid_seed", ["replay_c1_pred", "replay_c1_postrl"])],
    "reloc": [("multi_seed_reloc/reloc_seed",
               ["replay_reloc_pred", "replay_reloc_postA", "replay_reloc_postB",
                "cycle_1_goal", "cycle_2_goal", "goal_conditioned"])],
    "extinction": [("multi_seed_ext/ext_seed",
                    ["replay_ext_pred", "replay_ext_postLearn", "replay_ext_postExt"])],
}


def min_entropy(job):
    family, seed = job
    ents = [peak_entropy(str(RES / f"{prefix}{seed}" / "data"), p, cached=True)
            for prefix, periods in FAMILIES[family] for p in periods]
    return family, seed, (None if any(e is None for e in ents) else float(min(ents)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()
    jobs = []
    for fam in FAMILIES:
        launched = json.loads((RES / "aggregate" / f"until_clean_{fam}.json").read_text())
        jobs += [(fam, int(e["seed"])) for e in launched["entries"]]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(min_entropy, jobs))
    out = {"threshold_bits": MIN_ENTROPY}
    for fam in FAMILIES:
        rows = {s: e for f, s, e in res if f == fam}
        keep = sorted(s for s, e in rows.items() if e is not None and e >= MIN_ENTROPY)
        out[fam] = {"n_launched": len(rows), "n": len(keep), "seeds": keep,
                    "min_entropy": {str(s): e for s, e in rows.items()}}
        print(f"{fam}: {len(keep)} of {len(rows)} launched networks pass {MIN_ENTROPY} bits")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
