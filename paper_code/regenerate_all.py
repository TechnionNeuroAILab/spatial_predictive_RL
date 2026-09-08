#!/usr/bin/env python3
"""Regenerate every figure in paper/main_v4_reframed.tex from existing run data.

This encodes the canonical invocation of each generator -- in particular the
explicit `--run_dir` for fig2/fig3, whose `utils.py` DEFAULT_RUN points at a
different run from the shipped exemplar (see CLEANUP_TODO item 4). Before this
file existed those arguments lived only in README prose.

It does NOT train anything: every figure is built from the run data already in
Repro/ and New_Results_v3/. Rebuilding the JSON caches is a separate, much more
expensive step (see --with-caches).

Figures are independent processes, so they run in parallel (default: 6 at a
time -- several load 40-50 seeds of pickled activity and peak at a few GB each).

Usage:
    python paper_code/regenerate_all.py --list
    python paper_code/regenerate_all.py                     # all figures
    python paper_code/regenerate_all.py --only fig2 fig3
    python paper_code/regenerate_all.py --jobs 10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PC = Path(__file__).resolve().parent
REPO = PC.parent
FIG = PC / "figures"
LOGS = PC / "figures" / "_logs"

# name -> (script, extra args, produced pdf relative to figures/)
# The exemplar run dirs are the ones the shipped figures were made from
# (provenance: Repro/*/aggregate/fig*_multi_seed_summary.json + the
# select_best_seed_fig*.py rankings).
JOBS: dict[str, tuple[Path, list[str], str]] = {
    # fig1's exemplar run is nohead_seed0, NOT its utils.py DEFAULT_RUN
    # (nohead_seed42). Established by re-deriving the panel-d neuron selection
    # (seed 42, 8 cells) across all 24 candidate run dirs: only nohead_seed0
    # reproduces the shipped IDs [509, 352, 439, 324, 222, 227, 45, 391];
    # nohead_seed42 gives [510, 364, 442, 334, 227, 229, 47, 400].
    "fig1": (FIG/"fig1"/"generate_figure1.py",
             ["--run_dir", str(REPO/"Repro"/"multi_seed"/"nohead_seed0")],
             "fig1/figures/figure1_raw.pdf"),
    "fig2": (FIG/"fig2"/"generate_figure2.py",
             ["--run_dir", str(REPO/"Repro"/"multi_seed"/"nohead_seed16")],
             "fig2/figures/figure2_raw.pdf"),
    "fig3": (FIG/"fig3"/"generate_figure3.py",
             ["--run_dir", str(REPO/"Repro"/"multi_seed_head"/"head_seed28")],
             "fig3/figures/figure3_raw.pdf"),
    "fig4": (FIG/"fig4"/"generate_figure4.py",
             ["--run_dir", str(REPO/"Repro"/"multi_seed_reloc"/"reloc_seed9")],
             "fig4/figures/figure4_raw.pdf"),
    "fig5": (FIG/"fig5"/"generate_figure5_data.py", [],
             "fig5/figures/figure5_data_raw.pdf"),
    "figS_hypotheses": (FIG/"fig5"/"generate_figure5.py", [],
                        "fig5/figures/figure5_raw.pdf"),
    "figS_pipeline": (FIG/"fig1"/"generate_figure1_supp.py", [],
                      "fig1/figures_supp/figure1_supp_raw.pdf"),
    "figS_frozenreplay": (FIG/"fig2"/"generate_figure2_supp.py", [],
                          "fig2/figures_supp/figure2_supp_raw.pdf"),
    "figS_head": (FIG/"fig3"/"generate_figure3_supp.py", [],
                  "fig3/figures_supp/figure3_supp_raw.pdf"),
    "figS_relocation": (FIG/"fig4"/"generate_figure4_supp.py", [],
                        "fig4/figures_supp/figure4_supp_raw.pdf"),
    "figS_dissociation": (FIG/"fig5"/"generate_figure5_supp.py", [],
                          "fig5/figures_supp/figure5_supp_raw.pdf"),
    "figS_sr": (FIG/"figS_sr"/"generate_figure_sr_supp.py", [],
                "figS_sr/figures_supp/figure_sr_supp_raw.pdf"),
    "figS_rigid": (FIG/"figS_sr"/"generate_figure_rigid_supp.py", [],
                   "figS_sr/figures_supp/figure_rigid_supp_raw.pdf"),
    "figS_extinction": (FIG/"figS_extinction"/"generate_figure_extinction.py", [],
                        "figS_extinction/figures_supp/figure_extinction_supp_raw.pdf"),
    "figS_discriminator": (FIG/"figS_discriminator"/"generate_figure_discriminator.py", [],
                           "figS_discriminator/figures_supp/figure_discriminator_supp_raw.pdf"),
    "figS_rewardvector": (FIG/"figS_reward_vector"/"generate_figure_reward_vector_supp.py", [],
                          "figS_reward_vector/figures_supp/figure_reward_vector_supp_raw.pdf"),
    "figS_sosa": (FIG/"figS_sosa"/"generate_figure_sosa.py", [],
                  "figS_sosa/figures_supp/figure_sosa_supp_raw.pdf"),
}

# `--no_show` is universal EXCEPT for generate_figure_sosa.py, which parses no
# arguments at all and would choke on an unknown flag being ignored silently.
NO_SHOW_UNSUPPORTED = {"figS_sosa"}

# Expensive cache rebuilds, run before the figures when --with-caches is given.
CACHE_JOBS: dict[str, tuple[Path, list[str]]] = {
    "cache_extinction": (PC/"aggregate"/"run_extinction_analysis.py", []),
    "cache_estimator":  (PC/"aggregate"/"run_estimator_validation.py", []),
}


def run_one(name: str, script: Path, extra: list[str], timeout: int) -> dict:
    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), *extra]
    if name not in NO_SHOW_UNSUPPORTED:
        cmd.append("--no_show")
    log = LOGS / f"{name}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write(f"$ {' '.join(cmd)}\n\n")
        fh.flush()
        try:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               cwd=str(script.parent), timeout=timeout)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -1
            fh.write(f"\n*** TIMEOUT after {timeout}s ***\n")
    return {"name": name, "rc": rc, "secs": time.time() - t0, "log": log}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=10800, help="per figure, seconds")
    ap.add_argument("--with-caches", action="store_true",
                    help="also rebuild the aggregate JSON caches first (slow)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, (s, e, out) in JOBS.items():
            print(f"  {n:20s} {s.relative_to(PC)} {' '.join(e)}")
        return 0

    names = args.only or list(JOBS)
    unknown = [n for n in names if n not in JOBS]
    if unknown:
        print(f"unknown: {unknown}\navailable: {list(JOBS)}")
        return 1

    if args.with_caches:
        print("== rebuilding aggregate caches (serial, slow) ==")
        for n, (s, e) in CACHE_JOBS.items():
            r = run_one(n, s, e, args.timeout)
            print(f"  [{'ok' if r['rc']==0 else 'FAIL'}] {n}  {r['secs']:.0f}s  -> {r['log']}")

    print(f"== regenerating {len(names)} figures, {args.jobs} at a time ==")
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, n, *JOBS[n][:2], args.timeout): n for n in names}
        for f in futs:
            pass
        for f in list(futs):
            r = f.result()
            results.append(r)
            out = FIG / JOBS[r["name"]][2]
            status = "ok" if r["rc"] == 0 and out.exists() else "FAIL"
            print(f"  [{status}] {r['name']:20s} {r['secs']:7.0f}s  {r['log'].name}",
                  flush=True)

    fails = [r for r in results if r["rc"] != 0]
    print(f"\n{len(results)-len(fails)} ok, {len(fails)} failed"
          f"   (total wall {max((r['secs'] for r in results), default=0):.0f}s)")
    for r in fails:
        print(f"  FAILED {r['name']}: see {r['log']}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
