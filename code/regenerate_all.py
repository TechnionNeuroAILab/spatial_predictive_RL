#!/usr/bin/env python3
"""Regenerate every generated figure referenced by paper_repo/main.tex.

This is the canonical machine-readable map from manuscript figures to figure
generators and exemplar runs.  Generators for retired exploratory figures may
remain under code/figures/, but they are deliberately absent from ``JOBS``.

It does not train anything: every manuscript figure is built from run data
under results/. Rebuilding the JSON caches is a separate, much more expensive
step (see --with-caches).

Figures are independent processes, so they run in parallel (default: 6 at a
time -- several load 40-50 seeds of pickled activity and peak at a few GB each).

Usage:
    python code/regenerate_all.py --list
    python code/regenerate_all.py                     # all figures
    python code/regenerate_all.py --only fig2 fig3
    python code/regenerate_all.py --jobs 10
    python code/regenerate_all.py --plot-only   # redraw from figure_data.npz

The first full run writes ``figure_data.npz`` next to each figure. A later
colormap, label, or layout edit uses ``--plot-only`` and does not rescan the
runs. ``--assemble_only`` (on a single generator) only restacks panel PDFs, so
it cannot recolor a map. figS_sosa already reads its JSON cache and takes no
flag.
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
# (provenance: results/*/aggregate/fig*_multi_seed_summary.json + the
# select_best_seed_fig*.py rankings).
JOBS: dict[str, tuple[Path, list[str], str]] = {
    # fig1 and fig2 share one exemplar network: nohead_seed16, the paired-cohort
    # network closest to the cohort median on all four per-network shifts
    # (behavioural and fixed-trajectory, peak and centre of mass; see
    # results/aggregate/headline_stats.json). Their example cells are then the
    # same units of the same network. (nohead_seed0, the earlier fig1 exemplar,
    # has a replay shift *away* from reward, +0.043.)
    # fig1 also writes figure_model_raw.pdf: the architecture schematic, which
    # used to be its panel a and is now a figure in its own right.
    "fig1": (FIG/"fig1"/"generate_figure1.py",
             ["--run_dir", str(REPO/"results"/"multi_seed"/"nohead_seed16")],
             "fig1/figures/figure1_raw.pdf"),
    "fig2": (FIG/"fig2"/"generate_figure2.py",
             ["--run_dir", str(REPO/"results"/"multi_seed"/"nohead_seed16")],
             "fig2/figures/figure2_raw.pdf"),
    "fig3": (FIG/"fig3"/"generate_figure3.py",
             ["--run_dir", str(REPO/"results"/"multi_seed_head"/"head_seed28")],
             "fig3/figures/figure3_raw.pdf"),
    "fig4": (FIG/"fig4"/"generate_figure4.py",
             # reloc_seed40: the retained relocation network closest to the
             # cohort median on the shifts toward A and B, peak and centre of
             # mass (headline_stats.json). reloc_seed9, the earlier exemplar,
             # ranked 48/50 on the centre-of-mass shift toward A.
             ["--run_dir", str(REPO/"results"/"multi_seed_reloc"/"reloc_seed40")],
             "fig4/figures/figure4_raw.pdf"),
    # generate_figure5_data.py now writes two plates: the reward-omission
    # (Krishnan) figure and the reward-relocation (Sosa) figure. The single
    # 183 x 262 mm plate it used to write was taller than the text block.
    "fig5": (FIG/"fig5"/"generate_figure5_data.py", [],
             "fig5/figures/figure_omission_raw.pdf"),
    "figS_pipeline": (FIG/"fig1"/"generate_figure1_supp.py", [],
                      "fig1/figures_supp/figure1_supp_raw.pdf"),
    "figS_relocation": (FIG/"fig4"/"generate_figure4_supp.py", [],
                        "fig4/figures_supp/figure4_supp_raw.pdf"),
    "figS_dissociation": (FIG/"fig5"/"generate_figure5_supp.py", [],
                          "fig5/figures_supp/figure5_supp_raw.pdf"),
    "figS_sr": (FIG/"figS_sr"/"generate_figure_sr_supp.py", [],
                "figS_sr/figures_supp/figure_sr_supp_raw.pdf"),
    "figS_extinction": (FIG/"figS_extinction"/"generate_figure_extinction.py", [],
                        "figS_extinction/figures_supp/figure_extinction_supp_raw.pdf"),
    "figS_discriminator": (FIG/"figS_discriminator"/"generate_figure_discriminator.py", [],
                           "figS_discriminator/figures_supp/figure_discriminator_supp_raw.pdf"),
    "figS_rewardvector": (FIG/"figS_reward_vector"/"generate_figure_reward_vector_supp.py", [],
                          "figS_reward_vector/figures_supp/figure_reward_vector_supp_raw.pdf"),
    "figS_sosa": (FIG/"figS_sosa"/"generate_figure_sosa.py", [],
                  "figS_sosa/figures_supp/figure_sosa_supp_raw.pdf"),
}

# Jobs that intentionally emit a second manuscript plate.
ADDITIONAL_OUTPUTS: dict[str, list[str]] = {
    "fig1": ["fig1/figures/figure_model_raw.pdf"],
    "fig5": ["fig5/figures/figure_relocdata_raw.pdf"],
}

# Every canonical generator supports ``--no_show`` and ``--plot-only``.
NO_SHOW_UNSUPPORTED: set[str] = set()

# Expensive cache rebuilds, run before the figures when --with-caches is given.
CACHE_JOBS: dict[str, tuple[Path, list[str]]] = {
    # Cohort membership is upstream of every aggregate.
    "cache_cohorts":   (PC/"aggregate"/"cohorts.py", []),
    "cache_headlines": (PC/"aggregate"/"headline_stats.py", []),
    "cache_extinction": (PC/"aggregate"/"run_extinction_analysis.py", []),
    "cache_estimator":  (PC/"aggregate"/"run_estimator_validation.py", []),
    # Requires relocation pickles; refuses to overwrite the committed JSON
    # without --force (pass it here so --with-caches can refresh after a new cohort).
    "cache_reward_relative": (PC/"aggregate"/"run_reward_relative_analysis.py",
                              ["--force"]),
}


def run_one(name: str, script: Path, extra: list[str], timeout: int,
            plot_only: bool = False) -> dict:
    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), *extra]
    # Cache drivers are argparse-strict and do not accept --no_show.
    # figS_sosa parses no arguments; it already draws from sosa_tests.json.
    if name not in NO_SHOW_UNSUPPORTED and not name.startswith("cache_"):
        cmd.append("--no_show")
        if plot_only:
            cmd.append("--plot-only")
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
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw figures from the figure_data.npz each generator "
                         "saved on its last full run; do not rescan runs")
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
            if r["rc"] != 0:
                print("Cache rebuild failed; refusing to draw figures from stale inputs.")
                return 1

    print(f"== regenerating {len(names)} figures, {args.jobs} at a time ==")
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, n, *JOBS[n][:2], args.timeout, args.plot_only): n
                for n in names}
        for f in futs:
            pass
        for f in list(futs):
            r = f.result()
            results.append(r)
            outputs = [JOBS[r["name"]][2], *ADDITIONAL_OUTPUTS.get(r["name"], [])]
            outputs_exist = all((FIG / output).exists() for output in outputs)
            status = "ok" if r["rc"] == 0 and outputs_exist else "FAIL"
            if not outputs_exist and r["rc"] == 0:
                r["rc"] = 1
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
