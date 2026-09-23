#!/usr/bin/env python3
"""Produce figS_fate ("Place-cell status migration") for main_v4_reframed.tex.

This figure was originally made by the analysis script that sits next to this file
as `plot_replay_v11.py` (the 916-line standalone copy, not the trimmed 313-line
`paper_code/lib/` module of the same name). Its `fig_recruitment_fate()` is the only generator in the repo
that emits the panels in the shipped PDF -- identified by the verbatim titles
"Place-cell status migration", "Activity level change by fate" and
"Distance to reward by fate", and the legend format "Neither (n=87)".

The legacy script has no `--output_dir`: it writes every panel it makes into
`<run_dir>/figures/<fig_subdir>/`. This wrapper runs it and copies the one PDF
we need to `figures_supp/figS_fate_raw.pdf`, so `sync_paper_figures.py` has a
stable source path like every other figure directory.

NOTE: the legacy script regenerates its *whole* set of replay figures, not just
this panel, so a run takes a while and rewrites other files under `<run_dir>/figures/`.

WARNING -- unverified provenance: `--run_dir` below is an inference. The shipped
figS_fate.pdf was produced in July 2026 and the original invocation was not
recorded; `final_v11_fixed` is the null/no-head run whose data still contains the
required periods. Confirm the output matches the shipped PDF before replacing it
in the manuscript. See figures/figS_legacy/README.md.

Usage:
    python paper_code/figures/figS_legacy/generate_figS_fate.py
    python paper_code/figures/figS_legacy/generate_figS_fate.py --run_dir <other run>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LEGACY = HERE / "plot_replay_v11.py"
OUT_DIR = HERE / "figures_supp"

DEFAULT_RUN = REPO / "New_Results_v3" / "final_v11_fixed"
DEFAULT_CYCLE = 1
DEFAULT_SKAGGS = 1.0          # the legacy script's own default
FIG_SUBDIR = "replay"


def main() -> int:
    """Run the replay script on a run directory and copy the recruitment-fate PDF to figures_supp/."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=str, default=str(DEFAULT_RUN))
    ap.add_argument("--replay_cycle", type=int, default=DEFAULT_CYCLE)
    ap.add_argument("--skaggs_threshold", type=float, default=DEFAULT_SKAGGS)
    ap.add_argument("--dry_run", action="store_true",
                    help="print the legacy invocation without running it")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "data").is_dir():
        print(f"ERROR: no data/ under {run_dir}\n"
              f"       figS_fate needs a v11 run with replay periods "
              f"(df_activations_replay_c{args.replay_cycle}_*.pkl).")
        return 1

    cmd = [sys.executable, str(LEGACY),
           "--run_dir", str(run_dir),
           "--replay_cycle", str(args.replay_cycle),
           "--skaggs_threshold", str(args.skaggs_threshold),
           "--fig_subdir", FIG_SUBDIR,
           "--no_show"]
    print("running:", " ".join(cmd))
    if args.dry_run:
        return 0
    subprocess.run(cmd, check=True, cwd=str(HERE))

    produced = run_dir / "figures" / FIG_SUBDIR / \
        f"replay_c{args.replay_cycle}_recruitment_fate.pdf"
    if not produced.exists():
        print(f"ERROR: legacy script finished but {produced} is absent")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "figS_fate_raw.pdf"
    shutil.copy2(produced, dest)
    print(f"figS_fate_raw.pdf  <-  {produced.relative_to(REPO)}")
    print(f"wrote {dest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
